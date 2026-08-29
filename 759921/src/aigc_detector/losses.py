from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ClassSourceGroupDRO:
    """GroupDRO over (class, source) with symmetric class-hard mining.

    Keeping the class in the group identity prevents an easy real or fake source
    from hiding the opposite error mode.  The class-balance term penalizes a large
    gap between the real and fake surrogate risks, which is the differentiable
    counterpart of balancing false positives and false negatives.
    """

    def __init__(
        self,
        groups: list[str],
        *,
        eta: float,
        hard_fraction: float,
        hard_weight: float,
        class_balance_weight: float,
        device: torch.device,
    ) -> None:
        self.groups = sorted(set(groups))
        if not self.groups:
            raise ValueError("ClassSourceGroupDRO requires at least one group")
        self.index = {name: index for index, name in enumerate(self.groups)}
        self.q = torch.full(
            (len(self.groups),), 1.0 / len(self.groups), device=device
        )
        self.eta = eta
        self.hard_fraction = hard_fraction
        self.hard_weight = hard_weight
        self.class_balance_weight = class_balance_weight

    @staticmethod
    def group_name(label: int, source: str, degradation: str | None = None) -> str:
        base = f"{label}::{source}"
        return base if degradation is None else f"{base}::{degradation}"

    def __call__(
        self,
        per_sample_loss: torch.Tensor,
        labels: torch.Tensor,
        sources: list[str] | tuple[str, ...],
        degradations: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if degradations is None:
            degradations = [None] * len(sources)
        sample_groups = [
            self.group_name(int(label), str(source), degradation)
            for label, source, degradation in zip(
                labels.detach().cpu().tolist(), sources, degradations
            )
        ]
        unknown = sorted(set(sample_groups).difference(self.index))
        if unknown:
            raise RuntimeError(f"Unknown GroupDRO groups in batch: {unknown}")
        group_indices = torch.tensor(
            [self.index[name] for name in sample_groups],
            dtype=torch.long,
            device=per_sample_loss.device,
        )
        present_indices = torch.unique(group_indices, sorted=True)
        group_losses = torch.stack(
            [per_sample_loss[group_indices == index].mean() for index in present_indices]
        )
        with torch.no_grad():
            self.q[present_indices] *= torch.exp(self.eta * group_losses.detach())
            self.q /= self.q.sum().clamp_min(1e-12)
        present_weights = self.q[present_indices]
        robust = (present_weights * group_losses).sum() / present_weights.sum().clamp_min(1e-12)

        class_losses = []
        hard_losses = []
        for label in (0, 1):
            selected = per_sample_loss[labels == label]
            if selected.numel() == 0:
                continue
            class_losses.append(selected.mean())
            count = max(1, round(selected.numel() * self.hard_fraction))
            hard_losses.append(selected.topk(count).values.mean())
        hard = torch.stack(hard_losses).mean() if hard_losses else per_sample_loss.mean()
        class_gap = (
            torch.abs(class_losses[0] - class_losses[1])
            if len(class_losses) == 2
            else per_sample_loss.new_zeros(())
        )
        total = robust + self.hard_weight * hard + self.class_balance_weight * class_gap
        stats = {
            "group_dro": float(robust.detach()),
            "class_hard": float(hard.detach()),
            "class_risk_gap": float(class_gap.detach()),
            "robust_classification": float(total.detach()),
        }
        return total, stats

    def weights(self) -> dict[str, float]:
        return {
            name: float(self.q[index].detach().cpu())
            for name, index in self.index.items()
        }


def trace_loss_terms(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    labels = batch["label"]
    classification = F.binary_cross_entropy_with_logits(outputs["logits"], labels)
    tile_targets = labels[:, None].expand_as(outputs["tile_logits"])
    tile = F.binary_cross_entropy_with_logits(outputs["tile_logits"], tile_targets)
    feature_consistency = 1.0 - F.cosine_similarity(
        outputs["embedding"], outputs["clean_embedding"], dim=-1
    ).mean()
    consistency = F.smooth_l1_loss(
        outputs["aug_consistency_logits"], outputs["clean_consistency_logits"]
    ) + 0.5 * F.binary_cross_entropy_with_logits(outputs["aug_consistency_logits"], labels)
    degradation_classification = F.cross_entropy(
        outputs["degradation_logits"], batch["degradation_class"]
    )
    degradation_severity = F.smooth_l1_loss(
        outputs["degradation_severity"], batch["degradation_severity"]
    )
    terms = {
        "classification": classification,
        "consistency": consistency,
        "feature_consistency": feature_consistency,
        "degradation_classification": degradation_classification,
        "degradation_severity": degradation_severity,
        "tile_auxiliary": tile,
    }
    if "clean_logits" in outputs:
        clean_classification = F.binary_cross_entropy_with_logits(outputs["clean_logits"], labels)
        final_logit_consistency = F.smooth_l1_loss(
            outputs["logits"], outputs["clean_logits"]
        ) + F.smooth_l1_loss(
            torch.sigmoid(outputs["logits"]), torch.sigmoid(outputs["clean_logits"])
        )
        terms["clean_classification"] = clean_classification
        terms["final_logit_consistency"] = final_logit_consistency
    if "expert_gate" in outputs:
        mean_gate = outputs["expert_gate"].mean(dim=0)
        target = torch.full_like(mean_gate, 1.0 / mean_gate.numel())
        terms["gate_balance"] = F.mse_loss(mean_gate, target)
        # Avoid exact expert collapse without forcing large disagreements.
        diversity = outputs["expert_logits"].std(dim=-1, unbiased=False).mean()
        terms["expert_diversity"] = F.relu(outputs["logits"].new_tensor(0.10) - diversity)
    return terms


def trace_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict) -> tuple[torch.Tensor, dict[str, float]]:
    terms = trace_loss_terms(outputs, batch)
    total = sum(float(weights.get(name, 0.0)) * value for name, value in terms.items())
    scalar_terms = {name: float(value.detach()) for name, value in terms.items()}
    scalar_terms["total"] = float(total.detach())
    return total, scalar_terms


class UncertaintyWeightedLoss(nn.Module):
    """Learn interpretable homoscedastic uncertainty weights for loss groups.

    The primary real/fake objective is deliberately kept as a fixed anchor.  Each
    auxiliary group contributes ``exp(-s) * group_loss + s`` where ``s`` is a
    learned log variance.  Group membership is explicit in the configuration so
    the learned weights remain human-readable and checkpoint-auditable.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        if config.get("mode") != "uncertainty":
            raise ValueError("UncertaintyWeightedLoss requires mode=uncertainty")
        groups = config.get("groups", {})
        if not groups:
            raise ValueError("Uncertainty weighting requires at least one loss group")
        self.group_names = list(groups)
        self.group_terms = {
            name: tuple(str(term) for term in settings["terms"])
            for name, settings in groups.items()
        }
        if any(not terms for terms in self.group_terms.values()):
            raise ValueError("Every uncertainty loss group must contain a term")
        initial_weights = torch.tensor(
            [float(groups[name]["initial_weight"]) for name in self.group_names],
            dtype=torch.float32,
        )
        if bool(torch.any(initial_weights <= 0)):
            raise ValueError("Uncertainty initial weights must be positive")
        self.log_variances = nn.Parameter(-torch.log(initial_weights))
        self.classification_anchor = float(config.get("classification_anchor", 1.0))
        self.min_log_variance = float(config.get("min_log_variance", -3.0))
        self.max_log_variance = float(config.get("max_log_variance", 3.0))
        if self.min_log_variance >= self.max_log_variance:
            raise ValueError("min_log_variance must be smaller than max_log_variance")

    def _bounded_log_variances(self, *, learn: bool) -> torch.Tensor:
        values = self.log_variances.clamp(
            self.min_log_variance, self.max_log_variance
        )
        return values if learn else values.detach()

    def forward(
        self,
        terms: dict[str, torch.Tensor],
        *,
        primary_loss: torch.Tensor | None = None,
        learn: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        primary = terms["classification"] if primary_loss is None else primary_loss
        total = self.classification_anchor * primary
        data_objective = total
        regularizer_total = total.new_zeros(())
        stats: dict[str, float] = {
            "classification_anchor": self.classification_anchor,
            "primary_contribution": float(total.detach()),
            "uncertainty_learning_enabled": float(learn),
        }
        bounded = self._bounded_log_variances(learn=learn)
        for index, name in enumerate(self.group_names):
            missing = [term for term in self.group_terms[name] if term not in terms]
            if missing:
                raise KeyError(f"Missing terms for uncertainty group {name}: {missing}")
            group_loss = torch.stack(
                [terms[term].float() for term in self.group_terms[name]]
            ).sum()
            log_variance = bounded[index]
            weight = torch.exp(-log_variance)
            weighted = weight * group_loss
            total = total + weighted + log_variance
            data_objective = data_objective + weighted
            regularizer_total = regularizer_total + log_variance
            prefix = f"uncertainty/{name}"
            stats[f"{prefix}/group_loss"] = float(group_loss.detach())
            stats[f"{prefix}/weight"] = float(weight.detach())
            stats[f"{prefix}/weighted_contribution"] = float(weighted.detach())
            stats[f"{prefix}/log_variance_regularizer"] = float(
                log_variance.detach()
            )
        stats["uncertainty/data_objective"] = float(data_objective.detach())
        stats["uncertainty/log_variance_regularizer"] = float(
            regularizer_total.detach()
        )
        stats["total"] = float(total.detach())
        return total, stats

    @torch.no_grad()
    def clamp_parameters(self) -> None:
        self.log_variances.clamp_(self.min_log_variance, self.max_log_variance)

    @torch.no_grad()
    def summary(self) -> dict[str, dict[str, float | list[str]]]:
        bounded = self._bounded_log_variances(learn=False).cpu()
        return {
            name: {
                "terms": list(self.group_terms[name]),
                "log_variance": float(bounded[index]),
                "weight": float(torch.exp(-bounded[index])),
            }
            for index, name in enumerate(self.group_names)
        }
