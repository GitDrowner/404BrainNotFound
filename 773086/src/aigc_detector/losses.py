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


class NormalizedMLPLoss(nn.Module):
    """Dynamically balance auxiliary tasks without task-specific coefficients.

    Every auxiliary loss is divided by its detached exponential moving average.
    A shared MLP scores the resulting task statistics and a floored softmax maps
    those scores to a positive simplex.  The detector minimizes the weighted
    normalized losses, while the controller is trained adversarially to assign
    more mass to the currently harder normalized tasks.  Detaching the weights
    in the detector objective and using a zero-valued gradient proxy prevents
    the controller from winning by selecting only the easiest task.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        if config.get("mode") != "mlp_normalized":
            raise ValueError("NormalizedMLPLoss requires mode=mlp_normalized")
        groups = config.get("groups", {})
        if not groups:
            raise ValueError("MLP normalized weighting requires auxiliary groups")
        self.group_names = list(groups)
        self.group_terms = {
            name: tuple(str(term) for term in settings["terms"])
            for name, settings in groups.items()
        }
        if any(not terms for terms in self.group_terms.values()):
            raise ValueError("Every MLP loss group must contain a term")
        self.classification_anchor = float(config.get("classification_anchor", 1.0))
        self.auxiliary_budget = float(config.get("auxiliary_budget", 1.0))
        self.ema_decay = float(config.get("ema_decay", 0.98))
        self.min_weight = float(config.get("min_weight", 0.05))
        self.epsilon = float(config.get("epsilon", 1e-6))
        group_count = len(self.group_names)
        if self.auxiliary_budget <= 0:
            raise ValueError("auxiliary_budget must be positive")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if not 0.0 <= self.min_weight < 1.0 / group_count:
            raise ValueError("min_weight must be below 1 / number of groups")
        hidden_dim = int(config.get("hidden_dim", 32))
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        # The scorer is shared across tasks: it has no task ID or task-specific bias.
        self.scorer = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)
        self.register_buffer("ema_losses", torch.ones(group_count, dtype=torch.float32))
        self.register_buffer("initial_losses", torch.ones(group_count, dtype=torch.float32))
        self.register_buffer(
            "last_weights", torch.full((group_count,), 1.0 / group_count, dtype=torch.float32)
        )
        self.register_buffer("statistics_initialized", torch.tensor(False))
        self.register_buffer("controller_steps", torch.tensor(0, dtype=torch.long))

    def _group_losses(self, terms: dict[str, torch.Tensor]) -> torch.Tensor:
        losses = []
        for name in self.group_names:
            missing = [term for term in self.group_terms[name] if term not in terms]
            if missing:
                raise KeyError(f"Missing terms for MLP loss group {name}: {missing}")
            losses.append(
                torch.stack([terms[term].float() for term in self.group_terms[name]]).sum()
            )
        return torch.stack(losses)

    def forward(
        self,
        terms: dict[str, torch.Tensor],
        *,
        primary_loss: torch.Tensor | None = None,
        learn: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        primary = terms["classification"] if primary_loss is None else primary_loss
        group_losses = self._group_losses(terms)
        detached_losses = group_losses.detach().clamp_min(self.epsilon)
        if not bool(self.statistics_initialized):
            reference = detached_losses
            initial = detached_losses
            if self.training:
                self.ema_losses.copy_(detached_losses)
                self.initial_losses.copy_(detached_losses)
                self.statistics_initialized.fill_(True)
        else:
            reference = self.ema_losses.clamp_min(self.epsilon)
            initial = self.initial_losses.clamp_min(self.epsilon)
        normalized = group_losses / reference
        detached_normalized = detached_losses / reference
        log_normalized = torch.log(detached_normalized.clamp_min(self.epsilon))
        cross_task = (log_normalized - log_normalized.mean()) / (
            log_normalized.std(unbiased=False).clamp_min(self.epsilon)
        )
        trend = torch.log((detached_losses / initial).clamp_min(self.epsilon))
        trend = (trend - trend.mean()) / trend.std(unbiased=False).clamp_min(self.epsilon)
        progress = torch.full_like(
            cross_task,
            float(self.controller_steps.item()) / float(self.controller_steps.item() + 100),
        )
        features = torch.stack((cross_task, trend, progress), dim=-1)
        # Keep the simplex in FP32 even when the detector runs under BF16
        # autocast; otherwise five rounded 0.2 values sum to 1.0009765625.
        logits = self.scorer(features).squeeze(-1).float()
        uniform = torch.full_like(logits, 1.0 / len(self.group_names), dtype=torch.float32)
        if self.training and not learn:
            weights = uniform
        else:
            probabilities = torch.softmax(logits, dim=0)
            weights = self.min_weight + (1.0 - len(self.group_names) * self.min_weight) * probabilities
        model_auxiliary = self.auxiliary_budget * torch.sum(weights.detach() * normalized)
        # Numerically zero, but its gradient makes the controller emphasize the
        # hardest normalized task instead of collapsing onto the easiest one.
        controller_reward = torch.sum(weights * detached_normalized)
        controller_proxy = (
            -controller_reward + controller_reward.detach()
            if self.training and learn
            else group_losses.new_zeros(())
        )
        primary_contribution = self.classification_anchor * primary
        total = primary_contribution + model_auxiliary + controller_proxy
        if self.training:
            with torch.no_grad():
                self.ema_losses.mul_(self.ema_decay).add_(
                    detached_losses, alpha=1.0 - self.ema_decay
                )
                self.last_weights.copy_(weights.detach())
                self.controller_steps.add_(1)
        entropy = -torch.sum(weights * torch.log(weights.clamp_min(self.epsilon)))
        stats: dict[str, float] = {
            "classification_anchor": self.classification_anchor,
            "primary_contribution": float(primary_contribution.detach()),
            "mlp/learning_enabled": float(learn),
            "mlp/auxiliary_budget": self.auxiliary_budget,
            "mlp/weight_sum": float(weights.detach().sum()),
            "mlp/weight_entropy": float(entropy.detach()),
            "mlp/controller_reward": float(controller_reward.detach()),
            "mlp/normalized_auxiliary": float(model_auxiliary.detach()),
        }
        for index, name in enumerate(self.group_names):
            prefix = f"mlp/{name}"
            stats[f"{prefix}/group_loss"] = float(group_losses[index].detach())
            stats[f"{prefix}/ema_loss"] = float(reference[index].detach())
            stats[f"{prefix}/normalized_loss"] = float(normalized[index].detach())
            stats[f"{prefix}/weight"] = float(weights[index].detach())
            stats[f"{prefix}/weighted_contribution"] = float(
                (self.auxiliary_budget * weights[index].detach() * normalized[index].detach())
            )
            stats[f"{prefix}/score"] = float(logits[index].detach())
        stats["total"] = float(total.detach())
        return total, stats

    @torch.no_grad()
    def clamp_parameters(self) -> None:
        # Softmax and the explicit floor already bound the effective weights.
        return None

    @torch.no_grad()
    def summary(self) -> dict[str, dict[str, float | list[str]]]:
        return {
            name: {
                "terms": list(self.group_terms[name]),
                "weight": float(self.last_weights[index].cpu()),
                "ema_loss": float(self.ema_losses[index].cpu()),
            }
            for index, name in enumerate(self.group_names)
        }
