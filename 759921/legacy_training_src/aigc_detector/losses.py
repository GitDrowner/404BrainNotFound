from __future__ import annotations

import torch
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
    def group_name(label: int, source: str) -> str:
        return f"{label}::{source}"

    def __call__(
        self,
        per_sample_loss: torch.Tensor,
        labels: torch.Tensor,
        sources: list[str] | tuple[str, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        sample_groups = [
            self.group_name(int(label), str(source))
            for label, source in zip(labels.detach().cpu().tolist(), sources)
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


def trace_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict) -> tuple[torch.Tensor, dict[str, float]]:
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
    total = sum(weights[name] * value for name, value in terms.items())
    scalar_terms = {name: float(value.detach()) for name, value in terms.items()}
    scalar_terms["total"] = float(total.detach())
    return total, scalar_terms
