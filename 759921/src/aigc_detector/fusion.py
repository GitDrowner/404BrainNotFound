from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlattCalibrator(nn.Module):
    """Positive-temperature affine calibration on FP32 logits."""

    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.02, 100.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.float() / self.temperature + self.bias.float()


class DualScoreGate(nn.Module):
    """Tiny frozen-model stacker supporting fixed and quality-aware fusion."""

    def __init__(self, quality_dim: int = 0, dynamic: bool = True, hidden_dim: int = 16) -> None:
        super().__init__()
        self.quality_dim = quality_dim
        self.dynamic = dynamic
        if dynamic:
            self.gate = nn.Sequential(
                nn.LayerNorm(3 + quality_dim),
                nn.Linear(3 + quality_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.fixed_gate_logit = nn.Parameter(torch.zeros(()))
        self.calibrator = PlattCalibrator()

    def forward(
        self,
        logit_a: torch.Tensor,
        logit_b: torch.Tensor,
        quality: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        a, b = logit_a.float(), logit_b.float()
        if self.dynamic:
            parts = [a[:, None], b[:, None], (a - b).abs()[:, None]]
            if self.quality_dim:
                if quality is None or quality.shape[-1] != self.quality_dim:
                    raise ValueError(f"Expected {self.quality_dim} quality features")
                parts.append(quality.float())
            gate = torch.sigmoid(self.gate(torch.cat(parts, dim=-1)).float()).squeeze(-1)
        else:
            gate = torch.sigmoid(self.fixed_gate_logit.float()).expand_as(a)
        fused_raw = gate * a + (1.0 - gate) * b
        calibrated_logit = self.calibrator(fused_raw)
        return {
            "fused_raw_logit": fused_raw,
            "calibrated_logit": calibrated_logit,
            "probability_fake": torch.sigmoid(calibrated_logit),
            "gate_a": gate,
        }


def weighted_bce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction="none")
    return (losses * weights.float()).sum() / weights.sum().clamp_min(1e-12)


def expected_calibration_error(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    probabilities, labels = probabilities.float(), labels.float()
    total = max(1, probabilities.numel())
    value = probabilities.new_zeros(())
    for lower in torch.linspace(0, 1, bins + 1, device=probabilities.device)[:-1]:
        upper = lower + 1.0 / bins
        mask = (probabilities >= lower) & (probabilities < upper)
        if mask.any():
            value += mask.sum() / total * (probabilities[mask].mean() - labels[mask].mean()).abs()
    return float(value.detach().cpu())


def group_balanced_weights(labels: list[int], groups: list[str]) -> torch.Tensor:
    keys = [f"{int(label)}::{group}" for label, group in zip(labels, groups)]
    counts = {key: keys.count(key) for key in set(keys)}
    raw = torch.tensor([1.0 / counts[key] for key in keys], dtype=torch.float32)
    return raw / raw.mean().clamp_min(1e-12)


def robust_threshold(labels: list[int], probabilities: list[float], groups: list[str]) -> dict[str, float]:
    candidates = sorted(set([0.0, 0.5, 1.0, *probabilities]))
    best: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        recalls = []
        for label in (0, 1):
            selected = [index for index, value in enumerate(labels) if value == label]
            if selected:
                recalls.append(sum((probabilities[i] >= threshold) == bool(label) for i in selected) / len(selected))
        for group in sorted(set(groups)):
            selected = [index for index, value in enumerate(groups) if value == group]
            if selected:
                recalls.append(sum((probabilities[i] >= threshold) == bool(labels[i]) for i in selected) / len(selected))
        worst = min(recalls)
        accuracy = sum((p >= threshold) == bool(y) for p, y in zip(probabilities, labels)) / len(labels)
        candidate = (worst, accuracy, -abs(threshold - 0.5), threshold)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    assert best is not None
    return {"threshold": best[3], "worst_group_recall": best[0], "accuracy": best[1]}
