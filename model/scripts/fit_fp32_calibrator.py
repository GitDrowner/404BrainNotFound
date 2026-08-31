#!/usr/bin/env python3
"""Fit an independent FP32 Platt calibrator; never point this at final test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aigc_detector.fusion import (
    PlattCalibrator,
    expected_calibration_error,
    group_balanced_weights,
    robust_threshold,
    weighted_bce,
)


def read_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()
    rows = read_rows(args.scores)
    if not rows or {int(row["label"]) for row in rows} != {0, 1}:
        raise RuntimeError("Calibration scores must be non-empty and contain both classes")
    logits = torch.tensor([float(row.get("raw_logit_fp32", row.get("logit"))) for row in rows])
    labels = torch.tensor([float(row["label"]) for row in rows])
    group_names = [f"{row.get('source','unknown')}::{row.get('variant','clean')}" for row in rows]
    weights = group_balanced_weights([int(value) for value in labels.tolist()], group_names)
    model = PlattCalibrator()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.25, max_iter=args.steps, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = weighted_bce(model(logits), labels, weights)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        calibrated_logits = model(logits)
        probabilities = torch.sigmoid(calibrated_logits)
        threshold = robust_threshold([int(x) for x in labels.tolist()], probabilities.tolist(), group_names)
        result = {
            "format": "aigc_platt_v1", "fit_role": "calibration_only", "samples": len(rows),
            "temperature": float(model.temperature), "bias": float(model.bias),
            "threshold": threshold["threshold"], "worst_group_recall": threshold["worst_group_recall"],
            "accuracy_at_threshold": threshold["accuracy"],
            "nll": float(weighted_bce(calibrated_logits, labels, weights)),
            "ece_15": expected_calibration_error(probabilities, labels),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
