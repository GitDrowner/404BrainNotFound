#!/usr/bin/env python3
"""Select an operating threshold on calibration data only.

This does not refit the Platt temperature or bias. It replaces only the operating
point with the threshold that maximizes mean real/fake recall.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.scores.read_text().splitlines() if line]
    calibration = json.loads(args.calibration.read_text())
    temperature = float(calibration["temperature"])
    bias = float(calibration["bias"])
    if temperature <= 0:
        raise ValueError("Calibration temperature must be positive")
    labels = [int(row["label"]) for row in rows]
    if set(labels) != {0, 1}:
        raise ValueError("Calibration scores must contain both labels")
    probabilities = [
        sigmoid(
            float(row["raw_logit_fp32"] if "raw_logit_fp32" in row else row["logit"])
            / temperature
            + bias
        )
        for row in rows
    ]
    real_total = labels.count(0)
    fake_total = labels.count(1)
    best: tuple[float, float, float, float, float] | None = None
    for threshold in sorted(set([0.0, 1.0, *probabilities])):
        real_recall = sum(label == 0 and score < threshold for label, score in zip(labels, probabilities)) / real_total
        fake_recall = sum(label == 1 and score >= threshold for label, score in zip(labels, probabilities)) / fake_total
        balanced = 0.5 * (real_recall + fake_recall)
        candidate = (balanced, -abs(threshold - 0.5), threshold, real_recall, fake_recall)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    balanced, _, threshold, real_recall, fake_recall = best
    result = {
        "format": "aigc_platt_v1_balanced_operating_point",
        "fit_role": "calibration_only",
        "source_calibration": str(args.calibration),
        "source_calibration_sha256": sha256(args.calibration),
        "samples": len(rows),
        "temperature": temperature,
        "bias": bias,
        "threshold": threshold,
        "threshold_method": "maximize_class_balanced_accuracy_on_calibration_only",
        "calibration_real_recall": real_recall,
        "calibration_fake_recall": fake_recall,
        "calibration_balanced_accuracy": balanced,
        "external_test_labels_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
