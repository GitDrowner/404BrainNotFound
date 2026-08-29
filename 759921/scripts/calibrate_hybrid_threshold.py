#!/usr/bin/env python3
"""Fit one source-robust threshold on a dedicated, multi-domain calibration set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from aigc_detector.metrics import binary_metrics


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.scores.read_text().splitlines() if line.strip()]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["probability_fake"]) for row in rows])
    sources = np.asarray([str(row.get("source", "unknown")) for row in rows], dtype=object)
    if set(labels.tolist()) != {0, 1}:
        raise RuntimeError("Calibration requires both classes")

    best_key = (-1.0, -1.0, -1.0, -1.0)
    best_threshold = 0.5
    best_sources: dict[str, float] = {}
    for threshold in np.unique(np.r_[0.0, scores, 1.0]):
        prediction = scores >= threshold
        source_recalls = {
            source: float(np.mean(prediction[sources == source] == labels[sources == source]))
            for source in sorted(set(sources.tolist()))
        }
        real_recall = float(np.mean(~prediction[labels == 0]))
        fake_recall = float(np.mean(prediction[labels == 1]))
        key = (
            min(real_recall, fake_recall),
            min(source_recalls.values()),
            0.5 * (real_recall + fake_recall),
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_sources = source_recalls

    result = {
        "status": "PASS",
        "method": "single_global_threshold_maximin_class_then_source_recall",
        "scores": str(args.scores),
        "scores_sha256": sha256_path(args.scores),
        "total": len(rows),
        "real": int(np.sum(labels == 0)),
        "fake": int(np.sum(labels == 1)),
        "threshold": best_threshold,
        "worst_source_recall": best_key[1],
        "worst_class_recall": best_key[0],
        "source_recalls": best_sources,
        "metrics_at_0_5": binary_metrics(labels.tolist(), scores.tolist(), 0.5),
        "metrics_calibrated": binary_metrics(
            labels.tolist(), scores.tolist(), best_threshold
        ),
        "external_eval_only_used": False,
        "weights_modified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "hybrid_calibration_complete", **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
