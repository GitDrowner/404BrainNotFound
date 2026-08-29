#!/usr/bin/env python3
"""Convert audit logits to the competition's minimal confidence JSONL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    temperature, bias = float(calibration["temperature"]), float(calibration["bias"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit_handle = args.audit_output.open("w") if args.audit_output else None
    with args.scores.open() as source, args.output.open("w") as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            raw = float(row.get("raw_logit_fp32", row.get("logit")))
            calibrated = raw / temperature + bias
            probability = 1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, calibrated))))
            image_path = str(row.get("image_path", row.get("path")))
            target.write(json.dumps({"image_path": image_path, "pred": probability}) + "\n")
            if audit_handle:
                audit_handle.write(json.dumps({**row, "calibrated_logit": calibrated, "probability_fake": probability}) + "\n")
    if audit_handle:
        audit_handle.close()


if __name__ == "__main__":
    main()
