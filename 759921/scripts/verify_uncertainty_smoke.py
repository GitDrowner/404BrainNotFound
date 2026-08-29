from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_path = args.output_dir / "loss_weight_history.jsonl"
    checkpoint_path = args.output_dir / "best.pt"
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line]
    if len(rows) < 2:
        raise RuntimeError("Uncertainty smoke requires at least two optimizer-step records")
    if not any(bool(row["learning_enabled"]) for row in rows):
        raise RuntimeError("Uncertainty learning never activated after warm-up")
    initial = {name: float(value["weight"]) for name, value in rows[0]["groups"].items()}
    final = {name: float(value["weight"]) for name, value in rows[-1]["groups"].items()}
    if initial.keys() != final.keys() or len(final) != 5:
        raise RuntimeError("Expected the same five uncertainty groups throughout smoke")
    if not all(math.isfinite(value) and 0.049 <= value <= 20.1 for value in final.values()):
        raise RuntimeError(f"Invalid final uncertainty weights: {final}")
    if not any(abs(final[name] - initial[name]) > 1e-8 for name in final):
        raise RuntimeError("No uncertainty weight changed after learning was enabled")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    uncertainty = checkpoint.get("uncertainty_weighting")
    if not uncertainty or set(uncertainty["summary"]) != set(final):
        raise RuntimeError("Checkpoint is missing the uncertainty-weighting state")
    print(
        json.dumps(
            {
                "event": "uncertainty_smoke_verified",
                "records": len(rows),
                "initial_weights": initial,
                "final_weights": final,
                "checkpoint": str(checkpoint_path),
            }
        )
    )


if __name__ == "__main__":
    main()
