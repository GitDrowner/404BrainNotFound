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
    rows = [
        json.loads(line)
        for line in (args.output_dir / "loss_weight_history.jsonl").read_text().splitlines()
        if line
    ]
    if len(rows) < 3 or not any(row["learning_enabled"] for row in rows):
        raise RuntimeError("MLP controller did not execute after its uniform warm-up")
    if any(row.get("mode") != "mlp_normalized" for row in rows):
        raise RuntimeError("Unexpected loss-controller mode in smoke history")
    final = {name: float(value["weight"]) for name, value in rows[-1]["groups"].items()}
    if len(final) != 5 or not all(math.isfinite(value) and value >= 0.049 for value in final.values()):
        raise RuntimeError(f"Invalid MLP weights: {final}")
    if abs(sum(final.values()) - 1.0) > 1e-5:
        raise RuntimeError(f"MLP weights do not form a simplex: {final}")
    if max(final.values()) - min(final.values()) <= 1e-7:
        raise RuntimeError(f"MLP weights remained exactly uniform: {final}")
    checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    controller = checkpoint.get("mlp_normalized_weighting")
    if not controller or controller.get("mode") != "mlp_normalized":
        raise RuntimeError("Checkpoint is missing the MLP loss controller")
    if checkpoint.get("selection_metric") != "auroc":
        raise RuntimeError("Checkpoint was not selected by AUROC")
    validation = checkpoint.get("metrics", {})
    if not math.isfinite(float(validation.get("auroc", float("nan")))):
        raise RuntimeError("Checkpoint lacks finite held-out validation AUROC")
    if checkpoint.get("selection_manifest") is not None:
        raise RuntimeError("Smoke unexpectedly used an external selection manifest")
    print(
        json.dumps(
            {
                "event": "mlp_controller_smoke_verified",
                "records": len(rows),
                "final_weights": final,
                "selection_role": "held_out_training_corpus_validation",
                "selection_auroc": validation["auroc"],
            }
        )
    )


if __name__ == "__main__":
    main()
