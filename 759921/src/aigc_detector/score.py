from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import ManifestDataset
from .metrics import binary_metrics
from .model import TraceDetector


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    dataset = ManifestDataset(
        args.manifest,
        image_size=config["data"]["image_size"],
        semantic_image_size=config["data"]["semantic_image_size"],
        num_tiles=config["data"]["num_tiles"],
        training=False,
        max_samples=args.limit,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraceDetector(
        config,
        image_size=config["data"]["image_size"],
        semantic_image_size=config["data"]["semantic_image_size"],
    ).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or config["data"]["batch_size"],
        shuffle=False,
        num_workers=(
            args.num_workers if args.num_workers is not None else config["data"]["num_workers"]
        ),
        pin_memory=device.type == "cuda",
    )

    scored_records = []
    labels, probabilities = [], []
    offset = 0
    for batch_index, batch in enumerate(loader):
        moved = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with autocast_context(device, config["train"]["amp"]):
            outputs = model(moved)
        batch_logits = outputs["logits"].float().cpu().tolist()
        batch_probabilities = torch.sigmoid(outputs["logits"].float()).cpu().tolist()
        batch_gates = outputs.get("expert_gate")
        batch_experts = outputs.get("expert_logits")
        if batch_gates is not None:
            batch_gates = batch_gates.float().cpu().tolist()
            batch_experts = batch_experts.float().cpu().tolist()
        batch_labels = moved["label"].int().cpu().tolist()
        for index, (label, probability) in enumerate(zip(batch_labels, batch_probabilities)):
            original = dict(dataset.records[offset + index])
            original.update(
                {
                    "probability_fake": probability,
                    "raw_logit_fp32": batch_logits[index],
                    "prediction_at_0_5": int(probability >= 0.5),
                    "hardness": probability if label == 0 else 1.0 - probability,
                    "score_batch": batch_index,
                }
            )
            if batch_gates is not None:
                original["expert_gate"] = batch_gates[index]
                original["expert_logits_fp32"] = batch_experts[index]
            scored_records.append(original)
        offset += len(batch_labels)
        labels.extend(batch_labels)
        probabilities.extend(batch_probabilities)
        if (batch_index + 1) % 20 == 0:
            print(json.dumps({"event": "score_progress", "processed": offset}))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in scored_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    overall = binary_metrics(labels, probabilities)
    by_source: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"labels": [], "probabilities": []}
    )
    for record in scored_records:
        group = by_source[record.get("source", "unknown")]
        group["labels"].append(record["label"])
        group["probabilities"].append(record["probability_fake"])
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_path(args.checkpoint),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_path(args.manifest),
        "output": str(args.output),
        "output_sha256": sha256_path(args.output),
        "total": len(scored_records),
        "real": sum(label == 0 for label in labels),
        "fake": sum(label == 1 for label in labels),
        "metrics_at_0_5": overall,
        "by_source": {
            source: binary_metrics(values["labels"], values["probabilities"])
            for source, values in sorted(by_source.items())
        },
        "training": False,
        "optimizer_created": False,
        "weights_modified": False,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "score_complete", **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
