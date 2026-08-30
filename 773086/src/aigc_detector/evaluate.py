from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .augmentations import (
    apply_fixed,
    competition_grid,
    fixed_operation,
    native_spectral_signature,
    native_tiles,
    resize_tensor,
)
from .metrics import binary_metrics
from .model import TraceDetector
from .train import autocast_context, move_batch


class FixedTransformDataset(Dataset):
    def __init__(self, manifest: Path, config: dict, variant: str, transform_name: str, value, limit: int | None):
        with manifest.open() as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if limit is not None:
            self.records = self.records[:limit]
        self.data_config = config["data"]
        self.variant = variant
        self.transform_name = transform_name
        self.value = value

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with Image.open(record["path"]) as handle:
            original = handle.convert("RGB").copy()
        image = apply_fixed(original, self.transform_name, self.value, seed=index)
        image_size = self.data_config["image_size"]
        global_view = resize_tensor(image, image_size)
        return {
            "clean_global": global_view,
            "clean_semantic_view": resize_tensor(image, self.data_config["semantic_image_size"]),
            "global_view": global_view,
            "semantic_view": resize_tensor(image, self.data_config["semantic_image_size"]),
            "tiles": native_tiles(image, image_size, self.data_config["num_tiles"], training=False),
            "native_spectral": native_spectral_signature(image),
            "clean_native_spectral": native_spectral_signature(image),
            "label": torch.tensor(float(record["label"]), dtype=torch.float32),
            "degradation_class": torch.tensor(0, dtype=torch.long),
            "degradation_severity": torch.tensor(0.0, dtype=torch.float32),
            "degradation_name": "clean",
            "augmentation": json.dumps(
                {
                    "policy": "fixed_robustness_grid",
                    "variant": self.variant,
                    "operations": [fixed_operation(self.transform_name, self.value)],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "path": record["path"],
            "source": record.get("source", "unknown"),
            "generator": record.get("generator", "unknown"),
        }


@torch.inference_mode()
def evaluate_variant(
    model, loader, device, amp: str, variant: str, audit_path: Path, threshold: float,
    calibration: dict | None = None,
) -> dict[str, float]:
    labels, probabilities = [], []
    for batch_index, batch in enumerate(loader):
        batch = move_batch(batch, device)
        with autocast_context(device, amp):
            outputs = model(batch)
        raw_logits = outputs["logits"].float()
        if calibration is None:
            calibrated_logits = raw_logits
        else:
            calibrated_logits = raw_logits / float(calibration["temperature"]) + float(calibration["bias"])
        batch_logits = raw_logits.cpu().tolist()
        batch_calibrated_logits = calibrated_logits.cpu().tolist()
        batch_probabilities = torch.sigmoid(calibrated_logits).cpu().tolist()
        labels.extend(batch["label"].cpu().tolist())
        probabilities.extend(batch_probabilities)
        batch_labels = batch["label"].cpu().tolist()
        with audit_path.open("a") as handle:
            for index, path in enumerate(batch["path"]):
                handle.write(
                    json.dumps(
                        {
                            "split": "test_robustness",
                            "variant": variant,
                            "batch": batch_index,
                            "path": path,
                            "label": int(batch_labels[index]),
                            "source": batch["source"][index],
                            "generator": batch["generator"][index],
                            "probability_fake": batch_probabilities[index],
                            "raw_logit_fp32": batch_logits[index],
                            "calibrated_logit_fp32": batch_calibrated_logits[index],
                            "threshold": threshold,
                            "prediction": int(batch_probabilities[index] >= threshold),
                            "augmentation": json.loads(batch["augmentation"][index]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return binary_metrics(labels, probabilities, threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--calibration", type=Path, help="Independent FP32 Platt calibration JSON")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Evaluation device. auto prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument("--batch-size", type=int, help="Override the training batch size for evaluation")
    parser.add_argument("--num-workers", type=int, help="Override the training dataloader worker count")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    batch_size = args.batch_size or config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"] if args.num_workers is None else args.num_workers
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch size must be positive and num workers must be non-negative")
    print(json.dumps({"event": "evaluation_runtime", "device": str(device), "batch_size": batch_size, "num_workers": num_workers}))
    model = TraceDetector(
        config,
        image_size=config["data"]["image_size"],
        semantic_image_size=config["data"]["semantic_image_size"],
    ).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
    model.eval()
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(calibration["threshold"] if calibration else config.get("evaluation", {}).get("threshold", 0.5))
    )
    results = []
    audit_path = args.output.with_name(args.output.name + "_augmentation_audit.jsonl")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.unlink(missing_ok=True)
    for variant, transform_name, value in competition_grid():
        dataset = FixedTransformDataset(args.manifest, config, variant, transform_name, value, args.limit)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        metrics = evaluate_variant(
            model,
            loader,
            device,
            config["train"]["amp"],
            variant,
            audit_path,
            threshold,
            calibration,
        )
        row = {"variant": variant, "transform": transform_name, "value": value, **metrics}
        results.append(row)
        print(json.dumps(row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
