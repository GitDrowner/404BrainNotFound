#!/usr/bin/env python3
"""
Batch inference script for RobustFusion.

Usage:
    # Single image
    python scripts/batch_inference.py --input path/to/image.jpg

    # Directory of images
    python scripts/batch_inference.py --input path/to/images/

    # Run with all 16 transforms
    python scripts/batch_inference.py --input path/to/images/ --transforms all --output results.json

    # Run clean only (default)
    python scripts/batch_inference.py --input path/to/images/ --transforms clean --output results.json

    # Specify transforms
    python scripts/batch_inference.py --input path/to/images/ --transforms jpeg_q90 blur_sigma1 noise_sigma0.05
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from PIL import Image

# Add model src to path for imports
MODEL_ROOT = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(MODEL_ROOT / "src"))

from aigc_detector.augmentations import competition_grid, apply_fixed, fixed_operation
from aigc_detector.explain import score_images, sha256
from aigc_detector.model import TraceDetector
from aigc_detector.train import autocast_context

import torch

DEFAULT_CHECKPOINT = MODEL_ROOT / "checkpoint" / "best.pt"
DEFAULT_CALIBRATION = MODEL_ROOT / "checkpoint" / "calibration_balanced.json"

TRANSFORM_LABELS = {
    "clean": "原图（不变换）",
    "jpeg_q90": "JPEG · quality 90",
    "jpeg_q70": "JPEG · quality 70",
    "jpeg_q50": "JPEG · quality 50",
    "jpeg_q30": "JPEG · quality 30",
    "blur_sigma0.5": "Gaussian blur · σ 0.5",
    "blur_sigma1": "Gaussian blur · σ 1.0",
    "blur_sigma2": "Gaussian blur · σ 2.0",
    "resize_0.5x": "Resize · 0.5× 后放大",
    "resize_0.25x": "Resize · 0.25× 后放大",
    "noise_sigma0.02": "Gaussian noise · σ 0.02",
    "noise_sigma0.05": "Gaussian noise · σ 0.05",
    "noise_sigma0.10": "Gaussian noise · σ 0.10",
    "color_minus20": "Color jitter · −20%",
    "color_plus20": "Color jitter · +20%",
    "center_crop_80": "Center crop · 80%",
}


def load_runtime(
    checkpoint: Path,
    calibration: Path,
    device: str = "auto",
) -> tuple[TraceDetector, dict, torch.device, dict]:
    """Load model, config, device, and calibration settings."""
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(device)

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint_data["config"]
    model = TraceDetector(
        config,
        int(config["data"]["image_size"]),
        int(config["data"]["semantic_image_size"]),
    ).to(device)
    model.load_state_dict(checkpoint_data["model"], strict=False)
    model.eval()

    calibration_data = json.loads(calibration.read_text())
    return model, config, device, calibration_data


def apply_transform(image: Image.Image, transform_id: str, image_digest: str) -> Image.Image:
    """Apply a named transform to an image with seeded randomness for reproducibility."""
    for tid, family, value in competition_grid():
        if tid == transform_id:
            seed_material = f"{image_digest}:{tid}".encode()
            seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
            return apply_fixed(image, str(family), value, seed=seed)
    raise ValueError(f"Unknown transform: {transform_id}")


def predict_single(
    model: TraceDetector,
    config: dict,
    device: torch.device,
    calibration: dict,
    image: Image.Image,
    image_path: Path,
    transform_id: str = "clean",
) -> dict:
    """Run inference on a single image for a single transform."""
    image_digest = hashlib.sha256(image.tobytes()).hexdigest()

    # Apply transform
    if transform_id != "clean":
        image = apply_transform(image, transform_id, image_digest)

    # Score
    temperature = float(calibration["temperature"])
    bias = float(calibration["bias"])
    threshold = float(calibration["threshold"])

    with torch.inference_mode():
        rows, _ = score_images(
            model, [image], config, device,
            temperature, bias, batch_size=1,
            decision_threshold=threshold,
        )

    row = rows[0]
    return {
        "image_path": str(image_path),
        "image_sha256": image_digest,
        "transform": transform_id,
        "transform_label": TRANSFORM_LABELS.get(transform_id, transform_id),
        "raw_logit": row["raw_logit"],
        "calibrated_logit": row["calibrated_logit"],
        "probability_fake": row["probability_fake"],
        "aigc_confidence": row.get("aigc_confidence", row["probability_fake"]),
        "decision": "aigc" if row.get("aigc_confidence", row["probability_fake"]) >= 0.5 else "real",
        "wavelet_similarity": row.get("wavelet_similarity", None),
    }


def predict_image(
    model: TraceDetector,
    config: dict,
    device: torch.device,
    calibration: dict,
    image_path: Path,
    transform_ids: list[str],
    threshold: float,
) -> dict:
    """Run inference on one image across specified transforms."""
    image = Image.open(image_path).convert("RGB")
    image_digest = hashlib.sha256(image.tobytes()).hexdigest()

    results = []
    for tid in transform_ids:
        try:
            result = predict_single(model, config, device, calibration, image, image_path, tid)
            results.append(result)
        except Exception as e:
            results.append({
                "image_path": str(image_path),
                "image_sha256": image_digest,
                "transform": tid,
                "transform_label": TRANSFORM_LABELS.get(tid, tid),
                "error": str(e),
            })

    # Summary across transforms
    valid = [r for r in results if "error" not in r]
    if valid:
        prob_fake_values = [r["probability_fake"] for r in valid]
        summary = {
            "image_path": str(image_path),
            "image_sha256": image_digest,
            "transform_count": len(valid),
            "mean_probability_fake": sum(prob_fake_values) / len(prob_fake_values),
            "min_probability_fake": min(prob_fake_values),
            "max_probability_fake": max(prob_fake_values),
            "decision_at_threshold": "aigc" if (sum(prob_fake_values) / len(prob_fake_values)) >= threshold else "real",
        }
    else:
        summary = {
            "image_path": str(image_path),
            "image_sha256": image_digest,
            "transform_count": 0,
            "error": "All transforms failed",
        }

    return {"summary": summary, "per_transform": results}


def collect_images(input_path: Path) -> list[Path]:
    """Collect all image files from a path (file or directory)."""
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes
    )


def run_batch(
    model: TraceDetector,
    config: dict,
    device: torch.device,
    calibration: dict,
    image_paths: list[Path],
    transform_ids: list[str],
    threshold: float,
    workers: int,
) -> list[dict]:
    """Run batch inference across all images."""
    results = []
    total = len(image_paths)
    threshold = float(calibration["threshold"])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                predict_image, model, config, device, calibration,
                path, transform_ids, threshold,
            ): path
            for path in image_paths
        }
        for i, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = "✓" if "error" not in result.get("summary", {}) else "✗"
                print(f"[{i}/{total}] {status} {path.name}")
            except Exception as e:
                results.append({
                    "summary": {
                        "image_path": str(path),
                        "error": str(e),
                    }
                })
                print(f"[{i}/{total}] ✗ {path.name}: {e}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="RobustFusion batch inference")
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Input image file or directory"
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        help="Output JSON file (default: stdout)"
    )
    parser.add_argument(
        "--transforms", "-t", nargs="+",
        default=["clean"],
        choices=["all", "clean"] + list(TRANSFORM_LABELS.keys()),
        help="Transforms to apply. Use 'all' for all 16 conditions."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="Model checkpoint path"
    )
    parser.add_argument(
        "--calibration", type=Path, default=DEFAULT_CALIBRATION,
        help="Platt calibration file"
    )
    parser.add_argument(
        "--device", "-d", default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=4,
        help="Parallel workers for batch inference"
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Only output summary, skip per-transform details"
    )
    args = parser.parse_args()

    # Resolve transforms
    if "all" in args.transforms:
        transform_ids = [t for t in TRANSFORM_LABELS.keys()]
    else:
        transform_ids = args.transforms

    # Collect images
    image_paths = collect_images(args.input)
    if not image_paths:
        print(f"No images found in {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {args.checkpoint} ...")
    model, config, device, calibration = load_runtime(
        args.checkpoint, args.calibration, args.device
    )
    print(f"Model loaded on {device}")
    print(f"Running {len(image_paths)} images × {len(transform_ids)} transforms ...")

    start = time.time()
    results = run_batch(
        model, config, device, calibration,
        image_paths, transform_ids, args.workers
    )
    elapsed = time.time() - start

    # Build output
    threshold = float(calibration["threshold"])
    output = {
        "metadata": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "calibration": str(args.calibration),
            "calibration_sha256": sha256(args.calibration),
            "calibration_threshold": threshold,
            "device": str(device),
            "transforms": transform_ids,
            "image_count": len(image_paths),
            "transform_count": len(transform_ids),
            "elapsed_seconds": round(elapsed, 1),
            "throughput_per_second": round(len(image_paths) / elapsed, 2),
        },
        "results": results,
    }

    if args.summary_only:
        output["results"] = [
            {"summary": r["summary"]} for r in results
        ]

    # Write output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"\nResults saved to {args.output}")
    else:
        print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
