from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageFilter

from .augmentations import (
    apply_fixed,
    competition_grid,
    native_spectral_signature,
    native_tiles,
    resize_tensor,
)
from .explainability import (
    SCHEMA_VERSION,
    occlude_patch,
    patch_boxes,
    save_attribution_overlay,
    save_bar_svg,
    save_colorbar,
    save_dense_signed_heatmaps,
    save_line_svg,
    save_texture_heatmap,
    subdivide_box,
    suppress_high_frequency_patch,
    write_dashboard,
    write_schema,
)
from .model import TraceDetector
from .train import autocast_context


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_batch(images: list[Image.Image], config: dict, device: torch.device) -> dict[str, torch.Tensor]:
    data = config["data"]
    image_size = int(data["image_size"])
    semantic_size = int(data["semantic_image_size"])
    tile_count = int(data["num_tiles"])
    globals_ = torch.stack([resize_tensor(image, image_size) for image in images])
    semantics = torch.stack([resize_tensor(image, semantic_size) for image in images])
    batch = {
        "global_view": globals_, "clean_global": globals_.clone(),
        "semantic_view": semantics, "clean_semantic_view": semantics.clone(),
        "tiles": torch.stack([native_tiles(image, image_size, tile_count, False) for image in images]),
        "native_spectral": torch.stack([native_spectral_signature(image) for image in images]),
    }
    batch["clean_native_spectral"] = batch["native_spectral"].clone()
    return {key: value.to(device) for key, value in batch.items()}


def calibration_values(path: Path | None) -> tuple[float, float]:
    if path is None:
        return 1.0, 0.0
    payload = json.loads(path.read_text())
    return float(payload.get("temperature", 1.0)), float(payload.get("bias", 0.0))


def linearize_operating_point(probability: float, threshold: float) -> float:
    """Map a calibrated operating threshold to 0.5 without changing score order."""
    probability = min(1.0, max(0.0, float(probability)))
    threshold = float(threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("Operating-point threshold must be strictly between 0 and 1")
    if probability <= threshold:
        return 0.5 * probability / threshold
    return 0.5 + 0.5 * (probability - threshold) / (1.0 - threshold)


@torch.inference_mode()
def score_images(
    model: TraceDetector,
    images: list[Image.Image],
    config: dict,
    device: torch.device,
    temperature: float,
    bias: float,
    batch_size: int,
    decision_threshold: float | None = None,
) -> tuple[list[dict], list[dict[str, torch.Tensor]]]:
    rows, raw_outputs = [], []
    for start in range(0, len(images), batch_size):
        batch = make_batch(images[start : start + batch_size], config, device)
        with autocast_context(device, config["train"]["amp"]):
            outputs = model(batch)
        logits = outputs["logits"].float()
        calibrated = logits / temperature + bias
        probabilities = torch.sigmoid(calibrated)
        for index in range(len(logits)):
            row = {
                "raw_logit": float(logits[index].cpu()),
                "calibrated_logit": float(calibrated[index].cpu()),
                "probability_fake": float(probabilities[index].cpu()),
            }
            if decision_threshold is not None:
                row["aigc_confidence"] = linearize_operating_point(
                    row["probability_fake"], decision_threshold
                )
            if "wavelet_similarity" in outputs:
                row["wavelet_similarity"] = float(outputs["wavelet_similarity"][index].float().cpu())
            selected = {
                key: value[index].detach().float().cpu()
                for key, value in outputs.items()
                if isinstance(value, torch.Tensor) and value.shape[:1] == logits.shape[:1]
            }
            rows.append(row)
            raw_outputs.append(selected)
    return rows, raw_outputs


@torch.inference_mode()
def score_wavelet_only_counterfactuals(
    model: TraceDetector,
    original: Image.Image,
    images: list[Image.Image],
    config: dict,
    device: torch.device,
    temperature: float,
    bias: float,
    batch_size: int,
) -> tuple[dict, list[dict]]:
    """Keep every legacy fusion feature fixed except the real wavelet-similarity scalar.

    A temporary pre-hook is used only during explanation inference. It does not alter
    the module graph, parameters, checkpoint, or ``model.py``.
    """
    if model.use_three_expert or not model.use_wavelet:
        raise RuntimeError("wavelet-only counterfactual currently requires the 759921 legacy fusion path")
    captured: dict[str, torch.Tensor] = {}

    def capture_fusion_input(_module, inputs):
        captured["fusion_input"] = inputs[0].detach().clone()

    handle = model.fusion.register_forward_pre_hook(capture_fusion_input)
    try:
        with autocast_context(device, config["train"]["amp"]):
            base_outputs = model(make_batch([original], config, device))
    finally:
        handle.remove()
    base_input = captured["fusion_input"]
    base_raw = base_outputs["logits"].float()
    base_calibrated = base_raw / temperature + bias
    base = {
        "raw_logit": float(base_raw[0].cpu()),
        "probability_fake": float(torch.sigmoid(base_calibrated)[0].cpu()),
        "wavelet_similarity": float(base_outputs["wavelet_similarity"][0].float().cpu()),
    }
    rows = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]

        def replace_fusion_input(_module, inputs):
            incoming = inputs[0]
            replacement = base_input.to(device=incoming.device, dtype=incoming.dtype).expand(incoming.shape[0], -1).clone()
            replacement[:, -1] = incoming[:, -1]
            return (replacement,)

        handle = model.fusion.register_forward_pre_hook(replace_fusion_input)
        try:
            with autocast_context(device, config["train"]["amp"]):
                outputs = model(make_batch(batch_images, config, device))
        finally:
            handle.remove()
        logits = outputs["logits"].float()
        probabilities = torch.sigmoid(logits / temperature + bias)
        for index in range(len(batch_images)):
            rows.append({
                "raw_logit": float(logits[index].cpu()),
                "probability_fake": float(probabilities[index].cpu()),
                "wavelet_similarity": float(outputs["wavelet_similarity"][index].float().cpu()),
            })
    return base, rows


@torch.inference_mode()
def branch_counterfactuals(
    model: TraceDetector,
    image: Image.Image,
    config: dict,
    device: torch.device,
    temperature: float,
    bias: float,
    decision_threshold: float,
) -> list[dict]:
    batch = make_batch([image], config, device)
    with autocast_context(device, config["train"]["amp"]):
        base_outputs = model(batch)
    base_probability = float(
        torch.sigmoid(base_outputs["logits"].float() / temperature + bias).cpu()
    )
    base_confidence = linearize_operating_point(base_probability, decision_threshold)
    variants = {}
    components = {
        "global_spatial_and_wavelet_view": ("global_view",),
        "semantic_view": ("semantic_view",),
        "native_tiles": ("tiles",),
    }
    if model.use_three_expert:
        components["native_frequency_signature"] = ("native_spectral",)
    for name, keys in components.items():
        changed = {key: value.clone() for key, value in batch.items()}
        for key in keys:
            if key == "native_spectral":
                blurred = image.filter(ImageFilter.GaussianBlur(radius=max(5, min(image.size) / 12)))
                changed[key] = native_spectral_signature(blurred)[None].to(device)
            else:
                changed[key].fill_(0.5)
        with autocast_context(device, config["train"]["amp"]):
            changed_outputs = model(changed)
        probability = float(
            torch.sigmoid(changed_outputs["logits"].float() / temperature + bias).cpu()
        )
        variants[name] = (
            probability,
            linearize_operating_point(probability, decision_threshold),
        )
    return [
        {
            "component": name,
            "probability_when_neutralized": probability,
            "aigc_confidence_when_neutralized": confidence,
            "confidence_delta": base_confidence - confidence,
            "calibrated_probability_delta": base_probability - probability,
        }
        for name, (probability, confidence) in variants.items()
    ]


def generate_explanation(
    args,
    *,
    model: TraceDetector | None = None,
    config: dict | None = None,
    device: torch.device | None = None,
) -> dict:
    """Generate one explanation, optionally reusing an already loaded model.

    The reusable path is used by the local API. It keeps the CLI behavior intact
    while avoiding a 500M-parameter backbone reload for every uploaded image.
    Calls sharing a model must be serialized because wavelet attribution uses a
    temporary inference pre-hook.
    """
    if device is None and args.device == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        device = torch.device(name)
    elif device is None:
        name = args.device
        device = torch.device(name)
    if model is None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = TraceDetector(
            config,
            int(config["data"]["image_size"]),
            int(config["data"]["semantic_image_size"]),
        ).to(device)
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
    if config is None:
        raise ValueError("config is required when reusing a loaded model")
    model.eval()
    with Image.open(args.image) as opened:
        image = opened.convert("RGB").copy()
    temperature, bias = calibration_values(args.calibration)
    decision_threshold = (
        float(json.loads(args.calibration.read_text()).get("threshold", 0.5))
        if args.calibration
        else 0.5
    )
    base_rows, base_outputs = score_images(
        model, [image], config, device, temperature, bias, 1, decision_threshold
    )
    prediction, model_outputs = base_rows[0], base_outputs[0]
    prediction.update(
        {
            "label_at_display_threshold": (
                "aigc" if prediction["aigc_confidence"] >= 0.5 else "real"
            ),
            "display_threshold": 0.5,
            "calibrated_probability_threshold": decision_threshold,
            "confidence_semantics": (
                "piecewise-linearized FP32 Platt score; calibrated probability retained as probability_fake"
            ),
        }
    )

    coarse_boxes = patch_boxes(image.width, image.height, args.grid)
    occluded = [occlude_patch(image, box, args.occlusion) for box in coarse_boxes]
    coarse_scores, _ = score_images(
        model, occluded, config, device, temperature, bias, args.batch_size, decision_threshold
    )
    patches = []
    for index, (box, score) in enumerate(zip(coarse_boxes, coarse_scores)):
        patches.append({
            "patch_id": index, "box_xyxy": list(box),
            "stage": "coarse", "parent_patch_id": None,
            "probability_when_occluded": score["probability_fake"],
            "aigc_confidence_when_occluded": score["aigc_confidence"],
            "confidence_contribution": prediction["aigc_confidence"] - score["aigc_confidence"],
            "calibrated_probability_contribution": prediction["probability_fake"] - score["probability_fake"],
            "raw_logit_when_occluded": score["raw_logit"],
            "raw_logit_contribution": prediction["raw_logit"] - score["raw_logit"],
        })
    refine_count = min(max(0, args.refine_top_k), len(patches))
    selected = sorted(
        range(len(patches)),
        key=lambda index: abs(patches[index]["raw_logit_contribution"]),
        reverse=True,
    )[:refine_count]
    refined_boxes, refined_parents = [], []
    for parent in selected:
        children = subdivide_box(coarse_boxes[parent], args.refine_grid)
        refined_boxes.extend(children)
        refined_parents.extend([parent] * len(children))
    if refined_boxes:
        refined_images = [occlude_patch(image, box, args.occlusion) for box in refined_boxes]
        refined_scores, _ = score_images(
            model,
            refined_images,
            config,
            device,
            temperature,
            bias,
            args.batch_size,
            decision_threshold,
        )
        for box, parent, score in zip(refined_boxes, refined_parents, refined_scores):
            patches.append({
                "patch_id": len(patches), "box_xyxy": list(box),
                "stage": "refined", "parent_patch_id": parent,
                "probability_when_occluded": score["probability_fake"],
                "aigc_confidence_when_occluded": score["aigc_confidence"],
                "confidence_contribution": prediction["aigc_confidence"] - score["aigc_confidence"],
                "calibrated_probability_contribution": prediction["probability_fake"] - score["probability_fake"],
                "raw_logit_when_occluded": score["raw_logit"],
                "raw_logit_contribution": prediction["raw_logit"] - score["raw_logit"],
            })
    boxes = [tuple(row["box_xyxy"]) for row in patches]
    region_weights = [1.0 if row["stage"] == "coarse" else 2.0 for row in patches]

    frequency_images = [suppress_high_frequency_patch(image, box) for box in boxes]
    frequency_total_scores, _ = score_images(
        model,
        frequency_images,
        config,
        device,
        temperature,
        bias,
        args.batch_size,
        decision_threshold,
    )
    wavelet_base, wavelet_only_scores = score_wavelet_only_counterfactuals(
        model, image, frequency_images, config, device, temperature, bias, args.batch_size
    )
    for row, total_score, isolated_score in zip(patches, frequency_total_scores, wavelet_only_scores):
        row.update({
            "high_frequency_ablation": {
                "total_raw_logit_when_suppressed": total_score["raw_logit"],
                "total_raw_logit_contribution": prediction["raw_logit"] - total_score["raw_logit"],
                "wavelet_similarity_when_suppressed": isolated_score["wavelet_similarity"],
                "wavelet_similarity_contribution": wavelet_base["wavelet_similarity"] - isolated_score["wavelet_similarity"],
                "wavelet_only_raw_logit_when_suppressed": isolated_score["raw_logit"],
                "wavelet_only_raw_logit_contribution": wavelet_base["raw_logit"] - isolated_score["raw_logit"],
            }
        })

    transform_images, transforms = [], []
    for variant, operation, value in competition_grid():
        transform_images.append(apply_fixed(image, operation, value, seed=0))
        transforms.append({"variant": variant, "operation": operation, "value": value})
    transform_scores, _ = score_images(
        model,
        transform_images,
        config,
        device,
        temperature,
        bias,
        args.batch_size,
        decision_threshold,
    )
    for row, score in zip(transforms, transform_scores):
        row.update(score)

    branches = branch_counterfactuals(
        model, image, config, device, temperature, bias, decision_threshold
    )
    expert_details = []
    if "expert_gate" in model_outputs:
        gates = model_outputs["expert_gate"].tolist()
        logits = model_outputs["expert_logits"].tolist()
        for name_, gate, logit in zip(("spatial", "semantic", "native_frequency"), gates, logits):
            expert_details.append({"expert": name_, "gate": gate, "raw_logit": logit, "weighted_logit_contribution": gate * logit})
    tile_details = []
    if "tile_attention" in model_outputs:
        tile_logits = model_outputs["tile_logits"].tolist()
        for index, (attention, logit) in enumerate(zip(model_outputs["tile_attention"].tolist(), tile_logits)):
            tile_details.append({"tile": index, "attention": attention, "raw_logit": logit})

    args.output.mkdir(parents=True, exist_ok=True)
    raw_contributions = [row["raw_logit_contribution"] for row in patches]
    vmin, vmax = save_dense_signed_heatmaps(
        image,
        boxes,
        raw_contributions,
        args.output / "heatmap_attribution.png",
        args.output / "heatmap_attribution_overlay.png",
        weights=region_weights,
    )
    frequency_contributions = [
        row["high_frequency_ablation"]["wavelet_only_raw_logit_contribution"] for row in patches
    ]
    frequency_vmin, frequency_vmax = save_dense_signed_heatmaps(
        image,
        boxes,
        frequency_contributions,
        args.output / "heatmap_frequency_contribution.png",
        args.output / "heatmap_frequency_overlay.png",
        weights=region_weights,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "method": (
                "fusion_v2_three_expert"
                if expert_details
                else "773086_759921_architecture_mlp_normalized"
                if config.get("loss", {}).get("mode") == "mlp_normalized"
                else "759921_hybrid_legacy"
            ),
            "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
            "calibration": str(args.calibration) if args.calibration else None,
        },
        "image": {"path": str(args.image), "width": image.width, "height": image.height, "sha256": sha256(args.image)},
        "prediction": prediction,
        "attribution": {
            "method": f"hierarchical_native_image_patch_occlusion_{args.occlusion}",
            "semantics": "Primary map uses raw_logit_contribution = logit(original) - logit(region occluded), avoiding sigmoid saturation. Probability deltas are retained. Refined regions are selected only by coarse absolute raw-logit contribution. This is a local counterfactual, not causal proof.",
            "coarse_grid": [args.grid, args.grid],
            "refinement": {"top_k": refine_count, "local_grid": [args.refine_grid, args.refine_grid], "selected_parent_patch_ids": selected},
            "patches": patches,
        },
        "frequency_attribution": {
            "method": "local_low_pass_with_wavelet_only_fusion_counterfactual",
            "semantics": "Each region is locally low-passed and forwarded through the real model. For wavelet_only_raw_logit_contribution, a temporary inference hook fixes every legacy fusion feature except the model-computed wavelet_similarity scalar. No layer, parameter, architecture, or checkpoint is changed.",
            "base_wavelet_similarity": wavelet_base["wavelet_similarity"],
            "base_raw_logit": wavelet_base["raw_logit"],
        },
        "experts": expert_details, "branches": branches, "tiles": tile_details, "transforms": transforms,
        "visualizations": {
            "heatmap_attribution": {
                "file": "heatmap_attribution.png",
                "overlay_file": "heatmap_attribution_overlay.png",
                "signal": "raw_logit_contribution = logit(original) - logit(region occluded)",
                "colormap": "coolwarm",
                "normalization": "symmetric 99th-percentile per image",
                "value_range": [vmin, vmax],
            },
            "heatmap_texture": {
                "file": "heatmap_texture.png",
                "signal": "input-only per-pixel high-frequency residual magnitude; independent of model output",
                "colormap": "inferno",
                "normalization": "99th percentile",
            },
            "heatmap_frequency_contribution": {
                "file": "heatmap_frequency_contribution.png",
                "overlay_file": "heatmap_frequency_overlay.png",
                "signal": "wavelet-only raw-logit contribution under local high-frequency suppression",
                "colormap": "coolwarm",
                "normalization": "symmetric 99th-percentile per image",
                "value_range": [frequency_vmin, frequency_vmax],
            },
        },
    }
    image.save(args.output / "input.png")
    save_attribution_overlay(image, boxes, raw_contributions, args.output / "patch_attribution.png")
    save_colorbar(args.output / "heatmap_colorbar.png", vmin=vmin, vmax=vmax, center=0.0)
    save_colorbar(
        args.output / "frequency_colorbar.png",
        vmin=frequency_vmin,
        vmax=frequency_vmax,
        center=0.0,
    )
    save_texture_heatmap(image, args.output / "heatmap_texture.png")
    save_line_svg(transforms, args.output / "transform_trajectory.svg", value_key="aigc_confidence", title="Linearized AIGC confidence under competition transforms")
    component_items = (
        [(row["expert"], row["weighted_logit_contribution"]) for row in expert_details]
        if expert_details else [(row["component"], row["confidence_delta"]) for row in branches]
    )
    save_bar_svg(component_items, args.output / "components.svg", title="Expert weighted logits" if expert_details else "Branch neutralization confidence deltas")
    (args.output / "explanation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    with (args.output / "patches.jsonl").open("w") as handle:
        for row in patches:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_schema(args.output / "schema.json")
    write_dashboard(args.output, payload, component_title="专家贡献" if expert_details else "分支反事实贡献")
    print(json.dumps({"event": "explanation_complete", "output": str(args.output), "probability_fake": prediction["probability_fake"], "coarse_regions": len(coarse_boxes), "refined_regions": len(refined_boxes), "regions": len(patches)}))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate patch/expert/robustness explanations for one image")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--grid", type=int, default=6, help="coarse grid size")
    parser.add_argument("--refine-top-k", type=int, default=6, help="coarse cells refined by absolute raw-logit contribution")
    parser.add_argument("--refine-grid", type=int, default=3, help="local subdivisions inside every selected coarse cell")
    parser.add_argument("--occlusion", choices=("blur", "mean"), default="blur")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    generate_explanation(parser.parse_args())


if __name__ == "__main__":
    main()
