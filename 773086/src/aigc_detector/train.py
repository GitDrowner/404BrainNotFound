from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from .data import create_evaluation_loader, create_loaders
from .augmentations import DEGRADATION_NAMES
from .losses import (
    ClassSourceGroupDRO,
    NormalizedMLPLoss,
    UncertaintyWeightedLoss,
    trace_loss,
    trace_loss_terms,
)
from .metrics import binary_metrics
from .model import TraceDetector


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def write_augmentation_audit(
    destination: Path,
    batch: dict,
    *,
    split: str,
    epoch: int,
    batch_index: int,
    optimizer_step: int | None = None,
) -> None:
    """Append one exact augmentation record per image without bloating loss logs."""
    labels = batch["label"].detach().cpu().tolist()
    paths = list(batch["path"])
    sources = list(batch.get("source", ["unknown"] * len(paths)))
    generators = list(batch.get("generator", ["unknown"] * len(paths)))
    augmentations = list(batch.get("augmentation", ['{"policy":"unknown","operations":[]}'] * len(paths)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a") as handle:
        for index, path in enumerate(paths):
            record = {
                "split": split,
                "epoch": epoch,
                "batch": batch_index,
                "optimizer_step": optimizer_step,
                "path": path,
                "label": int(labels[index]),
                "source": sources[index],
                "generator": generators[index],
                "augmentation": json.loads(augmentations[index]),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def autocast_context(device: torch.device, amp: str):
    if device.type not in {"cuda", "mps"} or amp == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_optimizer(
    model: TraceDetector,
    config: dict,
    loss_balancer: torch.nn.Module | None = None,
) -> AdamW:
    main, lora = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (lora if "lora_" in name else main).append(parameter)
    groups = [{"params": main, "lr": config["train"]["learning_rate"]}]
    if lora:
        groups.append({"params": lora, "lr": config["train"]["lora_learning_rate"]})
    if loss_balancer is not None:
        groups.append(
            {
                "params": list(loss_balancer.parameters()),
                "lr": float(
                    config["loss"].get(
                        "controller_learning_rate",
                        config["loss"].get("learning_rate", 0.002),
                    )
                ),
                "weight_decay": 0.0,
            }
        )
    return AdamW(groups, weight_decay=config["train"]["weight_decay"])


def lr_multiplier(step: int, total_steps: int, warmup_ratio: float) -> float:
    warmup_steps = max(1, round(total_steps * warmup_ratio))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(
    model: TraceDetector,
    loader,
    device: torch.device,
    amp: str,
    max_steps: int | None,
    *,
    audit_path: Path | None = None,
    split: str = "validation",
    epoch: int = -1,
    loss_balancer: torch.nn.Module | None = None,
) -> dict[str, float]:
    model.eval()
    balancer_was_training = loss_balancer.training if loss_balancer is not None else False
    if loss_balancer is not None:
        loss_balancer.eval()
    labels, probabilities, sources = [], [], []
    total_loss = 0.0
    steps = 0
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        batch = move_batch(batch, device)
        if audit_path is not None:
            write_augmentation_audit(
                audit_path,
                batch,
                split=split,
                epoch=epoch,
                batch_index=step,
            )
        with autocast_context(device, amp):
            outputs = model(batch)
            if loss_balancer is None:
                loss, _ = trace_loss(outputs, batch, DEFAULT_LOSS_WEIGHTS)
            else:
                raw_terms = trace_loss_terms(outputs, batch)
                loss, _ = loss_balancer(raw_terms, learn=False)
        total_loss += float(loss)
        labels.extend(batch["label"].cpu().tolist())
        probabilities.extend(torch.sigmoid(outputs["logits"]).float().cpu().tolist())
        sources.extend(str(source) for source in batch.get("source", []))
        steps += 1
    metrics = binary_metrics(labels, probabilities)
    metrics["loss"] = total_loss / max(steps, 1)
    if len(set(int(label) for label in labels)) == 2:
        y = np.asarray(labels, dtype=np.int64)
        score = np.asarray(probabilities, dtype=np.float64)
        candidates = np.unique(np.concatenate(([0.0], score, [1.0])))
        best = (-1.0, -1.0, -1.0, 0.5)
        for threshold in candidates:
            prediction = score >= threshold
            real_recall = float(np.mean(~prediction[y == 0]))
            fake_recall = float(np.mean(prediction[y == 1]))
            candidate = (
                min(real_recall, fake_recall),
                0.5 * (real_recall + fake_recall),
                -abs(float(threshold) - 0.5),
                float(threshold),
            )
            if candidate[:3] > best[:3]:
                best = candidate
        selected_threshold = best[3]
        selected_prediction = score >= selected_threshold
        source_recalls = []
        source_array = np.asarray(sources, dtype=object)
        for source in sorted(set(sources)):
            mask = source_array == source
            source_recalls.append(float(np.mean(selected_prediction[mask] == y[mask])))
        metrics.update(
            {
                "balanced_threshold": selected_threshold,
                "worst_class_recall": best[0],
                "balanced_accuracy_at_balanced_threshold": best[1],
                "worst_source_recall_at_balanced_threshold": min(source_recalls),
            }
        )
    if loss_balancer is not None and balancer_was_training:
        loss_balancer.train()
    return metrics


DEFAULT_LOSS_WEIGHTS: dict[str, float] = {}


def train(config: dict, max_steps: int | None = None) -> dict:
    global DEFAULT_LOSS_WEIGHTS
    DEFAULT_LOSS_WEIGHTS = config["loss"]
    seed_everything(config["seed"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = create_loaders(config)
    evaluation_config = config.get("evaluation", {})
    selection_manifest = evaluation_config.get("selection_manifest")
    selection_loader = (
        create_evaluation_loader(
            config,
            selection_manifest,
            max_samples=evaluation_config.get("max_selection_samples"),
        )
        if selection_manifest
        else None
    )
    model = TraceDetector(
        config,
        image_size=config["data"]["image_size"],
        semantic_image_size=config["data"]["semantic_image_size"],
    ).to(device)
    initial_checkpoint_path = config["train"].get("initial_checkpoint")
    if initial_checkpoint_path:
        initial_checkpoint = torch.load(
            initial_checkpoint_path, map_location="cpu", weights_only=False
        )
        incompatible = model.load_state_dict(initial_checkpoint["model"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected initial checkpoint keys: {incompatible.unexpected_keys}"
            )
        print(
            json.dumps(
                {
                    "event": "initial_checkpoint_loaded",
                    "path": str(initial_checkpoint_path),
                    "sha256": sha256_path(Path(initial_checkpoint_path)),
                    "source_epoch": initial_checkpoint.get("epoch"),
                    "optimizer_restored": False,
                }
            )
        )
    if config["train"].get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)
    summary = model.trainable_parameter_summary()
    print(json.dumps({"event": "model", "device": str(device), "lora_blocks": model.lora_modules, **summary}))
    loss_balancer = None
    loss_mode = config["loss"].get("mode")
    if loss_mode == "uncertainty":
        loss_balancer = UncertaintyWeightedLoss(config["loss"]).to(device)
        print(
            json.dumps(
                {
                    "event": "uncertainty_weighting_initialized",
                    "classification_anchor": loss_balancer.classification_anchor,
                    "groups": loss_balancer.summary(),
                }
            )
        )
    elif loss_mode == "mlp_normalized":
        loss_balancer = NormalizedMLPLoss(config["loss"]).to(device)
        print(
            json.dumps(
                {
                    "event": "mlp_normalized_weighting_initialized",
                    "classification_anchor": loss_balancer.classification_anchor,
                    "auxiliary_budget": loss_balancer.auxiliary_budget,
                    "groups": loss_balancer.summary(),
                }
            )
        )
    optimizer = make_optimizer(model, config, loss_balancer)
    robust_config = config.get("robust_optimization", {})
    group_dro = None
    if robust_config.get("enabled", False):
        include_degradation = bool(robust_config.get("include_degradation", False))
        groups = [
            ClassSourceGroupDRO.group_name(
                int(record["label"]),
                str(record.get("source", "unknown")),
                degradation if include_degradation else None,
            )
            for record in train_loader.dataset.records
            for degradation in (DEGRADATION_NAMES if include_degradation else [None])
        ]
        group_dro = ClassSourceGroupDRO(
            groups,
            eta=float(robust_config["eta"]),
            hard_fraction=float(robust_config["hard_fraction"]),
            hard_weight=float(robust_config["hard_weight"]),
            class_balance_weight=float(robust_config["class_balance_weight"]),
            device=device,
        )
    base_learning_rates = [group["lr"] for group in optimizer.param_groups]
    writer = SummaryWriter(output_dir / "tensorboard")
    history_path = output_dir / "history.jsonl"
    checkpoint_metrics_path = output_dir / "checkpoint_metrics.csv"
    train_augmentation_path = output_dir / "augmentation_train.jsonl"
    validation_augmentation_path = output_dir / "augmentation_validation.jsonl"
    selection_augmentation_path = output_dir / "augmentation_checkpoint_selection.jsonl"
    test_augmentation_path = output_dir / "augmentation_test.jsonl"
    loss_weight_history_path = output_dir / "loss_weight_history.jsonl"
    accumulation = config["train"]["gradient_accumulation"]
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accumulation))
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    total_steps = steps_per_epoch * config["train"]["epochs"]
    uncertainty_warmup_steps = (
        max(1, round(total_steps * float(config["loss"].get("warmup_ratio", 0.10))))
        if loss_balancer is not None
        else 0
    )
    amp = config["train"]["amp"]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp == "fp16")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_score = -float("inf")
    best_epoch = -1
    patience = 0
    global_step = 0
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(config["train"]["epochs"]):
        model.train()
        if loss_balancer is not None:
            loss_balancer.train()
        running = 0.0
        optimizer_steps = 0
        for batch_index, batch in enumerate(train_loader):
            if max_steps is not None and optimizer_steps >= max_steps:
                break
            batch = move_batch(batch, device)
            write_augmentation_audit(
                train_augmentation_path,
                batch,
                split="train",
                epoch=epoch,
                batch_index=batch_index,
                optimizer_step=global_step,
            )
            with autocast_context(device, amp):
                outputs = model(batch)
                raw_terms = trace_loss_terms(outputs, batch)
                robust_classification = None
                robust_terms = {}
                if group_dro is not None:
                    per_sample_classification = F.binary_cross_entropy_with_logits(
                        outputs["logits"], batch["label"], reduction="none"
                    )
                    robust_classification, robust_terms = group_dro(
                        per_sample_classification,
                        batch["label"],
                        batch["source"],
                        batch.get("degradation_name") if include_degradation else None,
                    )
                if loss_balancer is None:
                    loss, terms = trace_loss(outputs, batch, config["loss"])
                    if robust_classification is not None:
                        ordinary_classification = per_sample_classification.mean()
                        loss = loss + float(config["loss"]["classification"]) * (
                            robust_classification - ordinary_classification
                        )
                        terms.update(robust_terms)
                        terms["total"] = float(loss.detach())
                else:
                    uncertainty_learning_enabled = global_step >= uncertainty_warmup_steps
                    loss, uncertainty_terms = loss_balancer(
                        raw_terms,
                        primary_loss=robust_classification,
                        learn=uncertainty_learning_enabled,
                    )
                    terms = {
                        **{name: float(value.detach()) for name, value in raw_terms.items()},
                        **robust_terms,
                        **uncertainty_terms,
                    }
                loss = loss / accumulation
            scaler.scale(loss).backward()
            running += terms["total"]
            if (batch_index + 1) % accumulation != 0:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            if loss_balancer is not None:
                loss_balancer.clamp_parameters()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            optimizer_steps += 1
            multiplier = lr_multiplier(global_step, total_steps, config["train"]["warmup_ratio"])
            for group, base_learning_rate in zip(optimizer.param_groups, base_learning_rates):
                group["lr"] = base_learning_rate * multiplier
            elapsed = time.monotonic() - started
            event = {
                "event": "train",
                "epoch": epoch,
                "step": global_step,
                "loss": terms["total"],
                "epoch_mean_loss": running / max(optimizer_steps, 1),
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "samples_per_second": global_step * config["data"]["batch_size"] * accumulation / max(elapsed, 1e-6),
            }
            if device.type == "cuda":
                event["allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
                event["reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
            event.update({f"loss/{name}": value for name, value in terms.items()})
            with history_path.open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            if loss_balancer is not None:
                with loss_weight_history_path.open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "step": global_step,
                                "mode": loss_mode,
                                "learning_enabled": global_step >= uncertainty_warmup_steps,
                                "groups": loss_balancer.summary(),
                            }
                        )
                        + "\n"
                    )
            for name, value in terms.items():
                writer.add_scalar(f"train/{name}", value, global_step)
            writer.add_scalar("train/epoch_mean_loss", event["epoch_mean_loss"], global_step)
            if global_step % config["train"]["log_every"] == 0:
                print(json.dumps(event))
        val_metrics = validate(
            model,
            val_loader,
            device,
            amp,
            max_steps,
            audit_path=validation_augmentation_path,
            split="validation",
            epoch=epoch,
            loss_balancer=loss_balancer,
        )
        test_metrics = {}
        if config.get("evaluation", {}).get("evaluate_test_each_epoch", True):
            test_metrics = validate(
                model,
                test_loader,
                device,
                amp,
                max_steps,
                audit_path=test_augmentation_path,
                split="test",
                epoch=epoch,
                loss_balancer=loss_balancer,
            )
        selection_metrics = {}
        if selection_loader is not None:
            selection_metrics = validate(
                model,
                selection_loader,
                device,
                amp,
                max_steps,
                audit_path=selection_augmentation_path,
                split=str(evaluation_config.get("selection_name", "checkpoint_selection")),
                epoch=epoch,
                loss_balancer=loss_balancer,
            )
        print(json.dumps({"event": "validation", "epoch": epoch, **val_metrics}))
        print(json.dumps({"event": "test", "epoch": epoch, **test_metrics}))
        if selection_loader is not None:
            print(
                json.dumps(
                    {
                        "event": "checkpoint_selection_validation",
                        "epoch": epoch,
                        "manifest": str(selection_manifest),
                        **selection_metrics,
                    }
                )
            )
        epoch_event = {
            "event": "checkpoint_metrics",
            "epoch": epoch,
            "train_loss": running / max(optimizer_steps, 1),
            "validation": val_metrics,
            "test": test_metrics,
            "checkpoint_selection": selection_metrics,
        }
        if loss_balancer is not None:
            epoch_event["uncertainty_weighting"] = loss_balancer.summary()
        with history_path.open("a") as handle:
            handle.write(json.dumps(epoch_event) + "\n")
        flat_checkpoint_metrics = {
            "epoch": epoch,
            "train_loss": epoch_event["train_loss"],
            **{f"validation_{name}": value for name, value in val_metrics.items()},
            **{f"test_{name}": value for name, value in test_metrics.items()},
            **{
                f"checkpoint_selection_{name}": value
                for name, value in selection_metrics.items()
            },
        }
        with checkpoint_metrics_path.open("a", newline="") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=list(flat_checkpoint_metrics))
            if handle.tell() == 0:
                csv_writer.writeheader()
            csv_writer.writerow(flat_checkpoint_metrics)
        for split, metrics in (("validation", val_metrics), ("test", test_metrics)):
            for name, value in metrics.items():
                writer.add_scalar(f"{split}/{name}", value, epoch)
        selection_metric = evaluation_config.get("selection_metric", "auroc")
        score_metrics = selection_metrics if selection_loader is not None else val_metrics
        score = score_metrics.get(selection_metric, score_metrics["balanced_accuracy"])
        if not math.isfinite(score):
            score = score_metrics["balanced_accuracy"]
        checkpoint = {
            "model": model.trainable_state_dict(),
            "trainable_only": True,
            "config": config,
            "epoch": epoch,
            "metrics": val_metrics,
            "test_metrics": test_metrics,
            "train_loss": running / max(optimizer_steps, 1),
            "selection_metric": selection_metric,
            "selection_score": score,
            "selection_manifest": str(selection_manifest) if selection_manifest else None,
            "checkpoint_selection_metrics": selection_metrics,
        }
        if group_dro is not None:
            checkpoint["group_dro_weights"] = group_dro.weights()
        if loss_balancer is not None:
            balancer_checkpoint = {
                "mode": loss_mode,
                "state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in loss_balancer.state_dict().items()
                },
                "summary": loss_balancer.summary(),
                "warmup_steps": uncertainty_warmup_steps,
            }
            checkpoint["loss_balancer"] = balancer_checkpoint
            if loss_mode == "uncertainty":
                checkpoint["uncertainty_weighting"] = balancer_checkpoint
            elif loss_mode == "mlp_normalized":
                checkpoint["mlp_normalized_weighting"] = balancer_checkpoint
        if config["train"].get("save_every_epoch", True):
            torch.save(checkpoint, output_dir / f"epoch-{epoch:02d}.pt")
        if score > best_score:
            best_score, best_epoch, patience = score, epoch, 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            patience += 1
            if patience >= config["train"]["early_stopping_patience"]:
                break
    result = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "elapsed_seconds": time.monotonic() - started,
        "device": str(device),
        **summary,
    }
    if device.type == "cuda":
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
        result["gpu_name"] = torch.cuda.get_device_name()
    if loss_balancer is not None:
        result["loss_mode"] = loss_mode
        result["loss_balancer"] = loss_balancer.summary()
        if loss_mode == "uncertainty":
            result["uncertainty_weighting"] = loss_balancer.summary()
        elif loss_mode == "mlp_normalized":
            result["mlp_normalized_weighting"] = loss_balancer.summary()
    if selection_manifest:
        result["selection_manifest"] = str(selection_manifest)
        result["selection_metric"] = evaluation_config.get("selection_metric", "auroc")
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    writer.close()
    print(json.dumps({"event": "complete", **result}))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    train(config, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
