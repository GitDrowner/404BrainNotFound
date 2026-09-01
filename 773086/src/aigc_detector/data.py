from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, WeightedRandomSampler

from .augmentations import (
    CompetitionDegradation,
    native_spectral_signature,
    native_tiles,
    resize_tensor,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        image_size: int,
        semantic_image_size: int,
        num_tiles: int,
        training: bool,
        clean_probability: float = 0.2,
        compound_probability: float = 0.3,
        max_samples: int | None = None,
    ) -> None:
        self.manifest = Path(manifest)
        with self.manifest.open() as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if training:
            forbidden_component = "external_eval_only"
            manifest_parts = {part.lower() for part in self.manifest.parts}
            record_paths = [
                str(record.get("path", "")).lower().replace("\\", "/")
                for record in self.records
            ]
            if forbidden_component in manifest_parts or any(
                f"/{forbidden_component}/" in f"/{path.strip('/')}/" for path in record_paths
            ):
                raise RuntimeError(
                    "Evaluation-only data cannot be loaded with training=True: "
                    f"{self.manifest}"
                )
        if max_samples is not None:
            rng = random.Random(0)
            rng.shuffle(self.records)
            self.records = self.records[:max_samples]
        self.image_size = image_size
        self.semantic_image_size = semantic_image_size
        self.num_tiles = num_tiles
        self.training = training
        self.degrade = CompetitionDegradation(clean_probability, compound_probability)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        with Image.open(record["path"]) as handle:
            clean = handle.convert("RGB").copy()
        degraded, degradation = self.degrade(clean) if self.training else (clean, None)
        if degradation is None:
            from .augmentations import Degradation

            degradation = Degradation("clean", 0, 0.0, ({"transform": "clean"},))
        return {
            "clean_global": resize_tensor(clean, self.image_size),
            "clean_semantic_view": resize_tensor(clean, self.semantic_image_size),
            "global_view": resize_tensor(degraded, self.image_size),
            "semantic_view": resize_tensor(degraded, self.semantic_image_size),
            "tiles": native_tiles(degraded, self.image_size, self.num_tiles, self.training),
            "native_spectral": native_spectral_signature(degraded),
            "clean_native_spectral": native_spectral_signature(clean),
            "label": torch.tensor(float(record["label"]), dtype=torch.float32),
            "degradation_class": torch.tensor(degradation.class_id, dtype=torch.long),
            "degradation_severity": torch.tensor(degradation.severity, dtype=torch.float32),
            "degradation_name": degradation.name,
            "augmentation": json.dumps(
                {"policy": degradation.name, "operations": degradation.operations},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "path": record["path"],
            "source": record.get("source", "unknown"),
            "generator": record.get("generator", "unknown"),
        }


def create_loaders(config: dict) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    section = config["data"]
    common = dict(
        image_size=section["image_size"],
        semantic_image_size=section["semantic_image_size"],
        num_tiles=section["num_tiles"],
        clean_probability=section.get("clean_probability", 0.2),
        compound_probability=section.get("compound_probability", 0.3),
    )
    train_dataset = ManifestDataset(
        section["train_manifest"],
        training=True,
        max_samples=section.get("max_train_samples"),
        **common,
    )
    val_dataset = ManifestDataset(
        section["val_manifest"],
        training=False,
        max_samples=section.get("max_val_samples"),
        **common,
    )
    test_dataset = ManifestDataset(
        section["test_manifest"],
        training=False,
        max_samples=section.get("max_test_samples"),
        **common,
    )
    loader_options = dict(
        batch_size=section["batch_size"],
        num_workers=section["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=section["num_workers"] > 0,
    )
    if section.get("balanced_sampler", False):
        grouping = section.get("balanced_sampler_grouping", "label_source")
        if grouping == "label":
            groups = [str(int(record["label"])) for record in train_dataset.records]
        elif grouping == "label_source":
            groups = [
                f"{int(record['label'])}::{record.get('source', 'unknown')}"
                for record in train_dataset.records
            ]
        else:
            raise ValueError(f"Unknown balanced_sampler_grouping: {grouping}")
        counts = Counter(groups)
        weights = torch.as_tensor(
            [1.0 / counts[group] for group in groups], dtype=torch.double
        )
        generator = torch.Generator().manual_seed(int(config["seed"]))
        sampler = WeightedRandomSampler(
            weights, len(weights), replacement=True, generator=generator
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=sampler, shuffle=False, drop_last=True, **loader_options
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, drop_last=True, **loader_options
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, drop_last=False, **loader_options
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, drop_last=False, **loader_options
    )
    return train_loader, val_loader, test_loader


def create_evaluation_loader(
    config: dict,
    manifest: str | Path,
    *,
    max_samples: int | None = None,
) -> torch.utils.data.DataLoader:
    """Create a deterministic, inference-only loader for checkpoint selection."""
    section = config["data"]
    dataset = ManifestDataset(
        manifest,
        image_size=section["image_size"],
        semantic_image_size=section["semantic_image_size"],
        num_tiles=section["num_tiles"],
        training=False,
        clean_probability=section.get("clean_probability", 0.2),
        compound_probability=section.get("compound_probability", 0.3),
        max_samples=max_samples,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=section["batch_size"],
        num_workers=section["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=section["num_workers"] > 0,
        shuffle=False,
        drop_last=False,
    )
