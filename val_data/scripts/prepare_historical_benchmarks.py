#!/usr/bin/env python3
"""Prepare the frozen COCO/DALL-E 3 and COCO/MidJourney benchmarks.

The two benchmarks are written to different directories and manifests. They are
external evaluation data: never training, checkpoint selection, hard mining, or
calibration data.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image


IMAGE_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
POLICY = "one dataset per manifest; never pool samples or metrics"
FORBIDDEN_TRAINING_FRAGMENTS = (
    "dall",
    "defactify",
    "openfake",
    "midjourney",
    "coco_real_test",
    "ms-coco-unique",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def image_bytes(value: dict[str, Any]) -> bytes:
    if value.get("bytes") is not None:
        return value["bytes"]
    if value.get("path"):
        return Path(value["path"]).read_bytes()
    raise ValueError("Image field has neither bytes nor path")


def image_metadata(payload: bytes) -> tuple[str, int, int, str]:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        image_format = (image.format or "UNKNOWN").upper()
        width, height = image.size
    return IMAGE_EXTENSIONS.get(image_format, ".img"), width, height, image_format


def dhash_bytes(payload: bytes) -> int:
    with Image.open(io.BytesIO(payload)) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


class BKNode:
    def __init__(self, value: int, payload: str) -> None:
        self.value = value
        self.payloads = [payload]
        self.children: dict[int, BKNode] = {}

    def add(self, value: int, payload: str) -> None:
        distance = (self.value ^ value).bit_count()
        if distance == 0:
            self.payloads.append(payload)
        elif distance in self.children:
            self.children[distance].add(value, payload)
        else:
            self.children[distance] = BKNode(value, payload)

    def query(self, value: int, radius: int, output: list[tuple[int, str]]) -> None:
        distance = (self.value ^ value).bit_count()
        if distance <= radius:
            output.extend((distance, payload) for payload in self.payloads)
        for edge, child in self.children.items():
            if distance - radius <= edge <= distance + radius:
                child.query(value, radius, output)


def resolve_record_path(record: dict[str, Any], image_root: Path) -> Path:
    path = Path(str(record["path"])).expanduser()
    return path if path.is_absolute() else image_root / path


def collect_blockers(
    training_manifests: list[Path],
    external_manifests: list[Path],
    image_root: Path,
) -> tuple[set[str], dict[str, int], BKNode | None, list[str]]:
    blocked: set[str] = set()
    counts: dict[str, int] = {}
    tree: BKNode | None = None
    training_names: set[str] = set()
    for manifest in [*training_manifests, *external_manifests]:
        records = read_jsonl(manifest)
        counts[str(manifest)] = len(records)
        for record in records:
            digest = record.get("sha256")
            if not digest:
                raise RuntimeError(f"Manifest lacks sha256: {manifest}")
            blocked.add(str(digest))
    for manifest in training_manifests:
        for record in read_jsonl(manifest):
            training_names.update(
                str(record.get(field, "")).casefold()
                for field in ("source", "generator", "dataset_repo")
            )
            path = resolve_record_path(record, image_root)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Training image needed for perceptual leakage gate: {path}. "
                    "Set --manifest-image-root to the directory against which manifest paths resolve."
                )
            value = dhash_bytes(path.read_bytes())
            if tree is None:
                tree = BKNode(value, str(path))
            else:
                tree.add(value, str(path))
    forbidden = sorted(
        name
        for name in training_names
        if any(fragment in name for fragment in FORBIDDEN_TRAINING_FRAGMENTS)
    )
    if forbidden:
        raise RuntimeError(f"Training manifest contains a historical holdout source: {forbidden}")
    return blocked, counts, tree, forbidden


def persist(
    root: Path,
    key: str,
    payload: bytes,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    extension, width, height, image_format = image_metadata(payload)
    relative = Path("images") / f"{key}{extension}"
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256_file(destination) != sha256_bytes(payload):
        raise RuntimeError(f"Refusing to overwrite different content: {destination}")
    if not destination.exists():
        destination.write_bytes(payload)
    return {
        **metadata,
        "path": str(destination),
        "sha256": sha256_bytes(payload),
        "dhash64": f"{dhash_bytes(payload):016x}",
        "width": width,
        "height": height,
        "format": image_format,
        "role": "external_evaluation_only",
    }


def finalise(
    root: Path,
    benchmark_id: str,
    records: list[dict[str, Any]],
    source: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not records or {str(row["benchmark_id"]) for row in records} != {benchmark_id}:
        raise RuntimeError(f"Invalid or mixed records in {benchmark_id}")
    hashes = [str(row["sha256"]) for row in records]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError(f"Duplicate image content in {benchmark_id}")
    manifest = root / "manifest.jsonl"
    write_jsonl(manifest, records)
    labels = Counter(int(row["label"]) for row in records)
    generators = Counter(str(row["generator"]) for row in records)
    receipt = {
        "benchmark_id": benchmark_id,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "records": len(records),
        "real": labels[0],
        "fake": labels[1],
        "supports_auroc": bool(labels[0] and labels[1]),
        "fake_only": bool(labels[1] and not labels[0]),
        "generators": dict(sorted(generators.items())),
        "seen_in_training": 0,
        "source": source,
        "training": False,
        "checkpoint_selection": False,
        "calibration": False,
    }
    if extra:
        receipt.update(extra)
    (root / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


def prepare_dalle3(
    config: dict[str, Any],
    output_root: Path,
    blocked: set[str],
    training_tree: BKNode | None,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    benchmark_id = str(config["benchmark_id"])
    root = output_root / benchmark_id
    root.mkdir(parents=True, exist_ok=True)
    (root / ".EVAL_ONLY_DO_NOT_TRAIN").touch()
    raw_root = root / "raw_validation_only"
    parquet_paths = [
        Path(
            hf_hub_download(
                repo_id=config["repo"],
                repo_type="dataset",
                revision=config["revision"],
                filename=filename,
                local_dir=raw_root,
            )
        )
        for filename in config["parquet_files"]
    ]
    reals: list[dict[str, Any]] = []
    fakes_by_caption: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    global_index = 0
    for parquet_path in parquet_paths:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(columns=["Caption", "Image", "Label_A", "Label_B"]):
            for row in batch.to_pylist():
                row["hf_validation_row"] = global_index
                global_index += 1
                labels = (int(row["Label_A"]), int(row["Label_B"]))
                if labels == (int(config["real_label_a"]), int(config["real_label_b"])):
                    reals.append(row)
                elif labels == (int(config["fake_label_a"]), int(config["dalle3_label_b"])):
                    fakes_by_caption[str(row["Caption"])].append(row)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for real in reals:
        candidates = fakes_by_caption.get(str(real["Caption"]))
        if candidates:
            pairs.append((real, candidates.popleft()))
        if len(pairs) == int(config["caption_pairs"]):
            break
    if len(pairs) != int(config["caption_pairs"]):
        raise RuntimeError(f"Only {len(pairs)} caption pairs found")

    radius = int(config["near_duplicate_dhash_radius"])
    quarantine: dict[int, list[dict[str, Any]]] = defaultdict(list)
    accepted_pairs: list[tuple[int, tuple[dict[str, Any], dict[str, Any]]]] = []
    for pair_id, pair in enumerate(pairs):
        reject = False
        for label, row in enumerate(pair):
            payload = image_bytes(row["Image"])
            digest = sha256_bytes(payload)
            if digest in blocked:
                quarantine[pair_id].append({"label": label, "reason": "exact_sha256"})
                reject = True
            if training_tree is not None:
                matches: list[tuple[int, str]] = []
                training_tree.query(dhash_bytes(payload), radius, matches)
                if matches:
                    quarantine[pair_id].append(
                        {
                            "label": label,
                            "reason": "training_near_duplicate",
                            "matches": [
                                {"distance": distance, "path": path}
                                for distance, path in sorted(matches)[:20]
                            ],
                        }
                    )
                    reject = True
        if not reject:
            accepted_pairs.append((pair_id, pair))

    records: list[dict[str, Any]] = []
    for pair_id, pair in accepted_pairs:
        for label, row, generator in (
            (0, pair[0], "coco_real"),
            (1, pair[1], "dalle3"),
        ):
            records.append(
                persist(
                    root,
                    f"{'fake' if label else 'real'}_{pair_id:05d}",
                    image_bytes(row["Image"]),
                    {
                        "label": label,
                        "source": "defactify_validation",
                        "generator": generator,
                        "benchmark_id": benchmark_id,
                        "caption_pair_id": pair_id,
                        "caption": str(row["Caption"]),
                        "dataset_repo": config["repo"],
                        "dataset_revision": config["revision"],
                        "dataset_split": config["split"],
                        "dataset_row": int(row["hf_validation_row"]),
                        "seen_in_training": False,
                    },
                )
            )
    return finalise(
        root,
        benchmark_id,
        records,
        {
            "repo": config["repo"],
            "revision": config["revision"],
            "split": config["split"],
            "license": config["license"],
        },
        {
            "candidate_caption_pairs": len(pairs),
            "quarantined_caption_pairs": len(quarantine),
            "quarantine": dict(sorted(quarantine.items())),
            "paired_after_quarantine": True,
        },
    )


def download_parquet(repo: str, config: str, split: str, shard: int, target: Path) -> Path:
    if target.is_file() and target.stat().st_size:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with requests.get(
        f"https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}/{shard}.parquet",
        timeout=300,
        stream=True,
    ) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(target)
    return target


def reservoir(paths: list[Path], count: int, seed: int) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    seen = 0
    row_index = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["image"], batch_size=16, use_threads=False):
            for row in batch.to_pylist():
                item = {"image": row["image"], "dataset_row": row_index}
                row_index += 1
                seen += 1
                if len(selected) < count:
                    selected.append(item)
                else:
                    replacement = rng.randrange(seen)
                    if replacement < count:
                        selected[replacement] = item
    rng.shuffle(selected)
    return selected


def prepare_coco_midjourney(
    config: dict[str, Any], output_root: Path, blocked: set[str]
) -> dict[str, Any]:
    benchmark_id = str(config["benchmark_id"])
    root = output_root / benchmark_id
    root.mkdir(parents=True, exist_ok=True)
    (root / ".EVAL_ONLY_DO_NOT_TRAIN").touch()
    target = int(config["sample_per_class"])
    records: list[dict[str, Any]] = []
    local_hashes: set[str] = set()
    source_receipts: list[dict[str, Any]] = []
    for source in config["sources"]:
        remote_sha = str(HfApi().dataset_info(source["repo"]).sha)
        if remote_sha != source["revision"]:
            raise RuntimeError(
                f"Revision drift for {source['repo']}: expected {source['revision']}, got {remote_sha}"
            )
        parquet_paths = [
            download_parquet(
                source["repo"], source["config"], source["split"], int(shard),
                root / "raw_parquet" / source["name"] / f"{int(shard):04d}.parquet",
            )
            for shard in source["shards"]
        ]
        candidates = reservoir(parquet_paths, max(target + 100, int(target * 1.15)), int(source["sampling_seed"]))
        accepted = 0
        invalid = duplicate_or_blocked = 0
        for row in candidates:
            try:
                payload = image_bytes(row["image"])
                image_metadata(payload)
            except (OSError, ValueError):
                invalid += 1
                continue
            digest = sha256_bytes(payload)
            if digest in blocked or digest in local_hashes:
                duplicate_or_blocked += 1
                continue
            records.append(
                persist(
                    root,
                    f"{source['name']}_{accepted:05d}",
                    payload,
                    {
                        "label": int(source["label"]),
                        "source": source["name"],
                        "generator": source["generator"],
                        "benchmark_id": benchmark_id,
                        "dataset_repo": source["repo"],
                        "dataset_revision": source["revision"],
                        "dataset_split": source["split"],
                        "dataset_row": int(row["dataset_row"]),
                        "seen_in_training": False,
                    },
                )
            )
            local_hashes.add(digest)
            accepted += 1
            if accepted == target:
                break
        if accepted != target:
            raise RuntimeError(f"{source['name']}: only {accepted}/{target} accepted")
        source_receipts.append(
            {
                "name": source["name"],
                "repo": source["repo"],
                "revision": source["revision"],
                "selected": accepted,
                "invalid_skipped": invalid,
                "blocked_or_duplicate_skipped": duplicate_or_blocked,
                "parquet_sha256": [sha256_file(path) for path in parquet_paths],
            }
        )
    random.Random(20260829).shuffle(records)
    return finalise(
        root,
        benchmark_id,
        records,
        {
            "repos": [source["repo"] for source in config["sources"]],
            "revisions": [source["revision"] for source in config["sources"]],
            "split": "train (upstream split; external-evaluation-only locally)",
            "license": config["license"],
        },
        {"source_receipts": source_receipts},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/historical_benchmarks.json"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--training-manifest", type=Path, action="append", default=[])
    parser.add_argument("--blocked-manifest", type=Path, action="append", default=[])
    parser.add_argument("--manifest-image-root", type=Path, default=Path.cwd())
    parser.add_argument("--allow-empty-blocklist", action="store_true")
    parser.add_argument(
        "--only",
        choices=["all", "dalle3", "coco-midjourney"],
        default="all",
        help="Prepare both historical benchmarks or one named benchmark.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not args.training_manifest and not args.blocked_manifest and not args.allow_empty_blocklist:
        raise RuntimeError(
            "No contamination manifests supplied. Provide --training-manifest and/or "
            "--blocked-manifest, or explicitly mark a non-audited copy with --allow-empty-blocklist."
        )
    if args.only in {"all", "dalle3"} and not args.training_manifest and not args.allow_empty_blocklist:
        raise RuntimeError(
            "DALL-E preparation requires --training-manifest so the dHash near-duplicate gate can run."
        )
    root = (args.output_root or Path(config["output_root"])).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / ".EVAL_ONLY_DO_NOT_TRAIN").touch()
    effective_config = root / "effective_config.json"
    effective_config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    blocked, blocked_counts, training_tree, _ = collect_blockers(
        args.training_manifest, args.blocked_manifest, args.manifest_image_root.resolve()
    )
    entries: list[dict[str, Any]] = []
    if args.only in {"all", "dalle3"}:
        entries.append(prepare_dalle3(config["datasets"]["dalle3_advanced"], root, blocked, training_tree))
    if args.only in {"all", "coco-midjourney"}:
        entries.append(prepare_coco_midjourney(config["datasets"]["coco_midjourney"], root, blocked))
    index = {
        "status": "PASS",
        "policy": POLICY,
        "root": str(root),
        "config": str(effective_config),
        "config_sha256": sha256_file(effective_config),
        "blocked_manifests": blocked_counts,
        "blocked_sha256_count": len(blocked),
        "audit_status": (
            "non-audited-empty-blocklist"
            if not args.training_manifest and not args.blocked_manifest
            else "audited-with-supplied-blocklists"
        ),
        "training": False,
        "checkpoint_selection": False,
        "calibration": False,
        "benchmarks": entries,
    }
    index_path = root / "benchmark_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"event": "historical_benchmarks_ready", **index}, ensure_ascii=False))


if __name__ == "__main__":
    main()
