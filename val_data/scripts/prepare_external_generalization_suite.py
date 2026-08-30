#!/usr/bin/env python3
"""Download small, revision-pinned, evaluation-only OOD benchmarks.

Every benchmark receives its own manifest. Samples from different datasets are
never concatenated. Exact overlap with training and previously frozen external
holdouts is rejected before a record is admitted.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image


IMAGE_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


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
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def image_metadata(payload: bytes) -> tuple[str, int, int, str]:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        image_format = image.format or "UNKNOWN"
        width, height = image.size
    return IMAGE_EXTENSIONS.get(image_format, ".img"), width, height, image_format


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "unknown"


def normalise_generator(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def collect_blocked_hashes(paths: list[str]) -> tuple[set[str], dict[str, int]]:
    hashes: set[str] = set()
    counts: dict[str, int] = {}
    for raw_path in paths:
        path = Path(raw_path)
        records = read_jsonl(path)
        missing = [record for record in records if not record.get("sha256")]
        if missing:
            raise RuntimeError(f"Manifest lacks sha256 values: {path}")
        hashes.update(str(record["sha256"]) for record in records)
        counts[str(path)] = len(records)
    return hashes, counts


def training_generators(paths: list[str]) -> set[str]:
    return {
        normalise_generator(str(record.get("generator", "")))
        for raw_path in paths
        for record in read_jsonl(Path(raw_path))
        if record.get("generator")
    }


def download_hf_file(repo: str, revision: str, repo_path: str) -> tuple[str, bytes]:
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            cached = Path(
                hf_hub_download(
                    repo_id=repo,
                    filename=repo_path,
                    repo_type="dataset",
                    revision=revision,
                )
            )
            return repo_path, cached.read_bytes()
        except Exception as error:  # Hub errors vary across HTTP/LFS/Xet backends.
            last_error = error
            if attempt == 7:
                break
            delay = min(60.0, 2.0 ** (attempt + 1))
            print(
                json.dumps(
                    {
                        "event": "hf_download_retry",
                        "repo_path": repo_path,
                        "attempt": attempt + 1,
                        "delay_seconds": delay,
                        "error": str(error),
                    }
                ),
                flush=True,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def download_hf_files(repo: str, revision: str, paths: list[str], workers: int) -> dict[str, bytes]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        downloaded = executor.map(
            lambda repo_path: download_hf_file(repo, revision, repo_path), paths
        )
    return dict(downloaded)


def persist_record(
    *,
    payload: bytes,
    benchmark_root: Path,
    image_key: str,
    record: dict[str, Any],
    blocked_hashes: set[str],
) -> dict[str, Any] | None:
    digest = sha256_bytes(payload)
    if digest in blocked_hashes:
        return None
    extension, width, height, image_format = image_metadata(payload)
    relative = Path("images") / f"{safe_name(image_key)}{extension}"
    path = benchmark_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and sha256_file(path) != digest:
        raise RuntimeError(f"Refusing to overwrite a different image: {path}")
    if not path.exists():
        path.write_bytes(payload)
    output = dict(record)
    output.update(
        {
            "path": str(path),
            "sha256": digest,
            "width": width,
            "height": height,
            "format": image_format,
            "role": "external_evaluation_only",
        }
    )
    return output


def prepare_ditfake(
    config: dict[str, Any],
    root: Path,
    blocked: set[str],
    seen_training_generators: set[str],
    seed: int,
    workers: int,
) -> list[dict[str, Any]]:
    api = HfApi()
    files = api.list_repo_files(
        config["repo"], repo_type="dataset", revision=config["revision"]
    )
    index = set(files)
    suite_entries: list[dict[str, Any]] = []
    target = int(config["sample_per_class_per_generator"])
    for benchmark_id, generator in config["generators"].items():
        benchmark_root = root / benchmark_id
        benchmark_root.mkdir(parents=True, exist_ok=True)
        (benchmark_root / ".EVAL_ONLY_DO_NOT_TRAIN").touch()
        records: list[dict[str, Any]] = []
        per_label_counts: Counter[int] = Counter()
        local_hashes: set[str] = set()
        candidate_paths: list[str] = []
        for label, folder in ((0, "0_real"), (1, "1_fake")):
            prefix = f"DiTFake/test/{generator}/{folder}/"
            candidates = sorted(path for path in index if path.startswith(prefix))
            if len(candidates) < target:
                raise RuntimeError(f"{benchmark_id} has only {len(candidates)} candidates for {folder}")
            rng = random.Random(f"{seed}:{benchmark_id}:{label}")
            rng.shuffle(candidates)
            candidate_paths.extend(candidates[: target + 50])
        payloads = download_hf_files(config["repo"], config["revision"], candidate_paths, workers)
        for repo_path in candidate_paths:
            label = int("/1_fake/" in repo_path)
            if per_label_counts[label] >= target:
                continue
            digest = sha256_bytes(payloads[repo_path])
            if digest in blocked or digest in local_hashes:
                continue
            record = persist_record(
                payload=payloads[repo_path],
                benchmark_root=benchmark_root,
                image_key=f"{label}_{Path(repo_path).stem}",
                blocked_hashes=blocked,
                record={
                    "label": label,
                    "source": "ditfake",
                    "generator": generator if label else "coco_real",
                    "benchmark_id": benchmark_id,
                    "dataset_repo": config["repo"],
                    "dataset_revision": config["revision"],
                    "dataset_split": config["split"],
                    "dataset_path": repo_path,
                    "seen_in_training": bool(
                        label and normalise_generator(generator) in seen_training_generators
                    ),
                },
            )
            if record is None:
                continue
            records.append(record)
            local_hashes.add(record["sha256"])
            per_label_counts[label] += 1
        if per_label_counts != Counter({0: target, 1: target}):
            raise RuntimeError(f"Insufficient leakage-free samples for {benchmark_id}: {per_label_counts}")
        suite_entries.append(finalise_benchmark(benchmark_root, benchmark_id, records, config))
    return suite_entries


def fetch_rows(repo: str, config: str, split: str, expected: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset in range(0, expected, 100):
        response = requests.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": repo,
                "config": config,
                "split": split,
                "offset": offset,
                "length": min(100, expected - offset),
            },
            timeout=120,
        )
        response.raise_for_status()
        rows.extend(response.json().get("rows", []))
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows from {repo}, received {len(rows)}")
    return rows


def fetch_url(url: str) -> bytes:
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    return response.content


def prepare_frontier_small(
    config: dict[str, Any],
    root: Path,
    blocked: set[str],
    seen_training_generators: set[str],
    workers: int,
) -> dict[str, Any]:
    benchmark_id = config["benchmark_id"]
    benchmark_root = root / benchmark_id
    benchmark_root.mkdir(parents=True, exist_ok=True)
    (benchmark_root / ".EVAL_ONLY_DO_NOT_TRAIN").touch()
    rows = fetch_rows(
        config["repo"], config["config"], config["split"], int(config["expected_rows"])
    )
    urls = [str(item["row"]["image"]["src"]) for item in rows]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        payloads = list(executor.map(fetch_url, urls))
    records: list[dict[str, Any]] = []
    local_hashes: set[str] = set()
    for item, payload in zip(rows, payloads):
        row = item["row"]
        label = 1 if str(row["label"]).casefold() in {"ai", "fake", "1"} else 0
        digest = sha256_bytes(payload)
        if digest in blocked or digest in local_hashes:
            continue
        generator = str(row.get("generator", "unknown"))
        record = persist_record(
            payload=payload,
            benchmark_root=benchmark_root,
            image_key=f"row_{int(item['row_idx']):04d}_{row.get('filename', '')}",
            blocked_hashes=blocked,
            record={
                "label": label,
                "source": str(row.get("source", "unknown")),
                "generator": generator,
                "benchmark_id": benchmark_id,
                "dataset_repo": config["repo"],
                "dataset_revision": config["revision"],
                "dataset_config": config["config"],
                "dataset_split": config["split"],
                "dataset_row": int(item["row_idx"]),
                "original_filename": row.get("filename"),
                "seen_in_training": bool(
                    label and normalise_generator(generator) in seen_training_generators
                ),
            },
        )
        if record is not None:
            records.append(record)
            local_hashes.add(record["sha256"])
    if len(records) != int(config["expected_rows"]):
        raise RuntimeError(
            f"Blocked/duplicate rows reduced {benchmark_id} to {len(records)}; "
            "the benchmark must be reviewed rather than silently resized"
        )
    return finalise_benchmark(benchmark_root, benchmark_id, records, config)


def prepare_qwen_image_bench(
    config: dict[str, Any],
    root: Path,
    blocked: set[str],
    seen_training_generators: set[str],
    seed: int,
    workers: int,
) -> dict[str, Any]:
    benchmark_id = config["benchmark_id"]
    benchmark_root = root / benchmark_id
    benchmark_root.mkdir(parents=True, exist_ok=True)
    (benchmark_root / ".EVAL_ONLY_DO_NOT_TRAIN").touch()
    files = HfApi().list_repo_files(
        config["repo"], repo_type="dataset", revision=config["revision"]
    )
    target = int(config["sample_per_generator"])
    candidates_by_generator: dict[str, list[str]] = {}
    download_paths: list[str] = []
    for generator in config["generators"]:
        prefix = f"images/{generator}/"
        candidates = sorted(path for path in files if path.startswith(prefix))
        rng = random.Random(f"{seed}:{benchmark_id}:{generator}")
        rng.shuffle(candidates)
        candidates_by_generator[generator] = candidates[: target + 5]
        download_paths.extend(candidates_by_generator[generator])
    payloads = download_hf_files(config["repo"], config["revision"], download_paths, workers)
    records: list[dict[str, Any]] = []
    local_hashes: set[str] = set()
    for generator, candidates in candidates_by_generator.items():
        accepted = 0
        for repo_path in candidates:
            digest = sha256_bytes(payloads[repo_path])
            if digest in blocked or digest in local_hashes:
                continue
            record = persist_record(
                payload=payloads[repo_path],
                benchmark_root=benchmark_root,
                image_key=f"{generator}_{Path(repo_path).stem}",
                blocked_hashes=blocked,
                record={
                    "label": 1,
                    "source": "qwen_image_bench",
                    "generator": generator,
                    "benchmark_id": benchmark_id,
                    "dataset_repo": config["repo"],
                    "dataset_revision": config["revision"],
                    "dataset_split": config["split"],
                    "dataset_path": repo_path,
                    "seen_in_training": normalise_generator(generator)
                    in seen_training_generators,
                },
            )
            if record is None:
                continue
            records.append(record)
            local_hashes.add(record["sha256"])
            accepted += 1
            if accepted == target:
                break
        if accepted != target:
            raise RuntimeError(f"Only {accepted}/{target} accepted for {generator}")
    return finalise_benchmark(benchmark_root, benchmark_id, records, config)


def finalise_benchmark(
    benchmark_root: Path,
    benchmark_id: str,
    records: list[dict[str, Any]],
    source_config: dict[str, Any],
) -> dict[str, Any]:
    if {record["benchmark_id"] for record in records} != {benchmark_id}:
        raise RuntimeError(f"Mixed dataset IDs in {benchmark_id}")
    manifest = benchmark_root / "manifest.jsonl"
    write_jsonl(manifest, records)
    labels = Counter(int(record["label"]) for record in records)
    generators = Counter(str(record["generator"]) for record in records)
    entry = {
        "benchmark_id": benchmark_id,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "records": len(records),
        "real": labels[0],
        "fake": labels[1],
        "supports_auroc": bool(labels[0] and labels[1]),
        "fake_only": bool(labels[1] and not labels[0]),
        "generators": dict(sorted(generators.items())),
        "seen_in_training": sum(bool(record["seen_in_training"]) for record in records),
        "source": {
            "repo": source_config["repo"],
            "revision": source_config["revision"],
            "license": source_config["license"],
            "split": source_config["split"],
        },
        "training": False,
        "checkpoint_selection": False,
        "calibration": False,
    }
    (benchmark_root / "receipt.json").write_text(
        json.dumps(entry, ensure_ascii=False, indent=2) + "\n"
    )
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/external_generalization_suite.json"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        action="append",
        help="Manifest used only for exact-SHA blocking and seen-generator metadata; repeatable.",
    )
    parser.add_argument(
        "--blocked-manifest",
        type=Path,
        action="append",
        help="Existing external holdout whose exact hashes must be excluded; repeatable.",
    )
    parser.add_argument(
        "--allow-empty-blocklist",
        action="store_true",
        help="Permit preparation without contamination manifests; not equivalent to the audited suite.",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.training_manifest is not None:
        config["training_manifests"] = [str(path) for path in args.training_manifest]
    if args.blocked_manifest is not None:
        config["blocked_external_manifests"] = [str(path) for path in args.blocked_manifest]
    blocked_paths = config["training_manifests"] + config["blocked_external_manifests"]
    if config.get("require_blocklist", True) and not blocked_paths and not args.allow_empty_blocklist:
        raise RuntimeError(
            "No contamination manifests were supplied. Pass --training-manifest and/or "
            "--blocked-manifest, or explicitly use --allow-empty-blocklist for a non-audited copy."
        )
    root = args.output_root or Path(config["output_root"])
    config["output_root"] = str(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / ".EVAL_ONLY_DO_NOT_TRAIN").touch()
    effective_config = root / "effective_config.json"
    effective_config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    blocked_hashes, blocked_counts = collect_blocked_hashes(blocked_paths)
    seen_generators = training_generators(config["training_manifests"])
    entries: list[dict[str, Any]] = []
    entries.extend(
        prepare_ditfake(
            config["datasets"]["ditfake"], root, blocked_hashes, seen_generators,
            int(config["seed"]), args.workers,
        )
    )
    entries.append(
        prepare_frontier_small(
            config["datasets"]["frontier_small"], root, blocked_hashes,
            seen_generators, args.workers,
        )
    )
    entries.append(
        prepare_qwen_image_bench(
            config["datasets"]["qwen_image_bench"], root, blocked_hashes,
            seen_generators, int(config["seed"]), args.workers,
        )
    )
    if len({entry["manifest"] for entry in entries}) != len(entries):
        raise RuntimeError("Benchmarks do not have one-to-one manifests")
    index = {
        "status": "PASS",
        "policy": "one dataset per manifest; never pool samples or metrics",
        "root": str(root),
        "config": str(effective_config),
        "config_sha256": sha256_file(effective_config),
        "blocked_manifests": blocked_counts,
        "blocked_sha256_count": len(blocked_hashes),
        "training": False,
        "checkpoint_selection": False,
        "calibration": False,
        "benchmarks": entries,
    }
    index_path = root / "benchmark_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"event": "external_generalization_suite_ready", **index}, ensure_ascii=False))


if __name__ == "__main__":
    main()
