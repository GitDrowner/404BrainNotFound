#!/usr/bin/env python3
"""Verify downloaded benchmarks without depending on detector training code."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(
            "data/external_eval_only/generalization_suite_20260830/benchmark_index.json"
        ),
    )
    parser.add_argument(
        "--blocked-manifest",
        type=Path,
        action="append",
        default=[],
        help="Optional contamination manifest to recheck; repeatable.",
    )
    args = parser.parse_args()
    index = json.loads(args.index.read_text())
    if index.get("status") != "PASS":
        raise RuntimeError("Benchmark index is not PASS")
    if index.get("policy") != "one dataset per manifest; never pool samples or metrics":
        raise RuntimeError("Dataset-isolation policy changed")
    for forbidden_role in ("training", "checkpoint_selection", "calibration"):
        if index.get(forbidden_role) is not False:
            raise RuntimeError(f"Suite role is not disabled: {forbidden_role}")
    config_path = Path(index["config"])
    if sha256(config_path) != index["config_sha256"]:
        raise RuntimeError("Effective config checksum mismatch")
    blocked_hashes = {
        str(record["sha256"])
        for path in args.blocked_manifest
        for record in read_jsonl(path)
    }
    manifest_paths: set[str] = set()
    suite_root = Path(index["root"]).resolve()
    verified = []
    for entry in index["benchmarks"]:
        benchmark_id = str(entry["benchmark_id"])
        manifest = Path(entry["manifest"])
        if str(manifest) in manifest_paths:
            raise RuntimeError(f"Manifest reused by multiple benchmarks: {manifest}")
        manifest_paths.add(str(manifest))
        if sha256(manifest) != entry["manifest_sha256"]:
            raise RuntimeError(f"Manifest checksum mismatch: {manifest}")
        if not (manifest.parent / ".EVAL_ONLY_DO_NOT_TRAIN").is_file():
            raise RuntimeError(f"Missing evaluation-only sentinel: {manifest.parent}")
        records = read_jsonl(manifest)
        if {record.get("benchmark_id") for record in records} != {benchmark_id}:
            raise RuntimeError(f"Mixed benchmark IDs: {manifest}")
        labels = Counter(int(record["label"]) for record in records)
        if labels[0] != int(entry["real"]) or labels[1] != int(entry["fake"]):
            raise RuntimeError(f"Class count mismatch: {benchmark_id}")
        local_paths: set[str] = set()
        local_hashes: set[str] = set()
        for record in records:
            path = Path(record["path"]).resolve()
            try:
                path.relative_to(suite_root)
            except ValueError as error:
                raise RuntimeError(f"Image escaped benchmark root: {path}") from error
            if str(path) in local_paths:
                raise RuntimeError(f"Duplicate path within {benchmark_id}: {path}")
            digest = sha256(path)
            if digest != record["sha256"]:
                raise RuntimeError(f"Image checksum mismatch: {path}")
            if digest in local_hashes:
                raise RuntimeError(f"Duplicate content within {benchmark_id}: {path}")
            if digest in blocked_hashes:
                raise RuntimeError(f"Blocked-manifest overlap: {path}")
            if record.get("role") != "external_evaluation_only":
                raise RuntimeError(f"Incorrect role: {path}")
            local_paths.add(str(path))
            local_hashes.add(digest)
        verified.append(
            {
                "benchmark_id": benchmark_id,
                "records": len(records),
                "real": labels[0],
                "fake": labels[1],
                "manifest_sha256": entry["manifest_sha256"],
            }
        )
    print(
        json.dumps(
            {
                "event": "external_generalization_suite_verified",
                "status": "PASS",
                "policy": "each dataset remains separate",
                "blocked_hashes_rechecked": len(blocked_hashes),
                "benchmarks": verified,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
