#!/usr/bin/env python3
"""Merge the 759921 corpus with diverse additions while preserving role isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


DALL_E_3 = re.compile(r"(?:dall[-_. ]?e|dalle)[-_. ]?3", re.IGNORECASE)


def read(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(record: dict) -> str:
    return " ".join(str(record.get(key, "")) for key in ("path", "source", "generator", "dataset_repo"))


def record_hash(record: dict, require_images: bool) -> str:
    path = Path(record["path"])
    if path.is_file():
        value = sha256(path)
        expected = record.get("sha256")
        if expected and value != expected:
            raise RuntimeError(f"Checksum mismatch: {path}")
        return value
    if require_images:
        raise FileNotFoundError(path)
    if not record.get("sha256"):
        raise RuntimeError(f"Missing image and recorded SHA-256: {path}")
    return str(record["sha256"])


def validate_train(records: list[dict], holdout_hashes: set[str], require_images: bool) -> list[dict]:
    clean, seen = [], set()
    for record in records:
        text = identity(record)
        if "external_eval_only" in text.replace("\\", "/").casefold():
            raise RuntimeError(f"Evaluation-only path/source in training: {text}")
        if DALL_E_3.search(text.replace("·", "")):
            raise RuntimeError(f"DALL-E 3 source in training: {text}")
        digest = record_hash(record, require_images)
        if digest in holdout_hashes:
            raise RuntimeError(f"Exact DALL-E Advanced overlap in training: {record['path']}")
        if digest in seen:
            continue
        seen.add(digest)
        clean.append({**record, "sha256": digest, "role": "train"})
    return clean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--additions", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, default=Path("data/external_eval_only/dalle3_advanced/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/fusion_v2/manifests"))
    parser.add_argument("--require-images", action="store_true")
    args = parser.parse_args()
    holdout = read(args.holdout)
    holdout_hashes = {str(record["sha256"]) for record in holdout}
    train = validate_train(read(args.base_train) + read(args.additions), holdout_hashes, args.require_images)
    roles = {
        "validation": read(args.validation),
        "calibration": read(args.calibration),
        "test": read(args.test),
    }
    role_quarantine: dict[str, int] = {}
    higher_priority_hashes: set[str] = set()
    for role in ("test", "calibration", "validation"):
        kept, seen = [], set()
        for record in roles[role]:
            digest = record_hash(record, args.require_images)
            if digest in seen or digest in higher_priority_hashes:
                continue
            seen.add(digest)
            kept.append(record)
        role_quarantine[role] = len(roles[role]) - len(kept)
        roles[role] = kept
        higher_priority_hashes.update(seen)
    reserved_hashes = {
        record_hash(record, args.require_images)
        for records in roles.values()
        for record in records
    }
    before_quarantine = len(train)
    train = [
        record for record in train
        if record_hash(record, args.require_images) not in reserved_hashes
    ]
    quarantined_train_records = before_quarantine - len(train)
    roles = {"train": train, **roles}
    role_hashes: dict[str, set[str]] = {}
    for role, records in roles.items():
        hashes = {record_hash(record, args.require_images) for record in records}
        role_hashes[role] = hashes
    overlaps = {}
    names = list(roles)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            count = len(role_hashes[left] & role_hashes[right])
            overlaps[f"{left}::{right}"] = count
            if count:
                raise RuntimeError(f"Cross-role SHA overlap: {left}/{right}={count}")
    args.output.mkdir(parents=True, exist_ok=True)
    for role, records in roles.items():
        with (args.output / f"{role}.jsonl").open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    receipt = {
        "status": "PASS", "dalle3_records_in_train": 0, "external_eval_records_in_train": 0,
        "exact_dalle_advanced_overlap": 0, "cross_role_sha_overlap": overlaps,
        "train_records_quarantined_for_later_role_overlap": quarantined_train_records,
        "later_role_records_quarantined_by_priority": role_quarantine,
        "counts": {role: len(records) for role, records in roles.items()},
        "train_labels": dict(Counter(str(int(record["label"])) for record in train)),
        "train_sources": dict(Counter(str(record.get("source", "unknown")) for record in train)),
        "train_generators": dict(Counter(str(record.get("generator", "unknown")) for record in train)),
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
