#!/usr/bin/env python3
"""Build the leak-free union used by the hybrid two-error-mode experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row, sort_keys=True) + "\n").encode())
    return digest.hexdigest()


def normalize_rows(
    rows: list[dict],
    *,
    project_root: Path,
    path_prefix: Path,
    source_prefix: str,
    role: str,
) -> list[dict]:
    normalized = []
    for original in rows:
        row = dict(original)
        relative = path_prefix / row["path"]
        absolute = project_root / relative
        if not absolute.is_file():
            raise FileNotFoundError(absolute)
        row["path"] = relative.as_posix()
        row["source"] = f"{source_prefix}::{row.get('source', 'unknown')}"
        row["hybrid_role"] = role
        row["sha256"] = row.get("sha256") or sha256_path(absolute)
        normalized.append(row)
    return normalized


def split_by_source(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    by_source: dict[str, list[dict]] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    selection, calibration = [], []
    for source, source_rows in sorted(by_source.items()):
        ordered = sorted(
            source_rows,
            key=lambda row: hashlib.sha256(
                f"{seed}:{source}:{row['sha256']}".encode()
            ).hexdigest(),
        )
        cut = len(ordered) // 2
        selection.extend(ordered[:cut])
        calibration.extend(ordered[cut:])
    return selection, calibration


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def role_summary(rows: list[dict]) -> dict:
    return {
        "total": len(rows),
        "real": sum(int(row["label"]) == 0 for row in rows),
        "fake": sum(int(row["label"]) == 1 for row in rows),
        "sources": dict(sorted(Counter(row["source"] for row in rows).items())),
        "manifest_sha256": manifest_sha(rows),
    }


def deduplicate_role(rows: list[dict], role: str) -> tuple[list[dict], int]:
    unique: list[dict] = []
    seen: dict[str, dict] = {}
    for row in rows:
        digest = row["sha256"]
        previous = seen.get(digest)
        if previous is None:
            seen[digest] = row
            unique.append(row)
            continue
        if int(previous["label"]) != int(row["label"]):
            raise RuntimeError(
                f"Conflicting labels for duplicate content in {role}: {digest}"
            )
    return unique, len(rows) - len(unique)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/hybrid_groupdro"))
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output

    formal = root / "data/formal_bundle/manifests"
    general_root = root / "test/test-1/data/full"
    general = general_root / "manifests"
    train = normalize_rows(
        read_jsonl(formal / "train.jsonl"), project_root=root, path_prefix=Path(),
        source_prefix="legacy", role="train",
    ) + normalize_rows(
        read_jsonl(general / "train.jsonl"), project_root=root,
        path_prefix=Path("test/test-1/data/full"), source_prefix="general", role="train",
    )
    formal_validation = normalize_rows(
        read_jsonl(formal / "val.jsonl"), project_root=root, path_prefix=Path(),
        source_prefix="legacy", role="selection",
    )
    general_validation = normalize_rows(
        read_jsonl(general / "validation.jsonl"), project_root=root,
        path_prefix=Path("test/test-1/data/full"), source_prefix="general",
        role="selection_or_calibration",
    )
    general_selection, general_calibration_holdout = split_by_source(
        general_validation, args.seed
    )
    validation = formal_validation + general_selection
    calibration = normalize_rows(
        read_jsonl(general / "calibration.jsonl"), project_root=root,
        path_prefix=Path("test/test-1/data/full"), source_prefix="general",
        role="calibration",
    ) + [dict(row, hybrid_role="calibration") for row in general_calibration_holdout]
    legacy_test = normalize_rows(
        read_jsonl(formal / "test.jsonl"), project_root=root, path_prefix=Path(),
        source_prefix="legacy", role="test_legacy",
    )
    general_test = normalize_rows(
        read_jsonl(general / "test.jsonl"), project_root=root,
        path_prefix=Path("test/test-1/data/full"), source_prefix="general",
        role="test_general",
    )
    raw_roles = {
        "train": train,
        "validation": validation,
        "calibration": calibration,
        "test_legacy": legacy_test,
        "test_general": general_test,
    }

    roles: dict[str, list[dict]] = {}
    duplicates_removed: dict[str, int] = {}
    for role, rows in raw_roles.items():
        roles[role], duplicates_removed[role] = deduplicate_role(rows, role)

    protected: dict[str, tuple[str, dict]] = {}
    cross_role_quarantined = {role: 0 for role in roles}
    # Preserve evaluation integrity.  Content present in more than one role is
    # retained only in the highest-priority evaluation role; lower-priority copies
    # are quarantined.  Label conflicts remain a hard failure.
    priority = ["test_legacy", "test_general", "calibration", "validation", "train"]
    for role in priority:
        filtered = []
        for row in roles[role]:
            digest = row["sha256"]
            previous = protected.get(digest)
            if previous is not None:
                previous_role, previous_row = previous
                if int(previous_row["label"]) != int(row["label"]):
                    raise RuntimeError(
                        "Conflicting labels across roles for content "
                        f"{digest}: {previous_role} vs {role}"
                    )
                cross_role_quarantined[role] += 1
                continue
            protected[digest] = (role, row)
            filtered.append(row)
        roles[role] = filtered

    manifests = output / "manifests"
    for role, rows in roles.items():
        write_jsonl(manifests / f"{role}.jsonl", rows)
    receipt = {
        "status": "PASS",
        "seed": args.seed,
        "policy": {
            "external_eval_only_used": False,
            "dalle_used": False,
            "group_validation_split": "deterministic_sha256_half_per_source",
            "selection_and_calibration_disjoint": True,
            "cross_role_sha256_overlap": 0,
            "same_label_duplicates_removed_within_role": duplicates_removed,
            "cross_role_duplicates_quarantined_from_lower_priority_role": (
                cross_role_quarantined
            ),
        },
        "roles": {role: role_summary(rows) for role, rows in roles.items()},
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"event": "hybrid_data_ready", **receipt}, ensure_ascii=False))


if __name__ == "__main__":
    main()
