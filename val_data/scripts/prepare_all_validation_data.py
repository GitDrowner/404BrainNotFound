#!/usr/bin/env python3
"""One command for the latest OOD suite plus both historical benchmarks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def run(arguments: list[str]) -> None:
    print(json.dumps({"event": "command", "argv": arguments}), flush=True)
    subprocess.run(arguments, check=True, cwd=PACKAGE_ROOT)


def add_repeated(command: list[str], flag: str, paths: list[Path]) -> None:
    for path in paths:
        command.extend([flag, str(path.resolve())])


def verify(index: Path, blocked: list[Path]) -> None:
    command = [
        sys.executable,
        str(PACKAGE_ROOT / "scripts/verify_external_generalization_suite.py"),
        "--index",
        str(index.resolve()),
    ]
    add_repeated(command, "--blocked-manifest", blocked)
    run(command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, action="append", default=[])
    parser.add_argument("--blocked-manifest", type=Path, action="append", default=[])
    parser.add_argument("--manifest-image-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-parent", type=Path, default=Path("data/external_eval_only"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-empty-blocklist", action="store_true")
    args = parser.parse_args()
    output_parent = args.output_parent.resolve()
    historical_root = output_parent / "historical_benchmarks"
    latest_root = output_parent / "generalization_suite_20260830"

    historical = [
        sys.executable,
        str(PACKAGE_ROOT / "scripts/prepare_historical_benchmarks.py"),
        "--config",
        str(PACKAGE_ROOT / "configs/historical_benchmarks.json"),
        "--output-root",
        str(historical_root),
        "--manifest-image-root",
        str(args.manifest_image_root.resolve()),
    ]
    add_repeated(historical, "--training-manifest", args.training_manifest)
    add_repeated(historical, "--blocked-manifest", args.blocked_manifest)
    if args.allow_empty_blocklist:
        historical.append("--allow-empty-blocklist")
    run(historical)
    verify(
        historical_root / "benchmark_index.json",
        [*args.training_manifest, *args.blocked_manifest],
    )

    historical_manifests = sorted(historical_root.glob("*/manifest.jsonl"))
    latest = [
        sys.executable,
        str(PACKAGE_ROOT / "scripts/prepare_external_generalization_suite.py"),
        "--config",
        str(PACKAGE_ROOT / "configs/external_generalization_suite.json"),
        "--output-root",
        str(latest_root),
        "--workers",
        str(args.workers),
    ]
    add_repeated(latest, "--training-manifest", args.training_manifest)
    add_repeated(latest, "--blocked-manifest", [*args.blocked_manifest, *historical_manifests])
    if args.allow_empty_blocklist:
        latest.append("--allow-empty-blocklist")
    run(latest)
    verify(
        latest_root / "benchmark_index.json",
        [*args.training_manifest, *args.blocked_manifest, *historical_manifests],
    )
    print(
        json.dumps(
            {
                "event": "all_validation_data_ready",
                "status": "PASS",
                "policy": "seven independent benchmarks; never pool samples or metrics",
                "historical_index": str(historical_root / "benchmark_index.json"),
                "latest_index": str(latest_root / "benchmark_index.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
