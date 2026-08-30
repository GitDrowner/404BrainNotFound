#!/usr/bin/env python3
"""One-command wrapper for download, preprocessing, and integrity verification."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(arguments: list[str]) -> None:
    print(json.dumps({"event": "command", "argv": arguments}), flush=True)
    subprocess.run(arguments, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/external_generalization_suite.json")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/external_eval_only/generalization_suite_20260830"),
    )
    parser.add_argument("--training-manifest", type=Path, action="append", default=[])
    parser.add_argument("--blocked-manifest", type=Path, action="append", default=[])
    parser.add_argument("--allow-empty-blocklist", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare = [
        sys.executable,
        "scripts/prepare_external_generalization_suite.py",
        "--config",
        str(args.config),
        "--output-root",
        str(args.output_root),
        "--workers",
        str(args.workers),
    ]
    for path in args.training_manifest:
        prepare.extend(["--training-manifest", str(path)])
    for path in args.blocked_manifest:
        prepare.extend(["--blocked-manifest", str(path)])
    if args.allow_empty_blocklist:
        prepare.append("--allow-empty-blocklist")
    run(prepare)
    verify = [
        sys.executable,
        "scripts/verify_external_generalization_suite.py",
        "--index",
        str(args.output_root / "benchmark_index.json"),
    ]
    for path in [*args.training_manifest, *args.blocked_manifest]:
        verify.extend(["--blocked-manifest", str(path)])
    run(verify)


if __name__ == "__main__":
    main()
