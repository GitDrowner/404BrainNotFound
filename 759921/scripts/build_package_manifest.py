#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "PACKAGE_MANIFEST.json"
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == output or "__pycache__" in path.parts:
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    payload = {
        "package_schema": "aigc-759921-repository-v1",
        "project_id": "759921",
        "entrypoints": {
            "explain": "python scripts/explain_hybrid.py",
            "train_learnable_loss": "bash scripts/run_learnable_loss_experiment.sh",
        },
        "contains_training_images": False,
        "selected_checkpoint_sha256": sha256(root / "checkpoint" / "best.pt"),
        "file_count": len(files),
        "files": files,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
