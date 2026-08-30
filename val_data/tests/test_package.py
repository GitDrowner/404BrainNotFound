from __future__ import annotations

import importlib.util
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_prepare_module():
    path = ROOT / "scripts" / "prepare_external_generalization_suite.py"
    spec = importlib.util.spec_from_file_location("prepare_external_suite", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PackageTests(unittest.TestCase):
    def test_catalog_and_config_are_consistent(self) -> None:
        config = json.loads((ROOT / "configs/external_generalization_suite.json").read_text())
        catalog = json.loads((ROOT / "metadata/DATASET_CATALOG.json").read_text())
        self.assertTrue(config["require_blocklist"])
        self.assertEqual(config["training_manifests"], [])
        self.assertEqual(config["blocked_external_manifests"], [])
        self.assertEqual(sum(row["records"] for row in catalog["benchmarks"]), 920)
        ids = {row["benchmark_id"] for row in catalog["benchmarks"]}
        expected = set(config["datasets"]["ditfake"]["generators"].keys()) | {
            config["datasets"]["frontier_small"]["benchmark_id"],
            config["datasets"]["qwen_image_bench"]["benchmark_id"],
        }
        self.assertEqual(ids, expected)
        revisions = {
            settings["revision"] for settings in config["datasets"].values()
        }
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{40}", value) for value in revisions))

    def test_fake_only_semantics_are_explicit(self) -> None:
        catalog = json.loads((ROOT / "metadata/DATASET_CATALOG.json").read_text())
        qwen = next(
            row
            for row in catalog["benchmarks"]
            if row["benchmark_id"] == "qwen_image_bench_frontier_fake_only"
        )
        self.assertTrue(qwen["fake_only"])
        self.assertFalse(qwen["supports_auroc"])
        self.assertEqual(qwen["real"], 0)
        self.assertEqual(len(qwen["generators"]), 18)

    def test_helpers_are_deterministic_and_safe(self) -> None:
        module = load_prepare_module()
        self.assertEqual(module.safe_name("GPT Image / 2"), "GPT_Image_2")
        self.assertEqual(module.normalise_generator("FLUX.1-schnell"), "flux1schnell")
        self.assertEqual(module.collect_blocked_hashes([]), (set(), {}))
        self.assertEqual(module.training_generators([]), set())

    def test_repository_contains_no_image_payloads(self) -> None:
        forbidden = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
        images = [path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in forbidden]
        self.assertEqual(images, [])


if __name__ == "__main__":
    unittest.main()
