from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from aigc_detector.explainability import (
    SCHEMA_VERSION,
    occlude_patch,
    patch_boxes,
    save_attribution_overlay,
    save_bar_svg,
    save_line_svg,
    write_schema,
)


class ExplainabilityTests(unittest.TestCase):
    def test_patch_grid_is_gap_free(self):
        boxes = patch_boxes(103, 79, 6)
        self.assertEqual(len(boxes), 36)
        self.assertEqual(boxes[0][:2], (0, 0))
        self.assertEqual(boxes[-1][2:], (103, 79))
        self.assertEqual(sum((right - left) * (bottom - top) for left, top, right, bottom in boxes), 103 * 79)

    def test_occlusion_and_artifacts(self):
        image = Image.new("RGB", (100, 80), (120, 90, 60))
        boxes = patch_boxes(*image.size, 2)
        self.assertEqual(occlude_patch(image, boxes[0], "blur").size, image.size)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_attribution_overlay(image, boxes, [-0.2, 0.0, 0.1, 0.3], root / "overlay.png")
            save_line_svg(
                [{"variant": "clean", "probability_fake": 0.4}, {"variant": "jpeg", "probability_fake": 0.6}],
                root / "line.svg", value_key="probability_fake", title="test",
            )
            save_bar_svg([("a", -0.2), ("b", 0.3)], root / "bars.svg", title="test")
            write_schema(root / "schema.json")
            self.assertTrue((root / "overlay.png").is_file())
            self.assertIn("<svg", (root / "line.svg").read_text())
            self.assertEqual(json.loads((root / "schema.json").read_text())["title"], SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
