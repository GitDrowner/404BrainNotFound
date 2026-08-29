from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from aigc_detector.explainability import (
    SCHEMA_VERSION,
    dense_region_field,
    diverging_colormap,
    inferno_colormap,
    occlude_patch,
    patch_boxes,
    save_attribution_overlay,
    save_bar_svg,
    save_colorbar,
    save_dense_signed_heatmaps,
    save_heatmap_overlay,
    save_line_svg,
    save_texture_heatmap,
    subdivide_box,
    suppress_high_frequency_patch,
    write_schema,
)


class ExplainabilityTests(unittest.TestCase):
    def test_patch_grid_is_gap_free(self):
        boxes = patch_boxes(103, 79, 6)
        self.assertEqual(len(boxes), 36)
        self.assertEqual(boxes[0][:2], (0, 0))
        self.assertEqual(boxes[-1][2:], (103, 79))
        self.assertEqual(sum((right - left) * (bottom - top) for left, top, right, bottom in boxes), 103 * 79)
        children = subdivide_box(boxes[0], 3)
        self.assertEqual(len(children), 9)
        self.assertEqual(sum((r - l) * (b - t) for l, t, r, b in children), (boxes[0][2] - boxes[0][0]) * (boxes[0][3] - boxes[0][1]))

    def test_occlusion_and_artifacts(self):
        image = Image.new("RGB", (100, 80), (120, 90, 60))
        boxes = patch_boxes(*image.size, 2)
        self.assertEqual(occlude_patch(image, boxes[0], "blur").size, image.size)
        self.assertEqual(suppress_high_frequency_patch(image, boxes[0]).size, image.size)
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

    def test_colormaps_are_uint8_rgb(self):
        self.assertEqual(diverging_colormap(np.array([-1.0, 0.0, 1.0], dtype=np.float32)).shape, (3, 3))
        self.assertEqual(inferno_colormap(np.array([0.0, 1.0], dtype=np.float32)).shape, (2, 3))
        blue, mid, red = diverging_colormap(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
        self.assertLess(int(blue[0]), int(red[0]))  # blue has less red channel than the red anchor
        self.assertGreater(int(mid[0]), int(blue[0]))  # midpoint is brighter than the blue anchor

    def test_heatmap_and_texture_artifacts(self):
        image = Image.new("RGB", (96, 64), (120, 90, 60))
        boxes = patch_boxes(96, 64, 4)
        contributions = [0.1, -0.2, 0.3, -0.4] * 4
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vmin, vmax = save_heatmap_overlay(image, boxes, contributions, root / "heatmap.png")
            self.assertTrue((root / "heatmap.png").is_file())
            self.assertAlmostEqual(vmin, -0.4)
            self.assertAlmostEqual(vmax, 0.4)
            save_colorbar(root / "colorbar.png", vmin=vmin, vmax=vmax, center=0.0)
            self.assertTrue((root / "colorbar.png").is_file())
            save_texture_heatmap(image, root / "texture.png")
            self.assertTrue((root / "texture.png").is_file())

    def test_hierarchical_dense_heatmap(self):
        image = Image.new("RGB", (96, 64), (120, 90, 60))
        coarse = patch_boxes(96, 64, 2)
        refined = subdivide_box(coarse[0], 2)
        boxes = coarse + refined
        values = [0.1, -0.2, 0.3, -0.4, 0.8, 0.7, -0.6, -0.5]
        weights = [1.0] * 4 + [2.0] * 4
        field = dense_region_field(image.size, boxes, values, weights=weights)
        self.assertEqual(field.shape, (64, 96))
        self.assertTrue(np.isfinite(field).all())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vmin, vmax = save_dense_signed_heatmaps(
                image, boxes, values, root / "standalone.png", root / "overlay.png", weights=weights
            )
            self.assertLess(vmin, 0)
            self.assertGreater(vmax, 0)
            self.assertTrue((root / "standalone.png").is_file())
            self.assertTrue((root / "overlay.png").is_file())


if __name__ == "__main__":
    unittest.main()
