from __future__ import annotations

import io
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from aigc_detector.api import (
    LocalModelRuntime,
    RuntimeSettings,
    apply_transform,
    create_app,
    resolve_transform,
    validate_image,
)
from aigc_detector.explain import linearize_operating_point


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), (20, 40, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeRuntime:
    loaded = True
    device = "cpu"

    def __init__(self) -> None:
        self.jobs = {}
        self.transform_scans = {}
        self.foreground_waiting = 0

    def model_info(self) -> dict:
        return {"method": "fake", "loaded": True}

    def predict(self, image, image_digest: str) -> dict:
        return {
            "prediction": {"probability_fake": 0.75},
            "image": {"width": image.width, "height": image.height, "sha256": image_digest},
        }

    def predict_selected(self, image, image_digest: str, transform_id: str) -> dict:
        transform = resolve_transform(transform_id)
        return {
            "prediction": {
                "probability_fake": 0.75,
                "aigc_confidence": 0.80,
                "label_at_display_threshold": "aigc",
            },
            "image": {"width": image.width, "height": image.height, "sha256": image_digest},
            "transform": transform,
            "branches_available": [],
        }

    def submit_transform_scan(
        self, payload, filename, image_digest, selected_result
    ) -> dict:
        transform_id = selected_result["transform"]["id"]
        scan = {
            "scan_id": "scan-1",
            "status": "running",
            "selected_transform": transform_id,
            "order": [transform_id],
            "completed_count": 1,
            "total_count": 16,
            "results": {transform_id: selected_result},
        }
        self.transform_scans[scan["scan_id"]] = scan
        return dict(scan)

    def get_transform_scan(self, scan_id: str) -> dict:
        if scan_id not in self.transform_scans:
            raise KeyError(scan_id)
        return dict(self.transform_scans[scan_id])

    def submit_analysis(self, payload, extension, filename, mode, occlusion) -> dict:
        job = {
            "job_id": "job-1",
            "status": "queued",
            "filename": filename,
            "mode": mode,
            "occlusion": occlusion,
        }
        self.jobs[job["job_id"]] = job
        return dict(job)

    def get_job(self, job_id: str) -> dict:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return dict(self.jobs[job_id])

    def result_urls(self, job_id: str) -> dict:
        return {"dashboard_url": f"/results/{job_id}/index.html"}


class ApiTests(unittest.TestCase):
    def test_operating_point_linearization(self) -> None:
        threshold = 0.2815194250040655
        self.assertEqual(linearize_operating_point(0.0, threshold), 0.0)
        self.assertAlmostEqual(linearize_operating_point(threshold, threshold), 0.5)
        self.assertEqual(linearize_operating_point(1.0, threshold), 1.0)
        values = [linearize_operating_point(value / 100, threshold) for value in range(101)]
        self.assertEqual(values, sorted(values))

    def test_image_validation(self) -> None:
        image, extension = validate_image(png_bytes())
        self.assertEqual(image.size, (12, 8))
        self.assertEqual(extension, ".png")

    def test_transform_contract_and_deterministic_noise(self) -> None:
        image = Image.new("RGB", (12, 8), (20, 40, 60))
        transform = resolve_transform("noise_sigma0.05")
        first = apply_transform(image, transform, "same-image")
        second = apply_transform(image, transform, "same-image")
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(resolve_transform(None)["id"], "clean")
        with self.assertRaises(ValueError):
            resolve_transform("not-a-transform")

    def test_foreground_prediction_preempts_next_background_transform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"stub")
            calibration = root / "calibration.json"
            calibration.write_text(
                '{"temperature": 1.0, "bias": 0.0, "threshold": 0.5}'
            )
            runtime = LocalModelRuntime(
                RuntimeSettings(
                    checkpoint=checkpoint,
                    calibration=calibration,
                    results_root=root / "results",
                    frontend_root=root,
                    device="cpu",
                )
            )
            order = []
            first_background_started = threading.Event()
            release_first_background = threading.Event()

            def fake_score(
                _runtime,
                transformed,
                original,
                image_digest,
                transform,
                *,
                include_model,
            ):
                transform_id = transform["id"]
                order.append(transform_id)
                if transform_id == "clean":
                    first_background_started.set()
                    self.assertTrue(release_first_background.wait(timeout=2))
                return {
                    "prediction": {"aigc_confidence": 0.5},
                    "image": {"sha256": image_digest},
                    "transform": transform,
                    "branches_available": [],
                }

            runtime._score_transformed = types.MethodType(fake_score, runtime)
            image = Image.new("RGB", (16, 16), (10, 20, 30))
            first_background = threading.Thread(
                target=runtime._predict_background,
                args=(image, "digest", resolve_transform("clean")),
            )
            next_background = threading.Thread(
                target=runtime._predict_background,
                args=(image, "digest", resolve_transform("jpeg_q90")),
            )
            foreground = threading.Thread(
                target=runtime.predict_selected,
                args=(image, "digest", "blur_sigma0.5"),
            )
            first_background.start()
            self.assertTrue(first_background_started.wait(timeout=2))
            next_background.start()
            foreground.start()
            deadline = time.monotonic() + 2
            while runtime.foreground_waiting == 0 and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertEqual(runtime.foreground_waiting, 1)
            release_first_background.set()
            for thread in (first_background, foreground, next_background):
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())
            runtime.shutdown()
            self.assertEqual(order, ["clean", "blur_sigma0.5", "jpeg_q90"])

    def test_route_contract_without_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"stub")
            frontend = root / "frontend"
            frontend.mkdir()
            (frontend / "index.html").write_text("ok")
            settings = RuntimeSettings(
                checkpoint=checkpoint,
                results_root=root / "results",
                frontend_root=frontend,
                device="cpu",
            )
            app = create_app(settings, FakeRuntime())
            with TestClient(app) as client:
                self.assertEqual(client.get("/api/health").status_code, 200)
                self.assertEqual(client.get("/").text, "ok")
                prediction = client.post(
                    "/api/v1/predict",
                    files={"file": ("sample.png", png_bytes(), "image/png")},
                    data={"transform": "jpeg_q70"},
                )
                self.assertEqual(prediction.status_code, 200)
                self.assertEqual(prediction.json()["prediction"]["probability_fake"], 0.75)
                self.assertEqual(prediction.json()["transform"]["id"], "jpeg_q70")
                self.assertEqual(prediction.json()["transform_scan"]["scan_id"], "scan-1")
                self.assertEqual(
                    client.get("/api/v1/transform-scans/scan-1").json()[
                        "selected_transform"
                    ],
                    "jpeg_q70",
                )
                catalog = client.get("/api/v1/transforms").json()
                self.assertEqual(catalog["count"], 16)
                self.assertEqual(catalog["transforms"][0]["id"], "clean")
                invalid = client.post(
                    "/api/v1/predict",
                    files={"file": ("sample.png", png_bytes(), "image/png")},
                    data={"transform": "unknown"},
                )
                self.assertEqual(invalid.status_code, 400)
                analysis = client.post(
                    "/api/v1/analyses",
                    files={"file": ("sample.png", png_bytes(), "image/png")},
                    data={"mode": "fast", "occlusion": "blur"},
                )
                self.assertEqual(analysis.status_code, 202)
                self.assertEqual(analysis.json()["job_id"], "job-1")
                self.assertEqual(
                    client.get("/api/v1/analyses/job-1").json()["status"], "queued"
                )


if __name__ == "__main__":
    unittest.main()
