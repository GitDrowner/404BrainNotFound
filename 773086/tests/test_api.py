from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from aigc_detector.api import RuntimeSettings, create_app, validate_image
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

    def model_info(self) -> dict:
        return {"method": "fake", "loaded": True}

    def predict(self, image, image_digest: str) -> dict:
        return {
            "prediction": {"probability_fake": 0.75},
            "image": {"width": image.width, "height": image.height, "sha256": image_digest},
        }

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
                )
                self.assertEqual(prediction.status_code, 200)
                self.assertEqual(prediction.json()["prediction"]["probability_fake"], 0.75)
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
