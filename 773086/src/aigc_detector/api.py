from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import traceback
import uuid
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .explain import generate_explanation, score_images, sha256
from .model import TraceDetector


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PACKAGE_ROOT / "checkpoint" / "best.pt"
DEFAULT_CALIBRATION = PACKAGE_ROOT / "checkpoint" / "calibration_balanced.json"
DEFAULT_RESULTS = PACKAGE_ROOT / "runtime_results"
DEFAULT_FRONTEND = PACKAGE_ROOT / "demos-v2"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
SUPPORTED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def validate_image(payload: bytes) -> tuple[Image.Image, str]:
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 20 MiB upload limit")
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            image_format = opened.format or "UNKNOWN"
            width, height = opened.size
            if image_format not in SUPPORTED_FORMATS:
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported image format: {image_format}; use JPEG, PNG, or WEBP",
                )
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image dimensions {width}x{height} exceed the 25 MP limit",
                )
            image = opened.convert("RGB").copy()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=415, detail="The upload is not a valid image") from error
    return image, SUPPORTED_FORMATS[image_format]


@dataclass(frozen=True)
class RuntimeSettings:
    checkpoint: Path = DEFAULT_CHECKPOINT
    calibration: Path = DEFAULT_CALIBRATION
    results_root: Path = DEFAULT_RESULTS
    frontend_root: Path = DEFAULT_FRONTEND
    device: str = "auto"

    @classmethod
    def from_environment(cls) -> "RuntimeSettings":
        return cls(
            checkpoint=Path(os.environ.get("AIGC_CHECKPOINT", DEFAULT_CHECKPOINT)),
            calibration=Path(os.environ.get("AIGC_CALIBRATION", DEFAULT_CALIBRATION)),
            results_root=Path(os.environ.get("AIGC_RESULTS_ROOT", DEFAULT_RESULTS)),
            frontend_root=Path(os.environ.get("AIGC_FRONTEND_ROOT", DEFAULT_FRONTEND)),
            device=os.environ.get("AIGC_DEVICE", "auto"),
        )


class LocalModelRuntime:
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.settings.results_root.mkdir(parents=True, exist_ok=True)
        self._load_lock = threading.Lock()
        self._inference_lock = threading.RLock()
        self._job_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aigc-explanation")
        self.model: TraceDetector | None = None
        self.config: dict | None = None
        self.device: torch.device | None = None
        self.checkpoint_sha256 = sha256(self.settings.checkpoint)
        self.calibration_sha256 = sha256(self.settings.calibration)
        self.calibration = json.loads(self.settings.calibration.read_text())
        required = {"temperature", "bias", "threshold"}
        missing = required.difference(self.calibration)
        if missing:
            raise RuntimeError(f"Calibration is missing required fields: {sorted(missing)}")
        if float(self.calibration["temperature"]) <= 0:
            raise RuntimeError("Calibration temperature must be positive")
        if not 0.0 <= float(self.calibration["threshold"]) <= 1.0:
            raise RuntimeError("Calibration threshold must be in [0, 1]")
        self.jobs: dict[str, dict] = {}

    @property
    def decision_threshold(self) -> float:
        return float(self.calibration["threshold"])

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def ensure_loaded(self) -> None:
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            device = resolve_device(self.settings.device)
            checkpoint = torch.load(
                self.settings.checkpoint, map_location="cpu", weights_only=False
            )
            config = checkpoint["config"]
            model = TraceDetector(
                config,
                int(config["data"]["image_size"]),
                int(config["data"]["semantic_image_size"]),
            ).to(device)
            incompatible = model.load_state_dict(checkpoint["model"], strict=False)
            if incompatible.unexpected_keys:
                raise RuntimeError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
            model.eval()
            self.config = config
            self.device = device
            self.model = model

    def model_info(self) -> dict:
        checkpoint = torch.load(
            self.settings.checkpoint, map_location="cpu", weights_only=False
        )
        config = checkpoint["config"]
        return {
            "method": "773086_759921_architecture_mlp_normalized",
            "checkpoint": str(self.settings.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "source_epoch": checkpoint.get("epoch"),
            "forensic_backbone": config["model"]["forensic_backbone"],
            "semantic_backbone": config["model"]["semantic_backbone"],
            "loss_mode": config.get("loss", {}).get("mode"),
            "confidence": "piecewise_linearized_fp32_platt_score",
            "calibration": str(self.settings.calibration),
            "calibration_sha256": self.calibration_sha256,
            "calibration_temperature": float(self.calibration["temperature"]),
            "calibration_bias": float(self.calibration["bias"]),
            "calibrated_probability_threshold": self.decision_threshold,
            "decision_threshold": 0.5,
            "confidence_mapping": (
                "p<=t: 0.5*p/t; p>t: 0.5+0.5*(p-t)/(1-t)"
            ),
            "device": str(self.device) if self.device is not None else self.settings.device,
            "loaded": self.loaded,
        }

    def predict(self, image: Image.Image, image_digest: str) -> dict:
        self.ensure_loaded()
        assert self.model is not None and self.config is not None and self.device is not None
        with self._inference_lock:
            rows, outputs = score_images(
                self.model,
                [image],
                self.config,
                self.device,
                float(self.calibration["temperature"]),
                float(self.calibration["bias"]),
                1,
                self.decision_threshold,
            )
        prediction = rows[0]
        model_outputs = outputs[0]
        prediction.update(
            {
                "label_at_display_threshold": (
                    "aigc"
                    if prediction["aigc_confidence"] >= 0.5
                    else "real"
                ),
                "display_threshold": 0.5,
                "calibrated_probability_threshold": self.decision_threshold,
                "confidence_semantics": (
                    "piecewise-linearized FP32 Platt score; probability_fake retains the pre-mapping calibrated value"
                ),
            }
        )
        return {
            "prediction": prediction,
            "image": {
                "sha256": image_digest,
                "width": image.width,
                "height": image.height,
            },
            "model": self.model_info(),
            "branches_available": [
                key
                for key in ("tile_attention", "tile_logits", "wavelet_similarity")
                if key in model_outputs
            ],
        }

    def submit_analysis(
        self,
        payload: bytes,
        extension: str,
        filename: str,
        mode: Literal["fast", "full"],
        occlusion: Literal["blur", "mean"],
    ) -> dict:
        job_id = uuid.uuid4().hex
        job_root = self.settings.results_root / job_id
        job_root.mkdir(parents=True, exist_ok=False)
        input_path = job_root / f"upload{extension}"
        input_path.write_bytes(payload)
        profile = (
            {"grid": 4, "refine_top_k": 3, "refine_grid": 2, "batch_size": 4}
            if mode == "fast"
            else {"grid": 6, "refine_top_k": 6, "refine_grid": 3, "batch_size": 8}
        )
        job = {
            "job_id": job_id,
            "status": "queued",
            "filename": filename,
            "mode": mode,
            "occlusion": occlusion,
            "profile": profile,
            "created_at": utc_now(),
            "started_at": None,
            "completed_at": None,
            "error": None,
        }
        with self._job_lock:
            self.jobs[job_id] = job
        self._executor.submit(self._run_analysis, job_id, input_path, job_root, occlusion, profile)
        return dict(job)

    def _run_analysis(
        self,
        job_id: str,
        input_path: Path,
        output: Path,
        occlusion: str,
        profile: dict,
    ) -> None:
        self._update_job(job_id, status="running", started_at=utc_now())
        try:
            self.ensure_loaded()
            assert self.model is not None and self.config is not None and self.device is not None
            args = Namespace(
                checkpoint=self.settings.checkpoint,
                image=input_path,
                output=output,
                calibration=self.settings.calibration,
                occlusion=occlusion,
                device=str(self.device),
                **profile,
            )
            with self._inference_lock:
                payload = generate_explanation(
                    args, model=self.model, config=self.config, device=self.device
                )
            self._update_job(
                job_id,
                status="completed",
                completed_at=utc_now(),
                prediction=payload["prediction"],
                result=self.result_urls(job_id),
            )
        except Exception as error:  # Preserve a concise public error and local traceback.
            (output / "error.log").write_text(traceback.format_exc())
            self._update_job(
                job_id,
                status="failed",
                completed_at=utc_now(),
                error=f"{type(error).__name__}: {error}",
            )

    def _update_job(self, job_id: str, **updates) -> None:
        with self._job_lock:
            self.jobs[job_id].update(updates)

    def get_job(self, job_id: str) -> dict:
        with self._job_lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return dict(job)

    def result_urls(self, job_id: str) -> dict:
        base = f"/results/{job_id}"
        return {
            "dashboard_url": f"{base}/index.html",
            "explanation_url": f"{base}/explanation.json",
            "patches_url": f"{base}/patches.jsonl",
            "assets": {
                name: f"{base}/{filename}"
                for name, filename in {
                    "input": "input.png",
                    "patch_attribution": "patch_attribution.png",
                    "attribution_heatmap": "heatmap_attribution.png",
                    "attribution_overlay": "heatmap_attribution_overlay.png",
                    "texture_heatmap": "heatmap_texture.png",
                    "frequency_heatmap": "heatmap_frequency_contribution.png",
                    "frequency_overlay": "heatmap_frequency_overlay.png",
                    "transform_trajectory": "transform_trajectory.svg",
                    "components": "components.svg",
                }.items()
            },
        }


def create_app(
    settings: RuntimeSettings | None = None,
    runtime: LocalModelRuntime | None = None,
) -> FastAPI:
    settings = settings or RuntimeSettings.from_environment()
    if not settings.checkpoint.is_file():
        raise FileNotFoundError(settings.checkpoint)
    settings.results_root.mkdir(parents=True, exist_ok=True)
    runtime = runtime or LocalModelRuntime(settings)
    app = FastAPI(
        title="773086 Calibrated Local AIGC Explainability API",
        version="1.0.0",
        description="Local-only confidence and counterfactual explanation service.",
    )
    app.state.runtime = runtime
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.mount("/results", StaticFiles(directory=settings.results_root), name="results")
    app.mount("/demos", StaticFiles(directory=settings.frontend_root, html=True), name="demos")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(settings.frontend_root / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_loaded": runtime.loaded,
            "device": str(runtime.device) if runtime.device else settings.device,
            "queued_or_running": sum(
                runtime.get_job(job_id)["status"] in {"queued", "running"}
                for job_id in list(runtime.jobs)
            ),
        }

    @app.get("/api/v1/model")
    def model_info() -> dict:
        return runtime.model_info()

    @app.post("/api/v1/predict")
    def predict(file: UploadFile = File(...)) -> dict:
        payload = file.file.read(MAX_UPLOAD_BYTES + 1)
        image, _extension = validate_image(payload)
        return runtime.predict(image, hashlib.sha256(payload).hexdigest())

    @app.post("/api/v1/analyses", status_code=202)
    def create_analysis(
        request: Request,
        file: UploadFile = File(...),
        mode: Literal["fast", "full"] = Form("fast"),
        occlusion: Literal["blur", "mean"] = Form("blur"),
    ) -> JSONResponse:
        payload = file.file.read(MAX_UPLOAD_BYTES + 1)
        _image, extension = validate_image(payload)
        job = runtime.submit_analysis(
            payload, extension, file.filename or "upload", mode, occlusion
        )
        job_id = job["job_id"]
        body = {
            **job,
            "status_url": str(request.url_for("analysis_status", job_id=job_id)),
            "result_url": str(request.url_for("analysis_result", job_id=job_id)),
        }
        return JSONResponse(status_code=202, content=body)

    @app.get("/api/v1/analyses/{job_id}", name="analysis_status")
    def analysis_status(job_id: str) -> dict:
        try:
            job = runtime.get_job(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Unknown analysis job") from error
        job["result_url"] = f"/api/v1/analyses/{job_id}/result"
        return job

    @app.get("/api/v1/analyses/{job_id}/result", name="analysis_result")
    def analysis_result(job_id: str) -> dict:
        try:
            job = runtime.get_job(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Unknown analysis job") from error
        if job["status"] == "failed":
            raise HTTPException(status_code=500, detail=job["error"])
        if job["status"] != "completed":
            raise HTTPException(status_code=409, detail=f"Analysis is {job['status']}")
        explanation_path = settings.results_root / job_id / "explanation.json"
        return {
            "job": job,
            "explanation": json.loads(explanation_path.read_text()),
            "result": runtime.result_urls(job_id),
        }

    return app


app = create_app()
