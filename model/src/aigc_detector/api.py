from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import threading
import traceback
import uuid
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
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

from .augmentations import apply_fixed, competition_grid, fixed_operation
from .explain import generate_explanation, score_images, sha256
from .model import TraceDetector


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PACKAGE_ROOT / "checkpoint" / "best.pt"
DEFAULT_CALIBRATION = PACKAGE_ROOT / "checkpoint" / "calibration_balanced.json"
DEFAULT_TRANSFORM_THRESHOLDS = (
    PACKAGE_ROOT / "checkpoint" / "transform_thresholds_aligned_640.json"
)
DEFAULT_RESULTS = PACKAGE_ROOT / "runtime_results"
DEFAULT_FRONTEND = PACKAGE_ROOT / "demos-v2"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
SUPPORTED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
INPUT_PROTOCOL_ID = "rgb_256x256_bicubic_jpeg_q95_420"
INPUT_PROTOCOL = {
    "id": INPUT_PROTOCOL_ID,
    "color_mode": "RGB",
    "width": 256,
    "height": 256,
    "resize": "Pillow BICUBIC",
    "encoding": "JPEG",
    "jpeg_quality": 95,
    "jpeg_subsampling": "4:2:0",
    "order": "align_then_selected_transform_then_model",
}


def transform_catalog() -> list[dict]:
    """Return the stable public transform order used by evaluation and the demo."""
    labels = {
        "clean": "原图（不变换）",
        "jpeg_q90": "JPEG · quality 90",
        "jpeg_q70": "JPEG · quality 70",
        "jpeg_q50": "JPEG · quality 50",
        "jpeg_q30": "JPEG · quality 30",
        "blur_sigma0.5": "Gaussian blur · σ 0.5",
        "blur_sigma1": "Gaussian blur · σ 1.0",
        "blur_sigma2": "Gaussian blur · σ 2.0",
        "resize_0.5x": "Resize · 0.5× 后放大",
        "resize_0.25x": "Resize · 0.25× 后放大",
        "noise_sigma0.02": "Gaussian noise · σ 0.02",
        "noise_sigma0.05": "Gaussian noise · σ 0.05",
        "noise_sigma0.10": "Gaussian noise · σ 0.10",
        "color_minus20": "Color jitter · −20%",
        "color_plus20": "Color jitter · +20%",
        "center_crop_80": "Center crop · 80%",
    }
    return [
        {
            "id": transform_id,
            "family": family,
            "value": value,
            "label": labels[transform_id],
            "operation": fixed_operation(family, value),
        }
        for transform_id, family, value in competition_grid()
    ]


TRANSFORM_CATALOG = tuple(transform_catalog())
TRANSFORMS_BY_ID = {item["id"]: item for item in TRANSFORM_CATALOG}


def resolve_transform(transform_id: str | None) -> dict:
    normalized = (transform_id or "clean").strip() or "clean"
    try:
        return dict(TRANSFORMS_BY_ID[normalized])
    except KeyError as error:
        raise ValueError(
            f"Unknown transform '{normalized}'; choose one of {list(TRANSFORMS_BY_ID)}"
        ) from error


def apply_transform(image: Image.Image, transform: dict, image_digest: str) -> Image.Image:
    # Noise must be repeatable so the immediate and background result agree.
    seed_material = f"{image_digest}:{transform['id']}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    return apply_fixed(
        image,
        str(transform["family"]),
        transform.get("value"),
        seed=seed,
    )


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


def aligned_inference_payload(image: Image.Image) -> bytes:
    """Apply the frozen anti-shortcut alignment used by 773086 validation."""
    aligned = image.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    aligned.save(
        buffer,
        format="JPEG",
        quality=95,
        subsampling=2,
        optimize=False,
        progressive=False,
    )
    return buffer.getvalue()


def align_for_inference(image: Image.Image) -> Image.Image:
    payload = aligned_inference_payload(image)
    with Image.open(io.BytesIO(payload)) as opened:
        opened.load()
        return opened.convert("RGB").copy()


@dataclass(frozen=True)
class RuntimeSettings:
    checkpoint: Path = DEFAULT_CHECKPOINT
    calibration: Path = DEFAULT_CALIBRATION
    transform_thresholds: Path = DEFAULT_TRANSFORM_THRESHOLDS
    results_root: Path = DEFAULT_RESULTS
    frontend_root: Path = DEFAULT_FRONTEND
    device: str = "auto"

    @classmethod
    def from_environment(cls) -> "RuntimeSettings":
        return cls(
            checkpoint=Path(os.environ.get("AIGC_CHECKPOINT", DEFAULT_CHECKPOINT)),
            calibration=Path(os.environ.get("AIGC_CALIBRATION", DEFAULT_CALIBRATION)),
            transform_thresholds=Path(
                os.environ.get("AIGC_TRANSFORM_THRESHOLDS", DEFAULT_TRANSFORM_THRESHOLDS)
            ),
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
        self._foreground_condition = threading.Condition()
        self._foreground_waiting = 0
        self._job_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aigc-explanation")
        self._scan_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="aigc-transform-scan"
        )
        self.model: TraceDetector | None = None
        self.config: dict | None = None
        self.device: torch.device | None = None
        self.source_epoch: int | None = None
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
        self.transform_thresholds_sha256 = sha256(self.settings.transform_thresholds)
        self.transform_thresholds_payload = json.loads(
            self.settings.transform_thresholds.read_text()
        )
        if self.transform_thresholds_payload.get("input_protocol") != INPUT_PROTOCOL_ID:
            raise RuntimeError("Transform thresholds use a different input protocol")
        thresholds = self.transform_thresholds_payload.get("thresholds", {})
        missing_thresholds = set(TRANSFORMS_BY_ID).difference(thresholds)
        extra_thresholds = set(thresholds).difference(TRANSFORMS_BY_ID)
        if missing_thresholds or extra_thresholds:
            raise RuntimeError(
                "Transform threshold IDs do not match the transform catalog: "
                f"missing={sorted(missing_thresholds)}, extra={sorted(extra_thresholds)}"
            )
        self.transform_thresholds = {
            transform_id: float(value) for transform_id, value in thresholds.items()
        }
        invalid_thresholds = {
            key: value
            for key, value in self.transform_thresholds.items()
            if not 0.0 < value < 1.0
        }
        if invalid_thresholds:
            raise RuntimeError(f"Transform thresholds must be in (0, 1): {invalid_thresholds}")
        self.jobs: dict[str, dict] = {}
        self.transform_scans: dict[str, dict] = {}

    @property
    def decision_threshold(self) -> float:
        """Return the historical calibration-only reference threshold."""
        return float(self.calibration["threshold"])

    def threshold_for_transform(self, transform_id: str) -> float:
        try:
            return self.transform_thresholds[transform_id]
        except KeyError as error:
            raise ValueError(f"No threshold configured for transform '{transform_id}'") from error

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def foreground_waiting(self) -> int:
        with self._foreground_condition:
            return self._foreground_waiting

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
            self.source_epoch = checkpoint.get("epoch")

    def model_info(self) -> dict:
        if self.config is None:
            checkpoint = torch.load(
                self.settings.checkpoint, map_location="cpu", weights_only=False
            )
            config = checkpoint["config"]
            source_epoch = checkpoint.get("epoch")
        else:
            config = self.config
            source_epoch = self.source_epoch
        return {
            "method": "773086_759921_architecture_mlp_normalized",
            "checkpoint": str(self.settings.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "source_epoch": source_epoch,
            "forensic_backbone": config["model"]["forensic_backbone"],
            "semantic_backbone": config["model"]["semantic_backbone"],
            "loss_mode": config.get("loss", {}).get("mode"),
            "confidence": "piecewise_linearized_fp32_platt_score",
            "calibration": str(self.settings.calibration),
            "calibration_sha256": self.calibration_sha256,
            "calibration_temperature": float(self.calibration["temperature"]),
            "calibration_bias": float(self.calibration["bias"]),
            "calibrated_probability_threshold": self.threshold_for_transform("clean"),
            "default_threshold_transform": "clean",
            "reference_calibration_threshold": self.decision_threshold,
            "threshold_strategy": "per_transform_aligned_640_test_oracle",
            "transform_thresholds": copy.deepcopy(self.transform_thresholds),
            "transform_thresholds_config": str(self.settings.transform_thresholds),
            "transform_thresholds_sha256": self.transform_thresholds_sha256,
            "transform_thresholds_statistical_status": self.transform_thresholds_payload.get(
                "statistical_status"
            ),
            "decision_threshold": 0.5,
            "confidence_mapping": (
                "p<=t: 0.5*p/t; p>t: 0.5+0.5*(p-t)/(1-t)"
            ),
            "input_protocol": copy.deepcopy(INPUT_PROTOCOL),
            "device": str(self.device) if self.device is not None else self.settings.device,
            "loaded": self.loaded,
        }

    def _score_transformed(
        self,
        transformed: Image.Image,
        original: Image.Image,
        image_digest: str,
        transform: dict,
        *,
        include_model: bool,
    ) -> dict:
        """Score one already transformed image while the inference lock is held."""
        self.ensure_loaded()
        assert self.model is not None and self.config is not None and self.device is not None
        selected_threshold = self.threshold_for_transform(str(transform["id"]))
        rows, outputs = score_images(
            self.model,
            [transformed],
            self.config,
            self.device,
            float(self.calibration["temperature"]),
            float(self.calibration["bias"]),
            1,
            selected_threshold,
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
                "calibrated_probability_threshold": selected_threshold,
                "threshold_source": "per_transform_aligned_640_test_oracle",
                "confidence_semantics": (
                    "piecewise-linearized FP32 Platt score; probability_fake retains the pre-mapping calibrated value"
                ),
            }
        )
        return {
            "prediction": prediction,
            "image": {
                "sha256": image_digest,
                "width": original.width,
                "height": original.height,
                "inference_base_width": INPUT_PROTOCOL["width"],
                "inference_base_height": INPUT_PROTOCOL["height"],
                "transformed_width": transformed.width,
                "transformed_height": transformed.height,
                "input_protocol_id": INPUT_PROTOCOL_ID,
            },
            "transform": copy.deepcopy(transform),
            **({"model": self.model_info()} if include_model else {}),
            "branches_available": [
                key
                for key in ("tile_attention", "tile_logits", "wavelet_similarity")
                if key in model_outputs
            ],
        }

    def predict_selected(
        self,
        image: Image.Image,
        image_digest: str,
        transform_id: str | None = "clean",
    ) -> dict:
        """Run one user-selected transform with priority over background scans."""
        transform = resolve_transform(transform_id)
        with self._foreground_condition:
            self._foreground_waiting += 1
            self._foreground_condition.notify_all()
        try:
            inference_base = align_for_inference(image)
            transformed = apply_transform(inference_base, transform, image_digest)
            with self._inference_lock:
                return self._score_transformed(
                    transformed,
                    image,
                    image_digest,
                    transform,
                    include_model=True,
                )
        finally:
            with self._foreground_condition:
                self._foreground_waiting -= 1
                self._foreground_condition.notify_all()

    def predict(self, image: Image.Image, image_digest: str) -> dict:
        """Backward-compatible clean prediction."""
        return self.predict_selected(image, image_digest, "clean")

    def _predict_background(
        self,
        image: Image.Image,
        image_digest: str,
        transform: dict,
    ) -> dict:
        inference_base = align_for_inference(image)
        transformed = apply_transform(inference_base, transform, image_digest)
        while True:
            with self._foreground_condition:
                while self._foreground_waiting:
                    self._foreground_condition.wait()
                acquired = self._inference_lock.acquire(blocking=False)
                if acquired:
                    break
                self._foreground_condition.wait(timeout=0.01)
        try:
            return self._score_transformed(
                transformed,
                image,
                image_digest,
                transform,
                include_model=False,
            )
        finally:
            self._inference_lock.release()

    @staticmethod
    def _compact_transform_result(result: dict) -> dict:
        return {
            key: copy.deepcopy(result[key])
            for key in ("prediction", "image", "transform", "branches_available")
            if key in result
        }

    def submit_transform_scan(
        self,
        payload: bytes,
        filename: str,
        image_digest: str,
        selected_result: dict,
    ) -> dict:
        selected_transform = str(selected_result["transform"]["id"])
        remaining = [
            item["id"] for item in TRANSFORM_CATALOG if item["id"] != selected_transform
        ]
        scan_id = uuid.uuid4().hex
        scan = {
            "scan_id": scan_id,
            "status": "queued",
            "filename": filename,
            "image_sha256": image_digest,
            "selected_transform": selected_transform,
            "order": [selected_transform, *remaining],
            "completed_count": 1,
            "total_count": len(TRANSFORM_CATALOG),
            "current_transform": None,
            "results": {
                selected_transform: self._compact_transform_result(selected_result)
            },
            "created_at": utc_now(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "scheduling": "foreground_selected_then_idle_background_sequential",
        }
        with self._job_lock:
            self.transform_scans[scan_id] = scan
        self._scan_executor.submit(
            self._run_transform_scan,
            scan_id,
            payload,
            image_digest,
            remaining,
        )
        return copy.deepcopy(scan)

    def _run_transform_scan(
        self,
        scan_id: str,
        payload: bytes,
        image_digest: str,
        remaining: list[str],
    ) -> None:
        self._update_transform_scan(scan_id, status="running", started_at=utc_now())
        try:
            image, _extension = validate_image(payload)
            for transform_id in remaining:
                self._update_transform_scan(scan_id, current_transform=transform_id)
                result = self._predict_background(
                    image,
                    image_digest,
                    resolve_transform(transform_id),
                )
                with self._job_lock:
                    scan = self.transform_scans[scan_id]
                    scan["results"][transform_id] = self._compact_transform_result(result)
                    scan["completed_count"] = len(scan["results"])
            self._update_transform_scan(
                scan_id,
                status="completed",
                current_transform=None,
                completed_at=utc_now(),
            )
        except Exception as error:
            self._update_transform_scan(
                scan_id,
                status="failed",
                current_transform=None,
                completed_at=utc_now(),
                error=f"{type(error).__name__}: {error}",
            )

    def _update_transform_scan(self, scan_id: str, **updates) -> None:
        with self._job_lock:
            self.transform_scans[scan_id].update(updates)

    def get_transform_scan(self, scan_id: str) -> dict:
        with self._job_lock:
            scan = self.transform_scans.get(scan_id)
            if scan is None:
                raise KeyError(scan_id)
            return copy.deepcopy(scan)

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
        original, _original_extension = validate_image(payload)
        input_path = job_root / "upload.jpg"
        input_path.write_bytes(aligned_inference_payload(original))
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
                decision_threshold=self.threshold_for_transform("clean"),
                transform_thresholds=copy.deepcopy(self.transform_thresholds),
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

    def shutdown(self) -> None:
        self._scan_executor.shutdown(wait=False, cancel_futures=True)
        self._executor.shutdown(wait=False, cancel_futures=True)


def create_app(
    settings: RuntimeSettings | None = None,
    runtime: LocalModelRuntime | None = None,
) -> FastAPI:
    settings = settings or RuntimeSettings.from_environment()
    if not settings.checkpoint.is_file():
        raise FileNotFoundError(settings.checkpoint)
    settings.results_root.mkdir(parents=True, exist_ok=True)
    runtime = runtime or LocalModelRuntime(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        shutdown = getattr(runtime, "shutdown", None)
        if shutdown is not None:
            shutdown()

    app = FastAPI(
        title="773086 Calibrated Local AIGC Explainability API",
        version="1.2.0",
        description=(
            "Local-only foreground-priority transform confidence and counterfactual explanation service."
        ),
        lifespan=lifespan,
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
        analysis_active = sum(
            runtime.get_job(job_id)["status"] in {"queued", "running"}
            for job_id in list(runtime.jobs)
        )
        transform_active = sum(
            runtime.get_transform_scan(scan_id)["status"] in {"queued", "running"}
            for scan_id in list(getattr(runtime, "transform_scans", {}))
        )
        return {
            "status": "ok",
            "model_loaded": runtime.loaded,
            "device": str(runtime.device) if runtime.device else settings.device,
            "queued_or_running": analysis_active + transform_active,
            "analysis_queued_or_running": analysis_active,
            "transform_scans_queued_or_running": transform_active,
            "foreground_detection_requests": getattr(runtime, "foreground_waiting", 0),
        }

    @app.get("/api/v1/model")
    def model_info() -> dict:
        return runtime.model_info()

    @app.get("/api/v1/transforms")
    def transforms() -> dict:
        catalog = copy.deepcopy(TRANSFORM_CATALOG)
        threshold_lookup = getattr(runtime, "threshold_for_transform", None)
        if callable(threshold_lookup):
            for item in catalog:
                item["calibrated_probability_threshold"] = threshold_lookup(item["id"])
        return {
            "default": "clean",
            "count": len(TRANSFORM_CATALOG),
            "transforms": catalog,
        }

    @app.post("/api/v1/predict")
    def predict(
        request: Request,
        file: UploadFile = File(...),
        transform: str = Form("clean"),
    ) -> dict:
        payload = file.file.read(MAX_UPLOAD_BYTES + 1)
        image, _extension = validate_image(payload)
        image_digest = hashlib.sha256(payload).hexdigest()
        try:
            selected = runtime.predict_selected(image, image_digest, transform)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        scan = runtime.submit_transform_scan(
            payload,
            file.filename or "upload",
            image_digest,
            selected,
        )
        scan_id = scan["scan_id"]
        selected["transform_scan"] = {
            **{key: value for key, value in scan.items() if key != "results"},
            "status_url": str(request.url_for("transform_scan_status", scan_id=scan_id)),
        }
        return selected

    @app.get("/api/v1/transform-scans/{scan_id}", name="transform_scan_status")
    def transform_scan_status(scan_id: str) -> dict:
        try:
            scan = runtime.get_transform_scan(scan_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Unknown transform scan") from error
        scan["status_url"] = f"/api/v1/transform-scans/{scan_id}"
        return scan

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
