from __future__ import annotations

import io
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torchvision.transforms import functional as TF


DEGRADATION_NAMES = ["clean", "jpeg", "blur", "resize", "noise", "color", "crop", "compound"]
JPEG_QUALITIES = [90, 70, 50, 30]
BLUR_SIGMAS = [0.5, 1.0, 2.0]
RESIZE_SCALES = [0.5, 0.25]
NOISE_SIGMAS = [0.02, 0.05, 0.10]
COLOR_STRENGTH = 0.20
CROP_RATIO = 0.80
NATIVE_SPECTRAL_BINS = 12
NATIVE_SPECTRAL_DIM = NATIVE_SPECTRAL_BINS + 6


@dataclass(frozen=True)
class Degradation:
    name: str
    class_id: int
    severity: float
    operations: tuple[dict, ...]


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, subsampling=2)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").copy()


def _resize_roundtrip(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    small = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BICUBIC,
    )
    return small.resize((width, height), Image.Resampling.BICUBIC)


def _noise(image: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    array = np.clip(array + rng.normal(0.0, sigma, array.shape), 0.0, 1.0)
    return Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="RGB")


def _color_jitter(image: Image.Image, rng: random.Random) -> tuple[Image.Image, dict]:
    operations = [
        ("brightness", ImageEnhance.Brightness, rng.uniform(1.0 - COLOR_STRENGTH, 1.0 + COLOR_STRENGTH)),
        ("contrast", ImageEnhance.Contrast, rng.uniform(1.0 - COLOR_STRENGTH, 1.0 + COLOR_STRENGTH)),
        ("saturation", ImageEnhance.Color, rng.uniform(1.0 - COLOR_STRENGTH, 1.0 + COLOR_STRENGTH)),
    ]
    rng.shuffle(operations)
    output = image
    for _, enhancer, factor in operations:
        output = enhancer(output).enhance(factor)
    factors = {name: factor for name, _, factor in operations}
    return output, {
        "transform": "color_jitter",
        **factors,
        "order": [name for name, _, _ in operations],
    }


def _center_crop(image: Image.Image, ratio: float) -> Image.Image:
    width, height = image.size
    crop_width, crop_height = max(1, round(width * ratio)), max(1, round(height * ratio))
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    crop = image.crop((left, top, left + crop_width, top + crop_height))
    return crop.resize((width, height), Image.Resampling.BICUBIC)


def apply_named(
    image: Image.Image,
    name: str,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[Image.Image, float, dict]:
    if name == "jpeg":
        value = rng.choice(JPEG_QUALITIES)
        return _jpeg(image, value), (100 - value) / 70, {"transform": "jpeg", "quality": value}
    if name == "blur":
        value = rng.choice(BLUR_SIGMAS)
        return (
            image.filter(ImageFilter.GaussianBlur(radius=value)),
            value / max(BLUR_SIGMAS),
            {"transform": "gaussian_blur", "sigma": value},
        )
    if name == "resize":
        value = rng.choice(RESIZE_SCALES)
        return (
            _resize_roundtrip(image, value),
            (1.0 - value) / 0.75,
            {"transform": "resize_roundtrip", "downscale": value, "upscale": "bicubic"},
        )
    if name == "noise":
        value = rng.choice(NOISE_SIGMAS)
        return (
            _noise(image, value, np_rng),
            value / max(NOISE_SIGMAS),
            {"transform": "gaussian_noise", "sigma": value, "pixel_range": "0_to_1"},
        )
    if name == "color":
        output, operation = _color_jitter(image, rng)
        return output, 1.0, operation
    if name == "crop":
        return (
            _center_crop(image, CROP_RATIO),
            1.0,
            {"transform": "center_crop", "ratio": CROP_RATIO, "restore": "bicubic"},
        )
    if name == "clean":
        return image.copy(), 0.0, {"transform": "clean"}
    raise KeyError(name)


def apply_fixed(image: Image.Image, name: str, value: float | int | None, seed: int = 0) -> Image.Image:
    """Apply one exact competition transform for deterministic evaluation."""
    if name == "clean":
        return image.copy()
    if name == "jpeg":
        return _jpeg(image, int(value))
    if name == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(value)))
    if name == "resize":
        return _resize_roundtrip(image, float(value))
    if name == "noise":
        return _noise(image, float(value), np.random.default_rng(seed))
    if name == "color":
        factor = float(value)
        output = ImageEnhance.Brightness(image).enhance(factor)
        output = ImageEnhance.Contrast(output).enhance(factor)
        return ImageEnhance.Color(output).enhance(factor)
    if name == "crop":
        return _center_crop(image, float(value))
    raise KeyError(name)


def fixed_operation(name: str, value: float | int | None) -> dict:
    """Return the auditable parameterization used by ``apply_fixed``."""
    if name == "clean":
        return {"transform": "clean"}
    if name == "jpeg":
        return {"transform": "jpeg", "quality": int(value)}
    if name == "blur":
        return {"transform": "gaussian_blur", "sigma": float(value)}
    if name == "resize":
        return {"transform": "resize_roundtrip", "downscale": float(value), "upscale": "bicubic"}
    if name == "noise":
        return {"transform": "gaussian_noise", "sigma": float(value), "pixel_range": "0_to_1"}
    if name == "color":
        factor = float(value)
        return {
            "transform": "color_jitter_fixed",
            "brightness": factor,
            "contrast": factor,
            "saturation": factor,
            "order": ["brightness", "contrast", "saturation"],
        }
    if name == "crop":
        return {"transform": "center_crop", "ratio": float(value), "restore": "bicubic"}
    raise KeyError(name)


def competition_grid() -> list[tuple[str, str, float | int | None]]:
    return [
        ("clean", "clean", None),
        *((f"jpeg_q{quality}", "jpeg", quality) for quality in JPEG_QUALITIES),
        *((f"blur_sigma{sigma:g}", "blur", sigma) for sigma in BLUR_SIGMAS),
        *((f"resize_{scale:g}x", "resize", scale) for scale in RESIZE_SCALES),
        *((f"noise_sigma{sigma:.2f}", "noise", sigma) for sigma in NOISE_SIGMAS),
        ("color_minus20", "color", 0.8),
        ("color_plus20", "color", 1.2),
        ("center_crop_80", "crop", CROP_RATIO),
    ]


class CompetitionDegradation:
    def __init__(self, clean_probability: float = 0.2, compound_probability: float = 0.3):
        self.clean_probability = clean_probability
        self.compound_probability = compound_probability

    def __call__(self, image: Image.Image) -> tuple[Image.Image, Degradation]:
        rng = random.Random(random.getrandbits(64))
        np_rng = np.random.default_rng(rng.getrandbits(64))
        draw = rng.random()
        if draw < self.clean_probability:
            return image.copy(), Degradation("clean", 0, 0.0, ({"transform": "clean"},))
        choices = DEGRADATION_NAMES[1:7]
        if draw < self.clean_probability + self.compound_probability:
            names = rng.sample(choices, k=rng.choice([2, 3]))
            output, severities, operations = image.copy(), [], []
            for name in names:
                output, severity, operation = apply_named(output, name, rng, np_rng)
                severities.append(severity)
                operations.append(operation)
            return output, Degradation("compound", 7, float(np.mean(severities)), tuple(operations))
        name = rng.choice(choices)
        output, severity, operation = apply_named(image, name, rng, np_rng)
        return output, Degradation(name, DEGRADATION_NAMES.index(name), severity, (operation,))


def resize_tensor(image: Image.Image, size: int) -> torch.Tensor:
    return TF.to_tensor(image.resize((size, size), Image.Resampling.BICUBIC))


def native_spectral_signature(
    image: Image.Image,
    *,
    max_crop: int = 256,
    bins: int = NATIVE_SPECTRAL_BINS,
) -> torch.Tensor:
    """Return fixed-size high-frequency statistics before any interpolation.

    The largest centred native-resolution crop is used.  No resize, resampling,
    JPEG round-trip, or padding is performed, so the signature cannot invent or
    erase frequencies merely to satisfy a backbone input size.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    height, width = gray.shape
    crop_height, crop_width = min(height, max_crop), min(width, max_crop)
    top = max(0, (height - crop_height) // 2)
    left = max(0, (width - crop_width) // 2)
    gray = gray[top : top + crop_height, left : left + crop_width]

    # Spatial derivatives retain camera/demosaicing and synthesis residuals.
    gx = np.diff(gray, axis=1) if gray.shape[1] > 1 else np.zeros_like(gray)
    gy = np.diff(gray, axis=0) if gray.shape[0] > 1 else np.zeros_like(gray)
    padded = np.pad(gray, 1, mode="reflect") if min(gray.shape) > 1 else np.pad(gray, 1)
    laplacian = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    spatial = np.asarray(
        [
            np.mean(np.abs(gx)), np.std(gx),
            np.mean(np.abs(gy)), np.std(gy),
            np.mean(np.abs(laplacian)), np.std(laplacian),
        ],
        dtype=np.float32,
    )

    # Radial log-power profile. Normalising by total log energy makes it less
    # sensitive to brightness while retaining relative spectral roll-off.
    centred = gray - float(gray.mean())
    window = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1])).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(centred * window))
    log_power = np.log1p(np.abs(spectrum) ** 2)
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    yy -= (gray.shape[0] - 1) / 2.0
    xx -= (gray.shape[1] - 1) / 2.0
    radius = np.sqrt(xx * xx + yy * yy)
    radius /= max(float(radius.max()), 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    radial = []
    for index in range(bins):
        mask = (radius >= edges[index]) & (radius < edges[index + 1])
        radial.append(float(log_power[mask].mean()) if mask.any() else 0.0)
    radial_array = np.asarray(radial, dtype=np.float32)
    radial_array /= max(float(radial_array.sum()), 1e-6)
    return torch.from_numpy(np.concatenate([radial_array, spatial]))


def native_tiles(image: Image.Image, size: int, count: int, training: bool) -> torch.Tensor:
    image = image.convert("RGB")
    width, height = image.size
    if min(width, height) < size:
        scale = size / min(width, height)
        image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
        width, height = image.size
    if training:
        positions = [
            (
                random.randint(0, max(0, width - size)),
                random.randint(0, max(0, height - size)),
            )
            for _ in range(count)
        ]
    else:
        candidates = [
            (0, 0),
            (max(0, width - size), 0),
            (0, max(0, height - size)),
            (max(0, width - size), max(0, height - size)),
            (max(0, (width - size) // 2), max(0, (height - size) // 2)),
        ]
        positions = [candidates[idx % len(candidates)] for idx in range(count)]
    return torch.stack(
        [TF.to_tensor(image.crop((x, y, x + size, y + size))) for x, y in positions]
    )


def haar_high_frequency_perturbation(images: torch.Tensor, strength: float = 0.12) -> torch.Tensor:
    """Add a bounded Haar-like high-frequency residual to [0, 1] tensors."""
    low = F.avg_pool2d(images, kernel_size=2, stride=2)
    low = F.interpolate(low, size=images.shape[-2:], mode="nearest")
    high = images - low
    return (images + strength * torch.sign(high) * torch.sqrt(high.abs() + 1e-6)).clamp(0, 1)
