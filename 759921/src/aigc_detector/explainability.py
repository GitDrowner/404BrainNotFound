from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


SCHEMA_VERSION = "aigc-explanation-v1"


def patch_boxes(width: int, height: int, columns: int, rows: int | None = None) -> list[tuple[int, int, int, int]]:
    """Partition an image into deterministic, gap-free boxes."""
    rows = rows or columns
    if width <= 0 or height <= 0 or columns <= 0 or rows <= 0:
        raise ValueError("image and grid dimensions must be positive")
    x_edges = [round(index * width / columns) for index in range(columns + 1)]
    y_edges = [round(index * height / rows) for index in range(rows + 1)]
    return [
        (x_edges[column], y_edges[row], x_edges[column + 1], y_edges[row + 1])
        for row in range(rows)
        for column in range(columns)
    ]


def subdivide_box(box: tuple[int, int, int, int], divisions: int) -> list[tuple[int, int, int, int]]:
    """Split one coarse region into a gap-free local grid."""
    if divisions <= 0:
        raise ValueError("divisions must be positive")
    left, top, right, bottom = box
    local = patch_boxes(right - left, bottom - top, divisions)
    return [(x0 + left, y0 + top, x1 + left, y1 + top) for x0, y0, x1, y1 in local]


def occlude_patch(image: Image.Image, box: tuple[int, int, int, int], mode: str = "blur") -> Image.Image:
    """Create a counterfactual by replacing exactly one native-image patch."""
    image = image.convert("RGB")
    output = image.copy()
    if mode == "blur":
        radius = max(3.0, min(image.size) / 24.0)
        replacement = image.filter(ImageFilter.GaussianBlur(radius=radius)).crop(box)
    elif mode == "mean":
        region = image.crop(box).resize((1, 1), Image.Resampling.BOX)
        replacement = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), region.getpixel((0, 0)))
    else:
        raise ValueError(f"Unknown occlusion mode: {mode}")
    output.paste(replacement, box)
    return output


def suppress_high_frequency_patch(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    radius: float | None = None,
) -> Image.Image:
    """Locally low-pass one region while retaining its colour and coarse structure."""
    base = image.convert("RGB")
    radius = radius or max(1.0, min(base.size) / 180.0)
    output = base.copy()
    output.paste(base.filter(ImageFilter.GaussianBlur(radius=radius)).crop(box), box)
    return output


def save_attribution_overlay(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    contributions: list[float],
    path: Path,
) -> None:
    """Save red/blue patch attribution overlay; red raises AIGC confidence."""
    if len(boxes) != len(contributions):
        raise ValueError("box/contribution length mismatch")
    base = image.convert("RGB")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    scale = max((abs(value) for value in contributions), default=0.0)
    for box, value in zip(boxes, contributions):
        strength = 0.0 if scale == 0 else min(1.0, abs(value) / scale)
        alpha = round(35 + 170 * strength) if strength else 18
        color = (225, 48, 48, alpha) if value >= 0 else (40, 105, 225, alpha)
        draw.rectangle(box, fill=color, outline=(255, 255, 255, 120), width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB").save(path)


def _map_anchors(values: np.ndarray, anchors: Iterable[float], colors: Iterable[tuple[int, int, int]]) -> np.ndarray:
    """Piecewise-linear colormap over anchor positions, returning an RGB uint8 array (..., 3)."""
    values = np.asarray(values, dtype=np.float32)
    anchors = np.asarray(list(anchors), dtype=np.float32)
    colors = np.asarray(list(colors), dtype=np.float32)
    out = np.empty(values.shape + (3,), dtype=np.float32)
    for channel in range(3):
        out[..., channel] = np.interp(values, anchors, colors[:, channel])
    return np.clip(out, 0, 255).astype(np.uint8)


def diverging_colormap(values: np.ndarray) -> np.ndarray:
    """Blue -> white -> red on [-1, 1] for signed contributions."""
    return _map_anchors(
        values,
        [-1.0, 0.0, 1.0],
        [(59, 76, 192), (221, 221, 221), (180, 4, 38)],
    )


def inferno_colormap(values: np.ndarray) -> np.ndarray:
    """Dark -> bright sequential map on [0, 1] for non-negative magnitudes."""
    return _map_anchors(
        values,
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [(0, 0, 4), (87, 15, 109), (188, 55, 84), (249, 142, 8), (252, 255, 164)],
    )


def _grid_shape(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int]:
    """Recover the (rows, columns) grid that produced these patch boxes."""
    columns = len({box[0] for box in boxes})
    rows = len({box[1] for box in boxes})
    if rows * columns == len(boxes):
        return rows, columns
    side = int(round(len(boxes) ** 0.5))
    if side * side == len(boxes):
        return side, side
    return len(boxes), 1


def _gaussian_blur_2d(field: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur, intended for small grid fields (not full images)."""
    if sigma <= 0:
        return field
    radius = max(1, int(round(3 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    out = field.astype(np.float32)
    out = np.apply_along_axis(lambda row: np.convolve(row, kernel, "same"), axis=1, arr=out)
    out = np.apply_along_axis(lambda col: np.convolve(col, kernel, "same"), axis=0, arr=out)
    return out


def save_heatmap_overlay(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    contributions: list[float],
    path: Path,
    *,
    colormap: str = "coolwarm",
) -> tuple[float, float]:
    """Render patch contributions as a smooth, semi-transparent heatmap over the image.

    The grid is Gaussian-smoothed and bicubically upsampled so the result reads as a
    continuous field rather than hard rectangles. Returns the (vmin, vmax) value range
    used for normalization, so a matching colorbar can be drawn.
    """
    if len(boxes) != len(contributions):
        raise ValueError("box/contribution length mismatch")
    base = image.convert("RGB")
    rows, columns = _grid_shape(boxes)
    field = np.asarray(contributions, dtype=np.float32).reshape(rows, columns)
    scale = float(np.abs(field).max()) if field.size else 0.0
    norm = field / scale if scale > 0 else np.zeros_like(field)
    if scale > 0:
        norm = _gaussian_blur_2d(norm, 0.6)
    low = Image.fromarray(norm.astype(np.float32), mode="F").resize(base.size, Image.Resampling.BICUBIC)
    norm_full = np.clip(np.asarray(low, dtype=np.float32), -1.0, 1.0)
    rgb = diverging_colormap(norm_full)
    alpha = 0.72 * np.minimum(1.0, np.abs(norm_full))
    layer = np.concatenate([rgb.astype(np.float32), (alpha * 255.0)[..., None]], axis=-1).astype(np.uint8)
    result = Image.alpha_composite(base.convert("RGBA"), Image.fromarray(layer, "RGBA")).convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(path)
    return -scale, scale


def dense_region_field(
    size: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    values: list[float],
    *,
    weights: list[float] | None = None,
) -> np.ndarray:
    """Rasterise arbitrary overlapping or hierarchical regions into one dense field."""
    if len(boxes) != len(values):
        raise ValueError("box/value length mismatch")
    if weights is None:
        weights = [1.0] * len(boxes)
    if len(weights) != len(boxes):
        raise ValueError("box/weight length mismatch")
    width, height = size
    total = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    for box, value, weight in zip(boxes, values, weights):
        left, top, right, bottom = box
        if right <= left or bottom <= top:
            continue
        total[top:bottom, left:right] += float(value) * float(weight)
        count[top:bottom, left:right] += float(weight)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def save_dense_signed_heatmaps(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    values: list[float],
    standalone_path: Path,
    overlay_path: Path,
    *,
    weights: list[float] | None = None,
) -> tuple[float, float]:
    """Save a vivid standalone map and a strong input overlay for signed model evidence."""
    base = image.convert("RGB")
    field = dense_region_field(base.size, boxes, values, weights=weights)
    scale = float(np.percentile(np.abs(field), 99.0)) if field.size else 0.0
    scale = max(scale, float(np.abs(field).max()) * 0.15, 1e-12)
    norm = np.clip(field / scale, -1.0, 1.0)
    smallest_region = min(
        (min(right - left, bottom - top) for left, top, right, bottom in boxes),
        default=min(base.size),
    )
    blur_radius = max(1.0, smallest_region / 2.5)
    encoded = Image.fromarray(np.round((norm + 1.0) * 127.5).astype(np.uint8), mode="L")
    smooth = encoded.filter(ImageFilter.GaussianBlur(blur_radius))
    norm = np.clip(np.asarray(smooth, dtype=np.float32) / 127.5 - 1.0, -1.0, 1.0)
    rgb = diverging_colormap(norm)
    standalone_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, "RGB").save(standalone_path)

    magnitude = np.abs(norm)
    alpha = 0.88 * np.sqrt(magnitude)
    rgba = np.concatenate([rgb, np.round(alpha[..., None] * 255.0).astype(np.uint8)], axis=-1)
    overlay = Image.alpha_composite(base.convert("RGBA"), Image.fromarray(rgba, "RGBA")).convert("RGB")
    overlay.save(overlay_path)
    return -scale, scale


def save_colorbar(
    path: Path,
    *,
    vmin: float,
    vmax: float,
    center: float | None = None,
    width: int = 480,
    bar_height: int = 34,
    font_size: int = 15,
) -> None:
    """Save a compact horizontal colorbar with value tick labels (for slides)."""
    text_height = 26
    canvas = Image.new("RGB", (width, text_height + bar_height), "white")
    strip = np.linspace(0.0, 1.0, width, dtype=np.float32)
    rgb = diverging_colormap((strip * 2.0) - 1.0) if center is not None else inferno_colormap(strip)
    gradient = Image.fromarray(rgb.reshape(1, width, 3), "RGB").resize((width, bar_height))
    canvas.paste(gradient, (0, text_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=font_size)
    draw.text((2, 4), f"{vmin:+.3f}", fill="black", font=font)
    if center is not None:
        label = "0.000" if abs(center) < 1e-9 else f"{center:+.3f}"
        label_width = draw.textlength(label, font=font)
        draw.text(((width - label_width) / 2, 4), label, fill="black", font=font)
    vmax_label = f"{vmax:+.3f}"
    vmax_width = draw.textlength(vmax_label, font=font)
    draw.text((width - vmax_width - 2, 4), vmax_label, fill="black", font=font)
    canvas.save(path)


def high_frequency_residual(image: Image.Image, *, blur_radius: float = 2.0) -> np.ndarray:
    """Per-pixel input high-frequency magnitude; independent of the detector output."""
    base = image.convert("RGB")
    low = np.asarray(base.filter(ImageFilter.GaussianBlur(radius=blur_radius)), dtype=np.float32)
    arr = np.asarray(base, dtype=np.float32)
    return np.abs(arr - low).mean(axis=2)


def save_texture_heatmap(image: Image.Image, path: Path, *, colormap: str = "inferno") -> None:
    """Render an input-only high-frequency diagnostic over a grayscale image."""
    base = image.convert("RGB")
    residual = high_frequency_residual(base)
    ceiling = float(np.percentile(residual, 99.0))
    norm = np.clip(residual / max(ceiling, 1e-6), 0.0, 1.0)
    norm_img = Image.fromarray((norm * 255.0).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(radius=1.5))
    norm = np.asarray(norm_img, dtype=np.float32) / 255.0
    rgb = inferno_colormap(norm)
    gray = np.asarray(base.convert("L"), dtype=np.float32)
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    blended = gray_rgb * (1.0 - 0.7 * norm)[..., None] + rgb.astype(np.float32) * (0.7 * norm)[..., None]
    result = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), "RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(path)


def save_line_svg(rows: list[dict], path: Path, *, value_key: str, title: str) -> None:
    width, height = 900, 330
    left, right, top, bottom = 64, 24, 38, 92
    plot_width, plot_height = width - left - right, height - top - bottom
    values = [float(row[value_key]) for row in rows]
    points = []
    for index, value in enumerate(values):
        x = left + (plot_width * index / max(1, len(values) - 1))
        y = top + (1.0 - max(0.0, min(1.0, value))) * plot_height
        points.append((x, y))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    labels = []
    for index, row in enumerate(rows):
        x, _ = points[index]
        label = html.escape(str(row.get("variant", index)))
        labels.append(
            f'<text x="{x:.1f}" y="{height - bottom + 18}" transform="rotate(48 {x:.1f} {height - bottom + 18})" '
            f'font-size="10" text-anchor="start">{label}</text>'
        )
    grid = []
    for tick in range(5):
        value = tick / 4
        y = top + (1 - value) * plot_height
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d6d6d6"/>')
        grid.append(f'<text x="{left-10}" y="{y+4:.1f}" font-size="11" text-anchor="end">{value:.2f}</text>')
    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#c9324d"><title>{values[index]:.6f}</title></circle>'
        for index, (x, y) in enumerate(points)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        f'<rect width="100%" height="100%" fill="white"/><text x="{left}" y="22" font-size="16">{html.escape(title)}</text>'
        + "".join(grid)
        + f'<polyline points="{polyline}" fill="none" stroke="#c9324d" stroke-width="3"/>{circles}'
        + "".join(labels)
        + f'<text x="18" y="{top + plot_height/2:.1f}" font-size="12" transform="rotate(-90 18 {top + plot_height/2:.1f})" text-anchor="middle">P(AIGC)</text></svg>'
    )
    path.write_text(svg)


def save_bar_svg(items: Iterable[tuple[str, float]], path: Path, *, title: str) -> None:
    items = list(items)
    width = 760
    row_height = 42
    height = 58 + row_height * max(1, len(items))
    label_width, plot_width = 190, 520
    maximum = max((abs(value) for _, value in items), default=1.0) or 1.0
    center = label_width + plot_width / 2
    rows = []
    for index, (name, value) in enumerate(items):
        y = 46 + index * row_height
        length = abs(value) / maximum * (plot_width / 2 - 24)
        x = center if value >= 0 else center - length
        color = "#d3424f" if value >= 0 else "#3469c8"
        rows.append(f'<text x="{label_width-12}" y="{y+16}" text-anchor="end" font-size="12">{html.escape(name)}</text>')
        rows.append(f'<rect x="{x:.1f}" y="{y}" width="{length:.1f}" height="22" fill="{color}"><title>{value:.6f}</title></rect>')
        rows.append(f'<text x="{x + (length + 6 if value >= 0 else -6):.1f}" y="{y+16}" text-anchor="{"start" if value >= 0 else "end"}" font-size="11">{value:+.4f}</text>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        f'<rect width="100%" height="100%" fill="white"/><text x="18" y="24" font-size="16">{html.escape(title)}</text>'
        f'<line x1="{center:.1f}" y1="36" x2="{center:.1f}" y2="{height-12}" stroke="#777"/>'
        + "".join(rows) + "</svg>"
    )
    path.write_text(svg)


def write_dashboard(output: Path, payload: dict, *, component_title: str) -> None:
    confidence = float(payload["prediction"]["probability_fake"])
    raw_logit = float(payload["prediction"]["raw_logit"])
    method = html.escape(str(payload["model"]["method"]))
    disclaimer = html.escape(str(payload["attribution"]["semantics"]))
    document = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AIGC explanation</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;max-width:1120px;color:#202124}} main{{display:grid;gap:22px}}
.hero{{display:grid;grid-template-columns:minmax(260px,1fr) minmax(260px,1fr);gap:18px;align-items:start}}
img,object{{max-width:100%;height:auto}} .metric{{font-size:2.3rem;font-variant-numeric:tabular-nums}}
.muted{{color:#5f6368}} .panel{{border:1px solid #ddd;padding:16px;border-radius:10px}}
code{{word-break:break-all}} @media(max-width:720px){{.hero{{grid-template-columns:1fr}} body{{margin:12px}}}}
</style></head><body><main>
<section><h1>单图 AIGC 可解释结果</h1><div class="metric">P(AIGC) = {confidence:.4f}</div>
<div class="muted">raw logit {raw_logit:+.4f} · {method}</div></section>
<section class="hero"><div class="panel"><h2>输入</h2><img src="input.png" alt="input image"></div>
<div class="panel"><h2>局部 patch 贡献</h2><img src="patch_attribution.png" alt="red patches increase AIGC confidence; blue patches decrease it"></div></section>
<section class="hero"><div class="panel"><h2>多尺度 raw-logit 贡献</h2><img src="heatmap_attribution.png" alt="standalone raw-logit attribution heatmap"><img src="heatmap_colorbar.png" alt="colorbar"></div>
<div class="panel"><h2>贡献叠加图</h2><img src="heatmap_attribution_overlay.png" alt="strong raw-logit attribution overlay"></div></section>
<section class="hero"><div class="panel"><h2>输入高频强度（非模型输出）</h2><img src="heatmap_texture.png" alt="input-only high-frequency residual"></div>
<div class="panel"><h2>模型 wavelet-only 高频贡献</h2><img src="heatmap_frequency_contribution.png" alt="model-grounded wavelet-only logit contribution"><img src="frequency_colorbar.png" alt="colorbar"></div></section>
<section class="panel"><h2>wavelet-only 贡献叠加图</h2><img src="heatmap_frequency_overlay.png" alt="model-grounded frequency contribution overlay"></section>
<section class="panel"><h2>六类失真下的置信度轨迹</h2><object data="transform_trajectory.svg" type="image/svg+xml"></object></section>
<section class="panel"><h2>{html.escape(component_title)}</h2><object data="components.svg" type="image/svg+xml"></object></section>
<section class="panel"><h2>解释边界</h2><p>{disclaimer}</p><p class="muted">独立热图使用 raw-logit，避免 sigmoid 饱和隐藏证据变化。蓝色拉低 / 红色抬高 AIGC logit。输入纹理图与 checkpoint 无关；wavelet-only 图固定其他 fusion 特征，只允许模型内部 wavelet similarity 变化。完整数值：<code>explanation.json</code>；逐区域：<code>patches.jsonl</code>。</p></section>
</main></body></html>"""
    (output / "index.html").write_text(document)


def write_schema(path: Path) -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": SCHEMA_VERSION,
        "type": "object",
        "required": ["schema_version", "model", "image", "prediction", "attribution", "transforms"],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "prediction": {"type": "object", "required": ["probability_fake", "raw_logit"]},
            "attribution": {"type": "object", "required": ["method", "semantics", "patches"]},
            "transforms": {"type": "array"},
        },
    }
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n")
