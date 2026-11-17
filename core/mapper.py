"""
Stage 5/6 - Waterlogging map generation and georeferenced output.

Converts U-Net segmentation predictions into georeferenced waterlogging extent
and severity maps (GeoTIFF / PNG) suitable for drainage planning and
crop-damage assessment.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import config
import numpy as np

LOGGER = logging.getLogger("deepwaterlog.mapper")

# RGB palette for the five classes (matches config.CLASS_COLORS)
DEFAULT_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (214, 40, 40),    # waterlogged_soil  - red
    1: (64, 145, 108),   # healthy_crop      - green
    2: (244, 162, 97),   # stressed_crop     - orange
    3: (42, 111, 183),   # open_water        - blue
    4: (150, 150, 150),  # non_agricultural  - grey
}


@dataclass
class MapResult:
    """Result of map generation."""

    extent_map: np.ndarray          # (H, W) integer class ids
    severity_map: np.ndarray | None = None  # (H, W) waterlogged severity 0-1
    class_counts: dict[int, int] = field(default_factory=dict)
    class_areas_ha: dict[int, float] = field(default_factory=dict)
    waterlogged_area_ha: float = 0.0
    total_area_ha: float = 0.0
    waterlogged_fraction: float = 0.0
    geotransform: tuple[float, ...] | None = None
    crs: str | None = None
    output_path: str | None = None


def generate_maps(
    predictions: np.ndarray,
    *,
    class_names: list[str] | None = None,
    resolution_m: float | None = None,
    geotransform: tuple[float, ...] | None = None,
    crs: str | None = None,
) -> MapResult:
    """
    Build per-class pixel counts and area estimates from U-Net predictions.

    ``predictions`` is an integer class-id map of shape (H, W).
    """
    class_names = class_names or config.CLASSES
    resolution_m = config.MAP_RESOLUTION_M if resolution_m is None else resolution_m
    predictions = np.asarray(predictions).astype(np.int64)

    n_classes = len(class_names)
    unique, counts = np.unique(predictions, return_counts=True)
    class_counts = {int(u): int(c) for u, c in zip(unique, counts)}
    for cls in range(n_classes):
        class_counts.setdefault(cls, 0)

    total_pixels = int(predictions.size)
    pixel_area_m2 = resolution_m * resolution_m
    pixel_area_ha = pixel_area_m2 / 10_000.0

    class_areas_ha = {
        cls: class_counts.get(cls, 0) * pixel_area_ha for cls in range(n_classes)
    }
    total_area_ha = total_pixels * pixel_area_ha

    # Waterlogged area = waterlogged_soil + stressed_crop (+ open_water optionally)
    waterlogged_classes = {0, 2}  # waterlogged_soil, stressed_crop
    waterlogged_pixels = sum(class_counts.get(cls, 0) for cls in waterlogged_classes)
    waterlogged_area_ha = waterlogged_pixels * pixel_area_ha
    waterlogged_fraction = waterlogged_pixels / total_pixels if total_pixels else 0.0

    # Severity map: 1.0 for waterlogged_soil and stressed_crop, 0 elsewhere
    severity_map = np.isin(predictions, list(waterlogged_classes)).astype(np.float32)

    result = MapResult(
        extent_map=predictions.copy(),
        severity_map=severity_map,
        class_counts=class_counts,
        class_areas_ha=class_areas_ha,
        waterlogged_area_ha=waterlogged_area_ha,
        total_area_ha=total_area_ha,
        waterlogged_fraction=waterlogged_fraction,
        geotransform=geotransform,
        crs=crs,
    )

    LOGGER.info(
        "Map generated: total=%.2f ha, waterlogged=%.2f ha (%.2f%%), "
        "class_counts=%s",
        total_area_ha, waterlogged_area_ha, waterlogged_fraction * 100,
        class_counts,
    )
    return result


def save_geotiff(
    array: np.ndarray,
    path: str,
    *,
    geotransform: tuple[float, ...] | None = None,
    crs: str | None = None,
    nodata: int | float | None = None,
) -> str:
    """
    Write a 2-D array to GeoTIFF if rasterio is available; otherwise save as
    raw .npy and log a warning.

    Returns the actual output path.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        out = path.replace(".tif", ".npy").replace(".tiff", ".npy") if path.lower().endswith((".tif", ".tiff")) else path + ".npy"
        np.save(out, array)
        LOGGER.warning(
            "rasterio not installed; saved array to %s instead of GeoTIFF. "
            "Install rasterio for georeferenced output.",
            out,
        )
        return out

    profile = {
        "driver": "GTiff",
        "dtype": str(array.dtype),
        "width": array.shape[1],
        "height": array.shape[0],
        "count": 1,
        "crs": crs or "EPSG:4326",
        "transform": geotransform or (1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
        "nodata": nodata,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(profile["dtype"]), 1)
    LOGGER.info("GeoTIFF written to %s", path)
    return path


# ---------------------------------------------------------------------------
# Full-scene inference
# ---------------------------------------------------------------------------
def sliding_window_predict(
    model: Any,
    cube: np.ndarray,
    *,
    patch_size: int | None = None,
    stride: int | None = None,
    num_classes: int | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a segmentation model over a whole hyperspectral cube using a sliding
    window and average the per-pixel class probabilities across overlapping
    windows.

    Parameters
    ----------
    model : torch.nn.Module
        Trained segmentation model producing logits (B, C, H, W).
    cube : np.ndarray
        Band-selected cube of shape (bands, height, width).
    patch_size, stride : int, optional
        Window size and step; default to config.PATCH_SIZE / config.STRIDE.
    num_classes : int, optional
        Number of classes; defaults to len(config.CLASSES).

    Returns
    -------
    (predictions, probabilities)
        predictions : (height, width) integer class ids
        probabilities : (num_classes, height, width) mean class probabilities
    """
    import torch

    ps = int(patch_size or config.PATCH_SIZE)
    st = int(stride or max(1, ps // 2))
    nc = int(num_classes or len(config.CLASSES))

    cube = np.asarray(cube, dtype=np.float32)
    bands, height, width = cube.shape

    # Pad the scene so it is at least one full window in each direction
    pad_h = max(0, ps - height)
    pad_w = max(0, ps - width)
    if pad_h or pad_w:
        cube = np.pad(cube, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")

    hh, ww = cube.shape[1], cube.shape[2]
    prob_sum = np.zeros((nc, hh, ww), dtype=np.float32)
    weight = np.zeros((hh, ww), dtype=np.float32)

    # Gaussian window weighting: centre pixels count more than window edges,
    # which removes the visible grid artefacts produced by uniform averaging.
    axis = np.arange(ps) - (ps - 1) / 2.0
    sigma = max(1.0, ps * 0.35)
    gauss_1d = np.exp(-(axis ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    window = np.outer(gauss_1d, gauss_1d)
    window /= window.max()

    tops = list(range(0, max(1, hh - ps + 1), st))
    lefts = list(range(0, max(1, ww - ps + 1), st))

    model.eval()
    with torch.no_grad():
        for top in tops:
            for left in lefts:
                patch = cube[:, top:top + ps, left:left + ps]
                if patch.shape[1] != ps or patch.shape[2] != ps:
                    continue
                tensor = torch.from_numpy(patch[None]).float().to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                prob_sum[:, top:top + ps, left:left + ps] += probs * window[None]
                weight[top:top + ps, left:left + ps] += window

    weight[weight == 0] = 1.0
    prob_mean = prob_sum / weight[None, :, :]
    predictions = prob_mean.argmax(axis=0).astype(np.int64)

    # Remove padding
    predictions = predictions[:height, :width]
    prob_mean = prob_mean[:, :height, :width]

    LOGGER.info(
        "Full-scene inference: %d windows over %dx%d pixels (%d bands).",
        len(tops) * len(lefts), height, width, bands,
    )
    return predictions, prob_mean


# ---------------------------------------------------------------------------
# PNG rendering (no GDAL required)
# ---------------------------------------------------------------------------
def _class_map_to_rgb(
    array: np.ndarray,
    palette: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Map integer class ids to an (H, W, 3) uint8 RGB image."""
    palette = palette or DEFAULT_PALETTE
    array = np.asarray(array)
    rgb = np.zeros((*array.shape, 3), dtype=np.uint8)
    for class_id, colour in palette.items():
        rgb[array == class_id] = colour
    return rgb


def save_class_map_png(
    array: np.ndarray,
    path: str,
    *,
    palette: dict[int, tuple[int, int, int]] | None = None,
    scale: int = 8,
) -> str:
    """
    Write an integer class map to a PNG using a fixed class palette.

    ``scale`` enlarges each pixel block so small test scenes remain legible.
    Requires Pillow; falls back to a .npy dump if Pillow is unavailable.
    """
    rgb = _class_map_to_rgb(array, palette)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    try:
        from PIL import Image
    except ImportError:
        fallback = path.replace(".png", ".npy")
        np.save(fallback, array)
        LOGGER.warning("Pillow not installed; saved class map to %s instead.", fallback)
        return fallback

    image = Image.fromarray(rgb, mode="RGB")
    if scale and scale > 1:
        image = image.resize(
            (image.width * scale, image.height * scale), Image.NEAREST
        )
    image.save(path, format="PNG")
    LOGGER.info("Class map PNG written to %s", path)
    return path


def save_severity_png(
    severity: np.ndarray,
    path: str,
    *,
    scale: int = 8,
) -> str:
    """
    Write a 0-1 severity map to a PNG using a yellow-to-red ramp.

    Requires Pillow. Returns the written path.
    """
    severity = np.clip(np.asarray(severity, dtype=np.float32), 0.0, 1.0)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    try:
        from PIL import Image
    except ImportError:
        fallback = path.replace(".png", ".npy")
        np.save(fallback, severity)
        LOGGER.warning("Pillow not installed; saved severity map to %s.", fallback)
        return fallback

    # Yellow (255, 237, 160) -> red (214, 40, 40)
    r = (255 - (255 - 214) * severity).astype(np.uint8)
    g = (237 - (237 - 40) * severity).astype(np.uint8)
    b = (160 - (160 - 40) * severity).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)

    image = Image.fromarray(rgb, mode="RGB")
    if scale and scale > 1:
        image = image.resize(
            (image.width * scale, image.height * scale), Image.NEAREST
        )
    image.save(path, format="PNG")
    LOGGER.info("Severity PNG written to %s", path)
    return path


def export_report(
    result: MapResult,
    class_names: list[str] | None = None,
    path: str = "waterlogging_report.md",
) -> str:
    """Write a Markdown report summarising the waterlogging map."""
    class_names = class_names or config.CLASSES
    lines = []
    lines.append("# Waterlogging Detection Report")
    lines.append("")
    lines.append(f"- Total area: {result.total_area_ha:.2f} ha")
    lines.append(f"- Waterlogged area: {result.waterlogged_area_ha:.2f} ha")
    lines.append(f"- Waterlogged fraction: {result.waterlogged_fraction:.2%}")
    lines.append("")
    lines.append("## Per-class area")
    lines.append("| class | pixels | area (ha) |")
    lines.append("|---|---|---|")
    for cls in range(len(class_names)):
        lines.append(
            f"| {class_names[cls]} | {result.class_counts.get(cls, 0)} | "
            f"{result.class_areas_ha.get(cls, 0.0):.2f} |"
        )
    lines.append("")
    lines.append("## Output files")
    if result.output_path:
        lines.append(f"- Map: {result.output_path}")
    report = "\n".join(lines)
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report)
    LOGGER.info("Report written to %s", path)
    return report
