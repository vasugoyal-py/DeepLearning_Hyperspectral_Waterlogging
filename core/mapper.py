"""
Stage 5/6 - Waterlogging map generation and georeferenced output.

Converts U-Net segmentation predictions into georeferenced waterlogging extent
and severity maps (GeoTIFF / PNG) suitable for drainage planning and
crop-damage assessment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import os

import config
import numpy as np

LOGGER = logging.getLogger("deepwaterlog.mapper")


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
