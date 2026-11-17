"""
Generate a demonstration run for the DeepWaterLog dashboard.

Runs the full six-stage pipeline on a synthetic hyperspectral scene with a small
number of epochs, so the dashboard has metrics, a classified map and a severity
map to display without waiting for a full training schedule.

Usage
-----
    python scripts/run_demo.py
    python scripts/run_demo.py --epochs 15 --size 96
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import config
from core.pipeline import run_pipeline


def make_scene(
    n_bands: int = 40,
    size: int = 96,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a synthetic hyperspectral scene with five class regions and a
    waterlogging pattern that follows low-lying areas.

    Class ids follow config.CLASSES:
      0 waterlogged_soil, 1 healthy_crop, 2 stressed_crop,
      3 open_water, 4 non_agricultural
    """
    rng = np.random.default_rng(seed)
    cube = np.zeros((n_bands, size, size), dtype=np.float32)
    labels = np.zeros((size, size), dtype=np.int64)

    # Distinct spectral signatures per class
    signatures = {
        0: np.linspace(0.12, 0.28, n_bands),   # waterlogged soil: damp, low NIR
        1: np.linspace(0.06, 0.62, n_bands),   # healthy crop: strong NIR rise
        2: np.linspace(0.09, 0.38, n_bands),   # stressed crop: reduced NIR
        3: np.linspace(0.02, 0.07, n_bands),   # open water: very dark
        4: np.full(n_bands, 0.22),             # non-agricultural: flat
    }

    # Background: healthy crop with mild spatial variation
    cube[:] = signatures[1][:, None, None]
    labels[:] = 1

    # Non-agricultural strip along the top (road / built-up)
    cube[:, 0:8, :] = signatures[4][:, None, None]
    labels[0:8, :] = 4

    # Open water pond in the lower-right
    cube[:, size - 20:size - 6, size - 22:size - 8] = signatures[3][:, None, None]
    labels[size - 20:size - 6, size - 22:size - 8] = 3

    # Waterlogged soil in the low-lying lower-left block
    cube[:, size - 34:size - 12, 8:34] = signatures[0][:, None, None]
    labels[size - 34:size - 12, 8:34] = 0

    # Stressed crop ringing the waterlogged block (drainage gradient)
    cube[:, size - 40:size - 34, 6:38] = signatures[2][:, None, None]
    labels[size - 40:size - 34, 6:38] = 2
    cube[:, size - 34:size - 12, 4:8] = signatures[2][:, None, None]
    labels[size - 34:size - 12, 4:8] = 2
    cube[:, size - 34:size - 12, 34:38] = signatures[2][:, None, None]
    labels[size - 34:size - 12, 34:38] = 2

    # Sensor noise
    cube += rng.normal(0, 0.012, size=cube.shape).astype(np.float32)
    cube = np.clip(cube, 0.0, 1.0)
    return cube, labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a dashboard demo run.")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs per model.")
    parser.add_argument("--size", type=int, default=96, help="Scene size in pixels.")
    parser.add_argument("--bands", type=int, default=40, help="Number of spectral bands.")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size.")
    parser.add_argument("--stride", type=int, default=8, help="Patch stride.")
    args = parser.parse_args()

    # Short training schedule for the demo
    config.CNN_EPOCHS = args.epochs
    config.UNET_EPOCHS = args.epochs
    config.CNN_EARLY_STOPPING_PATIENCE = args.epochs
    config.UNET_EARLY_STOPPING_PATIENCE = args.epochs
    config.PATCH_SIZE = args.patch_size
    config.STRIDE = args.stride
    config.BAND_SELECTION_METHOD = "none"  # keep all bands for the demo

    print(f"Building synthetic scene: {args.bands} bands, {args.size}x{args.size} pixels")
    cube, labels = make_scene(n_bands=args.bands, size=args.size)

    result, _artifacts = run_pipeline(
        cube,
        labels,
        scenario_id="demo_lowland_paddy",
        patch_size=args.patch_size,
        stride=args.stride,
        save_results=True,
    )

    metrics = result.get("metrics") or {}
    unet = metrics.get("unet") or {}
    map_info = result.get("map") or {}

    print("")
    print("=" * 68)
    print("DEMO RUN COMPLETE")
    print(f"  Run ID        : {result.get('run_id')}")
    print(f"  Processing    : {result.get('processing_seconds')} s")
    print(f"  U-Net accuracy: {unet.get('accuracy')}")
    print(f"  U-Net F1      : {unet.get('f1_macro')}")
    print(f"  U-Net IoU     : {unet.get('iou_macro')}")
    print(f"  Waterlogged   : {map_info.get('waterlogged_area_ha')} ha")
    print(f"  Artefacts     : {map_info.get('artifacts')}")
    print("=" * 68)
    print("Open the dashboard: python ui/app.py  ->  http://127.0.0.1:5000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
