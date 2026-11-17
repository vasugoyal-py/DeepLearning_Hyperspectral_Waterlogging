"""
Command-line health check for DeepWaterLog.

Runs the six-stage pipeline (preprocessing, band selection, dataset
preparation, CNN/U-Net training, evaluation, mapping) on a small synthetic
hyperspectral scene so the code path can be exercised offline without Earth
Engine or labelled field data.

Examples
--------
    python test_pipeline.py
    python test_pipeline.py --no-cnn
    python test_pipeline.py --no-unet
    python test_pipeline.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import config
import numpy as np
import torch

from core.pipeline import PipelineError, run_pipeline


def _make_synthetic_scene(
    n_bands: int = 30,
    height: int = 64,
    width: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a synthetic hyperspectral cube and label map for offline testing.

    Five classes are simulated with distinct spectral means so a CNN/U-Net can
    in principle learn to separate them:
      0 - waterlogged_soil   (high SWIR, low NIR)
      1 - healthy_crop       (high NIR, high red-edge)
      2 - stressed_crop      (moderate NIR, low red-edge)
      3 - open_water         (low visible/NIR, high SWIR absorption)
      4 - non_agricultural   (flat mid spectrum)
    """
    rng = np.random.default_rng(seed)
    cube = np.zeros((n_bands, height, width), dtype=np.float32)
    label_map = np.zeros((height, width), dtype=np.int64)

    signatures = {
        0: np.linspace(0.10, 0.30, n_bands),
        1: np.linspace(0.05, 0.55, n_bands),
        2: np.linspace(0.08, 0.35, n_bands),
        3: np.linspace(0.02, 0.10, n_bands),
        4: np.full(n_bands, 0.20),
    }

    block_h = height // 5
    block_w = width // 5
    for cls in range(5):
        r0, r1 = cls * block_h, (cls + 1) * block_h
        c0, c1 = cls * block_w, (cls + 1) * block_w
        sig = signatures[cls][:, None, None]
        cube[:, r0:r1, c0:c1] = sig + rng.normal(0, 0.02, size=(n_bands, 1, 1)).astype(np.float32)
        label_map[r0:r1, c0:c1] = cls

    cy, cx = height // 2, width // 2
    cube[:, cy - 4:cy + 4, cx - 4:cx + 4] = signatures[0][:, None, None]
    label_map[cy - 4:cy + 4, cx - 4:cx + 4] = 0

    return cube, label_map


def _print_summary(result: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("DeepWaterLog — pipeline health check")
    print(f"Run ID     : {result.get('run_id')}")
    print(f"Scenario   : {result.get('scenario_id')}")
    print(f"Theme      : {config.THEME}")
    print(f"Processing : {result.get('processing_seconds')} s")
    print("-" * 72)
    pp = result.get("preprocessing") or {}
    print(f"Preprocessing : bad_bands_removed={pp.get('bad_bands_removed')}, "
          f"output_shape={pp.get('output_shape')}")
    bs = result.get("band_selection") or {}
    print(f"Band selection: method={bs.get('method')}, "
          f"bands {bs.get('n_input_bands')} -> {bs.get('n_output_bands')}")
    ds = result.get("dataset") or {}
    print(f"Dataset       : {ds.get('n_patches')} patches "
          f"(train={ds.get('train')}, val={ds.get('val')}, test={ds.get('test')})")
    print(f"Class counts  : {ds.get('class_counts')}")
    if result.get("cnn"):
        c = result["cnn"]
        print(f"CNN           : best_f1={c.get('best_metric'):.4f} "
              f"@epoch {c.get('best_epoch')}, ckpt={c.get('checkpoint')}")
    if result.get("unet"):
        u = result["unet"]
        print(f"U-Net         : best_iou={u.get('best_metric'):.4f} "
              f"@epoch {u.get('best_epoch')}, ckpt={u.get('checkpoint')}")
    m = result.get("metrics") or {}
    if m.get("cnn"):
        print(f"CNN test      : acc={m['cnn']['accuracy']:.4f} "
              f"f1={m['cnn']['f1_macro']:.4f}")
    if m.get("unet"):
        print(f"U-Net test    : acc={m['unet']['accuracy']:.4f} "
              f"f1={m['unet']['f1_macro']:.4f} iou={m['unet']['iou_macro']:.4f}")
    mp = result.get("map") or {}
    if mp.get("waterlogged_area_ha") is not None:
        print(f"Map           : waterlogged={mp.get('waterlogged_area_ha'):.2f} ha "
              f"of {mp.get('total_area_ha'):.2f} ha "
              f"({mp.get('waterlogged_fraction'):.2%})")
    warnings = result.get("warnings") or []
    if warnings:
        print("-" * 72)
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the DeepWaterLog pipeline health check on synthetic data."
    )
    parser.add_argument("--no-cnn", action="store_true", help="Skip CNN training.")
    parser.add_argument("--no-unet", action="store_true", help="Skip U-Net training.")
    parser.add_argument("--bands", type=int, default=30, help="Number of synthetic bands.")
    parser.add_argument("--size", type=int, default=64, help="Scene size (pixels).")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size (pixels).")
    parser.add_argument("--stride", type=int, default=8, help="Patch stride (pixels).")
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    parser.add_argument("--device", default=None, help="Torch device (cuda/cpu).")
    args = parser.parse_args()

    cube, label_map = _make_synthetic_scene(
        n_bands=args.bands, height=args.size, width=args.size
    )

    try:
        result, _artifacts = run_pipeline(
            cube,
            label_map,
            scenario_id="synthetic_test",
            train_cnn_model=not args.no_cnn,
            train_unet_model=not args.no_unet,
            device=args.device,
            patch_size=args.patch_size,
            stride=args.stride,
        )
    except PipelineError as error:
        print(f"\nPIPELINE FAILED [{error.code}]: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001
        print(f"\nPIPELINE FAILED: {error}", file=sys.stderr)
        return 3

    _print_summary(result)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    print("\n>>> PIPELINE COMPLETED <<<")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
