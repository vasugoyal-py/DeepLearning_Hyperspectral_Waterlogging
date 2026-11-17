"""
Evaluate trained CNN and/or U-Net models on a test hyperspectral dataset.

Usage
-----
    python scripts/evaluate.py --data data/testing --model both
    python scripts/evaluate.py --data data/testing --model unet --checkpoint data/models/unet_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import config
from core.cnn_model import build_cnn
from core.dataset import HyperspectralWaterloggingDataset
from core.evaluator import (
    evaluate_classification,
    evaluate_segmentation,
    format_metrics_report,
)
from core.unet_model import build_unet


def _load_npz(path: Path):
    with np.load(path) as z:
        return z["cube"].astype(np.float32), z["labels"].astype(np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate CNN and/or U-Net waterlogging models."
    )
    parser.add_argument("--data", required=True, help="Path to labelled .npz file.")
    parser.add_argument(
        "--model", choices=["cnn", "unet", "both"], default="both",
        help="Which model to evaluate.",
    )
    parser.add_argument(
        "--cnn-checkpoint", default=None,
        help="Path to CNN checkpoint (default: data/models/cnn_best.pt).",
    )
    parser.add_argument(
        "--unet-checkpoint", default=None,
        help="Path to U-Net checkpoint (default: data/models/unet_best.pt).",
    )
    parser.add_argument("--device", default=None, help="Torch device (cuda/cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data file not found: {data_path}", file=sys.stderr)
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading data from: {data_path}")

    cube, label_map = _load_npz(data_path)

    ds = HyperspectralWaterloggingDataset(
        cube, label_map,
        patch_size=None, stride=None,
    )
    patches, patch_labels = ds.extract_patches()
    ds.normalise()
    splits = ds.split(seed=args.seed)

    def to_t(x):
        return torch.from_numpy(np.asarray(x, dtype=np.float32))

    test_patches = to_t(splits["test"][0])
    test_labels = torch.from_numpy(splits["test"][1].astype(np.int64))

    if args.model in ("cnn", "both"):
        ckpt = args.cnn_checkpoint or str(
            Path(config.DATA_DIR) / "models" / "cnn_best.pt"
        )
        print(f"\n=== Evaluating CNN ({ckpt}) ===")
        cnn = build_cnn()
        state = torch.load(ckpt, map_location=device)
        cnn.load_state_dict(state["model_state_dict"])
        cnn.to(device)
        cnn.eval()
        with torch.no_grad():
            logits = cnn(test_patches.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
        metrics = evaluate_classification(
            preds, test_labels.numpy(),
            num_classes=config.CNN_NUM_CLASSES,
            class_names=config.CLASSES,
        )
        print(format_metrics_report(metrics, "CNN", config.CLASSES))

    if args.model in ("unet", "both"):
        ckpt = args.unet_checkpoint or str(
            Path(config.DATA_DIR) / "models" / "unet_best.pt"
        )
        print(f"\n=== Evaluating U-Net ({ckpt}) ===")
        unet = build_unet()
        state = torch.load(ckpt, map_location=device)
        unet.load_state_dict(state["model_state_dict"])
        unet.to(device)
        unet.eval()
        with torch.no_grad():
            logits = unet(test_patches.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
        metrics = evaluate_segmentation(
            preds, test_labels.numpy(),
            num_classes=config.UNET_NUM_CLASSES,
            class_names=config.CLASSES,
        )
        print(format_metrics_report(metrics, "U-Net", config.CLASSES))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
