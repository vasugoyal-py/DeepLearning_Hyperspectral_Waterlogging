"""
Train CNN and/or U-Net models on a labelled hyperspectral dataset.

Usage
-----
    python scripts/train.py --data data/training --model both
    python scripts/train.py --data data/training --model cnn --epochs 50
    python scripts/train.py --data data/training --model unet --batch-size 16
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
from core.trainer import train_cnn, train_unet
from core.unet_model import build_unet


def _load_npz(path: Path):
    """Load a .npz with keys: cube (bands,H,W), labels (H,W)."""
    with np.load(path) as z:
        return z["cube"].astype(np.float32), z["labels"].astype(np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train CNN and/or U-Net waterlogging models."
    )
    parser.add_argument("--data", required=True, help="Path to labelled .npz file.")
    parser.add_argument(
        "--model", choices=["cnn", "unet", "both"], default="both",
        help="Which model to train.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--patch-size", type=int, default=None, help="Patch size.")
    parser.add_argument("--stride", type=int, default=None, help="Patch stride.")
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
        cube,
        label_map,
        patch_size=args.patch_size,
        stride=args.stride,
    )
    patches, patch_labels = ds.extract_patches()
    ds.normalise()
    splits = ds.split(seed=args.seed)

    def to_t(x):
        return torch.from_numpy(np.asarray(x, dtype=np.float32))

    train_patches = to_t(splits["train"][0])
    train_labels = torch.from_numpy(splits["train"][1].astype(np.int64))
    val_patches = to_t(splits["val"][0])
    val_labels = torch.from_numpy(splits["val"][1].astype(np.int64))

    common_kwargs = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )

    results = {}
    if args.model in ("cnn", "both"):
        print("\n=== Training CNN ===")
        cnn = build_cnn()
        results["cnn"] = train_cnn(
            cnn, train_patches, train_labels, val_patches, val_labels, **common_kwargs
        )
        print(f"CNN best F1: {results['cnn'].best_metric:.4f} "
              f"@epoch {results['cnn'].best_epoch}")

    if args.model in ("unet", "both"):
        print("\n=== Training U-Net ===")
        unet = build_unet()
        results["unet"] = train_unet(
            unet, train_patches, train_labels, val_patches, val_labels, **common_kwargs
        )
        print(f"U-Net best IoU: {results['unet'].best_metric:.4f} "
              f"@epoch {results['unet'].best_epoch}")

    print("\nTraining complete. Checkpoints saved to data/models/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
