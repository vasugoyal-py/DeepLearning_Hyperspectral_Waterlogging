"""
Stage 3 - Training data preparation.

Extracts labelled hyperspectral patches from preprocessed, band-selected cubes,
normalises them, applies data augmentation, and splits into train / validation /
test sets.

Classes (from config.CLASSES):
  0 - waterlogged_soil
  1 - healthy_crop
  2 - stressed_crop
  3 - open_water
  4 - non_agricultural
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import config
import numpy as np

LOGGER = logging.getLogger("deepwaterlog.dataset")


@dataclass
class DatasetDiagnostics:
    n_patches: int = 0
    patch_size: int = 0
    stride: int = 0
    n_classes: int = 0
    class_counts: dict[int, int] = field(default_factory=dict)
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    no_data_fraction_excluded: float = 0.0
    warnings: list[str] = field(default_factory=list)


class HyperspectralWaterloggingDataset:
    """
    In-memory dataset of labelled hyperspectral patches.

    Parameters
    ----------
    cube : np.ndarray
        Preprocessed, band-selected cube in (bands, height, width) layout.
    label_map : np.ndarray
        Integer class map of shape (height, width).
    patch_size : int
        Side length of square patches (pixels).
    stride : int
        Extraction stride; overlap = patch_size - stride.
    classes : list[str]
        Class names; index == class id.
    """

    def __init__(
        self,
        cube: np.ndarray,
        label_map: np.ndarray,
        patch_size: int | None = None,
        stride: int | None = None,
        classes: list[str] | None = None,
    ):
        self.cube = np.asarray(cube, dtype=np.float32)  # (bands, H, W)
        self.label_map = np.asarray(label_map, dtype=np.int64)  # (H, W)
        self.patch_size = patch_size or config.PATCH_SIZE
        self.stride = stride or config.STRIDE
        self.classes = classes or config.CLASSES
        self.n_classes = len(self.classes)
        self.diagnostics = DatasetDiagnostics(
            n_classes=self.n_classes,
            patch_size=self.patch_size,
            stride=self.stride,
        )
        self.patches: np.ndarray | None = None
        self.patch_labels: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    def extract_patches(
        self,
        min_class_frequency: float | None = None,
        max_no_data_fraction: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Slide a window over the cube and label each patch by the majority class
        in its label-map window.

        Returns (patches, patch_labels) where patches is
        (n_patches, bands, patch_size, patch_size).
        """
        min_class_frequency = (
            config.QC_MIN_CLASS_FREQUENCY if min_class_frequency is None else min_class_frequency
        )
        max_no_data_fraction = (
            config.QC_MAX_NO_DATA_FRACTION if max_no_data_fraction is None else max_no_data_fraction
        )

        bands, height, width = self.cube.shape
        ps = self.patch_size
        st = self.stride

        patch_list: list[np.ndarray] = []
        label_list: list[int] = []
        label_patch_list: list[np.ndarray] = []
        excluded_no_data = 0

        for top in range(0, height - ps + 1, st):
            for left in range(0, width - ps + 1, st):
                patch = self.cube[:, top:top + ps, left:left + ps]
                labels_window = self.label_map[top:top + ps, left:left + ps]

                # Exclude patches that are mostly NaN / no-data
                if np.isnan(patch).any():
                    no_data_frac = float(np.isnan(patch).mean())
                    if no_data_frac > max_no_data_fraction:
                        excluded_no_data += 1
                        continue
                    patch = np.nan_to_num(patch, nan=0.0)

                # Majority-class label for the patch (CNN target)
                values, counts = np.unique(labels_window, return_counts=True)
                majority_label = int(values[np.argmax(counts)])
                majority_frac = float(counts.max()) / float(counts.sum())

                if majority_frac < min_class_frequency:
                    continue

                patch_list.append(patch)
                label_list.append(majority_label)
                label_patch_list.append(labels_window.astype(np.int64))

        if not patch_list:
            raise ValueError(
                "No patches were extracted. Check patch_size, stride, label map "
                "coverage, and QC thresholds."
            )

        patches = np.stack(patch_list).astype(np.float32)
        patch_labels = np.asarray(label_list, dtype=np.int64)
        label_patches = np.stack(label_patch_list).astype(np.int64)

        self.patches = patches
        self.patch_labels = patch_labels
        self.label_patches = label_patches

        # Diagnostics
        self.diagnostics.n_patches = int(patches.shape[0])
        self.diagnostics.class_counts = {
            int(c): int((patch_labels == c).sum()) for c in range(self.n_classes)
        }
        self.diagnostics.no_data_fraction_excluded = float(excluded_no_data)

        total = self.diagnostics.n_patches
        for cls in range(self.n_classes):
            frac = self.diagnostics.class_counts.get(cls, 0) / total
            if frac < min_class_frequency:
                self.diagnostics.warnings.append(
                    f"Class {self.classes[cls]} ({cls}) has only "
                    f"{self.diagnostics.class_counts.get(cls, 0)} patches "
                    f"({frac:.2%}); consider rebalancing."
                )

        LOGGER.info(
            "Extracted %d patches (%d bands, %dx%d) across %d classes; "
            "%d excluded for excess no-data.",
            total, bands, ps, ps, self.n_classes, excluded_no_data,
        )
        return patches, patch_labels, label_patches

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------
    def normalise(self, method: str | None = None) -> None:
        """Normalise patches in place (min_max or z_score per band)."""
        method = method or config.NORMALIZATION
        if self.patches is None:
            raise RuntimeError("Extract patches before normalising.")
        if method == "min_max":
            mn = self.patches.min(axis=(0, 2, 3), keepdims=True)
            mx = self.patches.max(axis=(0, 2, 3), keepdims=True)
            self.patches = (self.patches - mn) / (mx - mn + 1e-8)
        elif method == "z_score":
            mu = self.patches.mean(axis=(0, 2, 3), keepdims=True)
            sd = self.patches.std(axis=(0, 2, 3), keepdims=True)
            self.patches = (self.patches - mu) / (sd + 1e-8)
        elif method == "none":
            pass
        else:
            raise ValueError(f"Unknown normalisation method: {method}")
        LOGGER.info("Normalised patches using method '%s'.", method)

    # ------------------------------------------------------------------
    # Train / validation / test split
    # ------------------------------------------------------------------
    def split(
        self,
        train_ratio: float | None = None,
        val_ratio: float | None = None,
        test_ratio: float | None = None,
        seed: int = 42,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Stratified split into train / validation / test.

        Returns dict with keys "train", "val", "test", each a
        (patches, patch_labels, label_patches) tuple.
        """
        train_ratio = config.TRAIN_RATIO if train_ratio is None else train_ratio
        val_ratio = config.VAL_RATIO if val_ratio is None else val_ratio
        test_ratio = config.TEST_RATIO if test_ratio is None else test_ratio

        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        if self.patches is None or self.patch_labels is None or self.label_patches is None:
            raise RuntimeError("Extract and normalise patches before splitting.")

        rng = np.random.default_rng(seed)
        n = self.patches.shape[0]
        indices = np.arange(n)

        # Stratified split
        splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        for cls in range(self.n_classes):
            cls_idx = indices[self.patch_labels == cls]
            rng.shuffle(cls_idx)
            n_cls = len(cls_idx)
            n_train = int(round(n_cls * train_ratio))
            n_val = int(round(n_cls * val_ratio))
            splits["train"].extend(cls_idx[:n_train].tolist())
            splits["val"].extend(cls_idx[n_train:n_train + n_val].tolist())
            splits["test"].extend(cls_idx[n_train + n_val:].tolist())

        rng.shuffle(splits["train"])
        rng.shuffle(splits["val"])
        rng.shuffle(splits["test"])

        out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for key, idx in splits.items():
            idx = np.asarray(idx, dtype=np.int64)
            out[key] = (
                self.patches[idx],
                self.patch_labels[idx],
                self.label_patches[idx],
            )

        self.diagnostics.train_count = len(out["train"][0])
        self.diagnostics.val_count = len(out["val"][0])
        self.diagnostics.test_count = len(out["test"][0])

        LOGGER.info(
            "Split: train=%d val=%d test=%d",
            self.diagnostics.train_count,
            self.diagnostics.val_count,
            self.diagnostics.test_count,
        )
        return out
