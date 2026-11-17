"""Tests for training data preparation (Stage 3)."""

import numpy as np

from core.dataset import HyperspectralWaterloggingDataset


def test_extract_patches_produces_patches_and_labels():
    cube = np.random.rand(10, 32, 32).astype(np.float32)
    labels = np.zeros((32, 32), dtype=np.int64)
    labels[:16, :] = 1
    labels[16:, :] = 2

    ds = HyperspectralWaterloggingDataset(cube, labels, patch_size=8, stride=8)
    patches, patch_labels, label_patches = ds.extract_patches()

    assert patches.ndim == 4
    assert patches.shape[1] == 10  # bands
    assert patches.shape[2] == 8
    assert patches.shape[3] == 8
    assert patch_labels.shape[0] == patches.shape[0]
    assert label_patches.shape == (patches.shape[0], 8, 8)
    assert set(patch_labels.tolist()) <= {0, 1, 2}


def test_normalise_changes_values():
    cube = np.random.rand(5, 16, 16).astype(np.float32) * 100
    labels = np.zeros((16, 16), dtype=np.int64)
    ds = HyperspectralWaterloggingDataset(cube, labels, patch_size=8, stride=8)
    ds.extract_patches()
    before = ds.patches.copy()
    ds.normalise(method="z_score")
    assert not np.allclose(before, ds.patches)


def test_split_proportions():
    cube = np.random.rand(5, 32, 32).astype(np.float32)
    labels = np.zeros((32, 32), dtype=np.int64)
    labels[:16, :] = 1
    labels[16:, :] = 2
    ds = HyperspectralWaterloggingDataset(cube, labels, patch_size=8, stride=8)
    ds.extract_patches()
    splits = ds.split()
    total = ds.diagnostics.n_patches
    assert abs(
        (ds.diagnostics.train_count + ds.diagnostics.val_count + ds.diagnostics.test_count)
        - total
    ) < 3  # rounding per class
