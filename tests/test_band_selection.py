"""Tests for band selection and dimensionality reduction (Stage 2)."""

import numpy as np

from core.band_selection import select_bands, select_by_correlation, select_by_pca


def test_select_none_returns_all_bands():
    cube = np.random.rand(20, 8, 8).astype(np.float32)
    out, diag = select_bands(cube, method="none")
    assert out.shape == cube.shape
    assert diag.n_output_bands == 20


def test_select_correlation_reduces_bands():
    # Highly correlated bands (duplicates)
    band = np.random.rand(1, 8, 8).astype(np.float32)
    cube = np.repeat(band, 10, axis=0)
    out, diag = select_by_correlation(cube, threshold=0.95)
    assert out.shape[0] < cube.shape[0]
    assert diag.n_output_bands == 1


def test_select_pca_reduces_to_components():
    cube = np.random.rand(20, 8, 8).astype(np.float32)
    out, diag = select_by_pca(cube, n_components=5)
    assert out.shape[0] == 5
    assert len(diag.explained_variance_ratio) == 5
    assert all(0 <= v <= 1 for v in diag.explained_variance_ratio)
    assert sum(diag.explained_variance_ratio) <= 1.0 + 1e-6


def test_select_mutual_information_with_labels():
    cube = np.random.rand(20, 8, 8).astype(np.float32)
    labels = np.zeros((8, 8), dtype=np.int64)
    labels[:4, :] = 1
    out, diag = select_bands(
        cube, method="mutual_information", labels=labels, top_k=5
    )
    assert out.shape[0] == 5
