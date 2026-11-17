"""
Stage 2 - Band selection and dimensionality reduction.

Selects waterlogging-sensitive spectral bands and reduces the dimensionality of
the hyperspectral cube while retaining diagnostic features in the visible,
red-edge, NIR and SWIR regions.

Methods:
  - correlation filtering (remove highly redundant bands)
  - mutual-information ranking (keep the most class-informative bands)
  - principal component analysis (PCA)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import config
import numpy as np

LOGGER = logging.getLogger("deepwaterlog.band_selection")


@dataclass
class BandSelectionDiagnostics:
    method: str = ""
    n_input_bands: int = 0
    n_output_bands: int = 0
    removed_band_indices: list[int] = field(default_factory=list)
    selected_band_indices: list[int] = field(default_factory=list)
    explained_variance_ratio: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def select_by_correlation(
    cube: np.ndarray,
    threshold: float | None = None,
) -> tuple[np.ndarray, BandSelectionDiagnostics]:
    """
    Remove bands that are highly correlated (|r| > threshold) with an earlier
    band, keeping the first occurrence of each correlated group.

    ``cube`` is expected in (bands, height, width) layout.
    """
    threshold = config.CORRELATION_THRESHOLD if threshold is None else threshold
    diag = BandSelectionDiagnostics(method="correlation")
    n_bands = cube.shape[0]
    diag.n_input_bands = n_bands

    # Flatten spatial dimensions and compute band-band correlation
    flat = cube.reshape(n_bands, -1).astype(np.float64)
    # Standardise each band
    flat = (flat - flat.mean(axis=1, keepdims=True)) / (
        flat.std(axis=1, keepdims=True) + 1e-8
    )
    corr = np.corrcoef(flat)
    corr = np.nan_to_num(corr, nan=0.0)

    keep = np.ones(n_bands, dtype=bool)
    removed: list[int] = []
    for i in range(n_bands):
        if not keep[i]:
            continue
        for j in range(i + 1, n_bands):
            if keep[j] and abs(corr[i, j]) > threshold:
                keep[j] = False
                removed.append(j)

    selected = np.where(keep)[0]
    diag.n_output_bands = int(selected.size)
    diag.removed_band_indices = removed
    diag.selected_band_indices = [int(i) for i in selected]

    if diag.n_output_bands < n_bands * 0.30:
        diag.warnings.append(
            f"Correlation filtering removed {n_bands - diag.n_output_bands} of "
            f"{n_bands} bands; consider relaxing the threshold."
        )

    return cube[selected], diag


def select_by_mutual_information(
    cube: np.ndarray,
    labels: np.ndarray | None = None,
    top_k: int | None = None,
) -> tuple[np.ndarray, BandSelectionDiagnostics]:
    """
    Rank bands by mutual information with the class labels and keep the top-K.

    If no labels are supplied, falls back to variance ranking (high-variance
    bands are more likely to carry information).

    ``cube`` is (bands, height, width); ``labels`` is (height, width) with
    integer class ids.
    """
    top_k = config.MI_TOP_K_BANDS if top_k is None else top_k
    diag = BandSelectionDiagnostics(method="mutual_information")
    n_bands = cube.shape[0]
    diag.n_input_bands = n_bands

    flat = cube.reshape(n_bands, -1).astype(np.float64)

    if labels is not None:
        label_flat = labels.reshape(-1)
        # Simple proxy: between-class variance ratio per band (ANOVA-like)
        scores = np.zeros(n_bands)
        unique_labels = np.unique(label_flat)
        overall_mean = flat.mean(axis=1)
        for b in range(n_bands):
            band = flat[b]
            between = 0.0
            total = 0.0
            for cls in unique_labels:
                mask = label_flat == cls
                if mask.sum() == 0:
                    continue
                cls_mean = band[mask].mean()
                between += mask.sum() * (cls_mean - overall_mean[b]) ** 2
                total += ((band[mask] - cls_mean) ** 2).sum()
            scores[b] = between / (total + 1e-8)
    else:
        scores = flat.var(axis=1)

    top_k = min(top_k, n_bands)
    selected = np.argsort(scores)[::-1][:top_k]
    selected = np.sort(selected)

    diag.n_output_bands = int(selected.size)
    diag.selected_band_indices = [int(i) for i in selected]
    diag.removed_band_indices = [int(i) for i in range(n_bands) if i not in selected]

    return cube[selected], diag


def select_by_pca(
    cube: np.ndarray,
    n_components: int | None = None,
) -> tuple[np.ndarray, BandSelectionDiagnostics]:
    """
    Project bands onto the first ``n_components`` principal components.

    Returns the PCA-transformed cube (components, height, width).
    """
    n_components = config.PCA_COMPONENTS if n_components is None else n_components
    diag = BandSelectionDiagnostics(method="pca")
    n_bands = cube.shape[0]
    diag.n_input_bands = n_bands

    flat = cube.reshape(n_bands, -1).astype(np.float64)
    flat = (flat - flat.mean(axis=1, keepdims=True)) / (
        flat.std(axis=1, keepdims=True) + 1e-8
    )

    # Covariance and eigendecomposition
    cov = np.cov(flat)
    cov = np.nan_to_num(cov, nan=0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    n_components = min(n_components, n_bands)
    components = eigenvectors[:, :n_components]
    projected = components.T @ flat  # (n_components, pixels)
    explained = eigenvalues[:n_components] / (eigenvalues.sum() + 1e-8)

    out_cube = projected.reshape(n_components, *cube.shape[1:])

    diag.n_output_bands = n_components
    diag.explained_variance_ratio = [float(v) for v in explained]
    diag.warnings.append(
        f"PCA reduced {n_bands} bands to {n_components} components "
        f"(cumulative variance {explained.sum():.2%})."
    )
    return out_cube, diag


def select_bands(
    cube: np.ndarray,
    method: str | None = None,
    labels: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, BandSelectionDiagnostics]:
    """
    Run band selection / dimensionality reduction.

    ``method``: "correlation", "mutual_information", "pca", or "none".
    """
    method = (method or config.BAND_SELECTION_METHOD).lower()
    diag = BandSelectionDiagnostics(method=method)
    diag.n_input_bands = cube.shape[0]

    if method == "none" or method == "":
        diag.n_output_bands = cube.shape[0]
        diag.selected_band_indices = list(range(cube.shape[0]))
        return cube, diag

    if method == "correlation":
        return select_by_correlation(cube, **kwargs)
    if method == "mutual_information":
        return select_by_mutual_information(cube, labels=labels, **kwargs)
    if method == "pca":
        return select_by_pca(cube, **kwargs)

    raise ValueError(f"Unknown band selection method: {method}")
