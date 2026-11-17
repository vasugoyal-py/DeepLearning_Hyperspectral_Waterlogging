"""Tests for hyperspectral preprocessing (Stage 1)."""

import numpy as np

from core.preprocessing import preprocess, remove_bad_bands


def test_remove_bad_bands_reduces_band_count():
    cube = np.random.rand(10, 8, 8).astype(np.float32)
    clean, diag = remove_bad_bands(
        cube,
        wavelengths_nm=np.linspace(400, 2500, 10),
        bad_band_regions_nm=[(1340, 1450), (1780, 2000)],
    )
    assert clean.shape[0] < cube.shape[0]
    assert diag.bad_bands_removed > 0
    assert diag.output_shape == clean.shape


def test_preprocess_runs_full_chain():
    cube = np.random.rand(20, 16, 16).astype(np.float32)
    out, diag = preprocess(cube)
    assert out.shape[0] <= cube.shape[0]
    assert diag.output_shape == out.shape
    assert isinstance(diag.warnings, list)


def test_preprocess_bad_band_warning_when_many_removed():
    # 9 of 10 bands in absorption region -> warning expected
    cube = np.random.rand(10, 8, 8).astype(np.float32)
    _, diag = remove_bad_bands(
        cube,
        wavelengths_nm=np.linspace(1300, 2000, 10),
        bad_band_regions_nm=[(1340, 2000)],
    )
    assert len(diag.warnings) > 0
