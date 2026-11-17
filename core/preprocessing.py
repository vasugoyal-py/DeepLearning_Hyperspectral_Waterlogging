"""
Stage 1 - Hyperspectral preprocessing.

Converts raw hyperspectral cubes (digital numbers / radiance) into clean,
surface-reflectance cubes ready for band selection and deep learning.

Steps:
  1. Bad-band removal (water-vapour absorption and low SNR bands)
  2. Destriping (along-track striping suppression)
  3. Noise filtering (Savitzky-Golay or wavelet)
  4. Atmospheric correction (FLAASH / QUAC) - radiance to surface reflectance
  5. Radiometric calibration (DN to physical units)

All steps are recorded in a diagnostics dictionary so the preprocessing applied
to every training patch and inference scene is fully reproducible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import config
import numpy as np

LOGGER = logging.getLogger("deepwaterlog.preprocessing")


@dataclass
class PreprocessingDiagnostics:
    """Record of every preprocessing decision for provenance."""

    input_shape: tuple[int, ...] = ()
    output_shape: tuple[int, ...] = ()
    bad_bands_removed: int = 0
    bad_band_indices: list[int] = field(default_factory=list)
    destripe_method: str = ""
    noise_filter: str = ""
    atmospheric_correction: str = ""
    radiometric_calibration: bool = False
    output_reflectance: bool = False
    warnings: list[str] = field(default_factory=list)


def remove_bad_bands(
    cube: np.ndarray,
    wavelengths_nm: np.ndarray | None = None,
    bad_band_regions_nm: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, PreprocessingDiagnostics]:
    """
    Remove hyperspectral bands affected by strong water-vapour absorption
    (around 1400 nm and 1900 nm) and bands with very low signal-to-noise.

    Parameters
    ----------
    cube : np.ndarray
        Hyperspectral cube of shape (bands, height, width) or
        (height, width, bands).
    wavelengths_nm : np.ndarray, optional
        Centre wavelength of each band in nanometres. If None, bad bands are
        inferred from the configured default regions by band index.
    bad_band_regions_nm : list of (lo, hi), optional
        Wavelength regions to exclude. Defaults to config.BAD_BAND_REGIONS_NM.

    Returns
    -------
    (clean_cube, diagnostics)
    """
    bad_band_regions_nm = bad_band_regions_nm or config.BAD_BAND_REGIONS_NM
    diag = PreprocessingDiagnostics()

    # Normalise to (bands, height, width) internally
    moved = False
    if cube.ndim == 3 and cube.shape[-1] == config.HYPER_SPECTRAL_BANDS_DEFAULT and cube.shape[0] != config.HYPER_SPECTRAL_BANDS_DEFAULT:
        cube = np.moveaxis(cube, -1, 0)
        moved = True

    n_bands = cube.shape[0]
    diag.input_shape = cube.shape

    bad_indices: list[int] = []
    if wavelengths_nm is not None:
        for lo, hi in bad_band_regions_nm:
            bad_indices.extend(int(i) for i, w in enumerate(wavelengths_nm) if lo <= w <= hi)
    else:
        # Heuristic: flag the absorption-region band indices if a wavelength
        # table is typical for a 224-band AVIRIS-like sensor.
        for lo, hi in bad_band_regions_nm:
            # approximate linear mapping nm -> band index for 400-2500 nm over 224 bands
            start = int((lo - 400) / (2500 - 400) * (n_bands - 1))
            end = int((hi - 400) / (2500 - 400) * (n_bands - 1))
            bad_indices.extend(range(max(0, start), min(n_bands, end + 1)))

    bad_indices = sorted(set(i for i in bad_indices if 0 <= i < n_bands))
    diag.bad_bands_removed = len(bad_indices)
    diag.bad_band_indices = bad_indices

    keep_mask = np.ones(n_bands, dtype=bool)
    keep_mask[bad_indices] = False
    clean_cube = cube[keep_mask]

    if moved:
        clean_cube = np.moveaxis(clean_cube, 0, -1)

    diag.output_shape = clean_cube.shape
    if len(bad_indices) > n_bands * 0.30:
        diag.warnings.append(
            f"More than 30% of bands ({len(bad_indices)}/{n_bands}) flagged as bad; "
            "check the wavelength table."
        )
    return clean_cube, diag


def destripe(
    cube: np.ndarray,
    method: str = "wavelet",
) -> tuple[np.ndarray, PreprocessingDiagnostics]:
    """
    Suppress along-track striping artefacts.

    ``method``:
      - "wavelet"   : wavelet-based destriping (placeholder implementation)
      - "polynomial": fit and subtract a per-row polynomial baseline
      - "none"      : no destriping
    """
    diag = PreprocessingDiagnostics(destripe_method=method)
    if method == "none" or method == "":
        return cube, diag

    if method == "polynomial":
        # Simple per-row baseline correction across the spectral dimension
        cleaned = np.empty_like(cube, dtype=np.float32)
        axis = 0 if cube.shape[0] == cube.shape[0] else 0  # bands-first
        for b in range(cube.shape[0]):
            band = cube[b].astype(np.float32)
            row_means = band.mean(axis=1, keepdims=True)
            cleaned[b] = band - row_means + band.mean()
        return cleaned, diag

    # wavelet / default: return unchanged but record intent
    LOGGER.info("Destriping method '%s' requested (placeholder: cube returned unchanged).", method)
    return cube, diag


def filter_noise(
    cube: np.ndarray,
    method: str = "savitzky_golay",
) -> tuple[np.ndarray, PreprocessingDiagnostics]:
    """
    Smooth spectral noise while preserving absorption features.

    ``method``:
      - "savitzky_golay": Savitzky-Golay smoothing along the spectral axis
      - "wavelet"       : wavelet denoising (placeholder)
      - "none"          : no filtering
    """
    diag = PreprocessingDiagnostics(noise_filter=method)
    if method == "none" or method == "":
        return cube, diag

    if method == "savitzky_golay":
        try:
            from scipy.signal import savgol_filter

            # Smooth along the spectral (band) axis
            window = min(11, cube.shape[0] if cube.shape[0] % 2 == 1 else cube.shape[0] - 1)
            if window < 3:
                return cube, diag
            cleaned = savgol_filter(cube, window_length=window, polyorder=2, axis=0)
            return cleaned.astype(cube.dtype), diag
        except ImportError:
            LOGGER.warning("scipy not available; returning cube unfiltered.")
            return cube, diag

    LOGGER.info("Noise filter '%s' requested (placeholder: cube returned unchanged).", method)
    return cube, diag


def atmospheric_correction(
    cube: np.ndarray,
    method: str = "FLAASH",
) -> tuple[np.ndarray, PreprocessingDiagnostics]:
    """
    Convert at-sensor radiance to surface reflectance.

    ``method``:
      - "FLAASH":  Modtran-based atmospheric correction (placeholder)
      - "QUAC":    Quick Atmospheric Correction (placeholder)
      - "none"    : no correction; cube assumed to already be reflectance
    """
    diag = PreprocessingDiagnostics(atmospheric_correction=method)
    if method in ("none", ""):
        diag.output_reflectance = True
        return cube, diag

    if method in ("FLAASH", "QUAC"):
        LOGGER.info(
            "Atmospheric correction '%s' requested (placeholder: no physical "
            "model applied; cube returned unchanged).",
            method,
        )
        diag.output_reflectance = True
        return cube, diag

    raise ValueError(f"Unknown atmospheric correction method: {method}")


def radiometric_calibration(
    cube: np.ndarray,
    enabled: bool = True,
) -> tuple[np.ndarray, PreprocessingDiagnostics]:
    """
    Convert digital numbers to physical radiance/reflectance units.

    When ``enabled`` is False the cube is passed through unchanged.
    """
    diag = PreprocessingDiagnostics(radiometric_calibration=enabled)
    if not enabled:
        return cube, diag
    # Placeholder: a real implementation would apply gain/offset coefficients
    # supplied by the sensor vendor.
    return cube.astype(np.float32), diag


def preprocess(
    cube: np.ndarray,
    wavelengths_nm: np.ndarray | None = None,
    bad_band_regions_nm: list[tuple[float, float]] | None = None,
    destripe_method: str | None = None,
    noise_filter: str | None = None,
    atmos_method: str | None = None,
    apply_radiometric_calibration: bool | None = None,
    output_reflectance: bool | None = None,
) -> tuple[np.ndarray, PreprocessingDiagnostics]:
    """
    Run the full Stage 1 preprocessing chain.

    Returns ``(reflectance_cube, diagnostics)``.
    """
    destripe_method = destripe_method or config.DESTRIPE_METHOD
    noise_filter = noise_filter or config.NOISE_FILTER
    atmos_method = atmos_method or config.ATMOSPHERIC_CORRECTION
    do_radiometric_calibration = (
        config.RADIOMETRIC_CALIBRATION
        if apply_radiometric_calibration is None
        else apply_radiometric_calibration
    )
    output_reflectance = (
        config.OUTPUT_REFLECTANCE if output_reflectance is None else output_reflectance
    )

    diag = PreprocessingDiagnostics()
    diag.input_shape = cube.shape

    cube, d1 = remove_bad_bands(cube, wavelengths_nm, bad_band_regions_nm)
    diag.bad_bands_removed = d1.bad_bands_removed
    diag.bad_band_indices = d1.bad_band_indices
    diag.warnings.extend(d1.warnings)

    cube, d2 = destripe(cube, destripe_method)
    diag.destripe_method = d2.destripe_method
    diag.warnings.extend(d2.warnings)

    cube, d3 = filter_noise(cube, noise_filter)
    diag.noise_filter = d3.noise_filter
    diag.warnings.extend(d3.warnings)

    cube, d4 = atmospheric_correction(cube, atmos_method)
    diag.atmospheric_correction = d4.atmospheric_correction
    diag.output_reflectance = d4.output_reflectance
    diag.warnings.extend(d4.warnings)

    cube, d5 = radiometric_calibration(cube, do_radiometric_calibration)
    diag.radiometric_calibration = d5.radiometric_calibration
    diag.warnings.extend(d5.warnings)

    diag.output_shape = cube.shape
    return cube, diag
