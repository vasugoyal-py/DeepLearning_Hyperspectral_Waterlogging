"""
Shared end-to-end orchestration pipeline for DeepWaterLog.

Six stages:
  1. Preprocessing        (core/preprocessing.py)
  2. Band selection       (core/band_selection.py)
  3. Dataset preparation  (core/dataset.py)
  4. Deep learning        (core/cnn_model.py, core/unet_model.py, core/trainer.py)
  5. Evaluation           (core/evaluator.py)
  6. Mapping / decision support (core/mapper.py, ui/app.py)

Kept outside the Flask/UI layer so the same code path is used by the CLI,
scripts, and web UI — ensuring reported numbers are reproducible.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import config
import numpy as np
import torch

from core import band_selection, dataset as dataset_module, evaluator, mapper, preprocessing
from core.cnn_model import build_cnn
from core.trainer import train_cnn, train_unet
from core.unet_model import build_unet

LOGGER = logging.getLogger("deepwaterlog.pipeline")


@dataclass
class PipelineArtifacts:
    """Objects retained across stages for inspection and map generation."""

    preprocessed_cube: np.ndarray | None = None
    band_selected_cube: np.ndarray | None = None
    preprocessing_diagnostics: dict[str, Any] = field(default_factory=dict)
    band_selection_diagnostics: dict[str, Any] = field(default_factory=dict)
    dataset_diagnostics: dict[str, Any] = field(default_factory=dict)
    cnn_result: Any = None
    unet_result: Any = None
    cnn_metrics: Any = None
    unet_metrics: Any = None
    map_result: Any = None
    run_id: str = ""
    warnings: list[str] = field(default_factory=list)


class PipelineError(RuntimeError):
    """A user-facing pipeline failure with an HTTP status hint."""

    def __init__(self, message: str, code: str = "PIPELINE_ERROR", http_status: int = 400):
        super().__init__(message)
        self.code = code
        self.http_status = http_status


def _utc_stamp() -> str:
    from datetime import UTC, datetime
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def run_pipeline(
    cube: np.ndarray,
    label_map: np.ndarray,
    wavelengths_nm: np.ndarray | None = None,
    *,
    scenario_id: str = "default",
    train_cnn_model: bool = True,
    train_unet_model: bool = True,
    run_id: str | None = None,
    device: str | None = None,
    save_results: bool | None = None,
    patch_size: int | None = None,
    stride: int | None = None,
) -> tuple[dict, PipelineArtifacts]:
    """
    Run all six stages for one hyperspectral scene.

    Parameters
    ----------
    cube : np.ndarray
        Preprocessed-ready hyperspectral cube (bands, height, width).
    label_map : np.ndarray
        Integer class-id map (height, width) with values 0..4.
    wavelengths_nm : np.ndarray, optional
        Centre wavelength of each band (nanometres).
    scenario_id : str
        Identifier for provenance.
    train_cnn_model / train_unet_model : bool
        Whether to train each deep-learning model.
    run_id : str, optional
        Custom run identifier; auto-generated if omitted.
    device : str, optional
        Torch device ("cuda" or "cpu").
    save_results : bool, optional
        Whether to persist run artefacts; defaults to config.SAVE_RESULTS.

    Returns
    -------
    (result, artifacts)
    """
    started = time.perf_counter()
    timings: dict[str, float] = {}
    warnings: list[str] = []
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    run_id = run_id or f"{scenario_id}_{_utc_stamp()}"
    save = config.SAVE_RESULTS if save_results is None else save_results

    # ------------------------------------------------------------------
    # Stage 1 - Preprocessing
    # ------------------------------------------------------------------
    LOGGER.info("[%s] Stage 1/6: hyperspectral preprocessing", scenario_id)
    t0 = time.perf_counter()
    preprocessed, prep_diag = preprocessing.preprocess(
        cube,
        wavelengths_nm=wavelengths_nm,
    )
    timings["stage1_preprocessing"] = round(time.perf_counter() - t0, 2)
    warnings.extend(prep_diag.warnings)

    # ------------------------------------------------------------------
    # Stage 2 - Band selection
    # ------------------------------------------------------------------
    LOGGER.info("[%s] Stage 2/6: band selection", scenario_id)
    t0 = time.perf_counter()
    band_sel_cube, bs_diag = band_selection.select_bands(preprocessed)
    timings["stage2_band_selection"] = round(time.perf_counter() - t0, 2)
    warnings.extend(bs_diag.warnings)

    n_bands = band_sel_cube.shape[0]

    # ------------------------------------------------------------------
    # Stage 3 - Dataset preparation
    # ------------------------------------------------------------------
    LOGGER.info("[%s] Stage 3/6: dataset preparation", scenario_id)
    t0 = time.perf_counter()
    ds = dataset_module.HyperspectralWaterloggingDataset(
        band_sel_cube,
        label_map,
        patch_size=patch_size,
        stride=stride,
    )
    patches, patch_labels, label_patches = ds.extract_patches()
    ds.normalise()
    splits = ds.split()
    timings["stage3_dataset"] = round(time.perf_counter() - t0, 2)
    warnings.extend(ds.diagnostics.warnings)

    # Convert splits to torch tensors
    def _to_tensor(arr):
        return torch.from_numpy(np.asarray(arr, dtype=np.float32))

    train_patches = _to_tensor(splits["train"][0])
    train_labels = torch.from_numpy(splits["train"][1].astype(np.int64))
    train_label_patches = torch.from_numpy(splits["train"][2].astype(np.int64))
    val_patches = _to_tensor(splits["val"][0])
    val_labels = torch.from_numpy(splits["val"][1].astype(np.int64))
    val_label_patches = torch.from_numpy(splits["val"][2].astype(np.int64))
    test_patches = _to_tensor(splits["test"][0])
    test_labels = torch.from_numpy(splits["test"][1].astype(np.int64))
    test_label_patches = torch.from_numpy(splits["test"][2].astype(np.int64))

    # ------------------------------------------------------------------
    # Stage 4 - Deep learning
    # ------------------------------------------------------------------
    cnn_result = None
    unet_result = None

    if train_cnn_model:
        LOGGER.info("[%s] Stage 4/6: CNN training", scenario_id)
        t0 = time.perf_counter()
        cnn = build_cnn(in_channels=n_bands)
        cnn_result = train_cnn(
            cnn,
            train_patches,
            train_labels,
            val_patches,
            val_labels,
            device=device,
        )
        timings["stage4_cnn"] = round(time.perf_counter() - t0, 2)

    if train_unet_model:
        LOGGER.info("[%s] Stage 4/6: U-Net training", scenario_id)
        t0 = time.perf_counter()
        unet = build_unet(in_channels=n_bands)
        unet_result = train_unet(
            unet,
            train_patches,
            train_label_patches,
            val_patches,
            val_label_patches,
            device=device,
        )
        timings["stage4_unet"] = round(time.perf_counter() - t0, 2)

    # ------------------------------------------------------------------
    # Stage 5 - Evaluation
    # ------------------------------------------------------------------
    LOGGER.info("[%s] Stage 5/6: evaluation", scenario_id)
    t0 = time.perf_counter()
    cnn_metrics = None
    unet_metrics = None

    if cnn_result is not None:
        cnn = build_cnn(in_channels=n_bands)
        cnn_state = torch.load(cnn_result.checkpoint_path, map_location=device)
        cnn.load_state_dict(cnn_state["model_state_dict"])
        cnn.eval()
        with torch.no_grad():
            test_logits = cnn(test_patches.to(device))
            cnn_preds = test_logits.argmax(dim=1).cpu().numpy()
        cnn_metrics = evaluator.evaluate_classification(
            cnn_preds, test_labels.numpy(),
            num_classes=config.CNN_NUM_CLASSES,
            class_names=config.CLASSES,
        )
        LOGGER.info(
            "[%s] CNN test accuracy=%.4f f1=%.4f",
            scenario_id, cnn_metrics.accuracy, cnn_metrics.f1_macro,
        )

    if unet_result is not None:
        unet = build_unet(in_channels=n_bands)
        unet_state = torch.load(unet_result.checkpoint_path, map_location=device)
        unet.load_state_dict(unet_state["model_state_dict"])
        unet.eval()
        with torch.no_grad():
            test_logits = unet(test_patches.to(device))
            unet_preds = test_logits.argmax(dim=1).cpu().numpy()
        unet_metrics = evaluator.evaluate_segmentation(
            unet_preds, test_label_patches.numpy(),
            num_classes=config.UNET_NUM_CLASSES,
            class_names=config.CLASSES,
        )
        LOGGER.info(
            "[%s] U-Net test accuracy=%.4f f1=%.4f iou=%.4f",
            scenario_id, unet_metrics.accuracy, unet_metrics.f1_macro, unet_metrics.iou_macro,
        )

    timings["stage5_evaluation"] = round(time.perf_counter() - t0, 2)

    # ------------------------------------------------------------------
    # Stage 6 - Mapping / decision support
    # ------------------------------------------------------------------
    LOGGER.info("[%s] Stage 6/6: mapping", scenario_id)
    t0 = time.perf_counter()
    map_result = None
    map_artifacts: dict[str, str] = {}
    if unet_result is not None:
        # Full-scene inference: slide the trained U-Net over the whole
        # band-selected cube and average overlapping class probabilities.
        unet = build_unet(in_channels=n_bands)
        state = torch.load(unet_result.checkpoint_path, map_location=device)
        unet.load_state_dict(state["model_state_dict"])
        unet.to(device)

        inference_cube = band_sel_cube.astype(np.float32)
        # Min-max normalise per band, matching the training normalisation
        mn = inference_cube.min(axis=(1, 2), keepdims=True)
        mx = inference_cube.max(axis=(1, 2), keepdims=True)
        inference_cube = (inference_cube - mn) / (mx - mn + 1e-8)

        scene_pred, scene_prob = mapper.sliding_window_predict(
            unet,
            inference_cube,
            patch_size=patch_size or config.PATCH_SIZE,
            stride=stride or max(1, (patch_size or config.PATCH_SIZE) // 2),
            num_classes=config.UNET_NUM_CLASSES,
            device=device,
        )
        map_result = mapper.generate_maps(scene_pred)

        if save:
            try:
                config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                map_png = str(config.RESULTS_DIR / f"{run_id}_map.png")
                sev_png = str(config.RESULTS_DIR / f"{run_id}_severity.png")
                map_artifacts["class_map_png"] = mapper.save_class_map_png(
                    scene_pred, map_png
                )
                map_artifacts["severity_png"] = mapper.save_severity_png(
                    scene_prob[0] + scene_prob[2], sev_png
                )
                npy_path = str(config.RESULTS_DIR / f"{run_id}_map.npy")
                np.save(npy_path, scene_pred)
                map_artifacts["class_map_npy"] = npy_path
                map_result.output_path = map_artifacts["class_map_png"]
            except Exception as error:  # noqa: BLE001
                LOGGER.warning("Could not persist map artefacts: %s", error)
                warnings.append(f"Map artefacts could not be saved: {error}")

    timings["stage6_mapping"] = round(time.perf_counter() - t0, 2)

    # ------------------------------------------------------------------
    # Provenance and result assembly
    # ------------------------------------------------------------------
    processing_seconds = round(time.perf_counter() - started, 2)
    timings["total"] = processing_seconds

    provenance = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "theme": config.THEME,
        "project": config.PROJECT_NAME,
        "version": config.PROJECT_VERSION,
        "parameters": config.config_snapshot(),
        "device": device,
        "timings_seconds": timings,
        "warnings": warnings,
    }

    result = {
        "run_id": run_id,
        "status": "ok",
        "scenario_id": scenario_id,
        "preprocessing": preprocessing.__dict__ if False else {
            "bad_bands_removed": prep_diag.bad_bands_removed,
            "output_shape": prep_diag.output_shape,
        },
        "band_selection": {
            "method": bs_diag.method,
            "n_input_bands": bs_diag.n_input_bands,
            "n_output_bands": bs_diag.n_output_bands,
        },
        "dataset": {
            "n_patches": ds.diagnostics.n_patches,
            "train": ds.diagnostics.train_count,
            "val": ds.diagnostics.val_count,
            "test": ds.diagnostics.test_count,
            "class_counts": ds.diagnostics.class_counts,
        },
        "cnn": {
            "best_metric": cnn_result.best_metric if cnn_result else None,
            "best_epoch": cnn_result.best_epoch if cnn_result else None,
            "checkpoint": cnn_result.checkpoint_path if cnn_result else None,
        } if cnn_result else None,
        "unet": {
            "best_metric": unet_result.best_metric if unet_result else None,
            "best_epoch": unet_result.best_epoch if unet_result else None,
            "checkpoint": unet_result.checkpoint_path if unet_result else None,
        } if unet_result else None,
        "metrics": {
            "cnn": {
                "accuracy": cnn_metrics.accuracy,
                "f1_macro": cnn_metrics.f1_macro,
                "precision_macro": cnn_metrics.precision_macro,
                "recall_macro": cnn_metrics.recall_macro,
            } if cnn_metrics else None,
            "unet": {
                "accuracy": unet_metrics.accuracy,
                "f1_macro": unet_metrics.f1_macro,
                "iou_macro": unet_metrics.iou_macro,
                "precision_macro": unet_metrics.precision_macro,
                "recall_macro": unet_metrics.recall_macro,
            } if unet_metrics else None,
        },
        "map": {
            "waterlogged_area_ha": map_result.waterlogged_area_ha if map_result else None,
            "total_area_ha": map_result.total_area_ha if map_result else None,
            "waterlogged_fraction": map_result.waterlogged_fraction if map_result else None,
            "class_counts": map_result.class_counts if map_result else None,
            "class_areas_ha": map_result.class_areas_ha if map_result else None,
            "artifacts": map_artifacts,
        } if map_result else None,
        "provenance": provenance,
        "processing_seconds": processing_seconds,
        "timings_seconds": timings,
        "warnings": warnings,
    }

    artifacts = PipelineArtifacts(
        preprocessed_cube=preprocessed,
        band_selected_cube=band_sel_cube,
        preprocessing_diagnostics={
            "bad_bands_removed": prep_diag.bad_bands_removed,
            "output_shape": prep_diag.output_shape,
            "warnings": prep_diag.warnings,
        },
        band_selection_diagnostics={
            "method": bs_diag.method,
            "n_input_bands": bs_diag.n_input_bands,
            "n_output_bands": bs_diag.n_output_bands,
            "selected_band_indices": bs_diag.selected_band_indices,
            "warnings": bs_diag.warnings,
        },
        dataset_diagnostics={
            "n_patches": ds.diagnostics.n_patches,
            "class_counts": ds.diagnostics.class_counts,
            "train_count": ds.diagnostics.train_count,
            "val_count": ds.diagnostics.val_count,
            "test_count": ds.diagnostics.test_count,
            "warnings": ds.diagnostics.warnings,
        },
        cnn_result=cnn_result,
        unet_result=unet_result,
        cnn_metrics=cnn_metrics,
        unet_metrics=unet_metrics,
        map_result=map_result,
        run_id=run_id,
        warnings=warnings,
    )

    if save:
        try:
            from core.reporting import save_run_result as _save
            _save(result)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("Could not persist run record: %s", error)
            warnings.append(f"Run record could not be saved: {error}")
            result["warnings"] = warnings

    LOGGER.info(
        "Pipeline complete: %s | %.2f s | cnn_f1=%.4f unet_iou=%.4f",
        run_id, processing_seconds,
        (cnn_metrics.f1_macro if cnn_metrics else 0.0),
        (unet_metrics.iou_macro if unet_metrics else 0.0),
    )
    return result, artifacts
