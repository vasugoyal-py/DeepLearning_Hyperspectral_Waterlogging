"""DeepWaterLog core package."""

from core.preprocessing import preprocess, PreprocessingDiagnostics
from core.band_selection import select_bands, BandSelectionDiagnostics
from core.dataset import HyperspectralWaterloggingDataset, DatasetDiagnostics
from core.evaluator import (
    evaluate_classification,
    evaluate_segmentation,
    ClassificationMetrics,
    SegmentationMetrics,
    format_metrics_report,
)
from core.mapper import generate_maps, save_geotiff, export_report, MapResult
from core.reporting import save_run_result

# Torch-dependent modules (deep learning). Imported lazily so the package can
# still be imported and tested without PyTorch installed.
try:
    from core.cnn_model import build_cnn, CustomCNN
    from core.unet_model import build_unet, UNet
    from core.trainer import train_cnn, train_unet, TrainResult
    _TORCH_AVAILABLE = True
except Exception:  # torch not installed or import failed
    build_cnn = CustomCNN = None  # type: ignore
    build_unet = UNet = None  # type: ignore
    train_cnn = train_unet = TrainResult = None  # type: ignore
    _TORCH_AVAILABLE = False

__all__ = [
    "preprocess",
    "PreprocessingDiagnostics",
    "select_bands",
    "BandSelectionDiagnostics",
    "HyperspectralWaterloggingDataset",
    "DatasetDiagnostics",
    "evaluate_classification",
    "evaluate_segmentation",
    "ClassificationMetrics",
    "SegmentationMetrics",
    "format_metrics_report",
    "generate_maps",
    "save_geotiff",
    "export_report",
    "MapResult",
    "save_run_result",
    "build_cnn",
    "CustomCNN",
    "build_unet",
    "UNet",
    "train_cnn",
    "train_unet",
    "TrainResult",
    "_TORCH_AVAILABLE",
]
