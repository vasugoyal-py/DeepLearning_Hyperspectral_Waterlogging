"""
Stage 5 - Evaluation.

Computes accuracy, precision, recall, F1-score, and Intersection over Union
(IoU) for both CNN classification (patch-level) and U-Net segmentation
(pixel-level), with per-class and macro-averaged metrics plus confusion matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import config
import numpy as np

LOGGER = logging.getLogger("deepwaterlog.evaluator")


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision_per_class: list[float]
    recall_per_class: list[float]
    f1_per_class: list[float]
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: np.ndarray


@dataclass
class SegmentationMetrics:
    accuracy: float
    precision_per_class: list[float]
    recall_per_class: list[float]
    f1_per_class: list[float]
    iou_per_class: list[float]
    precision_macro: float
    recall_macro: float
    f1_macro: float
    iou_macro: float
    confusion_matrix: np.ndarray


def _confusion_matrix(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute a (num_classes, num_classes) confusion matrix: rows=truth, cols=pred."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets.ravel(), preds.ravel()):
        cm[int(t), int(p)] += 1
    return cm


def evaluate_classification(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int | None = None,
    class_names: list[str] | None = None,
) -> ClassificationMetrics:
    """
    Evaluate patch-level CNN predictions.

    ``preds`` and ``targets`` are 1-D arrays of integer class ids.
    """
    num_classes = num_classes or config.CNN_NUM_CLASSES
    class_names = class_names or config.CLASSES
    preds = np.asarray(preds).ravel()
    targets = np.asarray(targets).ravel()

    cm = _confusion_matrix(preds, targets, num_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    accuracy = float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0

    return ClassificationMetrics(
        accuracy=accuracy,
        precision_per_class=[float(v) for v in precision],
        recall_per_class=[float(v) for v in recall],
        f1_per_class=[float(v) for v in f1],
        precision_macro=float(precision.mean()),
        recall_macro=float(recall.mean()),
        f1_macro=float(f1.mean()),
        confusion_matrix=cm,
    )


def evaluate_segmentation(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int | None = None,
    class_names: list[str] | None = None,
    iou_threshold: float | None = None,
) -> SegmentationMetrics:
    """
    Evaluate pixel-level U-Net predictions.

    ``preds`` and ``targets`` are integer class-id maps of the same shape
    (N, H, W) or (H, W).
    """
    num_classes = num_classes or config.UNET_NUM_CLASSES
    class_names = class_names or config.CLASSES
    iou_threshold = config.IOU_THRESHOLD if iou_threshold is None else iou_threshold
    preds = np.asarray(preds).ravel()
    targets = np.asarray(targets).ravel()

    cm = _confusion_matrix(preds, targets, num_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    union = tp + fp + fn

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    iou = tp / (union + 1e-12)

    accuracy = float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0

    return SegmentationMetrics(
        accuracy=accuracy,
        precision_per_class=[float(v) for v in precision],
        recall_per_class=[float(v) for v in recall],
        f1_per_class=[float(v) for v in f1],
        iou_per_class=[float(v) for v in iou],
        precision_macro=float(precision.mean()),
        recall_macro=float(recall.mean()),
        f1_macro=float(f1.mean()),
        iou_macro=float(iou.mean()),
        confusion_matrix=cm,
    )


def format_metrics_report(
    metrics: Any,
    model_type: str,
    class_names: list[str] | None = None,
) -> str:
    """Return a human-readable evaluation report string."""
    class_names = class_names or config.CLASSES
    lines = []
    lines.append(f"=== {model_type} Evaluation ===")
    if isinstance(metrics, ClassificationMetrics):
        lines.append(f"Accuracy        : {metrics.accuracy:.4f}")
        lines.append(f"Precision (macro): {metrics.precision_macro:.4f}")
        lines.append(f"Recall (macro)   : {metrics.recall_macro:.4f}")
        lines.append(f"F1-score (macro) : {metrics.f1_macro:.4f}")
    elif isinstance(metrics, SegmentationMetrics):
        lines.append(f"Accuracy        : {metrics.accuracy:.4f}")
        lines.append(f"Precision (macro): {metrics.precision_macro:.4f}")
        lines.append(f"Recall (macro)   : {metrics.recall_macro:.4f}")
        lines.append(f"F1-score (macro) : {metrics.f1_macro:.4f}")
        lines.append(f"IoU (macro)      : {metrics.iou_macro:.4f}")

    lines.append("")
    lines.append("Per-class metrics:")
    header = f"{'class':<22} {'precision':>10} {'recall':>10} {'f1':>10}"
    if isinstance(metrics, SegmentationMetrics):
        header += f" {'iou':>10}"
    lines.append(header)
    for i, name in enumerate(class_names):
        row = f"{name:<22} {metrics.precision_per_class[i]:>10.4f} {metrics.recall_per_class[i]:>10.4f} {metrics.f1_per_class[i]:>10.4f}"
        if isinstance(metrics, SegmentationMetrics):
            row += f" {metrics.iou_per_class[i]:>10.4f}"
        lines.append(row)

    lines.append("")
    lines.append("Confusion matrix (rows=truth, cols=pred):")
    cm = metrics.confusion_matrix
    lines.append("      " + " ".join(f"{i:>5}" for i in range(cm.shape[0])))
    for i, row in enumerate(cm):
        lines.append(f"class {i}: " + " ".join(f"{v:>5}" for v in row))

    return "\n".join(lines)
