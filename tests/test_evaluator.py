"""Tests for evaluation metrics (Stage 5)."""

import numpy as np
import pytest

from core.evaluator import (
    evaluate_classification,
    evaluate_segmentation,
)


def test_classification_perfect_score():
    preds = np.array([0, 1, 2, 3, 4])
    targets = np.array([0, 1, 2, 3, 4])
    m = evaluate_classification(preds, targets, num_classes=5)
    assert m.accuracy == 1.0
    assert m.f1_macro == pytest.approx(1.0)
    assert m.precision_macro == pytest.approx(1.0)
    assert m.recall_macro == pytest.approx(1.0)


def test_classification_confusion_matrix_shape():
    preds = np.array([0, 0, 1, 1])
    targets = np.array([0, 1, 0, 1])
    m = evaluate_classification(preds, targets, num_classes=2)
    assert m.confusion_matrix.shape == (2, 2)


def test_segmentation_perfect_score():
    preds = np.zeros((4, 4), dtype=np.int64)
    targets = np.zeros((4, 4), dtype=np.int64)
    preds[0, 0] = 1
    targets[0, 0] = 1
    m = evaluate_segmentation(preds, targets, num_classes=2)
    assert m.accuracy == 1.0
    assert m.iou_macro == pytest.approx(1.0)
