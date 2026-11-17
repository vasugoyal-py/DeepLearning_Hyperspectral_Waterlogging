"""Tests for waterlogging map generation (Stage 5/6)."""

import numpy as np

from core.mapper import generate_maps


def test_generate_maps_counts_and_area():
    predictions = np.array(
        [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]],
        dtype=np.int64,
    )
    result = generate_maps(predictions, resolution_m=10.0)

    assert result.total_area_ha == 16 * (10.0 * 10.0) / 10_000.0  # 0.16 ha
    assert result.class_counts[0] == 4
    assert result.class_counts[1] == 4
    assert result.class_counts[2] == 4
    assert result.class_counts[3] == 4
    # waterlogged = class 0 (waterlogged_soil) + class 2 (stressed_crop) = 8 px
    assert abs(result.waterlogged_fraction - 0.5) < 1e-6
    assert result.severity_map.shape == predictions.shape
