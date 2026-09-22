import numpy as np
import pytest

from align_toolbox.foundation.keypoint_detection import heatmap_to_keypoints_1D
from align_toolbox.foundation.utils import (
    find_best_string_match,
    interpolate_infs,
    interpolate_nans,
    interpolate_nans_infs,
)


def test_interpolate_nans_fills_gaps_linearly():
    signal = np.array([0.0, np.nan, 2.0, np.nan, np.nan, 5.0])
    np.testing.assert_allclose(interpolate_nans(signal), [0, 1, 2, 3, 4, 5])


def test_interpolate_nans_holds_edge_values_constant():
    signal = np.array([np.nan, 1.0, 2.0, np.nan])
    np.testing.assert_allclose(interpolate_nans(signal), [1, 1, 2, 2])


def test_interpolate_nans_all_nan_returns_all_nan():
    assert np.isnan(interpolate_nans(np.full(4, np.nan))).all()


def test_interpolate_infs_replaces_positive_and_negative_infinity():
    signal = np.array([0.0, np.inf, 2.0, -np.inf, 4.0])
    np.testing.assert_allclose(interpolate_infs(signal), [0, 1, 2, 3, 4])


def test_interpolate_nans_infs_handles_both():
    signal = np.array([0.0, np.nan, 2.0, np.inf, 4.0])
    np.testing.assert_allclose(interpolate_nans_infs(signal), [0, 1, 2, 3, 4])


@pytest.mark.parametrize(
    "reference, expected",
    [
        ("ch2_seg_str_volume", "ch2_seg_str_worm_type"),
        ("ch1_seg_str_volume", "ch1_seg_str_worm_type"),
    ],
)
def test_find_best_string_match_picks_matching_channel(reference, expected):
    candidates = ["ch1_seg_str_worm_type", "ch2_seg_str_worm_type"]
    assert find_best_string_match(reference, candidates) == expected


def test_find_best_string_match_empty_candidates_returns_none():
    assert find_best_string_match("volume", []) is None


def test_heatmap_to_keypoints_returns_argmax_per_class():
    heatmap = np.zeros((2, 10))
    heatmap[0, 3] = 0.9
    heatmap[1, 7] = 0.8
    keypoints = heatmap_to_keypoints_1D(heatmap, presence=np.array([0.9, 0.9]))
    np.testing.assert_array_equal(keypoints, [3, 7])


def test_heatmap_to_keypoints_absent_or_weak_classes_are_nan():
    heatmap = np.zeros((3, 10))
    heatmap[0, 3] = 0.9  # present but class marked absent
    heatmap[1, 5] = 0.1  # present but below the height threshold
    heatmap[2, 8] = 0.9
    keypoints = heatmap_to_keypoints_1D(heatmap, presence=np.array([0.2, 0.9, 0.9]))
    assert np.isnan(keypoints[0])
    assert np.isnan(keypoints[1])
    assert keypoints[2] == 8
