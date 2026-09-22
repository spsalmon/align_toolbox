import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from align_toolbox.foundation import zstack
from align_toolbox.foundation.utils import (
    NotImplementedError as TowbinNotImplementedError,
)


@pytest.fixture
def focus_stack(rng):
    """Five planes; plane 2 is sharp, the others are increasingly blurred."""
    sharp = (rng.random((64, 64)) > 0.5).astype(np.float64) * 1000
    sigmas = [4, 2, 0, 2, 4]
    return np.stack([gaussian_filter(sharp, s) if s else sharp for s in sigmas])


def test_normalize_zstack_each_plane_uses_full_range_per_plane():
    stack = np.stack([np.array([[0.0, 10.0]]), np.array([[100.0, 200.0]])])
    normalized = zstack.normalize_zstack(stack, each_plane=True, dest_dtype=np.uint8)
    np.testing.assert_array_equal(normalized, [[[0, 255]], [[0, 255]]])


def test_normalize_zstack_globally_preserves_relative_intensities():
    stack = np.stack([np.array([[0.0, 100.0]]), np.array([[100.0, 200.0]])])
    normalized = zstack.normalize_zstack(stack, each_plane=False, dest_dtype=np.float32)
    np.testing.assert_allclose(normalized, [[[0, 0.5]], [[0.5, 1]]])


def test_normalize_zstack_rejects_unsupported_dtype():
    with pytest.raises(ValueError):
        zstack.normalize_zstack(np.ones((2, 2, 2)), dest_dtype=np.int8)


def test_augment_contrast_zstack_keeps_shape(focus_stack):
    augmented = zstack.augment_contrast_zstack(focus_stack)
    assert augmented.shape == focus_stack.shape
    assert augmented.dtype == np.uint16


@pytest.mark.parametrize(
    "measure",
    ["lapv", "LAPM", "teng", "normalized_variance", "mlog"],
)
@pytest.mark.parametrize("contrast_augmentation", [False, True])
def test_find_best_plane_picks_sharpest_plane(
    focus_stack, measure, contrast_augmentation
):
    index, plane = zstack.find_best_plane(
        focus_stack, measure, contrast_augmentation=contrast_augmentation
    )
    assert index == 2
    np.testing.assert_array_equal(plane, focus_stack[2])


def test_find_best_plane_mean_picks_brightest_plane():
    stack = np.stack([np.full((4, 4), v, dtype=float) for v in (1, 5, 3)])
    stack[:, 0, 0] = 0  # avoid flat planes, which normalize to all zeros
    index, _ = zstack.find_best_plane(stack, "mean", each_plane=False)
    assert index == 1


def test_find_best_plane_multichannel_uses_requested_channel(focus_stack):
    reversed_focus = focus_stack[::-1]
    multichannel = np.stack([reversed_focus, focus_stack], axis=1)  # (Z, C, H, W)
    index, plane = zstack.find_best_plane(multichannel, "lapv", channel=1)
    assert index == 2
    assert plane.shape == (2, 64, 64)


def test_find_best_plane_multichannel_requires_channel(focus_stack):
    with pytest.raises(ValueError):
        zstack.find_best_plane(focus_stack[:, np.newaxis], "lapv")


def test_find_best_plane_unknown_measure_raises(focus_stack):
    with pytest.raises(TowbinNotImplementedError):
        zstack.find_best_plane(focus_stack, "sharpness")
