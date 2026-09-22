import numpy as np
import pytest
from tifffile import imwrite

from align_toolbox.segmentation import get_segmentation_function, segment_image
from align_toolbox.segmentation.segmentation_tools import (
    double_threshold_segmentation,
    threshold_segmentation,
)


def _iou(a, b):
    a, b = a > 0, b > 0
    return np.logical_and(a, b).sum() / np.logical_or(a, b).sum()


@pytest.mark.parametrize(
    "method, min_iou",
    [("threshold", 0.95), ("double_threshold", 0.5), ("edge_based", 0.75)],
)
def test_segment_image_recovers_synthetic_worm(
    curved_worm_image, curved_worm_mask, method, min_iou
):
    mask = segment_image(curved_worm_image, method, pixelsize=1.0, is_stack=False)
    assert mask.shape == curved_worm_image.shape
    assert mask.dtype == np.uint8
    assert _iou(mask, curved_worm_mask) > min_iou


@pytest.mark.parametrize("method", ["threshold", "double_threshold", "edge_based"])
def test_masks_are_zero_one(curved_worm_image, method):
    mask = segment_image(curved_worm_image, method, pixelsize=1.0, is_stack=False)
    assert set(np.unique(mask)) == {0, 1}


@pytest.mark.parametrize("method", ["otsu", "li", "yen", "triangle"])
def test_threshold_segmentation_methods(curved_worm_image, curved_worm_mask, method):
    mask = threshold_segmentation(curved_worm_image, pixelsize=1.0, method=method)
    assert _iou(mask, curved_worm_mask) > 0.5


def test_threshold_segmentation_rejects_unknown_method(curved_worm_image):
    with pytest.raises(ValueError):
        threshold_segmentation(curved_worm_image, 1.0, method="magic")


def test_small_objects_are_removed(curved_worm_image):
    image = curved_worm_image.copy()
    image[5:8, 5:8] = 4000  # a 9 px speck
    mask = double_threshold_segmentation(image, pixelsize=1.0)
    assert not mask[5:8, 5:8].any()


def test_segment_image_stack_segments_each_plane(curved_worm_image):
    stack = np.stack([curved_worm_image, curved_worm_image[::-1]])
    masks = segment_image(stack, "threshold", pixelsize=1.0, is_stack=True)
    assert masks.shape == stack.shape
    np.testing.assert_array_equal(masks[1], masks[0][::-1])


def test_segment_image_reads_tiff_path(tmp_path, curved_worm_image):
    path = tmp_path / "worm.tiff"
    imwrite(path, np.stack([np.zeros_like(curved_worm_image), curved_worm_image]))
    from_path = segment_image(
        str(path), "threshold", channels=[1], pixelsize=1.0, is_stack=False
    )
    from_array = segment_image(
        curved_worm_image, "threshold", pixelsize=1.0, is_stack=False
    )
    np.testing.assert_array_equal(from_path, from_array)


@pytest.mark.parametrize("function", [segment_image, get_segmentation_function])
def test_edge_based_requires_pixelsize(curved_worm_image, function):
    with pytest.raises(ValueError, match="Pixelsize"):
        if function is segment_image:
            function(curved_worm_image, "edge_based")
        else:
            function("edge_based")


@pytest.mark.parametrize("function", [segment_image, get_segmentation_function])
def test_unknown_method_raises(curved_worm_image, function):
    with pytest.raises(ValueError):
        if function is segment_image:
            function(curved_worm_image, "watershed", pixelsize=1.0)
        else:
            function("watershed", pixelsize=1.0)


@pytest.mark.parametrize("method", ["threshold", "double_threshold", "edge_based"])
def test_get_segmentation_function_matches_segment_image(curved_worm_image, method):
    segment = get_segmentation_function(method, pixelsize=1.0)
    np.testing.assert_array_equal(
        segment(curved_worm_image),
        segment_image(curved_worm_image, method, pixelsize=1.0, is_stack=False),
    )


def test_edge_based_segmentation_rejects_3d_input(curved_worm_image):
    from align_toolbox.segmentation import edge_based_segmentation

    with pytest.raises(ValueError, match="2D"):
        edge_based_segmentation(np.stack([curved_worm_image] * 2), pixelsize=1.0)
