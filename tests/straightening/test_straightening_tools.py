import numpy as np
import pytest

from align_toolbox.straightening import Warper
from align_toolbox.straightening.straightening_tools import (
    extract_midline,
    validate_mask,
)


def test_validate_mask_accepts_single_object(curved_worm_mask):
    validate_mask(curved_worm_mask)
    validate_mask(curved_worm_mask[np.newaxis])


@pytest.mark.parametrize(
    "mask, message",
    [
        (np.ones(5), "1D"),
        (np.ones((1, 1, 5, 5)), "4 dimensions"),
        (np.zeros((5, 5)), "empty"),
        (np.array([[1, 0, 1]] * 3, dtype=np.uint8), "more than 1 connected component"),
    ],
)
def test_validate_mask_rejects_invalid_masks(mask, message):
    with pytest.raises(ValueError, match=message):
        validate_mask(mask)


def test_extract_midline_is_ordered_and_spans_the_worm(curved_worm_mask):
    midline, distance_transform = extract_midline(curved_worm_mask, return_dt=True)
    assert midline.ndim == 2 and midline.shape[1] == 2
    assert distance_transform.shape == curved_worm_mask.shape
    # skeleton points are 8-connected neighbours; only the two tip extensions
    # (straight rays to the boundary) may jump
    assert np.abs(np.diff(midline[1:-1], axis=0)).max() <= 1
    # the worm spans columns 30..290 once its round caps are included
    assert midline[:, 1].min() == pytest.approx(30, abs=3)
    assert midline[:, 1].max() == pytest.approx(290, abs=3)


def test_from_img_rejects_shape_mismatch(curved_worm_mask):
    with pytest.raises(ValueError, match="shapes don't match"):
        Warper.from_img(curved_worm_mask, curved_worm_mask[:-1])


def test_warper_2d_measures_curved_worm(curved_worm_mask, curved_worm_expected_length):
    warper = Warper.from_img(curved_worm_mask, curved_worm_mask)
    assert len(warper.splines) == 1
    assert warper.length == pytest.approx(curved_worm_expected_length, rel=0.05)
    # width is the mean max distance-transform diameter, inflated by 20%
    assert warper.width == pytest.approx(21 * 1.2, rel=0.15)


def test_warp_2d_img_straightens_mask_into_a_band(curved_worm_mask):
    warper = Warper.from_img(curved_worm_mask, curved_worm_mask)
    straightened = warper.warp_2D_img(
        curved_worm_mask, 0, interpolation_order=0, preserve_dtype=True
    )
    assert straightened.shape == (np.ceil(warper.width), np.ceil(warper.length))
    assert straightened.dtype == curved_worm_mask.dtype
    assert straightened.sum() == pytest.approx(curved_worm_mask.sum(), rel=0.02)
    # every column of the band should be close to the true 21 px thickness
    widths = straightened.sum(axis=0)
    assert np.median(widths[widths > 0]) == pytest.approx(21, abs=2)


def test_warp_2d_img_scale_factor_and_mirror(curved_worm_mask, curved_worm_image):
    warper = Warper.from_img(curved_worm_mask, curved_worm_mask)
    base = warper.warp_2D_img(curved_worm_image, 0)
    scaled = warper.warp_2D_img(curved_worm_image, 0, scale_factor=2)
    mirrored = warper.warp_2D_img(curved_worm_image, 0, mirror=True)
    assert scaled.shape == tuple(np.ceil([2 * warper.width, 2 * warper.length]))
    assert mirrored.shape == base.shape
    assert not np.array_equal(mirrored, base)


def test_warper_pickle_roundtrip(tmp_path, curved_worm_mask, curved_worm_image):
    warper = Warper.from_img(curved_worm_mask, curved_worm_mask)
    path = tmp_path / "warper.pkl"
    warper.to_pickle(path)
    restored = Warper.from_pickle(path)
    assert (restored.length, restored.width) == (warper.length, warper.width)
    np.testing.assert_array_equal(
        restored.warp_2D_img(curved_worm_image, 0),
        warper.warp_2D_img(curved_worm_image, 0),
    )


def test_warper_3d_fits_one_spline_per_plane(curved_worm_stack):
    image = curved_worm_stack.astype(np.uint16) * 1000
    warper = Warper.from_img(image, curved_worm_stack)
    assert len(warper.splines) == 3
    warped = warper.warp_3D_img(image)
    assert warped.shape == (3, np.ceil(warper.width), np.ceil(warper.length))


def test_warper_3d_empty_plane_warps_to_blank(curved_worm_stack):
    stack = curved_worm_stack.copy()
    stack[1] = 0
    image = stack.astype(np.uint16) * 1000
    warper = Warper.from_img(image, stack)
    assert warper.splines[1] is None
    warped = warper.warp_3D_img(image)
    assert not warped[1].any()
    assert warped[0].any() and warped[2].any()


def test_warp_3d_img_rejects_plane_count_mismatch(curved_worm_stack):
    image = curved_worm_stack.astype(np.uint16)
    warper = Warper.from_img(image, curved_worm_stack)
    with pytest.raises(ValueError, match="Incompatible number of planes"):
        warper.warp_3D_img(image[:2])


def test_rescaled_3d_img_makes_spacing_isotropic(curved_worm_stack):
    image = curved_worm_stack[[0, 2]].astype(np.uint16) * 1000
    warper = Warper.from_img(image, curved_worm_stack[[0, 2]])
    rescaled, final_spacing = warper.rescaled_3D_img(
        image, spacing=(2.0, 0.5, 0.5), return_final_spacing=True
    )
    # z spacing is 4x the xy spacing, so the 2 planes become 8
    assert rescaled.shape[0] == 8
    assert rescaled.dtype == image.dtype
    np.testing.assert_allclose(final_spacing, [0.5, 0.5, 0.5])
