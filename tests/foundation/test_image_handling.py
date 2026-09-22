from datetime import datetime

import numpy as np
import pytest
from tifffile import imwrite

from align_toolbox.foundation import image_handling


@pytest.mark.parametrize("shape", [(5, 7), (3, 5, 7), (2, 3, 5, 7)])
def test_pad_to_dim_pads_bottom_right_only(shape):
    image = np.ones(shape)
    padded = image_handling.pad_to_dim(image, 8, 10, pad_value=-1)
    assert padded.shape == shape[:-2] + (8, 10)
    np.testing.assert_array_equal(padded[..., :5, :7], 1)
    assert (padded[..., 5:, :] == -1).all() and (padded[..., :, 7:] == -1).all()


def test_pad_to_dim_equally_centers_the_image():
    padded = image_handling.pad_to_dim_equally(np.ones((2, 2)), 5, 6)
    assert padded.shape == (5, 6)
    # odd remainder goes to the bottom/right
    np.testing.assert_array_equal(np.argwhere(padded)[[0, -1]], [[1, 2], [2, 3]])


def test_pad_to_dim_equally_rejects_smaller_target():
    with pytest.raises(ValueError):
        image_handling.pad_to_dim_equally(np.ones((4, 4)), 3, 5)


def test_crop_to_dim_keeps_top_left():
    image = np.arange(30).reshape(5, 6)
    np.testing.assert_array_equal(
        image_handling.crop_to_dim(image, 2, 3), image[:2, :3]
    )


def test_crop_to_dim_equally_keeps_center_and_leading_dims():
    image = np.arange(2 * 6 * 6).reshape(2, 6, 6)
    cropped = image_handling.crop_to_dim_equally(image, 2, 4)
    np.testing.assert_array_equal(cropped, image[:, 2:4, 1:5])


def test_crop_to_dim_equally_rejects_larger_target():
    with pytest.raises(ValueError):
        image_handling.crop_to_dim_equally(np.ones((4, 4)), 5, 2)


def test_pad_then_crop_equally_roundtrips(rng):
    image = rng.random((7, 9))
    padded = image_handling.pad_to_dim_equally(image, 16, 16)
    np.testing.assert_array_equal(
        image_handling.crop_to_dim_equally(padded, 7, 9), image
    )


def test_crop_and_pad_images_to_same_dim():
    a, b = np.ones((4, 10)), np.ones((8, 6))
    cropped_a, cropped_b = image_handling.crop_images_to_same_dim(a, b)
    assert cropped_a.shape == cropped_b.shape == (4, 6)
    padded_a, padded_b = image_handling.pad_images_to_same_dim(a, b)
    assert padded_a.shape == padded_b.shape == (8, 10)


@pytest.mark.parametrize("axes", [(-1,), (-2,), (-1, -2)])
def test_align_images_orientation_ssim_undoes_a_flip(rng, axes):
    reference = rng.random((20, 30))
    flipped = np.flip(reference, axis=axes)
    aligned = image_handling.align_images_orientation_ssim(flipped, reference)
    np.testing.assert_array_equal(aligned, reference)


@pytest.mark.parametrize(
    "dtype, max_value", [(np.uint8, 255), (np.uint16, 65535), (np.float32, 1.0)]
)
def test_normalize_image_spans_destination_range(dtype, max_value):
    image = np.array([[10.0, 20.0], [30.0, 50.0]])
    normalized = image_handling.normalize_image(image, dtype)
    assert normalized.dtype == dtype
    assert normalized.min() == pytest.approx(0, abs=1e-6)
    assert normalized.max() == pytest.approx(max_value)


def test_normalize_image_rejects_unsupported_dtype():
    with pytest.raises(ValueError):
        image_handling.normalize_image(np.ones((2, 2)), np.int32)


def test_augment_contrast_returns_uint16_image_of_same_shape(rng):
    image = rng.random((64, 64))
    augmented = image_handling.augment_contrast(image)
    assert augmented.shape == image.shape
    assert augmented.dtype == np.uint16


# --- TIFF I/O -----------------------------------------------------------------


@pytest.fixture
def channel_stack_tiff(tmp_path):
    """A (C=3, H, W) TIFF whose channel c is filled with the value c."""
    image = np.stack([np.full((8, 10), c, dtype=np.uint16) for c in range(3)])
    path = tmp_path / "channels.tiff"
    imwrite(path, image)
    return str(path), image


@pytest.fixture
def zc_stack_tiff(tmp_path):
    """A (Z=4, C=3, H, W) TIFF whose channel c is filled with the value c."""
    image = np.zeros((4, 3, 8, 10), dtype=np.uint16)
    for c in range(3):
        image[:, c] = c
    path = tmp_path / "zc_stack.tiff"
    imwrite(path, image)
    return str(path), image


def test_read_tiff_file_returns_everything_without_channel_selection(
    channel_stack_tiff,
):
    path, image = channel_stack_tiff
    np.testing.assert_array_equal(image_handling.read_tiff_file(path), image)


@pytest.mark.parametrize(
    "fixture_name, channels, expected_shape, expected_values",
    [
        ("channel_stack_tiff", [1], (8, 10), [1]),
        ("channel_stack_tiff", [0, 2], (2, 8, 10), [0, 2]),
        ("zc_stack_tiff", [2], (4, 8, 10), [2]),
        ("zc_stack_tiff", [0, 2], (4, 2, 8, 10), [0, 2]),
    ],
)
def test_read_tiff_file_selects_channels(
    request, fixture_name, channels, expected_shape, expected_values
):
    path, _ = request.getfixturevalue(fixture_name)
    image = image_handling.read_tiff_file(path, channels_to_keep=channels)
    assert image.shape == expected_shape
    assert sorted(np.unique(image)) == expected_values


@pytest.mark.parametrize(
    "fixture_name, channels",
    [
        ("channel_stack_tiff", None),
        ("channel_stack_tiff", [1]),
        ("channel_stack_tiff", [0, 2]),
        ("zc_stack_tiff", [2]),
        ("zc_stack_tiff", [0, 2]),
    ],
)
def test_get_shape_from_tiff_matches_read_tiff_file(request, fixture_name, channels):
    path, _ = request.getfixturevalue(fixture_name)
    expected = image_handling.read_tiff_file(path, channels_to_keep=channels).shape
    assert tuple(image_handling.get_shape_from_tiff(path, channels)) == expected


def test_read_tiff_file_missing_file_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="missing.tiff"):
        image_handling.read_tiff_file(str(tmp_path / "missing.tiff"))


def test_get_shape_from_tiff_missing_file_returns_none(tmp_path):
    assert image_handling.get_shape_from_tiff(str(tmp_path / "missing.tiff")) is None


@pytest.fixture
def ome_tiff(tmp_path):
    """A (T=2, Z=3, C=1, H, W) OME-TIFF."""
    path = tmp_path / "timelapse.ome.tiff"
    imwrite(
        path,
        np.zeros((2, 3, 1, 8, 10), dtype=np.uint16),
        ome=True,
        metadata={"axes": "TZCYX"},
    )
    return str(path)


def test_get_image_size_metadata_reads_ome_dimensions(ome_tiff):
    assert image_handling.get_image_size_metadata(ome_tiff) == {
        "x_dim": 10,
        "y_dim": 8,
        "z_dim": 3,
        "t_dim": 2,
        "c_dim": 1,
    }


def test_ome_stack_checks(ome_tiff):
    assert image_handling.check_if_stack(ome_tiff) == (True, (3, 2))
    assert image_handling.check_if_zstack(ome_tiff) is True
    assert image_handling.check_if_time_series(ome_tiff) is True


@pytest.mark.parametrize(
    "shape, channels, expected",
    [
        ((8, 10), None, (False, (1, 1))),
        ((5, 8, 10), None, (True, (5, 1))),
        ((2, 5, 8, 10), None, (True, (5, 2))),
        ((5, 3, 8, 10), [0], (True, (5, 1))),
        ((5, 3, 8, 10), [0, 1], (True, (5, 1))),
    ],
)
def test_check_if_stack_infers_from_shape_without_ome_metadata(
    tmp_path, shape, channels, expected
):
    path = tmp_path / "plain.tiff"
    imwrite(path, np.zeros(shape, dtype=np.uint8))
    assert image_handling.check_if_stack(str(path), channels) == expected


def test_metadata_helpers_return_none_for_plain_tiff(tmp_path):
    path = tmp_path / "plain.tiff"
    imwrite(path, np.zeros((8, 10), dtype=np.uint8))
    assert image_handling.get_image_size_metadata(str(path)) is None
    assert image_handling.get_acquisition_date(str(path)) is None


def test_get_acquisition_date_reads_ome_metadata(tmp_path):
    from ome_types import to_xml
    from ome_types.model import OME, Image, Pixels

    acquired = datetime(2024, 3, 1, 12, 30)
    ome = OME(
        images=[
            Image(
                acquisition_date=acquired,
                pixels=Pixels(
                    dimension_order="XYZCT",
                    type="uint8",
                    size_x=10,
                    size_y=8,
                    size_z=1,
                    size_c=1,
                    size_t=1,
                    metadata_only=True,
                ),
            )
        ]
    )
    path = tmp_path / "dated.ome.tiff"
    imwrite(path, np.zeros((8, 10), dtype=np.uint8), description=to_xml(ome))
    date = image_handling.get_acquisition_date(str(path))
    assert date.replace(tzinfo=None) == acquired
