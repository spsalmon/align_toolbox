from types import SimpleNamespace

import numpy as np
import pytest
from tifffile import imwrite

from align_toolbox.deep_learning.utils import augmentation as aug


@pytest.fixture
def sample(rng):
    image = rng.random((2, 8, 12)) * 1000
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[1:3, 2:9] = 1
    return {"image": image, "mask": mask}


def test_normalize_data_range_maps_to_unit_interval(sample):
    image = aug.NormalizeDataRange(["image"])(sample)["image"]
    assert image.min() == 0 and image.max() == 1


def test_normalize_mean_std(sample):
    image = aug.NormalizeMeanStd(["image"], mean=500.0, std=2.0)(sample)["image"]
    np.testing.assert_allclose(image, (sample["image"] - 500) / 2)


def test_normalize_percentile_clips_outliers_to_near_unit_range(sample):
    sample["image"][0, 0, 0] = 1e6
    image = aug.NormalizePercentile(["image"], lo=1, hi=99)(sample)["image"]
    assert np.percentile(image, 50) == pytest.approx(0.5, abs=0.2)
    assert image.max() > 1  # values beyond the percentile are not clipped


def test_normalization_converts_lists_to_arrays():
    data = aug.NormalizeDataRange(["image"])({"image": [[0.0, 2.0]]})
    np.testing.assert_array_equal(data["image"], [[0, 1]])


def test_transforms_leave_the_input_dict_untouched(sample):
    original = sample["image"].copy()
    aug.NormalizeMeanStd(["image"], 1.0, 1.0)(sample)
    np.testing.assert_array_equal(sample["image"], original)


@pytest.mark.parametrize(
    "shape, n_channels, expected_first_pixels",
    [
        ((4, 4), 3, [0, 0, 0]),
        ((2, 4, 4), 2, [0, 1]),
        ((2, 4, 4), 4, [0, 1, 0, 1]),
        ((2, 4, 4), 5, [0, 1, 0, 1, 0]),
    ],
)
def test_enforce_n_channels_tiles_channels(shape, n_channels, expected_first_pixels):
    image = np.zeros(shape)
    if image.ndim == 3:
        image[1] = 1
    tiled = aug.EnforceNChannels(["image"], n_channels)({"image": image})["image"]
    assert tiled.shape == (n_channels, 4, 4)
    assert list(tiled[:, 0, 0]) == expected_first_pixels


def test_enforce_n_channels_rejects_too_many_channels():
    with pytest.raises(ValueError):
        aug.EnforceNChannels(["image"], 1)({"image": np.zeros((2, 4, 4))})


def test_enforce_n_channels_rejects_multichannel_stacks():
    with pytest.raises(AssertionError):
        aug.EnforceNChannels(["image"], 4)({"image": np.zeros((2, 2, 4, 4))})


@pytest.mark.parametrize("transform_class", [aug.CustomFlip, aug.CustomRotate90])
def test_geometric_transforms_move_image_and_mask_together(sample, transform_class):
    transform = transform_class(keys=["image", "mask"], prob=1.0)
    for seed in range(5):
        transform.set_random_state(seed)
        out = transform(sample)
        assert not np.array_equal(out["mask"], sample["mask"])
        # the mask marks the same pixels in the transformed image
        bright = out["image"][0][out["mask"] == 1]
        np.testing.assert_array_equal(
            np.sort(bright), np.sort(sample["image"][0][sample["mask"] == 1])
        )


@pytest.mark.parametrize("transform_class", [aug.CustomFlip, aug.CustomRotate90])
def test_geometric_transforms_with_zero_probability_are_identity(
    sample, transform_class
):
    out = transform_class(keys=["image", "mask"], prob=0.0)(sample)
    np.testing.assert_array_equal(out["image"], sample["image"])


@pytest.mark.parametrize(
    "normalization_type, kwargs",
    [
        ("data_range", {}),
        ("mean_std", {"mean": 0.0, "std": 1.0}),
        ("percentile", {"lo": 1, "hi": 99}),
    ],
)
@pytest.mark.parametrize(
    "factory",
    [
        aug.get_training_augmentation,
        aug.get_qc_training_augmentation,
        aug.get_prediction_augmentation,
    ],
)
def test_pipelines_keep_image_and_mask_shapes(
    sample, factory, normalization_type, kwargs
):
    out = factory(normalization_type, **kwargs)(sample)
    assert out["image"].shape in {(2, 8, 12), (2, 12, 8)}
    assert out["mask"].shape in {(8, 12), (12, 8)}


def test_unknown_normalization_type_raises():
    with pytest.raises(ValueError):
        aug.get_prediction_augmentation("zscore")


@pytest.mark.parametrize(
    "factory",
    [
        aug.get_training_augmentation,
        aug.get_qc_training_augmentation,
        aug.get_prediction_augmentation,
    ],
)
def test_pipelines_enforce_channels(sample, factory):
    pipeline = factory("data_range", enforce_n_channels=3)
    assert pipeline(sample)["image"].shape[0] == 3


def test_prediction_augmentation_from_model_uses_stored_normalization(sample):
    model = SimpleNamespace(normalization={"type": "mean_std", "mean": 1.0, "std": 2.0})
    out = aug.get_prediction_augmentation_from_model(model)(sample)
    np.testing.assert_allclose(out["image"], (sample["image"] - 1) / 2)


def test_get_mean_and_std_uses_channel_two(tmp_path):
    image = np.zeros((3, 4, 4), dtype=np.uint16)
    image[2] = [[1, 3, 1, 3]] * 4
    path = tmp_path / "image.tiff"
    imwrite(path, image)
    assert aug.get_mean_and_std(str(path)) == (2.0, 1.0)
