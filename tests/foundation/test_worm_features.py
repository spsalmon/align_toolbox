import numpy as np
import pytest
from skimage.measure import label

from align_toolbox.foundation import worm_features
from align_toolbox.straightening import Warper

PIXELSIZE = 0.5


@pytest.fixture
def straightened_curved_worm(curved_worm_mask):
    warper = Warper.from_img(curved_worm_mask, curved_worm_mask)
    return warper.warp_2D_img(
        curved_worm_mask,
        0,
        interpolation_order=0,
        preserve_range=True,
        preserve_dtype=True,
    )


def test_feature_name_getters_return_module_constants():
    assert (
        worm_features.get_available_mask_features()
        == worm_features.AVAILABLE_MASK_FEATURES
    )
    assert (
        worm_features.get_features_to_compute_at_molt()
        == worm_features.FEATURES_TO_COMPUTE_AT_MOLT
    )


def test_regionprops_split_into_mask_and_intensity_properties():
    mask_props, image_props = worm_features.get_available_regionprops()
    assert "area" in mask_props and "area" not in image_props
    assert "intensity_mean" in image_props and "intensity_mean" not in mask_props


def test_rectangle_morphology_is_exact(rectangle_mask):
    assert worm_features.compute_mask_length(rectangle_mask, PIXELSIZE) == 50.0
    assert worm_features.compute_mask_area(rectangle_mask, PIXELSIZE) == 500.0
    # a cylinder of radius 10 px and length 100 px, scaled by pixelsize**3
    expected_volume = np.pi * 10**2 * 100 * PIXELSIZE**3
    assert worm_features.compute_mask_volume(
        rectangle_mask, PIXELSIZE
    ) == pytest.approx(expected_volume)


@pytest.mark.parametrize("aggregation", ["mean", "median"])
def test_compute_mask_average_width(rectangle_mask, aggregation):
    width = worm_features.compute_mask_average_width(
        rectangle_mask, PIXELSIZE, aggregation
    )
    assert width == pytest.approx(10.0)


def test_compute_mask_average_width_rejects_unknown_aggregation(rectangle_mask):
    with pytest.raises(ValueError):
        worm_features.compute_mask_average_width(rectangle_mask, PIXELSIZE, "max")


def test_compute_width_profile_ignores_empty_columns(rectangle_mask):
    profile = worm_features.compute_width_profile(rectangle_mask, 1.0, smooth=False)
    np.testing.assert_array_equal(profile, np.full(100, 20.0))


def test_compute_width_profile_falls_back_when_too_short_to_smooth():
    short = np.ones((4, 5), dtype=np.uint8)
    profile = worm_features.compute_width_profile(short, 1.0, savgol_window=21)
    np.testing.assert_array_equal(profile, np.full(5, 4.0))


def test_compute_max_width_averages_around_peak():
    profile = np.array([1, 1, 1, 1, 2, 4, 2, 1, 1, 1, 1], dtype=float)
    assert worm_features.compute_max_width(profile, window_size=1) == pytest.approx(
        8 / 3
    )


def test_compute_max_width_peak_at_start_of_profile():
    profile = np.array([5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    assert worm_features.compute_max_width(profile, window_size=2) == pytest.approx(4)


def test_compute_mid_width_averages_around_center():
    profile = np.array([1, 1, 3, 5, 7, 1, 1], dtype=float)
    assert worm_features.compute_mid_width(profile, window_size=1) == pytest.approx(5)


def test_curved_worm_morphology_matches_geometry(
    straightened_curved_worm, curved_worm_mask, curved_worm_expected_length
):
    features = worm_features.compute_mask_morphological_features(
        straightened_curved_worm,
        pixelsize=1.0,
        features=["length", "area", "volume", "width_median", "width_middle"],
    )
    assert features["length"] == pytest.approx(curved_worm_expected_length, rel=0.05)
    assert features["area"] == pytest.approx(curved_worm_mask.sum(), rel=0.02)
    # the band is 21 px thick; resampling along the curve adds up to ~2 px
    assert features["width_median"] == pytest.approx(21, abs=3)
    assert features["width_middle"] == pytest.approx(21, abs=3)
    # the ~1 px resampling overestimate of the width inflates the volume by ~13%
    cylinder_volume = np.pi * 10.5**2 * curved_worm_expected_length
    assert features["volume"] == pytest.approx(cylinder_volume, rel=0.2)


def test_morphological_features_computes_every_available_feature(
    straightened_curved_worm,
):
    features = worm_features.compute_mask_morphological_features(
        straightened_curved_worm, 1.0, worm_features.AVAILABLE_MASK_FEATURES
    )
    assert list(features) == worm_features.AVAILABLE_MASK_FEATURES
    assert all(np.isfinite(value) for value in features.values())


def test_morphological_features_empty_mask_gives_nan_instead_of_zero():
    features = worm_features.compute_mask_morphological_features(
        np.zeros((10, 10), dtype=np.uint8), 1.0, ["length", "area", "volume"]
    )
    assert all(np.isnan(value) for value in features.values())


def test_morphological_features_rejects_unknown_feature(rectangle_mask):
    with pytest.raises(ValueError, match="girth"):
        worm_features.compute_mask_morphological_features(
            rectangle_mask, 1.0, ["girth"]
        )


def test_compute_image_features_returns_requested_keys(rng):
    image = rng.random((32, 32))
    features = worm_features.compute_image_features(
        image, worm_features.AVAILABLE_IMAGE_FEATURES
    )
    assert list(features) == worm_features.AVAILABLE_IMAGE_FEATURES
    with pytest.raises(ValueError):
        worm_features.compute_image_features(image, ["sharpness"])


def test_compute_mask_type_features_on_rectangle(rectangle_mask):
    (
        length,
        volume,
        volume_per_length,
        width_mean,
        width_std,
        width_cv,
        entropy,
        eccentricity,
        solidity,
        perimeter,
    ) = worm_features.compute_mask_type_features(rectangle_mask, 1.0)
    assert (length, width_mean, width_std, width_cv) == (100, 20, 0, 0)
    assert volume_per_length == pytest.approx(np.pi * 100)
    assert entropy > 0
    assert solidity == pytest.approx(1.0)
    assert 0 < eccentricity < 1
    assert perimeter > 0


def test_intensity_statistics_use_only_region_pixels():
    intensity = np.array([[1.0, 2.0], [3.0, 100.0]])
    region = np.array([[True, True], [True, False]])
    assert worm_features.intensity_std(region, intensity) == pytest.approx(
        np.std([1, 2, 3])
    )
    assert worm_features.intensity_skew(region, intensity) == pytest.approx(0.0)
    assert worm_features.intensity_kurtosis(region, intensity) == pytest.approx(-1.5)


def test_compute_haralick_features_uniform_image():
    (
        contrast,
        dissimilarity,
        homogeneity,
        energy,
        _,
    ) = worm_features.compute_haralick_features(np.full((16, 16), 0.5))
    assert (contrast, dissimilarity, homogeneity, energy) == (0, 0, 1, 1)


@pytest.mark.parametrize("center", [(5, 5), (50, 50), (94, 94)])
def test_compute_patch_features_handles_regions_near_borders(rng, center):
    intensity = rng.random((100, 100))
    region = np.zeros((100, 100), dtype=np.uint8)
    region[center[0] - 2 : center[0] + 3, center[1] - 2 : center[1] + 3] = 1
    features = worm_features.compute_patch_features(region, intensity, patch_size=16)
    assert len(features) == 12
    assert all(np.isfinite(features))


@pytest.fixture
def labeled_blobs():
    """Five 3x3 blobs on a line; blob k has intensity k."""
    labels = np.zeros((10, 60), dtype=np.uint8)
    intensity = np.zeros((10, 60))
    for k in range(1, 6):
        labels[4:7, 10 * k : 10 * k + 3] = k
        intensity[4:7, 10 * k : 10 * k + 3] = k
    return labels, intensity


def test_compute_base_label_features_returns_one_value_per_feature(labeled_blobs):
    labels, intensity = labeled_blobs
    mask_of_label = (labels == 3).astype(np.uint8)
    features = worm_features.compute_base_label_features(
        mask_of_label,
        intensity,
        ["area", "intensity_mean"],
        [worm_features.intensity_std],
    )
    assert features == [9, 3, 0]


def test_get_context_returns_nearest_other_labels(labeled_blobs):
    labels, _ = labeled_blobs
    current = (labels == 1).astype(np.uint8)
    context = worm_features.get_context(1, current, labels, num_closest=2)
    assert set(np.unique(context)) == {0, 2, 3}


def test_get_context_minus_one_returns_all_other_labels(labeled_blobs):
    labels, _ = labeled_blobs
    context = worm_features.get_context(
        1, (labels == 1).astype(np.uint8), labels, num_closest=-1
    )
    assert set(np.unique(context)) == {0, 2, 3, 4, 5}


def test_get_context_features_aggregates_mean_and_std(labeled_blobs):
    labels, intensity = labeled_blobs
    features = worm_features.get_context_features(
        label(labels > 0), intensity, ["intensity_mean"], []
    )
    assert features == pytest.approx([3.0, np.std([1, 2, 3, 4, 5])])


def test_bending_energy_of_straight_line_is_near_zero():
    t = np.linspace(0, 100, 50)
    midline = np.stack([t, np.zeros_like(t)], axis=1)
    energy = worm_features.compute_bending_energy(midline, np.full(50, 10.0))
    assert energy == pytest.approx(0.0, abs=1e-6)


def test_bending_energy_grows_with_curvature():
    theta_small = np.linspace(0, np.pi / 4, 50)
    theta_large = np.linspace(0, np.pi, 50)
    widths = np.full(50, 10.0)
    gentle = np.stack([100 * np.cos(theta_small), 100 * np.sin(theta_small)], axis=1)
    tight = np.stack([20 * np.cos(theta_large), 20 * np.sin(theta_large)], axis=1)
    assert worm_features.compute_bending_energy(
        tight, widths
    ) > worm_features.compute_bending_energy(gentle, widths)


def test_bending_energy_mask_ranks_curved_worm_above_straight(
    curved_worm_mask, rectangle_mask
):
    curved = worm_features.compute_bending_energy_mask(curved_worm_mask, 1.0)
    straight = worm_features.compute_bending_energy_mask(rectangle_mask, 1.0)
    assert curved > 10 * straight


def test_bending_energy_mask_returns_nan_on_failure():
    energy = worm_features.compute_bending_energy_mask(
        np.zeros((20, 20), dtype=np.uint8), 1.0
    )
    assert np.isnan(energy)
