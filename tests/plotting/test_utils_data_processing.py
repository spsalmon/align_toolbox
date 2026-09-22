import numpy as np
import pytest

from align_toolbox.plotting import utils_data_processing as udp


@pytest.fixture
def small_struct():
    return [{"a": np.array([1.0, 4.0]), "b": np.array([2.0, 2.0])}]


@pytest.mark.parametrize(
    "operation, expected",
    [
        ("add", [3, 6]),
        ("subtract", [-1, 2]),
        ("multiply", [2, 8]),
        ("divide", [0.5, 2]),
    ],
)
def test_combine_series(small_struct, operation, expected):
    struct = udp.combine_series(small_struct, "a", "b", operation, "c")
    np.testing.assert_allclose(struct[0]["c"], expected)


@pytest.mark.parametrize(
    "operation, expected",
    [("log", np.log([1, 4])), ("exp", np.exp([1, 4])), ("sqrt", [1, 2])],
)
def test_transform_series(small_struct, operation, expected):
    struct = udp.transform_series(small_struct, "a", operation, "t")
    np.testing.assert_allclose(struct[0]["t"], expected)


def test_exclude_arrests_single_worm():
    filtered = udp.exclude_arrests_from_series_at_ecdysis(
        np.array([1.0, 2.0, np.nan, np.nan, 5.0])
    )
    np.testing.assert_array_equal(filtered, [1, np.nan, np.nan, np.nan, 5])


def test_exclude_arrests_multiple_worms():
    values = np.array([[1.0, 2.0, 3.0], [1.0, np.nan, np.nan]])
    filtered = udp.exclude_arrests_from_series_at_ecdysis(values)
    np.testing.assert_array_equal(filtered, [[1, 2, 3], [np.nan, np.nan, np.nan]])


def test_detrend_rescaled_series_population_mean_ignores_nans():
    struct = [{"s": np.array([[1.0, 2.0], [3.0, np.nan]])}]
    udp.detrend_rescaled_series_population_mean(struct, "s")
    np.testing.assert_array_equal(struct[0]["s_detrended"], [[-1, 0], [1, np.nan]])


def test_smooth_series_stores_per_worm_smoothed_series(conditions_struct):
    struct = udp.smooth_series(conditions_struct, "body_seg_volume", "smoothed")
    for condition in struct:
        assert condition["smoothed"].shape == condition["body_seg_volume"].shape
        # the median filter flattens the last frames of a growing series
        np.testing.assert_allclose(
            condition["smoothed"][:, 2:-2],
            condition["body_seg_volume"][:, 2:-2],
            rtol=0.02,
        )


def test_compute_growth_rate_matches_derivative_of_exponential(conditions_struct):
    struct = udp.compute_growth_rate(
        conditions_struct, "body_seg_volume", "growth_rate", experiment_time=False
    )
    condition = struct[0]
    expected = 0.03 * condition["body_seg_volume"]
    np.testing.assert_allclose(
        condition["growth_rate"][:, 10:-10], expected[:, 10:-10], rtol=0.05
    )


@pytest.mark.parametrize("function", [udp.rescale, udp.rescale_without_flattening])
def test_rescale_shapes(conditions_struct, function):
    struct = function(conditions_struct, "body_seg_volume", "rescaled", n_points=10)
    expected_shape = (30, 40) if function is udp.rescale else (30, 4, 10)
    assert struct[0]["rescaled"].shape == expected_shape
    rescaled = struct[0]["rescaled"].reshape(30, 4, 10)
    # each stage starts where the previous one ended (shared ecdysis frame)
    np.testing.assert_allclose(rescaled[:, 1, 0], rescaled[:, 0, -1])


def test_rescale_picks_qc_column_matching_the_series(conditions_struct):
    condition = conditions_struct[0]
    condition["pharynx_seg_qc"] = np.full_like(condition["body_seg_qc"], "error")
    struct = udp.rescale(conditions_struct[:1], "body_seg_volume", "r", n_points=5)
    assert np.isfinite(struct[0]["r"]).all()


@pytest.mark.parametrize("flatten, shape", [(True, (30, 40)), (False, (30, 4, 10))])
def test_smooth_and_rescale_series(conditions_struct, flatten, shape):
    struct = udp.smooth_and_rescale_series(
        conditions_struct,
        "body_seg_volume",
        "volume_smoothed",
        experiment_time=False,
        n_points=10,
        flatten=flatten,
    )
    assert struct[0]["volume_smoothed"].shape == (30, 100)
    assert struct[0]["volume_smoothed_rescaled"].shape == shape
