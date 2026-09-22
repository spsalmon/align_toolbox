import numpy as np
import pytest

from align_toolbox.data_analysis import time_series as ts

TIME = np.arange(100.0)
LINEAR = 2 * TIME + 3
ECDYSIS = np.array([0.0, 25.0, 50.0, 75.0, 99.0])
ALL_WORM = np.array(["worm"] * 100)


@pytest.mark.parametrize(
    "function, args",
    [
        (ts.resize_series_to_length, (5,)),
        (ts.pad_series_to_length, (5,)),
        (ts.crop_series_to_length, (5,)),
        (ts.random_crop_series_to_length, (5,)),
    ],
)
def test_series_resizers_reject_3d_input(function, args):
    with pytest.raises(ValueError):
        function(np.zeros((2, 2, 4)), *args)


@pytest.mark.parametrize(
    "function",
    [
        ts.resize_series_to_length,
        ts.pad_series_to_length,
        ts.crop_series_to_length,
        ts.random_crop_series_to_length,
    ],
)
def test_series_resizers_return_input_when_length_matches(function):
    series = np.arange(5.0)
    np.testing.assert_array_equal(function(series, 5), series)


def test_resize_series_to_length_interpolates_1d_and_2d():
    np.testing.assert_allclose(
        ts.resize_series_to_length(np.array([0.0, 10.0]), 3), [0, 5, 10]
    )
    resized = ts.resize_series_to_length(np.array([[0.0, 10.0], [0.0, 20.0]]), 5)
    np.testing.assert_allclose(resized, [[0, 2.5, 5, 7.5, 10], [0, 5, 10, 15, 20]])


def test_pad_series_to_length():
    np.testing.assert_array_equal(
        ts.pad_series_to_length(np.array([1.0, 2.0]), 4, pad_value=-1), [1, 2, -1, -1]
    )
    assert ts.pad_series_to_length(np.ones((3, 2)), 5).shape == (3, 5)


def test_crop_series_to_length_keeps_start():
    np.testing.assert_array_equal(ts.crop_series_to_length(np.arange(5), 2), [0, 1])
    np.testing.assert_array_equal(
        ts.crop_series_to_length(np.arange(10).reshape(2, 5), 2), [[0, 1], [5, 6]]
    )


def test_random_crop_series_to_length_returns_contiguous_window():
    series = np.arange(20)
    for _ in range(20):
        window = ts.random_crop_series_to_length(series, 5)
        assert len(window) == 5
        np.testing.assert_array_equal(np.diff(window), 1)
    assert ts.random_crop_series_to_length(np.ones((3, 20)), 5).shape == (3, 5)


def test_filter_and_correct_series_with_classification():
    series = np.array([1.0, 100.0, 3.0, 4.0])
    qc = np.array(["worm", "error", "worm", "worm"])
    filtered = ts.filter_series_with_classification(series, qc)
    np.testing.assert_array_equal(filtered, [1, np.nan, 3, 4])
    corrected = ts.correct_series_with_classification(series, qc)
    np.testing.assert_array_equal(corrected, [1, 2, 3, 4])
    assert series[1] == 100.0  # input untouched


def test_correct_series_with_classification_no_worms_gives_all_nan():
    corrected = ts.correct_series_with_classification(
        np.ones(3), np.array(["egg", "egg", "error"])
    )
    assert np.isnan(corrected).all()


def test_interpolate_larval_stage_spans_the_stage():
    interpolated_time, interpolated_series = ts.interpolate_larval_stage(
        LINEAR, TIME, ECDYSIS, larval_stage=2, n_points=5
    )
    np.testing.assert_allclose(interpolated_time, [25, 31.25, 37.5, 43.75, 50])
    np.testing.assert_allclose(interpolated_series, 2 * interpolated_time + 3)


def test_interpolate_larval_stage_skips_non_worm_points():
    series = LINEAR.copy()
    series[30:40] = 1e6
    qc = ALL_WORM.copy()
    qc[30:40] = "error"
    _, interpolated = ts.interpolate_larval_stage(
        series, TIME, ECDYSIS, larval_stage=2, qc=qc, n_points=5
    )
    np.testing.assert_allclose(interpolated, [53, 65.5, 78, 90.5, 103])


@pytest.mark.parametrize(
    "ecdysis, larval_stage",
    [
        (np.array([np.nan, 25, 50, 75, 99.0]), 1),
        (np.array([0, 60, 50, 75, 99.0]), 2),
    ],
    ids=["missing-bound", "reversed-bounds"],
)
def test_interpolate_larval_stage_invalid_bounds_give_nan(ecdysis, larval_stage):
    interpolated_time, interpolated_series = ts.interpolate_larval_stage(
        LINEAR, TIME, ecdysis, larval_stage=larval_stage, n_points=3
    )
    assert np.isnan(interpolated_time).all() and np.isnan(interpolated_series).all()


@pytest.mark.parametrize("larval_stage", [0, 5])
def test_interpolate_larval_stage_rejects_invalid_stage(larval_stage):
    with pytest.raises(ValueError):
        ts.interpolate_larval_stage(LINEAR, TIME, ECDYSIS, larval_stage)


def test_interpolate_entire_development_stacks_the_four_stages():
    interpolated_time, interpolated_series = ts.interpolate_entire_development(
        LINEAR, TIME, np.array([0.0, 25.0, np.nan, 75.0, 99.0]), n_points=4
    )
    assert interpolated_time.shape == interpolated_series.shape == (4, 4)
    np.testing.assert_allclose(interpolated_time[0], [0, 25 / 3, 50 / 3, 25])
    assert np.isnan(interpolated_series[1:3]).all()
    np.testing.assert_allclose(interpolated_series[3], 2 * interpolated_time[3] + 3)


def test_compute_exponential_series_at_time_classified():
    series = np.exp(0.05 * TIME)
    result = ts.compute_exponential_series_at_time_classified(
        series, np.array([30.0, np.nan]), ALL_WORM
    )
    assert result[0] == pytest.approx(np.exp(1.5))
    assert np.isnan(result[1])


def test_compute_exponential_series_at_time_without_worms_is_nan():
    qc = ALL_WORM.copy()
    qc[20:40] = "egg"
    result = ts.compute_exponential_series_at_time_classified(
        np.exp(0.05 * TIME), np.array([30.0]), qc, fit_width=5
    )
    assert np.isnan(result[0])


@pytest.mark.parametrize("series_time", [TIME, None], ids=["time", "indices"])
def test_smooth_series_reduces_noise(rng, series_time):
    clean = np.sin(TIME / 10)
    noisy = clean + rng.normal(0, 0.1, TIME.shape)
    # the default lambda is tuned for time in hours; frames need more smoothing
    smoothed = ts.smooth_series(noisy, series_time, lmbda=10)
    assert smoothed.shape == noisy.shape
    assert np.abs(smoothed - clean).mean() < np.abs(noisy - clean).mean() / 2


def test_smooth_series_handles_nan_time_by_padding():
    time = TIME.copy()
    time[-5:] = np.nan
    smoothed = ts.smooth_series(LINEAR.copy(), time)
    assert smoothed.shape == LINEAR.shape
    assert np.isfinite(smoothed).all()


@pytest.mark.parametrize("function", ["smooth_series", "smooth_series_classified"])
def test_smoothing_all_nan_returns_all_nan(function):
    args = (np.full(10, np.nan), None)
    if function == "smooth_series_classified":
        args += (np.array(["worm"] * 10),)
    assert np.isnan(getattr(ts, function)(*args)).all()


def test_smooth_series_classified_ignores_non_worm_outliers():
    series = LINEAR.copy()
    series[40:43] = 1e6
    qc = ALL_WORM.copy()
    qc[40:43] = "error"
    smoothed = ts.smooth_series_classified(series, TIME, qc)
    np.testing.assert_allclose(smoothed[5:-5], LINEAR[5:-5], atol=0.5)


def test_compute_series_at_time_classified_evaluates_smoothed_series():
    qc = ALL_WORM.copy()
    qc[[10, 11, 50]] = "error"
    result = ts.compute_series_at_time_classified(
        LINEAR, np.array([10.5, 20.0]), TIME, qc
    )
    np.testing.assert_allclose(result, [24, 43], atol=0.1)


def test_compute_series_at_time_classified_all_nan():
    result = ts.compute_series_at_time_classified(
        np.full(100, np.nan), np.array([1.0, 2.0]), TIME, ALL_WORM
    )
    assert result.shape == (2,) and np.isnan(result).all()


@pytest.fixture
def two_worms():
    series = np.stack([2 * TIME + 3, 3 * TIME])
    time = np.stack([TIME, TIME])
    ecdysis = np.stack([ECDYSIS, ECDYSIS])
    qc = np.stack([ALL_WORM, ALL_WORM])
    return series, time, ecdysis, qc


def test_rescale_series_selects_points(two_worms):
    series, time, ecdysis, qc = two_worms
    rescaled_time, rescaled = ts.rescale_series(
        series, time, ecdysis, qc, points=[1], n_points=5
    )
    assert rescaled_time.shape == rescaled.shape == (1, 4, 5)
    np.testing.assert_allclose(rescaled[0, 0], 3 * np.linspace(0, 25, 5))


def test_rescale_and_aggregate_means_and_rescaled_time(two_worms):
    series, time, ecdysis, qc = two_worms
    durations = np.diff(ecdysis, axis=1)
    rescaled_time, mean, std, _ = ts.rescale_and_aggregate(
        series, time, ecdysis, durations, qc, n_points=5
    )
    assert rescaled_time.shape == mean.shape == std.shape == (20,)
    np.testing.assert_allclose(rescaled_time[:5], [0, 6.25, 12.5, 18.75, 25])
    np.testing.assert_allclose(rescaled_time[-5:], [75, 81, 87, 93, 99])
    expected_mean = (2 * rescaled_time + 3 + 3 * rescaled_time) / 2
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(std[:5], np.abs(3 - rescaled_time[:5]) / 2)


def test_rescale_and_aggregate_median(two_worms):
    series, time, ecdysis, qc = two_worms
    durations = np.diff(ecdysis, axis=1)
    _, median, _, _ = ts.rescale_and_aggregate(
        series, time, ecdysis, durations, qc, aggregation="median", n_points=5
    )
    _, mean, _, _ = ts.rescale_and_aggregate(
        series, time, ecdysis, durations, qc, n_points=5
    )
    # with two worms, median and mean coincide
    np.testing.assert_allclose(median, mean)


def test_aggregate_standard_error_uses_number_of_worms(two_worms):
    series, time, ecdysis, qc = two_worms
    durations = np.diff(ecdysis, axis=1)
    _, _, std, ste = ts.rescale_and_aggregate(
        series, time, ecdysis, durations, qc, n_points=5
    )
    np.testing.assert_allclose(ste, std / np.sqrt(2))
