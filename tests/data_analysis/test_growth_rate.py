import numpy as np
import pytest

from align_toolbox.data_analysis import growth_rate as gr

TIME = np.arange(100.0)
ALL_WORM = np.array(["worm"] * 100)


def test_linear_growth_rate_recovers_slope():
    assert gr.compute_growth_rate_linear(2 * TIME + 3, TIME) == pytest.approx(
        2, abs=0.01
    )


def test_exponential_growth_rate_recovers_rate():
    assert gr.compute_growth_rate_exponential(
        np.exp(0.05 * TIME), TIME
    ) == pytest.approx(0.05, abs=0.001)


@pytest.mark.parametrize(
    "function", [gr.compute_growth_rate_linear, gr.compute_growth_rate_exponential]
)
def test_growth_rate_ignores_requested_fractions(function):
    # flat plateaus at both ends hide the exponential middle part
    series = np.exp(0.05 * TIME)
    series[:20] = 50.0
    series[80:] = 50.0
    middle_only = function(
        series, TIME, ignore_start_fraction=0.25, ignore_end_fraction=0.25
    )
    whole = function(series, TIME)
    reference = function(np.exp(0.05 * TIME)[25:75], TIME[25:75])
    assert middle_only == pytest.approx(reference, rel=0.05)
    assert whole != pytest.approx(reference, rel=0.05)


@pytest.mark.parametrize(
    "function", [gr.compute_growth_rate_linear, gr.compute_growth_rate_exponential]
)
def test_growth_rate_rejects_ignoring_everything(function):
    with pytest.raises(AssertionError):
        function(TIME + 1, TIME, ignore_start_fraction=0.5, ignore_end_fraction=0.5)


@pytest.mark.parametrize(
    "method, series, expected",
    [
        ("exponential", np.exp(0.05 * TIME), 0.05),
        ("linear", 2 * TIME + 3, 2.0),
    ],
)
def test_classified_growth_rate_ignores_non_worm_points(method, series, expected):
    corrupted = series.copy()
    corrupted[[10, 11, 50]] = 1e9
    qc = ALL_WORM.copy()
    qc[[10, 11, 50]] = "error"
    rate = gr.compute_growth_rate_classified(corrupted, TIME, qc, method=method)
    assert rate == pytest.approx(expected, rel=0.01)


def test_instantaneous_growth_rate_of_line_is_its_slope():
    rate = gr.compute_instantaneous_growth_rate(2 * TIME + 3, TIME)
    assert rate.shape == TIME.shape
    np.testing.assert_allclose(rate[10:-10], 2, atol=0.05)


def test_instantaneous_growth_rate_classified_ignores_errors():
    series = 2 * TIME + 3
    series[40:43] = 1e6
    qc = ALL_WORM.copy()
    qc[40:43] = "error"
    rate = gr.compute_instantaneous_growth_rate_classified(series, TIME, qc)
    np.testing.assert_allclose(rate[10:-10], 2, atol=0.05)


@pytest.mark.parametrize(
    "function, args",
    [
        (gr.compute_instantaneous_growth_rate, (np.ones(5), np.ones(4))),
        (
            gr.compute_instantaneous_growth_rate_classified,
            (np.ones(5), np.ones(5), np.array(["worm"] * 4)),
        ),
    ],
)
def test_instantaneous_growth_rate_length_mismatch_raises(function, args):
    with pytest.raises(AssertionError):
        function(*args)


@pytest.fixture
def staged_growth():
    """Exponential growth whose rate doubles at each of four stages of 25 frames."""
    rates = np.repeat([0.01, 0.02, 0.04, 0.08], 25)
    series = np.exp(np.cumsum(rates))
    ecdysis = {"HatchTime": 0, "M1": 25, "M2": 50, "M3": 75, "M4": 99}
    return series, ecdysis


def test_growth_rate_per_larval_stage(staged_growth):
    series, ecdysis = staged_growth
    rates = gr.compute_growth_rate_per_larval_stage(series, TIME, ALL_WORM, ecdysis)
    assert list(rates) == ["L1", "L2", "L3", "L4"]
    for stage, expected in zip(rates, [0.01, 0.02, 0.04, 0.08]):
        assert rates[stage] == pytest.approx(expected, rel=0.15)


def test_growth_rate_per_larval_stage_missing_molt_gives_nan(staged_growth):
    series, ecdysis = staged_growth
    ecdysis = {**ecdysis, "M2": np.nan}
    rates = gr.compute_growth_rate_per_larval_stage(series, TIME, ALL_WORM, ecdysis)
    assert np.isnan(rates["L2"]) and np.isnan(rates["L3"])
    assert np.isfinite(rates["L1"]) and np.isfinite(rates["L4"])


def test_growth_rate_per_larval_stage_accepts_float_indices(staged_growth):
    series, ecdysis = staged_growth
    ecdysis = {key: float(value) for key, value in ecdysis.items()}
    rates = gr.compute_growth_rate_per_larval_stage(series, TIME, ALL_WORM, ecdysis)
    assert rates["L1"] == pytest.approx(0.01, rel=0.15)


def test_growth_rate_per_larval_stage_accepts_find_molts_keys(staged_growth):
    series, ecdysis = staged_growth
    # find_molts returns "hatch_time", filemaps use "HatchTime"
    ecdysis = {
        "hatch_time": float(ecdysis["HatchTime"]),
        **{k: float(v) for k, v in ecdysis.items() if k != "HatchTime"},
    }
    rates = gr.compute_growth_rate_per_larval_stage(series, TIME, ALL_WORM, ecdysis)
    assert rates["L1"] == pytest.approx(0.01, rel=0.15)


def test_compute_larval_stage_duration():
    durations = gr.compute_larval_stage_duration(
        {"HatchTime": 10, "M1": 30, "M2": np.nan, "M3": 80, "M4": 110}
    )
    assert durations["L1"] == 20
    assert np.isnan(durations["L2"]) and np.isnan(durations["L3"])
    assert durations["L4"] == 30
