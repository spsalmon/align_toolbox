import numpy as np
import pytest

from align_toolbox.plotting import proportions

COLUMNS = ("body_seg_volume_at_ecdysis", "body_seg_length_at_ecdysis")


@pytest.fixture
def power_law():
    """Five molts of 40 worms where y = 2 * x ** 0.5 exactly."""
    rng = np.random.default_rng(0)
    x = rng.uniform(10, 1000, (40, 5))
    return x, 2 * x**0.5


def _mean_lines(fig):
    """Solid lines only; error bars are drawn with linestyle 'None'."""
    return [line for line in fig.axes[0].get_lines() if line.get_linestyle() == "-"]


def test_proportion_model_recovers_log_log_line(power_law):
    x, y = power_law
    model = proportions._get_proportion_model(x, y, plot_model=False)
    log_x = np.log(np.array([20.0, 500.0]))
    np.testing.assert_allclose(
        model.predict(log_x.reshape(-1, 1)), np.log(2) + 0.5 * log_x
    )
    prediction, lower, upper = model.get_confidence_intervals(log_x)
    assert (lower <= prediction).all() and (prediction <= upper).all()


def test_continuous_proportion_model_follows_the_data(power_law):
    x, y = power_law
    model = proportions._get_continuous_proportion_model(x, y)
    log_x = np.log(np.array([50.0, 500.0]))
    np.testing.assert_allclose(model(log_x), np.log(2) + 0.5 * log_x, atol=0.01)


def test_get_deviation_from_model(power_law):
    x, y = power_law
    model = proportions._get_proportion_model(x, y, plot_model=False)
    y_shifted = y * 1.1
    y_shifted[0, 0] = np.nan
    y_shifted[:, 4] = np.nan
    deviations = proportions.get_deviation_from_model(x, y_shifted, model)
    assert deviations.shape == (40, 5)
    np.testing.assert_allclose(deviations[1:, :4], np.log(1.1), atol=1e-8)
    assert np.isnan(deviations[0, 0]) and np.isnan(deviations[:, 4]).all()
    percent = proportions.get_deviation_from_model(
        x, y_shifted, model, deviation_scale="percent"
    )
    np.testing.assert_allclose(percent[1:, :4], 10, atol=1e-6)
    assert np.isnan(percent[0, 0]) and np.isnan(percent[:, 4]).all()


def test_log_deviations_are_symmetric(power_law):
    x, y = power_law
    model = proportions._get_proportion_model(x, y, plot_model=False)
    doubled = proportions.get_deviation_from_model(x, y * 2, model)
    halved = proportions.get_deviation_from_model(x, y * 0.5, model)
    np.testing.assert_allclose(doubled, np.log(2), atol=1e-8)
    np.testing.assert_allclose(halved, -doubled, atol=1e-8)


def test_deprecated_percentage_flag_reproduces_percent_output(power_law):
    x, y = power_law
    model = proportions._get_proportion_model(x, y, plot_model=False)
    with pytest.warns(DeprecationWarning, match="percentage"):
        deviations = proportions.get_deviation_from_model(
            x, y * 1.1, model, percentage=True
        )
    np.testing.assert_allclose(deviations, 10, atol=1e-6)


def test_compute_deviation_from_control_model(conditions_struct):
    struct = proportions.compute_deviation_from_model_at_ecdysis(
        conditions_struct,
        *COLUMNS,
        control_condition=0,
        output_column_name="dev",
    )
    assert struct[0]["dev"].shape == (30, 4)
    np.testing.assert_allclose(struct[0]["dev"], 0, atol=1e-6)
    np.testing.assert_allclose(struct[1]["dev"], np.log(1.1), atol=1e-8)


def test_compute_deviation_from_each_model_is_zero_for_exact_data(conditions_struct):
    struct = proportions.compute_deviation_from_each_model_at_ecdysis(
        conditions_struct,
        *COLUMNS,
        output_column_name="dev",
        remove_hatch=False,
    )
    assert struct[1]["dev"].shape == (30, 5)
    np.testing.assert_allclose(struct[1]["dev"], 0, atol=1e-6)


def test_compute_deviation_development_percentage(conditions_struct):
    struct = proportions.compute_deviation_from_model_development_percentage(
        conditions_struct,
        "body_seg_volume",
        "body_seg_length",
        control_condition=0,
        percentages=np.array([0.1, 0.5, 0.9]),
        output_column_name="dev",
    )
    assert struct[1]["dev"].shape == (30, 3)
    np.testing.assert_allclose(struct[1]["dev"], np.log(1.1), atol=1e-6)


@pytest.mark.parametrize("single_plot", [True, False])
def test_plot_model_comparison_at_ecdysis(conditions_struct, single_plot):
    fig = proportions.plot_model_comparison_at_ecdysis(
        conditions_struct, *COLUMNS, [0, 1], single_plot=single_plot
    )
    assert len(fig.axes) == (1 if single_plot else 2)


def test_plot_correlation_functions(conditions_struct):
    fig = proportions.plot_correlation(
        conditions_struct, "body_seg_volume", "body_seg_length", [0, 1]
    )
    assert len(fig.axes[0].get_lines()) == 2
    fig = proportions.plot_correlation_at_ecdysis(
        conditions_struct, *COLUMNS, [0, 1], x_axis_label="V", y_axis_label="L"
    )
    assert (fig.axes[0].get_xlabel(), fig.axes[0].get_ylabel()) == ("V", "L")


def test_plot_deviation_from_model_at_ecdysis_shows_ten_percent(
    conditions_struct, shown_figures
):
    fig = proportions.plot_deviation_from_model_at_ecdysis(
        conditions_struct,
        *COLUMNS,
        0,
        [0, 1],
        log_scale=False,
    )
    # the control model diagnostic is shown first, then the deviation plot
    assert shown_figures[-1] is fig and len(shown_figures) == 2
    control, longer = _mean_lines(fig)
    np.testing.assert_allclose(control.get_ydata(), 0, atol=1e-6)
    np.testing.assert_allclose(longer.get_ydata(), 10, atol=1e-6)


def test_plot_deviation_percent_summary_is_back_transformed_log_mean(
    conditions_struct,
):
    rng = np.random.default_rng(1)
    noisy = conditions_struct[1]
    noisy["body_seg_length_at_ecdysis"] = noisy[
        "body_seg_length_at_ecdysis"
    ] * rng.lognormal(0, 0.3, noisy["body_seg_length_at_ecdysis"].shape)
    log_deviations = proportions.compute_deviation_from_model_at_ecdysis(
        conditions_struct,
        *COLUMNS,
        control_condition=0,
        output_column_name="dev",
        remove_hatch=False,
    )[1]["dev"]
    log_mean = log_deviations.mean(axis=0)
    log_ste = log_deviations.std(axis=0) / np.sqrt(log_deviations.shape[0])

    fig = proportions.plot_deviation_from_model_at_ecdysis(
        conditions_struct, *COLUMNS, 0, [1], log_scale=False
    )
    (line,) = _mean_lines(fig)
    np.testing.assert_allclose(line.get_ydata(), np.expm1(log_mean) * 100)
    (error_bars,) = fig.axes[0].collections
    lower, upper = np.array([segment[:, 1] for segment in error_bars.get_segments()]).T
    np.testing.assert_allclose(lower, np.expm1(log_mean - log_ste) * 100)
    np.testing.assert_allclose(upper, np.expm1(log_mean + log_ste) * 100)
    # exp is convex, so the upper arm is always the longer one
    center = line.get_ydata()
    assert (upper - center > center - lower).all()
    assert fig.axes[0].get_ylabel().endswith("(%)")


def test_plot_deviation_from_model_development_percentage(conditions_struct):
    fig = proportions.plot_deviation_from_model_development_percentage(
        conditions_struct,
        "body_seg_volume",
        "body_seg_length",
        0,
        [0, 1],
        percentages=np.array([0.25, 0.5, 0.75]),
        log_scale=False,
    )
    control, longer = _mean_lines(fig)
    assert len(control.get_xdata()) == 3
    np.testing.assert_allclose(control.get_ydata(), 0, atol=1e-4)
    np.testing.assert_allclose(longer.get_ydata(), 10, atol=1e-4)


def test_plot_continuous_deviation_from_model(conditions_struct):
    fig = proportions.plot_continuous_deviation_from_model(
        conditions_struct,
        "body_seg_volume",
        "body_seg_length",
        0,
        [0, 1],
        log_scale=False,
    )
    control, longer = fig.axes[0].get_lines()
    np.testing.assert_allclose(control.get_ydata(), 0, atol=0.5)
    np.testing.assert_allclose(longer.get_ydata(), 10, atol=0.5)


def test_plot_normalized_proportions_at_ecdysis(conditions_struct):
    fig = proportions.plot_normalized_proportions_at_ecdysis(
        conditions_struct, *COLUMNS, 0, [0, 1], log_scale=False
    )
    control, longer = _mean_lines(fig)
    np.testing.assert_allclose(control.get_ydata(), 1)
    # worm sizes differ between conditions, so the ratio is only close to 1.1
    np.testing.assert_allclose(longer.get_ydata(), 1.1, rtol=0.05)
