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
    model = proportions._get_proportion_model(
        x, y, plot_model=False, remove_outliers=False
    )
    log_x = np.log(np.array([20.0, 500.0]))
    np.testing.assert_allclose(
        model.predict(log_x.reshape(-1, 1)), np.log(2) + 0.5 * log_x
    )
    prediction, lower, upper = model.get_confidence_intervals(log_x)
    assert (lower <= prediction).all() and (prediction <= upper).all()


def test_proportion_model_plot_and_outlier_removal(power_law):
    x, y = power_law
    y = y.copy()
    y[0, :] *= 50
    model = proportions._get_proportion_model(x, y, poly_degree=1)
    log_x = np.log(np.array([[100.0]]))
    assert model.predict(log_x)[0] == pytest.approx(np.log(20), abs=0.1)


def test_continuous_proportion_model_follows_the_data(power_law):
    x, y = power_law
    model = proportions._get_continuous_proportion_model(x, y)
    log_x = np.log(np.array([50.0, 500.0]))
    np.testing.assert_allclose(model(log_x), np.log(2) + 0.5 * log_x, atol=0.01)


def test_get_deviation_from_model(power_law):
    x, y = power_law
    model = proportions._get_proportion_model(
        x, y, plot_model=False, remove_outliers=False
    )
    y_shifted = y * 1.1
    y_shifted[0, 0] = np.nan
    y_shifted[:, 4] = np.nan
    deviations = proportions.get_deviation_from_model(x, y_shifted, model)
    assert deviations.shape == (40, 5)
    np.testing.assert_allclose(deviations[1:, :4], 10, atol=1e-6)
    assert np.isnan(deviations[0, 0]) and np.isnan(deviations[:, 4]).all()
    fraction = proportions.get_deviation_from_model(
        x, y_shifted, model, percentage=False
    )
    np.testing.assert_allclose(fraction[1:, :4], 0.1, atol=1e-8)


def test_compute_deviation_from_control_model(conditions_struct):
    struct = proportions.compute_deviation_from_model_at_ecdysis(
        conditions_struct,
        *COLUMNS,
        control_condition=0,
        output_column_name="dev",
        remove_outliers_fitting=False,
    )
    assert struct[0]["dev"].shape == (30, 4)
    np.testing.assert_allclose(struct[0]["dev"], 0, atol=1e-6)
    np.testing.assert_allclose(struct[1]["dev"], 10, atol=1e-6)


def test_compute_deviation_from_each_model_is_zero_for_exact_data(conditions_struct):
    struct = proportions.compute_deviation_from_each_model_at_ecdysis(
        conditions_struct,
        *COLUMNS,
        output_column_name="dev",
        remove_hatch=False,
        remove_outliers_fitting=False,
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
        remove_outliers_fitting=False,
    )
    assert struct[1]["dev"].shape == (30, 3)
    np.testing.assert_allclose(struct[1]["dev"], 10, atol=1e-4)


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
        remove_outliers_fitting=False,
    )
    # the control model diagnostic is shown first, then the deviation plot
    assert shown_figures[-1] is fig and len(shown_figures) == 2
    control, longer = _mean_lines(fig)
    np.testing.assert_allclose(control.get_ydata(), 0, atol=1e-6)
    np.testing.assert_allclose(longer.get_ydata(), 10, atol=1e-6)


def test_plot_deviation_from_model_development_percentage(conditions_struct):
    fig = proportions.plot_deviation_from_model_development_percentage(
        conditions_struct,
        "body_seg_volume",
        "body_seg_length",
        0,
        [0, 1],
        percentages=np.array([0.25, 0.5, 0.75]),
        log_scale=False,
        remove_outliers_fitting=False,
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
