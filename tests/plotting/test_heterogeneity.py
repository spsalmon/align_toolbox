import numpy as np
import pytest

from align_toolbox.plotting import heterogeneity


def _line_data(fig):
    return [line.get_ydata() for line in fig.axes[0].get_lines()]


@pytest.mark.parametrize(
    "function, statistic",
    [
        (heterogeneity.plot_cv_at_ecdysis, lambda v: v.std(0) / v.mean(0) * 100),
        (heterogeneity.plot_std_at_ecdysis, lambda v: v.std(0)),
    ],
)
@pytest.mark.parametrize("remove_hatch", [True, False])
def test_spread_at_ecdysis_plots_one_line_per_condition(
    conditions_struct, function, statistic, remove_hatch
):
    fig = function(
        conditions_struct,
        "body_seg_volume_at_ecdysis",
        [0, 1],
        remove_hatch=remove_hatch,
        legend={"description": ""},
    )
    lines = _line_data(fig)
    assert len(lines) == 2
    values = conditions_struct[1]["body_seg_volume_at_ecdysis"]
    if remove_hatch:
        values = values[:, 1:]
    np.testing.assert_allclose(lines[1], statistic(values))
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labels[0] == ("M1" if remove_hatch else "Hatch")


def test_cv_at_ecdysis_can_exclude_arrests(conditions_struct):
    conditions_struct[0]["body_seg_volume_at_ecdysis"][0, 3:] = np.nan
    fig = heterogeneity.plot_cv_at_ecdysis(
        conditions_struct,
        "body_seg_volume_at_ecdysis",
        [0],
        exclude_arrests=True,
        ax_size=(2, 2),
    )
    assert np.isfinite(_line_data(fig)[0]).all()


def test_cv_development_percentage_samples_requested_positions(conditions_struct):
    percentages = np.array([0.0, 0.5, 1.0])
    fig = heterogeneity.plot_cv_development_percentage(
        conditions_struct, "body_seg_volume", [0, 1], percentages
    )
    line = fig.axes[0].get_lines()[0]
    np.testing.assert_array_equal(line.get_xdata(), [0, 50, 100])
    values = conditions_struct[0]["body_seg_volume"][:, [0, 50, 99]]
    np.testing.assert_allclose(
        line.get_ydata(), values.std(0) / values.mean(0) * 100, rtol=1e-6
    )


@pytest.mark.parametrize("smooth", [False, True])
def test_cv_rescaled_data(conditions_struct, smooth):
    fig = heterogeneity.plot_cv_rescaled_data(
        conditions_struct, "body_seg_volume", [0, 1], smooth=smooth, ax_size=(2, 2)
    )
    lines = _line_data(fig)
    assert len(lines) == 2 and len(lines[0]) == 100
