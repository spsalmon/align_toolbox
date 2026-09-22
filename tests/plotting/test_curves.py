import numpy as np
import pytest

from align_toolbox.plotting import curves


@pytest.mark.parametrize(
    "x, experiment_time, expected_x_end",
    [("time", True, 99 / 6), ("time", False, 99 * 10 / 60), ("percentage", True, 100)],
)
def test_plot_aggregated_series_x_axis(
    conditions_struct, x, experiment_time, expected_x_end
):
    fig = curves.plot_aggregated_series(
        conditions_struct,
        "body_seg_volume",
        [0, 1],
        x=x,
        experiment_time=experiment_time,
        n_points=10,
    )
    lines = fig.axes[0].get_lines()
    assert len(lines) == 2
    assert lines[0].get_xdata()[-1] == pytest.approx(expected_x_end)
    assert fig.axes[0].get_yscale() == "log"
    # the mean of exponential growth from ~1e4 ends near 1e4 * exp(0.03 * 99)
    assert lines[0].get_ydata()[-1] == pytest.approx(1e4 * np.exp(0.03 * 99), rel=0.1)


def test_plot_aggregated_series_multiple_columns_and_xlim(conditions_struct):
    fig = curves.plot_aggregated_series(
        conditions_struct,
        ["body_seg_volume", "body_seg_length"],
        [0],
        n_points=10,
        xlim=(0, 8),
        y_axis_label="size",
        ax_size=(2, 2),
    )
    lines = fig.axes[0].get_lines()
    assert len(lines) == 2
    assert all(line.get_xdata().max() <= 8 for line in lines)
    # both columns share one legend entry per condition
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == [
        "Condition 0"
    ]
    assert fig.axes[0].get_ylabel() == "size"


def test_plot_aggregated_series_rejects_unknown_x(conditions_struct):
    with pytest.raises(ValueError):
        curves.plot_aggregated_series(
            conditions_struct, "body_seg_volume", [0], x="frames", n_points=5
        )


def test_plot_growth_curves_individuals_one_panel_per_condition(conditions_struct):
    fig = curves.plot_growth_curves_individuals(
        conditions_struct,
        "body_seg_volume",
        [0, 1],
        share_y_axis=True,
        legend={"description": ""},
        cut_after=10,
    )
    assert [ax.get_title() for ax in fig.axes] == ["condition 0", "condition 1"]
    assert all(len(ax.get_lines()) == 30 for ax in fig.axes)
    # traces keep one frame (10 min) past the cut
    assert all(line.get_xdata().max() <= 10 + 1 / 6 for line in fig.axes[0].get_lines())


def test_plot_growth_curves_individuals_single_condition(conditions_struct):
    fig = curves.plot_growth_curves_individuals(
        conditions_struct, "body_seg_volume", [1], share_y_axis=False, ax_size=(2, 2)
    )
    assert fig.axes[0].get_ylabel() == "body_seg_volume"


def test_plot_growth_curves_individuals_starts_at_hatch_when_time_is_offset(
    conditions_struct,
):
    condition = conditions_struct[0]
    condition["ecdysis_index"] = condition["ecdysis_index"] + 5
    condition["ecdysis_time_step"] = condition["ecdysis_index"] + 1000
    condition["ecdysis_experiment_time"] = condition["ecdysis_index"] * 600
    fig = curves.plot_growth_curves_individuals(
        conditions_struct, "body_seg_volume", [0], share_y_axis=False
    )
    first_x = fig.axes[0].get_lines()[0].get_xdata()[0]
    assert first_x == pytest.approx(0)
