import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from align_toolbox.plotting import boxplots, curves, heterogeneity, proportions
from align_toolbox.plotting import utils_plotting as up


def test_save_figure_creates_directory(tmp_path):
    fig = plt.figure()
    up.save_figure(fig, "plot", str(tmp_path / "figures"), format="png", dpi=50)
    assert (tmp_path / "figures" / "plot.png").stat().st_size > 0


def test_build_legend_default_uses_condition_id():
    assert up.build_legend({"condition_id": 3.0}, None) == "Condition 3"


def test_build_legend_joins_values_and_units():
    condition = {"temperature": 20, "food": "OP50"}
    legend = up.build_legend(condition, {"temperature": "°C", "food": ""})
    assert legend == "20 °C, OP50"


@pytest.fixture
def axes_with_lines():
    fig, ax = plt.subplots()
    ax.plot([0, 1], label="a")
    ax.plot([1, 0], label="b")
    ax.plot([1, 1], label="a")
    return fig, ax


def test_add_legend_deduplicates_labels(axes_with_lines):
    _, ax = axes_with_lines
    legend = up.add_legend(ax, deduplicate=True)
    assert [t.get_text() for t in legend.get_texts()] == ["a", "b"]


def test_add_legend_replaces_existing_legend(axes_with_lines):
    _, ax = axes_with_lines
    first = up.add_legend(ax)
    second = up.add_legend(ax, "lower right")
    assert ax.get_legend() is second is not first


def test_add_legend_on_figure_collects_from_all_axes():
    fig, (left, right) = plt.subplots(1, 2)
    left.plot([0], label="left")
    left.legend()
    right.plot([0], label="right")
    legend = up.add_legend(fig, "best")
    assert [t.get_text() for t in legend.get_texts()] == ["left", "right"]
    assert left.get_legend() is None
    assert fig.legends == [legend]


def test_add_legend_with_explicit_handles_and_none_placement(axes_with_lines):
    _, ax = axes_with_lines
    handles = ax.get_lines()[:1]
    legend = up.add_legend(ax, "outside right", handles=handles, labels=["only"])
    assert [t.get_text() for t in legend.get_texts()] == ["only"]
    assert up.add_legend(ax, None) is None
    assert ax.get_legend() is None


def test_add_legend_defaults_to_current_axes(axes_with_lines):
    _, ax = axes_with_lines
    assert up.add_legend() is ax.get_legend()


def test_add_legend_draws_line_styles_without_markers_or_caps():
    fig, ax = plt.subplots()
    ax.errorbar([0, 1], [0, 1], yerr=0.1, marker="o", linestyle="--", label="bars")
    ax.plot([0, 1], [1, 0], marker="o", linestyle=":", color="red", label="line")
    ax.plot([0, 1], [0.5, 0.5], "s", label="markers only")
    legend = up.add_legend(ax)
    handles = dict(zip([t.get_text() for t in legend.texts], legend.legend_handles))
    dashed, dotted, markers_only = (
        handles["bars"],
        handles["line"],
        handles["markers only"],
    )
    assert dashed.get_linestyle() == "--"
    assert dotted.get_linestyle() == ":"
    assert dashed.get_marker() == dotted.get_marker() == "None"
    assert dotted.get_color() == "red"
    assert markers_only.get_marker() == "s"
    assert legend.handlelength == up.LEGEND_HANDLE_LENGTH
    plt.close(fig)


@pytest.mark.parametrize(
    "kwargs",
    [{"placement": "somewhere"}, {"handles": []}],
    ids=["bad-placement", "handles-without-labels"],
)
def test_add_legend_rejects_invalid_arguments(axes_with_lines, kwargs):
    _, ax = axes_with_lines
    with pytest.raises(ValueError):
        up.add_legend(ax, **kwargs)


@pytest.mark.parametrize(
    "log_scale, expected",
    [
        (True, ("linear", "log")),
        (False, ("linear", "linear")),
        ((True, False), ("log", "linear")),
        ([False, True], ("linear", "log")),
    ],
)
def test_set_scale(log_scale, expected):
    _, ax = plt.subplots()
    up.set_scale(ax, log_scale)
    assert (ax.get_xscale(), ax.get_yscale()) == expected


def test_get_colors_orders_dict_by_conditions():
    assert up.get_colors([2, 0], {0: "red", 2: "blue"}) == ["blue", "red"]


@pytest.mark.parametrize(
    "colors", [["red"], {0: "red"}], ids=["list-too-short", "dict-missing-key"]
)
def test_get_colors_validates_user_colors(colors):
    with pytest.raises(AssertionError):
        up.get_colors([0, 1], colors)


@pytest.mark.parametrize(
    "nrows, ncols, axes_shape",
    [(1, 1, None), (1, 3, (3,)), (2, 1, (2,)), (2, 2, (2, 2))],
)
def test_create_fixed_ax_sized_fig_layout(nrows, ncols, axes_shape):
    fig, axes = up.create_fixed_ax_sized_fig(
        ax_w=2.0, ax_h=1.5, nrows=nrows, ncols=ncols, dpi=100
    )
    if axes_shape is None:
        axes = np.array([axes])
    else:
        assert axes.shape == axes_shape
    fig.canvas.draw()
    for ax in axes.flat:
        bbox = ax.get_window_extent()
        assert (bbox.width / 100, bbox.height / 100) == pytest.approx((2.0, 1.5))


def test_create_fixed_ax_sized_fig_rows_are_top_to_bottom():
    fig, axes = up.create_fixed_ax_sized_fig(nrows=2, ncols=1)
    fig.canvas.draw()
    assert axes[0].get_window_extent().y0 > axes[1].get_window_extent().y0


def test_get_linestyles_fills_missing_conditions_with_default():
    assert up.get_linestyles([0, 1, 5], {5: "--"}) == ["-", "-", "--"]
    assert up.get_linestyles([0, 1], None, default=":") == [":", ":"]
    assert up.get_linestyles([0, 1], ["-", "-."]) == ["-", "-."]


def test_get_hatches_leaves_unlisted_conditions_plain():
    assert up.get_hatches([0, 5], {5: "//"}) == [None, "//"]
    assert up.get_hatches([0, 5], None) == [None, None]


@pytest.mark.parametrize("function", [up.get_linestyles, up.get_hatches])
def test_condition_styles_reject_list_of_wrong_length(function):
    with pytest.raises(ValueError, match="does not match"):
        function([0, 1, 2], ["--", "--"])


def test_build_condition_table_accepts_lists_dicts_and_callables():
    conditions_struct = [
        {"condition_id": 0, "description": "no TIR, raga-1:AID"},
        {"condition_id": 1, "description": "col-10:TIR, raga-1:AID"},
        {"condition_id": 2, "description": "yap-1, col-10:TIR"},
    ]
    table = up.build_condition_table(
        conditions_struct,
        [2, 0, 1],
        {
            "raga-1:AID": lambda c: "raga-1:AID" in c["description"],
            "col-10:TIR": {1: True, 2: True},
            "yap-1": ["abt7", None, False],
        },
    )
    assert table == {
        "raga-1:AID": ["−", "+", "+"],
        "col-10:TIR": ["+", "−", "+"],
        "yap-1": ["abt7", "", "−"],
    }


@pytest.mark.parametrize("row", [[True], "raga-1:AID"])
def test_build_condition_table_rejects_invalid_rows(row):
    with pytest.raises(ValueError, match="(?i)condition table row"):
        up.build_condition_table([{}, {}], [0, 1], {"raga-1:AID": row})


def test_add_condition_table_aligns_columns_on_ticks_and_stacks_rows():
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, 2.5)
    table = {"AID": ["+", "−", "+"], "TIR": ["−", "+", "+"]}
    texts = up.add_condition_table(ax, table)
    fig.canvas.draw()

    labels = [text for text in texts if text.get_text() in table]
    entries = [text for text in texts if text.get_text() not in table]
    assert [text.get_text() for text in labels] == ["AID", "TIR"]
    assert [text.get_text() for text in entries] == table["AID"] + table["TIR"]
    tick_x = ax.transData.transform([(x, 0) for x in range(3)])[:, 0]
    for row in (entries[:3], entries[3:]):
        extents = [text.get_window_extent() for text in row]
        np.testing.assert_allclose([(e.x0 + e.x1) / 2 for e in extents], tick_x)
    axes_bottom = ax.get_window_extent().y0
    first_row_top = entries[0].get_window_extent().y1
    second_row_top = entries[3].get_window_extent().y1
    assert axes_bottom > first_row_top > second_row_top
    assert labels[0].get_window_extent().x1 < ax.get_window_extent().x0
    ticks = ax.xaxis.get_major_ticks()
    assert list(ax.get_xticks()) == [0, 1, 2]
    assert all(tick.tick1line.get_visible() for tick in ticks)
    assert not any(tick.label1.get_visible() for tick in ticks)
    tick_bottom = min(tick.tick1line.get_window_extent().y0 for tick in ticks)
    assert tick_bottom > first_row_top


def test_add_condition_table_can_hide_row_labels_and_ticks():
    fig, ax = plt.subplots()
    ax.tick_params(axis="x", bottom=False)
    texts = up.add_condition_table(
        ax, {"AID": ["+", "−"]}, show_row_labels=False, show_ticks=False
    )
    assert [text.get_text() for text in texts] == ["+", "−"]
    assert not any(tick.tick1line.get_visible() for tick in ax.xaxis.get_major_ticks())


def _legend_entries(legend):
    """Label, color, line style and hatch of each entry of a drawn legend."""
    entries = []
    for text, handle in zip(legend.get_texts(), legend.legend_handles):
        if isinstance(handle, Patch):
            style = (to_rgba(handle.get_facecolor()), None, handle.get_hatch())
        else:
            style = (to_rgba(handle.get_color()), handle.get_linestyle(), None)
        entries.append((text.get_text(), *style))
    return entries


def test_plot_legend_matches_the_legend_of_a_curve_plot(conditions_struct):
    styles = {
        "legend": {"description": "", "food": ""},
        "colors": ["red", "blue"],
        "linestyles": {1: "--"},
    }
    plot = curves.plot_aggregated_series(
        conditions_struct, "body_seg_volume", [0, 1], n_points=10, **styles
    )
    fig = up.plot_legend(conditions_struct, [0, 1], **styles)
    assert _legend_entries(fig.axes[0].get_legend()) == _legend_entries(
        plot.axes[0].get_legend()
    )
    assert _legend_entries(fig.axes[0].get_legend())[1] == (
        "condition 1, OP50",
        to_rgba("blue"),
        "--",
        None,
    )


def test_plot_legend_matches_the_hatched_legend_of_a_box_plot(conditions_struct):
    styles = {"colors": ["red", "blue"], "hatches": {1: "//"}}
    plot = boxplots.boxplot(
        conditions_struct, "body_seg_volume_at_ecdysis", [0, 1], **styles
    )
    fig = up.plot_legend(conditions_struct, [0, 1], kind="patch", **styles)
    entries = _legend_entries(fig.axes[0].get_legend())
    plot_entries = _legend_entries(plot.legends[0])
    assert [entry[0] for entry in entries] == [entry[0] for entry in plot_entries]
    assert [entry[3] for entry in entries] == [entry[3] for entry in plot_entries]
    assert [entry[1] for entry in entries] == [to_rgba("red"), to_rgba("blue")]


def test_plot_legend_figure_is_sized_to_the_legend_alone(conditions_struct):
    fig = up.plot_legend(conditions_struct, [0, 1], kind="marker")
    ax = fig.axes[0]
    assert not ax.axison
    legend_extent = ax.get_legend().get_window_extent()
    np.testing.assert_allclose(
        fig.get_size_inches() * fig.dpi,
        [legend_extent.width, legend_extent.height],
        rtol=0.01,
    )
    assert all(handle.get_marker() == "o" for handle in ax.get_legend().legend_handles)


def _five_conditions():
    return [
        {"condition_id": condition_id, "name": name}
        for condition_id, name in enumerate("abcde")
    ]


def _entry_positions(legend):
    """Label of each entry with its (column, row) rank in the drawn legend."""
    legend.figure.draw_without_rendering()
    extents = {text.get_text(): text.get_window_extent() for text in legend.get_texts()}
    xs = sorted({round(e.x0) for e in extents.values()})
    ys = sorted({round(e.y0) for e in extents.values()}, reverse=True)
    return {
        label: (xs.index(round(e.x0)), ys.index(round(e.y0)))
        for label, e in extents.items()
    }


@pytest.mark.parametrize(
    "fill, expected",
    [
        ("columns", {"a": (0, 0), "b": (0, 1), "c": (1, 0), "d": (1, 1), "e": (2, 0)}),
        ("rows", {"a": (0, 0), "b": (1, 0), "c": (2, 0), "d": (0, 1), "e": (1, 1)}),
    ],
)
def test_plot_legend_fills_columns_or_rows(fill, expected):
    fig = up.plot_legend(
        _five_conditions(), list(range(5)), legend={"name": ""}, ncols=3, fill=fill
    )
    assert _entry_positions(fig.axes[0].get_legend()) == expected


@pytest.mark.parametrize("nrows, n_columns", [(1, 5), (2, 3), (5, 1), (9, 1)])
def test_plot_legend_nrows_sets_the_fewest_columns_that_fit(nrows, n_columns):
    fig = up.plot_legend(
        _five_conditions(), list(range(5)), legend={"name": ""}, nrows=nrows
    )
    columns = {
        column for column, _ in _entry_positions(fig.axes[0].get_legend()).values()
    }
    assert len(columns) == n_columns


def test_plot_legend_margin_and_figsize_set_the_figure_size(conditions_struct):
    fitted = up.plot_legend(conditions_struct, [0, 1])
    padded = up.plot_legend(conditions_struct, [0, 1], margin=0.25)
    np.testing.assert_allclose(
        padded.get_size_inches(), fitted.get_size_inches() + 0.5, rtol=0.01
    )
    fixed = up.plot_legend(conditions_struct, [0, 1], figsize=(4, 2))
    np.testing.assert_allclose(fixed.get_size_inches(), (4, 2))
    extent = fixed.axes[0].get_legend().get_window_extent()
    np.testing.assert_allclose(
        [(extent.x0 + extent.x1) / 2, (extent.y0 + extent.y1) / 2],
        np.array([4, 2]) * fixed.dpi / 2,
        rtol=0.01,
    )


def test_plot_legend_larger_font_and_handles_make_a_larger_legend(conditions_struct):
    small = up.plot_legend(conditions_struct, [0, 1], fontsize=8, handlelength=2)
    large = up.plot_legend(conditions_struct, [0, 1], fontsize=16, handlelength=6)
    assert np.all(large.get_size_inches() > small.get_size_inches())
    assert large.axes[0].get_legend().handlelength == 6


@pytest.mark.parametrize("kwargs", [{"fill": "diagonal"}, {"ncols": 0}, {"nrows": 0}])
def test_plot_legend_rejects_invalid_layout(conditions_struct, kwargs):
    with pytest.raises(ValueError):
        up.plot_legend(conditions_struct, [0, 1], **kwargs)


def test_plot_legend_draws_into_given_axes(conditions_struct):
    fig, axes = plt.subplots(1, 2)
    returned = up.plot_legend(conditions_struct, [1, 0], ax=axes[1])
    assert returned is fig
    assert [text.get_text() for text in axes[1].get_legend().get_texts()] == [
        "Condition 1",
        "Condition 0",
    ]


def test_plot_legend_rejects_unknown_kind(conditions_struct):
    with pytest.raises(ValueError):
        up.plot_legend(conditions_struct, [0, 1], kind="bar")


VOLUME, LENGTH = "body_seg_volume_at_ecdysis", "body_seg_length_at_ecdysis"
PLOTS_WITHOUT_LEGEND = {
    "aggregated": lambda c, **kw: curves.plot_aggregated_series(
        c, "body_seg_volume", [0, 1], n_points=10, **kw
    ),
    "violin": lambda c, **kw: boxplots.violinplot(c, VOLUME, [0, 1], **kw),
    "box": lambda c, **kw: boxplots.boxplot(c, VOLUME, [0, 1], **kw),
    "swarm": lambda c, **kw: boxplots.swarmplot(c, VOLUME, [0, 1], **kw),
    "cv": lambda c, **kw: heterogeneity.plot_cv_at_ecdysis(c, VOLUME, [0, 1], **kw),
    "std": lambda c, **kw: heterogeneity.plot_std_at_ecdysis(c, VOLUME, [0, 1], **kw),
    "model_comparison": lambda c, **kw: proportions.plot_model_comparison_at_ecdysis(
        c, VOLUME, LENGTH, [0, 1], single_plot=False, **kw
    ),
    "correlation": lambda c, **kw: proportions.plot_correlation_at_ecdysis(
        c, VOLUME, LENGTH, [0, 1], **kw
    ),
    "deviation": lambda c, **kw: proportions.plot_deviation_from_model_at_ecdysis(
        c, VOLUME, LENGTH, 0, [0, 1], **kw
    ),
    "normalized": lambda c, **kw: proportions.plot_normalized_proportions_at_ecdysis(
        c, VOLUME, LENGTH, 0, [0, 1], **kw
    ),
}


@pytest.mark.parametrize("name", PLOTS_WITHOUT_LEGEND)
def test_plots_can_be_drawn_without_any_legend(conditions_struct, shown_figures, name):
    PLOTS_WITHOUT_LEGEND[name](conditions_struct, legend_placement=None)
    assert shown_figures
    for fig in shown_figures:
        assert not fig.legends
        assert all(ax.get_legend() is None for ax in fig.axes)
