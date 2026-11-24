import matplotlib.pyplot as plt
import numpy as np
import pytest

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
