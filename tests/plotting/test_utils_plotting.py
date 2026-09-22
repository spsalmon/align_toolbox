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


def test_add_legend_inside_axes(axes_with_lines):
    _, ax = axes_with_lines
    legend = up.add_legend(ax, "upper left")
    assert [t.get_text() for t in legend.get_texts()] == ["a", "b", "a"]
    assert legend._loc == 2  # "upper left"


def test_add_legend_deduplicates_labels(axes_with_lines):
    _, ax = axes_with_lines
    legend = up.add_legend(ax, deduplicate=True)
    assert [t.get_text() for t in legend.get_texts()] == ["a", "b"]


def test_add_legend_replaces_existing_legend(axes_with_lines):
    _, ax = axes_with_lines
    first = up.add_legend(ax)
    second = up.add_legend(ax, "lower right")
    assert ax.get_legend() is second is not first


def test_add_legend_outside_top_lays_entries_in_one_row(axes_with_lines):
    _, ax = axes_with_lines
    legend = up.add_legend(ax, "outside top")
    assert legend._ncols == 3


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


def test_get_colors_generates_palette():
    assert len(up.get_colors([0, 1, 2], None)) == 3


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


def test_create_fixed_ax_sized_fig_returns_divider():
    fig, ax, divider = up.create_fixed_ax_sized_fig(return_divider=True)
    assert divider is not None
    assert tuple(fig.get_size_inches()) == pytest.approx(
        (1.0 + 3.5 + 0.8, 0.6 + 3.0 + 0.3)
    )
