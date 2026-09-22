import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

from .utils_data_processing import exclude_arrests_from_series_at_ecdysis
from .utils_plotting import (
    add_legend,
    build_legend,
    create_fixed_ax_sized_fig,
    get_colors,
)


def plot_cv_at_ecdysis(
    conditions_struct: dict,
    column: str,
    conditions_to_plot: list[int],
    remove_hatch: bool = True,
    legend: dict | None = None,
    colors: list | dict | None = None,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    exclude_arrests: bool = False,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "best",
) -> matplotlib.figure.Figure:
    """
    Plot the coefficient of variation (CV) of a measurement at each molt event.

    One line per condition; x-axis ticks are labelled M1–M4 (or Hatch–M4 when
    ``remove_hatch=False``).

    Parameters:
        conditions_struct (list) : List of condition dicts.
        column (str) : Key of the per-molt measurement array
            (shape ``(n_worms, n_molts)``).
        conditions_to_plot (list[int]) : Indices of conditions to include.
        remove_hatch (bool) : If ``True``, drop the hatch (first) column before
            computing CV.  Defaults to ``True``.
        legend (dict or None) : Legend spec passed to ``build_legend``.
            Defaults to ``None``.
        colors (list or dict or None) : Color spec passed to ``get_colors``.
            Defaults to ``None``.
        x_axis_label (str or None) : X-axis label.  Defaults to ``None``.
        y_axis_label (str or None) : Y-axis label.  Defaults to ``None``.
        exclude_arrests (bool) : If ``True``, remove arrested worms before
            computing CV via ``exclude_arrests_from_series_at_ecdysis``.
            Defaults to ``False``.
        ax_size (tuple[float, float] or None) : If provided, fixes the axes area to
            ``(ax_w, ax_h)`` inches. Defaults to ``None``.
        legend_placement (str or None) : Legend placement passed to ``add_legend``;
            ``None`` hides the legend.  Defaults to ``"best"``.

    Returns:
        matplotlib.figure.Figure : The generated figure.
    """
    if ax_size is not None:
        create_fixed_ax_sized_fig(
            ax_w=ax_size[0], ax_h=ax_size[1]
        )  # sets pyplot current figure/axes

    color_palette = get_colors(conditions_to_plot, colors)

    for i, condition in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition]
        values = condition_dict[column]
        if remove_hatch:
            values = values[:, 1:]
        if exclude_arrests:
            values = exclude_arrests_from_series_at_ecdysis(values)
        cvs = np.nanstd(values, axis=0) / np.nanmean(values, axis=0) * 100
        label = build_legend(condition_dict, legend)
        plt.plot(cvs, label=label, marker="o", color=color_palette[i])
    # replace the ticks by [L1, L2, L3, L4]
    if remove_hatch:
        plt.xticks(range(4), ["M1", "M2", "M3", "M4"])
    else:
        plt.xticks(range(5), ["Hatch", "M1", "M2", "M3", "M4"])
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    add_legend(placement=legend_placement)
    fig = plt.gcf()
    plt.show()
    return fig


def plot_std_at_ecdysis(
    conditions_struct: dict,
    column: str,
    conditions_to_plot: list[int],
    remove_hatch: bool = True,
    legend: dict | None = None,
    colors: list | dict | None = None,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    exclude_arrests: bool = False,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "best",
) -> matplotlib.figure.Figure:
    """
    Plot the standard deviation of a measurement at each molt event.

    One line per condition; x-axis ticks are labelled M1–M4 (or Hatch–M4 when
    ``remove_hatch=False``).

    Parameters:
        conditions_struct (list) : List of condition dicts.
        column (str) : Key of the per-molt measurement array
            (shape ``(n_worms, n_molts)``).
        conditions_to_plot (list[int]) : Indices of conditions to include.
        remove_hatch (bool) : If ``True``, drop the hatch (first) column before
            computing std.  Defaults to ``True``.
        legend (dict or None) : Legend spec passed to ``build_legend``.
            Defaults to ``None``.
        colors (list or dict or None) : Color spec passed to ``get_colors``.
            Defaults to ``None``.
        x_axis_label (str or None) : X-axis label.  Defaults to ``None``.
        y_axis_label (str or None) : Y-axis label.  Defaults to ``None``.
        exclude_arrests (bool) : If ``True``, remove arrested worms before
            computing std via ``exclude_arrests_from_series_at_ecdysis``.
            Defaults to ``False``.
        ax_size (tuple[float, float] or None) : If provided, fixes the axes area to
            ``(ax_w, ax_h)`` inches. Defaults to ``None``.
        legend_placement (str or None) : Legend placement passed to ``add_legend``;
            ``None`` hides the legend.  Defaults to ``"best"``.

    Returns:
        matplotlib.figure.Figure : The generated figure.
    """
    if ax_size is not None:
        create_fixed_ax_sized_fig(
            ax_w=ax_size[0], ax_h=ax_size[1]
        )  # sets pyplot current figure/axes

    color_palette = get_colors(conditions_to_plot, colors)

    for i, condition in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition]
        values = condition_dict[column]
        if remove_hatch:
            values = values[:, 1:]
        if exclude_arrests:
            values = exclude_arrests_from_series_at_ecdysis(values)
        stds = np.nanstd(values, axis=0)
        label = build_legend(condition_dict, legend)
        plt.plot(stds, label=label, marker="o", color=color_palette[i])
    if remove_hatch:
        plt.xticks(range(4), ["M1", "M2", "M3", "M4"])
    else:
        plt.xticks(range(5), ["Hatch", "M1", "M2", "M3", "M4"])
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    add_legend(placement=legend_placement)
    fig = plt.gcf()
    plt.show()
    return fig


def plot_cv_development_percentage(
    conditions_struct: dict,
    column: str,
    conditions_to_plot: list[int],
    percentages: np.ndarray = np.linspace(0, 1, 11),
    legend: dict | None = None,
    colors: list | dict | None = None,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "best",
) -> matplotlib.figure.Figure:
    """
    Plot the coefficient of variation (CV) of a rescaled series at specific development percentages.

    ``column`` must be a pre-rescaled series of shape ``(n_worms, total_n_points)``.
    Values at the requested ``percentages`` are extracted by linear index into the
    rescaled time axis.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        column (str) : Key of the rescaled measurement array.
        conditions_to_plot (list[int]) : Indices of conditions to include.
        percentages (np.ndarray) : Fractional development positions (0–1) at which
            to compute CV.  Defaults to ``np.linspace(0, 1, 11)``.
        legend (dict or None) : Legend spec passed to ``build_legend``.
            Defaults to ``None``.
        colors (list or dict or None) : Color spec passed to ``get_colors``.
            Defaults to ``None``.
        x_axis_label (str or None) : X-axis label.  Defaults to ``None``.
        y_axis_label (str or None) : Y-axis label.  Defaults to ``None``.
        ax_size (tuple[float, float] or None) : If provided, fixes the axes area to
            ``(ax_w, ax_h)`` inches. Defaults to ``None``.
        legend_placement (str or None) : Legend placement passed to ``add_legend``;
            ``None`` hides the legend.  Defaults to ``"best"``.

    Returns:
        matplotlib.figure.Figure : The generated figure.
    """
    if ax_size is not None:
        create_fixed_ax_sized_fig(
            ax_w=ax_size[0], ax_h=ax_size[1]
        )  # sets pyplot current figure/axes

    color_palette = get_colors(conditions_to_plot, colors)
    for i, condition in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition]
        values = condition_dict[column]
        percentages_index = np.clip(
            percentages * values.shape[1], 0, values.shape[1] - 1
        )
        values = values[:, percentages_index.astype(int)]
        epsilon = np.finfo(values.dtype).eps
        cvs = np.nanstd(values, axis=0) / (np.nanmean(values, axis=0) + epsilon) * 100
        label = build_legend(condition_dict, legend)
        plt.plot(
            percentages * 100, cvs, label=label, color=color_palette[i], marker="o"
        )
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    add_legend(placement=legend_placement)
    fig = plt.gcf()
    plt.show()
    return fig


def plot_cv_rescaled_data(
    conditions_struct: dict,
    column: str,
    conditions_to_plot: list[int],
    smooth: bool = False,
    legend: dict | None = None,
    colors: list | dict | None = None,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "best",
) -> matplotlib.figure.Figure:
    """
    Plot the coefficient of variation (CV) across the full rescaled time axis.

    One line per condition plotted against the resampled index.  An optional
    median filter (window = 7) can be applied to the CV trace.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        column (str) : Key of the rescaled measurement array
            (shape ``(n_worms, n_points)``).
        conditions_to_plot (list[int]) : Indices of conditions to include.
        smooth (bool) : If ``True``, apply a median filter (window = 7) to each
            CV trace.  Defaults to ``False``.
        legend (dict or None) : Legend spec passed to ``build_legend``.
            Defaults to ``None``.
        colors (list or dict or None) : Color spec passed to ``get_colors``.
            Defaults to ``None``.
        x_axis_label (str or None) : X-axis label.  Defaults to ``None``.
        y_axis_label (str or None) : Y-axis label.  Defaults to ``None``.
        ax_size (tuple[float, float] or None) : If provided, fixes the axes area to
            ``(ax_w, ax_h)`` inches. Defaults to ``None``.
        legend_placement (str or None) : Legend placement passed to ``add_legend``;
            ``None`` hides the legend.  Defaults to ``"best"``.

    Returns:
        matplotlib.figure.Figure : The generated figure.
    """
    if ax_size is not None:
        create_fixed_ax_sized_fig(
            ax_w=ax_size[0], ax_h=ax_size[1]
        )  # sets pyplot current figure/axes

    color_palette = get_colors(conditions_to_plot, colors)

    for i, condition in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition]
        values = condition_dict[column]
        cvs = np.nanstd(values, axis=0) / np.nanmean(values, axis=0) * 100
        label = build_legend(condition_dict, legend)
        if smooth:
            cvs = medfilt(cvs, 7)
            # cvs = savgol_filter(cvs, 15, 3)
        plt.plot(cvs, label=label, color=color_palette[i])
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    add_legend(placement=legend_placement)
    fig = plt.gcf()
    plt.show()
    return fig
