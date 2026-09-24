import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from align_toolbox.data_analysis import rescale_and_aggregate
from align_toolbox.data_analysis.time_series import smooth_series_classified
from align_toolbox.foundation.utils import find_best_string_match

from .utils_plotting import (
    add_legend,
    build_legend,
    create_fixed_ax_sized_fig,
    get_colors,
    set_scale,
)


def plot_aggregated_series(
    conditions_struct: list,
    series_column: str | list[str],
    conditions_to_plot: list[int],
    x: str = "time",
    experiment_time: bool = True,
    aggregation: str = "mean",
    n_points: int = 100,
    time_step: int = 10,
    log_scale: bool = True,
    colors: list | dict | None = None,
    legend: dict | None = None,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    xlim: tuple[float, float] | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "best",
) -> matplotlib.figure.Figure:
    """
    Plot the time-rescaled aggregated series with 95% confidence intervals.

    Series are rescaled to a common time axis via ``rescale_and_aggregate``, then
    plotted as a solid line (mean or median) with a shaded 95% CI band.
    If ``series_column`` is a list, all columns are overlaid on the same axes.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        series_column (str or list[str]) : Key(s) of the measurement series to plot.
        conditions_to_plot (list[int]) : Indices of conditions to include.
        x (str) : X-axis variable.  ``"time"`` uses rescaled hours;
            ``"percentage"`` uses development completion (0–100 %).
            Defaults to ``"time"``.
        experiment_time (bool) : If ``True``, use absolute experiment time (hours);
            otherwise use time-step index scaled by ``time_step``.
            Defaults to ``True``.
        aggregation (str) : Aggregation function; ``"mean"`` or ``"median"``.
            Defaults to ``"mean"``.
        n_points (int) : Number of resampled points per larval stage.
            Defaults to ``100``.
        time_step (int) : Minutes per frame, used to convert time-step indices to
            hours when ``experiment_time=False``.  Defaults to ``10``.
        log_scale (bool) : If ``True``, set the y-axis to log scale.
            Defaults to ``True``.
        colors (list or dict or None) : Color spec passed to ``get_colors``.
            Defaults to ``None``.
        legend (dict or None) : Legend spec passed to ``build_legend``.
            Defaults to ``None``.
        x_axis_label (str or None) : X-axis label; auto-generated when ``None``.
            Defaults to ``None``.
        y_axis_label (str or None) : Y-axis label; falls back to ``series_column``
            when ``None``.  Defaults to ``None``.
        xlim (tuple[float, float] or None) : X-axis limits ``(xmin, xmax)`` used to
            crop the plotted range.  Defaults to ``None``.
        ax_size (tuple[float, float] or None) : If provided, fixes the axes area to
            ``(ax_w, ax_h)`` inches. Defaults to ``None``.
        legend_placement (str or None) : Legend placement passed to ``add_legend``;
            ``None`` hides the legend.  Defaults to ``"best"``.

    Returns:
        matplotlib.figure.Figure : The generated figure.

    Raises:
        ValueError : If ``x`` is not ``"time"`` or ``"percentage"``.
    """
    if ax_size is not None:
        create_fixed_ax_sized_fig(
            ax_w=ax_size[0], ax_h=ax_size[1]
        )  # sets pyplot current figure/axes
    color_palette = get_colors(conditions_to_plot, colors)

    def plot_single_series(column: str):
        for i, condition_id in enumerate(conditions_to_plot):
            condition_dict = conditions_struct[condition_id]
            if experiment_time:
                time = condition_dict["experiment_time_hours"]
                larval_stage_durations = condition_dict[
                    "larval_stage_durations_experiment_time_hours"
                ]
            else:
                time = condition_dict["time"]
                larval_stage_durations = condition_dict[
                    "larval_stage_durations_time_step"
                ]

            qc_keys = [key for key in condition_dict.keys() if "qc" in key]
            if len(qc_keys) == 1:
                qc_key = qc_keys[0]
            else:
                qc_key = find_best_string_match(column, qc_keys)

            rescaled_time, aggregated_series, _, ste_series = rescale_and_aggregate(
                condition_dict[column],
                time,
                condition_dict["ecdysis_index"],
                larval_stage_durations,
                condition_dict[qc_key],
                aggregation=aggregation,
                n_points=n_points,
            )
            ci_lower = aggregated_series - 1.96 * ste_series
            ci_upper = aggregated_series + 1.96 * ste_series
            if not experiment_time:
                rescaled_time = rescaled_time * time_step / 60
            label = build_legend(condition_dict, legend)

            if x == "time":
                x_values = rescaled_time
            elif x == "percentage":
                x_values = np.linspace(0, 100, len(rescaled_time))
            else:
                raise ValueError(
                    f"Invalid x value: {x}. Must be 'time' or 'percentage'."
                )

            if xlim is not None:
                x_values_not_in_xlim = (x_values < xlim[0]) | (x_values > xlim[1])
                x_values = x_values[~x_values_not_in_xlim]
                aggregated_series = aggregated_series[~x_values_not_in_xlim]
                ci_lower = ci_lower[~x_values_not_in_xlim]
                ci_upper = ci_upper[~x_values_not_in_xlim]

            plt.plot(x_values, aggregated_series, color=color_palette[i], label=label)
            plt.fill_between(
                x_values, ci_lower, ci_upper, color=color_palette[i], alpha=0.2
            )

    if isinstance(series_column, list):
        for column in series_column:
            plot_single_series(column)
    else:
        plot_single_series(series_column)
    add_legend(placement=legend_placement, deduplicate=True)
    plt.yscale("log" if log_scale else "linear")
    if y_axis_label is not None:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel(series_column)
    if x_axis_label is not None:
        plt.xlabel(x_axis_label)
    else:
        plt.xlabel("time (h)" if x == "time" else "development completion (%)")

    fig = plt.gcf()
    plt.show()
    return fig


def plot_growth_curves_individuals(
    conditions_struct: list,
    column: str,
    conditions_to_plot: list[int],
    share_y_axis: bool = True,
    log_scale: bool | tuple | list = True,
    figsize: tuple[float, float] | None = None,
    ax_size: tuple[float, float] | None = None,
    legend: dict | None = None,
    y_axis_label: str | None = None,
    cut_after: float | None = None,
    alpha: float = 1.0,
) -> matplotlib.figure.Figure:
    """
    Plot smoothed individual-worm growth curves with one subplot per condition.

    Each worm's series is smoothed via ``smooth_series_classified`` and plotted
    from hatch time.  Worms without a detected hatch event are skipped.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        column (str) : Key of the raw measurement series.
        conditions_to_plot (list[int]) : Indices of conditions to include.
        share_y_axis (bool) : If ``True``, all subplots share the same y-axis range.
            Defaults to ``True``.
        log_scale (bool or tuple or list) : Scale spec passed to ``set_scale``.
            Defaults to ``True`` (log y-axis only).
        figsize (tuple[float, float] or None) : Figure size ``(width, height)`` in inches.
            Defaults to ``(n_conditions * 8, 10)``.
        ax_size (tuple[float, float] or None) : If provided, each panel's axes area is fixed to
            ``(ax_w, ax_h)`` inches. Overrides ``figsize``. Note: ``share_y_axis`` is ignored
            when ``ax_size`` is provided. Defaults to ``None``.
        legend (dict or None) : Legend spec used to generate subplot titles.
            Defaults to ``None``.
        y_axis_label (str or None) : Y-axis label; falls back to ``column`` when ``None``.
            Defaults to ``None``.
        cut_after (float or None) : Truncate worm traces at this experiment time
            (hours after hatch).  ``None`` keeps full traces.  Defaults to ``None``.
        alpha (float) : Opacity of the individual curves, between 0 and 1.
            Defaults to ``1.0``.

    Returns:
        matplotlib.figure.Figure : The generated figure.
    """
    if ax_size is not None:
        fig, ax = create_fixed_ax_sized_fig(
            ncols=len(conditions_to_plot), ax_w=ax_size[0], ax_h=ax_size[1]
        )
    else:
        if figsize is None:
            figsize = (len(conditions_to_plot) * 8, 10)
        fig, ax = plt.subplots(
            1, len(conditions_to_plot), figsize=figsize, sharey=share_y_axis
        )
    for i, condition_id in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition_id]

        qc_keys = [key for key in condition_dict.keys() if "qc" in key]
        if len(qc_keys) == 1:
            qc_key = qc_keys[0]
        else:
            qc_key = find_best_string_match(column, qc_keys)

        for j in range(len(condition_dict[column])):
            time = condition_dict["experiment_time"][j] / 3600
            data = condition_dict[column][j]
            qc = condition_dict[qc_key][j]
            hatch = condition_dict["ecdysis_index"][j][0]
            hatch_experiment_time = (
                condition_dict["ecdysis_experiment_time"][j][0] / 3600
            )
            individual_counts = 0
            if not np.isnan(hatch):
                hatch = int(hatch)
                if cut_after is not None:
                    indexes_to_cut = np.where(time > cut_after)[0]
                    if len(indexes_to_cut) > 0:
                        data = data[: indexes_to_cut[0] + 1]
                        time = time[: indexes_to_cut[0] + 1]
                        qc = qc[: indexes_to_cut[0] + 1]

                time = time[hatch:]
                time = time - hatch_experiment_time
                data = data[hatch:]
                qc = qc[hatch:]
                filtered_data = smooth_series_classified(
                    data,
                    time,
                    qc,
                )
                label = build_legend(condition_dict, legend)
                try:
                    ax[i].plot(time, filtered_data, alpha=alpha)
                    set_scale(ax[i], log_scale)
                except TypeError:
                    ax.plot(time, filtered_data, alpha=alpha)
                    set_scale(ax, log_scale)
                    individual_counts += 1
            print(
                f"Individual counts for condition {condition_id}: {individual_counts}"
            )
        try:
            ax[i].title.set_text(label)
        except TypeError:
            ax.title.set_text(label)
    # Set labels
    if y_axis_label is not None:
        try:
            ax[0].set_ylabel(y_axis_label)
            ax[0].set_xlabel("time (h)")
        except TypeError:
            ax.set_ylabel(y_axis_label)
            ax.set_xlabel("time (h)")
    else:
        try:
            ax[0].set_ylabel(column)
            ax[0].set_xlabel("time (h)")
        except TypeError:
            ax.set_ylabel(column)
            ax.set_xlabel("time (h)")
    fig = plt.gcf()
    plt.show()
    return fig
