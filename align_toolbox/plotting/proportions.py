from typing import Any, Literal
from warnings import warn

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from align_toolbox.data_analysis import rescale_and_aggregate
from align_toolbox.foundation.utils import find_best_string_match

from .utils_plotting import (
    add_legend,
    build_legend,
    create_fixed_ax_sized_fig,
    get_colors,
    set_scale,
)

# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import RANSACRegressor

# from .utils_data_processing import exclude_arrests_from_series_at_ecdysis

# MODEL BUILDING


def _warn_remove_outliers_fitting_removed(
    remove_outliers_fitting: bool | None,
) -> None:
    """
    Warn that the removed ``remove_outliers_fitting`` argument was passed and is ignored.

    Parameters:
        remove_outliers_fitting (bool or None): Value passed by the caller; ``None``
            means the argument was not passed and nothing is emitted.
    """
    if remove_outliers_fitting is not None:
        warn(
            "`remove_outliers_fitting` has been removed and is ignored: outliers are "
            "no longer removed before fitting. Passing it will raise an error in a "
            "future release.",
            FutureWarning,
            stacklevel=3,
        )


DEVIATION_SCALES = ("log", "percent")


def log_ratio_to_percentage(log_ratios: np.ndarray) -> np.ndarray:
    """
    Convert natural-log ratios to percent deviations, ``(exp(x) - 1) * 100``.

    Use it to display log-ratio deviations, e.g. as the ``display_transform`` of
    ``boxplots.violinplot``; compute statistics on the log ratios themselves.

    Parameters:
        log_ratios (np.ndarray): Natural-log ratios ``log(observed / predicted)``.

    Returns:
        np.ndarray: Percent deviations, same shape as ``log_ratios``.
    """
    return np.expm1(log_ratios) * 100


def _resolve_deviation_scale(
    scale: str,
    scale_name: str,
    deprecated_name: str,
    deprecated_value: bool | None,
) -> str:
    """
    Validate a deviation scale, letting a deprecated boolean flag override it.

    Parameters:
        scale (str): Requested scale, ``"log"`` or ``"percent"``.
        scale_name (str): Name of the argument that replaces the deprecated one,
            used in the warning message.
        deprecated_name (str): Name of the deprecated boolean argument, used in the
            warning message.
        deprecated_value (bool or None): Value of the deprecated argument; ``None``
            means it was not passed. ``True`` maps to ``"percent"`` and ``False`` to
            ``"log"``, with a ``DeprecationWarning``.

    Returns:
        str: The resolved scale.

    Raises:
        ValueError: If ``scale`` is not one of ``DEVIATION_SCALES``.
    """
    if deprecated_value is not None:
        scale = "percent" if deprecated_value else "log"
        warn(
            f"`{deprecated_name}` is deprecated; use `{scale_name}={scale!r}` "
            "instead. Passing it will raise an error in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )
    if scale not in DEVIATION_SCALES:
        raise ValueError(
            f"Invalid deviation scale {scale!r}; expected one of {DEVIATION_SCALES}."
        )
    return scale


def _summarize_log_deviations(
    log_deviations: np.ndarray,
    display_scale: str,
    n_standard_errors: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Summarize log-ratio deviations per column as a center and an error interval.

    The mean and standard error are always computed on the log scale. For
    ``"percent"`` display, the center and bounds are back-transformed with
    ``(exp(x) - 1) * 100``, which gives a geometric-mean-based relative deviation
    with asymmetric bounds.

    Parameters:
        log_deviations (np.ndarray): Log-ratio deviations of shape
            ``(n_worms, n_points)``; NaN values are ignored.
        display_scale (str): ``"log"`` or ``"percent"``.
        n_standard_errors (float): Half-width of the interval in standard errors.
            (default: 1.0)

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Center, lower bound and upper
            bound, each of shape ``(n_points,)``, in the display scale.
    """
    mean = np.nanmean(log_deviations, axis=0)
    ste = np.nanstd(log_deviations, axis=0) / np.sqrt(
        np.sum(~np.isnan(log_deviations), axis=0)
    )
    center = mean
    lower = mean - n_standard_errors * ste
    upper = mean + n_standard_errors * ste
    if display_scale == "percent":
        center, lower, upper = (
            log_ratio_to_percentage(v) for v in (center, lower, upper)
        )
    return center, lower, upper


def _default_deviation_label(column: str, display_scale: str) -> str:
    """
    Build the default y-axis label for a deviation plot.

    Parameters:
        column (str): Name of the modeled measurement.
        display_scale (str): ``"log"`` or ``"percent"``.

    Returns:
        str: The axis label.
    """
    unit = "%" if display_scale == "percent" else "log ratio"
    return f"deviation from modeled {column} ({unit})"


def _get_continuous_proportion_model(
    rescaled_series_one: np.ndarray,
    rescaled_series_two: np.ndarray,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    plot_model: bool = True,
) -> Any:
    """
    Fit a LOWESS + linear-spline model to the log-log relationship between two series.

    Duplicate x values are averaged before fitting.  The resulting interpolant spans
    the observed x range.

    Parameters:
        rescaled_series_one (array-like) : X values (first measurement).
        rescaled_series_two (array-like) : Y values (second measurement).
        x_axis_label (str or None) : X-axis label for the diagnostic scatter plot.
            Defaults to ``"column one"``.
        y_axis_label (str or None) : Y-axis label for the diagnostic scatter plot.
            Defaults to ``"column two"``.
        plot_model (bool) : If ``True``, display a scatter + LOWESS plot.
            Defaults to ``True``.

    Returns:
        scipy.interpolate.BSpline : Fitted interpolant in log-log space.

    Raises:
        AssertionError : If ``rescaled_series_one`` and ``rescaled_series_two`` differ
            in length.
    """
    assert len(rescaled_series_one) == len(
        rescaled_series_two
    ), "The two series must have the same length."

    series_one = np.array(rescaled_series_one).flatten()
    series_two = np.array(rescaled_series_two).flatten()

    # remove elements that are nan in one of the two arrays
    correct_indices = ~np.isnan(series_one) & ~np.isnan(series_two)
    series_one = series_one[correct_indices]
    series_two = series_two[correct_indices]

    # log transform the data
    series_one = np.log(series_one)
    series_two = np.log(series_two)

    # for duplicate values, take the mean
    unique_series_one = np.unique(series_one)
    unique_series_two = np.array(
        [np.mean(series_two[series_one == value]) for value in unique_series_one]
    )

    series_one = unique_series_one
    series_two = unique_series_two

    plt.scatter(series_one, series_two)

    if x_axis_label is not None:
        plt.xlabel(x_axis_label)
    else:
        plt.xlabel("column one")

    if y_axis_label is not None:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel("column two")

    # lowess will return our "smoothed" data with a y value for at every x-value
    lowess = sm.nonparametric.lowess(series_two, series_one, frac=0.1)

    # unpack the lowess smoothed points to their values
    lowess_x = list(zip(*lowess))[0]
    lowess_y = list(zip(*lowess))[1]

    # interpolate the loess curve
    model = make_interp_spline(lowess_x, lowess_y, k=1)

    x = np.linspace(min(series_one), max(series_one), 500)
    y = model(x)

    plt.plot(x, y, color="red", linewidth=2)
    plt.show()

    return model


def _get_proportion_model(
    series_one_values: np.ndarray,
    series_two_values: np.ndarray,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    poly_degree: int = 1,
    plot_model: bool = True,
) -> Any:
    """
    Fit a polynomial OLS model to the log-log relationship between two series.

    Data from all molt events (axis 1) are pooled for fitting. After
    fitting, a ``get_confidence_intervals`` method is attached to the returned
    pipeline for downstream use.

    Parameters:
        series_one_values (np.ndarray) : X values of shape ``(n_worms, n_molts)``.
        series_two_values (np.ndarray) : Y values of shape ``(n_worms, n_molts)``.
        x_axis_label (str or None) : X-axis label for the diagnostic plot.
            Defaults to ``"column one"``.
        y_axis_label (str or None) : Y-axis label for the diagnostic plot.
            Defaults to ``"column two"``.
        poly_degree (int) : Degree of the polynomial features.  Defaults to ``1``.
        plot_model (bool) : If ``True``, display a scatter + fitted model plot with
            confidence bands.  Defaults to ``True``.

    Returns:
        sklearn.pipeline.Pipeline : Fitted pipeline with an additional
            ``get_confidence_intervals(x_pred)`` method returning
            ``(y_pred, ci_lower, ci_upper)``.

    Raises:
        AssertionError : If ``series_one_values`` and ``series_two_values`` differ
            in length.
    """
    assert len(series_one_values) == len(
        series_two_values
    ), "The two series must have the same length."

    alpha = 0.05  # significance level for confidence intervals

    fitting_x = []
    fitting_y = []

    model_plot_x = []

    for i in range(series_two_values.shape[-1]):
        values_one = series_one_values[:, i].flatten()
        values_two = series_two_values[:, i].flatten()
        correct_indices = ~np.isnan(values_one) & ~np.isnan(values_two)
        values_one = values_one[correct_indices]
        values_two = values_two[correct_indices]
        values_one = np.log(values_one)
        values_two = np.log(values_two)

        model_plot_x.extend(values_one)
        fitting_x.extend(values_one)
        fitting_y.extend(values_two)

    fitting_x = np.array(fitting_x).flatten()
    fitting_y = np.array(fitting_y).flatten()

    # Fit polynomial model with OLS
    model = Pipeline(
        [
            ("poly_features", PolynomialFeatures(degree=poly_degree)),
            ("regression", LinearRegression()),
        ]
    )
    model.fit(fitting_x.reshape(-1, 1), fitting_y)

    # Calculate confidence intervals
    def get_confidence_intervals(x_pred):
        # Transform features
        poly_features = model.named_steps["poly_features"]
        X_design = poly_features.fit_transform(fitting_x.reshape(-1, 1))
        X_pred = poly_features.transform(x_pred.reshape(-1, 1))

        # Get predictions
        y_pred = model.predict(x_pred.reshape(-1, 1))

        # Calculate residuals and standard error
        y_fitted = model.predict(fitting_x.reshape(-1, 1))
        residuals = fitting_y - y_fitted
        n = len(fitting_y)
        p = X_design.shape[1]  # number of parameters

        # Mean squared error
        mse = np.sum(residuals**2) / (n - p)

        # Standard error of prediction
        # SE = sqrt(MSE * (1 + x'(X'X)^(-1)x))
        try:
            XTX_inv = np.linalg.inv(X_design.T @ X_design)
            se_pred = np.sqrt(mse * (1 + np.sum((X_pred @ XTX_inv) * X_pred, axis=1)))
        except np.linalg.LinAlgError:
            # If matrix is singular, use simpler approximation
            se_pred = np.sqrt(mse) * np.ones(len(x_pred))

        # Critical value
        t_crit = stats.t.ppf(1 - alpha / 2, n - p)

        # Confidence intervals
        ci_lower = y_pred - t_crit * se_pred
        ci_upper = y_pred + t_crit * se_pred

        return y_pred, ci_lower, ci_upper

    if plot_model:
        plt.scatter(fitting_x, fitting_y, color="black", label="Data")

        if x_axis_label is not None:
            plt.xlabel(x_axis_label)
        else:
            plt.xlabel("column one")
        if y_axis_label is not None:
            plt.ylabel(y_axis_label)
        else:
            plt.ylabel("column two")

        # Plot model with confidence intervals
        x_plot = np.linspace(np.nanmin(model_plot_x), np.nanmax(model_plot_x), 100)
        y_pred, ci_lower, ci_upper = get_confidence_intervals(x_plot)

        plt.plot(x_plot, y_pred, color="red", label="Fitted model")
        plt.fill_between(
            x_plot,
            ci_lower,
            ci_upper,
            alpha=0.3,
            color="red",
            label=f"{int((1-alpha)*100)}% CI",
        )
        add_legend()
        plt.show()

    # Add method to model for getting confidence intervals
    model.get_confidence_intervals = get_confidence_intervals

    return model


# COMPARE MODELS FOR DIFFERENT CONDITIONS


def plot_model_comparison_at_ecdysis(
    conditions_struct: list,
    column_one: str,
    column_two: str,
    conditions_to_plot: list,
    remove_hatch: bool = True,
    poly_degree: int = 1,
    remove_outliers_fitting: bool | None = None,
    log_scale: tuple[bool, bool] | bool = (True, False),
    colors: list | dict | None = None,
    legend: dict | None = None,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    single_plot: bool = True,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "outside right",
) -> matplotlib.figure.Figure:
    """
    Scatter-plot the log-log relationship between two columns at molt events with fitted models.

    One polynomial model is fitted per condition.  A shared R² annotation is added to
    the legend.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        column_one (str) : Key of the X measurement (per-molt array).
        column_two (str) : Key of the Y measurement (per-molt array).
        conditions_to_plot (list) : Ordered condition identifiers.
        remove_hatch (bool) : If ``True``, drop the hatch column (index 0).
            Defaults to ``True``.
        poly_degree (int) : Polynomial degree for model fitting.  Defaults to ``1``.
        remove_outliers_fitting (bool or None) : Removed; ignored with a
            ``FutureWarning`` and will raise an error in a future release.
            Defaults to ``None``.
        log_scale (tuple[bool, bool] or bool) : Scale spec passed to ``set_scale``.
            Defaults to ``(True, False)``.
        colors (list or dict or None) : Color spec passed to ``get_colors``.
            Defaults to ``None``.
        legend (dict or None) : Legend spec passed to ``build_legend``.
            Defaults to ``None``.
        x_axis_label (str or None) : X-axis label; falls back to ``column_one``.
            Defaults to ``None``.
        y_axis_label (str or None) : Y-axis label; falls back to ``column_two``.
            Defaults to ``None``.
        single_plot (bool) : If ``True``, overlay all conditions on one axes;
            otherwise create one subplot per condition.  Defaults to ``True``.
        ax_size (tuple[float, float] or None) : If provided, each panel's axes area is fixed to
            ``(ax_w, ax_h)`` inches. Overrides the default figure size. Note: ``sharex`` and
            ``sharey`` are ignored when ``ax_size`` is provided. Defaults to ``None``.
        legend_placement (str or None) : Legend placement passed to ``add_legend``;
            ``None`` hides the legend.  With multiple subplots the legend is shared
            and placed relative to the figure.  Defaults to ``"outside right"``.

    Returns:
        matplotlib.figure.Figure : The generated figure.
    """
    _warn_remove_outliers_fitting_removed(remove_outliers_fitting)

    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    r_squared_texts = []

    if ax_size is not None:
        ncols = 1 if single_plot else len(conditions_to_plot)
        fig, axs = create_fixed_ax_sized_fig(
            ncols=ncols, ax_w=ax_size[0], ax_h=ax_size[1]
        )
    elif single_plot:
        fig, axs = plt.subplots(1, 1)
    else:
        fig, axs = plt.subplots(1, len(conditions_to_plot), sharey=True, sharex=True)

    for i, condition_idx in enumerate(conditions_to_plot):
        if single_plot or len(conditions_to_plot) == 1:
            current_ax = axs
        else:
            current_ax = axs[i]

        condition = conditions_struct[condition_idx]

        label = build_legend(
            condition,
            legend,
        )
        column_one_values = condition[column_one]
        column_two_values = condition[column_two]

        if remove_hatch:
            column_one_values = column_one_values[:, 1:]
            column_two_values = column_two_values[:, 1:]

        model = _get_proportion_model(
            column_one_values,
            column_two_values,
            poly_degree=poly_degree,
            plot_model=False,
        )

        scatter_x = []
        scatter_y = []
        model_plot_x = []
        model_plot_y = []

        for j in range(column_two_values.shape[-1]):
            values_one = column_one_values[:, j].flatten()
            values_two = column_two_values[:, j].flatten()
            correct_indices = ~np.isnan(values_one) & ~np.isnan(values_two)
            values_one = values_one[correct_indices]
            values_two = values_two[correct_indices]

            model_plot_x.extend(values_one)
            model_plot_y.extend(values_two)

            scatter_x.extend(values_one)
            scatter_y.extend(values_two)

        model_plot_x = np.array(model_plot_x)
        model_plot_y = np.array(model_plot_y)

        current_ax.scatter(
            scatter_x,
            scatter_y,
            color=color_palette[i],
            alpha=0.5,
        )

        # plot the model
        x_values = np.linspace(np.nanmin(model_plot_x), np.nanmax(model_plot_x), 100)
        y_values, ci_low, ci_high = model.get_confidence_intervals(np.log(x_values))
        y_values = np.exp(y_values)
        ci_low = np.exp(ci_low)
        ci_high = np.exp(ci_high)

        current_ax.plot(
            x_values,
            y_values,
            color=color_palette[i],
            linestyle="--",
            label=label,
        )

        current_ax.fill_between(
            x_values,
            ci_low,
            ci_high,
            color=color_palette[i],
            alpha=0.2,
        )

        # if not the first plot, remove the ticks on the y-axis
        if not single_plot and i > 0:
            current_ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        # text box with R^2 value
        r_squared = model.score(
            np.log(model_plot_x).reshape(-1, 1), np.log(model_plot_y)
        )
        textstr = f"$R^2$ = {r_squared:.2f}"

        r_squared_texts.append(textstr)

    # modify the legend to include R^2 values
    if single_plot or len(conditions_to_plot) == 1:
        handles, labels = axs.get_legend_handles_labels()
        new_labels = [
            f"{label} ({r_squared})"
            for label, r_squared in zip(labels, r_squared_texts)
        ]
        add_legend(axs, legend_placement, handles, new_labels)

        plt.xlabel(x_axis_label if x_axis_label else column_one)
        plt.ylabel(y_axis_label if y_axis_label else column_two)
    else:
        # For multiple plots, set labels for each subplot and create a shared legend
        for ax in axs:
            ax.set_xlabel(x_axis_label if x_axis_label else column_one)
        axs[0].set_ylabel(y_axis_label if y_axis_label else column_two)

        # Collect legend handles and labels from all subplots
        all_handles = []
        all_labels = []
        for ax in axs:
            handles, labels = ax.get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)

        # Create new labels with R^2 values
        new_labels = [
            f"{label} ({r_squared})"
            for label, r_squared in zip(all_labels, r_squared_texts)
        ]

        add_legend(fig, legend_placement, all_handles, new_labels)

    set_scale(plt.gca(), log_scale)
    fig = plt.gcf()
    plt.show()
    return fig


# COMPUTE DEVIATION FROM MODEL


def get_deviation_from_model(
    series_one_values: np.ndarray,
    series_two_values: np.ndarray,
    model: Any,
    deviation_scale: Literal["log", "percent"] = "log",
    percentage: bool | None = None,
) -> np.ndarray:
    """
    Compute per-worm deviations from a proportion model at each molt event.

    For each event (axis 1), the model is evaluated on log(series_one) to predict
    log(series_two). The native deviation is the log ratio
    ``r = log(actual) - log(predicted)``, which is symmetric (a ×2 and a ×0.5 shift
    have equal magnitude) and is the scale on which statistics should be computed.
    ``"percent"`` returns ``(exp(r) - 1) * 100`` for display.

    Parameters:
        series_one_values (np.ndarray) : X values of shape ``(n_worms, n_molts)``.
        series_two_values (np.ndarray) : Y values of shape ``(n_worms, n_molts)``.
        model : Fitted model with a ``predict`` method (sklearn Pipeline) or a
            callable (LOWESS spline) accepting log-transformed X values.
        deviation_scale (str) : ``"log"`` for natural-log ratios or ``"percent"``
            for percent relative deviations.  Defaults to ``"log"``.
        percentage (bool or None) : Deprecated; ``True`` maps to
            ``deviation_scale="percent"`` and ``False`` to ``"log"``.
            Defaults to ``None``.

    Returns:
        np.ndarray : Deviations of shape ``(n_worms, n_molts)``; NaN where either
            input value is NaN.

    Raises:
        ValueError : If ``deviation_scale`` is not ``"log"`` or ``"percent"``.
    """
    deviation_scale = _resolve_deviation_scale(
        deviation_scale, "deviation_scale", "percentage", percentage
    )
    deviations = []
    for i in range(series_two_values.shape[-1]):
        values_one = series_one_values[:, i].flatten()
        values_two = series_two_values[:, i].flatten()

        correct_indices = ~np.isnan(values_one) & ~np.isnan(values_two)
        values_one = values_one[correct_indices]
        values_two = values_two[correct_indices]

        if values_one.size == 0 or values_two.size == 0:
            deviations.append(np.array([]))
        else:
            try:
                log_expected_series_two = model.predict(
                    np.log(values_one).reshape(-1, 1)
                )
            except AttributeError:
                # Continuous models do not have predict method, use the model directly
                log_expected_series_two = model(np.log(values_one))
            deviation = np.log(values_two) - log_expected_series_two

            if deviation_scale == "percent":
                deviation = log_ratio_to_percentage(deviation)

            # Create full-length array with NaNs, then fill in the valid values
            full_deviation = np.full(len(correct_indices), np.nan)
            full_deviation[correct_indices] = deviation

            deviations.append(full_deviation)

    # Pad deviations to the same length with np.nan so they can be stacked into an array
    max_len = max(len(dev) for dev in deviations)
    padded_devs = [
        np.pad(dev, (0, max_len - len(dev)), constant_values=np.nan)
        for dev in deviations
    ]
    deviations = np.array(padded_devs).T
    return deviations


def plot_correlation(
    conditions_struct: list,
    column_one: str,
    column_two: str,
    conditions_to_plot: list,
    log_scale: bool | tuple | list = True,
    colors: list | dict | None = None,
    legend: dict | None = None,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "best",
) -> matplotlib.figure.Figure:
    """
    Plot the correlation between two measurements as aggregated rescaled series.

    Each condition is first rescaled and aggregated via ``rescale_and_aggregate``;
    the resulting mean traces are plotted against each other sorted by the x value.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        column_one (str) : Key of the X measurement series.
        column_two (str) : Key of the Y measurement series.
        conditions_to_plot (list) : Ordered condition identifiers.
        log_scale (bool or tuple or list) : Scale spec passed to ``set_scale``.
            Defaults to ``True``.
        colors (list or dict or None) : Color spec passed to ``get_colors``.
            Defaults to ``None``.
        legend (dict or None) : Legend spec passed to ``build_legend``.
            Defaults to ``None``.
        x_axis_label (str or None) : X-axis label; falls back to ``column_one``.
            Defaults to ``None``.
        y_axis_label (str or None) : Y-axis label; falls back to ``column_two``.
            Defaults to ``None``.
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
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    for i, condition_id in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition_id]

        qc_keys = [key for key in condition_dict.keys() if "qc" in key]
        if len(qc_keys) == 1:
            column_one_qc_key = qc_keys[0]
            column_two_qc_key = qc_keys[0]
        else:
            column_one_qc_key = find_best_string_match(column_one, qc_keys)
            column_two_qc_key = find_best_string_match(column_two, qc_keys)

        _, aggregated_series_one, _, _ = rescale_and_aggregate(
            condition_dict[column_one],
            condition_dict["time"],
            condition_dict["ecdysis_index"],
            condition_dict["larval_stage_durations_time_step"],
            condition_dict[column_one_qc_key],
            aggregation="mean",
        )

        _, aggregated_series_two, _, _ = rescale_and_aggregate(
            condition_dict[column_two],
            condition_dict["time"],
            condition_dict["ecdysis_index"],
            condition_dict["larval_stage_durations_time_step"],
            condition_dict[column_two_qc_key],
            aggregation="mean",
        )

        # sort the values
        order = np.argsort(aggregated_series_one)
        aggregated_series_one = aggregated_series_one[order]
        aggregated_series_two = aggregated_series_two[order]

        label = build_legend(condition_dict, legend)

        plt.plot(
            aggregated_series_one,
            aggregated_series_two,
            color=color_palette[i],
            label=label,
        )

    if x_axis_label is not None:
        plt.xlabel(x_axis_label)
    else:
        plt.xlabel(column_one)

    if y_axis_label is not None:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel(column_two)

    set_scale(plt.gca(), log_scale)

    add_legend(placement=legend_placement)

    fig = plt.gcf()
    plt.show()

    return fig


def plot_correlation_at_ecdysis(
    conditions_struct: list,
    column_one: str,
    column_two: str,
    conditions_to_plot: list,
    remove_hatch: bool = True,
    log_scale: bool | tuple | list = True,
    colors: list | dict | None = None,
    legend: dict | None = None,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "best",
) -> matplotlib.figure.Figure:
    """
    Plot the mean ± std of two measurements at each molt event as error-bar scatter.

    Each point on the plot represents one molt event; x and y error bars show the
    cross-worm standard deviation.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        column_one (str) : Key of the X measurement (per-molt array).
        column_two (str) : Key of the Y measurement (per-molt array).
        conditions_to_plot (list) : Ordered condition identifiers.
        remove_hatch (bool) : If ``True``, drop the hatch column (index 0).
            Defaults to ``True``.
        log_scale (bool or tuple or list) : Scale spec passed to ``set_scale``.
            Defaults to ``True``.
        colors (list or dict or None) : Color spec passed to ``get_colors``.
            Defaults to ``None``.
        legend (dict or None) : Legend spec passed to ``build_legend``.
            Defaults to ``None``.
        x_axis_label (str or None) : X-axis label; falls back to ``column_one``.
            Defaults to ``None``.
        y_axis_label (str or None) : Y-axis label; falls back to ``column_two``.
            Defaults to ``None``.
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
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    for i, condition_id in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition_id]

        if remove_hatch:
            column_one_values = condition_dict[column_one][:, 1:]
            column_two_values = condition_dict[column_two][:, 1:]
        else:
            column_one_values = condition_dict[column_one]
            column_two_values = condition_dict[column_two]

        x = np.nanmean(column_one_values, axis=0)
        x_std = np.nanstd(column_one_values, axis=0)

        y = np.nanmean(column_two_values, axis=0)
        y_std = np.nanstd(column_two_values, axis=0)

        label = build_legend(condition_dict, legend)
        plt.errorbar(
            x,
            y,
            xerr=x_std,
            yerr=y_std,
            fmt="o",
            color=color_palette[i],
            label=label,
            capsize=3,
        )
        plt.plot(x, y, color=color_palette[i])

    if x_axis_label is not None:
        plt.xlabel(x_axis_label)
    else:
        plt.xlabel(column_one)

    if y_axis_label is not None:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel(column_two)

    set_scale(plt.gca(), log_scale)

    add_legend(placement=legend_placement)

    fig = plt.gcf()
    plt.show()

    return fig


def plot_continuous_deviation_from_model(
    conditions_struct: list,
    rescaled_column_one: str,
    rescaled_column_two: str,
    control_condition_id: int,
    conditions_to_plot: list,
    display_scale: Literal["log", "percent"] = "percent",
    colors: list | dict | None = None,
    log_scale: tuple[bool, bool] | bool = (True, False),
    legend: dict | None = None,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    sort_values: bool = False,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "best",
    deviation_as_percentage: bool | None = None,
) -> matplotlib.figure.Figure:
    """
    Plot the deviation from a LOWESS model as a continuous line across the rescaled axis.

    A LOWESS model is fitted on the control condition's log-log data; all conditions
    (including the control) are then plotted as mean ± 95% CI of their per-worm
    deviations from that model.

    Deviations are log ratios; the mean and standard error are always computed on
    the log scale. With ``display_scale="percent"`` they are back-transformed with
    ``(exp(x) - 1) * 100``, so the plotted center is a geometric-mean-based relative
    deviation and the error bars are asymmetric.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        rescaled_column_one (str) : Key of the rescaled X series.
        rescaled_column_two (str) : Key of the rescaled Y series.
        control_condition_id (int) : Index of the control condition used to fit the model.
        conditions_to_plot (list) : Ordered condition identifiers.
        display_scale (str) : Axis scale, ``"percent"`` or ``"log"`` (log ratio).
            Only affects display; statistics are computed on log ratios.
            Defaults to ``"percent"``.
        colors (list or dict or None) : Color spec passed to ``get_colors``.
            Defaults to ``None``.
        log_scale (tuple[bool, bool] or bool) : Scale spec passed to ``set_scale``.
            Defaults to ``(True, False)``.
        legend (dict or None) : Legend spec passed to ``build_legend``.
            Defaults to ``None``.
        x_axis_label (str or None) : X-axis label; falls back to ``rescaled_column_one``.
            Defaults to ``None``.
        y_axis_label (str or None) : Y-axis label; auto-generated when ``None``.
            Defaults to ``None``.
        sort_values (bool) : If ``True``, sort both the x and residual arrays by x
            before averaging.  Defaults to ``False``.
        ax_size (tuple[float, float] or None) : If provided, fixes the axes area to
            ``(ax_w, ax_h)`` inches. Defaults to ``None``.
        legend_placement (str or None) : Legend placement passed to ``add_legend``;
            ``None`` hides the legend.  Defaults to ``"best"``.
        deviation_as_percentage (bool or None) : Deprecated; ``True`` maps to
            ``display_scale="percent"`` and ``False`` to ``"log"``.
            Defaults to ``None``.

    Returns:
        matplotlib.figure.Figure : The generated figure.

    Raises:
        ValueError : If ``display_scale`` is not ``"log"`` or ``"percent"``.
    """
    display_scale = _resolve_deviation_scale(
        display_scale,
        "display_scale",
        "deviation_as_percentage",
        deviation_as_percentage,
    )
    if ax_size is not None:
        create_fixed_ax_sized_fig(
            ax_w=ax_size[0], ax_h=ax_size[1]
        )  # sets pyplot current figure/axes
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    xlbl = rescaled_column_one
    ylbl = rescaled_column_two

    x_axis_label = x_axis_label if x_axis_label is not None else xlbl
    y_axis_label = (
        y_axis_label
        if y_axis_label is not None
        else _default_deviation_label(rescaled_column_two, display_scale)
    )

    control_condition = conditions_struct[control_condition_id]

    control_model = _get_continuous_proportion_model(
        control_condition[rescaled_column_one],
        control_condition[rescaled_column_two],
        x_axis_label=xlbl,
        y_axis_label=ylbl,
        plot_model=True,
    )

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]
        rescaled_column_one_values, rescaled_column_two_values = (
            condition[rescaled_column_one],
            condition[rescaled_column_two],
        )
        residuals = get_deviation_from_model(
            rescaled_column_one_values,
            rescaled_column_two_values,
            control_model,
            deviation_scale="log",
        )

        if sort_values:
            sorted_indices = np.argsort(rescaled_column_one_values, axis=1)
            rescaled_column_one_values = np.take_along_axis(
                rescaled_column_one_values, sorted_indices, axis=1
            )
            residuals = np.take_along_axis(residuals, sorted_indices, axis=1)

        average_column_one_values = np.nanmean(rescaled_column_one_values, axis=0)
        average_residuals, lower_residuals, upper_residuals = _summarize_log_deviations(
            residuals, display_scale, n_standard_errors=1.96
        )

        label = build_legend(condition, legend)
        plt.plot(
            average_column_one_values,
            average_residuals,
            label=label,
            color=color_palette[i],
        )
        plt.fill_between(
            average_column_one_values,
            lower_residuals,
            upper_residuals,
            color=color_palette[i],
            alpha=0.2,
        )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    set_scale(plt.gca(), log_scale)

    add_legend(placement=legend_placement)

    fig = plt.gcf()
    plt.show()

    return fig


def plot_deviation_from_model_at_ecdysis(
    conditions_struct: list,
    column_one: str,
    column_two: str,
    control_condition_id: int,
    conditions_to_plot: list,
    remove_hatch: bool = False,
    display_scale: Literal["log", "percent"] = "percent",
    log_scale: tuple[bool, bool] | bool = (True, False),
    colors: list | dict | None = None,
    legend: dict | None = None,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    poly_degree: int = 1,
    remove_outliers_fitting: bool | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "best",
    deviation_as_percentage: bool | None = None,
) -> matplotlib.figure.Figure:
    """
    Plot the per-condition deviation from a polynomial model at each molt event.

    A polynomial OLS model is fitted on the control condition; for each other
    condition, the mean deviation and its standard error are plotted as a line with
    error bars over the mean X values at each molt.

    Deviations are log ratios; the mean and standard error are always computed on
    the log scale. With ``display_scale="percent"`` they are back-transformed with
    ``(exp(x) - 1) * 100``, so the plotted center is a geometric-mean-based relative
    deviation and the error bars are asymmetric.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        column_one (str) : Key of the X measurement (per-molt array).
        column_two (str) : Key of the Y measurement (per-molt array).
        control_condition_id (int) : Index of the control condition used to fit the model.
        conditions_to_plot (list) : Ordered condition identifiers.
        remove_hatch (bool) : If ``True``, drop the hatch column (index 0).
            Defaults to ``False``.
        display_scale (str) : Axis scale, ``"percent"`` or ``"log"`` (log ratio).
            Only affects display; statistics are computed on log ratios.
            Defaults to ``"percent"``.
        log_scale (tuple[bool, bool] or bool) : Scale spec passed to ``set_scale``.
            Defaults to ``(True, False)``.
        colors (list or dict or None) : Color spec passed to ``get_colors``.
            Defaults to ``None``.
        legend (dict or None) : Legend spec passed to ``build_legend``.
            Defaults to ``None``.
        x_axis_label (str or None) : X-axis label; falls back to ``column_one``.
            Defaults to ``None``.
        y_axis_label (str or None) : Y-axis label; auto-generated when ``None``.
            Defaults to ``None``.
        poly_degree (int) : Polynomial degree for model fitting.  Defaults to ``1``.
        remove_outliers_fitting (bool or None) : Removed; ignored with a
            ``FutureWarning`` and will raise an error in a future release.
            Defaults to ``None``.
        ax_size (tuple[float, float] or None) : If provided, fixes the axes area to
            ``(ax_w, ax_h)`` inches. Defaults to ``None``.
        legend_placement (str or None) : Legend placement passed to ``add_legend``;
            ``None`` hides the legend.  Defaults to ``"best"``.
        deviation_as_percentage (bool or None) : Deprecated; ``True`` maps to
            ``display_scale="percent"`` and ``False`` to ``"log"``.
            Defaults to ``None``.

    Returns:
        matplotlib.figure.Figure : The generated figure.

    Raises:
        ValueError : If ``display_scale`` is not ``"log"`` or ``"percent"``.
    """
    display_scale = _resolve_deviation_scale(
        display_scale,
        "display_scale",
        "deviation_as_percentage",
        deviation_as_percentage,
    )
    _warn_remove_outliers_fitting_removed(remove_outliers_fitting)

    if ax_size is not None:
        create_fixed_ax_sized_fig(
            ax_w=ax_size[0], ax_h=ax_size[1]
        )  # sets pyplot current figure/axes
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    xlbl = column_one
    ylbl = column_two

    x_axis_label = x_axis_label if x_axis_label is not None else xlbl
    y_axis_label = (
        y_axis_label
        if y_axis_label is not None
        else _default_deviation_label(column_two, display_scale)
    )

    control_condition = conditions_struct[control_condition_id]

    column_one_values = control_condition[column_one]
    column_two_values = control_condition[column_two]

    if remove_hatch:
        column_one_values = column_one_values[:, 1:]
        column_two_values = column_two_values[:, 1:]

    control_model = _get_proportion_model(
        column_one_values,
        column_two_values,
        x_axis_label=xlbl,
        y_axis_label=ylbl,
        poly_degree=poly_degree,
        plot_model=True,
    )

    # figure got used for the control model, create a new one for the deviations plot
    if ax_size is not None:
        create_fixed_ax_sized_fig(
            ax_w=ax_size[0], ax_h=ax_size[1]
        )  # sets pyplot current figure/axes

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]
        column_one_values, column_two_values = (
            condition[column_one],
            condition[column_two],
        )
        if remove_hatch:
            column_one_values = column_one_values[:, 1:]
            column_two_values = column_two_values[:, 1:]
        deviations = get_deviation_from_model(
            column_one_values,
            column_two_values,
            control_model,
            deviation_scale="log",
        )

        mean_column_one_values = np.nanmean(column_one_values, axis=0)
        mean_deviations, lower_deviations, upper_deviations = _summarize_log_deviations(
            deviations, display_scale
        )

        label = build_legend(condition, legend)
        plt.plot(
            mean_column_one_values,
            mean_deviations,
            label=label,
            color=color_palette[i],
            marker="o",
        )
        plt.errorbar(
            mean_column_one_values,
            mean_deviations,
            yerr=[
                mean_deviations - lower_deviations,
                upper_deviations - mean_deviations,
            ],
            color=color_palette[i],
            fmt="o",
            capsize=3,
        )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    set_scale(plt.gca(), log_scale)

    add_legend(placement=legend_placement)

    fig = plt.gcf()
    plt.show()

    return fig


def plot_deviation_from_model_development_percentage(
    conditions_struct: list,
    column_one: str,
    column_two: str,
    control_condition_id: int,
    conditions_to_plot: list,
    percentages: np.ndarray,
    display_scale: Literal["log", "percent"] = "percent",
    log_scale: tuple[bool, bool] | bool = (True, False),
    colors: list | dict | None = None,
    legend: dict | None = None,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    poly_degree: int = 1,
    remove_outliers_fitting: bool | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "best",
    deviation_as_percentage: bool | None = None,
) -> matplotlib.figure.Figure:
    """
    Plot the deviation from a polynomial model at specified development percentages.

    A polynomial OLS model is fitted on the control condition at the sampled
    percentages; for each other condition, the mean deviation and its standard error
    are plotted with error bars over the mean X values at those percentages.

    Deviations are log ratios; the mean and standard error are always computed on
    the log scale. With ``display_scale="percent"`` they are back-transformed with
    ``(exp(x) - 1) * 100``, so the plotted center is a geometric-mean-based relative
    deviation and the error bars are asymmetric.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        column_one (str) : Key of the rescaled X measurement series.
        column_two (str) : Key of the rescaled Y measurement series.
        control_condition_id (int) : Index of the control condition used to fit the model.
        conditions_to_plot (list) : Ordered condition identifiers.
        percentages (np.ndarray) : Fractional development positions (0–1) at which
            to sample and fit the model.
        display_scale (str) : Axis scale, ``"percent"`` or ``"log"`` (log ratio).
            Only affects display; statistics are computed on log ratios.
            Defaults to ``"percent"``.
        log_scale (tuple[bool, bool] or bool) : Scale spec passed to ``set_scale``.
            Defaults to ``(True, False)``.
        colors (list or dict or None) : Color spec passed to ``get_colors``.
            Defaults to ``None``.
        legend (dict or None) : Legend spec passed to ``build_legend``.
            Defaults to ``None``.
        x_axis_label (str or None) : X-axis label; falls back to ``column_one``.
            Defaults to ``None``.
        y_axis_label (str or None) : Y-axis label; auto-generated when ``None``.
            Defaults to ``None``.
        poly_degree (int) : Polynomial degree for model fitting.  Defaults to ``1``.
        remove_outliers_fitting (bool or None) : Removed; ignored with a
            ``FutureWarning`` and will raise an error in a future release.
            Defaults to ``None``.
        ax_size (tuple[float, float] or None) : If provided, fixes the axes area to
            ``(ax_w, ax_h)`` inches. Defaults to ``None``.
        legend_placement (str or None) : Legend placement passed to ``add_legend``;
            ``None`` hides the legend.  Defaults to ``"best"``.
        deviation_as_percentage (bool or None) : Deprecated; ``True`` maps to
            ``display_scale="percent"`` and ``False`` to ``"log"``.
            Defaults to ``None``.

    Returns:
        matplotlib.figure.Figure : The generated figure.

    Raises:
        ValueError : If ``display_scale`` is not ``"log"`` or ``"percent"``.
    """
    display_scale = _resolve_deviation_scale(
        display_scale,
        "display_scale",
        "deviation_as_percentage",
        deviation_as_percentage,
    )
    _warn_remove_outliers_fitting_removed(remove_outliers_fitting)

    if ax_size is not None:
        create_fixed_ax_sized_fig(
            ax_w=ax_size[0], ax_h=ax_size[1]
        )  # sets pyplot current figure/axes
    color_palette = get_colors(conditions_to_plot, colors)

    xlbl = column_one
    ylbl = column_two

    x_axis_label = x_axis_label if x_axis_label is not None else xlbl
    y_axis_label = (
        y_axis_label
        if y_axis_label is not None
        else _default_deviation_label(column_two, display_scale)
    )

    control_condition = conditions_struct[control_condition_id]
    column_one_values = control_condition[column_one]
    column_two_values = control_condition[column_two]

    indices = np.clip(
        (percentages * column_one_values.shape[1]).astype(int),
        0,
        column_one_values.shape[1] - 1,
    ).astype(int)

    control_one_values = column_one_values[:, indices]
    control_two_values = column_two_values[:, indices]

    # Fit the model on the control condition
    control_model = _get_proportion_model(
        control_one_values,
        control_two_values,
        x_axis_label=xlbl,
        y_axis_label=ylbl,
        poly_degree=poly_degree,
        plot_model=True,
    )

    # figure got used for the control model, create a new one for the deviations plot
    if ax_size is not None:
        create_fixed_ax_sized_fig(
            ax_w=ax_size[0], ax_h=ax_size[1]
        )  # sets pyplot current figure/axes

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]
        column_one_values, column_two_values = (
            condition[column_one],
            condition[column_two],
        )
        column_one_values = column_one_values[:, indices]
        column_two_values = column_two_values[:, indices]

        deviations = get_deviation_from_model(
            column_one_values,
            column_two_values,
            control_model,
            deviation_scale="log",
        )

        mean_column_one_values = np.nanmean(column_one_values, axis=0)
        mean_deviations, lower_deviations, upper_deviations = _summarize_log_deviations(
            deviations, display_scale
        )

        label = build_legend(condition, legend)
        plt.plot(
            mean_column_one_values,
            mean_deviations,
            label=label,
            color=color_palette[i],
            marker="o",
        )
        plt.errorbar(
            mean_column_one_values,
            mean_deviations,
            yerr=[
                mean_deviations - lower_deviations,
                upper_deviations - mean_deviations,
            ],
            fmt="o",
            capsize=3,
            color=color_palette[i],
        )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    add_legend(placement=legend_placement)
    set_scale(plt.gca(), log_scale)
    fig = plt.gcf()
    plt.show()
    return fig


def plot_normalized_proportions_at_ecdysis(
    conditions_struct: list,
    column_one: str,
    column_two: str,
    control_condition_id: int,
    conditions_to_plot: list,
    colors: list | dict | None = None,
    aggregation: str = "mean",
    log_scale: tuple[bool, bool] | bool = (True, False),
    legend: dict | None = None,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "best",
) -> matplotlib.figure.Figure:
    """
    Plot the column_two/column_one ratio normalised to the control at each molt event.

    The control proportion (mean ratio across worms) is computed first; each condition's
    ratio is divided by the control proportion before averaging.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        column_one (str) : Key of the denominator measurement (per-molt array).
        column_two (str) : Key of the numerator measurement (per-molt array).
        control_condition_id (int) : Index of the control condition used for normalisation.
        conditions_to_plot (list) : Ordered condition identifiers.
        colors (list or dict or None) : Color spec passed to ``get_colors``.
            Defaults to ``None``.
        aggregation (str) : Aggregation function; currently only ``"mean"`` is used.
            Defaults to ``"mean"``.
        log_scale (tuple[bool, bool] or bool) : Scale spec passed to ``set_scale``.
            Defaults to ``(True, False)``.
        legend (dict or None) : Legend spec passed to ``build_legend``.
            Defaults to ``None``.
        x_axis_label (str or None) : X-axis label; falls back to ``column_one``.
            Defaults to ``None``.
        y_axis_label (str or None) : Y-axis label; auto-generated when ``None``.
            Defaults to ``None``.
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
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )
    control_condition = conditions_struct[control_condition_id]
    control_column_one, control_column_two = (
        control_condition[column_one],
        control_condition[column_two],
    )

    aggregation_function = np.nanmean
    control_proportion = aggregation_function(
        control_column_two / control_column_one, axis=0
    )

    x_axis_label = x_axis_label if x_axis_label is not None else column_one
    y_axis_label = (
        y_axis_label
        if y_axis_label is not None
        else f"normalized {column_two} to {column_one} ratio"
    )

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]

        condition_column_one = condition[column_one]
        condition_column_two = condition[column_two]

        proportion = condition_column_two / condition_column_one
        normalized_proportion = proportion / control_proportion

        y = aggregation_function(normalized_proportion, axis=0)
        y_err = np.nanstd(normalized_proportion, axis=0) / np.sqrt(
            len(normalized_proportion)
        )
        x = aggregation_function(condition_column_one, axis=0)

        label = build_legend(condition, legend)

        plt.plot(x, y, label=label, color=color_palette[i])
        plt.errorbar(x, y, yerr=y_err, fmt="o", capsize=3, color=color_palette[i])

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    set_scale(plt.gca(), log_scale)

    add_legend(placement=legend_placement)

    fig = plt.gcf()
    plt.show()

    return fig


def compute_deviation_from_model_at_ecdysis(
    conditions_struct: list,
    column_one: str,
    column_two: str,
    control_condition: int,
    output_column_name: str,
    remove_hatch: bool = True,
    deviation_scale: Literal["log", "percent"] = "log",
    poly_degree: int = 1,
    remove_outliers_fitting: bool | None = None,
    deviations_as_percentage: bool | None = None,
) -> list:
    """
    Compute per-worm deviations from a control-fitted model and store them in conditions_struct.

    A polynomial model is fitted on the control condition; deviations for all
    conditions (including control) are stored under ``output_column_name``.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        column_one (str) : Key of the X measurement (per-molt array).
        column_two (str) : Key of the Y measurement (per-molt array).
        control_condition (int) : Index of the control condition used to fit the model.
        output_column_name (str) : Key under which deviations are stored.
        remove_hatch (bool) : If ``True``, drop the hatch column (index 0) before
            fitting and computing deviations.  Defaults to ``True``.
        deviation_scale (str) : ``"log"`` stores natural-log ratios, the scale on
            which downstream statistics should run; ``"percent"`` stores
            ``(exp(r) - 1) * 100``.  Defaults to ``"log"``.
        poly_degree (int) : Polynomial degree for model fitting.  Defaults to ``1``.
        remove_outliers_fitting (bool or None) : Removed; ignored with a
            ``FutureWarning`` and will raise an error in a future release.
            Defaults to ``None``.
        deviations_as_percentage (bool or None) : Deprecated; ``True`` maps to
            ``deviation_scale="percent"`` and ``False`` to ``"log"``.
            Defaults to ``None``.

    Returns:
        list : The modified ``conditions_struct`` with deviations added in place.

    Raises:
        ValueError : If ``deviation_scale`` is not ``"log"`` or ``"percent"``.
    """
    _warn_remove_outliers_fitting_removed(remove_outliers_fitting)
    deviation_scale = _resolve_deviation_scale(
        deviation_scale,
        "deviation_scale",
        "deviations_as_percentage",
        deviations_as_percentage,
    )

    control_condition = conditions_struct[control_condition]
    control_column_one_values = control_condition[column_one]
    control_column_two_values = control_condition[column_two]

    if remove_hatch:
        control_column_one_values = control_column_one_values[:, 1:]
        control_column_two_values = control_column_two_values[:, 1:]

    control_model = _get_proportion_model(
        control_column_one_values,
        control_column_two_values,
        poly_degree=poly_degree,
        plot_model=True,
    )

    for condition in conditions_struct:
        column_one_values, column_two_values = (
            condition[column_one],
            condition[column_two],
        )
        if remove_hatch:
            column_one_values = column_one_values[:, 1:]
            column_two_values = column_two_values[:, 1:]

        deviations = get_deviation_from_model(
            column_one_values,
            column_two_values,
            control_model,
            deviation_scale=deviation_scale,
        )
        condition[output_column_name] = deviations

    return conditions_struct


def compute_deviation_from_each_model_at_ecdysis(
    conditions_struct: list,
    column_one: str,
    column_two: str,
    output_column_name: str,
    remove_hatch: bool = True,
    deviation_scale: Literal["log", "percent"] = "log",
    poly_degree: int = 1,
    remove_outliers_fitting: bool | None = None,
    deviations_as_percentage: bool | None = None,
) -> list:
    """
    Fit a separate model per condition and store each condition's self-deviation.

    Unlike ``compute_deviation_from_model_at_ecdysis``, the model is refitted
    independently for each condition using that condition's own data.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        column_one (str) : Key of the X measurement (per-molt array).
        column_two (str) : Key of the Y measurement (per-molt array).
        output_column_name (str) : Key under which deviations are stored.
        remove_hatch (bool) : If ``True``, drop the hatch column (index 0) before
            fitting.  Defaults to ``True``.
        deviation_scale (str) : ``"log"`` stores natural-log ratios, the scale on
            which downstream statistics should run; ``"percent"`` stores
            ``(exp(r) - 1) * 100``.  Defaults to ``"log"``.
        poly_degree (int) : Polynomial degree for model fitting.  Defaults to ``1``.
        remove_outliers_fitting (bool or None) : Removed; ignored with a
            ``FutureWarning`` and will raise an error in a future release.
            Defaults to ``None``.
        deviations_as_percentage (bool or None) : Deprecated; ``True`` maps to
            ``deviation_scale="percent"`` and ``False`` to ``"log"``.
            Defaults to ``None``.

    Returns:
        list : The modified ``conditions_struct`` with deviations added in place.

    Raises:
        ValueError : If ``deviation_scale`` is not ``"log"`` or ``"percent"``.
    """
    _warn_remove_outliers_fitting_removed(remove_outliers_fitting)
    deviation_scale = _resolve_deviation_scale(
        deviation_scale,
        "deviation_scale",
        "deviations_as_percentage",
        deviations_as_percentage,
    )

    for condition in conditions_struct:
        column_one_values, column_two_values = (
            condition[column_one],
            condition[column_two],
        )
        if remove_hatch:
            column_one_values = column_one_values[:, 1:]
            column_two_values = column_two_values[:, 1:]

        try:
            model = _get_proportion_model(
                column_one_values,
                column_two_values,
                poly_degree=poly_degree,
                plot_model=False,
            )

            deviations = get_deviation_from_model(
                column_one_values,
                column_two_values,
                model,
                deviation_scale=deviation_scale,
            )
        except Exception as e:
            print(f"Error fitting model for condition {condition['description']}: {e}")
            deviations = np.full_like(column_one_values, np.nan, dtype=float)
        condition[output_column_name] = deviations

    return conditions_struct


def compute_deviation_from_model_development_percentage(
    conditions_struct: list,
    column_one: str,
    column_two: str,
    control_condition: int,
    percentages: np.ndarray,
    output_column_name: str,
    deviation_scale: Literal["log", "percent"] = "log",
    poly_degree: int = 1,
    remove_outliers_fitting: bool | None = None,
    deviations_as_percentage: bool | None = None,
) -> list:
    """
    Compute per-worm deviations from a control model sampled at development percentages.

    The model is fitted on the control condition at the sampled indices; deviations
    for all conditions are stored under ``output_column_name``.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        column_one (str) : Key of the rescaled X measurement series.
        column_two (str) : Key of the rescaled Y measurement series.
        control_condition (int) : Index of the control condition used to fit the model.
        percentages (np.ndarray) : Fractional development positions (0–1) at which
            to sample and fit the model.
        output_column_name (str) : Key under which deviations are stored.
        deviation_scale (str) : ``"log"`` stores natural-log ratios, the scale on
            which downstream statistics should run; ``"percent"`` stores
            ``(exp(r) - 1) * 100``.  Defaults to ``"log"``.
        poly_degree (int) : Polynomial degree for model fitting.  Defaults to ``1``.
        remove_outliers_fitting (bool or None) : Removed; ignored with a
            ``FutureWarning`` and will raise an error in a future release.
            Defaults to ``None``.
        deviations_as_percentage (bool or None) : Deprecated; ``True`` maps to
            ``deviation_scale="percent"`` and ``False`` to ``"log"``.
            Defaults to ``None``.

    Returns:
        list : The modified ``conditions_struct`` with deviations added in place.

    Raises:
        ValueError : If ``deviation_scale`` is not ``"log"`` or ``"percent"``.
    """
    _warn_remove_outliers_fitting_removed(remove_outliers_fitting)
    deviation_scale = _resolve_deviation_scale(
        deviation_scale,
        "deviation_scale",
        "deviations_as_percentage",
        deviations_as_percentage,
    )

    control_condition = conditions_struct[control_condition]
    control_column_one_values = control_condition[column_one]
    control_column_two_values = control_condition[column_two]

    indices = np.clip(
        (percentages * control_column_one_values.shape[1]).astype(int),
        0,
        control_column_one_values.shape[1] - 1,
    ).astype(int)

    control_column_one_values = control_column_one_values[:, indices]
    control_column_two_values = control_column_two_values[:, indices]

    control_model = _get_proportion_model(
        control_column_one_values,
        control_column_two_values,
        poly_degree=poly_degree,
        plot_model=True,
    )

    for condition in conditions_struct:
        column_one_values, column_two_values = (
            condition[column_one],
            condition[column_two],
        )
        column_one_values = column_one_values[:, indices]
        column_two_values = column_two_values[:, indices]

        deviations = get_deviation_from_model(
            column_one_values,
            column_two_values,
            control_model,
            deviation_scale=deviation_scale,
        )
        condition[output_column_name] = deviations

    return conditions_struct
