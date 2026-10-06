from math import ceil
from typing import Any, Literal

import matplotlib.axes
import matplotlib.axis
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import (
    FuncFormatter,
    LogFormatterSciNotation,
    LogLocator,
    NullFormatter,
    PercentFormatter,
)
from scipy.stats import gaussian_kde

from align_toolbox.data_analysis.proportion_model import (
    SINGLE_FACTOR_NAME,
    GenotypeCoding,
    ProportionModelResult,
    _add_timepoint_columns,
    compare_experiments,
)

from .boxplots import _plot_violinplot
from .proportions import log_ratio_to_percentage
from .utils_plotting import add_legend, create_fixed_ax_sized_fig, get_colors

DEFAULT_PALETTE = "colorblind"
REFERENCE_COLOR = "grey"
REFERENCE_CURVE_COLOR = "black"
INTERVAL_LABEL = "95% CrI"

DEFAULT_AX_SIZE = (3.5, 3.0)
PANEL_AX_SIZE = (2.6, 2.6)
PANEL_WSPACE = 0.8
PANEL_HSPACE = 0.9
PANEL_TOP = 0.5
DEFAULT_PANEL_COLUMNS = 4

DEFAULT_X_NAME = "Body size"
DEFAULT_Y_NAME = "Organ size"
DEFAULT_KINDS = ("beta",)
DISPLAYS = ("percent", "log")
RATIO_KINDS = ("tau_ratio", "sigma_ratio")

EXPERIMENT_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">")
MISFIT_STYLES = {"line": ("o", "0.15"), "spline": ("s", "0.55")}
REPLICATE_COLOR = "0.6"

CellSpec = int | str | dict[str, str]
TimepointSpec = int | float | str


# SHARED HELPERS


def _default_cell_label(coding: GenotypeCoding, cell: str) -> str:
    """Display name of a cell: ``"Condition <id>"`` under single-factor coding, else the cell label."""
    if coding.factor_names == [SINGLE_FACTOR_NAME]:
        return f"Condition {coding.resolve_cell(cell)[SINGLE_FACTOR_NAME]}"
    return cell


def _values_by_cell(
    coding: GenotypeCoding, values: dict | list | None, name: str
) -> dict[str, Any]:
    """
    Key user-supplied per-cell values by cell label.

    Parameters:
        coding (GenotypeCoding): Coding of the fit.
        values (dict, list or None): Dict keyed by condition id or cell label, or a
            list with one value per cell in ``coding.cells`` order.
        name (str): Argument name, for error messages.

    Returns:
        dict[str, Any]: Values keyed by cell label; empty for ``None``.

    Raises:
        ValueError: If a key designates no fitted cell or a list has the wrong
            length.
    """
    if values is None:
        return {}
    if isinstance(values, dict):
        by_cell = {}
        for key, value in values.items():
            try:
                cell = coding.cell_label(coding.resolve_cell(key))
            except (ValueError, TypeError, KeyError):
                raise ValueError(
                    f"`{name}` key {key!r} is neither a condition id nor a cell label "
                    f"of the fit; its cells are {coding.cells}."
                ) from None
            by_cell[cell] = value
        return by_cell
    values = list(values)
    if len(values) != len(coding.cells):
        raise ValueError(
            f"`{name}` has {len(values)} entries but the fit has "
            f"{len(coding.cells)} cells {coding.cells}."
        )
    return dict(zip(coding.cells, values))


def _cell_styles(
    result: ProportionModelResult,
    colors: dict | list | None,
    labels: dict | list | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Map every cell of the fit to a color and a display name.

    Defaults give the reference cell ``REFERENCE_COLOR`` and the other cells the
    ``DEFAULT_PALETTE`` palette in ``coding.cells`` order, so a genotype keeps its
    color in every plot of the same fit. User values override the defaults cell
    by cell.

    Parameters:
        result (ProportionModelResult): Fitted model.
        colors (dict, list or None): Colors keyed by condition id or cell label,
            or one per cell in ``coding.cells`` order.
        labels (dict, list or None): Display names, in the same forms as
            ``colors``.

    Returns:
        tuple[dict[str, Any], dict[str, str]]: Color and display name of each
            cell label.
    """
    coding = result.coding
    mutants = coding.cells[1:]
    color_of = {coding.reference_cell: REFERENCE_COLOR}
    color_of.update(zip(mutants, get_colors(mutants, None, DEFAULT_PALETTE)))
    color_of.update(_values_by_cell(coding, colors, "colors"))
    label_of = {cell: _default_cell_label(coding, cell) for cell in coding.cells}
    label_of.update(_values_by_cell(coding, labels, "labels"))
    return color_of, label_of


def _resolve_cells(
    result: ProportionModelResult,
    cells: list[CellSpec] | None,
    include_reference: bool = True,
) -> list[str]:
    """Labels of the requested cells (all when ``None``), in ``coding.cells`` order."""
    coding = result.coding
    if cells is None:
        selected = set(coding.cells)
    else:
        selected = {coding.cell_label(coding.resolve_cell(c)) for c in cells}
    if not include_reference:
        selected.discard(coding.reference_cell)
    return [cell for cell in coding.cells if cell in selected]


def _row_cells(result: ProportionModelResult) -> np.ndarray:
    """Cell label of each row of ``result.table``."""
    return result.table["condition_id"].map(result.coding.cell_of_condition).to_numpy()


def _reference_range(result: ProportionModelResult) -> tuple[float, float]:
    """Min and max ``log_x`` of the reference cell's observations."""
    log_x = result.table["log_x"].to_numpy()[
        _row_cells(result) == result.coding.reference_cell
    ]
    return float(log_x.min()), float(log_x.max())


def _centroid_range(result: ProportionModelResult, cell: str) -> tuple[float, float]:
    """Min and max over timepoints of a cell's mean ``log_x`` at each timepoint."""
    rows = _row_cells(result) == cell
    centroids = result.table.loc[rows].groupby("timepoint")["log_x"].mean()
    return float(centroids.min()), float(centroids.max())


def _measurement_names(result: ProportionModelResult) -> tuple[str, str]:
    """Names of X and Y from ``table.attrs`` (``"column_x"``, ``"column_y"``), else generic ones."""
    attrs = result.table.attrs
    return attrs.get("column_x", DEFAULT_X_NAME), attrs.get("column_y", DEFAULT_Y_NAME)


def _timepoint_noun(result: ProportionModelResult) -> str:
    """``"molt"`` for ecdysis sampling, else ``"timepoint"``."""
    return "molt" if result.sampling == "ecdysis" else "timepoint"


def _timepoint_axis_label(result: ProportionModelResult) -> str:
    """Default label of an axis of timepoints."""
    return "Molt" if result.sampling == "ecdysis" else "Development"


def _grid_with_bounds(
    lower: float, upper: float, n_grid: int, bounds: tuple[float, ...]
) -> np.ndarray:
    """Evenly spaced grid on ``[lower, upper]`` that also contains the ``bounds`` inside it."""
    grid = np.linspace(lower, upper, n_grid)
    inside = [b for b in bounds if lower < b < upper]
    return np.unique(np.concatenate([grid, inside]))


def _interval_errors(
    center: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Error-bar lengths of shape ``(2, n)`` from interval bounds, clipped at 0."""
    center, lower, upper = (
        np.atleast_1d(np.asarray(v, float)) for v in (center, lower, upper)
    )
    return np.vstack([np.maximum(center - lower, 0), np.maximum(upper - center, 0)])


def _plot_segments(
    ax: matplotlib.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    solid: np.ndarray,
    label: str | None = None,
    **line_kwargs,
) -> list[Line2D]:
    """
    Draw a curve solid where ``solid`` is ``True`` and dashed elsewhere.

    Dashed runs extend to the neighbouring solid points so the curve stays
    continuous, while solid runs never reach into dashed territory.

    Parameters:
        ax (matplotlib.axes.Axes): Target axes.
        x (np.ndarray): X values of shape ``(n_points,)``.
        y (np.ndarray): Y values of shape ``(n_points,)``.
        solid (np.ndarray): Boolean mask of shape ``(n_points,)``.
        label (str or None): Legend label, put on the first solid segment, or on
            the first segment when none is solid. (default: None)
        **line_kwargs: Forwarded to ``ax.plot``.

    Returns:
        list[Line2D]: The drawn segments.
    """
    x, y = np.asarray(x), np.asarray(y)
    solid = np.asarray(solid, dtype=bool)
    starts = np.flatnonzero(np.r_[True, solid[1:] != solid[:-1]])
    ends = np.r_[starts[1:], len(x)]
    # the legend shows the curve's solid style unless it is dashed throughout
    labelled = starts[np.argmax(solid[starts])] if solid.any() else 0
    lines = []
    for start, end in zip(starts, ends):
        is_solid = solid[start]
        line_label = label if start == labelled and label is not None else "_nolegend_"
        if not is_solid:
            start, end = max(start - 1, 0), min(end + 1, len(x))
        (line,) = ax.plot(
            x[start:end],
            y[start:end],
            linestyle="-" if is_solid else "--",
            label=line_label,
            **line_kwargs,
        )
        lines.append(line)
    return lines


def _format_log_axis(axis: matplotlib.axis.Axis, plain: bool = False) -> None:
    """
    Label a log axis with readable, non-overlapping ticks for its current range.

    Parameters:
        axis (matplotlib.axis.Axis): A log-scaled axis whose limits are final.
        plain (bool): Write ticks as plain numbers (``0.5``, ``2``), suited to
            ratios, instead of ``2×10⁴`` notation. (default: False)
    """
    lower, upper = axis.get_view_interval()
    decades = np.log10(upper / lower)
    if decades > 3:
        subs = (1.0,)
    elif decades > 0.7:
        subs = (1.0, 2.0, 5.0)
    else:
        subs = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)
    axis.set_major_locator(LogLocator(subs=subs))
    axis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:g}")
        if plain
        else LogFormatterSciNotation(
            labelOnlyBase=False, minor_thresholds=(np.inf, np.inf)
        )
    )
    axis.set_minor_formatter(NullFormatter())


def _single_axes(
    ax: matplotlib.axes.Axes | None,
    ax_size: tuple[float, float] | None,
    default_size: tuple[float, float] = DEFAULT_AX_SIZE,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, bool]:
    """Return ``(figure, axes, owns_figure)``, creating a fixed-size figure unless ``ax`` is given."""
    if ax is not None:
        return ax.figure, ax, False
    ax_w, ax_h = default_size if ax_size is None else ax_size
    fig, ax = create_fixed_ax_sized_fig(ax_w=ax_w, ax_h=ax_h)
    return fig, ax, True


def _panel_axes(
    n_panels: int,
    ax: matplotlib.axes.Axes | np.ndarray | list | None,
    ax_size: tuple[float, float] | None,
    ncols: int | None = None,
    default_size: tuple[float, float] = PANEL_AX_SIZE,
) -> tuple[matplotlib.figure.Figure, np.ndarray, bool]:
    """
    Return ``(figure, axes, owns_figure)`` for a multi-panel plot.

    Parameters:
        n_panels (int): Number of panels needed.
        ax (Axes, np.ndarray, list or None): Axes to draw into; when ``None`` a
            fixed-size grid is created.
        ax_size (tuple[float, float] or None): Size of each new panel in inches.
        ncols (int or None): Columns of a new grid; ``None`` uses up to
            ``DEFAULT_PANEL_COLUMNS``. (default: None)
        default_size (tuple[float, float]): Panel size when ``ax_size`` is
            ``None``. (default: PANEL_AX_SIZE)

    Returns:
        tuple[matplotlib.figure.Figure, np.ndarray, bool]: The figure, a 1-D array
            of ``n_panels`` axes and whether the figure was created here.

    Raises:
        ValueError: If ``ax`` holds fewer than ``n_panels`` axes.
    """
    if ax is not None:
        axes = np.ravel(np.array(ax, dtype=object))
        if len(axes) < n_panels:
            raise ValueError(
                f"`ax` holds {len(axes)} axes but the plot needs {n_panels} panels."
            )
        return axes[0].figure, axes[:n_panels], False
    ncols = min(DEFAULT_PANEL_COLUMNS if ncols is None else ncols, n_panels)
    nrows = ceil(n_panels / ncols)
    ax_w, ax_h = default_size if ax_size is None else ax_size
    fig, axes = create_fixed_ax_sized_fig(
        ax_w=ax_w,
        ax_h=ax_h,
        nrows=nrows,
        ncols=ncols,
        wspace=PANEL_WSPACE,
        hspace=PANEL_HSPACE,
        top=PANEL_TOP,
    )
    axes = np.ravel(np.array(axes, dtype=object))
    for unused in axes[n_panels:]:
        unused.set_axis_off()
    return fig, axes[:n_panels], True


def _add_legend(
    target: matplotlib.axes.Axes | matplotlib.figure.Figure,
    axes: list | np.ndarray,
    placement: str | None,
) -> None:
    """Draw one legend on ``target`` from the labelled artists of ``axes``."""
    handles, labels = [], []
    for panel in axes:
        legend = panel.get_legend()
        if legend is not None:
            legend.remove()
        for handle, label in zip(*panel.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if placement is None or handles:
        add_legend(target, placement, handles, labels)


def _finish(
    fig: matplotlib.figure.Figure,
    owns_figure: bool,
) -> matplotlib.figure.Figure:
    """Show the figure if it was created here and return it."""
    if owns_figure:
        plt.show()
    return fig


def _select_parameters(
    table: pd.DataFrame, parameters: list[str] | str | None
) -> pd.DataFrame:
    """
    Select rows of a summary-like table by parameter name.

    Parameters:
        table (pd.DataFrame): Table with ``parameter`` and ``kind`` columns.
        parameters (list[str], str or None): Exact names (kept in the given
            order), a regular expression searched in each name, or ``None`` for
            the rows whose ``kind`` is in ``DEFAULT_KINDS``.

    Returns:
        pd.DataFrame: The selected rows.

    Raises:
        ValueError: If the table has no ``kind`` column, listed names are
            missing or nothing matches.
    """
    if "kind" not in table.columns:
        raise ValueError(
            "The table has no `kind` column; recompute it with "
            "ProportionModelResult.summary or compare_experiments."
        )
    if parameters is None:
        selected = table[table["kind"].isin(DEFAULT_KINDS)]
    elif isinstance(parameters, str):
        selected = table[table["parameter"].str.contains(parameters, regex=True)]
    else:
        order = {name: i for i, name in enumerate(parameters)}
        missing = [name for name in parameters if name not in set(table["parameter"])]
        if missing:
            raise ValueError(f"Parameters {missing} are not in the table.")
        selected = table[table["parameter"].isin(order)]
        selected = selected.iloc[
            np.argsort(selected["parameter"].map(order).to_numpy(), kind="stable")
        ]
    if selected.empty:
        raise ValueError(f"No parameter matches {parameters!r}.")
    return selected


def _display_values(rows: pd.DataFrame, display: str) -> dict[str, np.ndarray]:
    """``mean``, ``lower`` and ``upper`` to draw: the ``_percent`` columns of beta rows for ``"percent"``, raw values otherwise."""
    is_beta = (rows["kind"] == "beta").to_numpy()
    values = {}
    for key in ("mean", "lower", "upper"):
        values[key] = rows[key].to_numpy(dtype=float)
        if display == "percent":
            percent = rows[f"{key}_percent"].to_numpy(dtype=float)
            values[key] = np.where(is_beta, percent, values[key])
    return values


def _effect_axis(kinds: pd.Series, display: str) -> tuple[str, float]:
    """Default axis label and null value for summary rows of the given ``kind``."""
    is_beta = kinds == "beta"
    if kinds.isin(RATIO_KINDS).all():
        return f"Ratio to reference cell ({INTERVAL_LABEL})", 1.0
    if display == "percent" and is_beta.all():
        return f"Offset at x_ref (%, {INTERVAL_LABEL})", 0.0
    if display == "percent" and is_beta.any():
        return f"Estimate (% for beta, raw otherwise; {INTERVAL_LABEL})", 0.0
    return f"Estimate (natural-log units, {INTERVAL_LABEL})", 0.0


def _check_display(display: str) -> None:
    if display not in DISPLAYS:
        raise ValueError(f"`display` must be one of {DISPLAYS}, got {display!r}.")


def _format_p(p: float) -> str:
    return f"p = {p:.2g}"


# SCALING AND OFFSETS


def plot_proportion_scaling(
    result: ProportionModelResult,
    cells: list[CellSpec] | None = None,
    show_points: bool = False,
    show_reference: bool = True,
    show_predictive_band: bool = True,
    prob: float = 0.95,
    n_grid: int = 200,
    random_seed: int | None = None,
    colors: dict | list | None = None,
    labels: dict | list | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "outside right",
) -> matplotlib.figure.Figure:
    """
    Plot each genotype's per-timepoint geometric means of Y against X on log–log axes, with the reference fit.

    Points are ``exp(mean)`` of ``log_x`` and ``log_y`` from ``timepoint_positions``
    with asymmetric bars ``exp(mean ± sd)``. The reference curve from
    ``reference_curve`` is solid inside the reference cell's ``log_x`` range and
    dashed where it is extrapolated, with its 95% credible band and, lighter, the
    predictive band of a new reference worm from ``reference_predictive_interval``.

    Parameters:
        result (ProportionModelResult): Fitted model.
        cells (list or None): Cells to plot, as condition ids, cell labels or
            factor-level dicts; ``None`` plots all. (default: None)
        show_points (bool): Add the individual observations as faint points.
            (default: False)
        show_reference (bool): Draw the reference curve and its credible band.
            (default: True)
        show_predictive_band (bool): Draw the reference predictive band.
            (default: True)
        prob (float): Probability mass of the predictive band. (default: 0.95)
        n_grid (int): Number of points of the reference curve. (default: 200)
        random_seed (int or None): Seed of the predictive draws; ``None`` reuses
            the sampling seed. (default: None)
        colors (dict, list or None): Cell colors keyed by condition id or cell
            label, or one per cell in ``coding.cells`` order. (default: None)
        labels (dict, list or None): Cell display names, like ``colors``.
            (default: None)
        x_label (str or None): X-axis label; defaults to the X column name.
            (default: None)
        y_label (str or None): Y-axis label; defaults to the Y column name.
            (default: None)
        ax (matplotlib.axes.Axes or None): Axes to draw into; no figure is
            created or shown when given. (default: None)
        ax_size (tuple[float, float] or None): Axes size in inches of a new
            figure. (default: None)
        legend_placement (str or None): Placement passed to ``add_legend``;
            ``None`` hides the legend.
            (default: "outside right")

    Returns:
        matplotlib.figure.Figure: The figure holding the plot.
    """
    color_of, label_of = _cell_styles(result, colors, labels)
    selected = _resolve_cells(result, cells)
    x_name, y_name = _measurement_names(result)
    fig, ax, owns_figure = _single_axes(ax, ax_size)
    row_cells = _row_cells(result)
    positions = result.timepoint_positions()

    for cell in selected:
        color = color_of[cell]
        if show_points:
            rows = result.table[row_cells == cell]
            ax.scatter(
                np.exp(rows["log_x"]),
                np.exp(rows["log_y"]),
                s=6,
                color=color,
                alpha=0.25,
                linewidths=0,
                gid=f"points|{cell}",
            )
        timepoints = positions[positions["cell"] == cell]
        x_mean = timepoints["log_x_mean"].to_numpy()
        y_mean = timepoints["log_y_mean"].to_numpy()
        x_sd = np.nan_to_num(timepoints["log_x_sd"].to_numpy())
        y_sd = np.nan_to_num(timepoints["log_y_sd"].to_numpy())
        ax.errorbar(
            np.exp(x_mean),
            np.exp(y_mean),
            xerr=_interval_errors(
                np.exp(x_mean), np.exp(x_mean - x_sd), np.exp(x_mean + x_sd)
            ),
            yerr=_interval_errors(
                np.exp(y_mean), np.exp(y_mean - y_sd), np.exp(y_mean + y_sd)
            ),
            fmt="o",
            ms=5,
            color=color,
            capsize=2,
            zorder=4,
            label=label_of[cell],
            gid=f"molts|{cell}",
        )

    if show_reference or show_predictive_band:
        log_x = result.table["log_x"].to_numpy()[np.isin(row_cells, selected)]
        grid = _grid_with_bounds(
            log_x.min(), log_x.max(), n_grid, _reference_range(result)
        )
        band_color = color_of[result.coding.reference_cell]
        if show_predictive_band:
            band = result.reference_predictive_interval(
                grid, prob=prob, random_seed=random_seed
            )
            ax.fill_between(
                np.exp(grid),
                np.exp(band["lower"]),
                np.exp(band["upper"]),
                color=band_color,
                alpha=0.12,
                linewidth=0,
                gid="predictive_band",
            )
        if show_reference:
            curve = result.reference_curve(grid)
            ax.fill_between(
                np.exp(grid),
                np.exp(curve["lower"]),
                np.exp(curve["upper"]),
                color=band_color,
                alpha=0.35,
                linewidth=0,
                gid="reference_band",
            )
            _plot_segments(
                ax,
                np.exp(grid),
                np.exp(curve["mean"]),
                ~curve["extrapolated"].to_numpy(),
                color=REFERENCE_CURVE_COLOR,
                linewidth=2,
                zorder=3,
                gid="reference_curve",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    _format_log_axis(ax.xaxis)
    _format_log_axis(ax.yaxis)
    ax.set_xlabel(x_name if x_label is None else x_label)
    ax.set_ylabel(y_name if y_label is None else y_label)
    _add_legend(ax, [ax], legend_placement)
    return _finish(fig, owns_figure)


def plot_offset_curves(
    result: ProportionModelResult,
    cells: list[CellSpec] | None = None,
    markers: Literal["observed", "model"] | None = "observed",
    n_grid: int = 200,
    colors: dict | list | None = None,
    labels: dict | list | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "outside right",
) -> matplotlib.figure.Figure:
    """
    Plot each genotype's size-matched offset from the reference cell, in percent, against body size.

    The reference cell is the line at 0, with a band showing the 95% band of
    ``reference_curve`` around its mean, i.e. how precisely the reference curve
    itself is known. Bands and curves of every cell span only from its first to
    its last timepoint centroid, the cell's mean ``log_x`` at a timepoint. Each
    non-reference cell's ``group_offset_curve`` is solid inside the reference
    cell's overall observed ``log_x`` range, including gaps between reference
    timepoints, and dashed beyond it. Markers sit at each cell's timepoint
    centroids, the reference cell included, show the observed or the model
    offset from ``timepoint_positions`` with its interval, and are joined by a
    faint dotted line.

    Parameters:
        result (ProportionModelResult): Fitted model.
        cells (list or None): Cells to plot, as condition ids, cell labels or
            factor-level dicts; ``None`` plots all. The reference cell is the
            line at 0. (default: None)
        markers (str or None): ``"observed"`` (``observed_offset``), ``"model"``
            (``model_offset``) or ``None`` for no markers. (default: "observed")
        n_grid (int): Number of points of each curve. (default: 200)
        colors (dict, list or None): Cell colors keyed by condition id or cell
            label, or one per cell in ``coding.cells`` order. (default: None)
        labels (dict, list or None): Cell display names, like ``colors``.
            (default: None)
        x_label (str or None): X-axis label; defaults to the X column name.
            (default: None)
        y_label (str or None): Y-axis label. (default: None)
        ax (matplotlib.axes.Axes or None): Axes to draw into; no figure is
            created or shown when given. (default: None)
        ax_size (tuple[float, float] or None): Axes size in inches of a new
            figure. (default: None)
        legend_placement (str or None): Placement passed to ``add_legend``;
            ``None`` hides the legend.
            (default: "outside right")

    Returns:
        matplotlib.figure.Figure: The figure holding the plot.

    Raises:
        ValueError: If ``markers`` is not ``"observed"``, ``"model"`` or ``None``.
    """
    if markers not in ("observed", "model", None):
        raise ValueError(
            f"`markers` must be 'observed', 'model' or None, got {markers!r}."
        )
    color_of, label_of = _cell_styles(result, colors, labels)
    selected = _resolve_cells(result, cells, include_reference=False)
    reference = result.coding.reference_cell
    x_name, y_name = _measurement_names(result)
    fig, ax, owns_figure = _single_axes(ax, ax_size)
    reference_lower, reference_upper = _reference_range(result)
    positions = result.timepoint_positions() if markers is not None else None

    ax.axhline(
        0,
        color=color_of[reference],
        linewidth=2,
        zorder=1,
        label=label_of[reference],
    )
    # uncertainty of the reference curve itself, as a band around its line at 0
    reference_grid = np.linspace(*_centroid_range(result, reference), n_grid)
    reference_curve = result.reference_curve(reference_grid)
    ax.fill_between(
        np.exp(reference_grid),
        log_ratio_to_percentage(reference_curve["lower"] - reference_curve["mean"]),
        log_ratio_to_percentage(reference_curve["upper"] - reference_curve["mean"]),
        color=color_of[reference],
        alpha=0.25,
        linewidth=0,
        gid=f"offset_band|{reference}",
    )
    for cell in selected:
        color = color_of[cell]
        grid = _grid_with_bounds(
            *_centroid_range(result, cell), n_grid, (reference_lower, reference_upper)
        )
        curve = result.group_offset_curve(cell, grid)
        x = np.exp(grid)
        ax.fill_between(
            x,
            log_ratio_to_percentage(curve["lower"]),
            log_ratio_to_percentage(curve["upper"]),
            color=color,
            alpha=0.25,
            linewidth=0,
            gid=f"offset_band|{cell}",
        )
        within = (grid >= reference_lower) & (grid <= reference_upper)
        _plot_segments(
            ax,
            x,
            log_ratio_to_percentage(curve["mean"]),
            within,
            label=label_of[cell],
            color=color,
            linewidth=2,
            gid=f"offset|{cell}",
        )
    if markers is not None:
        prefix = f"{markers}_offset"
        for cell in [reference, *selected]:
            timepoints = positions[positions["cell"] == cell]
            center = timepoints[f"{prefix}_mean_percent"].to_numpy()
            centroids = np.exp(timepoints["log_x_mean"].to_numpy(dtype=float))
            ax.plot(
                centroids,
                center,
                linestyle=":",
                linewidth=1,
                alpha=0.6,
                color=color_of[cell],
                zorder=4,
                gid=f"offset_link|{cell}",
            )
            ax.errorbar(
                centroids,
                center,
                yerr=_interval_errors(
                    center,
                    timepoints[f"{prefix}_lower_percent"],
                    timepoints[f"{prefix}_upper_percent"],
                ),
                fmt="o",
                ms=4,
                color=color_of[cell],
                capsize=2,
                zorder=4,
                gid=f"offset_markers|{cell}",
            )

    ax.set_xscale("log")
    _format_log_axis(ax.xaxis)
    ax.set_xlabel(x_name if x_label is None else x_label)
    ax.set_ylabel(
        f"Size-matched {y_name} offset (%, {INTERVAL_LABEL})"
        if y_label is None
        else y_label
    )
    _add_legend(ax, [ax], legend_placement)
    return _finish(fig, owns_figure)


def plot_genotype_interaction(
    result: ProportionModelResult,
    log_x: float | None = None,
    x_factor: str | None = None,
    line_factor: str | None = None,
    annotate_interaction: bool = False,
    n_draws: int | None = None,
    colors: dict | list | None = None,
    labels: dict | list | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "outside right",
) -> matplotlib.figure.Figure:
    """
    Plot an interaction plot of the size-matched offsets of a two-factor design.

    Offsets come from ``cell_offsets(log_x)``. The levels of ``x_factor`` are on
    the x axis and each level of ``line_factor`` is a line, with points slightly
    dodged. Dashed lines with hollow markers show ``expected``, the offset an
    additive model predicts from the main effects. Cells for which ``log_x`` is
    outside their own observed range are drawn as diamonds, since their offset
    at that size is extrapolated.

    Each line takes the color of its cell at the reference level of
    ``x_factor``, so it matches that genotype in the other plots.

    Parameters:
        result (ProportionModelResult): Fitted model with two-factor coding.
        log_x (float or None): Natural-log body size at which offsets are
            compared; ``None`` uses ``x_ref``. (default: None)
        x_factor (str or None): Factor on the x axis; ``None`` uses the second
            factor. (default: None)
        line_factor (str or None): Factor drawn as lines; ``None`` uses the first
            factor. (default: None)
        annotate_interaction (bool): Write each interaction in percent with its
            interval next to its point. (default: False)
        n_draws (int or None): Posterior draws used by ``cell_offsets``; ``None``
            uses all. (default: None)
        colors (dict, list or None): Cell colors keyed by condition id or cell
            label, or one per cell in ``coding.cells`` order. (default: None)
        labels (dict, list or None): Cell display names, like ``colors``.
            (default: None)
        title (str or None): Axes title; ``None`` uses the default, ``""``
            hides it. (default: None)
        x_label (str or None): X-axis label; defaults to ``x_factor``.
            (default: None)
        y_label (str or None): Y-axis label. (default: None)
        ax (matplotlib.axes.Axes or None): Axes to draw into; no figure is
            created or shown when given. (default: None)
        ax_size (tuple[float, float] or None): Axes size in inches of a new
            figure. (default: None)
        legend_placement (str or None): Placement passed to ``add_legend``;
            ``None`` hides the legend.
            (default: "outside right")

    Returns:
        matplotlib.figure.Figure: The figure holding the plot.

    Raises:
        ValueError: If the coding does not have exactly two factors, or
            ``x_factor`` and ``line_factor`` are not its two distinct factors.
    """
    coding = result.coding
    if len(coding.factor_names) != 2:
        raise ValueError(
            "The interaction plot needs a two-factor coding; the fit has factors "
            f"{coding.factor_names}."
        )
    x_factor = coding.factor_names[1] if x_factor is None else x_factor
    line_factor = coding.factor_names[0] if line_factor is None else line_factor
    if {x_factor, line_factor} != set(coding.factor_names):
        raise ValueError(
            f"`x_factor` ({x_factor!r}) and `line_factor` ({line_factor!r}) must be "
            f"the two factors {coding.factor_names}."
        )
    color_of, label_of = _cell_styles(result, colors, labels)
    x_name, y_name = _measurement_names(result)
    fig, ax, owns_figure = _single_axes(ax, ax_size, default_size=(3.0, 3.0))
    offsets = result.cell_offsets(log_x=log_x, n_draws=n_draws).set_index("cell")

    x_levels = coding.levels[x_factor]
    line_levels = coding.levels[line_factor]
    fallback_colors = get_colors(line_levels, None, DEFAULT_PALETTE)
    dodge = 0.08 * (np.arange(len(line_levels)) - (len(line_levels) - 1) / 2)

    ax.axhline(0, color="0.5", linewidth=0.8, zorder=1)
    for k, level in enumerate(line_levels):
        anchor = coding.cell_label(
            {line_factor: level, x_factor: coding.reference[x_factor]}
        )
        color = color_of.get(anchor, fallback_colors[k])
        points = [
            (i + dodge[k], cell, offsets.loc[cell])
            for i, x_level in enumerate(x_levels)
            if (cell := coding.cell_label({line_factor: level, x_factor: x_level}))
            in offsets.index
        ]
        if not points:
            continue
        x = np.array([p[0] for p in points])
        rows = pd.DataFrame([p[2] for p in points])
        ax.plot(
            x,
            rows["expected_mean_percent"].to_numpy(dtype=float),
            linestyle="--",
            marker="o",
            ms=5,
            mfc="white",
            color=color,
            linewidth=1,
            zorder=2,
            gid=f"expected|{level}",
        )
        ax.plot(
            x,
            rows["offset_mean_percent"].to_numpy(dtype=float),
            color=color,
            linewidth=2,
            zorder=3,
            label=f"{line_factor}={level}",
            gid=f"line|{level}",
        )
        for xi, cell, row in points:
            extrapolated = not row["within_cell_range"]
            ax.errorbar(
                xi,
                row["offset_mean_percent"],
                yerr=_interval_errors(
                    row["offset_mean_percent"],
                    row["offset_lower_percent"],
                    row["offset_upper_percent"],
                ),
                fmt="D" if extrapolated else "o",
                ms=5,
                color=color,
                capsize=2,
                zorder=4,
                gid=f"offset|{cell}",
            )
            if annotate_interaction and not np.isnan(row["interaction_mean_percent"]):
                ax.annotate(
                    f"{row['interaction_mean_percent']:+.1f}% "
                    f"[{row['interaction_lower_percent']:+.1f}, "
                    f"{row['interaction_upper_percent']:+.1f}]",
                    xy=(xi, row["offset_mean_percent"]),
                    xytext=(9, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize="x-small",
                    color=color,
                    gid=f"interaction|{cell}",
                )

    size = np.exp(offsets["log_x"].iloc[0])
    if title is None:
        title = f"At {x_name} = {size:.3g}" + (" (x_ref)" if log_x is None else "")
    ax.set_title(title, fontsize="medium")
    ax.set_xticks(range(len(x_levels)), x_levels)
    ax.set_xlim(-0.5, len(x_levels) - 0.5)
    ax.set_xlabel(x_factor if x_label is None else x_label)
    ax.set_ylabel(
        f"Size-matched {y_name} offset (%, {INTERVAL_LABEL})"
        if y_label is None
        else y_label
    )
    _add_legend(ax, [ax], legend_placement)
    return _finish(fig, owns_figure)


# VARIANCE AND INDIVIDUALS


def plot_variance_components(
    result: ProportionModelResult,
    cells: list[CellSpec] | None = None,
    show_repeatability: bool = False,
    colors: dict | list | None = None,
    labels: dict | list | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ax: matplotlib.axes.Axes | np.ndarray | list | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "best",
) -> matplotlib.figure.Figure:
    """
    Plot each genotype's between-worm and within-worm SD relative to the reference cell.

    ``tau_ratio`` (between worms) and ``sigma_ratio`` (within worm) per
    non-reference cell come from ``summary`` and are drawn side by side on a log
    axis with a line at 1. With ``show_repeatability``, a second panel shows each
    cell's repeatability from ``repeatability``.

    Parameters:
        result (ProportionModelResult): Model fitted with ``group_variances=True``.
        cells (list or None): Cells to plot, as condition ids, cell labels or
            factor-level dicts; ``None`` plots all. (default: None)
        show_repeatability (bool): Add the repeatability panel. (default: False)
        colors (dict, list or None): Cell colors keyed by condition id or cell
            label, or one per cell in ``coding.cells`` order. (default: None)
        labels (dict, list or None): Cell display names, like ``colors``.
            (default: None)
        x_label (str or None): X-axis label. (default: None)
        y_label (str or None): Y-axis label of the ratio panel. (default: None)
        ax (Axes, np.ndarray, list or None): Axes to draw into, two with
            ``show_repeatability``; no figure is created or shown when given.
            (default: None)
        ax_size (tuple[float, float] or None): Size in inches of each panel of a
            new figure. (default: None)
        legend_placement (str or None): Placement passed to ``add_legend`` on the
            ratio panel; ``None`` hides the legend. (default: "best")

    Returns:
        matplotlib.figure.Figure: The figure holding the plot.

    Raises:
        ValueError: If the model was fitted without group variances.
    """
    if not result.group_variances:
        raise ValueError(
            "Variance components per genotype need a fit with group_variances=True."
        )
    color_of, label_of = _cell_styles(result, colors, labels)
    reference_label = label_of[result.coding.reference_cell]
    selected = _resolve_cells(result, cells, include_reference=False)
    fig, axes, owns_figure = _panel_axes(
        2 if show_repeatability else 1, ax, ax_size, default_size=DEFAULT_AX_SIZE
    )
    summary = result.summary()
    ratios = summary[summary["kind"].isin(RATIO_KINDS)].set_index(["kind", "cell"])

    ratio_ax = axes[0]
    for i, cell in enumerate(selected):
        for name, shift, marker in (("tau", -0.12, "o"), ("sigma", 0.12, "s")):
            row = ratios.loc[(f"{name}_ratio", cell)]
            ratio_ax.errorbar(
                i + shift,
                row["mean"],
                yerr=_interval_errors(row["mean"], row["lower"], row["upper"]),
                fmt=marker,
                ms=5,
                color=color_of[cell],
                capsize=2,
                gid=f"{name}_ratio|{cell}",
            )
    ratio_ax.axhline(1, color="0.5", linewidth=0.8, zorder=1)
    ratio_ax.set_yscale("log")
    _format_log_axis(ratio_ax.yaxis, plain=True)
    ratio_ax.set_xticks(
        range(len(selected)),
        [label_of[c] for c in selected],
        rotation=30,
        ha="right",
    )
    ratio_ax.set_xlim(-0.5, len(selected) - 0.5)
    if x_label is not None:
        ratio_ax.set_xlabel(x_label)
    ratio_ax.set_ylabel(
        f"SD ratio to {reference_label} ({INTERVAL_LABEL})"
        if y_label is None
        else y_label
    )
    _add_legend(ratio_ax, [ratio_ax], legend_placement)

    if show_repeatability:
        repeatability_ax = axes[1]
        repeatability = result.repeatability().set_index("cell")
        repeatability_cells = _resolve_cells(result, cells)
        for i, cell in enumerate(repeatability_cells):
            row = repeatability.loc[cell]
            repeatability_ax.errorbar(
                i,
                row["repeatability_mean"],
                yerr=_interval_errors(
                    row["repeatability_mean"],
                    row["repeatability_lower"],
                    row["repeatability_upper"],
                ),
                fmt="o",
                ms=5,
                color=color_of[cell],
                capsize=2,
                gid=f"repeatability|{cell}",
            )
        repeatability_ax.set_ylim(0, 1)
        repeatability_ax.set_xticks(
            range(len(repeatability_cells)),
            [label_of[c] for c in repeatability_cells],
            rotation=30,
            ha="right",
        )
        repeatability_ax.set_xlim(-0.5, len(repeatability_cells) - 0.5)
        if x_label is not None:
            repeatability_ax.set_xlabel(x_label)
        repeatability_ax.set_ylabel(f"Repeatability ({INTERVAL_LABEL})")
    return _finish(fig, owns_figure)


def plot_individual_consistency(
    result: ProportionModelResult,
    molt_a: TimepointSpec,
    molt_b: TimepointSpec,
    relative_to: Literal["own", "reference"] = "own",
    cells: list[CellSpec] | None = None,
    n_draws: int | None = None,
    ncols: int | None = None,
    colors: dict | list | None = None,
    labels: dict | list | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ax: matplotlib.axes.Axes | np.ndarray | list | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = None,
) -> matplotlib.figure.Figure:
    """
    Scatter each worm's deviation at one timepoint against another, one panel per genotype.

    Deviations come from ``worm_deviations(relative_to)`` and are shown in
    percent. Every panel has the identity line and the same, equal x and y
    limits, and is annotated with its number of worms and the cell's
    repeatability from ``repeatability`` (mean and interval). Worms missing
    either timepoint are dropped and counted.

    Parameters:
        result (ProportionModelResult): Fitted model.
        molt_a (int, float or str): Timepoint on the x axis: a molt number
            (molt ``k`` is timepoint ``k / 4``), a timepoint value or a timepoint
            label, as in ``ProportionModelResult.resolve_timepoint``.
        molt_b (int, float or str): Timepoint on the y axis, likewise.
        relative_to (str): ``"own"`` (residual around the genotype's curve) or
            ``"reference"`` (deviation from the reference curve).
            (default: "own")
        cells (list or None): Cells to plot, as condition ids, cell labels or
            factor-level dicts; ``None`` plots all. (default: None)
        n_draws (int or None): Posterior draws used by ``worm_deviations``;
            ``None`` uses all. (default: None)
        ncols (int or None): Panel columns of a new figure. (default: None)
        colors (dict, list or None): Cell colors keyed by condition id or cell
            label, or one per cell in ``coding.cells`` order. (default: None)
        labels (dict, list or None): Cell display names, like ``colors``.
            (default: None)
        x_label (str or None): X-axis label. (default: None)
        y_label (str or None): Y-axis label. (default: None)
        ax (Axes, np.ndarray, list or None): One axes per plotted cell; no figure
            is created or shown when given. (default: None)
        ax_size (tuple[float, float] or None): Size in inches of each panel of a
            new figure; keep it square for equal axes. (default: None)
        legend_placement (str or None): Placement of a cell legend passed to
            ``add_legend``; ``None`` relies on the panel titles. (default: None)

    Returns:
        matplotlib.figure.Figure: The figure holding the plot.

    Raises:
        ValueError: If ``molt_a`` or ``molt_b`` is not a timepoint of the table.
    """
    timepoint_a, timepoint_b = (result.resolve_timepoint(t) for t in (molt_a, molt_b))
    column_a, column_b = (
        result._deviation_column(t) for t in (timepoint_a, timepoint_b)
    )
    label_of_timepoint = result.timepoints().set_index("timepoint")["timepoint_label"]
    deviations = result.worm_deviations(relative_to, n_draws=n_draws)
    color_of, label_of = _cell_styles(result, colors, labels)
    selected = _resolve_cells(result, cells)
    fig, axes, owns_figure = _panel_axes(len(selected), ax, ax_size, ncols)
    repeatability = result.repeatability().set_index("cell")
    worm_cells = (
        deviations["condition_id"].map(result.coding.cell_of_condition).to_numpy()
    )

    lowest, highest = np.inf, -np.inf
    for panel, cell in zip(axes, selected):
        values = deviations.loc[worm_cells == cell, [column_a, column_b]]
        complete = values.dropna()
        a = log_ratio_to_percentage(complete[column_a].to_numpy())
        b = log_ratio_to_percentage(complete[column_b].to_numpy())
        panel.scatter(
            a,
            b,
            s=14,
            color=color_of[cell],
            alpha=0.7,
            linewidths=0,
            label=label_of[cell],
            gid=f"worms|{cell}",
        )
        if len(complete):
            lowest = min(lowest, a.min(), b.min())
            highest = max(highest, a.max(), b.max())
        row = repeatability.loc[cell if result.group_variances else "shared"]
        n_dropped = len(values) - len(complete)
        text = f"n = {len(complete)}"
        if n_dropped:
            text += f" ({n_dropped} missing a {_timepoint_noun(result)})"
        text += (
            f"\nR = {row['repeatability_mean']:.2f} "
            f"[{row['repeatability_lower']:.2f}, {row['repeatability_upper']:.2f}]"
        )
        panel.text(
            0.04,
            0.96,
            text,
            transform=panel.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            gid=f"annotation|{cell}",
        )
        panel.set_title(label_of[cell], fontsize="medium")
        panel.set_xlabel(
            f"Deviation at {label_of_timepoint[timepoint_a]} (%)"
            if x_label is None
            else x_label
        )
        panel.set_ylabel(
            f"Deviation at {label_of_timepoint[timepoint_b]} (%)"
            if y_label is None
            else y_label
        )

    if not np.isfinite(lowest):
        lowest, highest = -1.0, 1.0
    pad = 0.05 * (highest - lowest) or 1.0
    limits = (lowest - pad, highest + pad)
    for panel in axes:
        panel.plot(
            limits, limits, color="0.5", linestyle=":", linewidth=1, gid="identity"
        )
        panel.set_xlim(limits)
        panel.set_ylim(limits)

    _add_legend(fig if owns_figure else axes[0], axes, legend_placement)
    return _finish(fig, owns_figure)


def plot_penetrance(
    result: ProportionModelResult,
    prob: float = 0.95,
    cells: list[CellSpec] | None = None,
    split_direction: bool = True,
    penetrance: pd.DataFrame | None = None,
    n_draws: int | None = None,
    random_seed: int | None = None,
    colors: dict | list | None = None,
    labels: dict | list | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "outside right",
) -> matplotlib.figure.Figure:
    """
    Plot, per genotype, the fraction of worms outside the reference cell's predictive interval.

    Bars come from ``penetrance(prob)``; with ``split_direction`` the fractions
    below and above are stacked, each with its error bar. A dashed line marks
    ``1 - prob``, the fraction expected for the reference cell, as a calibration
    check. Cells with observations beyond the reference size range get a note of
    their ``extrapolated_fraction``.

    Parameters:
        result (ProportionModelResult): Fitted model.
        prob (float): Probability mass of the reference interval. (default: 0.95)
        cells (list or None): Cells to plot, as condition ids, cell labels or
            factor-level dicts; ``None`` plots all. (default: None)
        split_direction (bool): Stack the fractions below and above instead of
            one bar of the total. (default: True)
        penetrance (pd.DataFrame or None): Precomputed ``penetrance(prob)``
            table; ``None`` computes it. (default: None)
        n_draws (int or None): Posterior draws used when computing; ``None``
            uses all. (default: None)
        random_seed (int or None): Seed used when computing. (default: None)
        colors (dict, list or None): Cell colors keyed by condition id or cell
            label, or one per cell in ``coding.cells`` order. (default: None)
        labels (dict, list or None): Cell display names, like ``colors``.
            (default: None)
        x_label (str or None): X-axis label. (default: None)
        y_label (str or None): Y-axis label. (default: None)
        ax (matplotlib.axes.Axes or None): Axes to draw into; no figure is
            created or shown when given. (default: None)
        ax_size (tuple[float, float] or None): Axes size in inches of a new
            figure. (default: None)
        legend_placement (str or None): Placement passed to ``add_legend``;
            ``None`` hides the legend.
            (default: "outside right")

    Returns:
        matplotlib.figure.Figure: The figure holding the plot.
    """
    if penetrance is None:
        penetrance = result.penetrance(
            prob=prob, random_seed=random_seed, n_draws=n_draws
        )
    table = penetrance.set_index("cell")
    color_of, label_of = _cell_styles(result, colors, labels)
    selected = _resolve_cells(result, cells)
    fig, ax, owns_figure = _single_axes(ax, ax_size)

    def interval(x, center, lower, upper, gid):
        errors = _interval_errors(center, lower, upper)
        # a fraction pinned at 0 or 1 in every draw has no interval to draw
        if errors.any():
            ax.errorbar(
                x, center, yerr=errors, fmt="none", ecolor="black", capsize=2, gid=gid
            )

    for i, cell in enumerate(selected):
        row = table.loc[cell]
        color = color_of[cell]
        if split_direction:
            below, above = row["below_mean"], row["above_mean"]
            ax.bar(
                i,
                below,
                width=0.6,
                color=color,
                alpha=0.45,
                hatch="///",
                edgecolor=color,
                gid=f"below|{cell}",
            )
            ax.bar(
                i,
                above,
                bottom=below,
                width=0.6,
                color=color,
                edgecolor=color,
                gid=f"above|{cell}",
            )
            interval(
                i - 0.1,
                below,
                row["below_lower"],
                row["below_upper"],
                f"below_interval|{cell}",
            )
            interval(
                i + 0.1,
                below + above,
                below + row["above_lower"],
                below + row["above_upper"],
                f"above_interval|{cell}",
            )
            top = max(
                row["below_upper"],
                below + row["above_upper"],
            )
        else:
            ax.bar(
                i,
                row["outside_mean"],
                width=0.6,
                color=color,
                edgecolor=color,
                gid=f"outside|{cell}",
            )
            interval(
                i,
                row["outside_mean"],
                row["outside_lower"],
                row["outside_upper"],
                f"outside_interval|{cell}",
            )
            top = row["outside_upper"]
        if row["extrapolated_fraction"] > 0:
            ax.annotate(
                f"{row['extrapolated_fraction']:.0%}\nextrap.",
                xy=(i, top),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize="xx-small",
                linespacing=1.0,
                gid=f"extrapolated|{cell}",
            )

    ax.axhline(
        1 - prob,
        color="0.3",
        linestyle="--",
        linewidth=1,
        gid="expected_outside",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xticks(
        range(len(selected)), [label_of[c] for c in selected], rotation=30, ha="right"
    )
    ax.set_xlim(-0.5, len(selected) - 0.5)
    if x_label is not None:
        ax.set_xlabel(x_label)
    ax.set_ylabel(
        f"Worms outside reference {prob:.0%} interval ({INTERVAL_LABEL})"
        if y_label is None
        else y_label
    )
    _add_legend(ax, [ax], legend_placement)
    return _finish(fig, owns_figure)


# SUMMARIES


def plot_effects_forest(
    result_or_summary: ProportionModelResult | pd.DataFrame,
    parameters: list[str] | str | None = None,
    display: Literal["percent", "log"] = "percent",
    labels: dict[str, str] | None = None,
    color: Any = "black",
    x_label: str | None = None,
    y_label: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = None,
) -> matplotlib.figure.Figure:
    """
    Plot ``summary`` rows as a forest plot of posterior means with their intervals.

    With ``display="percent"``, ``beta`` rows use their ``_percent`` columns and
    all other rows their raw values, which the default axis label states. The
    vertical reference line is at 1 when every row is a ratio and at 0
    otherwise. Rows without an interval are skipped.

    Parameters:
        result_or_summary (ProportionModelResult or pd.DataFrame): A fit, whose
            ``summary`` is used, or a table with the columns of ``summary``.
        parameters (list[str], str or None): Parameter names, kept in the given
            order, or a regular expression; ``None`` selects the ``beta`` rows.
            (default: None)
        display (str): ``"percent"`` or ``"log"``. (default: "percent")
        labels (dict[str, str] or None): Display names keyed by parameter.
            (default: None)
        color (Any): Color of the points and bars. (default: "black")
        x_label (str or None): X-axis label. (default: None)
        y_label (str or None): Y-axis label. (default: None)
        ax (matplotlib.axes.Axes or None): Axes to draw into; no figure is
            created or shown when given. (default: None)
        ax_size (tuple[float, float] or None): Axes size in inches of a new
            figure; the height defaults to 0.3 inch per row. (default: None)
        legend_placement (str or None): Placement passed to ``add_legend``;
            ``None`` hides the legend. (default: None)

    Returns:
        matplotlib.figure.Figure: The figure holding the plot.

    Raises:
        ValueError: If ``display`` is unknown or no row matches ``parameters``.
    """
    _check_display(display)
    summary = (
        result_or_summary.summary()
        if isinstance(result_or_summary, ProportionModelResult)
        else result_or_summary
    )
    rows = _select_parameters(summary, parameters).dropna(
        subset=["mean", "lower", "upper"]
    )
    if rows.empty:
        raise ValueError(f"No row matching {parameters!r} has an interval.")
    values = _display_values(rows, display)
    default_label, null_value = _effect_axis(rows["kind"], display)
    fig, ax, owns_figure = _single_axes(
        ax, ax_size, default_size=(3.5, max(1.2, 0.3 * len(rows)))
    )

    y = np.arange(len(rows))[::-1]
    ax.axvline(null_value, color="0.5", linewidth=0.8, zorder=1)
    ax.errorbar(
        values["mean"],
        y,
        xerr=_interval_errors(values["mean"], values["lower"], values["upper"]),
        fmt="o",
        ms=5,
        color=color,
        capsize=2,
        gid="effects",
    )
    labels = {} if labels is None else labels
    ax.set_yticks(y, [labels.get(p, p) for p in rows["parameter"]])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(default_label if x_label is None else x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    _add_legend(ax, [ax], legend_placement)
    return _finish(fig, owns_figure)


def plot_linearity_check(
    result: ProportionModelResult,
    threshold: float | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "best",
) -> matplotlib.figure.Figure:
    """
    Plot the reference cell's mean residual per timepoint for the line and spline reference fits.

    Points and intervals come from ``result.linear_misfit`` and, when present,
    ``result.spline_misfit``, dodged and labelled ``"line"`` and ``"spline"``.
    The shaded band is the misfit tolerated by ``reference_shape="auto"``, i.e.
    an absolute log ratio up to ``log(1 + threshold)``.

    Parameters:
        result (ProportionModelResult): Fitted model with at least one misfit
            table.
        threshold (float or None): Tolerated misfit as a fraction; ``None`` uses
            ``result.linearity_threshold``, else 0.03. (default: None)
        title (str or None): Axes title; ``None`` uses the default, ``""``
            hides it. (default: None)
        x_label (str or None): X-axis label. (default: None)
        y_label (str or None): Y-axis label. (default: None)
        ax (matplotlib.axes.Axes or None): Axes to draw into; no figure is
            created or shown when given. (default: None)
        ax_size (tuple[float, float] or None): Axes size in inches of a new
            figure. (default: None)
        legend_placement (str or None): Placement passed to ``add_legend``;
            ``None`` hides the legend. (default: "best")

    Returns:
        matplotlib.figure.Figure: The figure holding the plot.

    Raises:
        ValueError: If the result has neither misfit table.
    """
    tables = [
        (name, _add_timepoint_columns(table))
        for name, table in (
            ("line", result.linear_misfit),
            ("spline", result.spline_misfit),
        )
        if table is not None
    ]
    if not tables:
        raise ValueError(
            "The result has no linear_misfit or spline_misfit table; they are set "
            "by fit_proportion_model."
        )
    if threshold is None:
        threshold = (
            0.03 if result.linearity_threshold is None else result.linearity_threshold
        )
    fig, ax, owns_figure = _single_axes(ax, ax_size)
    timepoints = (
        pd.concat([table[["timepoint", "timepoint_label"]] for _, table in tables])
        .drop_duplicates("timepoint")
        .sort_values("timepoint")
    )
    position = {t: i for i, t in enumerate(timepoints["timepoint"])}

    ax.axhspan(
        log_ratio_to_percentage(-np.log1p(threshold)),
        100 * threshold,
        color="0.9",
        zorder=0,
        gid="threshold",
    )
    ax.axhline(0, color="0.5", linewidth=0.8, zorder=1)
    for k, (name, table) in enumerate(tables):
        marker, color = MISFIT_STYLES[name]
        shift = 0.15 * (k - (len(tables) - 1) / 2)
        center = table["mean_percent"].to_numpy()
        ax.errorbar(
            [position[t] + shift for t in table["timepoint"]],
            center,
            yerr=_interval_errors(
                center, table["lower_percent"], table["upper_percent"]
            ),
            fmt=marker,
            ms=5,
            color=color,
            capsize=2,
            label=name,
            gid=f"misfit|{name}",
        )
    ax.set_xticks(range(len(timepoints)), timepoints["timepoint_label"].tolist())
    ax.set_xlim(-0.5, len(timepoints) - 0.5)
    if title is None:
        title = f"Reference shape used: {result.reference_shape_used}"
    ax.set_title(title, fontsize="medium")
    ax.set_xlabel(_timepoint_axis_label(result) if x_label is None else x_label)
    ax.set_ylabel(
        f"Reference residual (%, {INTERVAL_LABEL})" if y_label is None else y_label
    )
    _add_legend(ax, [ax], legend_placement)
    return _finish(fig, owns_figure)


def _draw_distribution(
    ax: matplotlib.axes.Axes,
    values: np.ndarray,
    kind: str,
    grid: np.ndarray | None,
    **line_kwargs,
) -> None:
    """Draw the ECDF of ``values``, or their Gaussian KDE on ``grid``."""
    if kind == "ecdf":
        ordered = np.sort(values)
        ax.plot(
            ordered,
            np.arange(1, len(ordered) + 1) / len(ordered),
            drawstyle="steps-post",
            **line_kwargs,
        )
    elif len(np.unique(values)) > 1:
        ax.plot(grid, gaussian_kde(values)(grid), **line_kwargs)


def plot_posterior_predictive_check(
    result: ProportionModelResult,
    kind: Literal["ecdf", "density"] = "ecdf",
    n_draws: int | None = 100,
    cells: list[CellSpec] | None = None,
    residuals: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    ppc: pd.DataFrame | None = None,
    random_seed: int | None = None,
    ncols: int | None = None,
    colors: dict | list | None = None,
    labels: dict | list | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ax: matplotlib.axes.Axes | np.ndarray | list | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "outside right",
) -> matplotlib.figure.Figure:
    """
    Compare each genotype's residual distribution with posterior predictive replicates.

    Each panel shows thin grey ECDFs (or densities) of the replicated marginal
    residuals from ``posterior_predictive_residuals``, one per draw, and on top
    the observed posterior mean residuals from ``residuals`` in the cell color,
    all in percent. The posterior predictive p-values of the SD and the excess
    kurtosis from ``ppc_summary`` are written in each panel.

    Parameters:
        result (ProportionModelResult): Fitted model.
        kind (str): ``"ecdf"`` or ``"density"``. (default: "ecdf")
        n_draws (int or None): Posterior draws simulated, and at most this many
            replicates drawn; ``None`` uses all. (default: 100)
        cells (list or None): Cells to plot, as condition ids, cell labels or
            factor-level dicts; ``None`` plots all. (default: None)
        residuals (tuple or None): Precomputed ``(replicated, observed)`` from
            ``posterior_predictive_residuals``; ``None`` computes them.
            (default: None)
        ppc (pd.DataFrame or None): Precomputed ``ppc_summary`` table; ``None``
            computes it from ``residuals``. (default: None)
        random_seed (int or None): Seed used when simulating. (default: None)
        ncols (int or None): Panel columns of a new figure. (default: None)
        colors (dict, list or None): Cell colors keyed by condition id or cell
            label, or one per cell in ``coding.cells`` order. (default: None)
        labels (dict, list or None): Cell display names, like ``colors``.
            (default: None)
        x_label (str or None): X-axis label. (default: None)
        y_label (str or None): Y-axis label. (default: None)
        ax (Axes, np.ndarray, list or None): One axes per plotted cell; no figure
            is created or shown when given. (default: None)
        ax_size (tuple[float, float] or None): Size in inches of each panel of a
            new figure. (default: None)
        legend_placement (str or None): Placement passed to ``add_legend``;
            ``None`` hides the legend. (default: "outside right")

    Returns:
        matplotlib.figure.Figure: The figure holding the plot.

    Raises:
        ValueError: If ``kind`` is not ``"ecdf"`` or ``"density"``.
    """
    if kind not in ("ecdf", "density"):
        raise ValueError(f"`kind` must be 'ecdf' or 'density', got {kind!r}.")
    if residuals is None:
        residuals = result.posterior_predictive_residuals(
            n_draws=n_draws, random_seed=random_seed
        )
    replicated, observed = residuals
    if ppc is None:
        ppc = result.ppc_summary(replicated, observed)
    shown_draws = replicated["draw"].unique()[:n_draws]
    replicated = replicated[replicated["draw"].isin(shown_draws)]
    posterior_mean = result.residuals()
    observed_cells = (
        posterior_mean["condition_id"].map(result.coding.cell_of_condition).to_numpy()
    )

    color_of, label_of = _cell_styles(result, colors, labels)
    selected = _resolve_cells(result, cells)
    fig, axes, owns_figure = _panel_axes(len(selected), ax, ax_size, ncols)
    p_values = ppc.set_index(["cell", "statistic"])["p_value"]

    for panel, cell in zip(axes, selected):
        replicates = [
            log_ratio_to_percentage(group["residual"].to_numpy())
            for _, group in replicated[replicated["cell"] == cell].groupby(
                "draw", sort=False
            )
        ]
        observed_values = log_ratio_to_percentage(
            posterior_mean["residual"].to_numpy()[observed_cells == cell]
        )
        grid = None
        if kind == "density":
            pooled = np.concatenate(replicates + [observed_values])
            grid = np.linspace(pooled.min(), pooled.max(), 200)
        for values in replicates:
            _draw_distribution(
                panel,
                values,
                kind,
                grid,
                color=REPLICATE_COLOR,
                linewidth=0.5,
                alpha=0.4,
                gid=f"replicated|{cell}",
            )
        _draw_distribution(
            panel,
            observed_values,
            kind,
            grid,
            color=color_of[cell],
            linewidth=2,
            zorder=3,
            gid=f"observed|{cell}",
        )
        # ECDFs leave the lower right empty, densities the upper corners
        panel.text(
            0.96,
            0.04 if kind == "ecdf" else 0.96,
            f"p(SD) = {p_values[(cell, 'sd')]:.2f}\n"
            f"p(excess kurtosis) = {p_values[(cell, 'excess_kurtosis')]:.2f}",
            transform=panel.transAxes,
            ha="right",
            va="bottom" if kind == "ecdf" else "top",
            fontsize="small",
            gid=f"p_values|{cell}",
        )
        panel.set_title(label_of[cell], fontsize="medium")
        panel.set_xlabel(
            "Residual around own curve (%)" if x_label is None else x_label
        )
        panel.set_ylabel(
            ("ECDF" if kind == "ecdf" else "Density") if y_label is None else y_label
        )

    _add_legend(fig if owns_figure else axes[0], axes, legend_placement)
    return _finish(fig, owns_figure)


def plot_experiment_consistency(
    results: dict[str, ProportionModelResult] | pd.DataFrame,
    parameters: list[str] | str | None = None,
    pooled: ProportionModelResult | None = None,
    display: Literal["percent", "log"] = "percent",
    labels: dict[str, str] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "outside right",
) -> matplotlib.figure.Figure:
    """
    Plot each parameter's estimate per experiment, to check that effects replicate.

    Rows come from ``compare_experiments`` (or a table it returned). Experiments
    are dodged within each parameter row and told apart by marker. With
    ``pooled``, its ``summary`` estimate and interval are drawn as a shaded band
    behind each row. ``beta`` rows are in percent with ``display="percent"``.

    Parameters:
        results (dict or pd.DataFrame): Fits keyed by experiment, e.g. from
            ``fit_per_experiment``, or a ``compare_experiments`` table.
        parameters (list[str], str or None): Parameter names, kept in the given
            order, or a regular expression; ``None`` selects the ``beta`` rows.
            (default: None)
        pooled (ProportionModelResult or None): Fit to all experiments together.
            (default: None)
        display (str): ``"percent"`` or ``"log"``. (default: "percent")
        labels (dict[str, str] or None): Display names keyed by parameter.
            (default: None)
        x_label (str or None): X-axis label. (default: None)
        y_label (str or None): Y-axis label. (default: None)
        ax (matplotlib.axes.Axes or None): Axes to draw into; no figure is
            created or shown when given. (default: None)
        ax_size (tuple[float, float] or None): Axes size in inches of a new
            figure; the height defaults to 0.45 inch per parameter.
            (default: None)
        legend_placement (str or None): Placement passed to ``add_legend``;
            ``None`` hides the legend.
            (default: "outside right")

    Returns:
        matplotlib.figure.Figure: The figure holding the plot.

    Raises:
        ValueError: If ``display`` is unknown or no row matches ``parameters``.
    """
    _check_display(display)
    table = compare_experiments(results) if isinstance(results, dict) else results
    rows = _select_parameters(table, parameters)
    names = list(dict.fromkeys(rows["parameter"]))
    experiments = list(dict.fromkeys(rows["experiment"]))
    row_of = {name: len(names) - 1 - i for i, name in enumerate(names)}
    spread = 0.5 / max(len(experiments), 1)
    default_label, null_value = _effect_axis(
        rows.drop_duplicates("parameter")["kind"], display
    )
    fig, ax, owns_figure = _single_axes(
        ax, ax_size, default_size=(3.5, max(1.2, 0.45 * len(names)))
    )

    ax.axvline(null_value, color="0.5", linewidth=0.8, zorder=1)
    if pooled is not None:
        summary = pooled.summary()
        pooled_rows = summary[summary["parameter"].isin(names)]
        pooled_values = _display_values(pooled_rows, display)
        for k, name in enumerate(pooled_rows["parameter"]):
            y = row_of[name]
            ax.fill_betweenx(
                [y - 0.4, y + 0.4],
                pooled_values["lower"][k],
                pooled_values["upper"][k],
                color="0.85",
                linewidth=0,
                zorder=0,
                gid=f"pooled|{name}",
            )
            ax.plot(
                [pooled_values["mean"][k]] * 2,
                [y - 0.4, y + 0.4],
                color="0.55",
                linewidth=1,
                zorder=1,
                gid=f"pooled_mean|{name}",
            )

    for k, experiment in enumerate(experiments):
        sub = rows[rows["experiment"] == experiment]
        values = _display_values(sub, display)
        offset = spread * ((len(experiments) - 1) / 2 - k)
        ax.errorbar(
            values["mean"],
            [row_of[name] + offset for name in sub["parameter"]],
            xerr=_interval_errors(values["mean"], values["lower"], values["upper"]),
            fmt=EXPERIMENT_MARKERS[k % len(EXPERIMENT_MARKERS)],
            ms=5,
            color="black",
            capsize=2,
            label=str(experiment),
            gid=f"experiment|{experiment}",
        )
    labels = {} if labels is None else labels
    ax.set_yticks(
        [row_of[name] for name in names], [labels.get(name, name) for name in names]
    )
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel(default_label if x_label is None else x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    _add_legend(ax, [ax], legend_placement)
    return _finish(fig, owns_figure)


def plot_timepoint_dispersion(
    result: ProportionModelResult,
    comparisons: list[tuple[CellSpec, CellSpec]] | None = None,
    family: Literal["all", "comparison"] = "all",
    show_swarm: bool = True,
    tests: pd.DataFrame | None = None,
    n_draws: int | None = None,
    ncols: int | None = None,
    colors: dict | list | None = None,
    labels: dict | list | None = None,
    titles: dict | list | None = None,
    y_label: str | None = None,
    ax: matplotlib.axes.Axes | np.ndarray | list | None = None,
    ax_size: tuple[float, float] | None = None,
    legend_placement: str | None = "outside right",
) -> matplotlib.figure.Figure:
    """
    Plot, timepoint by timepoint, each genotype's residuals around its own curve with dispersion tests.

    Each timepoint is a panel of violins of the posterior mean marginal
    residuals from ``residuals``, in percent, titled by its ``timepoint_label``
    unless ``titles`` says otherwise.
    Brackets show the Holm-adjusted p-values (``p_holm``) of the Brown–Forsythe
    tests from ``timepoint_dispersion_tests``; tests that were not run get no
    bracket. ``plot_molt_dispersion`` is an alias.

    Parameters:
        result (ProportionModelResult): Fitted model.
        comparisons (list[tuple] or None): Pairs of cells to test, as in
            ``timepoint_dispersion_tests``; ``None`` compares each non-reference
            cell with the reference. Ignored when ``tests`` is given.
            (default: None)
        family (str): ``"all"`` or ``"comparison"``, as in
            ``timepoint_dispersion_tests``. (default: "all")
        show_swarm (bool): Overlay the residuals as a swarm. (default: True)
        tests (pd.DataFrame or None): Precomputed ``timepoint_dispersion_tests``
            table; ``None`` computes it. (default: None)
        n_draws (int or None): Posterior draws used when computing the tests;
            ``None`` uses all. (default: None)
        ncols (int or None): Panel columns of a new figure. (default: None)
        colors (dict, list or None): Cell colors keyed by condition id or cell
            label, or one per cell in ``coding.cells`` order. (default: None)
        labels (dict, list or None): Cell display names, like ``colors``.
            (default: None)
        titles (dict, list or None): Panel titles, either keyed by timepoint
            (anything ``ProportionModelResult.resolve_timepoint`` accepts, e.g.
            ``1``, ``"M1"`` or ``0.375``; timepoints left out keep their label)
            or one per plotted timepoint in development order. (default: None)
        y_label (str or None): Y-axis label of the first panel. (default: None)
        ax (Axes, np.ndarray, list or None): One axes per timepoint; no figure
            is created or shown when given. (default: None)
        ax_size (tuple[float, float] or None): Size in inches of each panel of a
            new figure. (default: None)
        legend_placement (str or None): Placement passed to ``add_legend``;
            ``None`` hides the legend. (default: "outside right")

    Returns:
        matplotlib.figure.Figure: The figure holding the plot.

    Raises:
        ValueError: If ``titles`` is a list whose length differs from the
            number of timepoints, or a dict with a key that is not a timepoint
            of the table.
    """
    if tests is None:
        tests = result.timepoint_dispersion_tests(comparisons, family, n_draws)
    tests = _add_timepoint_columns(tests)
    coding = result.coding
    involved = set(tests["cell_a"]) | set(tests["cell_b"])
    plot_cells = [cell for cell in coding.cells if cell in involved]
    color_of, label_of = _cell_styles(result, colors, labels)

    residuals = result.residuals()
    residuals["cell"] = residuals["condition_id"].map(coding.cell_of_condition)
    residuals["timepoint"] = result.table.loc[residuals.index, "timepoint"]
    residuals = residuals[residuals["cell"].isin(plot_cells)]
    label_of_timepoint = result.timepoints().set_index("timepoint")["timepoint_label"]
    timepoints = sorted(residuals["timepoint"].unique())
    order = {timepoint: i for i, timepoint in enumerate(timepoints)}
    title_of = label_of_timepoint.to_dict()
    if isinstance(titles, dict):
        title_of.update({result.resolve_timepoint(k): v for k, v in titles.items()})
    elif titles is not None:
        if len(titles) != len(timepoints):
            raise ValueError(
                f"Got {len(titles)} titles for {len(timepoints)} timepoints "
                f"{[label_of_timepoint[t] for t in timepoints]}."
            )
        title_of.update(zip(timepoints, titles))
    data = pd.DataFrame(
        {
            "Order": residuals["timepoint"].map(order).to_numpy(),
            "Condition": residuals["cell"].to_numpy(),
            "residual": residuals["residual"].to_numpy(),
        }
    )
    annotations = {}
    for timepoint, group in tests.groupby("timepoint"):
        tested = group[group["p_holm"].notna()]
        if timepoint in order:
            annotations[order[timepoint]] = (
                list(zip(tested["cell_a"], tested["cell_b"])),
                [_format_p(p) for p in tested["p_holm"]],
            )

    fig, axes, owns_figure = _panel_axes(len(timepoints), ax, ax_size, ncols)
    _plot_violinplot(
        data,
        plot_cells,
        "residual",
        [color_of[cell] for cell in plot_cells],
        axes if len(axes) > 1 else axes[0],
        titles=[title_of[timepoint] for timepoint in timepoints],
        share_y_axis=False,
        plot_significance=True,
        significance_pairs=None,
        log_scale=False,
        show_swarm=show_swarm,
        display_transform=log_ratio_to_percentage,
        custom_annotations=annotations,
    )
    axes[0].set_ylabel("Residual around own curve (%)" if y_label is None else y_label)

    extra = [(Patch(facecolor=color_of[cell]), label_of[cell]) for cell in plot_cells]
    for panel in axes:
        legend = panel.get_legend()
        if legend is not None:
            legend.remove()
    add_legend(
        fig if owns_figure else axes[0],
        legend_placement,
        [handle for handle, _ in extra],
        [label for _, label in extra],
    )
    return _finish(fig, owns_figure)


plot_molt_dispersion = plot_timepoint_dispersion
