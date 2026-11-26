import os
from collections.abc import Callable

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.container import ErrorbarContainer
from matplotlib.font_manager import FontProperties
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Annotation
from mpl_toolkits.axes_grid1 import Divider, Size

# THIS PART IS MOSTLY ABOUT HANDLING LEGENDS, SAVING FIGURES, ETC.

# (loc, bbox_to_anchor) for legends placed just outside an axes, in axes coordinates
LEGEND_AXES_OUTSIDE_ANCHORS = {
    "outside right": ("center left", (1.02, 0.5)),
    "outside top": ("lower center", (0.5, 1.02)),
    "outside bottom": ("upper center", (0.5, -0.15)),
}

# (loc, bbox_to_anchor) for legends placed just outside a figure, in figure coordinates
LEGEND_FIGURE_OUTSIDE_ANCHORS = {
    "outside right": ("center left", (1.001, 0.5)),
    "outside top": ("lower center", (0.5, 1.0)),
    "outside bottom": ("upper center", (0.5, 0.0)),
}

AVAILABLE_LEGEND_PLACEMENTS = list(Legend.codes) + list(LEGEND_AXES_OUTSIDE_ANCHORS)

# long enough for a few periods of a dashed or dash-dotted pattern to show
LEGEND_HANDLE_LENGTH = 3.0

# looks of the entries drawn by plot_legend, matching curves, boxes/violins and swarms
AVAILABLE_LEGEND_KINDS = ["line", "patch", "marker"]

# symbols written in a condition table for a factor a condition has or lacks
CONDITION_TABLE_PRESENT = "+"
CONDITION_TABLE_ABSENT = "\u2212"


def save_figure(
    fig: matplotlib.figure.Figure,
    name: str,
    directory: str,
    format: str = "svg",
    dpi: int = 300,
    transparent: bool = True,
) -> None:
    """
    Save a given matplotlib figure to the specified directory with the given name, in the chose format.

    Parameters:
        fig (matplotlib.figure.Figure) : Figure to save
        name (str) : Name of the file (without extension)
        directory (str) : Directory to save the file in
        format (str) : File format to save the figure in
        dpi (int) : Resolution of the saved figure
        transparent (bool) : Whether to save the figure with a transparent background

    Returns:
        None
    """

    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Construct full file path
    filename = f"{name}.{format}"
    filepath = os.path.join(directory, filename)

    # Save the figure
    fig.savefig(
        filepath,
        format=format,
        dpi=dpi,
        bbox_inches="tight",
        transparent=transparent,
    )


def build_legend(single_condition_dict: dict, legend: dict | None) -> str:
    """
    Build a legend label string for a single condition.

    Parameters:
        single_condition_dict (dict) : Condition dict containing metadata fields.
        legend (dict or None) : Mapping of condition dict keys to unit/suffix strings.
            Each key whose value is truthy is appended as ``"<value> <unit>"``;
            a falsy value appends only ``"<value>"``.  If ``None``, the label
            defaults to ``"Condition <condition_id>"``.

    Returns:
        str : Legend label for the condition.
    """
    if legend is None:
        return f'Condition {int(single_condition_dict["condition_id"])}'
    else:
        legend_string = ""
        for i, (key, value) in enumerate(legend.items()):
            if value:
                legend_string += f"{single_condition_dict[key]} {value}"
            else:
                legend_string += f"{single_condition_dict[key]}"
            if i < len(legend) - 1:
                legend_string += ", "
        return legend_string


def add_legend(
    target: matplotlib.axes.Axes | matplotlib.figure.Figure | None = None,
    placement: str | None = "best",
    handles: list | None = None,
    labels: list[str] | None = None,
    deduplicate: bool = False,
    **legend_kwargs,
) -> Legend | None:
    """
    Draw a single legend on an axes or a figure at a named placement.

    Any legend already drawn on the target (or, for a figure, on any of its axes, e.g.
    by seaborn) is removed first, so repeated calls never stack legends. Line handles
    are drawn long and without markers or error bar caps, so their line style stays
    readable.

    Parameters:
        target (matplotlib.axes.Axes or matplotlib.figure.Figure or None): Where to draw
            the legend. ``None`` uses the current pyplot axes. (default: ``None``)
        placement (str or None): Either a matplotlib ``loc`` string (``"best"``,
            ``"upper left"``, ...) placing the legend inside the target, or one of
            ``"outside right"``, ``"outside top"``, ``"outside bottom"`` anchoring it
            just outside. ``"best"`` falls back to ``"upper right"`` for figures.
            ``None`` removes the legend. (default: ``"best"``)
        handles (list or None): Legend artists. If ``None``, collected from the target's
            axes. Must be given together with ``labels``. (default: ``None``)
        labels (list[str] or None): Legend labels matching ``handles``. (default: ``None``)
        deduplicate (bool): If ``True``, keep only the first entry for each label when
            handles are collected automatically. (default: ``False``)
        **legend_kwargs: Extra keyword arguments forwarded to ``legend()``. They take
            precedence over the ``loc``, ``bbox_to_anchor`` and ``ncols`` derived from
            ``placement`` and over the default ``handlelength``.

    Returns:
        matplotlib.legend.Legend or None: The created legend, or ``None`` when
            ``placement`` is ``None``.

    Raises:
        ValueError: If ``placement`` is not in ``AVAILABLE_LEGEND_PLACEMENTS``, or if
            only one of ``handles`` and ``labels`` is given.
    """
    if placement is not None and placement not in AVAILABLE_LEGEND_PLACEMENTS:
        raise ValueError(
            f"Invalid legend placement: {placement}. Must be None or one of {AVAILABLE_LEGEND_PLACEMENTS}."
        )
    if (handles is None) != (labels is None):
        raise ValueError("handles and labels must be given together.")

    if target is None:
        target = plt.gca()
    is_figure = isinstance(target, matplotlib.figure.Figure)
    axes = target.axes if is_figure else [target]

    existing_legends = [ax.get_legend() for ax in axes]
    if is_figure:
        existing_legends.extend(target.legends)
    for existing_legend in existing_legends:
        if existing_legend is not None:
            existing_legend.remove()

    if placement is None:
        return None

    if handles is None:
        handles, labels = [], []
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if deduplicate and label in labels:
                    continue
                handles.append(handle)
                labels.append(label)

    handles = [_line_legend_proxy(handle) for handle in handles]

    outside_anchors = (
        LEGEND_FIGURE_OUTSIDE_ANCHORS if is_figure else LEGEND_AXES_OUTSIDE_ANCHORS
    )
    if placement in outside_anchors:
        loc, bbox_to_anchor = outside_anchors[placement]
        placement_kwargs = {"loc": loc, "bbox_to_anchor": bbox_to_anchor}
        if (
            placement in ("outside top", "outside bottom")
            and "ncol" not in legend_kwargs
        ):
            # lay entries out in a single row above/below the target
            placement_kwargs["ncols"] = max(len(labels), 1)
    elif is_figure and placement == "best":
        placement_kwargs = {"loc": "upper right"}
    else:
        placement_kwargs = {"loc": placement}
    placement_kwargs["handlelength"] = LEGEND_HANDLE_LENGTH

    return target.legend(handles, labels, **{**placement_kwargs, **legend_kwargs})


def _condition_legend_handle(
    kind: str,
    color,
    linestyle: str,
    hatch: str | None,
    marker: str,
    linewidth: float | None,
    markersize: float | None,
):
    """Build the legend artist of one condition for ``plot_legend``."""
    if kind == "line":
        return Line2D([], [], color=color, linestyle=linestyle, linewidth=linewidth)
    if kind == "patch":
        return Patch(facecolor=color, hatch=hatch, linewidth=linewidth)
    return Line2D(
        [], [], color=color, marker=marker, markersize=markersize, linestyle="None"
    )


def _row_major_order(n_entries: int, ncols: int) -> list[int]:
    """
    Order entries so a column-filled legend reads them row by row.

    Matplotlib fills legend columns top to bottom, giving the first
    ``n_entries % ncols`` columns one extra entry, which is the shape of a grid
    filled row by row with a shorter last row. Listing the entries column by column
    of that grid thus makes them read left to right, then top to bottom.
    """
    return [
        entry for column in range(ncols) for entry in range(column, n_entries, ncols)
    ]


def plot_legend(
    conditions_struct: list,
    conditions_to_plot: list,
    legend: dict | None = None,
    colors: list | dict | None = None,
    linestyles: list | dict | None = None,
    hatches: list | dict | None = None,
    kind: str = "line",
    marker: str = "o",
    ncols: int = 1,
    nrows: int | None = None,
    fill: str = "columns",
    title: str | None = None,
    fontsize: float | str | None = None,
    title_fontsize: float | str | None = None,
    handlelength: float = LEGEND_HANDLE_LENGTH,
    handleheight: float | None = None,
    handletextpad: float | None = None,
    labelspacing: float | None = None,
    columnspacing: float | None = None,
    borderpad: float | None = None,
    frameon: bool | None = None,
    linewidth: float | None = None,
    markersize: float | None = None,
    figsize: tuple[float, float] | None = None,
    margin: float = 0.0,
    ax: matplotlib.axes.Axes | None = None,
    **legend_kwargs,
) -> matplotlib.figure.Figure:
    """
    Render only the legend of a set of conditions, as a figure of its own.

    Entries use the same labels, colors, line styles and hatches as the plots given
    the same arguments, so a set of plots drawn with ``legend_placement=None`` can
    share one separate legend. Unless ``figsize`` is given, a new figure is sized to
    the legend, so it saves without margins.

    Spacings and handle sizes are in multiples of the font size, as in
    ``matplotlib.axes.Axes.legend``; ``None`` uses the matplotlib rcParams.

    Parameters:
        conditions_struct (list): List of condition dicts.
        conditions_to_plot (list): Ordered condition identifiers, one entry each.
        legend (dict or None): Legend spec passed to ``build_legend``.
            (default: ``None``)
        colors (list or dict or None): Color spec passed to ``get_colors``.
            (default: ``None``)
        linestyles (list or dict or None): Line style spec passed to
            ``get_linestyles``; only used when ``kind`` is ``"line"``.
            (default: ``None``)
        hatches (list or dict or None): Hatch spec passed to ``get_hatches``; only
            used when ``kind`` is ``"patch"``. (default: ``None``)
        kind (str): Look of the entries, matching the plot they go with:
            ``"line"`` for curves, ``"patch"`` for box and violin plots,
            ``"marker"`` for swarm and scatter plots. (default: ``"line"``)
        marker (str): Matplotlib marker of ``"marker"`` entries. (default: ``"o"``)
        ncols (int): Number of columns. Ignored when ``nrows`` is given.
            (default: 1)
        nrows (int or None): Number of rows; the number of columns is then the
            fewest that fit every entry, e.g. ``nrows=1`` lays the legend out in a
            single row. (default: ``None``)
        fill (str): ``"columns"`` lists the entries top to bottom, then left to
            right; ``"rows"`` lists them left to right, then top to bottom.
            (default: ``"columns"``)
        title (str or None): Title written above the entries. (default: ``None``)
        fontsize (float or str or None): Font size of the labels, in points or as
            a name such as ``"small"``; it also scales every spacing below.
            (default: ``None``)
        title_fontsize (float or str or None): Font size of the title.
            (default: ``None``)
        handlelength (float): Length of the handles. (default: ``LEGEND_HANDLE_LENGTH``)
        handleheight (float or None): Height of the handles; makes ``"patch"``
            entries taller. (default: ``None``)
        handletextpad (float or None): Gap between a handle and its label.
            (default: ``None``)
        labelspacing (float or None): Vertical gap between rows. (default: ``None``)
        columnspacing (float or None): Horizontal gap between columns.
            (default: ``None``)
        borderpad (float or None): Gap between the frame and the entries.
            (default: ``None``)
        frameon (bool or None): Whether to draw the frame. (default: ``None``)
        linewidth (float or None): Line width of ``"line"`` handles and edge width
            of ``"patch"`` handles, in points. (default: ``None``)
        markersize (float or None): Marker size of ``"marker"`` handles, in points.
            (default: ``None``)
        figsize (tuple[float, float] or None): Size of a new figure in inches, with
            the legend centred in it. ``None`` fits the figure to the legend.
            (default: ``None``)
        margin (float): Blank space added around the legend on each side, in inches,
            when the figure is fitted to it. (default: 0.0)
        ax (matplotlib.axes.Axes or None): Axes to draw the legend in, centred, with
            the axes itself hidden; ``figsize`` and ``margin`` are then ignored.
            ``None`` creates a new figure. (default: ``None``)
        **legend_kwargs: Any other keyword argument of
            ``matplotlib.axes.Axes.legend``, e.g. ``alignment`` or ``markerfirst``.

    Returns:
        matplotlib.figure.Figure: The figure holding the legend.

    Raises:
        ValueError: If ``kind`` is not in ``AVAILABLE_LEGEND_KINDS``, ``fill`` is
            neither ``"columns"`` nor ``"rows"``, or ``ncols`` or ``nrows`` is
            smaller than 1.
    """
    if kind not in AVAILABLE_LEGEND_KINDS:
        raise ValueError(
            f"Invalid legend kind: {kind}. Must be one of {AVAILABLE_LEGEND_KINDS}."
        )
    if fill not in ("columns", "rows"):
        raise ValueError(f"Invalid fill: {fill}. Must be 'columns' or 'rows'.")
    if ncols < 1 or (nrows is not None and nrows < 1):
        raise ValueError("ncols and nrows must be at least 1.")

    color_palette = get_colors(conditions_to_plot, colors)
    line_styles = get_linestyles(conditions_to_plot, linestyles)
    hatch_patterns = get_hatches(conditions_to_plot, hatches)
    handles = [
        _condition_legend_handle(
            kind, color, linestyle, hatch, marker, linewidth, markersize
        )
        for color, linestyle, hatch in zip(color_palette, line_styles, hatch_patterns)
    ]
    labels = [
        build_legend(conditions_struct[condition_id], legend)
        for condition_id in conditions_to_plot
    ]

    n_entries = len(labels)
    if nrows is not None:
        ncols = -(-n_entries // nrows)
    ncols = max(min(ncols, n_entries), 1)
    if fill == "rows":
        order = _row_major_order(n_entries, ncols)
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]

    style_kwargs = {
        "title": title,
        "fontsize": fontsize,
        "title_fontsize": title_fontsize,
        "handleheight": handleheight,
        "handletextpad": handletextpad,
        "labelspacing": labelspacing,
        "columnspacing": columnspacing,
        "borderpad": borderpad,
        "frameon": frameon,
    }
    style_kwargs = {
        key: value for key, value in style_kwargs.items() if value is not None
    }
    legend_kwargs = {
        "ncols": ncols,
        "handlelength": handlelength,
        **style_kwargs,
        **legend_kwargs,
    }

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(layout="none")
    else:
        fig = ax.figure
    ax.set_axis_off()
    drawn_legend = add_legend(ax, "center", handles, labels, **legend_kwargs)

    if owns_figure:
        if figsize is None:
            legend_extent = drawn_legend.get_window_extent()
            figsize = (
                legend_extent.width / fig.dpi + 2 * margin,
                legend_extent.height / fig.dpi + 2 * margin,
            )
        fig.set_size_inches(figsize)
        ax.set_position((0, 0, 1, 1))
        plt.show()
    return fig


def _condition_table_entry(entry: bool | str | None) -> str:
    """Convert one condition table entry to the text written in the table."""
    if entry is None:
        return ""
    if isinstance(entry, bool | np.bool_):
        return CONDITION_TABLE_PRESENT if entry else CONDITION_TABLE_ABSENT
    return str(entry)


def build_condition_table(
    conditions_struct: list,
    conditions_to_plot: list,
    condition_table: dict[str, list | dict | Callable[[dict], bool | str | None]],
) -> dict[str, list[str]]:
    """
    Build the text of a condition table, one row per factor and one column per condition.

    A condition table replaces a legend of long condition descriptions by a grid of
    ``+`` and ``−`` under the plot, telling for each factor (genotype, treatment, ...)
    whether a condition has it.

    Parameters:
        conditions_struct (list): List of condition dicts, indexed by condition id.
        conditions_to_plot (list): Ordered condition identifiers, one table column each.
        condition_table (dict): Maps each row label (e.g. ``"raga-1:AID"``) to its
            entries, given as a list with one entry per condition in
            ``conditions_to_plot``, a dict mapping condition identifiers to entries
            (missing conditions lack the factor), or a callable taking a condition
            dict and returning its entry, e.g.
            ``lambda c: "raga-1:AID" in c["description"]``. An entry is ``True``
            (written ``+``), ``False`` (written ``−``), a string written as is
            (e.g. an allele name), or ``None`` (left blank).

    Returns:
        dict[str, list[str]]: Row label to the text of each column, in the order of
            ``conditions_to_plot``.

    Raises:
        ValueError: If a list of entries has a different length than
            ``conditions_to_plot``, or a row is neither a list, a dict nor a callable.
    """
    table = {}
    for row_label, entries in condition_table.items():
        if callable(entries):
            entries = [
                entries(conditions_struct[condition_id])
                for condition_id in conditions_to_plot
            ]
        elif isinstance(entries, dict):
            entries = [
                entries.get(condition_id, False) for condition_id in conditions_to_plot
            ]
        elif isinstance(entries, list | tuple):
            if len(entries) != len(conditions_to_plot):
                raise ValueError(
                    f"Length of condition table row {row_label!r} ({len(entries)}) does not match number of conditions to plot ({len(conditions_to_plot)})."
                )
        else:
            raise ValueError(
                f"Condition table row {row_label!r} must be a list, a dict or a callable, got {type(entries).__name__}."
            )
        table[row_label] = [_condition_table_entry(entry) for entry in entries]
    return table


def add_condition_table(
    ax: matplotlib.axes.Axes,
    table: dict[str, list[str]],
    show_row_labels: bool = True,
    row_spacing: float = 1.5,
    fontsize: float | str | None = None,
    show_ticks: bool = True,
) -> list[Annotation]:
    """
    Write a condition table under an axes, one row per factor and one column per x tick.

    Column ``i`` is centered on x = ``i`` in data coordinates, which is where seaborn
    draws the ``i``-th box, violin or swarm. Rows are spaced in points, so the table
    keeps its look whatever the axes size; it is drawn outside the axes and is kept in
    the figure by ``bbox_inches="tight"`` or a constrained layout. A tick marks each
    column on the bottom spine, and the table starts below it, where tick labels
    would be.

    Parameters:
        ax (matplotlib.axes.Axes): Axes to write the table under.
        table (dict[str, list[str]]): Row label to the text of each column, e.g. from
            ``build_condition_table``.
        show_row_labels (bool): If ``True``, write the row labels left of the axes; turn
            off for every panel but the first of a row of panels. (default: ``True``)
        row_spacing (float): Distance between rows, in multiples of the font size.
            (default: 1.5)
        fontsize (float or str or None): Font size of the table. ``None`` uses the x
            tick label size. (default: ``None``)
        show_ticks (bool): If ``True``, draw an unlabelled x tick at each column.
            (default: ``True``)

    Returns:
        list[matplotlib.text.Annotation]: The texts of the table, row by row, with each
            row label (when shown) before its entries.
    """
    if fontsize is None:
        fontsize = plt.rcParams["xtick.labelsize"]
    fontsize_points = FontProperties(size=fontsize).get_size_in_points()
    row_height = row_spacing * fontsize_points
    label_padding = 0.5 * fontsize_points
    n_columns = max((len(entries) for entries in table.values()), default=0)

    top_offset = 0.5 * row_height
    if show_ticks:
        ax.set_xticks(range(n_columns))
        ax.tick_params(axis="x", which="major", bottom=True, labelbottom=False)
        # leave the tick and its label pad clear, as a tick label would
        top_offset = max(
            top_offset,
            plt.rcParams["xtick.major.size"] + plt.rcParams["xtick.major.pad"],
        )

    texts = []
    for row_index, (row_label, entries) in enumerate(table.items()):
        y_offset = -(top_offset + row_index * row_height)
        if show_row_labels:
            texts.append(
                ax.annotate(
                    row_label,
                    xy=(0, 0),
                    xycoords="axes fraction",
                    xytext=(-label_padding, y_offset),
                    textcoords="offset points",
                    ha="right",
                    va="top",
                    fontsize=fontsize,
                    annotation_clip=False,
                )
            )
        for column_index, entry in enumerate(entries):
            texts.append(
                ax.annotate(
                    entry,
                    xy=(column_index, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=fontsize,
                    annotation_clip=False,
                )
            )
    return texts


def _line_legend_proxy(handle):
    """
    Replace a line-like legend handle by a plain line so its line style is legible.

    Markers drawn in the middle of a legend line and the caps of error bars hide
    dashes and dots, so a handle that draws a line is replaced by an unmarked line
    with the same color, width, style and alpha. Handles that draw no line (markers
    only, patches) are returned unchanged.

    Parameters:
        handle: A legend artist, e.g. a ``Line2D``, an ``ErrorbarContainer`` or a patch.

    Returns:
        The proxy ``Line2D``, or ``handle`` itself when it draws no line.
    """
    line = handle.lines[0] if isinstance(handle, ErrorbarContainer) else handle
    if not isinstance(line, Line2D) or line.get_linestyle() in ("None", "", " "):
        return handle
    return Line2D(
        [],
        [],
        color=line.get_color(),
        linewidth=line.get_linewidth(),
        linestyle=line.get_linestyle(),
        alpha=line.get_alpha(),
    )


def set_scale(ax: matplotlib.axes.Axes, log_scale: bool | tuple | list) -> None:
    """
    Set the x- and/or y-axis scale of a matplotlib Axes object.

    Parameters:
        ax (matplotlib.axes.Axes) : Axes to configure.
        log_scale (bool or tuple or list) : Scale specification.
            - ``bool``: applies to the y-axis only (``True`` → log, ``False`` → linear).
            - ``tuple`` or ``list`` of two bools ``(x_log, y_log)``: sets both axes independently.

    Returns:
        None
    """
    if isinstance(log_scale, bool):
        ax.set_yscale("log" if log_scale else "linear")
    elif isinstance(log_scale, tuple):
        ax.set_yscale("log" if log_scale[1] else "linear")
        ax.set_xscale("log" if log_scale[0] else "linear")
    elif isinstance(log_scale, list):
        ax.set_yscale("log" if log_scale[1] else "linear")
        ax.set_xscale("log" if log_scale[0] else "linear")


def get_colors(
    conditions_to_plot: list,
    colors: list | dict | None,
    base_palette: str = "colorblind",
) -> list:
    """
    Return a list of colors, one per condition, validating user-supplied values.

    Parameters:
        conditions_to_plot (list) : Ordered list of condition identifiers.
        colors (list or dict or None) : Color specification.
            - ``None``: a seaborn ``base_palette`` palette is generated automatically.
            - ``list``: must have the same length as ``conditions_to_plot``.
            - ``dict``: must contain a key for every entry in ``conditions_to_plot``;
              values are returned in the same order.
        base_palette (str) : Seaborn palette name used when ``colors`` is ``None``.
            Defaults to ``"colorblind"``.

    Returns:
        list : Colors in the same order as ``conditions_to_plot``.

    Raises:
        AssertionError : If a list ``colors`` has a different length than ``conditions_to_plot``,
            or if a dict ``colors`` is missing entries for any condition.
    """
    if colors is None:
        colors = sns.color_palette(base_palette, len(conditions_to_plot))
    else:
        if isinstance(colors, list):
            assert len(colors) == len(
                conditions_to_plot
            ), f"Length of colors list ({len(colors)}) does not match number of conditions to plot ({len(conditions_to_plot)})"
            colors = colors
        elif isinstance(colors, dict):
            assert np.all(
                [key in colors.keys() for key in conditions_to_plot]
            ), "Some conditions to plot are not in the colors dictionary"
            colors = [colors[condition] for condition in conditions_to_plot]

    return colors


def _get_condition_styles(
    conditions_to_plot: list,
    styles: list | dict | None,
    default: str | None,
    style_name: str,
) -> list:
    """
    Return one style value per condition from a list, dict or ``None`` specification.

    Parameters:
        conditions_to_plot (list): Ordered list of condition identifiers.
        styles (list or dict or None): ``None`` gives every condition ``default``; a
            list must have one entry per condition; a dict maps condition identifiers
            to styles, and conditions missing from it get ``default``.
        default (str or None): Style used for conditions without an explicit style.
        style_name (str): Name of the argument, used in error messages.

    Returns:
        list: Styles in the same order as ``conditions_to_plot``.

    Raises:
        ValueError: If a list ``styles`` has a different length than
            ``conditions_to_plot``, or ``styles`` is neither a list, a dict nor ``None``.
    """
    if styles is None:
        return [default] * len(conditions_to_plot)
    if isinstance(styles, dict):
        return [styles.get(condition, default) for condition in conditions_to_plot]
    if isinstance(styles, list | tuple):
        if len(styles) != len(conditions_to_plot):
            raise ValueError(
                f"Length of {style_name} ({len(styles)}) does not match number of conditions to plot ({len(conditions_to_plot)})."
            )
        return list(styles)
    raise ValueError(
        f"{style_name} must be a list, a dict or None, got {type(styles).__name__}."
    )


def get_linestyles(
    conditions_to_plot: list,
    linestyles: list | dict | None,
    default: str = "-",
) -> list:
    """
    Return one matplotlib line style per condition.

    Lets conditions that share a color be told apart, e.g. ``{5: "--"}`` draws
    condition 5 dashed and every other condition solid.

    Parameters:
        conditions_to_plot (list): Ordered list of condition identifiers.
        linestyles (list or dict or None): Line style specification, using any
            matplotlib ``linestyle`` (``"-"``, ``"--"``, ``":"``, ``"-."``, or a
            ``(offset, (on, off, ...))`` dash tuple). ``None`` gives every condition
            ``default``; a list must have one entry per condition; a dict maps
            condition identifiers to styles, and conditions missing from it get
            ``default``.
        default (str): Line style of conditions without an explicit style.
            (default: ``"-"``)

    Returns:
        list: Line styles in the same order as ``conditions_to_plot``.

    Raises:
        ValueError: If a list ``linestyles`` has a different length than
            ``conditions_to_plot``.
    """
    return _get_condition_styles(conditions_to_plot, linestyles, default, "linestyles")


def get_hatches(
    conditions_to_plot: list,
    hatches: list | dict | None,
) -> list:
    """
    Return one matplotlib hatch pattern per condition, ``None`` meaning no hatch.

    Lets conditions that share a color be told apart in box and violin plots, e.g.
    ``{5: "//"}`` hatches condition 5 and leaves every other condition plain.

    Parameters:
        conditions_to_plot (list): Ordered list of condition identifiers.
        hatches (list or dict or None): Hatch specification, using any matplotlib
            ``hatch`` pattern (``"/"``, ``"\\\\"``, ``"x"``, ``"."``, ...; repeating a
            symbol makes it denser, e.g. ``"///"``). ``None`` hatches nothing; a
            list must have one entry per condition; a dict maps condition
            identifiers to patterns, and conditions missing from it are not hatched.

    Returns:
        list: Hatch patterns (or ``None``) in the same order as ``conditions_to_plot``.

    Raises:
        ValueError: If a list ``hatches`` has a different length than
            ``conditions_to_plot``.
    """
    return _get_condition_styles(conditions_to_plot, hatches, None, "hatches")


def create_fixed_ax_sized_fig(
    ax_w: float = 3.5,
    ax_h: float = 3.0,
    left: float = 1.0,
    right: float = 0.8,
    bottom: float = 0.6,
    top: float = 0.3,
    nrows: int = 1,
    ncols: int = 1,
    hspace: float = 0.4,
    wspace: float = 0.2,
    dpi: float | None = None,
    return_divider: bool = False,
) -> tuple:
    """
    Create a figure whose axes panels each have a guaranteed physical size in inches.

    All measurements are in inches.

    Parameters:
        ax_w (float): Width of each axes area, excluding any labels or ticks. (default: 3.5)
        ax_h (float): Height of each axes area, excluding any labels or ticks. (default: 3.0)
        left (float): Left margin of the figure, in inches.  Should accommodate
            y-axis tick labels and title of the leftmost column. (default: 1.0)
        right (float): Right margin of the figure, in inches. (default: 0.8)
        bottom (float): Bottom margin of the figure, in inches.  Should accommodate
            x-axis tick labels and title of the bottom row. (default: 0.6)
        top (float): Top margin of the figure, in inches. (default: 0.3)
        nrows (int): Number of panel rows in the grid. (default: 1)
        ncols (int): Number of panel columns in the grid. (default: 1)
        hspace (float): Vertical gap between rows, in inches. (default: 0.4)
        wspace (float): Horizontal gap between columns, in inches. (default: 0.4)
        dpi (float or None): Resolution of the figure in dots per inch.  ``None``
            uses matplotlib's default (typically 100 for screen). (default: ``None``)
        return_divider (bool): If ``True``, include the ``Divider`` object as the last
            element of the returned tuple — useful for placing additional axes (e.g.
            colorbars) in the same coordinate system. (default: ``False``)

    Returns:
        tuple: ``(fig, ax)`` for a single panel (``nrows=ncols=1``);
            ``(fig, axes)`` for a grid, where ``axes`` is a 2-D ``np.ndarray`` of
            shape ``(nrows, ncols)`` — or a 1-D array when one dimension is 1.
            When ``return_divider=True``, the ``Divider`` is appended as the final
            element: ``(fig, ax, div)`` or ``(fig, axes, div)``.
    """
    fig_w = left + ncols * ax_w + (ncols - 1) * wspace + right
    fig_h = bottom + nrows * ax_h + (nrows - 1) * hspace + top

    # The Divider fixes every axes position, so a layout engine enabled through
    # rcParams would only fight it (and warn that it finds no gridspec).
    kwargs = {"figsize": (fig_w, fig_h), "layout": "none"}
    if dpi is not None:
        kwargs["dpi"] = dpi
    fig = plt.figure(**kwargs)

    h = [Size.Fixed(left)]
    for c in range(ncols):
        h.append(Size.Fixed(ax_w))
        if c < ncols - 1:
            h.append(Size.Fixed(wspace))
    h.append(Size.Fixed(right))

    v = [Size.Fixed(bottom)]
    for r in range(nrows):
        v.append(Size.Fixed(ax_h))
        if r < nrows - 1:
            v.append(Size.Fixed(hspace))
    v.append(Size.Fixed(top))

    div = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)

    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            nx = 1 + c * 2
            # v is indexed bottom-up, so row 0 (top) maps to the highest ny index
            ny = 1 + (nrows - 1 - r) * 2
            axes[r, c] = fig.add_axes(
                div.get_position(), axes_locator=div.new_locator(nx=nx, ny=ny)
            )

    # Squeeze: single panel → bare Axes; one row/col → 1-D array
    if nrows == 1 and ncols == 1:
        out_axes = axes[0, 0]
    elif nrows == 1:
        out_axes = axes[0, :]
    elif ncols == 1:
        out_axes = axes[:, 0]
    else:
        out_axes = axes

    if return_divider:
        return fig, out_axes, div
    return fig, out_axes
