import os

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.legend import Legend
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
    by seaborn) is removed first, so repeated calls never stack legends.

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
            ``placement``.

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

    return target.legend(handles, labels, **{**placement_kwargs, **legend_kwargs})


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

    kwargs = {"figsize": (fig_w, fig_h)}
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
