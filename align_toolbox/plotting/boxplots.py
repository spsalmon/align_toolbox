from itertools import combinations

import bottleneck as bn
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import STATTEST_LIBRARY, StatTest

from .utils_data_processing import rescale_without_flattening
from .utils_plotting import (
    add_legend,
    build_legend,
    create_fixed_ax_sized_fig,
    get_colors,
)

STATANNOTATIONS_TESTS = STATTEST_LIBRARY.keys()
CUSTOM_TESTS = ["Feltz-Miller", "MSLR"]


def _setup_figure(
    df: pd.DataFrame,
    figsize: tuple[float, float] | None,
    titles: list[str] | None,
    ax_size: tuple[float, float] | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes | np.ndarray]:
    """
    Create a figure and axes grid sized to the number of unique ordering groups.

    Parameters:
        df (pandas.DataFrame) : Data DataFrame containing an ``"Order"`` column
            whose unique values determine the number of subplots.
        figsize (tuple[float, float] or None) : Explicit figure size.
            Ignored when ``ax_size`` is provided. Defaults to ``(6 * n_groups, 10)``
            when both are ``None``.
        titles (list[str] or None) : Subplot titles; set to ``None`` internally
            if the length does not match the number of groups.
        ax_size (tuple[float, float] or None) : If provided, each panel's axes area
            is fixed to ``(ax_w, ax_h)`` inches via ``create_fixed_ax_sized_fig``
            instead of using ``figsize``. Defaults to ``None``.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes or np.ndarray] :
            The created figure and axes (scalar or array depending on group count).
    """
    n_groups = df["Order"].nunique()
    if titles is not None and len(titles) != n_groups:
        print("Number of titles does not match the number of ecdysis events.")
        titles = None

    if ax_size is not None:
        fig, ax = create_fixed_ax_sized_fig(
            ncols=n_groups, ax_w=ax_size[0], ax_h=ax_size[1]
        )
    else:
        if figsize is None:
            figsize = (6 * n_groups, 10)
        fig, ax = plt.subplots(
            1,
            n_groups,
            figsize=(figsize[0] + 3, figsize[1]),
            sharey=False,
            layout="constrained",
        )

    return fig, ax


def feltz_miller_asymptotic_cv_test(
    sample1: np.ndarray, sample2: np.ndarray
) -> tuple[float, float]:
    """
    Perform the Feltz-Miller asymptotic test for equality of CV on two samples.

    Adapted from: https://github.com/benmarwick/cvequality/blob/master/R/functions.R

    Parameters:
        sample1 (array-like) : First sample values.
        sample2 (array-like) : Second sample values.

    Returns:
        tuple[float, float] : Test statistic ``D_AD`` and two-sided p-value.
    """
    k = 2
    n_j = [len(sample1), len(sample2)]
    s_j = [bn.nanstd(sample1), bn.nanstd(sample2)]
    x_j = [bn.nanmean(sample1), bn.nanmean(sample2)]

    n_j, s_j, x_j = np.array(n_j), np.array(s_j), np.array(x_j)

    m_j = n_j - 1

    D = (np.sum(m_j * (s_j / x_j))) / np.sum(m_j)

    # test statistic
    D_AD = (np.sum(m_j * (s_j / x_j - D) ** 2)) / (D**2 * (0.5 + D**2))

    # D_AD distributes as a Chi-squared distribution with k-1 degrees of freedom
    p_value = 1 - stats.chi2.cdf(D_AD, k - 1)
    return D_AD, p_value


def _LRT_STAT(n: np.ndarray, x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Compute the likelihood-ratio test statistic required by ``mslr_test``.

    Adapted from: https://github.com/benmarwick/cvequality/blob/master/R/functions.R

    Parameters:
        n (array-like) : Sample sizes for each group.
        x (array-like) : Sample means for each group.
        s (array-like) : Sample standard deviations for each group.

    Returns:
        np.ndarray : Concatenated array ``[uh_0, ..., uh_{k-1}, tauh, stat]`` where
            ``uh`` are the MLE group means, ``tauh`` is the MLE CV, and ``stat`` is
            the log-likelihood-ratio statistic.
    """
    n = np.asarray(n)
    x = np.asarray(x)
    s = np.asarray(s)

    k = len(x)
    df = n - 1
    ssq = s**2
    vsq = df * ssq / n
    v = np.sqrt(vsq)
    sn = np.sum(n)

    # MLES
    tau0 = np.sum(n * vsq / x**2) / sn
    iteration = 1
    while True:
        uh = (-x + np.sqrt(x**2 + 4.0 * tau0 * (vsq + x**2))) / (2.0 * tau0)
        tau = np.sum(n * (vsq + (x - uh) ** 2) / uh**2) / sn
        if abs(tau - tau0) <= 1.0e-7 or iteration > 30:
            break
        iteration += 1
        tau0 = tau

    tauh = np.sqrt(tau)

    elf = 0.0
    clf = 0.0
    for j in range(k):
        clf = (
            clf
            - n[j] * np.log(tauh * uh[j])
            - (n[j] * (vsq[j] + (x[j] - uh[j]) ** 2)) / (2.0 * tauh**2 * uh[j] ** 2)
        )
        elf = elf - n[j] * np.log(v[j]) - n[j] / 2.0

    stat = 2.0 * (elf - clf)
    return np.concatenate([uh, [tauh, stat]])


def mslr_test(
    sample1: np.ndarray, sample2: np.ndarray, nr: int = 1000
) -> tuple[float, float]:
    """
    Perform the Modified Signed-Likelihood Ratio Test (MSLR) for equality of CVs.

    Adapted from: https://github.com/benmarwick/cvequality/blob/master/R/functions.R

    Parameters:
        sample1 (array-like) : First sample values.
        sample2 (array-like) : Second sample values.
        nr (int) : Number of parametric bootstrap replicates used to calibrate the
            test statistic.  Defaults to ``1000``.

    Returns:
        tuple[float, float] : Modified test statistic ``statm`` and two-sided p-value.
    """
    k = 2

    n = np.array([len(sample1), len(sample2)])
    x = np.array([bn.nanmean(sample1), bn.nanmean(sample2)])
    s = np.array([bn.nanstd(sample1), bn.nanstd(sample2)])

    gv = np.zeros(nr)
    df = n - 1
    xst0 = _LRT_STAT(n, x, s)
    uh0 = xst0[:k]
    tauh0 = xst0[k]
    stat0 = xst0[k + 1]
    sh0 = tauh0 * uh0
    se0 = tauh0 * uh0 / np.sqrt(n)

    # PB estimates of the mean and SD of the LRT
    for ii in range(nr):
        z = np.random.normal(size=k)
        x_sim = uh0 + z * se0
        ch = np.random.chisquare(df)
        s_sim = sh0 * np.sqrt(ch / df)
        xst = _LRT_STAT(n, x_sim, s_sim)
        gv[ii] = xst[k + 1]

    am = np.mean(gv)
    sd = np.std(gv, ddof=1)
    # end PB estimates

    statm = np.sqrt(2.0 * (k - 1)) * (stat0 - am) / sd + (k - 1)
    pval = 1.0 - stats.chi2.cdf(statm, k - 1)

    return statm, pval


def _annotate_significance(
    df: pd.DataFrame,
    conditions_to_plot: list,
    column: str,
    boxplot: matplotlib.axes.Axes,
    significance_pairs: list[tuple] | None,
    event_index: int,
    plot_type: str = "boxplot",
    test: str = "Mann-Whitney",
    verbose: bool = True,
) -> None:
    """
    Add significance annotations to a single subplot using statannotations.

    Parameters:
        df (pandas.DataFrame) : Full data DataFrame with ``"Order"`` and
            ``"Condition"`` columns.
        conditions_to_plot (list) : Ordered condition identifiers.
        column (str) : Column name of the y-variable.
        boxplot (matplotlib.axes.Axes) : Axes object of the target subplot.
        significance_pairs (list[tuple] or None) : Explicit pairs to annotate;
            all pairwise combinations are used when ``None``.
        event_index (int) : The ``"Order"`` value identifying the current subplot.
        plot_type (str) : ``"boxplot"`` or ``"violinplot"``.  Defaults to ``"boxplot"``.
        test (str) : Statistical test name.  Statannotations built-in tests are
            supported as well as ``"Feltz-Miller"`` and ``"MSLR"``.
            Defaults to ``"Mann-Whitney"``.
        verbose (bool) : If ``True``, print sample sizes and test details.
            Defaults to ``True``.

    Returns:
        None
    """
    # Filter data for the current event
    df_filtered = df[df["Order"] == event_index]

    # Print non-NaN counts for each condition
    print(f"\nSample sizes (non-NaN) for event index {event_index}, column '{column}':")
    if verbose:
        for condition in conditions_to_plot:
            condition_data = df_filtered[df_filtered["Condition"] == condition][column]
            n = condition_data.notna().sum()
            print(f"Condition {condition}: n={n}")

    # Original code continues...
    if significance_pairs is None:
        pairs = list(combinations(df["Condition"].unique(), 2))
    else:
        pairs = significance_pairs
    annotator = Annotator(
        ax=boxplot,
        pairs=pairs,
        data=df_filtered,
        x="Condition",
        order=conditions_to_plot,
        y=column,
        plot=plot_type,
    )
    if test in STATANNOTATIONS_TESTS:
        if test != "Mann-Whitney":
            annotator.configure(
                test=test,
                text_format="simple",
                loc="inside",
                verbose=verbose,
                test_short_name=test.capitalize(),
            )
        else:
            annotator.configure(
                test=test, text_format="star", loc="inside", verbose=verbose
            )
    else:
        if test == "Feltz-Miller":
            custom_long_name = "Feltz-Miller Asymptotic Test"
            custom_short_name = "Feltz-Miller"
            custom_func = feltz_miller_asymptotic_cv_test
            stat_test = StatTest(custom_func, custom_long_name, custom_short_name)
            annotator.configure(
                test=stat_test,
                text_format="simple",
                loc="inside",
                verbose=verbose,
            )
        elif test == "MSLR":
            custom_long_name = "Modified Signed Likelihood Ratio Test"
            custom_short_name = "MSLR"
            custom_func = mslr_test
            stat_test = StatTest(custom_func, custom_long_name, custom_short_name)
            annotator.configure(
                test=stat_test,
                text_format="simple",
                loc="inside",
                verbose=verbose,
            )
        else:
            raise ValueError(
                f"Test {test} is not supported. Please use one of the following: "
                f"{list(STATANNOTATIONS_TESTS) + CUSTOM_TESTS}"
            )
    annotator.apply_and_annotate()


def _add_metric_text(
    df: pd.DataFrame,
    conditions_to_plot: list,
    column: str,
    ax: matplotlib.axes.Axes,
    event_index: int,
    log_scale: bool,
    test: str = "Mann-Whitney",
    y_offset_pct: float = 0.1,
    significant_digits: int = 3,
) -> None:
    """
    Annotate each condition with its relevant summary statistic below the plot area.

    The statistic displayed depends on the test: median (Mann-Whitney, Kruskal-Wallis,
    Wilcoxon), mean (t-test, Welch), std (Levene), or CV % (Feltz-Miller, MSLR).

    Parameters:
        df (pandas.DataFrame) : Full data DataFrame with ``"Order"`` and
            ``"Condition"`` columns.
        conditions_to_plot (list) : Ordered condition identifiers.
        column (str) : Column name of the y-variable.
        ax (matplotlib.axes.Axes) : Axes object of the target subplot.
        event_index (int) : The ``"Order"`` value identifying the current subplot.
        log_scale (bool) : If ``True``, adjust y-position calculation for log-scale axes.
        test (str) : Statistical test name; determines which statistic to display.
            Defaults to ``"Mann-Whitney"``.
        y_offset_pct (float) : Downward offset of the text box as a fraction of the
            y-axis range.  Defaults to ``0.1``.
        significant_digits (int) : Number of significant digits in the displayed value.
            Defaults to ``3``.

    Returns:
        None

    Raises:
        ValueError : If ``test`` is not in the supported list.
    """
    test_metrics = {
        "Mann-Whitney": ("median", "M"),
        "Levene": ("std", "σ"),
        "t-test": ("mean", "μ"),
        "Kruskal-Wallis": ("median", "M"),
        "Welch": ("mean", "μ"),
        "Wilcoxon": ("median", "M"),
        "Feltz-Miller": ("cv", "CV"),
        "MSLR": ("cv", "CV"),
    }

    if test not in test_metrics:
        raise ValueError(
            f"Test '{test}' not supported. Available tests: {list(test_metrics.keys())}"
        )

    metric_type, symbol = test_metrics[test]

    data = df[df["Order"] == event_index]

    y_min, y_max = ax.get_ylim()

    if log_scale:
        log_y_min = np.log10(y_min) if y_min > 0 else np.log10(y_max) - 1
        log_y_max = np.log10(y_max)
        log_range = log_y_max - log_y_min
        y_position = 10 ** (log_y_min - log_range * y_offset_pct)
    else:
        y_range = y_max - y_min
        y_position = y_min - (y_range * y_offset_pct)

    for i, condition in enumerate(conditions_to_plot):
        condition_data = data[data["Condition"] == condition][column]

        if len(condition_data) == 0 or condition_data.isna().all():
            continue

        if metric_type == "mean":
            metric_value = condition_data.mean()
        elif metric_type == "median":
            metric_value = condition_data.median()
        elif metric_type == "std":
            metric_value = condition_data.std()
        elif metric_type == "cv":
            metric_value = condition_data.std() / condition_data.mean() * 100
        if np.isnan(metric_value):
            continue

        text = f"{symbol} = {metric_value:.{significant_digits}g}"
        if metric_type == "cv":
            text += " %"

        ax.text(
            i,
            y_position,
            text,
            ha="center",
            va="top",
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                linestyle="-.",
                alpha=0.8,
            ),
        )

    if log_scale:
        ax.set_ylim(10 ** (log_y_min - log_range * (y_offset_pct + 0.04)), y_max)
    else:
        ax.set_ylim(y_position - (y_range * 0.04), y_max)


def _plot_violinplot(
    df: pd.DataFrame,
    conditions_to_plot: list,
    column: str,
    color_palette: list,
    ax: matplotlib.axes.Axes | np.ndarray,
    titles: list[str] | None,
    share_y_axis: bool,
    plot_significance: bool,
    significance_pairs: list[tuple] | None,
    log_scale: bool,
    show_metric: bool = False,
    test: str = "Mann-Whitney",
    show_swarm: bool = True,
    hide_outliers: bool = False,
    inner: str | None = "box",
) -> tuple[list[float], list[float]]:
    """
    Draw violin + swarm subplots for each ordering group.

    Parameters:
        df (pandas.DataFrame) : Data with ``"Order"``, ``"Condition"``, and
            ``column`` columns.
        conditions_to_plot (list) : Ordered condition identifiers.
        column (str) : Y-variable column name.
        color_palette (list) : Colors in the same order as ``conditions_to_plot``.
        ax (np.ndarray or matplotlib.axes.Axes) : Axes array (or scalar) produced
            by ``_setup_figure``.
        titles (list[str] or None) : Subplot titles.
        share_y_axis (bool) : If ``True``, hide y-axis ticks on all but the first subplot.
        plot_significance (bool) : If ``True``, add significance brackets.
        significance_pairs (list[tuple] or None) : Pairs to annotate; all pairs when ``None``.
        log_scale (bool) : Passed to ``_add_metric_text`` for back-transformation.
        show_metric (bool) : If ``True``, display summary statistics below the plot.
            Defaults to ``False``.
        test (str) : Statistical test for significance annotation.
            Defaults to ``"Mann-Whitney"``.
        show_swarm (bool) : If ``True``, overlay a swarm plot on the violin plot.
            Defaults to ``True``.
        hide_outliers (bool) : If ``True``, remove data points beyond ±3 std in the
            swarm plot (violin retains them).  Defaults to ``False``.
        inner (str or None) : Passed to seaborn violinplot ``inner`` parameter. Defaults to ``None``.

    Returns:
        tuple[list[float], list[float]] : Per-subplot y-axis minima and maxima.
    """
    y_min, y_max = [], []
    for event_index in range(df["Order"].nunique()):
        if share_y_axis:
            if event_index > 0:
                ax[event_index].tick_params(
                    axis="y", which="both", left=False, labelleft=False
                )

        if isinstance(ax, np.ndarray):
            current_ax = ax[event_index]
        else:
            current_ax = ax

        violinplot = sns.violinplot(
            data=df[df["Order"] == event_index],
            x="Condition",
            y=column,
            order=conditions_to_plot,
            hue_order=conditions_to_plot,
            hue="Condition",
            palette=color_palette,
            cut=0,
            ax=current_ax,
            linewidth=2,
            legend="full",
            inner=inner,
        )

        plot_df = df.copy()
        if hide_outliers:
            data = df[df["Order"] == event_index]
            for condition in conditions_to_plot:
                condition_data = data[data["Condition"] == condition]
                mean = condition_data[column].mean()
                std = condition_data[column].std()
                outliers = condition_data[
                    (condition_data[column] < mean - 3 * std)
                    | (condition_data[column] > mean + 3 * std)
                ]

            plot_df.loc[
                (plot_df["Order"] == event_index)
                & (plot_df["Condition"] == condition)
                & (plot_df[column].isin(outliers[column])),
                column,
            ] = np.nan

        if show_swarm:
            dot_size = _swarm_dot_size(plot_df, event_index, column)
            sns.swarmplot(
                data=plot_df[plot_df["Order"] == event_index],
                x="Condition",
                order=conditions_to_plot,
                y=column,
                ax=current_ax,
                alpha=0.5,
                color="black",
                dodge=False,
                size=dot_size,
            )

        current_ax.set_xlabel("")
        if event_index > 0:
            current_ax.set_ylabel("")

        if titles is not None:
            current_ax.set_title(titles[event_index])

        if log_scale:
            current_ax.set_yscale("log")

        current_ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

        if plot_significance:
            _annotate_significance(
                df,
                conditions_to_plot,
                column,
                violinplot,
                significance_pairs,
                event_index,
                plot_type="violinplot",
                test=test,
            )

            if show_metric:
                _add_metric_text(
                    df,
                    conditions_to_plot,
                    column,
                    violinplot,
                    event_index,
                    log_scale,
                    test=test,
                )

        min_y, max_y = current_ax.get_ylim()
        y_min.append(min_y)
        y_max.append(max_y)

    return y_min, y_max


def _plot_boxplot(
    df: pd.DataFrame,
    conditions_to_plot: list,
    column: str,
    color_palette: list,
    ax: matplotlib.axes.Axes | np.ndarray,
    titles: list[str] | None,
    share_y_axis: bool,
    plot_significance: bool,
    significance_pairs: list[tuple] | None,
    log_scale: bool,
    show_metric: bool = False,
    show_swarm: bool = True,
    hide_outliers: bool = False,
    test: str = "Mann-Whitney",
    return_data: bool = False,
) -> tuple[list[float], list[float]]:
    """
    Draw box + swarm subplots for each ordering group.

    Parameters:
        df (pandas.DataFrame) : Data with ``"Order"``, ``"Condition"``, and
            ``column`` columns.
        conditions_to_plot (list) : Ordered condition identifiers.
        column (str) : Y-variable column name.
        color_palette (list) : Colors in the same order as ``conditions_to_plot``.
        ax (np.ndarray or matplotlib.axes.Axes) : Axes array (or scalar) produced
            by ``_setup_figure``.
        titles (list[str] or None) : Subplot titles.
        share_y_axis (bool) : If ``True``, hide y-axis ticks on all but the first subplot.
        plot_significance (bool) : If ``True``, add significance brackets.
        significance_pairs (list[tuple] or None) : Pairs to annotate; all pairs when ``None``.
        log_scale (bool) : Passed to seaborn and ``_add_metric_text`` for log-scale handling.
        show_metric (bool) : If ``True``, display summary statistics below the plot.
            Defaults to ``False``.
        show_swarm (bool) : If ``True``, overlay a swarm plot on the box plot.
            Defaults to ``True``.
        hide_outliers (bool) : If ``True``, remove data points beyond ±3 std in the
            swarm plot.  Defaults to ``False``.
        test (str) : Statistical test for significance annotation.
            Defaults to ``"Mann-Whitney"``.
        return_data (bool) : Unused; reserved for future use.  Defaults to ``False``.

    Returns:
        tuple[list[float], list[float]] : Per-subplot y-axis minima and maxima.
    """
    y_min, y_max = [], []
    for event_index in range(df["Order"].nunique()):
        if share_y_axis:
            if event_index > 0:
                ax[event_index].tick_params(
                    axis="y", which="both", left=False, labelleft=False
                )

        if isinstance(ax, np.ndarray):
            current_ax = ax[event_index]
        else:
            current_ax = ax

        boxplot = sns.boxplot(
            data=df[df["Order"] == event_index],
            x="Condition",
            y=column,
            order=conditions_to_plot,
            hue_order=conditions_to_plot,
            hue="Condition",
            palette=color_palette,
            showfliers=False,
            ax=current_ax,
            dodge=False,
            linewidth=2,
            legend="full",
            linecolor="black",
        )

        plot_df = df.copy()
        if hide_outliers:
            data = df[df["Order"] == event_index]
            for condition in conditions_to_plot:
                condition_data = data[data["Condition"] == condition]
                mean = condition_data[column].mean()
                std = condition_data[column].std()
                outliers = condition_data[
                    (condition_data[column] < mean - 3 * std)
                    | (condition_data[column] > mean + 3 * std)
                ]

            plot_df.loc[
                (plot_df["Order"] == event_index)
                & (plot_df["Condition"] == condition)
                & (plot_df[column].isin(outliers[column])),
                column,
            ] = np.nan

        if log_scale:
            current_ax.set_yscale("log")

        if show_swarm:
            dot_size = _swarm_dot_size(plot_df, event_index, column)
            sns.swarmplot(
                data=plot_df[plot_df["Order"] == event_index],
                x="Condition",
                order=conditions_to_plot,
                y=column,
                ax=current_ax,
                alpha=0.5,
                color="black",
                dodge=False,
                size=dot_size,
            )

        current_ax.set_xlabel("")
        # Hide y-axis labels and ticks for all subplots except the first one
        if event_index > 0:
            current_ax.set_ylabel("")

        if titles is not None:
            current_ax.set_title(titles[event_index])

        # remove ticks
        current_ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

        if plot_significance:
            _annotate_significance(
                df,
                conditions_to_plot,
                column,
                boxplot,
                significance_pairs,
                event_index,
                test=test,
            )

            if show_metric:
                _add_metric_text(
                    df,
                    conditions_to_plot,
                    column,
                    boxplot,
                    event_index,
                    log_scale,
                    test=test,
                )

        min_y, max_y = current_ax.get_ylim()
        y_min.append(min_y)
        y_max.append(max_y)

    return y_min, y_max


def _swarm_dot_size(df: pd.DataFrame, event_index: int, column: str) -> float:
    """
    Compute a dot size for swarm plots that shrinks as sample count grows.

    Uses ``max(3, 6 * sqrt(20 / max(20, n_max)))`` so dots stay at 6 pt up to
    20 points and decay smoothly above that, flooring at 3 pt.

    Parameters:
        df (pandas.DataFrame) : Full data DataFrame with ``"Order"`` and ``"Condition"`` columns.
        event_index (int) : The ``"Order"`` value identifying the current subplot.
        column (str) : Column name used to count non-NaN values.

    Returns:
        float : Dot size in points for ``sns.swarmplot``.
    """
    event_data = df[df["Order"] == event_index]
    n_max = (
        event_data.groupby("Condition")[column].apply(lambda s: s.notna().sum()).max()
    )
    n_max = max(20, int(n_max))
    return max(3.0, 6.0 * (20.0 / n_max) ** 0.5)


def _set_all_y_limits(ax: np.ndarray, y_min: list[float], y_max: list[float]) -> None:
    """
    Synchronise y-axis limits across all subplots with 10% padding.

    Parameters:
        ax (np.ndarray) : Array of Axes objects.
        y_min (list[float]) : Per-subplot y-axis minima.
        y_max (list[float]) : Per-subplot y-axis maxima.

    Returns:
        None
    """
    global_min = min(y_min)
    global_max = max(y_max)
    range_padding = (global_max - global_min) * 0.1  # 5% padding
    global_min = global_min - range_padding
    global_max = global_max + range_padding
    for axes in np.atleast_1d(ax):
        axes.set_ylim(global_min, global_max)


def _set_labels_and_legend(
    ax: matplotlib.axes.Axes | np.ndarray,
    fig: matplotlib.figure.Figure,
    conditions_struct: list,
    conditions_to_plot: list,
    column: str,
    y_axis_label: str | None,
    legend: dict | None,
    legend_placement: str | None = "outside right",
) -> None:
    """
    Set the y-axis label and place a shared figure legend.

    Individual subplot legends are removed; a single legend is added to the figure.

    Parameters:
        ax (np.ndarray or matplotlib.axes.Axes) : Axes array or scalar.
        fig (matplotlib.figure.Figure) : Parent figure.
        conditions_struct (list) : List of condition dicts (used to build legend labels).
        conditions_to_plot (list) : Ordered condition identifiers.
        column (str) : Column name; used as the y-axis label fallback.
        y_axis_label (str or None) : Explicit y-axis label; falls back to ``column``.
        legend (dict or None) : Legend spec passed to ``build_legend``.
        legend_placement (str or None) : Figure legend placement passed to
            ``add_legend``; ``None`` hides the legend.  Defaults to ``"outside right"``.

    Returns:
        None
    """
    if not isinstance(ax, np.ndarray):
        ax = [ax]

    # Set y label for the first plot
    if y_axis_label is not None:
        ax[0].set_ylabel(y_axis_label)
    else:
        ax[0].set_ylabel(column)

    legend_labels = [
        build_legend(conditions_struct[condition_id], legend)
        for condition_id in conditions_to_plot
    ]
    legend_handles = ax[0].get_legend_handles_labels()[0]

    add_legend(fig, legend_placement, legend_handles, legend_labels)


def violinplot(
    conditions_struct: list,
    column: str,
    conditions_to_plot: list,
    events_to_plot: list[int] | None = None,
    log_scale: bool = True,
    figsize: tuple[float, float] | None = None,
    ax_size: tuple[float, float] | None = None,
    colors: list | dict | None = None,
    plot_significance: bool = False,
    show_metric: bool = False,
    significance_pairs: list[tuple] | None = None,
    significance_test: str = "Mann-Whitney",
    legend: dict | None = None,
    y_axis_label: str | None = None,
    titles: list[str] | None = None,
    share_y_axis: bool = False,
    show_swarm: bool = True,
    hide_outliers: bool = True,
    return_data: bool = False,
    legend_placement: str | None = "outside right",
) -> matplotlib.figure.Figure:
    """
    Create violin plots for a per-molt measurement across conditions.

    Each column in ``column`` (axis 1) corresponds to one molt event subplot.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        column (str) : Key of the per-molt measurement array
            (shape ``(n_worms, n_molts)``).
        conditions_to_plot (list) : Ordered condition identifiers.
        events_to_plot (list[int] or None) : Column indices (molt events) to include.
            All events are plotted when ``None``.  Defaults to ``None``.
        log_scale (bool) : If ``True``, render y-axis in log scale via ``set_yscale``.
            Defaults to ``True``.
        figsize (tuple[float, float] or None) : Figure size; auto-sized when ``None``.
            Defaults to ``None``.
        ax_size (tuple[float, float] or None) : If provided, each panel's axes area is fixed to
            ``(ax_w, ax_h)`` inches. Overrides ``figsize``. Defaults to ``None``.
        colors (list or dict or None) : Color spec passed to ``get_colors``.
            Defaults to ``None``.
        plot_significance (bool) : If ``True``, add significance brackets.
            Defaults to ``False``.
        show_metric (bool) : If ``True``, display summary statistics below the plot.
            Defaults to ``False``.
        significance_pairs (list[tuple] or None) : Pairs to annotate; all pairs when ``None``.
            Defaults to ``None``.
        significance_test (str) : Statistical test for annotation.
            Defaults to ``"Mann-Whitney"``.
        legend (dict or None) : Legend spec passed to ``build_legend``.
            Defaults to ``None``.
        y_axis_label (str or None) : Y-axis label; falls back to ``column``.
            Defaults to ``None``.
        titles (list[str] or None) : Subplot titles.  Defaults to ``None``.
        share_y_axis (bool) : If ``True``, synchronise y-axis limits.
            Defaults to ``False``.
        show_swarm (bool) : If ``True``, overlay a swarm plot on the violin plot.
            Defaults to ``True``.
        hide_outliers (bool) : If ``True``, hide swarm-plot points beyond ±3 std.
            Defaults to ``True``.
        return_data (bool) : If ``True``, also return the intermediate DataFrame.
            Defaults to ``False``.
        legend_placement (str or None) : Figure legend placement passed to
            ``add_legend``; ``None`` hides the legend.  Defaults to ``"outside right"``.

    Returns:
        matplotlib.figure.Figure : The generated figure.
        tuple[matplotlib.figure.Figure, pandas.DataFrame] : Figure and DataFrame if
            ``return_data=True``.
    """
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    # Prepare data
    data_list = []
    for condition_id in conditions_to_plot:
        condition_dict = conditions_struct[condition_id]
        data = condition_dict[column]
        if not events_to_plot:
            events_to_plot = range(conditions_struct[condition_id][column].shape[1])

        for idx, j in enumerate(events_to_plot):
            for value in data[:, j]:
                order = idx
                data_list.append(
                    {
                        "Condition": condition_id,
                        "Order": order,
                        "Description": condition_dict["description"],
                        column: value,
                    }
                )

    df = pd.DataFrame(data_list)

    fig, ax = _setup_figure(
        df,
        figsize,
        titles,
        ax_size=ax_size,
    )

    y_min, y_max = _plot_violinplot(
        df,
        conditions_to_plot,
        column,
        color_palette,
        ax,
        titles,
        share_y_axis,
        plot_significance,
        significance_pairs,
        log_scale=log_scale,
        show_metric=show_metric,
        show_swarm=show_swarm,
        hide_outliers=hide_outliers,
        test=significance_test,
    )

    _set_labels_and_legend(
        ax,
        fig,
        conditions_struct,
        conditions_to_plot,
        column,
        y_axis_label,
        legend,
        legend_placement,
    )

    if share_y_axis:
        _set_all_y_limits(ax, y_min, y_max)
        # set the figure to sharey
        all_axes = np.atleast_1d(ax)
        for axes in all_axes:
            axes.sharey(all_axes[0])

    fig = plt.gcf()
    plt.show()

    if return_data:
        return fig, df

    return fig


def boxplot(
    conditions_struct: list,
    column: str,
    conditions_to_plot: list,
    events_to_plot: list[int] | None = None,
    log_scale: bool = True,
    figsize: tuple[float, float] | None = None,
    ax_size: tuple[float, float] | None = None,
    colors: list | dict | None = None,
    plot_significance: bool = False,
    show_metric: bool = False,
    significance_pairs: list[tuple] | None = None,
    significance_test: str = "Mann-Whitney",
    legend: dict | None = None,
    y_axis_label: str | None = None,
    titles: list[str] | None = None,
    share_y_axis: bool = False,
    show_swarm: bool = True,
    hide_outliers: bool = True,
    return_data: bool = False,
    legend_placement: str | None = "outside right",
) -> matplotlib.figure.Figure:
    """
    Create box plots for a per-molt measurement across conditions.

    Log scaling is handled natively by seaborn (unlike ``violinplot`` which
    pre-transforms values).  Each column in ``column`` (axis 1) corresponds to
    one molt event subplot.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        column (str) : Key of the per-molt measurement array
            (shape ``(n_worms, n_molts)``).
        conditions_to_plot (list) : Ordered condition identifiers.
        events_to_plot (list[int] or None) : Column indices (molt events) to include.
            All events are plotted when ``None``.  Defaults to ``None``.
        log_scale (bool) : If ``True``, render y-axis in log scale via ``set_yscale``.
            Defaults to ``True``.
        figsize (tuple[float, float] or None) : Figure size; auto-sized when ``None``.
            Defaults to ``None``.
        ax_size (tuple[float, float] or None) : If provided, each panel's axes area is fixed to
            ``(ax_w, ax_h)`` inches. Overrides ``figsize``. Defaults to ``None``.
        colors (list or dict or None) : Color spec passed to ``get_colors``.
            Defaults to ``None``.
        plot_significance (bool) : If ``True``, add significance brackets.
            Defaults to ``False``.
        show_metric (bool) : If ``True``, display summary statistics below the plot.
            Defaults to ``False``.
        significance_pairs (list[tuple] or None) : Pairs to annotate; all pairs when ``None``.
            Defaults to ``None``.
        significance_test (str) : Statistical test for annotation.
            Defaults to ``"Mann-Whitney"``.
        legend (dict or None) : Legend spec passed to ``build_legend``.
            Defaults to ``None``.
        y_axis_label (str or None) : Y-axis label; falls back to ``column``.
            Defaults to ``None``.
        titles (list[str] or None) : Subplot titles.  Defaults to ``None``.
        share_y_axis (bool) : If ``True``, synchronise y-axis limits.
            Defaults to ``False``.
        show_swarm (bool) : If ``True``, overlay a swarm plot on the box plot.
            Defaults to ``True``.
        hide_outliers (bool) : If ``True``, hide swarm-plot points beyond ±3 std.
            Defaults to ``True``.
        return_data (bool) : If ``True``, also return the intermediate DataFrame.
            Defaults to ``False``.
        legend_placement (str or None) : Figure legend placement passed to
            ``add_legend``; ``None`` hides the legend.  Defaults to ``"outside right"``.

    Returns:
        matplotlib.figure.Figure : The generated figure.
        tuple[matplotlib.figure.Figure, pandas.DataFrame] : Figure and DataFrame if
            ``return_data=True``.
    """
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    # Prepare data
    data_list = []
    for condition_id in conditions_to_plot:
        condition_dict = conditions_struct[condition_id]
        data = condition_dict[column]
        if not events_to_plot:
            events_to_plot = range(conditions_struct[condition_id][column].shape[1])

        for idx, j in enumerate(events_to_plot):
            for value in data[:, j]:
                order = idx
                data_list.append(
                    {
                        "Condition": condition_id,
                        "Order": order,
                        "Description": condition_dict["description"],
                        # column: np.log10(value) if log_scale else value,
                        column: value,
                    }
                )

    df = pd.DataFrame(data_list)

    fig, ax = _setup_figure(
        df,
        figsize,
        titles,
        ax_size=ax_size,
    )

    y_min, y_max = _plot_boxplot(
        df,
        conditions_to_plot,
        column,
        color_palette,
        ax,
        titles,
        share_y_axis,
        plot_significance,
        significance_pairs,
        show_swarm=show_swarm,
        hide_outliers=hide_outliers,
        log_scale=log_scale,
        show_metric=show_metric,
        test=significance_test,
    )

    _set_labels_and_legend(
        ax,
        fig,
        conditions_struct,
        conditions_to_plot,
        column,
        y_axis_label,
        legend,
        legend_placement,
    )

    if share_y_axis:
        _set_all_y_limits(ax, y_min, y_max)
        # set the figure to sharey
        all_axes = np.atleast_1d(ax)
        for axes in all_axes:
            axes.sharey(all_axes[0])

    fig = plt.gcf()
    plt.show()

    if return_data:
        return fig, df

    return fig


def violinplot_larval_stage(
    conditions_struct: list,
    column: str,
    conditions_to_plot: list,
    aggregation: str = "mean",
    n_points: int = 100,
    fraction: tuple[float, float] = (0.2, 0.8),
    log_scale: bool = True,
    figsize: tuple[float, float] | None = None,
    ax_size: tuple[float, float] | None = None,
    colors: list | dict | None = None,
    plot_significance: bool = False,
    significance_pairs: list[tuple] | None = None,
    significance_test: str = "Mann-Whitney",
    legend: dict | None = None,
    y_axis_label: str | None = None,
    titles: list[str] | None = None,
    share_y_axis: bool = False,
    show_metric: bool = False,
    show_swarm: bool = True,
    hide_outliers: bool = True,
    legend_placement: str | None = "outside right",
) -> matplotlib.figure.Figure:
    """
    Create violin plots with per-worm values aggregated within a fraction of each larval stage.

    If ``column`` does not contain ``"rescaled"``, the series is first rescaled via
    ``rescale_without_flattening`` to shape ``(n_worms, 4, n_points)``.  The middle
    fraction of each stage (controlled by ``fraction``) is averaged per worm before
    plotting.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        column (str) : Key of the measurement series.
        conditions_to_plot (list) : Ordered condition identifiers.
        aggregation (str) : Per-worm aggregation within the stage fraction;
            ``"mean"`` or ``"median"``.  Defaults to ``"mean"``.
        n_points (int) : Number of resampled points per larval stage.
            Defaults to ``100``.
        fraction (tuple[float, float]) : Start and end fractions of each stage
            to include in the aggregation.  Defaults to ``(0.2, 0.8)``.
        log_scale (bool) : If ``True``, render y-axis in log scale via ``set_yscale``.
            Defaults to ``True``.
        figsize (tuple[float, float] or None) : Figure size; auto-sized when ``None``.
            Defaults to ``None``.
        ax_size (tuple[float, float] or None) : If provided, each panel's axes area is fixed to
            ``(ax_w, ax_h)`` inches. Overrides ``figsize``. Defaults to ``None``.
        colors (list or dict or None) : Color spec passed to ``get_colors``.
            Defaults to ``None``.
        plot_significance (bool) : If ``True``, add significance brackets.
            Defaults to ``False``.
        significance_pairs (list[tuple] or None) : Pairs to annotate; all pairs when ``None``.
            Defaults to ``None``.
        significance_test (str) : Statistical test for annotation.
            Defaults to ``"Mann-Whitney"``.
        legend (dict or None) : Legend spec passed to ``build_legend``.
            Defaults to ``None``.
        y_axis_label (str or None) : Y-axis label; falls back to ``column``.
            Defaults to ``None``.
        titles (list[str] or None) : Subplot titles.  Defaults to ``None``.
        share_y_axis (bool) : If ``True``, synchronise y-axis limits.
            Defaults to ``False``.
        show_metric (bool) : If ``True``, display summary statistics below the plot.
            Defaults to ``False``.
        show_swarm (bool) : If ``True``, overlay a swarm plot on the violin plot.
            Defaults to ``True``.
        hide_outliers (bool) : If ``True``, hide swarm-plot points beyond ±3 std.
            Defaults to ``True``.
        legend_placement (str or None) : Figure legend placement passed to
            ``add_legend``; ``None`` hides the legend.  Defaults to ``"outside right"``.

    Returns:
        matplotlib.figure.Figure : The generated figure.
    """
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    if "rescaled" not in column:
        rescaled_column = column + "_rescaled"
        conditions_struct = rescale_without_flattening(
            conditions_struct, column, rescaled_column, n_points=n_points
        )
        column = rescaled_column

    # Prepare data
    data_list = []
    for condition_id in conditions_to_plot:
        condition_dict = conditions_struct[condition_id]
        data = condition_dict[column]
        for i in range(data.shape[1]):
            data_of_stage = data[:, i]
            data_of_stage = data_of_stage[
                :,
                int(fraction[0] * data_of_stage.shape[1]) : int(
                    fraction[1] * data_of_stage.shape[1]
                ),
            ]

            if aggregation == "mean":
                aggregated_data_of_stage = np.nanmean(data_of_stage, axis=1)
            elif aggregation == "median":
                aggregated_data_of_stage = np.nanmedian(data_of_stage, axis=1)

            for j in range(aggregated_data_of_stage.shape[0]):
                data_list.append(
                    {
                        "Condition": condition_id,
                        "Order": i,
                        column: aggregated_data_of_stage[j],
                    }
                )

    df = pd.DataFrame(data_list)

    fig, ax = _setup_figure(
        df,
        figsize,
        titles,
        ax_size=ax_size,
    )

    y_min, y_max = _plot_violinplot(
        df,
        conditions_to_plot,
        column,
        color_palette,
        ax,
        titles,
        share_y_axis,
        plot_significance,
        significance_pairs,
        log_scale=log_scale,
        show_metric=show_metric,
        show_swarm=show_swarm,
        hide_outliers=hide_outliers,
        test=significance_test,
    )

    _set_labels_and_legend(
        ax,
        fig,
        conditions_struct,
        conditions_to_plot,
        column,
        y_axis_label,
        legend,
        legend_placement,
    )

    if share_y_axis:
        _set_all_y_limits(ax, y_min, y_max)

    fig = plt.gcf()
    plt.show()

    return fig


def boxplot_larval_stage(
    conditions_struct: list,
    column: str,
    conditions_to_plot: list,
    aggregation: str = "mean",
    n_points: int = 100,
    fraction: tuple[float, float] = (0.2, 0.8),
    log_scale: bool = True,
    figsize: tuple[float, float] | None = None,
    ax_size: tuple[float, float] | None = None,
    colors: list | dict | None = None,
    plot_significance: bool = False,
    significance_pairs: list[tuple] | None = None,
    significance_test: str = "Mann-Whitney",
    legend: dict | None = None,
    y_axis_label: str | None = None,
    titles: list[str] | None = None,
    share_y_axis: bool = False,
    show_metric: bool = False,
    show_swarm: bool = True,
    hide_outliers: bool = True,
    legend_placement: str | None = "outside right",
) -> matplotlib.figure.Figure:
    """
    Create box plots with per-worm values aggregated within a fraction of each larval stage.

    Equivalent to ``violinplot_larval_stage`` but renders box plots instead of violin plots.
    If ``column`` does not contain ``"rescaled"``, the series is first rescaled via
    ``rescale_without_flattening``.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        column (str) : Key of the measurement series.
        conditions_to_plot (list) : Ordered condition identifiers.
        aggregation (str) : Per-worm aggregation within the stage fraction;
            ``"mean"`` or ``"median"``.  Defaults to ``"mean"``.
        n_points (int) : Number of resampled points per larval stage.
            Defaults to ``100``.
        fraction (tuple[float, float]) : Start and end fractions of each stage
            to include in the aggregation.  Defaults to ``(0.2, 0.8)``.
        log_scale (bool) : If ``True``, render y-axis in log scale via ``set_yscale``.
            Defaults to ``True``.
        figsize (tuple[float, float] or None) : Figure size; auto-sized when ``None``.
            Defaults to ``None``.
        ax_size (tuple[float, float] or None) : If provided, each panel's axes area is fixed to
            ``(ax_w, ax_h)`` inches. Overrides ``figsize``. Defaults to ``None``.
        colors (list or dict or None) : Color spec passed to ``get_colors``.
            Defaults to ``None``.
        plot_significance (bool) : If ``True``, add significance brackets.
            Defaults to ``False``.
        significance_pairs (list[tuple] or None) : Pairs to annotate; all pairs when ``None``.
            Defaults to ``None``.
        significance_test (str) : Statistical test for annotation.
            Defaults to ``"Mann-Whitney"``.
        legend (dict or None) : Legend spec passed to ``build_legend``.
            Defaults to ``None``.
        y_axis_label (str or None) : Y-axis label; falls back to ``column``.
            Defaults to ``None``.
        titles (list[str] or None) : Subplot titles.  Defaults to ``None``.
        share_y_axis (bool) : If ``True``, synchronise y-axis limits.
            Defaults to ``False``.
        show_metric (bool) : If ``True``, display summary statistics below the plot.
            Defaults to ``False``.
        show_swarm (bool) : If ``True``, overlay a swarm plot on the box plot.
            Defaults to ``True``.
        hide_outliers (bool) : If ``True``, hide swarm-plot points beyond ±3 std.
            Defaults to ``True``.
        legend_placement (str or None) : Figure legend placement passed to
            ``add_legend``; ``None`` hides the legend.  Defaults to ``"outside right"``.

    Returns:
        matplotlib.figure.Figure : The generated figure.
    """
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    if "rescaled" not in column:
        rescaled_column = column + "_rescaled"
        conditions_struct = rescale_without_flattening(
            conditions_struct, column, rescaled_column, n_points=n_points
        )
        column = rescaled_column

    # Prepare data
    data_list = []
    for condition_id in conditions_to_plot:
        condition_dict = conditions_struct[condition_id]
        data = condition_dict[column]
        for i in range(data.shape[1]):
            data_of_stage = data[:, i]
            data_of_stage = data_of_stage[
                :,
                int(fraction[0] * data_of_stage.shape[1]) : int(
                    fraction[1] * data_of_stage.shape[1]
                ),
            ]

            if aggregation == "mean":
                aggregated_data_of_stage = np.nanmean(data_of_stage, axis=1)
            elif aggregation == "median":
                aggregated_data_of_stage = np.nanmedian(data_of_stage, axis=1)

            for j in range(aggregated_data_of_stage.shape[0]):
                data_list.append(
                    {
                        "Condition": condition_id,
                        "Order": i,
                        column: aggregated_data_of_stage[j],
                    }
                )

    df = pd.DataFrame(data_list)

    fig, ax = _setup_figure(
        df,
        figsize,
        titles,
        ax_size=ax_size,
    )

    y_min, y_max = _plot_boxplot(
        df,
        conditions_to_plot,
        column,
        color_palette,
        ax,
        titles,
        share_y_axis,
        plot_significance,
        significance_pairs,
        log_scale=log_scale,
        show_metric=show_metric,
        show_swarm=show_swarm,
        hide_outliers=hide_outliers,
        test=significance_test,
    )

    _set_labels_and_legend(
        ax,
        fig,
        conditions_struct,
        conditions_to_plot,
        column,
        y_axis_label,
        legend,
        legend_placement,
    )

    if share_y_axis:
        _set_all_y_limits(ax, y_min, y_max)

    fig = plt.gcf()
    plt.show()

    return fig
