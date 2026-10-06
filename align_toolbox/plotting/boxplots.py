from collections.abc import Callable
from itertools import combinations
from typing import Literal

import bottleneck as bn
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statannotations.Annotator import Annotator
from statannotations.format_annotations import pval_annotation_text, simple_text
from statannotations.stats.StatResult import StatResult
from statannotations.stats.StatTest import STATTEST_LIBRARY, StatTest
from statsmodels.stats.multitest import multipletests

from .utils_data_processing import rescale_without_flattening
from .utils_plotting import (
    add_legend,
    build_legend,
    create_fixed_ax_sized_fig,
    get_colors,
)

STATANNOTATIONS_TESTS = STATTEST_LIBRARY.keys()
CUSTOM_TESTS = ["Feltz-Miller", "MSLR", "Brown-Forsythe"]
HOLM_FAMILIES = ["panel", "pair", "figure"]


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


def brown_forsythe_test(
    sample1: np.ndarray, sample2: np.ndarray
) -> tuple[float, float]:
    """
    Perform the Brown-Forsythe test for equality of variances on two samples.

    This is Levene's test with deviations taken from each sample's median rather
    than its mean, which keeps it robust to skewed and heavy-tailed data.

    Parameters:
        sample1 (array-like) : First sample values.
        sample2 (array-like) : Second sample values.

    Returns:
        tuple[float, float] : Test statistic ``W`` and p-value.
    """
    result = stats.levene(sample1, sample2, center="median")
    return result.statistic, result.pvalue


def _get_stat_test(test: str) -> StatTest:
    """
    Build the statannotations test for ``test``.

    Parameters:
        test (str): Statannotations built-in test name, ``"Feltz-Miller"``,
            ``"MSLR"`` or ``"Brown-Forsythe"``.

    Returns:
        StatTest: The test; calling it on two samples returns a ``StatResult``.

    Raises:
        ValueError: If ``test`` is not supported.
    """
    if test in STATTEST_LIBRARY:
        stat_test = STATTEST_LIBRARY[test]
    elif test == "Feltz-Miller":
        stat_test = StatTest(
            feltz_miller_asymptotic_cv_test,
            "Feltz-Miller Asymptotic Test",
            "Feltz-Miller",
        )
    elif test == "MSLR":
        stat_test = StatTest(mslr_test, "Modified Signed Likelihood Ratio Test", "MSLR")
    elif test == "Brown-Forsythe":
        stat_test = StatTest(
            brown_forsythe_test, "Brown-Forsythe Test", "Brown-Forsythe"
        )
    else:
        raise ValueError(
            f"Test {test} is not supported. Please use one of the following: "
            f"{list(STATANNOTATIONS_TESTS) + CUSTOM_TESTS}"
        )
    return stat_test


STAR_THRESHOLDS = [[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]
SIMPLE_THRESHOLDS = [[1e-5, "1e-5"], [1e-4, "1e-4"], [1e-3, "0.001"], [1e-2, "0.01"]]


def _format_test_result(test: str, result: StatResult) -> str:
    """
    Format one test result as a line of bracket text, prefixed with the test name.

    Mann-Whitney results are written as stars, all other tests as a p-value.

    Parameters:
        test (str): Test name, as passed to ``_get_stat_test``.
        result (StatResult): Result of the test on one pair.

    Returns:
        str: The annotation line, e.g. ``"Mann-Whitney ***"`` or
        ``"Levene p = 0.03"``.
    """
    if test == "Mann-Whitney":
        text = pval_annotation_text([result], STAR_THRESHOLDS)[0][0]
    else:
        text = simple_text(result, "{:.2f}", SIMPLE_THRESHOLDS, short_test_name=False)
    name = test if test[0].isupper() else test.capitalize()
    return f"{name} {text}"


def _compute_significance_annotations(
    df: pd.DataFrame,
    conditions_to_plot: list,
    column: str,
    significance_pairs: list[tuple] | None,
    test: str | list[str] = "Mann-Whitney",
    holm_correction: Literal["panel", "pair", "figure"] | None = "pair",
    verbose: bool = True,
) -> dict[int, tuple[list[tuple], list[str]]]:
    """
    Run the significance tests of every subplot and format their bracket texts.

    Each test is run on every pair in every subplot.  With ``holm_correction``,
    the p-values of each test are Holm-adjusted within a family: the pairs of one
    subplot (``"panel"``), one pair across all subplots (``"pair"``), or every
    pair of every subplot (``"figure"``).  Tests with a NaN p-value are left out
    of their family.

    Parameters:
        df (pandas.DataFrame) : Full data DataFrame with ``"Order"`` and
            ``"Condition"`` columns.
        conditions_to_plot (list) : Ordered condition identifiers.
        column (str) : Column name of the y-variable.
        significance_pairs (list[tuple] or None) : Explicit pairs to annotate;
            all pairwise combinations are used when ``None``.
        test (str or list[str]) : Statistical test name, or several names to share
            each bracket.  Statannotations built-in tests are supported as well as
            ``"Feltz-Miller"``, ``"MSLR"`` and ``"Brown-Forsythe"``.
            Defaults to ``"Mann-Whitney"``.
        holm_correction (str or None) : Holm family, ``"panel"``, ``"pair"`` or
            ``"figure"``; ``None`` shows the raw p-values.  Defaults to ``"pair"``.
        verbose (bool) : If ``True``, print sample sizes and test details.
            Defaults to ``True``.

    Returns:
        dict[int, tuple[list[tuple], list[str]]] : ``(pairs, texts)`` keyed by
            ``"Order"`` value, one text per pair.

    Raises:
        ValueError : If a test or ``holm_correction`` is not supported.
    """
    if holm_correction not in (*HOLM_FAMILIES, None):
        raise ValueError(
            f"holm_correction must be one of {HOLM_FAMILIES} or None, "
            f"got {holm_correction!r}."
        )
    tests = [test] if isinstance(test, str) else list(test)
    stat_tests = [_get_stat_test(name) for name in tests]
    if significance_pairs is None:
        pairs = list(combinations(df["Condition"].unique(), 2))
    else:
        pairs = list(significance_pairs)
    panels = range(df["Order"].nunique())

    # keyed by (test index, panel, pair index)
    results = {}
    for panel in panels:
        panel_df = df[df["Order"] == panel]
        if verbose:
            print(
                f"\nSample sizes (non-NaN) for event index {panel}, column '{column}':"
            )
            for condition in conditions_to_plot:
                n = panel_df.loc[panel_df["Condition"] == condition, column].count()
                print(f"Condition {condition}: n={n}")
        for k, pair in enumerate(pairs):
            samples = [
                panel_df.loc[panel_df["Condition"] == condition, column].dropna()
                for condition in pair
            ]
            for i, stat_test in enumerate(stat_tests):
                results[(i, panel, k)] = stat_test(*samples)

    if holm_correction is not None:
        families = {}
        for (i, panel, k), result in results.items():
            family = {"panel": (i, panel), "pair": (i, k), "figure": (i,)}[
                holm_correction
            ]
            families.setdefault(family, []).append(result)
        for family_results in families.values():
            _apply_holm_correction(family_results)

    annotations = {}
    for panel in panels:
        texts = []
        for k, (first, second) in enumerate(pairs):
            pair_results = [results[(i, panel, k)] for i in range(len(tests))]
            if verbose:
                for result in pair_results:
                    print(f"{first} vs. {second}: {result.formatted_output}")
            texts.append(
                "\n".join(
                    _format_test_result(name, result)
                    for name, result in zip(tests, pair_results)
                )
            )
        annotations[panel] = (pairs, texts)
    return annotations


def _annotate_significance(
    conditions_to_plot: list,
    column: str,
    ax: matplotlib.axes.Axes,
    pairs: list[tuple],
    texts: list[str],
    event_index: int,
    shown_df: pd.DataFrame,
    plot_type: str = "boxplot",
) -> None:
    """
    Draw significance brackets with precomputed texts on a single subplot.

    The brackets are placed from ``shown_df``, so the plot can display
    transformed values while the texts come from tests on the original ones.

    Parameters:
        conditions_to_plot (list) : Ordered condition identifiers.
        column (str) : Column name of the y-variable.
        ax (matplotlib.axes.Axes) : Axes object of the target subplot.
        pairs (list[tuple]) : Pairs of conditions to bracket.
        texts (list[str]) : Bracket texts, one per pair.
        event_index (int) : The ``"Order"`` value identifying the current subplot.
        shown_df (pandas.DataFrame) : The values drawn on the axes, with
            ``"Order"``, ``"Condition"`` and ``column`` columns.
        plot_type (str) : ``"boxplot"``, ``"violinplot"`` or ``"swarmplot"``.
            Defaults to ``"boxplot"``.

    Returns:
        None

    Raises:
        ValueError : If ``texts`` does not have one text per pair.
    """
    if len(texts) != len(pairs):
        raise ValueError("texts needs one text per pair.")
    shown_values = shown_df.loc[shown_df["Order"] == event_index]

    def draw_brackets() -> None:
        annotator = Annotator(
            ax=ax,
            pairs=pairs,
            data=shown_values,
            x="Condition",
            order=conditions_to_plot,
            y=column,
            plot=plot_type,
        )
        annotator.configure(loc="inside", verbose=False)
        annotator.set_custom_annotations(list(texts))
        annotator.annotate()

    # statannotations stacks brackets in axes fractions, then raises the y limit to
    # fit them, which squeezes the text it measured into overlap. The text and
    # bracket heights are fixed in axes fractions while the data shrinks as the
    # limit rises, so the first pass tells us the limit at which everything fits.
    n_lines, n_texts = len(ax.lines), len(ax.texts)
    bottom, top = ax.get_ylim()
    draw_brackets()
    stack_top = _axis_fraction(ax, bottom, ax.get_ylim()[1]) / 1.04
    for artist in ax.lines[n_lines:] + ax.texts[n_texts:]:
        artist.remove()
    ax.set_ylim(bottom, top)

    data_top = _axis_fraction(ax, bottom, np.nanmax(shown_values[column]))
    annotations_height = stack_top - data_top
    # past 90 % of the axis the brackets cannot fit; leave the first-pass limit
    axis_top = data_top / max(1 / 1.04 - annotations_height, 0.1)
    to_axes = ax.transScale + ax.transLimits
    new_top = to_axes.inverted().transform((0, axis_top))[1]
    ax.set_ylim(bottom, max(new_top, top))
    draw_brackets()


def _apply_holm_correction(results: list[StatResult]) -> None:
    """
    Replace the p-values of a family of test results with Holm-adjusted ones.

    Parameters:
        results (list[StatResult]): Results of one test across pairs; their
            ``pvalue`` is overwritten in place.

    Returns:
        None
    """
    tested = [result for result in results if not np.isnan(result.pvalue)]
    if not tested:
        return
    adjusted = multipletests([result.pvalue for result in tested], method="holm")[1]
    for result, pvalue in zip(tested, adjusted):
        result.pvalue = pvalue
        result.correction_method = "Holm-Bonferroni"


def _axis_fraction(ax: matplotlib.axes.Axes, low: float, high: float) -> float:
    """
    Measure the distance between two y values as a fraction of the axis height.

    Parameters:
        ax (matplotlib.axes.Axes): Axes whose current y limits and scale are used.
        low (float): Lower y value, in data coordinates.
        high (float): Upper y value, in data coordinates.

    Returns:
        float: ``(high - low)`` in axes-fraction units.
    """
    to_axes = ax.transScale + ax.transLimits
    return to_axes.transform((0, high))[1] - to_axes.transform((0, low))[1]


def _add_metric_text(
    df: pd.DataFrame,
    conditions_to_plot: list,
    column: str,
    ax: matplotlib.axes.Axes,
    event_index: int,
    log_scale: bool,
    test: str | list[str] = "Mann-Whitney",
    y_offset_pct: float = 0.1,
    significant_digits: int = 3,
    display_transform: Callable | None = None,
) -> None:
    """
    Annotate each condition with its relevant summary statistic below the plot area.

    The statistic displayed depends on the test: median (Mann-Whitney, Kruskal-Wallis,
    Wilcoxon), mean (t-test, Welch), std (Levene, Brown-Forsythe), or CV %
    (Feltz-Miller, MSLR).
    With several tests, the statistic of each is stacked in one box, in test order
    and without repeats.

    Parameters:
        df (pandas.DataFrame) : Full data DataFrame with ``"Order"`` and
            ``"Condition"`` columns.
        conditions_to_plot (list) : Ordered condition identifiers.
        column (str) : Column name of the y-variable.
        ax (matplotlib.axes.Axes) : Axes object of the target subplot.
        event_index (int) : The ``"Order"`` value identifying the current subplot.
        log_scale (bool) : If ``True``, adjust y-position calculation for log-scale axes.
        test (str or list[str]) : Statistical test name(s); determines which
            statistics to display.  Defaults to ``"Mann-Whitney"``.
        y_offset_pct (float) : Downward offset of the text box as a fraction of the
            y-axis range.  Defaults to ``0.1``.
        significant_digits (int) : Number of significant digits in the displayed value.
            Defaults to ``3``.
        display_transform (Callable or None) : If given, the mean and median are
            computed on ``column`` and then mapped through this function to the
            displayed scale; spread metrics stay on the original scale.
            Defaults to ``None``.

    Returns:
        None

    Raises:
        ValueError : If a test is not in the supported list.
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
        "Brown-Forsythe": ("std", "σ"),
    }

    tests = [test] if isinstance(test, str) else list(test)
    for name in tests:
        if name not in test_metrics:
            raise ValueError(
                f"Test '{name}' not supported. "
                f"Available tests: {list(test_metrics.keys())}"
            )
    metrics = list(dict.fromkeys(test_metrics[name] for name in tests))

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

        lines = []
        for metric_type, symbol in metrics:
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
            if display_transform is not None and metric_type in ("mean", "median"):
                metric_value = display_transform(metric_value)

            line = f"{symbol} = {metric_value:.{significant_digits}g}"
            if metric_type == "cv":
                line += " %"
            lines.append(line)
        if not lines:
            continue
        text = "\n".join(lines)

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

    # each extra stacked line needs roughly another 6 % of the axis below the plot
    bottom_pct = y_offset_pct + 0.04 + 0.06 * (len(metrics) - 1)
    if log_scale:
        ax.set_ylim(10 ** (log_y_min - log_range * bottom_pct), y_max)
    else:
        ax.set_ylim(y_min - y_range * bottom_pct, y_max)


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
    test: str | list[str] = "Mann-Whitney",
    show_swarm: bool = True,
    hide_outliers: bool = False,
    inner: str | None = "box",
    display_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    custom_annotations: dict[int, tuple[list[tuple], list[str]]] | None = None,
    holm_correction: Literal["panel", "pair", "figure"] | None = "pair",
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
        test (str or list[str]) : Statistical test for significance annotation, or
            several tests stacked on each bracket.  Defaults to ``"Mann-Whitney"``.
        show_swarm (bool) : If ``True``, overlay a swarm plot on the violin plot.
            Defaults to ``True``.
        hide_outliers (bool) : If ``True``, remove data points beyond ±3 std in the
            swarm plot (violin retains them).  Defaults to ``False``.
        inner (str or None) : Passed to seaborn violinplot ``inner`` parameter. Defaults to ``None``.
        display_transform (Callable or None) : Applied to the values only when
            drawing them; tests, outlier detection and metrics use the original
            values.  Defaults to ``None``.
        custom_annotations (dict or None) : Precomputed brackets per ``"Order"``
            value, as ``(pairs, texts)``.  When given, these are drawn instead of
            running ``test``; groups without an entry or without pairs get no
            brackets.  Defaults to ``None``.
        holm_correction (str or None) : Holm family passed to
            ``_compute_significance_annotations``; ignored with
            ``custom_annotations``.  Defaults to ``"pair"``.

    Returns:
        tuple[list[float], list[float]] : Per-subplot y-axis minima and maxima.
    """
    shown_df = df.copy()
    if display_transform is not None:
        shown_df[column] = display_transform(df[column].to_numpy())
    if plot_significance and custom_annotations is None:
        custom_annotations = _compute_significance_annotations(
            df, conditions_to_plot, column, significance_pairs, test, holm_correction
        )
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
            data=shown_df[shown_df["Order"] == event_index],
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
            linecolor="black",
        )

        plot_df = shown_df.copy()
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
                plot_df.loc[outliers.index, column] = np.nan

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
            # metrics first: they lower the y limit, which would squeeze brackets
            if show_metric:
                _add_metric_text(
                    df,
                    conditions_to_plot,
                    column,
                    violinplot,
                    event_index,
                    log_scale,
                    test=test,
                    display_transform=display_transform,
                )
            pairs, texts = custom_annotations.get(event_index, ([], []))
            if pairs:
                _annotate_significance(
                    conditions_to_plot,
                    column,
                    violinplot,
                    pairs,
                    texts,
                    event_index,
                    shown_df,
                    plot_type="violinplot",
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
    test: str | list[str] = "Mann-Whitney",
    return_data: bool = False,
    display_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    holm_correction: Literal["panel", "pair", "figure"] | None = "pair",
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
        test (str or list[str]) : Statistical test for significance annotation, or
            several tests stacked on each bracket.  Defaults to ``"Mann-Whitney"``.
        return_data (bool) : Unused; reserved for future use.  Defaults to ``False``.
        display_transform (Callable or None) : Applied to the values only when
            drawing them; tests, outlier detection and metrics use the original
            values.  Defaults to ``None``.
        holm_correction (str or None) : Holm family passed to
            ``_compute_significance_annotations``; ``None`` shows the raw p-values.
            Defaults to ``"pair"``.

    Returns:
        tuple[list[float], list[float]] : Per-subplot y-axis minima and maxima.
    """
    shown_df = df.copy()
    if display_transform is not None:
        shown_df[column] = display_transform(df[column].to_numpy())
    if plot_significance:
        annotations = _compute_significance_annotations(
            df, conditions_to_plot, column, significance_pairs, test, holm_correction
        )
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
            data=shown_df[shown_df["Order"] == event_index],
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

        plot_df = shown_df.copy()
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
                plot_df.loc[outliers.index, column] = np.nan

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
            # metrics first: they lower the y limit, which would squeeze brackets
            if show_metric:
                _add_metric_text(
                    df,
                    conditions_to_plot,
                    column,
                    boxplot,
                    event_index,
                    log_scale,
                    test=test,
                    display_transform=display_transform,
                )
            pairs, texts = annotations[event_index]
            _annotate_significance(
                conditions_to_plot,
                column,
                boxplot,
                pairs,
                texts,
                event_index,
                shown_df,
            )

        min_y, max_y = current_ax.get_ylim()
        y_min.append(min_y)
        y_max.append(max_y)

    return y_min, y_max


def _plot_swarmplot(
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
    hide_outliers: bool = False,
    test: str | list[str] = "Mann-Whitney",
    display_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    holm_correction: Literal["panel", "pair", "figure"] | None = "pair",
) -> tuple[list[float], list[float]]:
    """
    Draw swarm subplots, colored by condition, for each ordering group.

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
        log_scale (bool) : If ``True``, use a log y axis; also passed to
            ``_add_metric_text``.
        show_metric (bool) : If ``True``, display summary statistics below the plot.
            Defaults to ``False``.
        hide_outliers (bool) : If ``True``, remove data points beyond ±3 std from the
            swarm; tests and metrics still use them.  Defaults to ``False``.
        test (str or list[str]) : Statistical test for significance annotation, or
            several tests stacked on each bracket.  Defaults to ``"Mann-Whitney"``.
        display_transform (Callable or None) : Applied to the values only when
            drawing them; tests, outlier detection and metrics use the original
            values.  Defaults to ``None``.
        holm_correction (str or None) : Holm family passed to
            ``_compute_significance_annotations``; ``None`` shows the raw p-values.
            Defaults to ``"pair"``.

    Returns:
        tuple[list[float], list[float]] : Per-subplot y-axis minima and maxima.
    """
    shown_df = df.copy()
    if display_transform is not None:
        shown_df[column] = display_transform(df[column].to_numpy())
    if plot_significance:
        annotations = _compute_significance_annotations(
            df, conditions_to_plot, column, significance_pairs, test, holm_correction
        )
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

        plot_df = shown_df.copy()
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
                plot_df.loc[outliers.index, column] = np.nan

        # swarm layout is computed in display space, so the scale must be set first
        if log_scale:
            current_ax.set_yscale("log")

        swarmplot = sns.swarmplot(
            data=plot_df[plot_df["Order"] == event_index],
            x="Condition",
            y=column,
            order=conditions_to_plot,
            hue_order=conditions_to_plot,
            hue="Condition",
            palette=color_palette,
            ax=current_ax,
            dodge=False,
            size=_swarm_dot_size(plot_df, event_index, column),
            edgecolor="black",
            linewidth=0.5,
            legend="full",
        )

        current_ax.set_xlabel("")
        if event_index > 0:
            current_ax.set_ylabel("")

        if titles is not None:
            current_ax.set_title(titles[event_index])

        current_ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

        if plot_significance:
            # metrics first: they lower the y limit, which would squeeze brackets
            if show_metric:
                _add_metric_text(
                    df,
                    conditions_to_plot,
                    column,
                    swarmplot,
                    event_index,
                    log_scale,
                    test=test,
                    display_transform=display_transform,
                )
            pairs, texts = annotations[event_index]
            _annotate_significance(
                conditions_to_plot,
                column,
                swarmplot,
                pairs,
                texts,
                event_index,
                shown_df,
                plot_type="swarmplot",
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
    legend_as_xticks: bool = False,
) -> None:
    """
    Set the y-axis label and place a shared figure legend.

    Individual subplot legends are removed; a single legend is added to the figure,
    or, with ``legend_as_xticks``, the legend labels are written as x tick labels
    under each box or violin instead.

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
        legend_as_xticks (bool) : If ``True``, label each box or violin with its legend
            text on the x axis and draw no legend; ``legend_placement`` is ignored.
            Defaults to ``False``.

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

    if legend_as_xticks:
        for axes in ax:
            axes.set_xticks(range(len(legend_labels)), legend_labels)
            axes.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
        add_legend(fig, None)
        return

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
    significance_test: str | list[str] = "Mann-Whitney",
    holm_correction: Literal["panel", "pair", "figure"] | None = "pair",
    legend: dict | None = None,
    y_axis_label: str | None = None,
    titles: list[str] | None = None,
    share_y_axis: bool = False,
    show_swarm: bool = True,
    hide_outliers: bool = True,
    return_data: bool = False,
    legend_placement: str | None = "outside right",
    legend_as_xticks: bool = False,
    display_transform: Callable[[np.ndarray], np.ndarray] | None = None,
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
        significance_test (str or list[str]) : Statistical test for annotation.  With
            a list, e.g. ``["Mann-Whitney", "Levene"]``, every bracket shows one line
            per test, stacked in list order and labelled with the test name.
            Defaults to ``"Mann-Whitney"``.
        holm_correction (str or None) : Correct for multiple comparisons with the
            Holm method; the brackets show the adjusted p-values of each test.
            ``"pair"`` adjusts each pair across all subplots, ``"panel"`` adjusts
            the pairs within each subplot, ``"figure"`` adjusts every pair of
            every subplot together, and ``None`` shows the raw p-values.
            Defaults to ``"pair"``.
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
        legend_as_xticks (bool) : If ``True``, draw no legend and instead label each
            box or violin with its legend text on the x axis; ``legend_placement``
            is ignored.  Defaults to ``False``.
        display_transform (Callable or None) : Function applied to the values only
            when drawing them, e.g. ``proportions.log_ratio_to_percentage`` to show
            log-ratio deviations as percents.  Significance tests, outlier hiding
            and spread metrics use the untransformed values; mean and median
            metrics are computed on them and then transformed.
            Defaults to ``None``.

    Returns:
        matplotlib.figure.Figure : The generated figure.
        tuple[matplotlib.figure.Figure, pandas.DataFrame] : Figure and DataFrame if
            ``return_data=True``; the DataFrame holds the untransformed values.
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
        holm_correction=holm_correction,
        display_transform=display_transform,
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
        legend_as_xticks,
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
    significance_test: str | list[str] = "Mann-Whitney",
    holm_correction: Literal["panel", "pair", "figure"] | None = "pair",
    legend: dict | None = None,
    y_axis_label: str | None = None,
    titles: list[str] | None = None,
    share_y_axis: bool = False,
    show_swarm: bool = True,
    hide_outliers: bool = True,
    return_data: bool = False,
    legend_placement: str | None = "outside right",
    legend_as_xticks: bool = False,
    display_transform: Callable[[np.ndarray], np.ndarray] | None = None,
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
        significance_test (str or list[str]) : Statistical test for annotation.  With
            a list, e.g. ``["Mann-Whitney", "Levene"]``, every bracket shows one line
            per test, stacked in list order and labelled with the test name.
            Defaults to ``"Mann-Whitney"``.
        holm_correction (str or None) : Correct for multiple comparisons with the
            Holm method; the brackets show the adjusted p-values of each test.
            ``"pair"`` adjusts each pair across all subplots, ``"panel"`` adjusts
            the pairs within each subplot, ``"figure"`` adjusts every pair of
            every subplot together, and ``None`` shows the raw p-values.
            Defaults to ``"pair"``.
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
        legend_as_xticks (bool) : If ``True``, draw no legend and instead label each
            box or violin with its legend text on the x axis; ``legend_placement``
            is ignored.  Defaults to ``False``.
        display_transform (Callable or None) : Function applied to the values only
            when drawing them, e.g. ``proportions.log_ratio_to_percentage`` to show
            log-ratio deviations as percents.  Significance tests, outlier hiding
            and spread metrics use the untransformed values; mean and median
            metrics are computed on them and then transformed.
            Defaults to ``None``.

    Returns:
        matplotlib.figure.Figure : The generated figure.
        tuple[matplotlib.figure.Figure, pandas.DataFrame] : Figure and DataFrame if
            ``return_data=True``; the DataFrame holds the untransformed values.
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
        holm_correction=holm_correction,
        display_transform=display_transform,
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
        legend_as_xticks,
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


def swarmplot(
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
    significance_test: str | list[str] = "Mann-Whitney",
    holm_correction: Literal["panel", "pair", "figure"] | None = "pair",
    legend: dict | None = None,
    y_axis_label: str | None = None,
    titles: list[str] | None = None,
    share_y_axis: bool = False,
    hide_outliers: bool = False,
    return_data: bool = False,
    legend_placement: str | None = "outside right",
    legend_as_xticks: bool = False,
    display_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> matplotlib.figure.Figure:
    """
    Create swarm plots for a per-molt measurement across conditions.

    Every worm is drawn as one dot colored by its condition.  Each column in
    ``column`` (axis 1) corresponds to one molt event subplot.

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
        significance_test (str or list[str]) : Statistical test for annotation.  With
            a list, e.g. ``["Mann-Whitney", "Levene"]``, every bracket shows one line
            per test, stacked in list order and labelled with the test name.
            Defaults to ``"Mann-Whitney"``.
        holm_correction (str or None) : Correct for multiple comparisons with the
            Holm method; the brackets show the adjusted p-values of each test.
            ``"pair"`` adjusts each pair across all subplots, ``"panel"`` adjusts
            the pairs within each subplot, ``"figure"`` adjusts every pair of
            every subplot together, and ``None`` shows the raw p-values.
            Defaults to ``"pair"``.
        legend (dict or None) : Legend spec passed to ``build_legend``.
            Defaults to ``None``.
        y_axis_label (str or None) : Y-axis label; falls back to ``column``.
            Defaults to ``None``.
        titles (list[str] or None) : Subplot titles.  Defaults to ``None``.
        share_y_axis (bool) : If ``True``, synchronise y-axis limits.
            Defaults to ``False``.
        hide_outliers (bool) : If ``True``, hide points beyond ±3 std.  Unlike box and
            violin plots, the swarm is the only element drawn, so this is off by
            default.  Defaults to ``False``.
        return_data (bool) : If ``True``, also return the intermediate DataFrame.
            Defaults to ``False``.
        legend_placement (str or None) : Figure legend placement passed to
            ``add_legend``; ``None`` hides the legend.  Defaults to ``"outside right"``.
        legend_as_xticks (bool) : If ``True``, draw no legend and instead label each
            swarm with its legend text on the x axis; ``legend_placement`` is
            ignored.  Defaults to ``False``.
        display_transform (Callable or None) : Function applied to the values only
            when drawing them, e.g. ``proportions.log_ratio_to_percentage`` to show
            log-ratio deviations as percents.  Significance tests, outlier hiding
            and spread metrics use the untransformed values; mean and median
            metrics are computed on them and then transformed.
            Defaults to ``None``.

    Returns:
        matplotlib.figure.Figure : The generated figure.
        tuple[matplotlib.figure.Figure, pandas.DataFrame] : Figure and DataFrame if
            ``return_data=True``; the DataFrame holds the untransformed values.
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
                data_list.append(
                    {
                        "Condition": condition_id,
                        "Order": idx,
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

    y_min, y_max = _plot_swarmplot(
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
        hide_outliers=hide_outliers,
        test=significance_test,
        holm_correction=holm_correction,
        display_transform=display_transform,
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
        legend_as_xticks,
    )

    if share_y_axis:
        _set_all_y_limits(ax, y_min, y_max)
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
    significance_test: str | list[str] = "Mann-Whitney",
    holm_correction: Literal["panel", "pair", "figure"] | None = "pair",
    legend: dict | None = None,
    y_axis_label: str | None = None,
    titles: list[str] | None = None,
    share_y_axis: bool = False,
    show_metric: bool = False,
    show_swarm: bool = True,
    hide_outliers: bool = True,
    legend_placement: str | None = "outside right",
    legend_as_xticks: bool = False,
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
        significance_test (str or list[str]) : Statistical test for annotation.  With
            a list, e.g. ``["Mann-Whitney", "Levene"]``, every bracket shows one line
            per test, stacked in list order and labelled with the test name.
            Defaults to ``"Mann-Whitney"``.
        holm_correction (str or None) : Correct for multiple comparisons with the
            Holm method; the brackets show the adjusted p-values of each test.
            ``"pair"`` adjusts each pair across all subplots, ``"panel"`` adjusts
            the pairs within each subplot, ``"figure"`` adjusts every pair of
            every subplot together, and ``None`` shows the raw p-values.
            Defaults to ``"pair"``.
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
        legend_as_xticks (bool) : If ``True``, draw no legend and instead label each
            box or violin with its legend text on the x axis; ``legend_placement``
            is ignored.  Defaults to ``False``.

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
        holm_correction=holm_correction,
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
        legend_as_xticks,
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
    significance_test: str | list[str] = "Mann-Whitney",
    holm_correction: Literal["panel", "pair", "figure"] | None = "pair",
    legend: dict | None = None,
    y_axis_label: str | None = None,
    titles: list[str] | None = None,
    share_y_axis: bool = False,
    show_metric: bool = False,
    show_swarm: bool = True,
    hide_outliers: bool = True,
    legend_placement: str | None = "outside right",
    legend_as_xticks: bool = False,
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
        significance_test (str or list[str]) : Statistical test for annotation.  With
            a list, e.g. ``["Mann-Whitney", "Levene"]``, every bracket shows one line
            per test, stacked in list order and labelled with the test name.
            Defaults to ``"Mann-Whitney"``.
        holm_correction (str or None) : Correct for multiple comparisons with the
            Holm method; the brackets show the adjusted p-values of each test.
            ``"pair"`` adjusts each pair across all subplots, ``"panel"`` adjusts
            the pairs within each subplot, ``"figure"`` adjusts every pair of
            every subplot together, and ``None`` shows the raw p-values.
            Defaults to ``"pair"``.
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
        legend_as_xticks (bool) : If ``True``, draw no legend and instead label each
            box or violin with its legend text on the x axis; ``legend_placement``
            is ignored.  Defaults to ``False``.

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
        holm_correction=holm_correction,
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
        legend_as_xticks,
    )

    if share_y_axis:
        _set_all_y_limits(ax, y_min, y_max)

    fig = plt.gcf()
    plt.show()

    return fig
