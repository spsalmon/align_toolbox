import matplotlib.figure
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from scipy import stats

from align_toolbox.plotting import boxplots, proportions


@pytest.fixture
def equal_cv_samples():
    rng = np.random.default_rng(0)
    return rng.normal(100, 10, 200), rng.normal(50, 5, 200)


@pytest.fixture
def different_cv_samples():
    rng = np.random.default_rng(0)
    return rng.normal(100, 5, 200), rng.normal(100, 30, 200)


def test_feltz_miller_identical_samples_have_zero_statistic(equal_cv_samples):
    sample, _ = equal_cv_samples
    statistic, p_value = boxplots.feltz_miller_asymptotic_cv_test(sample, sample)
    assert statistic == pytest.approx(0)
    assert p_value == pytest.approx(1)


def test_feltz_miller_detects_different_cvs(equal_cv_samples, different_cv_samples):
    _, p_equal = boxplots.feltz_miller_asymptotic_cv_test(*equal_cv_samples)
    _, p_different = boxplots.feltz_miller_asymptotic_cv_test(*different_cv_samples)
    assert p_equal > 0.05
    assert p_different < 0.001


def test_mslr_detects_different_cvs(equal_cv_samples, different_cv_samples):
    np.random.seed(0)
    _, p_equal = boxplots.mslr_test(*equal_cv_samples, nr=200)
    _, p_different = boxplots.mslr_test(*different_cv_samples, nr=200)
    assert p_equal > 0.05
    assert p_different < 0.001


def test_brown_forsythe_statistic_matches_hand_computation():
    # absolute deviations from the medians are [1, 0, 1] and [2, 0, 2]:
    # between-group SS = 2/3 on 1 df, within-group SS = 10/3 on 4 df
    statistic, _ = boxplots.brown_forsythe_test(
        np.array([1.0, 2.0, 3.0]), np.array([0.0, 2.0, 4.0])
    )
    assert statistic == pytest.approx(0.8)


def test_brown_forsythe_detects_different_spreads_not_different_centers():
    rng = np.random.default_rng(0)
    _, p_shifted = boxplots.brown_forsythe_test(
        rng.normal(0, 1, 200), rng.normal(5, 1, 200)
    )
    _, p_spread = boxplots.brown_forsythe_test(
        rng.normal(0, 1, 200), rng.normal(0, 3, 200)
    )
    assert p_shifted > 0.05
    assert p_spread < 0.001


@pytest.mark.parametrize(
    "plot_function", [boxplots.boxplot, boxplots.violinplot, boxplots.swarmplot]
)
def test_event_plots_draw_one_panel_per_event(conditions_struct, plot_function):
    fig, data = plot_function(
        conditions_struct,
        "body_seg_volume_at_ecdysis",
        [0, 1],
        events_to_plot=[1, 2, 3],
        titles=["M1", "M2", "M3"],
        return_data=True,
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 3
    assert [ax.get_title() for ax in fig.axes] == ["M1", "M2", "M3"]
    assert len(data) == 2 * 30 * 3
    assert set(data["Order"]) == {0, 1, 2}
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [
        "Condition 0",
        "Condition 1",
    ]


@pytest.mark.parametrize(
    "plot_function", [boxplots.boxplot, boxplots.violinplot, boxplots.swarmplot]
)
@pytest.mark.parametrize(
    "test", ["Mann-Whitney", "Feltz-Miller", "MSLR", "Brown-Forsythe"]
)
def test_event_plots_annotate_significance(conditions_struct, plot_function, test):
    np.random.seed(0)
    fig = plot_function(
        conditions_struct,
        "body_seg_length_at_ecdysis",
        [0, 1],
        events_to_plot=[3, 4],
        plot_significance=True,
        significance_test=test,
        show_metric=True,
        share_y_axis=True,
        log_scale=False,
        legend={"description": ""},
        ax_size=(2, 2),
    )
    assert len(fig.axes) == 2
    assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert texts, "expected significance and metric annotations"


@pytest.mark.parametrize(
    "plot_function", [boxplots.boxplot, boxplots.violinplot, boxplots.swarmplot]
)
def test_event_plots_accept_statannotations_tests(conditions_struct, plot_function):
    fig = plot_function(
        conditions_struct,
        "body_seg_length_at_ecdysis",
        [0, 1],
        events_to_plot=[3, 4],
        plot_significance=True,
        significance_test="t-test_welch",
    )
    assert fig.axes[0].texts


def test_unknown_significance_test_raises_value_error(conditions_struct):
    with pytest.raises(ValueError, match="not supported"):
        boxplots.boxplot(
            conditions_struct,
            "body_seg_length_at_ecdysis",
            [0, 1],
            events_to_plot=[3, 4],
            plot_significance=True,
            significance_test="bogus",
        )


@pytest.mark.parametrize(
    "plot_function", [boxplots.boxplot, boxplots.violinplot, boxplots.swarmplot]
)
def test_event_plots_share_y_axis_with_single_event(conditions_struct, plot_function):
    fig = plot_function(
        conditions_struct,
        "body_seg_volume_at_ecdysis",
        [0, 1],
        events_to_plot=[4],
        share_y_axis=True,
    )
    assert len(fig.axes) == 1


@pytest.mark.parametrize(
    "plot_function",
    [boxplots.boxplot_larval_stage, boxplots.violinplot_larval_stage],
)
def test_larval_stage_plots_draw_four_panels(conditions_struct, plot_function):
    fig = plot_function(
        conditions_struct, "body_seg_volume", [0, 1], n_points=10, legend_placement=None
    )
    assert len(fig.axes) == 4
    assert "body_seg_volume_rescaled" in conditions_struct[0]
    assert not fig.legends


@pytest.mark.parametrize(
    "plot_function", [boxplots.boxplot, boxplots.violinplot, boxplots.swarmplot]
)
def test_event_plots_legend_as_xticks(conditions_struct, plot_function):
    fig = plot_function(
        conditions_struct,
        "body_seg_volume_at_ecdysis",
        [1, 0],
        events_to_plot=[1, 2],
        legend_as_xticks=True,
    )
    assert not fig.legends
    for ax in fig.axes:
        assert ax.get_legend() is None
        assert list(ax.get_xticks()) == [0, 1]
        assert [t.get_text() for t in ax.get_xticklabels()] == [
            "Condition 1",
            "Condition 0",
        ]


@pytest.mark.parametrize(
    "plot_function",
    [boxplots.boxplot_larval_stage, boxplots.violinplot_larval_stage],
)
def test_larval_stage_plots_legend_as_xticks(conditions_struct, plot_function):
    fig = plot_function(
        conditions_struct, "body_seg_volume", [0, 1], n_points=10, legend_as_xticks=True
    )
    assert not fig.legends
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == [
        "Condition 0",
        "Condition 1",
    ]


@pytest.mark.parametrize(
    "plot_function", [boxplots.boxplot, boxplots.violinplot, boxplots.swarmplot]
)
def test_display_transform_changes_drawn_values_not_tests(
    conditions_struct, plot_function
):
    # equal spread in log ratios, so Levene only differs on the percent scale
    rng = np.random.default_rng(0)
    for center, condition in zip([0.0, 0.7], conditions_struct):
        condition["log_dev"] = center + rng.normal(0, 0.2, (30, 1))
    log_values = [c["log_dev"][:, 0] for c in conditions_struct]
    percent_values = [proportions.log_ratio_to_percentage(v) for v in log_values]
    assert stats.levene(*log_values).pvalue > 0.05
    assert stats.levene(*percent_values).pvalue < 0.001

    kwargs = dict(
        log_scale=False,
        plot_significance=True,
        significance_test="Levene",
        return_data=True,
    )
    raw_fig, _ = plot_function(conditions_struct, "log_dev", [0, 1], **kwargs)
    fig, data = plot_function(
        conditions_struct,
        "log_dev",
        [0, 1],
        display_transform=proportions.log_ratio_to_percentage,
        **kwargs,
    )

    np.testing.assert_array_equal(data["log_dev"], np.concatenate(log_values))
    swarm = np.concatenate(
        [
            c.get_offsets()[:, 1]
            for c in fig.axes[0].collections
            if isinstance(c, PathCollection)
        ]
    )
    np.testing.assert_allclose(np.sort(swarm), np.sort(np.concatenate(percent_values)))
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert texts == [t.get_text() for t in raw_fig.axes[0].texts]
    (text,) = texts
    # the bracket is placed from the drawn values, above the largest percent
    assert fig.axes[0].texts[0].xy[1] > np.concatenate(percent_values).max()
    shown_p_value = float(text.split("=")[1])
    assert shown_p_value == pytest.approx(stats.levene(*log_values).pvalue, abs=0.01)


@pytest.mark.parametrize(
    "plot_function", [boxplots.boxplot, boxplots.violinplot, boxplots.swarmplot]
)
def test_several_tests_share_one_bracket(conditions_struct, plot_function):
    # same center, five times the spread: only Levene sees a difference
    rng = np.random.default_rng(0)
    for spread, condition in zip([0.1, 0.5], conditions_struct):
        condition["value"] = 1.0 + spread * rng.standard_normal((30, 1))
    values = [c["value"][:, 0] for c in conditions_struct]
    assert stats.mannwhitneyu(*values).pvalue > 0.05
    assert stats.levene(*values).pvalue < 1e-5

    fig = plot_function(
        conditions_struct,
        "value",
        [0, 1],
        log_scale=False,
        plot_significance=True,
        significance_test=["Mann-Whitney", "Levene"],
        show_metric=True,
    )

    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "Mann-Whitney ns\nLevene p ≤ 1e-5" in texts
    metric_texts = [text for text in texts if text.startswith("M = ")]
    for value, expected in zip(values, metric_texts, strict=True):
        median_line, std_line = expected.split("\n")
        assert float(median_line.split("=")[1]) == pytest.approx(
            np.median(value), rel=5e-3
        )
        assert float(std_line.split("=")[1]) == pytest.approx(
            np.std(value, ddof=1), rel=5e-3
        )


def _holm(p_values):
    # the k-th smallest p is scaled by (m - k + 1), kept monotone, capped at 1
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(
        np.maximum.accumulate(p_values[order] * (m - np.arange(m))), 1
    )
    return adjusted


@pytest.mark.parametrize(
    "plot_function", [boxplots.boxplot, boxplots.violinplot, boxplots.swarmplot]
)
@pytest.mark.parametrize("family", [None, "panel", "pair", "figure"])
def test_holm_correction_adjusts_shown_p_values_within_family(plot_function, family):
    rng = np.random.default_rng(14)
    conditions_struct = [
        {"description": f"condition {i}", "value": rng.normal(1, spread, (25, 2))}
        for i, spread in enumerate([0.1, 0.15, 0.2])
    ]
    pairs = [(0, 1), (0, 2), (1, 2)]
    # raw[panel, pair]
    raw = np.array(
        [
            [
                stats.levene(
                    conditions_struct[a]["value"][:, panel],
                    conditions_struct[b]["value"][:, panel],
                ).pvalue
                for a, b in pairs
            ]
            for panel in range(2)
        ]
    )
    assert raw.min() > 0.01, "p-values must be shown as numbers, not thresholds"
    expected = {
        None: raw,
        "panel": np.array([_holm(row) for row in raw]),
        "pair": np.array([_holm(column) for column in raw.T]).T,
        "figure": _holm(raw.ravel()).reshape(raw.shape),
    }[family]

    fig = plot_function(
        conditions_struct,
        "value",
        [0, 1, 2],
        log_scale=False,
        plot_significance=True,
        significance_test="Levene",
        holm_correction=family,
        legend={"description": ""},
    )
    for ax, expected_panel in zip(fig.axes, expected, strict=True):
        shown = [float(t.get_text().split("=")[1]) for t in ax.texts]
        np.testing.assert_allclose(np.sort(shown), np.sort(expected_panel), atol=0.005)


def test_unknown_holm_family_raises_value_error(conditions_struct):
    with pytest.raises(ValueError, match="holm_correction"):
        boxplots.boxplot(
            conditions_struct,
            "body_seg_length_at_ecdysis",
            [0, 1],
            plot_significance=True,
            holm_correction="bogus",
        )


def test_swarmplot_draws_every_worm_in_its_condition_color(conditions_struct):
    fig = boxplots.swarmplot(
        conditions_struct,
        "body_seg_volume_at_ecdysis",
        [0, 1],
        events_to_plot=[2],
        colors=["red", "blue"],
        log_scale=False,
    )
    swarms = [c for c in fig.axes[0].collections if isinstance(c, PathCollection)]
    assert len(swarms) == 2
    for swarm, condition, color in zip(swarms, conditions_struct, ["red", "blue"]):
        np.testing.assert_allclose(
            np.sort(swarm.get_offsets()[:, 1]),
            np.sort(condition["body_seg_volume_at_ecdysis"][:, 2]),
        )
        np.testing.assert_allclose(swarm.get_facecolors()[:, :3], [to_rgb(color)] * 30)


@pytest.fixture
def outlier_struct(conditions_struct):
    """
    Condition 0 worms come from two filemaps (worms 0-14, then 15-29) and are all
    1 except worms 3 and 20 at event 0 (z = 3.68) and worm 7 at event 1 (z = 5.30).
    Condition 1 has no spread, so no z-score.
    """
    for condition in conditions_struct:
        condition["custom"] = np.ones((30, 2))
    conditions_struct[0]["custom"][[3, 20], 0] = 100
    conditions_struct[0]["custom"][7, 1] = 100
    conditions_struct[0]["filemap_path"] = np.array([["a.csv"]] * 15 + [["b.csv"]] * 15)
    return conditions_struct


@pytest.mark.parametrize(
    "threshold, expected",
    [
        (2.0, {"a.csv": ["Point 3", "Point 7"], "b.csv": ["Point 20"]}),
        (4.0, {"a.csv": ["Point 7"]}),
    ],
)
def test_report_outliers_groups_points_by_filemap(
    outlier_struct, capsys, threshold, expected
):
    boxplots.boxplot(
        outlier_struct,
        "custom",
        [0, 1],
        log_scale=False,
        report_outliers=True,
        outlier_threshold=threshold,
    )
    lines = capsys.readouterr().out.splitlines()
    report = lines[lines.index(next(line for line in lines if "std from" in line)) :]
    reported = {}
    for line in report[1:]:
        if line.endswith(".csv"):
            filemap = line
            reported[filemap] = []
        elif line.startswith("  Point"):
            reported[filemap].append(line.split(":")[0].strip())
    assert reported == expected
