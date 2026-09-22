import matplotlib.figure
import numpy as np
import pytest

from align_toolbox.plotting import boxplots


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


@pytest.mark.parametrize("plot_function", [boxplots.boxplot, boxplots.violinplot])
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


@pytest.mark.parametrize("plot_function", [boxplots.boxplot, boxplots.violinplot])
@pytest.mark.parametrize("test", ["Mann-Whitney", "Feltz-Miller", "MSLR"])
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


@pytest.mark.parametrize("plot_function", [boxplots.boxplot, boxplots.violinplot])
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


@pytest.mark.parametrize("plot_function", [boxplots.boxplot, boxplots.violinplot])
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
