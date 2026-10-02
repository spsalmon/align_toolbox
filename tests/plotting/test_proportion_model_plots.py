from dataclasses import replace
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from xarray import Dataset, DataTree

from align_toolbox.data_analysis import proportion_model as pmod
from align_toolbox.plotting import proportion_model_plots as pmp
from align_toolbox.plotting.proportions import log_ratio_to_percentage

# 3 x 2 factorial; condition 1's worms are e^0.5 times bigger than the others'
FACTORS = {
    0: {"yap1": "WT", "tir": "none"},
    1: {"yap1": "abt7", "tir": "none"},
    2: {"yap1": "xyz", "tir": "none"},
    3: {"yap1": "WT", "tir": "col-10"},
    4: {"yap1": "abt7", "tir": "col-10"},
    5: {"yap1": "xyz", "tir": "col-10"},
}
REFERENCE = {"yap1": "WT", "tir": "none"}
SHIFTED_CELL = "yap1=abt7, tir=none"
LOG_X_SHIFT = {1: 0.5}
MOLT_LOG_X = {1: 10.0, 2: 11.0, 3: 12.0, 4: 13.0}
LOG_X_SD = 0.08
N_WORMS = 10
N_DRAWS = 200

A, B = 1.0, 0.8
BETA = np.array([0.10, -0.08, 0.05, 0.06, 0.0])
GAMMA = np.array([0.0, 0.02, 0.0, 0.0, 0.0])
TAU = np.array([0.03, 0.06, 0.03, 0.03, 0.03, 0.03])
SIGMA = np.array([0.02, 0.02, 0.04, 0.02, 0.02, 0.02])
EXPERIMENT_EFFECT = 0.01


def _simulate_table(coding, rng):
    """Every worm at 4 molts, except worm 0 of each condition, which misses molt 2."""
    rows = []
    worm_effects = {}
    for condition_id in FACTORS:
        cell = coding.cell_of_condition(condition_id)
        k = coding.cells.index(cell)
        d = coding.effect_vector(FACTORS[condition_id])
        for worm in range(N_WORMS):
            worm_id = f"c{condition_id}_w{worm}"
            worm_effects[worm_id] = rng.normal(0, TAU[k])
            experiment = "E1" if worm % 2 == 0 else "E2"
            for molt, center in MOLT_LOG_X.items():
                if worm == 0 and molt == 2:
                    continue
                log_x = (
                    center + LOG_X_SHIFT.get(condition_id, 0) + rng.normal(0, LOG_X_SD)
                )
                rows.append(
                    {
                        "condition_id": condition_id,
                        "worm_id": worm_id,
                        "experiment": experiment,
                        "molt": molt,
                        "log_x": log_x,
                        "xc_free_log_y": worm_effects[worm_id]
                        + (
                            EXPERIMENT_EFFECT
                            if experiment == "E1"
                            else -EXPERIMENT_EFFECT
                        )
                        + rng.normal(0, SIGMA[k]),
                        "d": d,
                    }
                )
    table = pd.DataFrame(rows)
    reference = table["condition_id"] == 0
    x_ref = float(np.median(table.loc[reference, "log_x"]))
    xc = table["log_x"] - x_ref
    design = np.stack(table.pop("d").to_numpy())
    table["log_y"] = (
        A
        + B * xc
        + design @ BETA
        + (design * xc.to_numpy()[:, np.newaxis]) @ GAMMA
        + table.pop("xc_free_log_y")
    )
    return table, x_ref, worm_effects


@pytest.fixture(scope="module")
def result():
    """A Student-t, group-variance, spline-reference fit built from synthetic draws."""
    rng = np.random.default_rng(0)
    coding = pmod.build_genotype_coding(list(FACTORS), FACTORS, REFERENCE)
    table, x_ref, worm_effects = _simulate_table(coding, rng)
    reference_log_x = table.loc[table["condition_id"] == 0, "log_x"].to_numpy()
    basis = pmod._build_spline_basis(reference_log_x, x_ref, n_knots=4)
    n_spline = basis.penalty_map.shape[1]
    worms = sorted(worm_effects)
    n_effects = len(coding.effect_names)

    def jitter(value, sd, shape=()):
        return value + rng.normal(0, sd, (N_DRAWS, *shape))

    experiment = jitter(EXPERIMENT_EFFECT, 0.002)
    posterior = {
        "a": ((), jitter(A, 0.005)),
        "b": ((), jitter(B, 0.005)),
        "beta": (("effect",), jitter(BETA, 0.01, (n_effects,))),
        "gamma": (("effect",), jitter(GAMMA, 0.005, (n_effects,))),
        "experiment_effect": (
            ("experiment",),
            np.column_stack([experiment, -experiment]),
        ),
        "tau": (("cell",), np.abs(jitter(TAU, 0.003, (len(TAU),)))),
        "sigma": (("cell",), np.abs(jitter(SIGMA, 0.002, (len(SIGMA),)))),
        "nu": ((), np.abs(jitter(10.0, 1.0))),
        "sd_s": ((), np.abs(jitter(0.0, 0.003))),
        "spline_z": (("spline_basis",), rng.normal(0, 1, (N_DRAWS, n_spline))),
        "worm_effect": (
            ("worm",),
            jitter(np.array([worm_effects[w] for w in worms]), 0.005, (len(worms),)),
        ),
    }
    data = {
        name: (("chain", "draw", *dims), np.asarray(values)[np.newaxis])
        for name, (dims, values) in posterior.items()
    }
    coords = {
        "chain": [0],
        "draw": np.arange(N_DRAWS),
        "effect": coding.effect_names,
        "experiment": ["E1", "E2"],
        "worm": worms,
        "cell": coding.cells,
        "spline_basis": np.arange(n_spline),
    }
    fit = pmod.ProportionModelResult(
        idata=DataTree.from_dict({"posterior": Dataset(data, coords=coords)}),
        table=table,
        coding=coding,
        x_ref=x_ref,
        experiment_labels={"E1": "E1", "E2": "E2"},
        counts=None,
        diagnostics={},
        likelihood="student_t",
        group_variances=True,
        random_seed=0,
        reference_shape_used="spline",
        spline_basis=basis,
        linearity_threshold=0.03,
    )
    fit.linear_misfit = replace(
        fit, spline_basis=None, reference_shape_used="linear"
    ).reference_residuals_by_molt()
    fit.spline_misfit = fit.reference_residuals_by_molt()
    return fit


@pytest.fixture(scope="module")
def tables(result):
    """The expensive model tables, computed once on a few draws."""
    residuals = result.posterior_predictive_residuals(n_draws=20)
    return SimpleNamespace(
        penetrance=result.penetrance(n_draws=20, n_simulations=200),
        residuals=residuals,
        ppc=result.ppc_summary(*residuals),
        dispersion=result.molt_dispersion_tests(),
        experiments={
            "E1": result,
            "E2": replace(result, random_seed=1),
        },
    )


def _calls(result, tables):
    """Each plot with its precomputed tables, and the number of axes it draws on."""
    n_cells = len(result.coding.cells)
    return {
        "scaling": (lambda **kw: pmp.plot_proportion_scaling(result, **kw), 1),
        "offsets": (lambda **kw: pmp.plot_offset_curves(result, **kw), 1),
        "interaction": (lambda **kw: pmp.plot_genotype_interaction(result, **kw), 1),
        "variance": (
            lambda **kw: pmp.plot_variance_components(
                result, show_repeatability=True, **kw
            ),
            2,
        ),
        "consistency": (
            lambda **kw: pmp.plot_individual_consistency(result, 1, 3, **kw),
            n_cells,
        ),
        "penetrance": (
            lambda **kw: pmp.plot_penetrance(
                result, penetrance=tables.penetrance, **kw
            ),
            1,
        ),
        "forest": (lambda **kw: pmp.plot_effects_forest(result, **kw), 1),
        "linearity": (lambda **kw: pmp.plot_linearity_check(result, **kw), 1),
        "ppc": (
            lambda **kw: pmp.plot_posterior_predictive_check(
                result, n_draws=20, residuals=tables.residuals, ppc=tables.ppc, **kw
            ),
            n_cells,
        ),
        "experiments": (
            lambda **kw: pmp.plot_experiment_consistency(
                tables.experiments, pooled=result, **kw
            ),
            1,
        ),
        "dispersion": (
            lambda **kw: pmp.plot_molt_dispersion(
                result, tests=tables.dispersion, show_swarm=False, **kw
            ),
            len(MOLT_LOG_X),
        ),
    }


PLOTS = [
    "scaling",
    "offsets",
    "interaction",
    "variance",
    "consistency",
    "penetrance",
    "forest",
    "linearity",
    "ppc",
    "experiments",
    "dispersion",
]


def _with_gid(ax, prefix):
    """Artists whose gid starts with ``prefix``, without error-bar whiskers and caps."""
    return [
        a
        for a in ax.get_children()
        if (a.get_gid() or "").startswith(prefix)
        and not isinstance(a, LineCollection)
        and not (isinstance(a, Line2D) and a.get_marker() in ("|", "_"))
    ]


def _gid_cells(ax, prefix):
    return {a.get_gid().split("|", 1)[1] for a in _with_gid(ax, prefix)}


@pytest.mark.parametrize("name", PLOTS)
def test_plot_returns_figure_with_expected_panels(result, tables, name):
    plot, n_axes = _calls(result, tables)[name]
    fig = plot()
    assert isinstance(fig, Figure)
    drawn = [ax for ax in fig.axes if ax.axison]
    assert len(drawn) == n_axes


@pytest.mark.parametrize("name", PLOTS)
def test_plot_draws_into_given_axes_without_new_figure(result, tables, name):
    plot, n_axes = _calls(result, tables)[name]
    fig, axes = plt.subplots(1, n_axes, squeeze=False)
    before = plt.get_fignums()
    returned = plot(ax=axes[0] if n_axes > 1 else axes[0, 0])
    assert returned is fig
    assert plt.get_fignums() == before
    assert all(ax.get_children() for ax in axes[0])
    assert plt.fignum_exists(fig.number)


def test_series_counts_follow_the_design(result, tables):
    cells = result.coding.cells
    mutants = cells[1:]

    ax = pmp.plot_proportion_scaling(result, show_points=True).axes[0]
    assert _gid_cells(ax, "molts|") == set(cells)
    assert _gid_cells(ax, "points|") == set(cells)
    for line in _with_gid(ax, "molts|"):
        assert len(line.get_xdata()) == len(MOLT_LOG_X)

    ax = pmp.plot_offset_curves(result).axes[0]
    assert _gid_cells(ax, "offset|") == set(mutants)
    assert _gid_cells(ax, "offset_markers|") == set(mutants)

    ax = pmp.plot_genotype_interaction(result).axes[0]
    assert _gid_cells(ax, "line|") == {"WT", "abt7", "xyz"}
    assert _gid_cells(ax, "offset|") == set(cells)

    fig = pmp.plot_variance_components(result, show_repeatability=True)
    assert _gid_cells(fig.axes[0], "tau_ratio|") == set(mutants)
    assert _gid_cells(fig.axes[0], "sigma_ratio|") == set(mutants)
    assert _gid_cells(fig.axes[1], "repeatability|") == set(cells)
    assert fig.axes[0].get_yscale() == "log"

    ax = pmp.plot_penetrance(result, penetrance=tables.penetrance).axes[0]
    assert _gid_cells(ax, "below|") == _gid_cells(ax, "above|") == set(cells)
    expected = _with_gid(ax, "expected_outside")[0]
    assert expected.get_ydata()[0] == pytest.approx(0.05)

    ax = pmp.plot_effects_forest(result).axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        f"beta[{e}]" for e in result.coding.effect_names
    ]

    ax = pmp.plot_linearity_check(result).axes[0]
    assert _gid_cells(ax, "misfit|") == {"line", "spline"}
    assert ax.get_title() == "Reference shape used: spline"

    fig = pmp.plot_posterior_predictive_check(
        result, n_draws=20, residuals=tables.residuals, ppc=tables.ppc
    )
    for ax, cell in zip(fig.axes, cells):
        assert len(_with_gid(ax, f"replicated|{cell}")) == 20
        assert len(_with_gid(ax, f"observed|{cell}")) == 1

    ax = pmp.plot_experiment_consistency(tables.experiments, pooled=result).axes[0]
    assert _gid_cells(ax, "experiment|") == {"E1", "E2"}
    markers = {a.get_marker() for a in _with_gid(ax, "experiment|")}
    assert len(markers) == 2
    assert len(_with_gid(ax, "pooled|")) == len(result.coding.effect_names)


def test_offset_curves_stay_in_cell_range_and_dash_beyond_reference(result):
    ax = pmp.plot_offset_curves(result, markers=None).axes[0]
    row_cells = result.table["condition_id"].map(result.coding.cell_of_condition)
    reference_log_x = result.table.loc[
        row_cells == result.coding.reference_cell, "log_x"
    ]
    reference_lower, reference_upper = np.exp(reference_log_x.agg(["min", "max"]))

    for cell in result.coding.cells[1:]:
        lines = _with_gid(ax, f"offset|{cell}")
        x = np.concatenate([line.get_xdata() for line in lines])
        cell_log_x = result.table.loc[row_cells == cell, "log_x"]
        assert x.min() == pytest.approx(np.exp(cell_log_x.min()))
        assert x.max() == pytest.approx(np.exp(cell_log_x.max()))
        for line in lines:
            line_x = line.get_xdata()
            inside = (line_x >= reference_lower * (1 - 1e-12)) & (
                line_x <= reference_upper * (1 + 1e-12)
            )
            if line.get_linestyle() == "-":
                assert inside.all()
            else:
                # a dashed segment only touches the reference range at its boundary
                assert (~inside).sum() >= len(line_x) - 1

    # the shifted genotype runs past the reference range and through its gaps
    lines = _with_gid(ax, f"offset|{SHIFTED_CELL}")
    assert any(line.get_linestyle() == "--" for line in lines)
    gap = np.exp(MOLT_LOG_X[1] + LOG_X_SHIFT[1])
    solid_x = np.concatenate(
        [line.get_xdata() for line in lines if line.get_linestyle() == "-"]
    )
    ref_x = np.exp(np.sort(reference_log_x.to_numpy()))
    assert not ((ref_x > gap * 0.99) & (ref_x < gap * 1.01)).any()
    assert ((solid_x > gap * 0.99) & (solid_x < gap * 1.01)).any()


def test_reference_curve_is_dashed_only_where_extrapolated(result):
    ax = pmp.plot_proportion_scaling(result).axes[0]
    row_cells = result.table["condition_id"].map(result.coding.cell_of_condition)
    reference_log_x = result.table.loc[
        row_cells == result.coding.reference_cell, "log_x"
    ]
    lower, upper = np.exp(reference_log_x.agg(["min", "max"]))
    segments = _with_gid(ax, "reference_curve")
    (solid,) = (s for s in segments if s.get_linestyle() == "-")
    dashed = [s for s in segments if s.get_linestyle() == "--"]
    # the shifted genotype reaches past the largest reference worm
    assert dashed
    assert solid.get_xdata().min() == pytest.approx(lower)
    assert solid.get_xdata().max() == pytest.approx(upper)
    for segment in dashed:
        x = segment.get_xdata()
        assert (x <= lower * (1 + 1e-12)).all() or (x >= upper * (1 - 1e-12)).all()


def test_percent_display_matches_model_log_ratios(result):
    ax = pmp.plot_offset_curves(result, markers="model").axes[0]
    for cell in result.coding.cells[1:]:
        for line in _with_gid(ax, f"offset|{cell}"):
            log_x = np.log(line.get_xdata())
            expected = result.group_offset_curve(cell, log_x)["mean"]
            np.testing.assert_allclose(
                line.get_ydata(), log_ratio_to_percentage(expected)
            )
        (markers,) = (
            a for a in _with_gid(ax, f"offset_markers|{cell}") if a.get_marker() == "o"
        )
        positions = result.molt_positions()
        positions = positions[positions["cell"] == cell]
        np.testing.assert_allclose(
            markers.get_ydata(),
            log_ratio_to_percentage(positions["model_offset_mean"]),
        )

    fig = pmp.plot_individual_consistency(result, 1, 3)
    deviations = result.worm_deviations("own")
    cell = result.coding.reference_cell
    worm_cells = deviations["condition_id"].map(result.coding.cell_of_condition)
    complete = deviations.loc[worm_cells == cell, [1, 3]].dropna()
    (points,) = _with_gid(fig.axes[0], f"worms|{cell}")
    np.testing.assert_allclose(
        points.get_offsets(), log_ratio_to_percentage(complete.to_numpy())
    )


def test_individual_consistency_shares_equal_limits_and_counts_dropped_worms(result):
    fig = pmp.plot_individual_consistency(result, 1, 2)
    limits = {(ax.get_xlim(), ax.get_ylim()) for ax in fig.axes if ax.axison}
    assert len(limits) == 1
    ((x_limits, y_limits),) = limits
    assert x_limits == y_limits
    # worm 0 of every condition misses molt 2
    text = _with_gid(fig.axes[0], "annotation|")[0].get_text()
    assert text.startswith(f"n = {N_WORMS - 1} (1 missing a molt)")
    repeatability = result.repeatability().set_index("cell")
    reference = repeatability.loc[result.coding.reference_cell]
    assert f"R = {reference['repeatability_mean']:.2f}" in text


def _color(artist):
    if hasattr(artist, "get_facecolor") and not hasattr(artist, "get_marker"):
        color = artist.get_facecolor()
        return tuple(np.ravel(color)[:3])
    return to_rgba(artist.get_color())[:3]


def test_a_cell_has_the_same_color_in_every_plot(result, tables):
    cell = SHIFTED_CELL
    expected = to_rgba(pmp._cell_styles(result, None, None)[0][cell])[:3]
    reference = to_rgba(pmp.REFERENCE_COLOR)[:3]
    drawn = [
        _with_gid(pmp.plot_proportion_scaling(result).axes[0], f"molts|{cell}")[0],
        _with_gid(pmp.plot_offset_curves(result).axes[0], f"offset|{cell}")[0],
        _with_gid(pmp.plot_genotype_interaction(result).axes[0], "line|abt7")[0],
        _with_gid(pmp.plot_variance_components(result).axes[0], f"tau_ratio|{cell}")[0],
        _with_gid(
            pmp.plot_penetrance(result, penetrance=tables.penetrance).axes[0],
            f"above|{cell}",
        )[0],
    ]
    fig = pmp.plot_individual_consistency(result, 1, 3)
    drawn.append(_with_gid(fig.axes[1], f"worms|{cell}")[0])
    fig = pmp.plot_posterior_predictive_check(
        result, n_draws=20, residuals=tables.residuals, ppc=tables.ppc
    )
    drawn.append(_with_gid(fig.axes[1], f"observed|{cell}")[0])
    for artist in drawn:
        np.testing.assert_allclose(_color(artist), expected)

    fig = pmp.plot_molt_dispersion(result, tests=tables.dispersion, show_swarm=False)
    legend = fig.legends[0]
    colors = {
        text.get_text(): handle.get_facecolor()[:3]
        for text, handle in zip(legend.get_texts(), legend.legend_handles)
    }
    np.testing.assert_allclose(colors[cell], expected)
    np.testing.assert_allclose(colors[result.coding.reference_cell], reference)

    ax = pmp.plot_proportion_scaling(result).axes[0]
    reference_line = _with_gid(ax, f"molts|{result.coding.reference_cell}")[0]
    np.testing.assert_allclose(_color(reference_line), reference)


@pytest.mark.parametrize("key", [1, SHIFTED_CELL])
def test_custom_colors_and_labels_by_condition_or_cell(result, key):
    fig = pmp.plot_offset_curves(
        result, colors={key: "#ff0000"}, labels={key: "big worms"}
    )
    ax = fig.axes[0]
    line = _with_gid(ax, f"offset|{SHIFTED_CELL}")[0]
    assert to_rgba(line.get_color()) == to_rgba("#ff0000")
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "big worms" in legend_texts
    # the other cells keep their default colors
    other = result.coding.cells[2]
    default = pmp._cell_styles(result, None, None)[0][other]
    other_line = _with_gid(ax, f"offset|{other}")[0]
    assert to_rgba(other_line.get_color()) == to_rgba(default)


def test_custom_colors_reject_unknown_cells_and_wrong_lengths(result):
    with pytest.raises(ValueError, match="neither a condition id"):
        pmp.plot_offset_curves(result, colors={99: "red"})
    with pytest.raises(ValueError, match="entries"):
        pmp.plot_offset_curves(result, colors=["red", "blue"])


def test_interaction_plot_requires_two_factors(result):
    single = pmod.build_genotype_coding(list(FACTORS), reference_condition=0)
    with pytest.raises(ValueError, match="two-factor"):
        pmp.plot_genotype_interaction(replace(result, coding=single))


def test_interaction_plot_marks_extrapolated_cells_and_annotates(result):
    # at the reference's smallest molt the shifted genotype has no data
    log_x = MOLT_LOG_X[1] - 0.2
    ax = pmp.plot_genotype_interaction(
        result, log_x=log_x, annotate_interaction=True
    ).axes[0]
    offsets = result.cell_offsets(log_x=log_x).set_index("cell")
    for artist in _with_gid(ax, "offset|"):
        if artist.get_marker() in ("o", "D"):
            cell = artist.get_gid().split("|", 1)[1]
            within = offsets.loc[cell, "within_cell_range"]
            assert artist.get_marker() == ("o" if within else "D")
    assert not offsets.loc[SHIFTED_CELL, "within_cell_range"]
    interactions = offsets.dropna(subset=["interaction_mean_percent"])
    annotated = _gid_cells(ax, "interaction|")
    assert annotated == set(interactions.index)
    assert f"{np.exp(log_x):.3g}" in ax.get_title()


def test_variance_plot_requires_group_variances(result):
    with pytest.raises(ValueError, match="group_variances"):
        pmp.plot_variance_components(replace(result, group_variances=False))


def test_linearity_plot_requires_a_misfit_table(result):
    with pytest.raises(ValueError, match="misfit"):
        pmp.plot_linearity_check(
            replace(result, linear_misfit=None, spline_misfit=None)
        )


def test_linearity_threshold_band_matches_the_auto_rule(result):
    ax = pmp.plot_linearity_check(result, threshold=0.05).axes[0]
    (band,) = _with_gid(ax, "threshold")
    assert band.get_y() + band.get_height() == pytest.approx(5.0)
    assert band.get_y() == pytest.approx(100 * (1 / 1.05 - 1))


def test_forest_uses_percent_for_beta_and_raw_values_otherwise(result):
    summary = result.summary()
    ratios = summary[summary["parameter"].str.contains("ratio")]
    ax = pmp.plot_effects_forest(result, parameters=r"_ratio\[").axes[0]
    (points,) = _with_gid(ax, "effects")
    np.testing.assert_allclose(np.sort(points.get_xdata()), np.sort(ratios["mean"]))
    assert ax.lines[0].get_xdata()[0] == 1

    names = ["beta[yap1[abt7]]", "gamma[yap1[abt7]]"]
    ax = pmp.plot_effects_forest(result, parameters=names).axes[0]
    (points,) = _with_gid(ax, "effects")
    rows = summary.set_index("parameter").loc[names]
    np.testing.assert_allclose(
        points.get_xdata(), [rows["mean_percent"].iloc[0], rows["mean"].iloc[1]]
    )
    assert "% for beta, raw otherwise" in ax.get_xlabel()


def test_molt_dispersion_brackets_show_holm_adjusted_p(result, tables):
    fig = pmp.plot_molt_dispersion(result, tests=tables.dispersion, show_swarm=False)
    molts = sorted(MOLT_LOG_X)
    for ax, molt in zip(fig.axes, molts):
        tested = tables.dispersion[
            (tables.dispersion["molt"] == molt) & tables.dispersion["p_holm"].notna()
        ]
        brackets = sorted(
            t.get_text() for t in ax.texts if t.get_text().startswith("p =")
        )
        assert brackets == sorted(f"p = {p:.2g}" for p in tested["p_holm"])
