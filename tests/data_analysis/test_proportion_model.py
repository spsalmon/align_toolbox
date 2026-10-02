import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import statsmodels.formula.api as smf
from scipy.stats import expon
from xarray import Dataset, DataTree

from align_toolbox.data_analysis import proportion_model as pmod
from tests.data_analysis.proportion_simulation import (
    ALLOMETRY_CELLS,
    ALLOMETRY_LOW_GAP_CELLS,
    CELLS,
    EXPERIMENTS,
    X0,
    simulate_allometry,
    simulate_proportions,
)

COLUMNS = ("volume_at_ecdysis", "pharynx_volume_at_ecdysis")
ALLOMETRY_COLUMNS = ("length_at_ecdysis", "volume_at_ecdysis")
FACTORS_2X2 = {
    0: {"yap1": "WT", "tir": "none"},
    1: {"yap1": "abt7", "tir": "none"},
    2: {"yap1": "WT", "tir": "col-10"},
    3: {"yap1": "abt7", "tir": "col-10"},
}
REFERENCE_2X2 = {"yap1": "WT", "tir": "none"}


def _condition(condition_id, experiments, points, x, y):
    return {
        "condition_id": condition_id,
        "experiment": np.array(experiments)[:, np.newaxis],
        "point": np.array(points)[:, np.newaxis],
        "x": np.array(x, dtype=float),
        "y": np.array(y, dtype=float),
    }


# DATA PREPARATION


def test_same_point_in_two_experiments_gives_distinct_worms():
    condition = _condition(
        0, ["/data/exp_b", "/data/exp_a"], [3, 3], [[1, 2, 3]] * 2, [[1, 2, 3]] * 2
    )
    table = pmod.build_proportion_table([condition], "x", "y", [0])
    assert table["worm_id"].nunique() == 2
    # labels follow sorted paths, whatever the order the worms come in
    assert table.attrs["experiment_labels"] == {
        "E1": "/data/exp_a",
        "E2": "/data/exp_b",
    }
    assert list(table.drop_duplicates("worm_id")["experiment"]) == ["E2", "E1"]


def test_nan_and_non_positive_values_are_dropped_and_counted():
    x = [[1, 2, 4], [1, np.nan, 4], [1, 2, 0], [1, 2, 4]]
    y = [[1, 3, 9], [1, 3, 9], [1, 3, 9], [1, -3, np.nan]]
    condition = _condition(7, ["/e"] * 4, [0, 1, 2, 3], x, y)
    table = pmod.build_proportion_table([condition], "x", "y", [7])
    counts = table.attrs["counts"].set_index("molt")
    assert list(counts.index) == [1, 2]
    assert counts.loc[1, ["n_worms", "n_nan", "n_non_positive", "n_kept"]].tolist() == [
        4,
        1,
        1,
        2,
    ]
    assert counts.loc[2, ["n_nan", "n_non_positive", "n_kept"]].tolist() == [1, 1, 2]
    assert table.loc[table["molt"] == 1, "worm_id"].tolist() == [
        "c7_E1_p0",
        "c7_E1_p2",
    ]
    np.testing.assert_allclose(
        table.loc[table["molt"] == 2, "log_x"], np.log([4.0, 4.0])
    )


def test_remove_hatch_drops_event_zero():
    condition = _condition(0, ["/e"], [0], [[1, 2, 3]], [[1, 2, 3]])
    without_hatch = pmod.build_proportion_table([condition], "x", "y", [0])
    with_hatch = pmod.build_proportion_table(
        [condition], "x", "y", [0], remove_hatch=False
    )
    assert sorted(without_hatch["molt"]) == [1, 2]
    assert sorted(with_hatch["molt"]) == [0, 1, 2]


def test_requested_condition_must_exist():
    condition = _condition(0, ["/e"], [0], [[1, 2]], [[1, 2]])
    with pytest.raises(ValueError, match=r"\[5\]"):
        pmod.build_proportion_table([condition], "x", "y", [0, 5])


# GENOTYPE CODING


def test_factorial_design_has_main_effects_and_interactions():
    factors = {
        i: {"yap1": yap1, "tir": tir}
        for i, (yap1, tir) in enumerate(
            (y, t) for y in ("WT", "abt7", "abt8") for t in ("none", "col-10")
        )
    }
    coding = pmod.build_genotype_coding(
        list(factors), factors, {"yap1": "WT", "tir": "none"}
    )
    design = coding.design_matrix(list(factors))
    assert list(design.columns) == [
        "yap1[abt7]",
        "yap1[abt8]",
        "tir[col-10]",
        "yap1[abt7]:tir[col-10]",
        "yap1[abt8]:tir[col-10]",
    ]
    # rows: (WT,none) (WT,col-10) (abt7,none) (abt7,col-10) (abt8,none) (abt8,col-10)
    np.testing.assert_array_equal(
        design.to_numpy(),
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 0, 1],
        ],
    )
    assert coding.reference_cell == "yap1=WT, tir=none"


def test_single_factor_coding_uses_reference_condition():
    coding = pmod.build_genotype_coding([2, 0, 5], reference_condition=2)
    assert coding.effect_names == ["condition[0]", "condition[5]"]
    assert coding.reference_cell == "condition=2"
    np.testing.assert_array_equal(coding.design_matrix([2]).to_numpy(), [[0, 0]])


def test_uncoded_condition_raises():
    with pytest.raises(ValueError, match=r"\[9\]"):
        pmod.build_genotype_coding([0, 1, 9], FACTORS_2X2, REFERENCE_2X2)


def test_missing_reference_cell_raises():
    with pytest.raises(ValueError, match="reference cell"):
        pmod.build_genotype_coding([1, 2, 3], FACTORS_2X2, REFERENCE_2X2)


# RESULT METHODS ON A HAND-BUILT POSTERIOR


def _result(
    table,
    coding,
    posterior,
    x_ref,
    group_variances=False,
    likelihood="normal",
    spline_basis=None,
):
    """Wrap fixed posterior draws, given as ``{name: (dims, values)}``, in a result."""
    n_draws = len(posterior["a"][1])
    data = {
        name: (("chain", "draw", *dims), np.asarray(values)[np.newaxis])
        for name, (dims, values) in posterior.items()
    }
    coords = {
        "chain": [0],
        "draw": np.arange(n_draws),
        "effect": coding.effect_names,
        "experiment": sorted(table["experiment"].unique()),
        # deliberately not in table order
        "worm": sorted(table["worm_id"].unique(), reverse=True),
        "cell": coding.cells,
    }
    if spline_basis is not None:
        coords["spline_basis"] = np.arange(spline_basis.penalty_map.shape[1])
    idata = DataTree.from_dict({"posterior": Dataset(data, coords=coords)})
    return pmod.ProportionModelResult(
        idata=idata,
        table=table,
        coding=coding,
        x_ref=x_ref,
        experiment_labels={},
        counts=None,
        diagnostics={},
        likelihood=likelihood,
        group_variances=group_variances,
        random_seed=0,
        reference_shape_used="linear" if spline_basis is None else "spline",
        spline_basis=spline_basis,
    )


def _reference_table(log_x_by_molt, bump_by_molt=None, condition_id=0):
    """Two worms per molt in two experiments, exactly on ``log_y = 1 + 0.5 * log_x``."""
    rows = []
    for molt, values in log_x_by_molt.items():
        for i, log_x in enumerate(values):
            rows.append(
                {
                    "condition_id": condition_id,
                    "worm_id": f"w{i}",
                    "experiment": f"E{i % 2 + 1}",
                    "molt": molt,
                    "log_x": log_x,
                    "log_y": 1 + 0.5 * log_x + (bump_by_molt or {}).get(molt, 0.0),
                }
            )
    return pd.DataFrame(rows)


def test_group_offset_curve_combines_main_effects_and_interaction():
    coding = pmod.build_genotype_coding(list(FACTORS_2X2), FACTORS_2X2, REFERENCE_2X2)
    table = _reference_table({1: [1.0, 2.0]})
    beta = np.array([0.10, 0.20, -0.05])
    gamma = np.array([0.01, 0.02, 0.03])
    # two draws spread symmetrically around beta/gamma
    result = _result(
        table,
        coding,
        {
            "a": ((), [1.0, 1.0]),
            "b": ((), [0.5, 0.5]),
            "beta": (("effect",), [beta - 0.01, beta + 0.01]),
            "gamma": (("effect",), [gamma, gamma]),
        },
        x_ref=2.0,
    )
    grid = np.array([1.0, 2.0, 4.0])
    double_mutant = result.group_offset_curve({"yap1": "abt7", "tir": "col-10"}, grid)
    np.testing.assert_allclose(double_mutant["mean"], 0.25 + 0.06 * (grid - 2.0))
    assert (double_mutant["lower"] < double_mutant["mean"]).all()
    single_mutant = result.group_offset_curve(1, grid)
    np.testing.assert_allclose(single_mutant["mean"], 0.10 + 0.01 * (grid - 2.0))
    np.testing.assert_allclose(
        result.group_offset_curve("yap1=WT, tir=none", grid)["mean"], 0
    )
    with pytest.raises(ValueError, match="fitted cells"):
        result.group_offset_curve({"yap1": "abt9", "tir": "none"}, grid)


def test_support_flags_label_gaps_and_out_of_range_values():
    coding = pmod.build_genotype_coding([0, 1], reference_condition=0)
    reference = _reference_table({1: [1.0, 1.2], 2: [2.0, 2.2]})
    other = pd.DataFrame(
        {
            "condition_id": 1,
            "worm_id": ["m0", "m1", "m2", "m3"],
            "experiment": "E1",
            "molt": [1, 1, 2, 2],
            "log_x": [1.1, 1.6, 2.5, 0.5],
            "log_y": 0.0,
        }
    )
    table = pd.concat([reference, other], ignore_index=True)
    result = _result(table, coding, {"a": ((), [1.0])}, x_ref=1.5)
    flags = result.support_flags()
    assert flags.tolist() == ["in"] * 4 + ["in", "gap", "out", "out"]


def test_reference_residuals_by_molt_recover_curvature():
    coding = pmod.build_genotype_coding([0], reference_condition=0)
    bumps = {1: 0.0, 2: np.log(1.05), 3: -np.log(1.02)}
    table = _reference_table(
        {1: [1.0, 1.1], 2: [2.0, 2.1], 3: [3.0, 3.1]}, bump_by_molt=bumps
    )
    # a, b and the experiment effects reproduce the line exactly; the worm
    # effects of the posterior are left out of the marginal residuals
    experiment_effect = [0.03, -0.03]
    worm_effect = {"w0": 0.02, "w1": -0.01}
    table["log_y"] += table["experiment"].map({"E1": 0.03, "E2": -0.03}) - 0.5 * 2.0
    worms = sorted(worm_effect, reverse=True)
    result = _result(
        table,
        coding,
        {
            "a": ((), [1.0]),
            "b": ((), [0.5]),
            "experiment_effect": (("experiment",), [experiment_effect]),
            "worm_effect": (("worm",), [[worm_effect[w] for w in worms]]),
        },
        x_ref=2.0,
    )
    residuals = result.reference_residuals_by_molt()
    np.testing.assert_allclose(residuals["mean_percent"], [0.0, 5.0, -100 / 51])
    assert residuals["n_observations"].tolist() == [2, 2, 2]
    effects = result.worm_effects().set_index("worm_id")
    assert effects.loc["w0", "mean"] == pytest.approx(0.02)
    assert effects.loc["w1", "mean"] == pytest.approx(-0.01)


def test_reference_residuals_do_not_depend_on_worm_effects():
    coding = pmod.build_genotype_coding([0], reference_condition=0)
    # molt 2 has only worm w0, so its intercept could absorb the molt-2 misfit
    table = _reference_table({1: [1.0, 1.1], 2: [2.0]}, bump_by_molt={2: 0.04})
    worm_effect = np.array([[0.01, 0.04], [-0.02, 0.03]])
    posterior = {
        "a": ((), [1.0, 1.1]),
        "b": ((), [0.5, 0.5]),
        "worm_effect": (("worm",), worm_effect),
    }
    before = _result(table, coding, posterior, x_ref=0.0)
    shifted = dict(posterior, worm_effect=(("worm",), worm_effect + 0.5))
    after = _result(table, coding, shifted, x_ref=0.0)
    pd.testing.assert_frame_equal(
        before.reference_residuals_by_molt(), after.reference_residuals_by_molt()
    )
    # a = 1.05 on average, so the residuals are -0.05 and -0.05 + 0.04
    np.testing.assert_allclose(
        before.reference_residuals_by_molt()["mean_percent"],
        [100 * np.expm1(-0.05), 100 * np.expm1(-0.01)],
    )


def test_residuals_use_each_genotype_curve_without_worm_effects():
    coding = pmod.build_genotype_coding([0, 1], reference_condition=0)
    worm_offset = {"w0": 0.02, "w1": -0.01, "m0": 0.03, "m1": -0.04}
    reference = _reference_table({1: [1.0, 1.1], 2: [2.0, 2.1]})
    mutant = reference.assign(
        condition_id=1, worm_id=reference["worm_id"].str.replace("w", "m")
    )
    # mutant curve: beta = 0.1 and gamma = 0.2 around x_ref = 1.5
    mutant["log_y"] += 0.1 + 0.2 * (mutant["log_x"] - 1.5)
    table = pd.concat([reference, mutant], ignore_index=True)
    table["log_y"] += table["experiment"].map({"E1": 0.03, "E2": -0.03}) - 0.5 * 1.5
    table["log_y"] += table["worm_id"].map(worm_offset)
    worms = sorted(worm_offset, reverse=True)
    result = _result(
        table,
        coding,
        {
            "a": ((), [1.0]),
            "b": ((), [0.5]),
            "beta": (("effect",), [[0.1]]),
            "gamma": (("effect",), [[0.2]]),
            "experiment_effect": (("experiment",), [[0.03, -0.03]]),
            "worm_effect": (("worm",), [[worm_offset[w] for w in worms]]),
        },
        x_ref=1.5,
    )
    residuals = result.residuals()
    assert list(residuals.columns) == [
        "condition_id",
        "worm_id",
        "molt",
        "experiment",
        "residual",
    ]
    np.testing.assert_allclose(
        residuals["residual"], table["worm_id"].map(worm_offset), atol=1e-12
    )


def test_reference_predictive_interval_matches_residual_spread():
    coding = pmod.build_genotype_coding([0], reference_condition=0)
    table = _reference_table({1: [1.0, 2.0]})
    n_draws = 40_000
    result = _result(
        table,
        coding,
        {
            "a": ((), np.full(n_draws, 1.0)),
            "b": ((), np.full(n_draws, 0.5)),
            "tau": ((), np.full(n_draws, 0.03)),
            "sigma": ((), np.full(n_draws, 0.04)),
        },
        x_ref=1.5,
    )
    grid = np.array([1.0, 3.0])
    interval = result.reference_predictive_interval(grid)
    line = 1.0 + 0.5 * (grid - 1.5)
    np.testing.assert_allclose(interval["mean"], line)
    # worm and residual variances add: sd = 0.05
    np.testing.assert_allclose(interval["upper"] - line, 1.96 * 0.05, atol=0.002)
    np.testing.assert_allclose(line - interval["lower"], 1.96 * 0.05, atol=0.002)


def test_summary_converts_beta_to_percent_and_reports_variance_ratios():
    coding = pmod.build_genotype_coding([0, 1], reference_condition=0)
    table = _reference_table({1: [1.0, 2.0]})
    result = _result(
        table,
        coding,
        {
            "a": ((), [1.0, 1.0]),
            "b": ((), [0.5, 0.5]),
            "beta": (("effect",), [[np.log(1.1)], [np.log(1.1)]]),
            "gamma": (("effect",), [[0.0], [0.0]]),
            "tau": (("cell",), [[0.02, 0.03], [0.04, 0.06]]),
            "sigma": (("cell",), [[0.02, 0.02], [0.04, 0.08]]),
        },
        x_ref=1.5,
        group_variances=True,
    )
    summary = result.summary().set_index("parameter")
    assert summary.loc["beta[condition[1]]", "mean_percent"] == pytest.approx(10.0)
    assert np.isnan(summary.loc["gamma[condition[1]]", "mean_percent"])
    assert summary.loc["tau_ratio[condition=1]", "mean"] == pytest.approx(1.5)
    # ratios are taken per draw: (1 + 2) / 2, not 0.05 / 0.03
    assert summary.loc["sigma_ratio[condition=1]", "mean"] == pytest.approx(1.5)
    assert summary.loc["sigma[condition=1]", "mean"] == pytest.approx(0.05)


def test_summary_reports_student_t_standard_deviation():
    coding = pmod.build_genotype_coding([0], reference_condition=0)
    table = _reference_table({1: [1.0, 2.0]})
    posterior = {
        "a": ((), [1.0, 1.0, 1.0]),
        "b": ((), [0.5, 0.5, 0.5]),
        "tau": ((), [0.1, 0.1, 0.1]),
        "sigma": ((), [0.1, 0.2, 0.3]),
    }
    normal = _result(table, coding, posterior, x_ref=1.5).summary()
    assert not normal["parameter"].str.startswith("sigma_sd").any()

    posterior["nu"] = ((), [1.5, 4.0, 6.0])
    student_t = (
        _result(table, coding, posterior, x_ref=1.5, likelihood="student_t")
        .summary()
        .set_index("parameter")
    )
    # sd = sigma * sqrt(nu / (nu - 2)); undefined for the draw with nu <= 2
    expected = [0.2 * np.sqrt(2), 0.3 * np.sqrt(1.5)]
    assert student_t.loc["sigma_sd", "mean"] == pytest.approx(np.mean(expected))
    assert student_t.loc["sigma_sd_undefined_fraction", "mean"] == pytest.approx(1 / 3)
    assert student_t.loc["sigma", "mean"] == pytest.approx(0.2)
    assert student_t.attrs["reference_shape"] == "linear"


# REFERENCE SPLINE


def _spline_reference_table():
    return _reference_table(
        {
            molt: center + np.linspace(-0.1, 0.1, 5)
            for molt, center in enumerate([1.0, 1.4, 1.8, 2.2], 1)
        }
    )


def test_spline_basis_is_pure_curvature_inside_the_reference_range():
    table = _spline_reference_table()
    log_x = table["log_x"].to_numpy()
    basis = pmod._build_spline_basis(log_x, x_ref=1.6, n_knots=10)
    columns = basis.design(log_x)
    linear = np.column_stack([np.ones_like(log_x), log_x - 1.6])
    np.testing.assert_allclose(linear.T @ columns, 0, atol=1e-10)
    # with z ~ Normal(0, 1), E[mean(s²)] over the reference rows is sd_s²
    assert np.sum(columns**2) / len(log_x) == pytest.approx(1.0)
    # rotating keeps the prior covariance Z Zᵀ of s
    rotated = pmod._rotate_spline_basis(basis, log_x, linear)
    np.testing.assert_allclose(
        rotated.design(log_x) @ rotated.design(log_x).T, columns @ columns.T, atol=1e-10
    )


def test_spline_reference_curve_extrapolates_linearly_at_the_boundary_slope():
    coding = pmod.build_genotype_coding([0], reference_condition=0)
    table = _spline_reference_table()
    basis = pmod._build_spline_basis(table["log_x"], x_ref=1.6, n_knots=10)
    rng = np.random.default_rng(0)
    n_draws, n_columns = 3, basis.penalty_map.shape[1]
    result = _result(
        table,
        coding,
        {
            "a": ((), [1.0, 1.1, 0.9]),
            "b": ((), [0.5, 0.6, 0.4]),
            "sd_s": ((), [0.05, 0.1, 0.2]),
            "spline_z": (("spline_basis",), rng.standard_normal((n_draws, n_columns))),
        },
        x_ref=1.6,
        spline_basis=basis,
    )
    step, h = 0.1, 1e-6
    for boundary, direction in ((basis.upper, 1), (basis.lower, -1)):
        outside = boundary + direction * step * np.arange(1, 6)
        curve = result.reference_curve(outside)
        assert curve["extrapolated"].all()
        np.testing.assert_allclose(np.diff(curve["mean"], n=2), 0, atol=1e-10)
        # one-sided derivative from inside the range
        inside = result.reference_curve([boundary - direction * h, boundary])["mean"]
        boundary_slope = direction * (inside[1] - inside[0]) / h
        slope = np.diff(curve["mean"]) / (direction * step)
        np.testing.assert_allclose(slope, boundary_slope, rtol=1e-4)
    curve = result.reference_curve([basis.lower, 1.6, basis.upper])
    assert not curve["extrapolated"].any()
    # the spline is not a line inside the range
    grid = np.linspace(basis.lower, basis.upper, 7)
    assert np.abs(np.diff(result.reference_curve(grid)["mean"], n=2)).max() > 1e-3


# constant fake draws leave R-hat and ESS undefined
@pytest.mark.filterwarnings("ignore")
def test_reference_curve_matches_the_fitted_curve_of_the_model(monkeypatch):
    # one genotype, one experiment and zero worm effects: the model mean is f
    table = pmod.build_proportion_table(
        simulate_allometry(0, cells={0: ALLOMETRY_CELLS[0]}),
        *ALLOMETRY_COLUMNS,
        [0],
    )
    table = table[table["experiment"] == "E1"].reset_index(drop=True)
    model_mean = {}

    def fake_sample(draws, chains, **kwargs):
        model = pm.modelcontext(None)
        rng = np.random.default_rng(1)
        point = {
            name: value + rng.normal(0, 0.5, np.shape(value))
            for name, value in model.initial_point().items()
        }
        point["worm_z"] = np.zeros_like(point["worm_z"])
        observed = model["log_y"]
        mu = model.replace_rvs_by_values(
            [observed.owner.op.dist_params(observed.owner)[0]]
        )
        mean = model.compile_fn(
            mu[0], inputs=model.value_vars, on_unused_input="ignore"
        )
        model_mean["value"] = np.broadcast_to(mean(point), (len(table),))
        values = model.compile_fn(
            model.unobserved_value_vars,
            inputs=model.value_vars,
            on_unused_input="ignore",
        )(point)
        posterior = {}
        for variable, value in zip(model.unobserved_value_vars, values):
            # transformed value variables such as ``sd_s_log__`` sit beside
            # their constrained twin
            name = variable.name
            if name.endswith("__"):
                continue
            dims = model.named_vars_to_dims.get(name, ())
            posterior[name] = (
                ("chain", "draw", *dims),
                np.broadcast_to(value, (chains, draws, *np.shape(value))),
            )
        return DataTree.from_dict(
            {
                "posterior": Dataset(
                    posterior,
                    coords={dim: list(values) for dim, values in model.coords.items()},
                ),
                "sample_stats": Dataset(
                    {"diverging": (("chain", "draw"), np.zeros((chains, draws), bool))}
                ),
            }
        )

    monkeypatch.setattr(pmod.pm, "sample", fake_sample)
    result = pmod.fit_proportion_model(
        table,
        reference_condition=0,
        reference_shape="spline",
        draws=2,
        chains=2,
        random_seed=0,
    )
    assert result.spline_basis.rotation is not None
    curve = result.reference_curve(table["log_x"].to_numpy())
    np.testing.assert_allclose(curve["mean"], model_mean["value"], rtol=1e-10)
    # the spline term is active, so f is not a line
    log_x = table["log_x"].to_numpy()
    line = np.polyval(np.polyfit(log_x, curve["mean"], 1), log_x)
    assert np.abs(curve["mean"] - line).max() > 1e-3


def test_smoothness_prior_rate_puts_the_requested_mass_above_upper():
    rate = pmod._smoothness_prior_rate(0.05, 0.05)
    assert expon(scale=1 / rate).sf(0.05) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="between 0 and 1"):
        pmod._smoothness_prior_rate(0.05, 1.0)


def _diagnostics_data(chain_means, n_divergences):
    rng = np.random.default_rng(0)
    draws = rng.standard_normal((2, 1000)) + np.array(chain_means)[:, np.newaxis]
    diverging = np.zeros((2, 1000), dtype=bool)
    diverging[0, :n_divergences] = True
    return DataTree.from_dict(
        {
            "posterior": Dataset(
                {
                    "a": (("chain", "draw"), draws),
                    "z": (("chain", "draw", "worm"), np.stack([draws, -draws], -1)),
                }
            ),
            "sample_stats": Dataset({"diverging": (("chain", "draw"), diverging)}),
        }
    )


def test_diagnostics_pass_for_independent_well_mixed_chains():
    diagnostics = pmod._compute_diagnostics(_diagnostics_data([0, 0], 0), ["a", "z"])
    assert diagnostics["max_rhat"] < pmod.MAX_RHAT
    assert diagnostics["min_ess_bulk"] > 1500
    assert diagnostics["n_divergences"] == 0


def test_diagnostics_flag_chains_that_disagree():
    diagnostics = pmod._compute_diagnostics(_diagnostics_data([0, 3], 4), ["a", "z"])
    assert diagnostics["max_rhat"] > 1.5
    assert diagnostics["min_ess_bulk"] < pmod.MIN_ESS
    assert diagnostics["n_divergences"] == 4


# SAMPLING


def _table(seed):
    return pmod.build_proportion_table(
        simulate_proportions(seed), *COLUMNS, list(CELLS)
    )


@pytest.fixture(scope="module")
def recovery_fit(simulated_proportions):
    table = pmod.build_proportion_table(simulated_proportions, *COLUMNS, list(CELLS))
    # the default chains/draws: 2 x 500 draws leaves the bulk ESS of `a` below 400
    return pmod.fit_proportion_model(
        table,
        reference_condition=0,
        likelihood="student_t",
        group_variances=True,
        x_ref=X0,
        random_seed=3,
    )


def _interval(summary, parameter):
    row = summary.loc[parameter]
    return row["lower"], row["upper"]


@pytest.mark.slow
def test_fit_recovers_genotype_and_experiment_effects(recovery_fit):
    summary = recovery_fit.summary().set_index("parameter")
    for condition_id, cell in CELLS.items():
        if condition_id == 0:
            continue
        for name in ("beta", "gamma"):
            lower, upper = _interval(summary, f"{name}[condition[{condition_id}]]")
            assert lower < cell[name] < upper, (name, condition_id)
    for label, offset in zip(("E1", "E2"), EXPERIMENTS.values()):
        lower, upper = _interval(summary, f"experiment_effect[{label}]")
        assert lower < offset < upper


@pytest.mark.slow
def test_fit_recovers_variance_ratios(recovery_fit):
    summary = recovery_fit.summary().set_index("parameter")
    for condition_id in (1, 2, 3):
        cell = CELLS[condition_id]
        true_ratio = cell["tau"] / CELLS[0]["tau"]
        lower, upper = _interval(summary, f"tau_ratio[condition={condition_id}]")
        assert lower < true_ratio < upper, condition_id
    for condition_id in (1, 2):
        lower, upper = _interval(summary, f"sigma_ratio[condition={condition_id}]")
        assert lower < 1 < upper, condition_id


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="LIMITATION: nu is shared by all cells, so the normal cells pull it "
    "up (~20) and the t(3) tails of cell 3 inflate its sigma ratio (~2.6 vs 2)",
)
def test_fit_recovers_sigma_ratio_of_heavy_tailed_cell(recovery_fit):
    summary = recovery_fit.summary().set_index("parameter")
    lower, upper = _interval(summary, "sigma_ratio[condition=3]")
    assert lower < CELLS[3]["sigma"] / CELLS[0]["sigma"] < upper


@pytest.mark.slow
def test_fit_diagnostics_pass(recovery_fit):
    diagnostics = recovery_fit.diagnostics
    assert diagnostics["max_rhat"] <= pmod.MAX_RHAT
    assert diagnostics["min_ess_bulk"] >= pmod.MIN_ESS
    assert diagnostics["min_ess_tail"] >= pmod.MIN_ESS
    assert diagnostics["n_divergences"] == 0


@pytest.mark.slow
def test_normal_shared_variance_fit_matches_mixed_model(simulated_proportions):
    table = pmod.build_proportion_table(simulated_proportions, *COLUMNS, list(CELLS))
    result = pmod.fit_proportion_model(
        table, reference_condition=0, draws=500, chains=2, random_seed=3
    )
    data = table.assign(x_c=table["log_x"] - result.x_ref)
    mixed = smf.mixedlm(
        "log_y ~ x_c * C(condition_id) + C(experiment)",
        data,
        groups=data["worm_id"],
    ).fit(reml=False)
    fixed = mixed.fe_params
    posterior = result.summary().set_index("parameter")["mean"]

    # sum-to-zero experiments: `a` is the mean of the two experiment intercepts
    expected = {
        "a": fixed["Intercept"] + fixed["C(experiment)[T.E2]"] / 2,
        "b": fixed["x_c"],
    }
    for condition_id in (1, 2, 3):
        expected[f"beta[condition[{condition_id}]]"] = fixed[
            f"C(condition_id)[T.{condition_id}]"
        ]
        expected[f"gamma[condition[{condition_id}]]"] = fixed[
            f"x_c:C(condition_id)[T.{condition_id}]"
        ]
    for parameter, value in expected.items():
        assert posterior[parameter] == pytest.approx(value, abs=0.01), parameter


@pytest.mark.slow
def test_student_t_is_closer_to_truth_in_heavy_tailed_cell():
    errors = {"normal": [], "student_t": []}
    for seed in (2, 3, 4, 5):
        table = _table(seed)
        for likelihood in errors:
            result = pmod.fit_proportion_model(
                table,
                reference_condition=0,
                likelihood=likelihood,
                group_variances=True,
                x_ref=X0,
                draws=500,
                chains=2,
                random_seed=seed,
            )
            beta = (
                result.summary()
                .set_index("parameter")
                .loc["beta[condition[3]]", "mean"]
            )
            errors[likelihood].append(abs(beta - CELLS[3]["beta"]))
    assert np.mean(errors["student_t"]) < np.mean(errors["normal"])


# REFERENCE SHAPE


def _allometry_fit(exponent_range, reference_shape, cells=ALLOMETRY_CELLS):
    table = pmod.build_proportion_table(
        simulate_allometry(1, exponent_range, cells), *ALLOMETRY_COLUMNS, [0, 1]
    )
    return pmod.fit_proportion_model(
        table,
        reference_condition=0,
        reference_shape=reference_shape,
        draws=500,
        chains=2,
        random_seed=4,
    )


@pytest.fixture(scope="module")
def curved_auto_fit():
    return _allometry_fit((2.4, 3.3), "auto")


@pytest.fixture(scope="module")
def curved_linear_fit():
    return _allometry_fit((2.4, 3.3), "linear")


@pytest.fixture(scope="module")
def low_gap_fits():
    return {
        shape: _allometry_fit((2.4, 3.3), shape, ALLOMETRY_LOW_GAP_CELLS)
        for shape in ("linear", "spline")
    }


@pytest.fixture(scope="module")
def straight_auto_fit():
    return _allometry_fit((3.0, 3.0), "auto")


@pytest.fixture(scope="module")
def straight_spline_fit():
    return _allometry_fit((3.0, 3.0), "spline")


@pytest.mark.slow
def test_auto_switches_to_spline_for_curved_allometry(curved_auto_fit):
    assert curved_auto_fit.reference_shape_used == "spline"
    assert curved_auto_fit.linearity_threshold == 0.03
    assert curved_auto_fit.linear_misfit["mean_percent"].abs().max() > 3
    assert (curved_auto_fit.spline_misfit["mean_percent"].abs() < 1).all()
    summary = curved_auto_fit.summary()
    assert summary.attrs["reference_shape"] == "spline"
    assert "sd_s" in summary["parameter"].values


@pytest.mark.slow
def test_spline_recovers_beta_of_mutant_in_gaps(curved_auto_fit, curved_linear_fit):
    true_beta = ALLOMETRY_CELLS[1]["beta"]
    spline = curved_auto_fit.summary().set_index("parameter")
    lower, upper = _interval(spline, "beta[condition[1]]")
    assert lower < true_beta < upper
    linear = curved_linear_fit.summary().set_index("parameter")
    lower, upper = _interval(linear, "beta[condition[1]]")
    assert upper < true_beta


@pytest.mark.slow
def test_spline_recovers_gamma_of_mutant_in_low_gaps(low_gap_fits):
    true_gamma = 0.0
    spline = low_gap_fits["spline"].summary().set_index("parameter")
    lower, upper = _interval(spline, "gamma[condition[1]]")
    assert lower < true_gamma < upper
    linear = low_gap_fits["linear"].summary().set_index("parameter")
    lower, upper = _interval(linear, "gamma[condition[1]]")
    assert upper < true_gamma


@pytest.mark.slow
def test_auto_keeps_line_for_straight_allometry(straight_auto_fit):
    assert straight_auto_fit.reference_shape_used == "linear"
    assert straight_auto_fit.spline_misfit is None
    assert straight_auto_fit.linear_misfit["mean_percent"].abs().max() < 3


@pytest.mark.slow
def test_forced_spline_matches_line_for_straight_allometry(
    straight_auto_fit, straight_spline_fit
):
    spline = straight_spline_fit.summary().set_index("parameter")["mean"]
    linear = straight_auto_fit.summary().set_index("parameter")["mean"]
    assert spline["sd_s"] < 0.01
    for parameter in ("beta[condition[1]]", "gamma[condition[1]]"):
        assert spline[parameter] == pytest.approx(linear[parameter], abs=0.01)


# PUBLICATION ANALYSES ON A HAND-BUILT POSTERIOR


def test_cell_offsets_split_offset_into_main_effects_and_interaction():
    coding = pmod.build_genotype_coding(list(FACTORS_2X2), FACTORS_2X2, REFERENCE_2X2)
    # the double mutant (condition 3) only spans log_x 3-4
    table = pd.concat(
        [_reference_table({1: [1.0, 2.0]}, condition_id=c) for c in (0, 1, 2)]
        + [_reference_table({1: [3.0, 4.0]}, condition_id=3)],
        ignore_index=True,
    )
    beta = np.array([0.10, 0.20, -0.05])
    gamma = np.array([0.01, 0.02, 0.03])
    result = _result(
        table,
        coding,
        {
            "a": ((), [1.0, 1.0]),
            "b": ((), [0.5, 0.5]),
            "beta": (("effect",), [beta - 0.01, beta + 0.01]),
            "gamma": (("effect",), [gamma, gamma]),
        },
        x_ref=2.0,
    )
    offsets = result.cell_offsets(log_x=3.0).set_index("cell")
    double = offsets.loc["yap1=abt7, tir=col-10"]
    # main effects: 0.10 + 0.20 + (0.01 + 0.02) * 1; interaction: -0.05 + 0.03 * 1
    assert double["expected_mean"] == pytest.approx(0.33)
    assert double["interaction_mean"] == pytest.approx(-0.02)
    assert double["offset_mean"] == pytest.approx(0.31)
    assert double["offset_mean_percent"] == pytest.approx(100 * np.expm1(0.31))
    # interaction draws are -0.03 and -0.01
    assert double["interaction_prob_positive"] == 0
    single = offsets.loc["yap1=abt7, tir=none"]
    assert single["expected_mean"] == pytest.approx(single["offset_mean"])
    assert np.isnan(single["interaction_mean"])
    assert np.isnan(single["interaction_prob_positive"])
    assert offsets["within_cell_range"].tolist() == [False, False, False, True]
    assert not offsets["within_reference_range"].any()
    at_x_ref = result.cell_offsets().set_index("cell")
    assert at_x_ref["within_reference_range"].all()
    assert at_x_ref["within_cell_range"].tolist() == [True, True, True, False]


def test_cell_offsets_with_single_factor_coding_have_no_interaction():
    coding = pmod.build_genotype_coding([0, 1], reference_condition=0)
    table = pd.concat(
        [_reference_table({1: [1.0, 2.0]}, condition_id=c) for c in (0, 1)],
        ignore_index=True,
    )
    result = _result(
        table,
        coding,
        {
            "a": ((), [1.0]),
            "b": ((), [0.5]),
            "beta": (("effect",), [[0.1]]),
            "gamma": (("effect",), [[0.2]]),
        },
        x_ref=1.5,
    )
    offsets = result.cell_offsets(log_x=2.0)
    np.testing.assert_allclose(offsets["offset_mean"], [0.0, 0.2])
    np.testing.assert_allclose(offsets["expected_mean"], offsets["offset_mean"])
    assert offsets["interaction_mean"].isna().all()


def _mutant_rows(log_x_by_molt, offset, condition_id=1):
    """Mutant worms exactly ``offset`` above the line of ``_reference_table``."""
    table = _reference_table(log_x_by_molt, condition_id=condition_id)
    return table.assign(worm_id="m" + table["worm_id"], log_y=table["log_y"] + offset)


def test_molt_positions_compare_data_and_model_offsets():
    coding = pmod.build_genotype_coding([0, 1], reference_condition=0)
    reference = _reference_table({1: [1.0, 1.2], 2: [2.0, 2.2]})
    # mutant molt means: 0.5 (below the reference), 1.6 (gap), 2.5 (above)
    mutant = _mutant_rows({1: [0.4, 0.6], 2: [1.5, 1.7], 3: [2.4, 2.6]}, 0.1)
    table = pd.concat([reference, mutant], ignore_index=True)
    table["log_y"] += -0.5 * 1.5
    result = _result(
        table,
        coding,
        {
            "a": ((), [1.0, 1.0]),
            "b": ((), [0.5, 0.5]),
            "beta": (("effect",), [[0.09], [0.11]]),
            "gamma": (("effect",), [[0.0], [0.0]]),
        },
        x_ref=1.5,
    )
    positions = result.molt_positions().set_index(["cell", "molt"])
    mutant_rows = positions.loc["condition=1"]
    assert mutant_rows["extrapolated"].tolist() == [True, False, True]
    np.testing.assert_allclose(mutant_rows["log_x_mean"], [0.5, 1.6, 2.5])
    np.testing.assert_allclose(mutant_rows["log_x_sd"], np.std([0.4, 0.6], ddof=1))
    np.testing.assert_allclose(mutant_rows["observed_offset_mean"], 0.1)
    np.testing.assert_allclose(mutant_rows["model_offset_mean"], 0.1)
    assert (mutant_rows["n"] == 2).all()
    np.testing.assert_allclose(
        positions.loc["condition=0", "observed_offset_mean"], 0, atol=1e-12
    )
    assert not positions.loc["condition=0", "extrapolated"].any()


def test_repeatability_uses_the_residual_standard_deviation():
    coding = pmod.build_genotype_coding([0, 1], reference_condition=0)
    table = _reference_table({1: [1.0, 2.0]})
    posterior = {
        "a": ((), [1.0, 1.0, 1.0]),
        "b": ((), [0.5, 0.5, 0.5]),
        "tau": (("cell",), [[0.03, 0.06]] * 3),
        "sigma": (("cell",), [[0.03, 0.03]] * 3),
    }
    normal = _result(table, coding, posterior, x_ref=1.5, group_variances=True)
    repeatability = normal.repeatability().set_index("cell")
    np.testing.assert_allclose(repeatability["repeatability_mean"], [0.5, 0.8])
    np.testing.assert_allclose(repeatability["repeatability_ratio_mean"], [1.0, 1.6])
    assert (repeatability["sd_undefined_fraction"] == 0).all()

    # nu = 4 doubles the residual variance; the nu = 1.5 draw is excluded
    student_t = _result(
        table,
        coding,
        dict(posterior, nu=((), [1.5, 4.0, 4.0])),
        x_ref=1.5,
        group_variances=True,
        likelihood="student_t",
    )
    repeatability = student_t.repeatability().set_index("cell")
    np.testing.assert_allclose(repeatability["repeatability_mean"], [1 / 3, 2 / 3])
    assert repeatability["sd_undefined_fraction"].iloc[0] == pytest.approx(1 / 3)

    shared = _result(
        table,
        coding,
        dict(posterior, tau=((), [0.03] * 3), sigma=((), [0.03] * 3)),
        x_ref=1.5,
    ).repeatability()
    assert shared["cell"].tolist() == ["shared"]
    assert shared["repeatability_mean"].iloc[0] == pytest.approx(0.5)
    assert np.isnan(shared["repeatability_ratio_mean"].iloc[0])


def test_worm_deviations_are_wide_and_differ_by_the_cell_offset():
    coding = pmod.build_genotype_coding([0, 1], reference_condition=0)
    reference = _reference_table({1: [1.0, 1.1], 2: [2.0, 2.1]})
    # worm m1 is missing at molt 2
    mutant = _mutant_rows({1: [1.0, 1.1], 2: [2.0]}, 0.0)
    table = pd.concat([reference, mutant], ignore_index=True)
    rng = np.random.default_rng(0)
    table["log_y"] += -0.5 * 1.5 + rng.normal(0, 0.02, len(table))
    result = _result(
        table,
        coding,
        {
            "a": ((), [1.0]),
            "b": ((), [0.5]),
            "beta": (("effect",), [[0.1]]),
            "gamma": (("effect",), [[0.2]]),
        },
        x_ref=1.5,
    )
    own = result.worm_deviations("own").set_index("worm_id")
    to_reference = result.worm_deviations("reference").set_index("worm_id")
    assert list(own.columns) == ["condition_id", "experiment", 1, 2]
    assert list(own.index) == ["w0", "w1", "mw0", "mw1"]
    assert np.isnan(own.loc["mw1", 2])
    np.testing.assert_allclose(
        to_reference.loc[["w0", "w1"], [1, 2]], own.loc[["w0", "w1"], [1, 2]]
    )
    # the mutant's offset is 0.1 + 0.2 * (log_x - 1.5)
    difference = (
        to_reference.loc[["mw0", "mw1"], [1, 2]] - own.loc[["mw0", "mw1"], [1, 2]]
    )
    expected = [
        [0.1 + 0.2 * (1.0 - 1.5), 0.1 + 0.2 * (2.0 - 1.5)],
        [0.1 + 0.2 * (1.1 - 1.5), np.nan],
    ]
    np.testing.assert_allclose(difference, expected)
    with pytest.raises(ValueError, match="relative_to"):
        result.worm_deviations("other")


@pytest.mark.parametrize("likelihood", ["normal", "student_t"])
def test_penetrance_interval_narrows_with_the_number_of_molts(likelihood):
    coding = pmod.build_genotype_coding([0, 1], reference_condition=0)
    tau, sigma = 0.03, 0.04
    # 95% bound of a worm mean over n molts: 1.96 * sqrt(tau² + sigma² / n)
    bound = {n: 1.96 * np.sqrt(tau**2 + sigma**2 / n) for n in (1, 3)}
    rows = []
    for name, n, deviation in (
        ("inside_1", 1, 0.8 * bound[1]),
        ("above_1", 1, 1.2 * bound[1]),
        ("inside_3", 3, -0.8 * bound[3]),
        ("below_3", 3, -1.2 * bound[3]),
        # inside the 1-molt interval but outside the 3-molt one
        ("above_3", 3, 1.2 * bound[3]),
    ):
        for molt in range(1, n + 1):
            rows.append(
                {
                    "condition_id": 1,
                    "worm_id": name,
                    "experiment": "E1",
                    "molt": molt,
                    "log_x": 1.5,
                    "log_y": 1.0 + deviation,
                }
            )
    rows.append(dict(rows[0], condition_id=0, worm_id="ref", log_y=1.0))
    table = pd.DataFrame(rows)
    assert 1.2 * bound[3] < bound[1]
    posterior = {
        "a": ((), [1.0]),
        "b": ((), [0.5]),
        "beta": (("effect",), [[0.0]]),
        "gamma": (("effect",), [[0.0]]),
        "tau": ((), [tau]),
        "sigma": ((), [sigma]),
    }
    if likelihood == "student_t":
        # a t distribution with huge nu is normal
        posterior["nu"] = ((), [1e6])
    result = _result(table, coding, posterior, x_ref=1.5, likelihood=likelihood)
    penetrance = result.penetrance(random_seed=0, n_simulations=20_000).set_index(
        "cell"
    )
    mutant = penetrance.loc["condition=1"]
    assert mutant["n_worms"] == 5
    assert mutant["below_mean"] == pytest.approx(1 / 5)
    assert mutant["above_mean"] == pytest.approx(2 / 5)
    assert mutant["outside_mean"] == pytest.approx(3 / 5)
    assert penetrance.loc["condition=0", "outside_mean"] == 0


def _dispersion_result(sd_by_condition, n_worms=150):
    coding = pmod.build_genotype_coding(list(sd_by_condition), reference_condition=0)
    rng = np.random.default_rng(0)
    rows = []
    for condition_id, sd in sd_by_condition.items():
        for worm in range(n_worms):
            for molt in range(1, 5):
                log_x = 10.0 + molt
                rows.append(
                    {
                        "condition_id": condition_id,
                        "worm_id": f"c{condition_id}_{worm}",
                        "experiment": "E1",
                        "molt": molt,
                        "log_x": log_x,
                        "log_y": 1.0 + 0.5 * (log_x - 12.0) + rng.normal(0, sd),
                    }
                )
    n_effects = len(coding.effect_names)
    result = _result(
        pd.DataFrame(rows),
        coding,
        {
            "a": ((), [1.0]),
            "b": ((), [0.5]),
            "beta": (("effect",), [np.zeros(n_effects)]),
            "gamma": (("effect",), [np.zeros(n_effects)]),
        },
        x_ref=12.0,
    )
    return result


def test_molt_dispersion_tests_detect_doubled_residual_sd_with_holm_adjustment():
    from statsmodels.stats.multitest import multipletests

    result = _dispersion_result({0: 0.03, 1: 0.06, 2: 0.03})
    for family, family_size in (("all", 8), ("comparison", 4)):
        tests = result.molt_dispersion_tests(family=family)
        assert len(tests) == 8
        assert (tests["family_size"] == family_size).all()
        assert (tests["p_holm"] >= tests["p_raw"]).all()
        assert (tests[["n_a", "n_b"]] == 150).all().all()
        groups = (
            [tests] if family == "all" else [g for _, g in tests.groupby("comparison")]
        )
        for group in groups:
            np.testing.assert_allclose(
                group["p_holm"], multipletests(group["p_raw"], method="holm")[1]
            )
        doubled = tests[tests["cell_a"] == "condition=1"]
        assert doubled["molt"].tolist() == [1, 2, 3, 4]
        assert (doubled["p_holm"] < 0.05).all()
        assert doubled["comparison"].iloc[0] == "condition=1 vs condition=0"

    explicit = result.molt_dispersion_tests(comparisons=[(1, 2)])
    assert (explicit["comparison"] == "condition=1 vs condition=2").all()
    with pytest.raises(ValueError, match="family"):
        result.molt_dispersion_tests(family="molt")


def test_fit_per_experiment_refuses_auto_reference_shape():
    with pytest.raises(ValueError, match="auto"):
        pmod.fit_per_experiment(
            _table(0), reference_shape="auto", reference_condition=0, random_seed=0
        )


# PUBLICATION ANALYSES ON SAMPLED FITS


def _cell(n, beta=0.0, tau=0.03, sigma=0.03, nu=None, x_shift=0.0):
    return {
        "n": n,
        "beta": beta,
        "gamma": 0.0,
        "tau": tau,
        "sigma": sigma,
        "nu": nu,
        "x_shift": x_shift,
    }


def _fit_simulated(cells, seed, **fit_kwargs):
    table = pmod.build_proportion_table(
        simulate_proportions(seed, cells), *COLUMNS, list(cells)
    )
    fit_kwargs.setdefault("reference_condition", 0)
    return pmod.fit_proportion_model(
        table, x_ref=X0, draws=500, chains=2, random_seed=seed, **fit_kwargs
    )


# 3 x 2 factorial: yap1 in {WT, abt7, abt8} x tir in {none, col-10}
FACTORIAL_MAIN = {"abt7": 0.10, "abt8": -0.06, "col-10": 0.05}
FACTORIAL_INTERACTION = {"abt7": 0.08, "abt8": 0.0}
FACTORIAL_FACTORS = {
    0: {"yap1": "WT", "tir": "none"},
    1: {"yap1": "abt7", "tir": "none"},
    2: {"yap1": "abt8", "tir": "none"},
    3: {"yap1": "WT", "tir": "col-10"},
    4: {"yap1": "abt7", "tir": "col-10"},
    5: {"yap1": "abt8", "tir": "col-10"},
}


def _factorial_beta(levels):
    beta = sum(FACTORIAL_MAIN.get(level, 0.0) for level in levels.values())
    if levels["tir"] == "col-10":
        beta += FACTORIAL_INTERACTION.get(levels["yap1"], 0.0)
    return beta


@pytest.fixture(scope="module")
def factorial_fit():
    cells = {
        # the abt8 double mutant is small: its whole log_x range is below X0
        c: _cell(80, _factorial_beta(levels), x_shift=-2.0 if c == 5 else 0.0)
        for c, levels in FACTORIAL_FACTORS.items()
    }
    return _fit_simulated(
        cells,
        seed=7,
        factors=FACTORIAL_FACTORS,
        reference={"yap1": "WT", "tir": "none"},
        reference_condition=None,
    )


@pytest.mark.slow
def test_cell_offsets_recover_factorial_interaction(factorial_fit):
    offsets = factorial_fit.cell_offsets().set_index("cell")
    coding = factorial_fit.coding
    for condition_id, levels in FACTORIAL_FACTORS.items():
        row = offsets.loc[coding.cell_label(levels)]
        main = sum(FACTORIAL_MAIN.get(level, 0.0) for level in levels.values())
        if levels["tir"] == "col-10" and levels["yap1"] != "WT":
            truth = FACTORIAL_INTERACTION[levels["yap1"]]
            assert row["interaction_lower"] < truth < row["interaction_upper"]
        else:
            assert np.isnan(row["interaction_mean"])
        # expected is the sum of the fitted main effects at x_ref
        fitted_main = sum(
            offsets.loc[
                coding.cell_label({**coding.reference, name: level}), "offset_mean"
            ]
            for name, level in levels.items()
            if level != coding.reference[name]
        )
        assert row["expected_mean"] == pytest.approx(fitted_main)
        assert row["expected_lower"] <= main <= row["expected_upper"]
        assert row["within_cell_range"] == (condition_id != 5)
    assert offsets["within_reference_range"].all()


@pytest.fixture(scope="module")
def variance_fit():
    cells = {
        0: _cell(400),
        1: _cell(150, beta=0.3),
        2: _cell(150, tau=0.05, sigma=0.02),
        3: _cell(150, tau=0.02, sigma=0.04),
    }
    # with this seed every cell's realized REML repeatability is within 0.6 SD of
    # its truth; on some seeds a cell's data alone sit 2 SD away
    return cells, _fit_simulated(cells, seed=15, group_variances=True)


@pytest.mark.slow
def test_molt_positions_agree_with_model_when_correctly_specified(variance_fit):
    _, result = variance_fit
    positions = result.molt_positions()
    np.testing.assert_allclose(
        positions["observed_offset_mean"], positions["model_offset_mean"], atol=0.01
    )
    # every cell shares the reference sizes
    assert not positions["extrapolated"].any()


@pytest.mark.slow
def test_repeatability_recovers_simulated_variance_shares(variance_fit):
    cells, result = variance_fit
    repeatability = result.repeatability().set_index("cell")
    for condition_id, cell in cells.items():
        truth = cell["tau"] ** 2 / (cell["tau"] ** 2 + cell["sigma"] ** 2)
        row = repeatability.loc[f"condition={condition_id}"]
        assert row["repeatability_lower"] < truth < row["repeatability_upper"]


@pytest.mark.slow
def test_penetrance_is_calibrated_on_the_reference_cell(variance_fit):
    _, result = variance_fit
    penetrance = result.penetrance(prob=0.95).set_index("cell")
    assert penetrance.loc["condition=0", "outside_mean"] == pytest.approx(
        0.05, abs=0.03
    )
    assert penetrance.loc["condition=1", "outside_mean"] > 0.95
    assert penetrance.loc["condition=1", "above_mean"] > 0.95


@pytest.fixture(scope="module")
def heavy_tailed_fits():
    cells = {0: _cell(100), 1: _cell(150, nu=3)}
    return {
        likelihood: _fit_simulated(
            cells, seed=6, group_variances=True, likelihood=likelihood
        )
        for likelihood in ("normal", "student_t")
    }


@pytest.mark.slow
def test_posterior_predictive_kurtosis_flags_heavy_tails_only_under_normal(
    heavy_tailed_fits,
):
    p_values = {}
    for likelihood, result in heavy_tailed_fits.items():
        replicated, observed = result.posterior_predictive_residuals(random_seed=0)
        assert len(replicated) == len(observed) == 200 * len(result.table)
        summary = result.ppc_summary(replicated, observed).set_index(
            ["cell", "statistic"]
        )
        p_values[likelihood] = summary.loc[
            ("condition=1", "excess_kurtosis"), "p_value"
        ]
    assert not 0.05 <= p_values["normal"] <= 0.95
    assert 0.05 <= p_values["student_t"] <= 0.95


@pytest.mark.slow
def test_fit_per_experiment_recovers_common_effects():
    cells = {c: CELLS[c] for c in (0, 1, 2)}
    table = pmod.build_proportion_table(
        simulate_proportions(9, cells), *COLUMNS, list(cells)
    )
    # condition 2 was not imaged in E2
    table = table[~((table["condition_id"] == 2) & (table["experiment"] == "E2"))]
    with pytest.warns(
        UserWarning, match=r"E2 has no observations of cells \['condition=2'\]"
    ):
        results = pmod.fit_per_experiment(
            table,
            reference_shape="linear",
            reference_condition=0,
            draws=500,
            chains=2,
            random_seed=8,
        )
    assert list(results) == ["E1", "E2"]
    assert results["E2"].coding.cells == ["condition=0", "condition=1"]
    assert results["E1"].x_ref == results["E2"].x_ref
    comparison = pmod.compare_experiments(results).set_index(
        ["experiment", "parameter"]
    )
    for experiment, result in results.items():
        for condition_id in (1, 2):
            if f"condition={condition_id}" not in result.coding.cells:
                continue
            row = comparison.loc[(experiment, f"beta[condition[{condition_id}]]")]
            assert row["lower"] < CELLS[condition_id]["beta"] < row["upper"]


# NAMES WITH PUNCTUATION

COLON_FACTORS = {
    0: {"yap1": "wild type", "tir": "no TIR"},
    1: {"yap1": "abt7", "tir": "no TIR"},
    2: {"yap1": "wild type", "tir": "col-10:TIR"},
    3: {"yap1": "abt7", "tir": "col-10:TIR"},
}
COLON_REFERENCE = {"yap1": "wild type", "tir": "no TIR"}
NAME_CHARACTERS = [":", ";", ",", "(", ")", "[", "]", "=", "+", "/", " ", "µ"]
# 3 x 2 factorial without the (xyz, col-10) cell, with plain names
PLAIN_FACTORS = {
    0: {"yap1": "WT", "tir": "none"},
    1: {"yap1": "abt7", "tir": "none"},
    2: {"yap1": "xyz", "tir": "none"},
    3: {"yap1": "WT", "tir": "col-10"},
    4: {"yap1": "abt7", "tir": "col-10"},
}
PLAIN_REFERENCE = {"yap1": "WT", "tir": "none"}


def test_level_with_colon_keeps_its_main_effect_and_interaction():
    coding = pmod.build_genotype_coding(
        list(COLON_FACTORS), COLON_FACTORS, COLON_REFERENCE
    )
    assert coding.effects == [
        (("yap1", "abt7"),),
        (("tir", "col-10:TIR"),),
        (("yap1", "abt7"), ("tir", "col-10:TIR")),
    ]
    np.testing.assert_array_equal(coding.is_main_effect, [True, True, False])
    np.testing.assert_array_equal(
        coding.design_matrix([0, 1, 2, 3]).to_numpy(),
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]],
    )

    table = pd.concat(
        [_reference_table({1: [1.0, 2.0]}, condition_id=c) for c in COLON_FACTORS],
        ignore_index=True,
    )
    beta = np.array([0.10, 0.20, -0.05])
    gamma = np.array([0.01, 0.02, 0.03])
    result = _result(
        table,
        coding,
        {
            "a": ((), [1.0]),
            "b": ((), [0.5]),
            "beta": (("effect",), [beta]),
            "gamma": (("effect",), [gamma]),
        },
        x_ref=2.0,
    )
    offsets = result.cell_offsets(log_x=3.0).set_index("cell")
    double = offsets.loc["yap1=abt7, tir=col-10:TIR"]
    assert double["expected_mean"] == pytest.approx(0.10 + 0.20 + 0.01 + 0.02)
    assert double["interaction_mean"] == pytest.approx(-0.05 + 0.03)
    tir = offsets.loc["yap1=wild type, tir=col-10:TIR"]
    assert tir["offset_mean"] == pytest.approx(0.22)
    assert np.isnan(tir["interaction_mean"])


def _punctuated(factors, reference, character):
    """Rename every factor and level of a coding to contain ``character``."""

    def rename(name):
        return f"{name}{character}x{character}"

    return (
        {
            c: {rename(f): rename(level) for f, level in levels.items()}
            for c, levels in factors.items()
        },
        {rename(f): rename(level) for f, level in reference.items()},
        rename,
    )


@pytest.mark.parametrize("character", NAME_CHARACTERS)
def test_coding_with_punctuated_names_matches_plain_coding(character):
    factors, reference, rename = _punctuated(PLAIN_FACTORS, PLAIN_REFERENCE, character)
    plain = pmod.build_genotype_coding(
        list(PLAIN_FACTORS), PLAIN_FACTORS, PLAIN_REFERENCE
    )
    coding = pmod.build_genotype_coding(list(factors), factors, reference)

    assert coding.effects == [
        tuple((rename(f), rename(level)) for f, level in effect)
        for effect in plain.effects
    ]
    np.testing.assert_array_equal(coding.is_main_effect, plain.is_main_effect)
    np.testing.assert_array_equal(
        coding.design_matrix(list(factors)).to_numpy(),
        plain.design_matrix(list(factors)).to_numpy(),
    )
    assert coding.cells == [
        ", ".join(f"{rename(f)}={rename(level)}" for f, level in levels.items())
        for levels in plain.cell_levels
    ]
    for condition_id, levels in factors.items():
        np.testing.assert_array_equal(
            coding.effect_vector(levels),
            plain.effect_vector(PLAIN_FACTORS[condition_id]),
        )
        label = coding.cell_of_condition(condition_id)
        for cell in (condition_id, label, levels):
            assert coding.resolve_cell(cell) == levels
    with pytest.raises(ValueError, match="fitted cells"):
        coding.resolve_cell(rename("absent"))


def test_effects_with_the_same_label_raise():
    # the level "1]:y[2" makes the main effect read like the interaction x[1]:y[2]
    factors = {
        0: {"x": "0", "y": "0"},
        1: {"x": "1]:y[2", "y": "0"},
        2: {"x": "1", "y": "0"},
        3: {"x": "0", "y": "2"},
        4: {"x": "1", "y": "2"},
    }
    with pytest.raises(ValueError, match=r"effects .*'x\[1\]:y\[2\]'"):
        pmod.build_genotype_coding(list(factors), factors, {"x": "0", "y": "0"})


def test_cells_with_the_same_label_raise():
    factors = {
        0: {"a": "0", "b": "0"},
        1: {"a": "1, b=2", "b": "3"},
        2: {"a": "1", "b": "2, b=3"},
    }
    with pytest.raises(ValueError, match="cells .*'a=1, b=2, b=3'"):
        pmod.build_genotype_coding(list(factors), factors, {"a": "0", "b": "0"})


def test_summary_kind_effect_and_cell_columns_with_bracketed_names():
    factors = {
        0: {"yap[1]": "WT", "tir:": "none"},
        1: {"yap[1]": "abt[7]:x", "tir:": "none"},
        2: {"yap[1]": "WT", "tir:": "col-10:TIR"},
        3: {"yap[1]": "abt[7]:x", "tir:": "col-10:TIR"},
    }
    coding = pmod.build_genotype_coding(
        list(factors), factors, {"yap[1]": "WT", "tir:": "none"}
    )
    table = pd.concat(
        [_reference_table({1: [1.0, 2.0]}, condition_id=c) for c in factors],
        ignore_index=True,
    )
    beta = np.array([0.1, 0.2, 0.3])
    tau = np.array([0.01, 0.02, 0.03, 0.04])
    posterior = {
        "a": ((), [1.0]),
        "b": ((), [0.5]),
        "beta": (("effect",), [beta]),
        "gamma": (("effect",), [-beta]),
        "experiment_effect": (("experiment",), [[0.01, -0.01]]),
        "nu": ((), [4.0]),
        "tau": (("cell",), [tau]),
        "sigma": (("cell",), [2 * tau]),
    }
    result = _result(
        table,
        coding,
        posterior,
        x_ref=1.5,
        group_variances=True,
        likelihood="student_t",
    )
    summary = result.summary()
    assert list(summary.columns[:4]) == ["parameter", "kind", "effect", "cell"]

    def rows(kind):
        return summary[summary["kind"] == kind]

    interaction = "yap[1][abt[7]:x]:tir:[col-10:TIR]"
    assert rows("beta")["effect"].tolist() == [
        "yap[1][abt[7]:x]",
        "tir:[col-10:TIR]",
        interaction,
    ]
    np.testing.assert_allclose(rows("beta")["mean"], beta)
    np.testing.assert_allclose(rows("gamma")["mean"], -beta)
    assert rows("beta")["cell"].isna().all()
    assert rows("beta")["parameter"].iloc[2] == f"beta[{interaction}]"
    assert rows("experiment_effect")[["effect", "cell"]].isna().all().all()
    assert len(rows("experiment_effect")) == 2
    for kind in ("tau", "sigma", "sigma_sd"):
        assert rows(kind)["cell"].tolist() == coding.cells
        assert rows(kind)["effect"].isna().all()
    np.testing.assert_allclose(rows("tau")["mean"], tau)
    for kind in ("tau_ratio", "sigma_ratio"):
        assert rows(kind)["cell"].tolist() == coding.cells[1:]
        np.testing.assert_allclose(rows(kind)["mean"], tau[1:] / tau[0])
    for kind in ("a", "b", "nu", "sigma_sd_undefined_fraction"):
        assert len(rows(kind)) == 1
        assert rows(kind)[["effect", "cell"]].isna().all().all()

    comparison = pmod.compare_experiments({"E1": result})
    assert (
        comparison["kind"].tolist()
        == ["beta"] * 3 + ["gamma"] * 3 + ["tau_ratio"] * 3 + ["sigma_ratio"] * 3
    )
    assert comparison["effect"].iloc[:6].tolist() == rows("beta")["effect"].tolist() * 2
    assert comparison["cell"].iloc[6:].tolist() == coding.cells[1:] * 2


def test_molt_dispersion_families_do_not_merge_comparisons_with_the_same_text():
    # both comparisons read "g=a vs g=b vs g=c"
    factors = {
        0: {"g": "a"},
        1: {"g": "a vs g=b"},
        2: {"g": "b vs g=c"},
        3: {"g": "c"},
    }
    coding = pmod.build_genotype_coding(list(factors), factors, {"g": "a"})
    rng = np.random.default_rng(0)
    table = pd.DataFrame(
        [
            {
                "condition_id": c,
                "worm_id": f"c{c}_w{w}",
                "experiment": "E1",
                "molt": molt,
                "log_x": 1.0 + molt,
                "log_y": rng.normal(0, 0.1),
            }
            for c in factors
            for w in range(6)
            for molt in (1, 2)
        ]
    )
    result = _result(
        table,
        coding,
        {
            "a": ((), [0.0]),
            "b": ((), [0.0]),
            "beta": (("effect",), [np.zeros(3)]),
            "gamma": (("effect",), [np.zeros(3)]),
        },
        x_ref=1.5,
    )
    tests = result.molt_dispersion_tests(
        [(1, 3), ({"g": "a"}, "g=b vs g=c")], family="comparison"
    )
    assert tests["comparison"].nunique() == 1
    assert (tests["family_size"] == 2).all()


def _old_coding(coding, effect_names):
    """A coding as pickled before effects were structured: names only."""
    old = object.__new__(pmod.GenotypeCoding)
    old.__dict__.update(
        factor_names=coding.factor_names,
        reference=coding.reference,
        levels=coding.levels,
        condition_levels=coding.condition_levels,
        effect_names=effect_names,
        cells=coding.cells,
    )
    return old


@pytest.mark.parametrize(
    "factors, reference, old_effects",
    [
        (FACTORS_2X2, REFERENCE_2X2, None),
        # the old coding dropped the effects whose level contains ":"
        (COLON_FACTORS, COLON_REFERENCE, ["yap1[abt7]"]),
    ],
)
def test_results_pickled_with_the_old_coding_load(factors, reference, old_effects):
    coding = pmod.build_genotype_coding(list(factors), factors, reference)
    effect_names = coding.effect_names if old_effects is None else old_effects
    table = pd.concat(
        [_reference_table({1: [1.0, 2.0]}, condition_id=c) for c in factors],
        ignore_index=True,
    )
    n_effects = len(effect_names)
    result = _result(
        table,
        SimpleNamespace(effect_names=effect_names, cells=coding.cells),
        {
            "a": ((), [1.0]),
            "b": ((), [0.5]),
            "tau": ((), [0.1]),
            "sigma": ((), [0.1]),
            "beta": (("effect",), [0.1 * np.arange(1, n_effects + 1)]),
            "gamma": (("effect",), [np.zeros(n_effects)]),
        },
        x_ref=1.5,
    )
    result.coding = _old_coding(coding, effect_names)

    loaded = pickle.loads(pickle.dumps(result))
    assert loaded.coding == coding
    if old_effects is None:
        summary = loaded.summary()
        np.testing.assert_allclose(
            summary.loc[summary["kind"] == "beta", "mean"], [0.1, 0.2, 0.3]
        )
        assert pickle.loads(pickle.dumps(loaded)).coding == coding
    else:
        with pytest.raises(ValueError, match="must be refitted"):
            loaded.summary()
        with pytest.raises(ValueError, match="must be refitted"):
            loaded.cell_offsets()


@pytest.mark.slow
def test_fit_recovers_interaction_of_level_with_colon():
    beta = {0: 0.0, 1: 0.10, 2: -0.05, 3: 0.10 - 0.05 + 0.08}
    cells = {c: _cell(80, beta[c]) for c in COLON_FACTORS}
    result = _fit_simulated(
        cells,
        seed=11,
        factors=COLON_FACTORS,
        reference=COLON_REFERENCE,
        reference_condition=None,
        reference_shape="linear",
    )
    summary = result.summary()
    interaction = summary[
        (summary["kind"] == "beta")
        & (summary["effect"] == "yap1[abt7]:tir[col-10:TIR]")
    ].iloc[0]
    assert interaction["lower"] < 0.08 < interaction["upper"]
    assert interaction["lower"] > 0
    offsets = result.cell_offsets().set_index("cell")
    double = offsets.loc["yap1=abt7, tir=col-10:TIR"]
    assert double["interaction_lower"] < 0.08 < double["interaction_upper"]
    assert double["expected_lower"] < 0.05 < double["expected_upper"]
