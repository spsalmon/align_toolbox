import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from xarray import Dataset, DataTree

from align_toolbox.data_analysis import proportion_model as pmod
from tests.data_analysis.proportion_simulation import (
    CELLS,
    EXPERIMENTS,
    X0,
    simulate_proportions,
)

COLUMNS = ("volume_at_ecdysis", "pharynx_volume_at_ecdysis")
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


def _result(table, coding, posterior, x_ref, group_variances=False):
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
    idata = DataTree.from_dict({"posterior": Dataset(data, coords=coords)})
    return pmod.ProportionModelResult(
        idata=idata,
        table=table,
        coding=coding,
        x_ref=x_ref,
        experiment_labels={},
        counts=None,
        diagnostics={},
        likelihood="normal",
        group_variances=group_variances,
        random_seed=0,
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
    # a, b and the worm/experiment effects reproduce the line exactly
    experiment_effect = [0.03, -0.03]
    worm_effect = {"w0": 0.02, "w1": -0.01}
    table["log_y"] += (
        table["experiment"].map({"E1": 0.03, "E2": -0.03})
        + table["worm_id"].map(worm_effect)
        - 0.5 * 2.0
    )
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
