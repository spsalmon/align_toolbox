import json
import pickle

import numpy as np
import pytest
from xarray import Dataset, DataTree

from align_toolbox.data_analysis import proportion_model as pmod
from align_toolbox.data_analysis import proportion_model_cache as pmc
from tests.data_analysis.proportion_simulation import simulate_series_proportions

COLUMNS = ("volume_at_ecdysis", "pharynx_volume_at_ecdysis")
SERIES_COLUMNS = ("volume", "pharynx_volume")
PERCENTAGES = [0.1, 0.25, 0.6]
FIT_KWARGS = {"reference_condition": 0, "random_seed": 1, "draws": 10}


def _fake_result(table, **kwargs):
    """A result with a one-draw posterior and the provenance a real fit records."""
    coding = pmod.build_genotype_coding(
        sorted(int(c) for c in table["condition_id"].unique()),
        kwargs.get("factors"),
        kwargs.get("reference"),
        kwargs.get("reference_condition"),
    )
    posterior = Dataset(
        {"a": (("chain", "draw"), [[1.0]])}, coords={"chain": [0], "draw": [0]}
    )
    result = pmod.ProportionModelResult(
        idata=DataTree.from_dict({"posterior": posterior}),
        table=table,
        coding=coding,
        x_ref=0.0,
        experiment_labels={},
        counts=None,
        diagnostics={},
        likelihood=kwargs.get("likelihood", "normal"),
        group_variances=False,
        random_seed=kwargs["random_seed"],
    )
    result.provenance = pmod.fit_provenance(table, kwargs)
    return result


@pytest.fixture
def fits(monkeypatch):
    """Tables the fake ``fit_proportion_model`` was called with."""
    calls = []

    def fake_fit(table, **kwargs):
        calls.append(table)
        return _fake_result(table, **kwargs)

    monkeypatch.setattr(pmc, "fit_proportion_model", fake_fit)
    return calls


@pytest.fixture(scope="module")
def series():
    worms = {0: {"n": 6, "beta": 0.0}, 1: {"n": 6, "beta": 0.1}}
    return simulate_series_proportions(seed=2, cells=worms)


def _ecdysis_fit(cache_dir, conditions_struct, columns=COLUMNS, **kwargs):
    return pmc.fit_or_load_proportion_model(
        cache_dir, conditions_struct, *columns, [0, 1], **{**FIT_KWARGS, **kwargs}
    )


def _percentage_fit(cache_dir, series, percentages=PERCENTAGES, **kwargs):
    return pmc.fit_or_load_proportion_model(
        cache_dir,
        series,
        *SERIES_COLUMNS,
        [0, 1],
        percentages=percentages,
        **{**FIT_KWARGS, **kwargs},
    )


def test_repeat_call_loads_without_fitting(tmp_path, fits, simulated_proportions):
    first = _ecdysis_fit(tmp_path, simulated_proportions)
    second = _ecdysis_fit(tmp_path, simulated_proportions)
    assert len(fits) == 1
    assert second.provenance == first.provenance
    # an argument left at its default and the same argument passed explicitly
    _ecdysis_fit(tmp_path, simulated_proportions, likelihood="normal")
    assert len(fits) == 1


def test_reordered_or_duplicated_percentages_hit_the_same_entry(tmp_path, fits, series):
    _percentage_fit(tmp_path, series)
    _percentage_fit(tmp_path, series, percentages=[0.6, 0.1, 0.25])
    _percentage_fit(tmp_path, series, percentages=np.array([0.25, 0.1, 0.6, 0.1]))
    assert len(fits) == 1
    assert fits[0].attrs["percentages"] == PERCENTAGES


def _change_data(conditions_struct):
    changed = [dict(condition) for condition in conditions_struct]
    values = changed[1][COLUMNS[1]].copy()
    values[0, 1] *= 1.01
    changed[1][COLUMNS[1]] = values
    return changed


@pytest.mark.parametrize(
    "change",
    [
        lambda fit, data, path: fit(path, data, columns=COLUMNS[::-1]),
        lambda fit, data, path: fit(path, _change_data(data)),
        lambda fit, data, path: fit(path, data, likelihood="student_t"),
        lambda fit, data, path: fit(path, data, random_seed=2),
        lambda fit, data, path: fit(path, data, refit=True),
    ],
    ids=["columns", "data", "likelihood", "seed", "refit"],
)
def test_changed_request_or_refit_fits_again(
    tmp_path, fits, simulated_proportions, change
):
    _ecdysis_fit(tmp_path, simulated_proportions)
    change(_ecdysis_fit, simulated_proportions, tmp_path)
    assert len(fits) == 2
    _ecdysis_fit(tmp_path, simulated_proportions)
    assert len(fits) == 2


def test_different_percentages_fit_again(tmp_path, fits, series):
    _percentage_fit(tmp_path, series)
    _percentage_fit(tmp_path, series, percentages=[0.1, 0.25, 0.65])
    assert len(fits) == 2


def test_different_smoothing_fits_again(tmp_path, fits, series):
    _percentage_fit(tmp_path, series)
    _percentage_fit(tmp_path, series, lmbda=0.1)
    assert len(fits) == 2
    _percentage_fit(tmp_path, series, lmbda=0.1)
    assert len(fits) == 2


def _only_entry(cache_dir):
    (pickle_path,) = cache_dir.glob("*.pkl")
    return pickle_path, pickle_path.with_suffix(".json")


def test_corrupted_pickle_warns_and_refits(tmp_path, fits, simulated_proportions):
    _ecdysis_fit(tmp_path, simulated_proportions)
    pickle_path, _ = _only_entry(tmp_path)
    pickle_path.write_bytes(b"not a pickle")
    with pytest.warns(UserWarning, match="could not be read"):
        _ecdysis_fit(tmp_path, simulated_proportions)
    assert len(fits) == 2
    _ecdysis_fit(tmp_path, simulated_proportions)
    assert len(fits) == 2


def test_entry_of_another_request_is_not_loaded(tmp_path, fits, simulated_proportions):
    other = _ecdysis_fit(tmp_path / "other", simulated_proportions, random_seed=9)
    _ecdysis_fit(tmp_path, simulated_proportions)
    pickle_path, _ = _only_entry(tmp_path)
    pickle_path.write_bytes(pickle.dumps(other))
    with pytest.warns(UserWarning, match="does not record"):
        result = _ecdysis_fit(tmp_path, simulated_proportions)
    assert len(fits) == 3
    assert result.random_seed == FIT_KWARGS["random_seed"]


def test_old_result_without_provenance_in_cache_refits(
    tmp_path, fits, simulated_proportions
):
    _ecdysis_fit(tmp_path, simulated_proportions)
    pickle_path, _ = _only_entry(tmp_path)
    old = pickle.loads(pickle_path.read_bytes())
    del old.__dict__["provenance"]
    pickle_path.write_bytes(pickle.dumps(old))
    assert pickle.loads(pickle_path.read_bytes()).provenance is None
    with pytest.warns(UserWarning, match="does not record"):
        _ecdysis_fit(tmp_path, simulated_proportions)
    assert len(fits) == 2


def test_sidecar_matches_the_result_provenance(tmp_path, fits, series):
    result = _percentage_fit(tmp_path, series, likelihood="student_t")
    pickle_path, sidecar_path = _only_entry(tmp_path)
    provenance = json.loads(sidecar_path.read_text())
    assert provenance == result.provenance
    assert provenance["cache"]["file"] == pickle_path.name
    assert provenance["percentages"] == PERCENTAGES
    assert provenance["lmbda"] == 0.0075
    assert provenance["cache"]["request"]["medfilt_window"] == 5
    assert provenance["column_x"] == SERIES_COLUMNS[0]
    assert provenance["condition_ids"] == [0, 1]
    assert provenance["reference_condition"] == 0
    assert provenance["fit_arguments"]["likelihood"] == "student_t"
    assert provenance["data_hash"] == pmod.table_data_hash(fits[0])
    # no temporary files left behind
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [pickle_path.name, sidecar_path.name]
    )


def test_list_cached_models_lists_every_entry(
    tmp_path, fits, simulated_proportions, series
):
    _ecdysis_fit(tmp_path, simulated_proportions)
    _percentage_fit(tmp_path, series, likelihood="student_t")
    listing = pmc.list_cached_models(tmp_path).set_index("sampling")
    assert len(listing) == 2
    assert listing.loc["ecdysis", "percentages"] is None
    assert listing.loc["percentages", "percentages"] == PERCENTAGES
    assert listing.loc["percentages", "likelihood"] == "student_t"
    assert listing.loc["ecdysis", "column_y"] == COLUMNS[1]
    assert listing.loc["ecdysis", "conditions"] == [0, 1]
    assert listing.loc["ecdysis", "reference_shape"] == "auto"
    assert set(listing["file"]) == {p.name for p in tmp_path.glob("*.pkl")}
    assert pmc.list_cached_models(tmp_path / "empty").empty


def test_per_experiment_fits_are_cached(monkeypatch, tmp_path, series):
    calls = []

    def fake_fit_per_experiment(table, reference_shape, **kwargs):
        calls.append(reference_shape)
        return {
            experiment: _fake_result(rows.reset_index(drop=True), **kwargs)
            for experiment, rows in table.groupby("experiment")
        }

    monkeypatch.setattr(pmc, "fit_per_experiment", fake_fit_per_experiment)

    def fit(**kwargs):
        return pmc.fit_or_load_per_experiment(
            tmp_path,
            series,
            *SERIES_COLUMNS,
            [0, 1],
            percentages=PERCENTAGES,
            **{**FIT_KWARGS, "reference_shape": "linear", **kwargs},
        )

    first = fit()
    second = fit()
    assert calls == ["linear"]
    assert isinstance(second, pmc.PerExperimentFits)
    assert sorted(second) == ["E1", "E2"]
    assert second.provenance == first.provenance
    assert second.provenance["fit_arguments"]["reference_shape"] == "linear"
    fit(reference_shape="spline")
    assert calls == ["linear", "spline"]
    listing = pmc.list_cached_models(tmp_path)
    assert listing["kind"].tolist() == ["per_experiment"] * 2
