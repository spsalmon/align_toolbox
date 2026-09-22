import numpy as np
import polars as pl
import pytest
import yaml

from align_toolbox.plotting import plotting_structure as ps

N_FRAMES = 40
FIRST_TIME = 100
MOLTS = ["HatchTime", "M1", "M2", "M3", "M4"]
MOLT_INDICES = [2, 10, 20, 30, 38]


def _point_rows(point, molt_indices, n_frames=N_FRAMES, with_values_at_molt=True):
    """Long-format filemap rows for one point; Time starts at FIRST_TIME."""
    index = np.arange(n_frames)
    volume = 1e4 * (point + 1) * np.exp(0.05 * index)
    rows = {
        "Time": FIRST_TIME + index,
        "Point": np.full(n_frames, point),
        "ExperimentTime": index * 600.0,
        "raw": [f"/raw/Time{t}_Point{point}.tiff" for t in FIRST_TIME + index],
        "ch2_seg_str_volume": volume,
        "ch2_seg_str_worm_type": np.where(index < 2, "egg", "worm"),
        "Food": np.full(n_frames, "OP50"),
    }
    for molt, molt_index in zip(MOLTS, molt_indices):
        missing = molt_index is None
        molt_time = np.nan if missing else FIRST_TIME + molt_index
        value = volume[molt_index] if with_values_at_molt and not missing else np.nan
        rows[molt] = np.full(n_frames, molt_time, dtype=float)
        rows[f"ch2_seg_str_volume_at_{molt}"] = np.full(n_frames, value, dtype=float)
    return pl.DataFrame(rows)


@pytest.fixture
def conditions_yaml(tmp_path):
    path = tmp_path / "conditions.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "conditions": [
                    {
                        "point_range": [[0, 0], [1, 1]],
                        "description": ["wild type", "mutant"],
                        "strain": ["N2", "CB4856"],
                    }
                ]
            }
        )
    )
    return str(path)


@pytest.fixture
def experiment_filemap(tmp_path):
    """Point 0 has every molt and stored at-molt values; point 1 misses M4 and values."""
    filemap = pl.concat(
        [
            _point_rows(0, MOLT_INDICES),
            _point_rows(1, MOLT_INDICES[:4] + [None], with_values_at_molt=False),
        ]
    )
    path = tmp_path / "filemap.csv"
    filemap.write_csv(path)
    return str(path)


@pytest.fixture
def plotting_struct(experiment_filemap, conditions_yaml, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return ps.build_plotting_struct(
        str(tmp_path),
        experiment_filemap,
        conditions_yaml,
        organ_channels={"body": "ch2"},
    )


# --- build_conditions ---------------------------------------------------------------


def test_build_conditions_expands_factorized_lists():
    conditions = ps.build_conditions(
        {
            "conditions": [
                {"point_range": [[0, 9], [10, 19]], "strain": "N2", "food": ["a", "b"]},
                {"pad": "A", "strain": "CB"},
            ]
        }
    )
    assert conditions == [
        {"point_range": [0, 9], "strain": "N2", "food": "a", "condition_id": 0},
        {"point_range": [10, 19], "strain": "N2", "food": "b", "condition_id": 1},
        {"pad": "A", "strain": "CB", "condition_id": 2},
    ]


def test_build_conditions_reads_yaml_file(conditions_yaml):
    assert [c["description"] for c in ps.build_conditions(conditions_yaml)] == [
        "wild type",
        "mutant",
    ]


@pytest.mark.parametrize(
    "condition",
    [
        {"a": [1, 2], "b": [1, 2, 3]},
        {"a": [1, 2], "b": [1, 2, 3], "c": 1},
    ],
)
def test_build_conditions_rejects_incompatible_list_lengths(condition):
    with pytest.raises(ValueError):
        ps.build_conditions({"conditions": [condition]})


# --- filemap helpers -------------------------------------------------------------------


def test_add_conditions_to_filemap_by_point_range_and_pad():
    filemap = pl.DataFrame({"Point": [0, 1, 2, 3], "Pad": ["A", "A", "B", "C"]})
    conditions = [
        {"point_range": [[0, 0], [2, 2]], "strain": "N2", "condition_id": 0},
        {"pad": "C", "strain": "CB", "condition_id": 1},
        {"strain": "ignored", "condition_id": 2},
    ]
    result = ps._add_conditions_to_filemap(filemap, conditions)
    assert result["strain"].to_list() == ["N2", None, "N2", "CB"]
    assert result["condition_id"].to_list() == [0, None, 0, 1]


def test_separate_column_by_point_numeric_pads_with_nan():
    filemap = pl.DataFrame({"Point": [3, 3, 1], "v": [1, None, 5]})
    np.testing.assert_array_equal(
        ps.separate_column_by_point(filemap, "v"), [[1, np.nan], [5, np.nan]]
    )


def test_separate_column_by_point_strings_pad_with_error():
    filemap = pl.DataFrame({"Point": [0, 0, 1], "qc": ["worm", None, "egg"]})
    result = ps.separate_column_by_point(filemap, "qc")
    assert result.dtype == object
    assert result.tolist() == [["worm", None], ["egg", "error"]]


@pytest.mark.parametrize(
    "values, expected",
    [
        ([True, False, True], [[1.0, 0.0], [1.0, np.nan]]),
        ([None, None, None], [[np.nan, np.nan], [np.nan, np.nan]]),
    ],
    ids=["boolean", "all-null"],
)
def test_separate_column_by_point_boolean_and_null_become_float(values, expected):
    filemap = pl.DataFrame({"Point": [0, 0, 1], "v": values})
    np.testing.assert_array_equal(ps.separate_column_by_point(filemap, "v"), expected)


def test_separate_column_by_point_other_dtypes_pad_with_none():
    filemap = pl.DataFrame({"Point": [0, 0, 1], "v": [[1], [2], [3]]})
    result = ps.separate_column_by_point(filemap, "v")
    assert result[1, 1] is None
    assert list(result[0, 0]) == [1]


def test_separate_column_by_point_edge_cases():
    empty = pl.DataFrame({"Point": [], "v": []})
    assert ps.separate_column_by_point(empty, "v").shape == (0, 0)
    with pytest.raises(ValueError, match="'missing'"):
        ps.separate_column_by_point(pl.DataFrame({"Point": [0]}), "missing")


def test_remove_ignored_molts_nulls_molts_on_ignored_frames():
    filemap = pl.DataFrame(
        {
            "Point": [0, 0, 1, 1],
            "Time": [5, 6, 5, 6],
            "Ignore": [False, True, False, False],
            "HatchTime": [5.0, 5.0, 5.0, 5.0],
            "M1": [6.0, 6.0, 6.0, 6.0],
            "M2": [None] * 4,
            "M3": [None] * 4,
            "M4": [None] * 4,
        }
    )
    result = ps.remove_ignored_molts(filemap)
    assert result["M1"].to_list() == [None, None, 6.0, 6.0]
    assert result["HatchTime"].to_list() == [5.0] * 4


def test_remove_ignored_molts_without_ignore_column_is_identity():
    filemap = pl.DataFrame({"Point": [0]})
    assert ps.remove_ignored_molts(filemap) is filemap


def test_remove_unwanted_info():
    info = [{"description": "x", "condition_id": 0, "strain": "N2"}]
    assert ps.remove_unwanted_info(info) == [{"strain": "N2"}]


# --- build_plotting_struct ----------------------------------------------------------------


def test_build_plotting_struct_one_dict_per_condition(plotting_struct):
    conditions_struct, conditions_info = plotting_struct
    assert conditions_info == [
        {"description": "wild type", "strain": "N2", "condition_id": 0},
        {"description": "mutant", "strain": "CB4856", "condition_id": 1},
    ]
    assert [c["condition_id"] for c in conditions_struct] == [0, 1]
    assert conditions_struct[1]["point"].tolist() == [[1]]


def test_build_plotting_struct_time_and_ecdysis(plotting_struct):
    control, mutant = plotting_struct[0]
    np.testing.assert_array_equal(control["time"], [FIRST_TIME + np.arange(N_FRAMES)])
    np.testing.assert_array_equal(control["ecdysis_index"], [MOLT_INDICES])
    np.testing.assert_array_equal(
        control["ecdysis_time_step"], [np.array(MOLT_INDICES) + FIRST_TIME]
    )
    np.testing.assert_allclose(
        control["ecdysis_experiment_time_hours"], [np.array(MOLT_INDICES) / 6]
    )
    np.testing.assert_array_equal(
        control["larval_stage_durations_time_step"], [[8, 10, 10, 8]]
    )
    np.testing.assert_array_equal(
        mutant["larval_stage_durations_time_step"], [[8, 10, 10, np.nan]]
    )
    assert np.isnan(mutant["ecdysis_index"][0, 4])


def test_build_plotting_struct_features_qc_and_metadata(plotting_struct, tmp_path):
    control = plotting_struct[0][0]
    assert control["body_seg_str_volume"].shape == (1, N_FRAMES)
    assert control["body_seg_str_qc"][0, :3].tolist() == ["egg", "egg", "worm"]
    assert control["Food"].shape == (1, N_FRAMES)
    assert control["experiment"].tolist() == [[str(tmp_path)]]
    assert np.isnan(control["death"]).all()
    assert not control["arrest"].any()
    assert (control["description"], control["strain"]) == ("wild type", "N2")


def test_build_plotting_struct_keeps_stored_values_at_molt(plotting_struct):
    control = plotting_struct[0][0]
    volume = control["body_seg_str_volume"][0]
    np.testing.assert_allclose(
        control["body_seg_str_volume_at_ecdysis"], [volume[MOLT_INDICES]]
    )


def test_build_plotting_struct_computes_missing_values_at_molt(plotting_struct):
    mutant = plotting_struct[0][1]
    volume = mutant["body_seg_str_volume"][0]
    values = mutant["body_seg_str_volume_at_ecdysis"][0]
    np.testing.assert_allclose(values[:4], volume[MOLT_INDICES[:4]], rtol=0.05)
    assert np.isnan(values[4])


def test_build_plotting_struct_recompute_values_at_molt(
    tmp_path, conditions_yaml, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    rows = _point_rows(0, MOLT_INDICES)
    rows = rows.with_columns(pl.lit(1.0).alias("ch2_seg_str_volume_at_M1"))
    path = tmp_path / "filemap.parquet"
    pl.concat([rows, _point_rows(1, MOLT_INDICES)]).write_parquet(path)
    conditions_struct, _ = ps.build_plotting_struct(
        str(tmp_path),
        str(path),
        conditions_yaml,
        organ_channels={"body": "ch2"},
        recompute_values_at_molt=True,
    )
    volume = conditions_struct[0]["body_seg_str_volume"][0]
    assert conditions_struct[0]["body_seg_str_volume_at_ecdysis"][
        0, 1
    ] == pytest.approx(volume[MOLT_INDICES[1]], rel=0.05)


def test_build_plotting_struct_drops_ignored_frames_and_molts(
    tmp_path, conditions_yaml, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    filemap = pl.concat([_point_rows(0, MOLT_INDICES), _point_rows(1, MOLT_INDICES)])
    ignored_time = FIRST_TIME + MOLT_INDICES[1]
    filemap = filemap.with_columns(
        ((pl.col("Point") == 0) & (pl.col("Time") == ignored_time)).alias("Ignore")
    )
    path = tmp_path / "filemap.csv"
    filemap.write_csv(path)
    conditions_struct, _ = ps.build_plotting_struct(
        str(tmp_path), str(path), conditions_yaml, organ_channels={"body": "ch2"}
    )
    control = conditions_struct[0]
    assert control["time"].shape == (1, N_FRAMES - 1)
    assert np.isnan(control["ecdysis_time_step"][0, 1])


def test_build_plotting_struct_does_not_write_to_cwd(plotting_struct, tmp_path):
    assert not (tmp_path / "test.csv").exists()


# --- combine_experiments ----------------------------------------------------------------------


@pytest.fixture
def two_experiments(tmp_path, conditions_yaml):
    paths = []
    for name, n_frames in [("a", N_FRAMES), ("b", N_FRAMES + 5)]:
        (tmp_path / name).mkdir()
        filemap = pl.concat(
            [
                _point_rows(0, MOLT_INDICES, n_frames=n_frames),
                _point_rows(1, MOLT_INDICES, n_frames=n_frames),
            ]
        )
        path = tmp_path / name / "filemap.csv"
        filemap.write_csv(path)
        paths.append(str(path))
    return paths, [conditions_yaml, conditions_yaml]


def test_combine_experiments_merges_matching_conditions(
    two_experiments, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    filemaps, configs = two_experiments
    merged = ps.combine_experiments(filemaps, configs, organ_channels={"body": "ch2"})
    assert [c["condition_id"] for c in merged] == [0, 1]
    control = merged[0]
    assert control["body_seg_str_volume"].shape == (2, N_FRAMES + 5)
    assert np.isnan(control["body_seg_str_volume"][0, N_FRAMES:]).all()
    assert control["ecdysis_index"].shape == (2, 5)
    assert control["experiment"].ravel().tolist() == [
        str(tmp_path / "a"),
        str(tmp_path / "b"),
    ]


def test_combine_experiments_uses_explicit_experiment_dirs(
    two_experiments, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    filemaps, configs = two_experiments
    merged = ps.combine_experiments(
        filemaps,
        configs,
        experiment_dirs=["exp_a", "exp_b"],
        organ_channels=[{"body": "ch2"}, {"body": "ch2"}],
    )
    assert merged[0]["experiment"].ravel().tolist() == ["exp_a", "exp_b"]


def test_combine_experiments_merges_conditions_differing_only_by_description(
    two_experiments, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    filemaps, _ = two_experiments
    config = {
        "conditions": [
            {"point_range": [[0, 0], [1, 1]], "description": ["a", "b"], "strain": "N2"}
        ]
    }
    merged = ps.combine_experiments(
        filemaps, [config, config], organ_channels={"body": "ch2"}
    )
    assert len(merged) == 1
    assert merged[0]["body_seg_str_volume"].shape == (4, N_FRAMES + 5)


def test_combine_experiments_rejects_mismatched_organ_channels(two_experiments):
    filemaps, configs = two_experiments
    with pytest.raises(ValueError):
        ps.combine_experiments(filemaps, configs, organ_channels=[{}, {}, {}])
