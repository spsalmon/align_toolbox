import polars as pl
import pytest
from polars.testing import assert_frame_equal

from align_toolbox.foundation import file_handling


def _touch(directory, *names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")


def test_extract_time_point_parses_default_pattern():
    assert file_handling.extract_time_point("/x/Time00012_Point0003.tiff") == (12, 3)


def test_extract_time_point_custom_regex():
    assert file_handling.extract_time_point("t5_p7.tiff", r"t(\d+)", r"p(\d+)") == (
        5,
        7,
    )


def test_extract_time_point_raises_without_match():
    with pytest.raises(ValueError):
        file_handling.extract_time_point("image.tiff")


def test_get_all_timepoints_from_dir_skips_unmatched_files_and_subdirs(tmp_path):
    _touch(
        tmp_path, "Time00000_Point0000.tiff", "Time00001_Point0000.tiff", "notes.txt"
    )
    (tmp_path / "Time00002_Point0000").mkdir()
    entries = file_handling.get_all_timepoints_from_dir(str(tmp_path))
    assert sorted((e["Time"], e["Point"]) for e in entries) == [(0, 0), (1, 0)]
    assert all(e["ImagePath"].startswith(str(tmp_path)) for e in entries)


def test_fill_empty_timepoints_adds_null_rows_and_sorts():
    filemap = pl.DataFrame(
        {
            "Time": [2, 0, 1, 0, 2],
            "Point": [0, 0, 0, 1, 1],
            "ImagePath": ["a2", "a0", "a1", "b0", "b2"],
        }
    )
    filled = file_handling.fill_empty_timepoints(filemap)
    expected = pl.DataFrame(
        {
            "Time": [0, 1, 2, 0, 1, 2],
            "Point": [0, 0, 0, 1, 1, 1],
            "ImagePath": ["a0", "a1", "a2", "b0", None, "b2"],
        }
    )
    assert_frame_equal(filled, expected)


def test_fill_empty_timepoints_point_with_single_timepoint():
    filemap = pl.DataFrame(
        {"Time": [0, 1, 0], "Point": [0, 0, 1], "ImagePath": ["a0", "a1", "b0"]}
    )
    filled = file_handling.fill_empty_timepoints(filemap)
    assert filled.filter(pl.col("ImagePath").is_null()).rows() == [(1, 1, None)]


def test_get_dir_filemap_fills_missing_timepoints(tmp_path):
    _touch(tmp_path, "Time00000_Point0000.tiff", "Time00001_Point0000.tiff")
    _touch(tmp_path, "Time00002_Point0000.tiff")
    _touch(tmp_path, "Time00000_Point0001.tiff", "Time00002_Point0001.tiff")
    filemap = file_handling.get_dir_filemap(str(tmp_path))
    assert filemap.shape == (6, 3)
    missing = filemap.filter(pl.col("ImagePath").is_null())
    assert missing.select("Time", "Point").rows() == [(1, 1)]


@pytest.fixture
def experiment_dir(tmp_path):
    raw_files = [f"Time0000{t}_Point0000.tiff" for t in range(3)]
    _touch(tmp_path / "raw", *raw_files)
    _touch(tmp_path / "analysis" / "ch1_seg", raw_files[0], raw_files[2])
    return tmp_path


def test_get_experiment_dir_filemap_names_raw_column(experiment_dir):
    filemap = file_handling.get_experiment_dir_filemap(str(experiment_dir))
    assert "raw" in filemap.columns


def test_get_experiment_dir_filemap_joins_analysis_subdirs(experiment_dir):
    filemap = file_handling.get_experiment_dir_filemap(str(experiment_dir))
    seg_column = str(experiment_dir / "analysis" / "ch1_seg")
    assert filemap.height == 3
    seg_paths = filemap.sort("Time")[seg_column].to_list()
    assert seg_paths[1] is None
    assert seg_paths[2].endswith("Time00002_Point0000.tiff")


def test_add_dir_to_experiment_filemap_replaces_existing_column(experiment_dir):
    filemap = pl.DataFrame(
        {"Time": [0, 1, 2], "Point": [0, 0, 0], "ch1_seg": ["stale"] * 3}
    )
    updated = file_handling.add_dir_to_experiment_filemap(
        filemap, str(experiment_dir / "analysis" / "ch1_seg"), "ch1_seg"
    )
    assert updated.columns == ["Time", "Point", "ch1_seg"]
    values = updated.sort("Time")["ch1_seg"].to_list()
    assert values[0].endswith("Time00000_Point0000.tiff")
    assert values[1] is None


def test_normalize_filemap_dtypes_heals_empty_strings_and_nans():
    filemap = pl.DataFrame(
        {
            "path": ["a", "", None],
            "volume": ["1.5", "", "2"],
            "count": [" 3", "", "4"],
            "label": ["x", "1", ""],
            "fluo": [1.0, float("nan"), 2.0],
        }
    )
    normalized = file_handling.normalize_filemap_dtypes(filemap)
    assert normalized["path"].to_list() == ["a", None, None]
    assert normalized["volume"].dtype == pl.Float64
    assert normalized["volume"].to_list() == [1.5, None, 2.0]
    assert normalized["count"].dtype == pl.Int64
    assert normalized["count"].to_list() == [3, None, 4]
    assert normalized["label"].dtype == pl.String
    assert normalized["fluo"].to_list() == [1.0, None, 2.0]


def test_normalize_filemap_dtypes_lazy_skips_type_inference():
    lazy = pl.LazyFrame({"volume": ["1.5", ""]})
    normalized = file_handling.normalize_filemap_dtypes(lazy).collect()
    assert normalized["volume"].dtype == pl.String
    assert normalized["volume"].to_list() == ["1.5", None]


@pytest.mark.parametrize("extension", [".csv", ".parquet"])
def test_write_then_read_filemap_roundtrips(tmp_path, extension):
    filemap = pl.DataFrame(
        {
            "Time": [0, 1],
            "Point": [0, 0],
            "raw": ["a.tiff", None],
            "volume": [1.5, float("nan")],
        }
    )
    path = str(tmp_path / f"filemap{extension}")
    file_handling.write_filemap(filemap, path)
    loaded = file_handling.read_filemap(path)
    assert loaded["raw"].to_list() == ["a.tiff", None]
    assert loaded["volume"].to_list() == [1.5, None]


def test_read_filemap_lazy_returns_lazyframe(tmp_path):
    path = str(tmp_path / "filemap.parquet")
    file_handling.write_filemap(pl.DataFrame({"Time": [0]}), path)
    assert isinstance(file_handling.read_filemap(path, lazy_loading=True), pl.LazyFrame)


def test_read_filemap_falls_back_to_other_extension(tmp_path):
    file_handling.write_filemap(
        pl.DataFrame({"Time": [0, 1]}), str(tmp_path / "filemap.parquet")
    )
    loaded = file_handling.read_filemap(str(tmp_path / "filemap.csv"))
    assert loaded["Time"].to_list() == [0, 1]
