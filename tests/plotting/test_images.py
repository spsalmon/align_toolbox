import numpy as np
import polars as pl
import pytest
from tifffile import imwrite

from align_toolbox.plotting import images


def test_filter_non_worm_data():
    data = np.array([1.0, 2.0, np.nan, 4.0])
    qc = np.array(["worm", "egg", "egg", "worm"])
    np.testing.assert_array_equal(
        images.filter_non_worm_data(data, qc), [1, np.nan, np.nan, 4]
    )


def test_get_indices_from_percentages_interpolates_within_stages():
    ecdysis = np.array([[0, 10, 30, 60, 100], [5, 15, 25, 35, 45]])
    indices = images.get_indices_from_percentages(
        np.array([0, 0.125, 0.25, 0.625, 1]), ecdysis
    )
    np.testing.assert_array_equal(indices, [[0, 5, 10, 45, 100], [5, 10, 15, 30, 45]])


@pytest.fixture
def image_filemap(tmp_path):
    rows = []
    for point in range(3):
        for time in range(2):
            path = tmp_path / f"Time{time}_Point{point}.tiff"
            imwrite(path, np.full((8, 8), point, dtype=np.uint16))
            rows.append(
                {
                    "Time": time,
                    "Point": point,
                    "raw": str(path),
                    "analysis/ch1_seg": f"seg_{time}_{point}.tiff",
                    "volume": 1.0,
                }
            )
    path = tmp_path / "filemap.csv"
    pl.DataFrame(rows).write_csv(path)
    return str(path)


def test_get_condition_filemaps_images_keeps_image_columns_and_points(image_filemap):
    condition = {
        "filemap_path": np.array([[image_filemap], [image_filemap]]),
        "point": np.array([[0], [2]]),
    }
    filemaps = images.get_condition_filemaps_images(condition)
    filemap = filemaps[image_filemap]
    assert filemap.columns == ["Time", "Point", "raw", "analysis/ch1_seg"]
    assert sorted(set(filemap["Point"])) == [0, 2]


def test_get_image_paths_of_time_point(image_filemap):
    filemap = pl.read_csv(image_filemap)
    path = images.get_image_paths_of_time_point(2, 1, filemap, "raw")
    assert path.endswith("Time1_Point2.tiff")
    both = images.get_image_paths_of_time_point(
        2, 1, filemap, ["raw", "analysis/ch1_seg"]
    )
    assert both[1] == "seg_1_2.tiff"


@pytest.mark.parametrize(
    "criterion, expected_point", [("min", 0), ("max", 2), ("median", 1)]
)
def test_get_images_ecdysis_shows_the_representative_worm(
    image_filemap, criterion, expected_point, capsys, shown_figures
):
    condition = {
        "filemap_path": np.array([[image_filemap]] * 3),
        "point": np.array([[0], [1], [2]]),
        "ecdysis_time_step": np.array([[0, 1, 1, 1, 1]] * 3, dtype=float),
        "volume_at_ecdysis": np.array([[1, 1, 1, 1, 1], [2] * 5, [3] * 5], dtype=float),
    }
    images.get_images_ecdysis(
        [condition],
        "volume_at_ecdysis",
        "raw",
        criterion,
        conditions_to_plot=[0],
        molts_to_plot=["M1", "M2"],
        show_scalebar=False,
    )
    shown = [
        line for line in capsys.readouterr().out.splitlines() if "Condition 0" in line
    ]
    assert len(shown) == 2
    assert all(line.endswith(f"Time1_Point{expected_point}.tiff") for line in shown)
    assert len(shown_figures) == 2
