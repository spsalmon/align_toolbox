import numpy as np
import pytest

from align_toolbox.quantification import (
    compute_background_fluorescence,
    compute_fluorescence_in_mask,
)


@pytest.fixture
def image_and_mask():
    image = np.full((4, 4), 10.0)
    image[1:3, 1:3] = [[20.0, 30.0], [40.0, 50.0]]
    image[0, 0] = 2.0
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    return image, mask


def test_compute_fluorescence_in_mask_all_aggregations(image_and_mask):
    image, mask = image_and_mask
    result = compute_fluorescence_in_mask(
        image, mask, aggregations=["sum", "mean", "median", "max", "min", "std"]
    )
    assert result == pytest.approx(
        {
            "sum": 140,
            "mean": 35,
            "median": 35,
            "max": 50,
            "min": 20,
            "std": np.std([20, 30, 40, 50]),
        }
    )


def test_compute_fluorescence_in_mask_rejects_unknown_aggregation(image_and_mask):
    with pytest.raises(ValueError):
        compute_fluorescence_in_mask(*image_and_mask, aggregations=["mode"])


@pytest.mark.parametrize(
    "aggregation, expected", [("mean", (11 * 10 + 2) / 12), ("median", 10), ("min", 2)]
)
def test_compute_background_fluorescence(image_and_mask, aggregation, expected):
    image, mask = image_and_mask
    assert compute_background_fluorescence(image, mask, aggregation) == pytest.approx(
        expected
    )


def test_compute_background_fluorescence_rejects_unknown_aggregation(image_and_mask):
    with pytest.raises(ValueError):
        compute_background_fluorescence(*image_and_mask, aggregation="max")


def test_background_subtraction_is_clipped_at_zero(image_and_mask):
    image, mask = image_and_mask
    result = compute_fluorescence_in_mask(
        image, mask, aggregations=["sum", "min"], background_aggregation="median"
    )
    assert result == {"sum": 100, "min": 10}
    image[1, 1] = 0.0
    result = compute_fluorescence_in_mask(
        image, mask, aggregations=["min"], background_aggregation="median"
    )
    assert result["min"] == 0
