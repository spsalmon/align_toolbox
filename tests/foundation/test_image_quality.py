import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from align_toolbox.foundation import image_quality

FOCUS_MEASURES = [
    image_quality.LAPV,
    image_quality.LAPM,
    image_quality.TENG,
    image_quality.MLOG,
    image_quality.TENG_VARIANCE,
]


@pytest.fixture
def sharp_and_blurred(rng):
    sharp = (rng.random((64, 64)) > 0.5).astype(np.float64) * 255
    blurred = gaussian_filter(sharp, 3)
    return sharp, blurred


@pytest.mark.parametrize("measure", FOCUS_MEASURES, ids=lambda f: f.__name__)
def test_focus_measures_rank_sharp_image_above_blurred(measure, sharp_and_blurred):
    sharp, blurred = sharp_and_blurred
    assert measure(sharp) > measure(blurred)


@pytest.mark.parametrize("measure", FOCUS_MEASURES, ids=lambda f: f.__name__)
def test_focus_measures_are_zero_on_flat_image(measure):
    assert measure(np.full((32, 32), 7.0)) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "measure", [image_quality.TENG, image_quality.TENG_VARIANCE], ids=["TENG", "TENGV"]
)
def test_tenengrad_is_rotation_invariant(measure):
    gradient_along_x = np.tile(np.arange(32, dtype=float), (32, 1))
    assert measure(gradient_along_x) == pytest.approx(measure(gradient_along_x.T))


def test_normalized_variance_is_variance_over_mean():
    image = np.array([[1.0, 3.0], [5.0, 7.0]])
    assert image_quality.normalized_variance_measure(image) == pytest.approx(5 / 4)
