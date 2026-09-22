import numpy as np
import pytest

from align_toolbox.foundation import binary_image


def test_find_endpoints_of_a_line_marks_its_two_tips():
    line = np.zeros((10, 10), dtype=np.uint8)
    line[5, 2:8] = 1
    endpoints = binary_image.find_endpoints(line)
    np.testing.assert_array_equal(np.argwhere(endpoints), [[5, 2], [5, 7]])


def test_connect_endpoints_bridges_a_small_gap():
    broken = np.zeros((10, 20), dtype=np.uint8)
    broken[5, 2:8] = 1
    broken[5, 11:17] = 1
    connected = binary_image.connect_endpoints(broken, max_distance=5)
    assert (connected[5, 2:17] == 1).all()


def test_connect_endpoints_ignores_gaps_beyond_max_distance():
    broken = np.zeros((10, 20), dtype=np.uint8)
    broken[5, 2:8] = 1
    broken[5, 11:17] = 1
    connected = binary_image.connect_endpoints(broken, max_distance=3)
    assert (connected[5, 8:11] == 0).all()


def test_connect_endpoints_without_endpoints_returns_input():
    empty = np.zeros((10, 10), dtype=np.uint8)
    assert binary_image.connect_endpoints(empty) is empty


@pytest.fixture
def ring_with_two_holes():
    """A filled square with a bright hole and a dark hole inside it."""
    image = np.full((40, 40), 100.0)
    image += np.random.default_rng(0).normal(0, 5, image.shape)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5:35, 5:35] = 1
    mask[10:15, 10:15] = 0  # bright hole
    mask[25:30, 25:30] = 0  # dark hole
    image[10:15, 10:15] = 1000
    image[25:30, 25:30] = 100
    return image, mask


def test_fill_bright_holes_fills_only_bright_holes(ring_with_two_holes):
    image, mask = ring_with_two_holes
    filled = binary_image.fill_bright_holes(image, mask.copy(), scale=3)
    assert (filled[10:15, 10:15] == 1).all()
    assert (filled[25:30, 25:30] == 0).all()


def test_fill_bright_holes_without_holes_returns_mask_unchanged(rectangle_mask):
    image = np.ones(rectangle_mask.shape)
    result = binary_image.fill_bright_holes(image, rectangle_mask.copy(), scale=3)
    np.testing.assert_array_equal(result, rectangle_mask)


def test_fill_bright_holes_does_not_mutate_input(ring_with_two_holes):
    image, mask = ring_with_two_holes
    original = mask.copy()
    binary_image.fill_bright_holes(image, mask, scale=3)
    np.testing.assert_array_equal(mask, original)


def test_get_biggest_object_keeps_largest_component():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[10:18, 10:18] = 1
    biggest = binary_image.get_biggest_object(mask)
    assert biggest.dtype == np.uint8
    assert biggest.sum() == 64
    assert biggest[1, 1] == 0


def test_get_biggest_object_connectivity_matters():
    diagonal = np.zeros((6, 6), dtype=np.uint8)
    diagonal[[0, 1, 2], [0, 1, 2]] = 1
    assert binary_image.get_biggest_object(diagonal, connectivity=4).sum() == 1
    assert binary_image.get_biggest_object(diagonal, connectivity=8).sum() == 3


def test_get_biggest_object_on_empty_mask_returns_zeros():
    result = binary_image.get_biggest_object(np.zeros((5, 5), dtype=np.uint8))
    assert result.shape == (5, 5) and not result.any()
