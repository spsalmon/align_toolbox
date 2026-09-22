import numpy as np
import pytest
from tifffile import imwrite

from align_toolbox.classification import compute_qc_features
from align_toolbox.classification.qc_tools import get_all_skimage_regionprops


def test_get_all_skimage_regionprops_mask_only_drops_intensity_properties():
    all_props = get_all_skimage_regionprops()
    mask_props = get_all_skimage_regionprops(mask_only=True)
    assert "intensity_mean" in all_props
    assert "intensity_mean" not in mask_props
    assert "area" in mask_props
    assert not any("image" in p or "coords" in p for p in all_props)


def test_compute_qc_features_mask_only_uses_all_mask_properties(curved_worm_mask):
    features = compute_qc_features(curved_worm_mask, None)
    assert len(features) == 1
    assert features["area"][0] == curved_worm_mask.sum()
    assert "NORMALIZED_VARIANCE_MEASURE" not in features


def test_compute_qc_features_with_image(curved_worm_mask, curved_worm_image):
    features = compute_qc_features(
        curved_worm_mask, curved_worm_image, features=["area", "intensity_mean"]
    )
    assert list(features.columns) == [
        "area",
        "intensity_mean",
        "NORMALIZED_VARIANCE_MEASURE",
    ]
    assert features["intensity_mean"][0] > 0.5  # worm pixels are the bright ones


def test_compute_qc_features_reads_paths_and_multichannel_images(
    tmp_path, curved_worm_mask, curved_worm_image
):
    mask_path, image_path = tmp_path / "mask.tiff", tmp_path / "image.tiff"
    imwrite(mask_path, curved_worm_mask * 255)
    imwrite(image_path, np.stack([curved_worm_image, curved_worm_image]))
    features = compute_qc_features(
        str(mask_path), str(image_path), features=["area", "intensity_mean"]
    )
    assert features["area"][0] == curved_worm_mask.sum()
    assert {"intensity_mean-0", "intensity_mean-1"} <= set(features.columns)


def test_compute_qc_features_pads_mismatched_shapes(
    curved_worm_mask, curved_worm_image
):
    features = compute_qc_features(
        curved_worm_mask, curved_worm_image[:100, :300], features=["area"]
    )
    assert features["area"][0] == curved_worm_mask.sum()


@pytest.mark.parametrize(
    "mask, image",
    [
        (np.zeros((10, 10)), None),
        (np.ones((10, 10)), np.ones((2, 2, 10, 10))),
    ],
    ids=["empty-mask", "4d-image"],
)
def test_compute_qc_features_returns_none_on_bad_input(mask, image):
    assert compute_qc_features(mask, image, features=["area"]) is None


def test_compute_qc_features_rejects_4d_image_file(tmp_path):
    path = tmp_path / "zc_stack.tiff"
    imwrite(path, np.ones((2, 2, 10, 10), dtype=np.uint16))
    assert compute_qc_features(np.ones((10, 10)), str(path)) is None
