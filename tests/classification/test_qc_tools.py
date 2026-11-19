import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
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
    features = compute_qc_features(curved_worm_mask, None, pixelsize=1.0)
    assert len(features) == 1
    assert features["area"][0] == curved_worm_mask.sum()
    assert "NORMALIZED_VARIANCE_MEASURE" not in features


def test_compute_qc_features_with_image(curved_worm_mask, curved_worm_image):
    features = compute_qc_features(
        curved_worm_mask,
        curved_worm_image,
        pixelsize=1.0,
        features=["area", "intensity_mean"],
    )
    assert list(features.columns) == [
        "area",
        "intensity_mean",
        "scharr_mask_intensity_mean",
        "scharr_image_intensity_mean",
        "NORMALIZED_VARIANCE_MEASURE",
    ]
    assert features["intensity_mean"][0] > 0.5  # worm pixels are the bright ones


def test_compute_qc_features_scharr_response_drops_with_blur(
    curved_worm_mask, curved_worm_image
):
    sharp, blurred = (
        compute_qc_features(
            curved_worm_mask, image, pixelsize=1.0, features=["intensity_mean"]
        )
        for image in (curved_worm_image, gaussian_filter(curved_worm_image, 3))
    )
    for region in ("mask", "image"):
        column = f"scharr_{region}_intensity_mean"
        assert sharp[column][0] > blurred[column][0]


def test_compute_qc_features_reads_paths_and_multichannel_images(
    tmp_path, curved_worm_mask, curved_worm_image
):
    mask_path, image_path = tmp_path / "mask.tiff", tmp_path / "image.tiff"
    imwrite(mask_path, curved_worm_mask * 255)
    imwrite(image_path, np.stack([curved_worm_image, curved_worm_image]))
    features = compute_qc_features(
        str(mask_path),
        str(image_path),
        pixelsize=1.0,
        features=["area", "intensity_mean"],
    )
    assert features["area"][0] == curved_worm_mask.sum()
    assert {"intensity_mean-0", "intensity_mean-1"} <= set(features.columns)
    assert features["scharr_image_intensity_mean-0"][0] == pytest.approx(
        features["scharr_image_intensity_mean-1"][0]
    )


def test_compute_qc_features_pads_mismatched_shapes(
    curved_worm_mask, curved_worm_image
):
    features = compute_qc_features(
        curved_worm_mask,
        curved_worm_image[:100, :300],
        pixelsize=1.0,
        features=["area"],
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
    assert compute_qc_features(mask, image, pixelsize=1.0, features=["area"]) is None


def test_compute_qc_features_rejects_4d_image_file(tmp_path):
    path = tmp_path / "zc_stack.tiff"
    imwrite(path, np.ones((2, 2, 10, 10), dtype=np.uint16))
    assert compute_qc_features(np.ones((10, 10)), str(path), pixelsize=1.0) is None


def test_compute_qc_features_scales_size_properties_by_pixelsize(rectangle_mask):
    features = compute_qc_features(
        rectangle_mask,
        None,
        pixelsize=0.5,
        features=["area", "perimeter", "eccentricity"],
    )
    reference = compute_qc_features(
        rectangle_mask,
        None,
        pixelsize=1.0,
        features=["area", "perimeter", "eccentricity"],
    )
    assert features["area"][0] == rectangle_mask.sum() * 0.25
    assert features["perimeter"][0] == reference["perimeter"][0] * 0.5
    assert features["eccentricity"][0] == reference["eccentricity"][0]


def test_compute_qc_features_keeps_hu_moments_and_order_with_pixelsize(
    curved_worm_mask, curved_worm_image
):
    features = ["moments_weighted_hu", "area", "moments_hu"]
    scaled = compute_qc_features(
        curved_worm_mask, curved_worm_image, pixelsize=0.65, features=features
    )
    reference = compute_qc_features(
        curved_worm_mask, curved_worm_image, pixelsize=1.0, features=features
    )
    assert scaled.columns[0] == "moments_weighted_hu-0"
    assert scaled.columns[-2] == "moments_hu-6"
    hu_columns = [c for c in scaled.columns if "hu" in c]
    assert len(hu_columns) == 7 + 7
    np.testing.assert_allclose(scaled[hu_columns], reference[hu_columns])


def test_compute_qc_features_without_pixelsize_keeps_pixel_units(curved_worm_mask):
    features = compute_qc_features(
        curved_worm_mask, None, features=["area", "moments_hu"]
    )
    assert features["area"][0] == curved_worm_mask.sum()
    assert "moments_hu-0" in features
