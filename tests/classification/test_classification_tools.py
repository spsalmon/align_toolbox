import joblib
import numpy as np
import pytest
import xgboost

from align_toolbox.classification import classification_tools as ct
from align_toolbox.foundation.worm_features import intensity_std

FEATURES_DICT = {
    "all_features": ["area", "intensity_mean"],
    "extra_properties": [intensity_std],
    "intensity_features": ["intensity_mean"],
    "extra_intensity_features": [],
    "num_closest": None,
    "patches": None,
}
BASE_ARGS = [
    FEATURES_DICT["all_features"],
    FEATURES_DICT["extra_properties"],
    FEATURES_DICT["intensity_features"],
    FEATURES_DICT["extra_intensity_features"],
]


@pytest.fixture
def labeled_plane():
    """Four 5x5 nuclei on a 40x60 plane: labels 1, 3 are dim, labels 2, 4 bright."""
    mask = np.zeros((40, 60), dtype=np.uint8)
    image = np.zeros((40, 60))
    for label, (row, col) in enumerate([(5, 5), (5, 30), (25, 5), (25, 30)], start=1):
        mask[row : row + 5, col : col + 5] = label
        image[row : row + 5, col : col + 5] = 100 if label % 2 == 0 else 10
    return mask, image


@pytest.fixture(scope="module")
def brightness_classifier():
    """XGBoost model on [area, intensity_mean, intensity_std]: class 1 if bright."""
    rng = np.random.default_rng(0)
    means = np.concatenate([rng.uniform(0, 30, 50), rng.uniform(70, 130, 50)])
    features = np.column_stack([np.full(100, 25), means, np.zeros(100)])
    labels = (means > 50).astype(int)
    classifier = xgboost.XGBClassifier(n_estimators=10, max_depth=2)
    classifier.fit(features, labels)
    return classifier


def test_compute_features_of_label_single_channel(labeled_plane):
    mask, image = labeled_plane
    features = ct.compute_features_of_label(2, mask, image, *BASE_ARGS)
    assert features == [25, 100, 0]


def test_compute_features_of_label_extra_channels_add_intensity_features(labeled_plane):
    mask, image = labeled_plane
    two_channels = np.stack([image, image * 2])
    features = ct.compute_features_of_label(2, mask, two_channels, *BASE_ARGS)
    assert features == [25, 100, 0, 200]


@pytest.mark.parametrize(
    "channels, num_closest, patches, expected_length",
    [
        (1, 2, None, 3 + 2 * 3),
        (1, None, [8, 16], 3 + 2 * 12),
        (2, 2, [8], 4 + 2 * 3 + 2 * 1 + 2 * 12),
    ],
)
def test_compute_features_of_label_context_and_patch_lengths(
    labeled_plane, channels, num_closest, patches, expected_length
):
    mask, image = labeled_plane
    # patch texture features go through img_as_ubyte, so use a microscope dtype
    image = image.astype(np.uint16)
    if channels == 2:
        image = np.stack([image, image])
    features = ct.compute_features_of_label(
        1, mask, image, *BASE_ARGS, num_closest=num_closest, patches=patches
    )
    assert len(features) == expected_length


def test_compute_features_of_plane_parallel_matches_sequential(labeled_plane):
    mask, image = labeled_plane
    sequential = ct.compute_features_of_plane(mask, image, *BASE_ARGS, parallel=False)
    # threads instead of loky: the assertion is about dispatch, and spawning
    # interpreters that re-import the package costs minutes
    with joblib.parallel_config(backend="threading"):
        parallel = ct.compute_features_of_plane(
            mask, image, *BASE_ARGS, parallel=True, n_jobs=2
        )
    assert len(sequential) == 4
    assert sequential == parallel


def test_classify_plane_predicts_per_label(labeled_plane, brightness_classifier):
    mask, image = labeled_plane
    predictions = ct.classify_plane(
        mask, image, brightness_classifier, *BASE_ARGS, parallel=False
    )
    np.testing.assert_array_equal(predictions, [0, 1, 0, 1])


def test_classify_plane_low_confidence_predictions_become_minus_one(
    labeled_plane, brightness_classifier
):
    mask, image = labeled_plane
    predictions = ct.classify_plane(
        mask,
        image,
        brightness_classifier,
        *BASE_ARGS,
        parallel=False,
        confidence_threshold=1.01,
    )
    np.testing.assert_array_equal(predictions, [-1, -1, -1, -1])


def test_classify_plane_without_labels_returns_none(brightness_classifier):
    empty = np.zeros((10, 10), dtype=np.uint8)
    assert ct.classify_plane(empty, empty, brightness_classifier, *BASE_ARGS) is None


def test_classify_labels_zstack_classifies_each_plane(
    labeled_plane, brightness_classifier
):
    mask, image = labeled_plane
    mask_stack = np.stack([mask, np.zeros_like(mask)])
    image_stack = np.stack([image, image])
    predictions = ct.classify_labels(
        mask_stack,
        image_stack,
        brightness_classifier,
        *BASE_ARGS,
        parallel=False,
        is_zstack=True,
    )
    np.testing.assert_array_equal(predictions[0], [0, 1, 0, 1])
    assert predictions[1] is None


def test_features_dict_wrappers_match_explicit_arguments(
    labeled_plane, brightness_classifier
):
    mask, image = labeled_plane
    explicit = ct.classify_labels_and_convert_to_dataframe(
        mask, image, brightness_classifier, *BASE_ARGS, parallel=False
    )
    from_dict = ct.classify_labels_and_convert_to_dataframe_features_dict(
        mask, image, brightness_classifier, FEATURES_DICT, parallel=False
    )
    assert explicit.equals(from_dict)
    assert explicit.to_dict("list") == {
        "Plane": [0, 0, 0, 0],
        "Label": [1, 2, 3, 4],
        "Class": [0, 1, 0, 1],
    }


def test_classify_labels_and_convert_to_mask(labeled_plane, brightness_classifier):
    mask, image = labeled_plane
    explicit = ct.classify_labels_and_convert_to_mask(
        mask, image, brightness_classifier, *BASE_ARGS, parallel=False
    )
    from_dict = ct.classify_labels_and_convert_to_mask_features_dict(
        mask, image, brightness_classifier, FEATURES_DICT, parallel=False
    )
    np.testing.assert_array_equal(explicit, from_dict)
    # class index + 1: dim nuclei become 1, bright nuclei become 2
    assert explicit[7, 7] == 1 and explicit[7, 32] == 2
    assert explicit[0, 0] == 0


def test_convert_classification_to_mask_and_dataframe_for_stacks():
    mask = np.zeros((3, 4, 4), dtype=np.uint8)
    mask[0, 0, 0], mask[0, 3, 3] = 1, 5
    mask[2, 1, 1] = 2
    classification = [np.array([0, 2]), None, np.array([1])]
    converted = ct.convert_classification_to_mask(mask, classification)
    assert (converted[0, 0, 0], converted[0, 3, 3], converted[2, 1, 1]) == (1, 3, 2)
    assert not converted[1].any()
    dataframe = ct.convert_classification_to_dataframe(mask, classification)
    assert dataframe.to_dict("list") == {
        "Plane": [0, 0, 2],
        "Label": [1, 5, 2],
        "Class": [0, 2, 1],
    }


def test_convert_classification_none_for_2d_mask_gives_empty_outputs():
    mask = np.ones((3, 3), dtype=np.uint8)
    assert not ct.convert_classification_to_mask(mask, None).any()
    assert ct.convert_classification_to_dataframe(mask, None).empty


def test_classify_image_returns_class_name(brightness_classifier):
    prediction = ct.classify_image(
        np.full((5, 5), 100.0),
        lambda image: np.array([[25, image.mean(), image.std()]]),
        brightness_classifier,
        classes=["dim", "bright"],
    )
    assert prediction == "bright"


def test_classify_image_class_count_mismatch_raises(brightness_classifier):
    with pytest.raises(AssertionError):
        ct.classify_image(
            np.zeros((5, 5)),
            lambda image: np.array([[25, 0, 0]]),
            brightness_classifier,
            classes=["a", "b", "c"],
        )


def test_classify_image_wraps_feature_errors(brightness_classifier):
    def broken_features(image):
        raise RuntimeError("boom")

    with pytest.raises(Exception, match="Error extracting features.*boom"):
        ct.classify_image(np.zeros((5, 5)), broken_features, brightness_classifier, [])
