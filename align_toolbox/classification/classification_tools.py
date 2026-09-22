from collections.abc import Callable

import numpy as np
import pandas as pd
import xgboost
from joblib import Parallel, delayed

from align_toolbox.foundation import worm_features


def classify_image(
    image: np.ndarray,
    features_function: Callable,
    classifier: xgboost.XGBClassifier,
    classes: list,
    **kwargs,
) -> str:
    """
    Classify an image based on features extracted by a provided function.

    Parameters:
        image (np.ndarray): The image (or array of images) to classify.
        features_function (callable): A function that extracts a feature vector from
            ``image``; called as ``features_function(image, **kwargs)``.
        classifier (xgboost.XGBClassifier): Trained XGBoost classifier.
        classes (list): Ordered list of class labels matching the classifier's output
            columns (e.g. ``["egg", "worm", "error"]``).
        **kwargs: Additional keyword arguments forwarded to ``features_function``.

    Returns:
        str: The predicted class label (element of ``classes``).
    """

    # feature extraction
    try:
        features = features_function(image, **kwargs)
    except Exception as e:
        raise Exception(f"Error extracting features from image. {e}")
    # classification
    try:
        prediction = classifier.predict_proba(features).squeeze()
    except Exception as e:
        raise Exception(f"Error predicting class of image. {e}")

    assert len(prediction) == len(
        classes
    ), f"Number of provided classes and predicted classes do not match. len(prediction) = {len(prediction)}, len(classes) = {len(classes)}"
    # convert proba to one hot encoding
    pred_class = np.argmax(prediction)
    prediction = classes[pred_class]
    return prediction


def compute_features_of_label(
    current_label: int,
    mask_plane: np.ndarray,
    image_plane: np.ndarray,
    all_features: list[str],
    extra_properties: list,
    intensity_features: list[str],
    extra_intensity_features: list,
    num_closest: int | None = None,
    patches: list[int] | None = None,
) -> list:
    """
    Compute a set of features for a single label, including context features and patch features.

    Parameters:
        current_label (int): The label of the current region.
        mask_plane (np.ndarray): The mask of all regions.
        image_plane (np.ndarray): The intensity image.
        all_features (list): The list of features to compute.
        extra_properties (list): The list of extra properties to compute.
        intensity_features (list): The list of intensity features to compute.
        extra_intensity_features (list): The list of extra intensity features to compute.
        num_closest (int): The number of closest regions to consider.
        patches (list): The list of patch sizes to consider.

    Returns:
        list: A list of features for the label.
    """

    mask_of_current_label = (mask_plane == current_label).astype("uint8")
    # check if image_plane has multiple channels
    if len(image_plane.shape) == 3:
        # compute all the features on the first channel and then intensity features on the other ones
        feature_vector = worm_features.compute_base_label_features(
            mask_of_current_label,
            image_plane[0],
            all_features,
            extra_properties,
        )
        for i in range(1, image_plane.shape[0]):
            other_channel_intensity_features = (
                worm_features.compute_base_label_features(
                    mask_of_current_label,
                    image_plane[i],
                    intensity_features,
                    extra_intensity_features,
                )
            )
            feature_vector += other_channel_intensity_features
    else:
        feature_vector = worm_features.compute_base_label_features(
            mask_of_current_label, image_plane, all_features, extra_properties
        )

    if patches is not None:
        for patch_size in patches:
            if len(image_plane.shape) == 3:
                patch_features = worm_features.compute_patch_features(
                    mask_of_current_label,
                    image_plane[0],
                    patch_size=patch_size,
                )
                for i in range(1, image_plane.shape[0]):
                    patch_features += worm_features.compute_patch_features(
                        mask_of_current_label,
                        image_plane[i],
                        patch_size=patch_size,
                    )
                feature_vector += patch_features
            else:
                patch_features = worm_features.compute_patch_features(
                    mask_of_current_label, image_plane, patch_size=patch_size
                )
                feature_vector += patch_features

    if num_closest is not None:
        context = worm_features.get_context(
            current_label,
            mask_of_current_label,
            mask_plane,
            num_closest=num_closest,
        )
        if len(image_plane.shape) == 3:
            context_features = worm_features.get_context_features(
                context, image_plane[0], all_features, extra_properties
            )
            for i in range(1, image_plane.shape[0]):
                context_features += worm_features.get_context_features(
                    context,
                    image_plane[i],
                    intensity_features,
                    extra_intensity_features,
                )
            feature_vector += context_features
        else:
            context_features = worm_features.get_context_features(
                context, image_plane, all_features, extra_properties
            )
            feature_vector += context_features

    return feature_vector


def compute_features_of_plane(
    mask_plane: np.ndarray,
    image_plane: np.ndarray,
    all_features: list[str],
    extra_properties: list,
    intensity_features: list[str],
    extra_intensity_features: list,
    num_closest: int | None = None,
    patches: list[int] | None = None,
    parallel: bool = True,
    n_jobs: int = -1,
) -> list:
    """
    Compute a set of features for a single label, including context features and patch features for all labels in a plane.

    Parameters:
        mask_plane (np.ndarray): The mask of all regions.
        image_plane (np.ndarray): The intensity image.
        all_features (list): The list of features to compute.
        extra_properties (list): The list of extra properties to compute.
        intensity_features (list): The list of intensity features to compute.
        extra_intensity_features (list): The list of extra intensity features to compute.
        num_closest (int): The number of closest regions to consider.
        patches (list): The list of patch sizes to consider.
        parallel (bool): Whether to compute features in parallel.
        n_jobs (int): The number of jobs to run in parallel.

    Returns:
        list: A list of lists of features for all labels.
    """

    if parallel:
        features_of_all_labels = Parallel(n_jobs=n_jobs)(
            delayed(compute_features_of_label)(
                current_label,
                mask_plane,
                image_plane,
                all_features,
                extra_properties,
                intensity_features,
                extra_intensity_features,
                num_closest=num_closest,
                patches=patches,
            )
            for current_label in np.unique(mask_plane)[1:]
        )
    else:
        features_of_all_labels = [
            compute_features_of_label(
                current_label,
                mask_plane,
                image_plane,
                all_features,
                extra_properties,
                intensity_features,
                extra_intensity_features,
                num_closest=num_closest,
                patches=patches,
            )
            for current_label in np.unique(mask_plane)[1:]
        ]
    return features_of_all_labels


def classify_plane(
    mask_plane: np.ndarray,
    image_plane: np.ndarray,
    classifier: xgboost.XGBClassifier,
    all_features: list[str],
    extra_properties: list,
    intensity_features: list[str],
    extra_intensity_features: list,
    num_closest: int | None = None,
    patches: list[int] | None = None,
    parallel: bool = True,
    n_jobs: int = -1,
    confidence_threshold: float | None = None,
) -> np.ndarray | None:
    """
    Compute the features of all the labels in a plane and classify them using an XGBoost classifier.

    Parameters:
        mask_plane (np.ndarray): The mask of all regions.
        image_plane (np.ndarray): The intensity image.
        classifier (xgboost.XGBClassifier): The trained classifier object.
        all_features (list): The list of features to compute.
        extra_properties (list): The list of extra properties to compute.
        intensity_features (list): The list of intensity features to compute.
        extra_intensity_features (list): The list of extra intensity features to compute.
        num_closest (int): The number of closest regions to consider.
        patches (list): The list of patch sizes to consider.
        parallel (bool): Whether to compute features in parallel.
        n_jobs (int): The number of jobs to run in parallel.
        confidence_threshold (float): The confidence threshold for predictions to be considered valid.

    Returns:
        list: A list of predicted classes for all labels in the plane.
    """

    features = compute_features_of_plane(
        mask_plane,
        image_plane,
        all_features,
        extra_properties,
        intensity_features,
        extra_intensity_features,
        num_closest=num_closest,
        patches=patches,
        parallel=parallel,
        n_jobs=n_jobs,
    )
    if len(features) == 0:
        return None
    predictions = classifier.predict_proba(features)
    predicted_classes = np.argmax(predictions, axis=1)
    if confidence_threshold is not None:
        for i in range(len(predicted_classes)):
            if np.max(predictions[i]) < confidence_threshold:
                predicted_classes[i] = -1
    return predicted_classes


def classify_labels(
    mask: np.ndarray,
    image: np.ndarray,
    classifier: xgboost.XGBClassifier,
    all_features: list[str],
    extra_properties: list,
    intensity_features: list[str],
    extra_intensity_features: list,
    num_closest: int | None = None,
    patches: list[int] | None = None,
    parallel: bool = True,
    n_jobs: int = -1,
    is_zstack: bool = False,
    confidence_threshold: float | None = None,
) -> list:
    """
    Compute the features of all the labels in a mask and classify them using an XGBoost classifier.

    Parameters:
        mask (np.ndarray): The mask of all regions.
        image (np.ndarray): The intensity image.
        classifier (xgboost.XGBClassifier): The trained classifier object.
        all_features (list): The list of features to compute.
        extra_properties (list): The list of extra properties to compute.
        intensity_features (list): The list of intensity features to compute.
        extra_intensity_features (list): The list of extra intensity features to compute.
        num_closest (int): The number of closest regions to consider.
        patches (list): The list of patch sizes to consider.
        parallel (bool): Whether to compute features in parallel.
        n_jobs (int): The number of jobs to run in parallel.
        is_zstack (bool): Whether the image is a z-stack.
        confidence_threshold (float): The confidence threshold for predictions to be considered valid.

    Returns:
        list: A list of predicted classes for all labels in the mask.
    """

    if is_zstack or len(image.shape) > 3:
        assert (
            mask.shape[0] == image.shape[0]
        ), "The number of planes in the mask and the image should be the same."
        return [
            classify_plane(
                mask_plane,
                image_plane,
                classifier,
                all_features,
                extra_properties,
                intensity_features,
                extra_intensity_features,
                num_closest=num_closest,
                patches=patches,
                parallel=parallel,
                n_jobs=n_jobs,
                confidence_threshold=confidence_threshold,
            )
            for mask_plane, image_plane in zip(mask, image)
        ]
    else:
        return classify_plane(
            mask,
            image,
            classifier,
            all_features,
            extra_properties,
            intensity_features,
            extra_intensity_features,
            num_closest=num_closest,
            patches=patches,
            parallel=parallel,
            n_jobs=n_jobs,
            confidence_threshold=confidence_threshold,
        )


def classify_labels_features_dict(
    mask: np.ndarray,
    image: np.ndarray,
    clf: xgboost.XGBClassifier,
    features_dict: dict,
    parallel: bool = True,
    n_jobs: int = -1,
    is_zstack: bool = False,
    confidence_threshold: float | None = None,
) -> list:
    """
    Classify all labels in a mask using a features dictionary.

    Convenience wrapper around :func:`classify_labels` that unpacks feature
    configuration from a dictionary rather than requiring individual arguments.

    Parameters:
        mask (np.ndarray): Labeled mask of all regions.
        image (np.ndarray): Intensity image.
        clf (xgboost.XGBClassifier): Trained XGBoost classifier.
        features_dict (dict): Dictionary with keys ``"all_features"``,
            ``"extra_properties"``, ``"intensity_features"``,
            ``"extra_intensity_features"``, ``"num_closest"``, and ``"patches"``.
        parallel (bool, optional): Whether to compute features in parallel.
            (default: True)
        n_jobs (int, optional): Number of parallel jobs (passed to joblib).
            (default: -1)
        is_zstack (bool, optional): Whether the image is a z-stack. (default: False)
        confidence_threshold (float, optional): Minimum prediction confidence;
            predictions below this threshold are set to -1. (default: None)

    Returns:
        list: Predicted class indices for all labels, structured as returned by
            :func:`classify_labels`.
    """
    return classify_labels(
        mask,
        image,
        clf,
        features_dict["all_features"],
        features_dict["extra_properties"],
        features_dict["intensity_features"],
        features_dict["extra_intensity_features"],
        num_closest=features_dict["num_closest"],
        patches=features_dict["patches"],
        parallel=parallel,
        n_jobs=n_jobs,
        is_zstack=is_zstack,
        confidence_threshold=confidence_threshold,
    )


def convert_classification_to_mask(
    mask: np.ndarray,
    classification: list,
    is_zstack: bool = False,
) -> np.ndarray:
    """
    Convert a classification (list of predicted classes) to a mask.

    Parameters:
        mask (np.ndarray): The mask of all regions.
        classification (list): The list of predicted classes for all labels.
        is_zstack (bool): Whether the image is a z-stack.

    Returns:
        np.ndarray: The given mask with pixel values replaced with class number + 1.
    """

    new_mask = np.zeros_like(mask)

    if is_zstack or len(mask.shape) > 2:
        for i, plane_classification in enumerate(classification):
            if plane_classification is not None:
                for j, label in enumerate(np.unique(mask[i])[1:]):
                    new_mask[i][mask[i] == label] = plane_classification[j] + 1
    else:
        if classification is not None:
            for i, label in enumerate(np.unique(mask)[1:]):
                new_mask[mask == label] = classification[i] + 1

    return new_mask


def convert_classification_to_dataframe(
    mask: np.ndarray,
    classification: list,
    is_zstack: bool = False,
) -> pd.DataFrame:
    """
    Convert a classification (list of predicted classes) to a pandas DataFrame.

    Parameters:
        mask (np.ndarray): The mask of all regions.
        classification (list): The list of predicted classes for all labels.
        is_zstack (bool): Whether the image is a z-stack.

    Returns:
        pd.DataFrame: A DataFrame with columns "Plane", "Label", and "Class".
    """

    data = []
    if is_zstack or len(mask.shape) > 2:
        for i, plane_classification in enumerate(classification):
            if plane_classification is not None:
                for j, label in enumerate(np.unique(mask[i])[1:]):
                    data.append(
                        {
                            "Plane": i,
                            "Label": int(label),
                            "Class": plane_classification[j],
                        }
                    )
    else:
        if classification is not None:
            for i, label in enumerate(np.unique(mask)[1:]):
                data.append(
                    {
                        "Plane": 0,
                        "Label": int(label),
                        "Class": classification[i],
                    }
                )
    return pd.DataFrame(data)


def classify_labels_and_convert_to_mask(
    mask: np.ndarray,
    image: np.ndarray,
    classifier: xgboost.XGBClassifier,
    all_features: list[str],
    extra_properties: list,
    intensity_features: list[str],
    extra_intensity_features: list,
    num_closest: int | None = None,
    patches: list[int] | None = None,
    parallel: bool = True,
    n_jobs: int = -1,
    is_zstack: bool = False,
    confidence_threshold: float | None = None,
) -> np.ndarray:
    """
    Classify all the labels in a mask using an XGBoost classifier and convert the classification to a mask.

    Parameters:
        mask (np.ndarray): The mask of all regions.
        image (np.ndarray): The intensity image.
        classifier (xgboost.XGBClassifier): The trained classifier object.
        all_features (list): The list of features to compute.
        extra_properties (list): The list of extra properties to compute.
        intensity_features (list): The list of intensity features to compute.
        extra_intensity_features (list): The list of extra intensity features to compute.
        num_closest (int): The number of closest regions to consider.
        patches (list): The list of patch sizes to consider.
        parallel (bool): Whether to compute features in parallel.
        n_jobs (int): The number of jobs to run in parallel.
        is_zstack (bool): Whether the image is a z-stack.
        confidence_threshold (float): The confidence threshold for predictions to be considered valid.

    Returns:
        np.ndarray: The given mask with pixel values replaced with class number + 1.
    """

    classification = classify_labels(
        mask,
        image,
        classifier,
        all_features,
        extra_properties,
        intensity_features,
        extra_intensity_features,
        num_closest=num_closest,
        patches=patches,
        parallel=parallel,
        n_jobs=n_jobs,
        is_zstack=is_zstack,
        confidence_threshold=confidence_threshold,
    )
    return convert_classification_to_mask(mask, classification)


def classify_labels_and_convert_to_dataframe(
    mask: np.ndarray,
    image: np.ndarray,
    classifier: xgboost.XGBClassifier,
    all_features: list[str],
    extra_properties: list,
    intensity_features: list[str],
    extra_intensity_features: list,
    num_closest: int | None = None,
    patches: list[int] | None = None,
    parallel: bool = True,
    n_jobs: int = -1,
    is_zstack: bool = False,
    confidence_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Classify all the labels in a mask using an XGBoost classifier and convert the classification to a pandas DataFrame.

    Parameters:
        mask (np.ndarray): The mask of all regions.
        image (np.ndarray): The intensity image.
        classifier (xgboost.XGBClassifier): The trained classifier object.
        all_features (list): The list of features to compute.
        extra_properties (list): The list of extra properties to compute.
        intensity_features (list): The list of intensity features to compute.
        extra_intensity_features (list): The list of extra intensity features to compute.
        num_closest (int): The number of closest regions to consider.
        patches (list): The list of patch sizes to consider.
        parallel (bool): Whether to compute features in parallel.
        n_jobs (int): The number of jobs to run in parallel.
        is_zstack (bool): Whether the image is a z-stack.
        confidence_threshold (float): The confidence threshold for predictions to be considered valid.

    Returns:
        pd.DataFrame: A DataFrame with columns "Plane", "Label", and "Class".
    """

    classification = classify_labels(
        mask,
        image,
        classifier,
        all_features,
        extra_properties,
        intensity_features,
        extra_intensity_features,
        num_closest=num_closest,
        patches=patches,
        parallel=parallel,
        n_jobs=n_jobs,
        is_zstack=is_zstack,
        confidence_threshold=confidence_threshold,
    )
    return convert_classification_to_dataframe(mask, classification)


def classify_labels_and_convert_to_mask_features_dict(
    mask: np.ndarray,
    image: np.ndarray,
    clf: xgboost.XGBClassifier,
    features_dict: dict,
    parallel: bool = True,
    n_jobs: int = -1,
    is_zstack: bool = False,
    confidence_threshold: float | None = None,
) -> np.ndarray:
    """
    Classify all labels using a features dictionary and return the result as a mask.

    Combines :func:`classify_labels_features_dict` and
    :func:`convert_classification_to_mask`.

    Parameters:
        mask (np.ndarray): Labeled mask of all regions.
        image (np.ndarray): Intensity image.
        clf (xgboost.XGBClassifier): Trained XGBoost classifier.
        features_dict (dict): Feature configuration dictionary (see
            :func:`classify_labels_features_dict`).
        parallel (bool, optional): Whether to compute features in parallel.
            (default: True)
        n_jobs (int, optional): Number of parallel jobs. (default: -1)
        is_zstack (bool, optional): Whether the image is a z-stack. (default: False)
        confidence_threshold (float, optional): Minimum prediction confidence.
            (default: None)

    Returns:
        np.ndarray: Mask with pixel values replaced by predicted class index + 1.
    """
    classification = classify_labels_features_dict(
        mask,
        image,
        clf,
        features_dict,
        parallel=parallel,
        n_jobs=n_jobs,
        is_zstack=is_zstack,
        confidence_threshold=confidence_threshold,
    )
    return convert_classification_to_mask(mask, classification, is_zstack=is_zstack)


def classify_labels_and_convert_to_dataframe_features_dict(
    mask: np.ndarray,
    image: np.ndarray,
    clf: xgboost.XGBClassifier,
    features_dict: dict,
    parallel: bool = True,
    n_jobs: int = -1,
    is_zstack: bool = False,
    confidence_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Classify all labels using a features dictionary and return the result as a DataFrame.

    Combines :func:`classify_labels_features_dict` and
    :func:`convert_classification_to_dataframe`.

    Parameters:
        mask (np.ndarray): Labeled mask of all regions.
        image (np.ndarray): Intensity image.
        clf (xgboost.XGBClassifier): Trained XGBoost classifier.
        features_dict (dict): Feature configuration dictionary (see
            :func:`classify_labels_features_dict`).
        parallel (bool, optional): Whether to compute features in parallel.
            (default: True)
        n_jobs (int, optional): Number of parallel jobs. (default: -1)
        is_zstack (bool, optional): Whether the image is a z-stack. (default: False)
        confidence_threshold (float, optional): Minimum prediction confidence.
            (default: None)

    Returns:
        pd.DataFrame: DataFrame with columns ``"Plane"``, ``"Label"``, and ``"Class"``.
    """
    classification = classify_labels_features_dict(
        mask,
        image,
        clf,
        features_dict,
        parallel=parallel,
        n_jobs=n_jobs,
        is_zstack=is_zstack,
        confidence_threshold=confidence_threshold,
    )
    return convert_classification_to_dataframe(
        mask, classification, is_zstack=is_zstack
    )
