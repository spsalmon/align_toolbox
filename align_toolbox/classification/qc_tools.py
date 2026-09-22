import numpy as np
import pandas as pd
from csbdeep.utils import normalize
from skimage.measure import regionprops_table
from skimage.measure._regionprops import RegionProperties

from align_toolbox.foundation.image_handling import (
    pad_images_to_same_dim,
    read_tiff_file,
)
from align_toolbox.foundation.image_quality import normalized_variance_measure


def get_all_skimage_regionprops(mask_only: bool = False) -> list[str]:
    """
    Return all available scikit-image region-property names.

    Excludes image-array and coordinates properties. When ``mask_only`` is
    ``True``, intensity-based properties are also excluded so that the result
    can be used on masks without an intensity image.

    Parameters:
        mask_only (bool, optional): If ``True``, omit intensity-based properties
            (e.g. ``intensity_mean``, ``moments_weighted``). (default: False)

    Returns:
        list[str]: Names of the available region properties.
    """
    features = [
        attr
        for attr in dir(RegionProperties)
        if not attr.startswith("_")
        and isinstance(getattr(RegionProperties, attr), property)
        if "image" not in attr and "coords" not in attr
    ]
    # remove intensity based features if mask_only is True
    if mask_only:
        intensity_features = [
            "centroid_weighted",
            "centroid_weighted_local",
            "intensity_min",
            "intensity_max",
            "intensity_mean",
            "intensity_median",
            "intensity_std",
            "moments_weighted",
            "moments_weighted_central",
            "moments_weighted_hu",
            "moments_weighted_normalized",
        ]
        features = [f for f in features if f not in intensity_features]
    return features


def compute_qc_features(
    mask: str | np.ndarray,
    image: str | np.ndarray | None,
    features: list[str] | None = None,
    channels: list[int] | None = None,
) -> pd.DataFrame | None:
    """
    Compute quality-control features from a mask and an optional intensity image.

    Accepts file paths or NumPy arrays for both ``mask`` and ``image``. Computes
    scikit-image region properties for the largest foreground object, plus a
    normalized-variance focus measure when an intensity image is provided.

    Parameters:
        mask (str or np.ndarray): Binary mask (path to TIFF or array). Pixels > 0
            are treated as foreground.
        image (str, np.ndarray, or None): Intensity image (path to TIFF, array, or
            ``None`` for mask-only features).
        features (list[str], optional): Region-property names to compute. If
            ``None``, all properties returned by
            :func:`get_all_skimage_regionprops` are used. (default: None)
        channels (list[int], optional): Channel indices to load when ``image`` is a
            file path. (default: None)

    Returns:
        pd.DataFrame or None: DataFrame of computed features, or ``None`` if the
            mask is empty or an error occurs during extraction.
    """
    try:
        if isinstance(image, str):
            image = read_tiff_file(image, channels_to_keep=channels)
            # skimage expects the channel dimension last, in our case it's first if the image is not a stack
            if image.ndim > 3:
                raise ValueError(
                    "Input image has more than 3 dimensions, which is currently not supported."
                )
        if isinstance(mask, str):
            mask = read_tiff_file(mask)

        mask = (mask > 0).astype(np.uint8)
        if np.max(mask) == 0:
            return None

        if features is None:
            features = get_all_skimage_regionprops(
                mask_only=image is None,
            )

        if image is not None:
            # normalize image
            image = normalize(image, 1, 99, axis=None)

            if image.shape[-2:] != mask.shape[-2:]:
                image, mask = pad_images_to_same_dim(image, mask)

            if image.ndim == 3:
                image = np.transpose(image, (1, 2, 0))

            props = regionprops_table(
                mask,
                intensity_image=image,
                properties=features,
            )
            props_df = pd.DataFrame(props)
            other_features = {
                "NORMALIZED_VARIANCE_MEASURE": normalized_variance_measure(image),
            }

            other_features_df = pd.DataFrame([other_features])
            props_df = pd.concat([props_df, other_features_df], axis=1)
        else:
            props = regionprops_table(
                mask,
                properties=features,
            )
            props_df = pd.DataFrame(props)
        return props_df
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None
