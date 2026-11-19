import numpy as np
import pandas as pd
from csbdeep.utils import normalize
from skimage.filters import scharr
from skimage.measure import regionprops_table
from skimage.measure._regionprops import RegionProperties

from align_toolbox.foundation.image_handling import (
    pad_images_to_same_dim,
    read_tiff_file,
)
from align_toolbox.foundation.image_quality import normalized_variance_measure

# Hu moments are scale-invariant, and scikit-image refuses to compute them with a
# non-unit spacing.
SCALE_INVARIANT_REGIONPROPS = ["moments_hu", "moments_weighted_hu"]


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
    pixelsize: float | None = None,
    features: list[str] | None = None,
    channels: list[int] | None = None,
) -> pd.DataFrame | None:
    """
    Compute quality-control features from a mask and an optional intensity image.

    Accepts file paths or NumPy arrays for both ``mask`` and ``image``. Computes
    scikit-image region properties for the largest foreground object. When an
    intensity image is provided, also computes a normalized-variance focus measure
    and the requested ``intensity_*`` properties on the Scharr edge magnitude of
    each channel, which drops as the image gets blurrier, both inside the mask
    (prefixed ``scharr_mask_``) and over the whole image (``scharr_image_``). Size
    properties are expressed in µm (lengths), µm² (areas) and so on, so that
    features from microscopes with different pixel sizes are comparable.

    Parameters:
        mask (str or np.ndarray): Binary mask (path to TIFF or array). Pixels > 0
            are treated as foreground.
        image (str, np.ndarray, or None): Intensity image (path to TIFF, array, or
            ``None`` for mask-only features).
        pixelsize (float, optional): The size of a pixel in µm, used to scale size
            properties. If ``None``, size properties are left in pixels, as expected
            by QC models trained before pixel sizes were taken into account.
            (default: None)
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

            # per channel, as scharr would otherwise also smooth across channels
            edges = np.stack(
                [scharr(plane) for plane in image.reshape(-1, *image.shape[-2:])]
            )
            edges = edges.reshape(image.shape)

            if image.ndim == 3:
                image = np.transpose(image, (1, 2, 0))
                edges = np.transpose(edges, (1, 2, 0))

            props_df = _compute_regionprops(mask, image, pixelsize, features)
            edge_features = [f for f in features if f.startswith("intensity_")]
            edge_props_df = pd.concat(
                [
                    _compute_regionprops(
                        region, edges, pixelsize, edge_features
                    ).add_prefix(prefix)
                    for region, prefix in (
                        (mask, "scharr_mask_"),
                        (np.ones_like(mask), "scharr_image_"),
                    )
                ],
                axis=1,
            )
            other_features = {
                "NORMALIZED_VARIANCE_MEASURE": normalized_variance_measure(image),
            }

            other_features_df = pd.DataFrame([other_features])
            props_df = pd.concat([props_df, edge_props_df, other_features_df], axis=1)
        else:
            props_df = _compute_regionprops(mask, None, pixelsize, features)
        return props_df
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None


def _compute_regionprops(
    mask: np.ndarray,
    image: np.ndarray | None,
    pixelsize: float | None,
    features: list[str],
) -> pd.DataFrame:
    if pixelsize is None:
        pixelsize = 1.0
    scaled = [f for f in features if f not in SCALE_INVARIANT_REGIONPROPS]
    unscaled = [f for f in features if f in SCALE_INVARIANT_REGIONPROPS]
    props = regionprops_table(
        mask,
        intensity_image=image,
        properties=scaled,
        spacing=(pixelsize, pixelsize),
    )
    props.update(regionprops_table(mask, intensity_image=image, properties=unscaled))
    props_df = pd.DataFrame(props)

    # restore the requested feature order, each property may span several columns
    columns = [
        column
        for feature in features
        for column in props_df.columns
        if column == feature or column.startswith(f"{feature}-")
    ]
    return props_df[columns]
