from collections.abc import Callable

import cv2
import numpy as np
import skimage.exposure
import skimage.morphology
from skimage.filters import (
    scharr,
    threshold_li,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)

from align_toolbox.foundation import image_handling
from align_toolbox.foundation.binary_image import fill_bright_holes

# def old_edge_based_segmentation(
#     image: np.ndarray,
#     pixelsize: float,
#     sigma_canny: float = 1,
#     low_threshold_ratio: float = 5,
#     high_threshold_ratio: float = 2.5,
#     kernel_size: int = 5,
#     final_threshold_percentile: float = 30,
#     **kwargs,
# ) -> np.ndarray:
#     """
#     Python adaptation of the OG Matlab code for Sobel-based segmentation

#     Parameters:
#             image (np.ndarray): The input 2D grayscale image as a NumPy array.
#             pixelsize (float): Pixel size to consider when removing small objects.
#             sigma_canny (float, optional): Standard deviation for the Gaussian filter used in Canny edge detection. Default is 1.

#     Returns:
#             np.ndarray: The segmented image as a binary mask (NumPy array).

#     Raises:
#             ValueError: If the input image is not 2D.
#     """

#     if image.ndim > 2:
#         raise ValueError("Image must be 2D.")

#     thresh_otsu = threshold_otsu(image)

#     edges = skimage.feature.canny(
#         image,
#         sigma=sigma_canny,
#         low_threshold=thresh_otsu / low_threshold_ratio,
#         high_threshold=thresh_otsu / high_threshold_ratio,
#     ).astype(np.uint8)

#     edges = skimage.morphology.remove_small_objects(
#         edges.astype(bool), 3, connectivity=2
#     ).astype(np.uint8)

#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#     if len(contours) > 10000:
#         return np.zeros_like(image).astype(np.uint8)

#     edges = binary_image.connect_endpoints(edges, max_distance=200)

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
#     edges = (cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) > 0).astype(int)

#     mask = fill_bright_holes(image, edges, 10).astype(np.uint8)
#     mask = skimage.morphology.remove_small_objects(
#         mask.astype(bool), 422.5 / (pixelsize**2), connectivity=2
#     ).astype(np.uint8)

#     mask = fill_bright_holes(image, mask, 5).astype(np.uint8)

#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#     if not contours:
#         return np.zeros_like(image, dtype=np.uint8)

#     out = np.zeros_like(mask)
#     cv2.drawContours(out, contours, -1, 1, 1)

#     threshold = np.percentile(image[out > 0], final_threshold_percentile)  # type: ignore

#     final_mask = image > threshold
#     final_mask = skimage.morphology.remove_small_objects(
#         final_mask, 422.5 / (pixelsize**2), connectivity=2
#     ).astype(np.uint8)

#     final_mask = (cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel) > 0).astype(
#         np.uint8
#     )

#     final_mask = fill_bright_holes(image, final_mask, 10).astype(np.uint8)
#     return final_mask


def edge_based_segmentation(
    image: np.ndarray,
    pixelsize: float,
    gaussian_filter_sigma: float = 1,
    kernel_size: int = 30,
    final_threshold_percentile: float = 87.5,
    minimal_object_area_um2: float = 422.5,
    **kwargs,
) -> np.ndarray:
    """
    Optimized edge-based segmentation using Scharr edge detection and intensity thresholding.

    Computes Scharr edges, closes edge contours with a morphological ellipse kernel,
    then thresholds by a percentile of pixels along the closed contour boundary.

    Parameters:
        image (np.ndarray): 2D grayscale input image.
        pixelsize (float): Physical pixel size (µm/px), used to convert
            ``minimal_object_area_um2`` to pixels.
        gaussian_filter_sigma (float, optional): Sigma for the Gaussian pre-smoothing
            applied before Scharr edge detection. (default: 1)
        kernel_size (int, optional): Diameter of the elliptical structuring element
            used for morphological closing of edge contours. (default: 30)
        final_threshold_percentile (float, optional): Percentile of contour-boundary
            pixel intensities used as the final threshold. (default: 87.5)
        minimal_object_area_um2 (float, optional): Minimum object area in µm² below
            which objects are removed. (default: 422.5)
        **kwargs: Ignored (accepted for API compatibility).

    Returns:
        np.ndarray: Binary mask of shape ``(H, W)`` with dtype ``np.uint8``.

    Raises:
        ValueError: If the input image is not 2D.
    """

    if image.ndim > 2:
        raise ValueError("Image must be 2D.")

    smoothed = skimage.filters.gaussian(image, sigma=gaussian_filter_sigma)

    edge_magnitudes = scharr(smoothed)

    thresh = threshold_otsu(edge_magnitudes)

    edge_mask = (edge_magnitudes > thresh).astype(np.uint8)

    contours, _ = cv2.findContours(edge_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    out = np.zeros_like(edge_mask)
    cv2.drawContours(out, contours, -1, 1, 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    out = cv2.morphologyEx(
        out,
        cv2.MORPH_CLOSE,
        kernel,
    )

    new_contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    new_out = np.zeros_like(out)
    cv2.drawContours(new_out, new_contours, -1, 1, 1)

    threshold = np.percentile(image[new_out > 0], final_threshold_percentile)  # type: ignore

    final_mask = image > threshold
    final_mask = skimage.morphology.remove_small_objects(
        final_mask, minimal_object_area_um2 / (pixelsize**2), connectivity=2
    ).astype(np.uint8)

    final_mask = (cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel) > 0).astype(
        np.uint8
    )

    final_mask = fill_bright_holes(image, final_mask, 10).astype(np.uint8)

    return final_mask


def double_threshold_segmentation(
    image: np.ndarray,
    pixelsize: float,
    minimal_object_area_um2: float = 422.5,
) -> np.ndarray:
    """
    Segment an image using the custom iterative Otsu threshold.

    Computes a threshold with :func:`_custom_threshold_otsu` (which iteratively
    refines the Otsu threshold using Mode-Limited Mean histogram cropping) and
    removes small objects.

    Parameters:
        image (np.ndarray): 2D grayscale input image.
        pixelsize (float): Physical pixel size (µm/px), used to convert
            ``minimal_object_area_um2`` to pixels.
        minimal_object_area_um2 (float, optional): Minimum object area in µm² below
            which objects are removed. (default: 422.5)

    Returns:
        np.ndarray: Binary mask of shape ``(H, W)`` with dtype ``uint8``,
            foreground 1 and background 0.
    """
    # keep bins 2**8 even though our images are 2**16 because none of the images cover the whole dynamic range of 2**16. This will bin lower abundance signal pixels into fewer histogram bins
    mask = image > _custom_threshold_otsu(image, nbins=2**8)

    mask = skimage.morphology.remove_small_objects(
        mask.astype(bool), minimal_object_area_um2 / (pixelsize**2), connectivity=2
    )
    mask = mask.astype(np.uint8)
    return mask


def threshold_segmentation(
    image: np.ndarray,
    pixelsize: float,
    method: str = "otsu",
    minimal_object_area_um2: float = 422.5,
    **kwargs,
) -> np.ndarray:
    """
    Segment an image using a global threshold computed by a standard algorithm.

    Supported thresholding algorithms: ``"otsu"``, ``"li"``, ``"yen"``,
    ``"triangle"``.

    Parameters:
        image (np.ndarray): 2D grayscale input image.
        pixelsize (float): Physical pixel size (µm/px), used to convert
            ``minimal_object_area_um2`` to pixels.
        method (str, optional): Thresholding algorithm to use. (default: ``"otsu"``)
        minimal_object_area_um2 (float, optional): Minimum object area in µm² below
            which objects are removed. (default: 422.5)
        **kwargs: Ignored (accepted for API compatibility).

    Returns:
        np.ndarray: Binary mask of shape ``(H, W)`` with dtype ``uint8``,
            foreground 1 and background 0.

    Raises:
        ValueError: If ``method`` is not one of the supported algorithms.
    """
    if method == "otsu":
        thresh = threshold_otsu(image)
    elif method == "li":
        thresh = threshold_li(image)
    elif method == "yen":
        thresh = threshold_yen(image)
    elif method == "triangle":
        thresh = threshold_triangle(image)
    else:
        raise ValueError("Invalid thresholding method.")

    mask = image > thresh

    mask = skimage.morphology.remove_small_objects(
        mask.astype(bool), minimal_object_area_um2 / (pixelsize**2), connectivity=2
    )
    mask = mask.astype(np.uint8)
    return mask


def get_segmentation_function(
    method: str,
    pixelsize: float | None = None,
    **kwargs,
) -> Callable:
    """
    Return a segmentation callable configured for the specified method.

    Parameters:
        method (str): Segmentation method. Supported values: ``"edge_based"``,
            ``"double_threshold"``, ``"threshold"``.
        pixelsize (float, optional): Physical pixel size (µm/px). Required for
            ``"edge_based"`` and ``"double_threshold"``; optional for
            ``"threshold"``. (default: None)
        **kwargs: Additional keyword arguments forwarded to the segmentation
            function (e.g. ``gaussian_filter_sigma`` for ``"edge_based"``).

    Returns:
        callable: A function ``segment_fn(image) -> np.ndarray`` that applies the
            configured segmentation method to a 2D image.

    Raises:
        ValueError: If ``method`` is not recognized or ``pixelsize`` is ``None``
            for ``"edge_based"``.
    """
    if method == "edge_based":
        if pixelsize is None:
            raise ValueError("Pixelsize must be specified for edge-based segmentation.")

        def segment_fn(x):
            return edge_based_segmentation(x, pixelsize, **kwargs)

    elif method == "double_threshold":

        def segment_fn(x):
            return double_threshold_segmentation(x, pixelsize, **kwargs)

    elif method == "threshold":

        def segment_fn(x):
            return threshold_segmentation(x, pixelsize, **kwargs)

    else:
        raise ValueError("Invalid segmentation method.")

    return segment_fn


def segment_image(
    image: str | np.ndarray,
    method: str,
    channels: list[int] | None = None,
    pixelsize: float | None = None,
    is_stack: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Segment an image using the specified method.

    Accepts either a file path (TIFF) or a NumPy array. When ``is_stack`` is
    ``True``, the segmentation function is applied plane-by-plane along axis 0.

    Parameters:
        image (str or np.ndarray): Input image path or array. A string is
            interpreted as a path to a TIFF file.
        method (str): Segmentation method. Supported: ``"edge_based"``,
            ``"double_threshold"``, ``"threshold"``.
        channels (list[int], optional): Channel indices to keep when reading a
            multi-channel TIFF. (default: None)
        pixelsize (float, optional): Physical pixel size (µm/px). Required for
            ``"edge_based"`` and ``"double_threshold"``. (default: None)
        is_stack (bool, optional): If ``True``, iterate over planes along axis 0.
            (default: True)
        **kwargs: Additional keyword arguments forwarded to the segmentation function.

    Returns:
        np.ndarray: Binary mask as ``uint8`` array. When ``is_stack`` is ``True``,
            shape is ``(N, H, W)``; otherwise ``(H, W)``.

    Raises:
        ValueError: If ``method`` is not recognized or ``pixelsize`` is ``None``
            for ``"edge_based"``.
    """
    if isinstance(image, str):
        image = image_handling.read_tiff_file(image, channels_to_keep=channels)

    if method == "edge_based":
        if pixelsize is None:
            raise ValueError("Pixelsize must be specified for edge-based segmentation.")

        def segment_fn(x):
            return edge_based_segmentation(x, pixelsize, **kwargs)

    elif method == "double_threshold":

        def segment_fn(x):
            return double_threshold_segmentation(x, pixelsize, **kwargs)

    elif method == "threshold":

        def segment_fn(x):
            return threshold_segmentation(x, pixelsize, **kwargs)

    else:
        raise ValueError("Invalid segmentation method.")

    if is_stack:
        mask = np.zeros(
            (image.shape[0], image.shape[-2], image.shape[-1]), dtype=np.uint8
        )
        for i, plane in enumerate(image):
            mask[i] = segment_fn(plane).squeeze()
        return mask

    return segment_fn(image)


def _mode_limited_mean(
    image: np.ndarray,
    nbins: int = 2**8,
    histogram: tuple | None = None,
) -> float:
    """
    Compute the Mode-Limited Mean (MoLiM) of an image.

    Returns the mean of all pixels whose intensity is strictly above the histogram
    mode. This limits the influence of the dominant background peak and is used
    as an intermediate threshold in :func:`_custom_threshold_otsu`.

    Reference: Brocher (2014), IJIP 8, 30-48, Section 2.2.

    Parameters:
        image (np.ndarray): Input image (float in [0, 1] expected).
        nbins (int, optional): Number of histogram bins. (default: 256)
        histogram (tuple, optional): Pre-computed ``(hist, bin_centers)`` tuple as
            returned by ``skimage.exposure.histogram``; computed from ``image``
            when ``None``. (default: None)

    Returns:
        float: Mean intensity of pixels above the histogram mode.
    """
    # basic idea of the paper and MoLiM is that most thresholding algorithms work on histograms
    # if most of the image is 'background', then the histogram will be dominated by a single background peak, which will affect how the threshold is calculated
    # MoLiM is the mean of the image with some of the background already classified as background
    # this limited mean value can either itself be a threshold, or can be used as a starting point for more sophisticated thresholding algorithms
    if histogram is None:
        hist, bin_centers = skimage.exposure.histogram(image, nbins=nbins)
    else:
        hist, bin_centers = histogram

    mode = bin_centers[hist.argmax()]
    molim = np.mean(image[image > mode])
    return molim


def _custom_threshold_otsu(image: np.ndarray, nbins: int = 2**8):
    """
    Compute an Otsu threshold with iterative histogram cropping guided by Mode-Limited Mean.

    Rescales the image to [0, 1], uses :func:`_mode_limited_mean` to find an
    approximate background boundary, then iteratively crops the histogram from below
    (using Triangle thresholding) until the lower bound exceeds the MoLiM. A final
    Otsu threshold is computed on the cropped histogram and rescaled back to the
    original image dtype and value range.

    Parameters:
        image (np.ndarray): Input image (any dtype; internally rescaled to float).
        nbins (int, optional): Number of histogram bins. (default: 256)

    Returns:
        scalar: Threshold value in the original image dtype and value range.
    """
    # for consistent behaviour with regards to dtype and nbins, explicitly convert image to float
    # this is because skimage.filters.threshold_... and skimage.exposure.histogram ignore nbins in case of integer dtype
    # additionally, I think skimage trims histogram values below image.min() and above image.max() since they are zero
    # so all in all, when trying to troubleshoot threshold algorithm behaviour, and make it consistent across different channels, rescale the image as float in range (0, 1)

    # save the image range and dtype to later convert threshold back to what is expected
    image_range = image.min(), image.max()
    image_dtype = image.dtype
    image = skimage.exposure.rescale_intensity(image, out_range=(0, 1))
    # calculate histogram once and reuse in later thresholding steps
    histogram = skimage.exposure.histogram(image, nbins=nbins)
    hist_vals, bin_centers = histogram
    # limited_mean lies somewhere in the histogram peak for background
    limited_mean = _mode_limited_mean(image, histogram=histogram)
    # threshold_triangle by-and-large finds the threshold for the bottom of background peak
    # so it's a good starting point for otsu threshold, which can struggle in the case when background is so large as a proportion of the image that the overall histogram appears to be unimodal
    thresh_lower_bound = image.min()
    i = 0
    while thresh_lower_bound < limited_mean:
        # >= to allow possibility of thresh_lower_bound == limited_mean
        thresh_lower_bound = threshold_triangle(
            image[image >= thresh_lower_bound], nbins=nbins
        )
        if i > 64:
            break
        i += 1

    # don't start right at lower bound because the histogram information below thresh_lower_bound may be useful in determining a good threshold in the last iteration
    # for example, when thresh_lower_bound is already a decent threshold, in which case the the very information used to calculate it would have been discarded
    thresh = image.min()
    i = 0
    while thresh < thresh_lower_bound:
        # crop histogram to only include values above the current threshold
        hist_vals = hist_vals[bin_centers > thresh]
        bin_centers = bin_centers[bin_centers > thresh]
        thresh = threshold_otsu(hist=(hist_vals, bin_centers))

        if i > 64:
            break
        i += 1

    # rescale threshold back to original image range and dtype since threshold is in (0, 1)
    # rather than do the math ourselves, let skimage do it as it also handles float quantization
    thresh = skimage.exposure.rescale_intensity(
        np.array(thresh).reshape(1, 1), in_range=(0, 1), out_range=image_range
    ).astype(image_dtype)[0, 0]

    return thresh
