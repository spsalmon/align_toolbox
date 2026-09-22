from datetime import datetime
from itertools import combinations

import cv2
import numpy as np
import ome_types
import skimage.metrics
from tifffile import TiffFile, imread


def pad_to_dim(
    image: np.ndarray,
    xdim: int,
    ydim: int,
    pad_value: float = 0,
) -> np.ndarray:
    """
    Pad an image to target dimensions by padding on its right and bottom.

    Parameters:
        image (np.ndarray): Input image array of shape (..., H, W).
        xdim (int): Desired height of the padded image.
        ydim (int): Desired width of the padded image.
        pad_value (float): Value to use for padding (default: 0).

    Returns:
            np.ndarray: Padded image of shape (..., xdim, ydim).
    """
    xpad = xdim - image.shape[-2]
    ypad = ydim - image.shape[-1]

    pad_width = [(0, xpad), (0, ypad)]

    if image.ndim > 2:
        pad_width = [(0, 0)] * (image.ndim - 2) + pad_width

    return np.pad(image, pad_width, mode="constant", constant_values=pad_value)


def pad_to_dim_equally(
    image: np.ndarray,
    xdim: int,
    ydim: int,
    pad_value: float = 0,
) -> np.ndarray:
    """
    Efficiently pad an image equally on all sides to reach target dimensions.

    Parameters:
        image (np.ndarray): Input image array of shape (..., H, W).
        xdim (int): Desired height of the padded image.
        ydim (int): Desired width of the padded image.
        pad_value (float): Value to use for padding (default: 0).

    Returns:
        np.ndarray: Padded image of shape (..., xdim, ydim).

    Raises:
        ValueError: If target dimensions are smaller than input dimensions.
    """
    if xdim < image.shape[-2] or ydim < image.shape[-1]:
        raise ValueError("Target dimensions cannot be smaller than image dimensions")

    x_total = xdim - image.shape[-2]
    y_total = ydim - image.shape[-1]

    x_start = x_total >> 1
    y_start = y_total >> 1

    x_end = x_total - x_start
    y_end = y_total - y_start

    pad_width = [(x_start, x_end), (y_start, y_end)]

    # Add padding configuration for additional dimensions if they exist
    if image.ndim > 2:
        pad_width = [(0, 0)] * (image.ndim - 2) + pad_width

    return np.pad(image, pad_width, mode="constant", constant_values=pad_value)


def crop_to_dim(
    image: np.ndarray,
    xdim: int,
    ydim: int,
) -> np.ndarray:
    """
    Crop an image to target dimensions by removing pixels from the bottom and right

    Parameters:
        image (np.ndarray): Input image array of shape (... , H, W).
        xdim (int): Desired height of the cropped image.
        ydim (int): Desired width of the cropped image.

    Returns:
        np.ndarray: Cropped image of shape (... , xdim, ydim).
    """

    return image[..., :xdim, :ydim]


def crop_to_dim_equally(image: np.ndarray, xdim: int, ydim: int) -> np.ndarray:
    """
    Crop an image equally to the specified dimensions by removing pixels from both sides.

    Parameters:
        image (np.ndarray): Input image array of shape (... , H, W).
        xdim (int): Desired height of the cropped image.
        ydim (int): Desired width of the cropped image.

    Returns:
        np.ndarray: Cropped image of shape (... , xdim, ydim).

    Raises:
        ValueError: If target dimensions are larger than input dimensions.
    """
    if xdim > image.shape[-2] or ydim > image.shape[-1]:
        raise ValueError("Target dimensions cannot be larger than image dimensions")

    x_total = image.shape[-2] - xdim
    y_total = image.shape[-1] - ydim

    x_start = x_total >> 1
    y_start = y_total >> 1

    x_end = x_start + xdim
    y_end = y_start + ydim

    return image[..., x_start:x_end, y_start:y_end]


def crop_images_to_same_dim(
    image1: np.ndarray,
    image2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop two images to the same dimensions by taking the minimum height and width.

    Parameters:
            image1 (np.ndarray): The first input image as a NumPy array.
            image2 (np.ndarray): The second input image as a NumPy array.

    Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the cropped images.
    """
    # Determine the minimum dimensions for cropping.
    min_height = min(image1.shape[0], image2.shape[0])
    min_width = min(image1.shape[1], image2.shape[1])

    # Crop the images to the same dimensions.
    image1 = crop_to_dim_equally(image1, min_height, min_width)
    image2 = crop_to_dim_equally(image2, min_height, min_width)

    return image1, image2


def pad_images_to_same_dim(
    image1: np.ndarray,
    image2: np.ndarray,
    pad_value: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad two images to the same dimensions by taking the maximum height and width.

    Parameters:
            image1 (np.ndarray): The first input image as a NumPy array.
            image2 (np.ndarray): The second input image as a NumPy array.
            pad_value (float): The value to use for padding (default: 0).

    Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the padded images.
    """
    # Determine the maximum dimensions for padding.
    max_height = max(image1.shape[0], image2.shape[0])
    max_width = max(image1.shape[1], image2.shape[1])

    # Pad the images to the same dimensions.
    image1 = pad_to_dim_equally(image1, max_height, max_width, pad_value=pad_value)
    image2 = pad_to_dim_equally(image2, max_height, max_width, pad_value=pad_value)

    return image1, image2


def align_images_orientation_ssim(
    image: np.ndarray,
    reference_image: np.ndarray,
    axes_to_flip: list[int] = [-1, -2],
) -> np.ndarray:
    """
    Align the orientation of an image based on structural similarity index (SSIM) comparison with a reference image.
    All combination of flipped images along the chosen axis will be generated. The one with the highest SSIM to the reference image will be returned.

    Parameters:
            image (np.ndarray): The input worm image as a NumPy array.
            reference_image (np.ndarray): The reference image for comparison as a NumPy array.
            axes_to_flip (list): List of axes to flip.

    Returns:
            np.ndarray: The aligned worm image as a NumPy array.
    """
    # Pad the input and reference images to the same dimensions.
    image_pad, reference_pad = pad_images_to_same_dim(
        image.copy(), reference_image.copy()
    )

    # every subset of the axes, including the empty one (the unflipped image);
    # combinations never repeats an axis, which np.flip rejects
    axis_subsets = [
        subset
        for size in range(len(axes_to_flip) + 1)
        for subset in combinations(axes_to_flip, size)
    ]

    flipped_images = [
        image_pad if not axes else np.flip(image_pad, axis=axes)
        for axes in axis_subsets
    ]

    # Calculate the SSIM for each flipped image.
    ssim_values = []
    for flipped_image in flipped_images:
        ssim = skimage.metrics.structural_similarity(
            flipped_image,
            reference_pad,
            data_range=image_pad.max() - image_pad.min(),
        )
        ssim_values.append(ssim)

    best_index = np.argmax(ssim_values)
    return flipped_images[best_index]


def normalize_image(
    image: np.ndarray,
    dest_dtype: type = np.uint16,
) -> np.ndarray:
    """
    Normalize an image to a specified data type's range.

    Parameters:
            image (np.ndarray): The input image as a NumPy array.
            dest_dtype (type): The desired data type for the output normalized image.
                                               Allowed values are np.uint16, np.uint8, np.float32, np.float64.

    Returns:
            np.ndarray: The normalized image with the specified data type.

    Raises:
            ValueError: If dest_dtype is not one of the allowed data types.
    """
    dtype_mapping = {
        np.uint16: cv2.CV_16U,  # type: ignore
        np.uint8: cv2.CV_8U,  # type: ignore
        np.float32: cv2.CV_32F,  # type: ignore
        np.float64: cv2.CV_64F,  # type: ignore
    }

    if dest_dtype not in dtype_mapping:
        raise ValueError(
            "dest_dtype must be one of np.uint16, np.uint8, np.float32, np.float64"
        )

    dest_dtype_cv2 = dtype_mapping[dest_dtype]
    max_value = np.iinfo(dest_dtype).max if dest_dtype in [np.uint16, np.uint8] else 1

    return cv2.normalize(image, None, 0, max_value, cv2.NORM_MINMAX, dtype=dest_dtype_cv2)  # type: ignore


def augment_contrast(
    image: np.ndarray,
    clip_limit: float = 5,
    tile_size: int = 8,
) -> np.ndarray:
    """
    Augment the contrast of an image using the CLAHE (Contrast Limited Adaptive Histogram Equalization) method.

    Parameters:
            image (np.ndarray): The input image as a NumPy array.
            clip_limit (float): The clipping limit for contrast enhancement. Values above this limit get clipped.
            tile_size (int): The size of the tile grid for the CLAHE method.

    Returns:
            np.ndarray: The contrast-enhanced image as a NumPy array of dtype np.uint16.
    """
    image = normalize_image(image, np.uint16)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    image = clahe.apply(image).astype(np.uint16)
    return image


def read_tiff_file(
    file_path: str,
    channels_to_keep: list[int] | None = None,
) -> np.ndarray:
    """
    Read a TIFF file and optionally select specific channels from the image.

    Parameters:
            file_path (str): Path to the TIFF image file.
            channels_to_keep (list): List of channel indices to keep. If empty or None, all channels are kept.

    Returns:
            np.ndarray: The image data as a NumPy array. The number of dimensions may vary depending on the input and selected channels.

    Raises:
            ValueError: If the image file cannot be read.
    """
    try:
        image = imread(file_path)
    except Exception as e:
        raise ValueError(f"Error while reading image file {file_path} : {e}")

    # If no channels are specified, return the image as is.
    if image.ndim == 2 or not channels_to_keep:
        return image

    if image.ndim == 3:
        return image[channels_to_keep, ...].squeeze()  # type: ignore
    else:
        return image[:, channels_to_keep, ...].squeeze()  # type: ignore


def get_shape_from_tiff(
    file_path: str, channels_to_keep: list[int] | None = None
) -> tuple | None:
    """
    Get the shape of the image stored in the TIFF file.

    Parameters:
            file_path (str): Path to the TIFF image file.
            channels_to_keep (Optional[list[int]]): List of channel indices to keep. If None, all channels are considered.

    Returns:
            Tuple: The shape of the image as a tuple.
                   Returns None if the file cannot be read.
    Raises:
            ValueError: If the image file cannot be read or if the shape cannot be determined.
    """
    try:
        with TiffFile(file_path) as tif:
            n_series = len(tif.series)
            if n_series > 1:
                print(f"Warning: {file_path} has multiple series ({n_series})")
            shapes = [series.shape for series in tif.series]
            if len(set(shapes)) > 1:
                print(
                    f"Warning: {file_path} has different shapes across series: {shapes}"
                )

            shape = shapes[0]
            if channels_to_keep is not None and len(shape) > 2:
                if len(shape) == 3:
                    shape = (
                        (len(channels_to_keep), shape[1], shape[2])
                        if len(channels_to_keep) > 1
                        else (shape[1], shape[2])
                    )
                elif len(shape) == 4:
                    shape = (
                        (shape[0], len(channels_to_keep), shape[2], shape[3])
                        if len(channels_to_keep) > 1
                        else (shape[0], shape[2], shape[3])
                    )

            return shape
    except Exception as e:
        print(f"Error while reading file {file_path} : {e}")
        return None


def get_image_size_metadata(file_path: str) -> dict | None:
    """
    Extract and return the size metadata of an image from its OME-TIFF file.

    Includes its width (x dimension), height (y dimension), depth (z dimension),
    timepoints (t dimension), and number of channels (c dimension).

    Parameters:
        file_path (str): Path to the OME-TIFF image file.

    Returns:
        dict: A dictionary containing the dimensions of the image (x_dim, y_dim, z_dim, t_dim, c_dim).
              Returns None if the metadata cannot be read or the file is not a valid OME-TIFF.
    """
    try:
        ome_metadata = ome_types.from_tiff(file_path).images[0].pixels  # type: ignore
        xdim = ome_metadata.size_x
        ydim = ome_metadata.size_y
        zdim = ome_metadata.size_z
        tdim = ome_metadata.size_t
        cdim = ome_metadata.size_c

        return {
            "x_dim": xdim,
            "y_dim": ydim,
            "z_dim": zdim,
            "t_dim": tdim,
            "c_dim": cdim,
        }
    except Exception as e:
        print(
            f"Caught exception when trying to extract dimensions from OME-TIFF metadata in {file_path}: {e}"
        )
        return None


def check_if_stack(
    file_path: str,
    channels_to_keep: list[int] | None = None,
) -> tuple[bool, tuple[int, int]]:
    """
    Determine whether the given OME-TIFF file represents a stack (either z-stack or time series).
    Checks the z-dimension and t-dimension sizes in the OME-TIFF metadata to determine
    if the image is a stack (more than one z-plane or more than one time point).
    If the metadata cannot be read, it will try to infer the stack status from the image shape.

    Parameters:
        file_path (str): Path to the OME-TIFF image file.
        channels_to_keep (list[int], optional): Channel indices used to infer stack
            status when OME-TIFF metadata is unavailable. (default: None)

    Returns:
        tuple: ``(is_stack, (z_dim, t_dim))`` where ``is_stack`` is ``True`` when
            either z_dim > 1 or t_dim > 1, and ``(z_dim, t_dim)`` are the
            corresponding dimension sizes.
    """

    dimensions = get_image_size_metadata(file_path)

    if dimensions is None:
        # try to infer it from the image shape
        shape = get_shape_from_tiff(file_path, channels_to_keep=channels_to_keep)
        if shape is None:
            shape = read_tiff_file(file_path, channels_to_keep=channels_to_keep).shape
        if channels_to_keep is None:
            channels_to_keep = []  # ensure it's a list
        if len(shape) == 3 and (
            len(channels_to_keep) == 1 or len(channels_to_keep) == 0
        ):
            # assume (z,y,x)
            is_stack = True
            t_dim, z_dim = 1, shape[0]
        elif len(shape) == 4 and (
            len(channels_to_keep) == 1 or len(channels_to_keep) == 0
        ):
            # assume (t,z,y,x)
            is_stack = True
            t_dim, z_dim = shape[0], shape[1]
        elif len(shape) == 4 and len(channels_to_keep) > 1:
            # assume (z,c,y,x)
            is_stack = True
            t_dim, z_dim = 1, shape[0]
        elif len(shape) > 4 and len(channels_to_keep) > 1:
            # assume (t,z,c,y,x)
            is_stack = True
            t_dim, z_dim = shape[0], shape[1]
        else:
            is_stack = False
            t_dim, z_dim = 1, 1  # assume single image
    else:
        t_dim, z_dim = dimensions.get("t_dim", 1), dimensions.get("z_dim", 1)
        is_stack = (t_dim > 1) or (z_dim > 1)

    return is_stack, (z_dim, t_dim)


def check_if_zstack(file_path: str) -> bool:
    """
    Determine whether the given OME-TIFF file represents a z-stack.

    Checks the z-dimension size in the OME-TIFF metadata to determine
    if the image is a z-stack (more than one z-plane).

    Returns False if the file is not a valid OME-TIFF or the metadata cannot be read.

    Parameters:
        file_path (str): Path to the OME-TIFF image file.

    Returns:
        bool: True if the image is a z-stack (z_dim > 1), False otherwise or if the file is not a valid OME-TIFF or the metadata cannot be read.
    """
    try:
        _, (z_dim, _) = check_if_stack(file_path)
        return z_dim > 1
    except Exception as e:
        print(
            f"Caught exception when trying to check if given file is a z-stack using OME-TIFF metadata in {file_path}: {e}"
        )
        return False


def check_if_time_series(file_path: str) -> bool:
    """
    Determine whether the given OME-TIFF file represents a time series.

    Checks the t-dimension size in the OME-TIFF metadata to determine
    if the image is a time series (more than one time point).

    Returns False if the file is not a valid OME-TIFF or the metadata cannot be read.

    Parameters:
        file_path (str): Path to the OME-TIFF image file.

    Returns:
        bool: True if the image is a time series (t_dim > 1), False otherwise or if the file is not a valid OME-TIFF or the metadata cannot be read.
    """
    try:
        _, (_, t_dim) = check_if_stack(file_path)
        return t_dim > 1
    except Exception as e:
        print(
            f"Caught exception when trying to check if given file is a timeseries using OME-TIFF metadata in {file_path}: {e}"
        )
        return False


def get_acquisition_date(file_path: str) -> datetime | None:
    """
    Extract the acquisition date from the OME-TIFF metadata of the given file.

    Returns None if the metadata cannot be read or the file is not a valid OME-TIFF.

    Parameters:
        file_path (str): Path to the OME-TIFF image file.

    Returns:
        Optional[datetime]: The acquisition date of the image, or None if the metadata cannot be read or the file is not a valid OME-TIFF.

    """
    try:
        ome_metadata = ome_types.from_tiff(file_path)
        return ome_metadata.images[0].acquisition_date
    except Exception as e:
        print(f"Error extracting date using OME-TIFF metadata from {file_path}: {e}")
        return None
