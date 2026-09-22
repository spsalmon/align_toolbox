import cv2
import numpy as np
import skimage.measure

from .image_quality import LAPM, LAPV, MLOG, TENG, normalized_variance_measure
from .utils import NotImplementedError


def normalize_zstack(
    zstack: np.ndarray,
    each_plane: bool = True,
    dest_dtype: np.dtype = np.uint16,  # type: ignore
) -> np.ndarray:
    """
    Normalize a z-stack of images, either plane by plane or the whole stack, to the desired data type range.

    Parameters:
        zstack (np.ndarray): The input z-stack as a 3D NumPy array, with the third dimension representing the z-plane.
        each_plane (bool, optional): Flag to determine if each z-plane should be normalized independently. (default: True)
        dest_dtype (np.dtype, optional): The desired data type for the output normalized z-stack.
                                         Can be one of the following: np.uint16, np.uint8, np.float32, np.float64. (default: np.uint16)

    Raises:
        ValueError: If dest_dtype is not one of the specified types.

    Returns:
        np.ndarray: The normalized z-stack as a 3D NumPy array.
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

    output_zstack = np.zeros_like(zstack, dtype=dest_dtype)
    if each_plane:
        output_zstack = np.array(
            [
                cv2.normalize(
                    plane,
                    None,  # type: ignore
                    0,
                    max_value,  # type: ignore
                    cv2.NORM_MINMAX,
                    dtype=dest_dtype_cv2,
                )
                for plane in zstack
            ]
        )
    else:
        output_zstack = cv2.normalize(
            zstack,
            None,  # type: ignore
            0,
            max_value,  # type: ignore
            cv2.NORM_MINMAX,
            dtype=dest_dtype_cv2,
        )
    return output_zstack


def augment_contrast_zstack(
    zstack: np.ndarray,
    clip_limit: int = 5,
    tile_size: int = 8,
    normalize_each_plane: bool = True,
) -> np.ndarray:
    """
    Augment the contrast of a z-stack of images using the CLAHE algorithm,
    with an optional initial normalization step on each z-plane.

    Parameters:
        zstack (np.ndarray): The input z-stack as a 3D NumPy array, with the third dimension representing the z-plane.
        clip_limit (int, optional): The clipping limit used in the CLAHE algorithm. Higher values increase contrast. (default: 5)
        tile_size (int, optional): The side length (in pixels) of the square tiles the image is divided into
                                   for the CLAHE algorithm. (default: 8)
        normalize_each_plane (bool, optional): Flag to determine if each z-plane should be normalized
                                               independently before contrast augmentation. (default: True)

    Returns:
        np.ndarray: The contrast-augmented z-stack as a 3D NumPy array.
    """

    zstack = normalize_zstack(zstack, normalize_each_plane, np.uint16)  # type: ignore
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    output_zstack = np.array([clahe.apply(plane) for plane in zstack.copy()])
    return output_zstack


def find_best_plane(
    zstack: np.ndarray,
    measure: str,
    channel: int | None = None,
    dest_dtype: np.dtype = np.uint16,  # type: ignore
    each_plane: bool = True,
    contrast_augmentation: bool = False,
    clip_limit: int = 2,
) -> tuple[int, np.ndarray]:
    """
    Find the best plane (z-slice) of a z-stack based on a specified measure.

    Parameters:
        zstack (np.ndarray): The input z-stack, potentially multi-channel.
        measure (str): The measure used to identify the "best" plane. Can be 'shannon_entropy', 'mean', 'normalized_variance', 'lapv', 'lapm', 'teng' or 'mlog'.
        channel (int, optional): If the z-stack has more than 3 dimensions,
                                 specifies which channel to use. (default: None)
        dest_dtype (np.dtype, optional): Desired data type after normalization. (default: np.uint16)
        each_plane (bool, optional): Flag to determine if each z-plane should be normalized independently. (default: True)
        contrast_augmentation (bool, optional): Whether to augment contrast of the z-stack or not. (default: False)
        clip_limit (int, optional): The clipping limit used for contrast augmentation
                                    if activated. (default: 2)

    Returns:
        tuple: A tuple containing:
            - int: The index of the best z-plane.
            - np.ndarray: The best z-plane image slice.

    Raises:
        ValueError: If the z-stack has more than 3 dimensions and the channel is not specified.

    Notes:
        The 'measure' determines which z-plane is considered "best". For example, if 'measure' is 'mean',
        then the z-plane with the highest mean pixel intensity is considered the best.
    """
    if zstack.ndim > 3 and channel is None:
        raise ValueError(
            "If the z-stack has more than 3 dimensions, the channel must be specified."
        )

    elif zstack.ndim > 3:
        zstack_for_measure = zstack.copy()[:, channel, ...].squeeze()
    else:
        zstack_for_measure = zstack.copy()

    if contrast_augmentation:
        zstack_for_measure = augment_contrast_zstack(
            zstack_for_measure,
            normalize_each_plane=each_plane,
            clip_limit=clip_limit,
        )

    else:
        zstack_for_measure = normalize_zstack(
            zstack_for_measure, each_plane, dest_dtype=dest_dtype
        )

    if measure == "shannon_entropy":
        measure_function = skimage.measure.shannon_entropy
    elif measure == "mean":
        measure_function = np.mean
    elif measure == "normalized_variance":
        measure_function = normalized_variance_measure
    elif measure == "lapv" or measure == "LAPV":
        measure_function = LAPV
    elif measure == "lapm" or measure == "LAPM":
        measure_function = LAPM
    elif measure == "teng" or measure == "TENG":
        measure_function = TENG
    elif measure == "mlog" or measure == "MLOG":
        measure_function = MLOG
    else:
        raise NotImplementedError(
            f"The {measure} measure is not implemented. Please choose one of the following: 'shannon_entropy', 'mean', 'normalized_variance', 'lapv', 'lapm', 'teng', 'mlog'"
        )

    best_plane_index = np.argmax(
        [measure_function(plane) for plane in zstack_for_measure]  # type: ignore
    )  # type: ignore

    return best_plane_index, zstack[best_plane_index]
