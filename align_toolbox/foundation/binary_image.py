import cv2
import numpy as np
import scipy.ndimage
from scipy.spatial import distance


def find_endpoints(
    binary_image: np.ndarray,
) -> np.ndarray:
    """
    Find the endpoints of a binary image using the Hit-Or-Miss morphology operation.

    Parameters:
            binary_image (np.ndarray): A binary image where the foreground is represented by 1s
            and the background is represented by 0s.

    Returns:
            np.ndarray: A binary image with the same dimensions as the input, where 1s represent
            the locations of the endpoints.
    """

    # Define kernels for the Hit-Or-Miss morphology operation.
    kernel_0 = np.array(([-1, -1, -1], [-1, 1, -1], [-1, 1, -1]), dtype="int")
    kernel_1 = np.array(([-1, -1, -1], [-1, 1, -1], [1, -1, -1]), dtype="int")
    kernel_2 = np.array(([-1, -1, -1], [1, 1, -1], [-1, -1, -1]), dtype="int")
    kernel_3 = np.array(([1, -1, -1], [-1, 1, -1], [-1, -1, -1]), dtype="int")
    kernel_4 = np.array(([-1, 1, -1], [-1, 1, -1], [-1, -1, -1]), dtype="int")
    kernel_5 = np.array(([-1, -1, 1], [-1, 1, -1], [-1, -1, -1]), dtype="int")
    kernel_6 = np.array(([-1, -1, -1], [-1, 1, 1], [-1, -1, -1]), dtype="int")
    kernel_7 = np.array(([-1, -1, -1], [-1, 1, -1], [-1, -1, 1]), dtype="int")
    kernel_8 = np.array(([-1, -1, -1], [-1, 1, -1], [-1, 1, 1]), dtype="int")
    kernels = np.array(
        (
            kernel_0,
            kernel_1,
            kernel_2,
            kernel_3,
            kernel_4,
            kernel_5,
            kernel_6,
            kernel_7,
            kernel_8,
        )
    )

    output_image = np.zeros(binary_image.shape)
    # Apply the Hit-Or-Miss morphology operation for all the kernels.
    for kernel in kernels:
        out = cv2.morphologyEx(binary_image, cv2.MORPH_HITMISS, kernel)
        output_image = output_image + out

    return output_image


def connect_endpoints(
    binary_image: np.ndarray,
    max_distance: int = 200,
) -> np.ndarray:
    """
    Connect endpoints of a binary image if they are within a specified maximum distance.

    First identifies the endpoints in the binary image using a Hit-Or-Miss morphology operation.
    If there are at least two endpoints, calculates the pairwise distances between them. Endpoints that are
    within the specified maximum distance from each other are connected using a straight line.

    Parameters:
            binary_image (np.ndarray): A binary image where the foreground is represented by 1s
                                                               and the background is represented by 0s.
            max_distance (int, optional): The maximum distance between two endpoints to consider
                                                                      connecting them. (default: 200)

    Returns:
            np.ndarray: A binary image with the same dimensions as the input, where endpoints within the maximum distance are connected by straight lines.
    """

    # Find the endpoints of a binary image using the Hit-Or-Miss morphology operation.
    endpoints = find_endpoints(binary_image)
    if np.sum(endpoints) < 2:
        return binary_image
    endpoints_x, endpoints_y = np.where(endpoints)
    endpoints_coordinates = np.column_stack((endpoints_x, endpoints_y))
    distances = distance.cdist(endpoints_coordinates, endpoints_coordinates)

    output_image = binary_image.copy()
    for i, distance_for_point in enumerate(distances):
        nonzero_distances = distance_for_point[np.nonzero(distance_for_point)]
        if nonzero_distances.size == 0:
            return output_image
        if np.min(nonzero_distances) < max_distance:
            mindist_index = np.where(distance_for_point == np.min(nonzero_distances))[
                0
            ][0]
            start_point = (
                endpoints_coordinates[i][1],
                endpoints_coordinates[i][0],
            )
            end_point = (
                endpoints_coordinates[mindist_index][1],
                endpoints_coordinates[mindist_index][0],
            )
            cv2.line(output_image, start_point, end_point, 1, 1)

    return output_image


def fill_bright_holes(
    image: np.ndarray,
    mask: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Fill bright holes in an image based on a given mask and statistical properties of the background.

    Identifies holes in a provided mask and evaluates the brightness of these holes in the
    original image. Bright holes with a median brightness significantly greater than the background mean are filled.

    Parameters:
            image (np.ndarray): Grayscale input image.
            mask (np.ndarray): Binary mask with foreground equal to 1.
            scale (float): A scaling factor that defines how many standard deviations above the background mean a hole needs
                                       to be in order for it to be considered 'bright' and filled.

    Returns:
            np.ndarray: A binary mask with the same dimensions as the input, where bright holes have been filled.
    """

    mask = mask.copy()
    filled_mask = scipy.ndimage.binary_fill_holes(mask)
    holes = (filled_mask - mask).astype(np.uint8)

    # Early exit if no holes
    if not np.any(holes):
        return mask

    number_of_holes, holes_labels = cv2.connectedComponents(holes, connectivity=4)

    hole_medians = np.zeros(number_of_holes - 1)
    for label in range(1, number_of_holes):
        hole_pixels = image[holes_labels == label]
        hole_medians[label - 1] = np.median(hole_pixels)

    image_without_object = image * (1 - filled_mask)
    background_pixels = image_without_object[image_without_object > 0]

    if len(background_pixels) == 0:
        return mask

    background_thresholds = np.percentile(background_pixels, [10, 90])
    background_mask = (background_pixels >= background_thresholds[0]) & (
        background_pixels <= background_thresholds[1]
    )
    background = background_pixels[background_mask]

    background_mean = np.mean(background)
    background_std = np.std(background)
    threshold = background_mean + scale * background_std

    # Fill holes that meet the brightness criterion
    for label, median in enumerate(hole_medians, start=1):
        if median > threshold:
            mask[holes_labels == label] = 1

    return mask


def get_biggest_object(
    mask: np.ndarray,
    connectivity: int = 4,
) -> np.ndarray:
    """
    Retrieve the largest connected object from a binary mask.

    Identifies connected components in the provided binary mask
    and returns a mask of the largest object. If no objects are identified,
    returns a zero mask of the same shape as the input.

    Parameters:
            mask (np.ndarray): Binary image mask where the objects are set to 1.
            connectivity (int, optional): Connectivity for connected components. (default: 4)

    Returns:
            np.ndarray: Binary mask of the biggest object.
    """
    # Get the mask's connected components
    nb_labels, labels = cv2.connectedComponents(mask, connectivity=connectivity)

    if nb_labels >= 2:
        # Find the biggest object, ignoring the background
        biggest_object_label = np.argmax(np.bincount(labels.flatten())[1:]) + 1
        biggest_object_mask = (labels == biggest_object_label).astype(np.uint8)
    else:
        biggest_object_label = 0
        biggest_object_mask = np.zeros(mask.shape, dtype=np.uint8)

    return biggest_object_mask
