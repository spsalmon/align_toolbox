import numpy as np


def compute_fluorescence_in_mask(
    image: np.ndarray,
    mask: np.ndarray,
    aggregations: list[str] = ["sum"],
    background_aggregation: str | None = None,
) -> dict[str, float]:
    """Quantify fluorescence of an image in a mask.

    Parameters:
        image (np.ndarray): The image as a NumPy array.
        mask (np.ndarray): The binary mask as a NumPy array.
        aggregations (list): The list of aggregation methods to use to quantify the fluorescence.
                             Can be one or more of 'sum', 'mean', 'median', 'max', 'min', or 'std'.

    Returns:
        dict: A dictionary with the aggregation methods as keys and the corresponding fluorescence values as values.
    """

    # compute on background substracted image if a way to compute background is provided
    if background_aggregation is not None:
        background_value = compute_background_fluorescence(
            image, mask, aggregation=background_aggregation
        )
        image = image - background_value
        image = np.clip(image, a_min=0, a_max=None)

    results = {}
    for agg in aggregations:
        if agg == "sum":
            results[agg] = np.sum(image[mask > 0])
        elif agg == "mean":
            results[agg] = np.mean(image[mask > 0])
        elif agg == "median":
            results[agg] = np.median(image[mask > 0])
        elif agg == "max":
            results[agg] = np.max(image[mask > 0])
        elif agg == "min":
            results[agg] = np.min(image[mask > 0])
        elif agg == "std":
            results[agg] = np.std(image[mask > 0])
        else:
            raise ValueError(
                'Aggregation must be one of "sum", "mean", "median", "max", "min", or "std".'
            )
    return results


def compute_background_fluorescence(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    aggregation: str = "mean",
) -> float:
    """Estimate the background value of an image.

    Parameters:
        image (np.ndarray): The image as a NumPy array.
        foreground_mask (np.ndarray): The binary mask of the foreground as a NumPy array.
        aggregation (str): The aggregation method to use to estimate the background value.
                           Can be either 'mean', 'median', or 'min'.

    Returns:
        float: The estimated background value of the image.
    """

    background_mask = np.logical_not(foreground_mask > 0)

    if aggregation == "mean":
        return np.mean(image[background_mask])
    elif aggregation == "median":
        return np.median(image[background_mask])
    elif aggregation == "min":
        return np.min(image[background_mask])
    else:
        raise ValueError('Aggregation must be one of "mean", "median", or "min".')
