import re
from collections.abc import Callable
from difflib import SequenceMatcher

import numpy as np


def nan_helper(
    y: np.ndarray,
) -> tuple[np.ndarray, Callable]:
    """
    Return logical indices of NaNs and a conversion function for use with np.interp.

    Parameters:
        y (np.ndarray): 1D array with possible NaN values.

    Returns:
        tuple: ``(nans, index)`` where ``nans`` is a boolean array marking NaN
            positions and ``index`` is a callable that converts a boolean index
            array to integer indices (e.g. ``index(nans)`` returns positions of NaNs).

    Example:
        >>> nans, x = nan_helper(y)
        >>> y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def inf_helper(
    y: np.ndarray,
) -> tuple[np.ndarray, Callable]:
    """
    Return logical indices of infinities and a conversion function for use with np.interp.

    Parameters:
        y (np.ndarray): 1D array with possible infinity values.

    Returns:
        tuple: ``(infs, index)`` where ``infs`` is a boolean array marking infinity
            positions and ``index`` is a callable that converts a boolean index
            array to integer indices (e.g. ``index(infs)`` returns positions of infinities).

    Example:
        >>> infs, x = inf_helper(y)
        >>> y[infs] = np.interp(x(infs), x(~infs), y[~infs])
    """

    return np.isinf(y), lambda z: z.nonzero()[0]


def interpolate_nans(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Interpolate NaN values in a given signal.

    Uses linear interpolation to estimate and replace NaN values in the provided
    signal based on the values of non-NaN neighbors.

    Parameters:
        signal (np.ndarray): The input signal array, which might contain NaN values.

    Returns:
        np.ndarray: The signal array with NaN values interpolated.
    """

    nans, x = nan_helper(signal)
    try:
        signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])
    except ValueError:
        # if all values are NaN, we cannot interpolate
        signal = np.full_like(signal, np.nan)
    return signal


def interpolate_infs(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Interpolate infinity values in a given signal.

    Uses linear interpolation to estimate and replace infinity values in the provided
    signal based on the values of non-infinity neighbors.

    Parameters:
        signal (np.ndarray): The input signal array, which might contain infinity values.

    Returns:
        np.ndarray: The signal array with infinity values interpolated.
    """

    infs, x = inf_helper(signal)
    try:
        signal[infs] = np.interp(x(infs), x(~infs), signal[~infs])
    except ValueError:
        # if all values are inf, we cannot interpolate
        signal = np.full_like(signal, np.nan)
    return signal


def interpolate_nans_infs(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Interpolate NaN and infinity values in a given signal.

    Uses linear interpolation to estimate and replace NaN and infinity values in the provided
    signal based on the values of non-NaN and non-infinity neighbors.

    Parameters:
        signal (np.ndarray): The input signal array, which might contain NaN and infinity values.

    Returns:
        np.ndarray: The signal array with NaN and infinity values interpolated.
    """

    signal = interpolate_nans(signal)
    signal = interpolate_infs(signal)
    return signal


# Exception class for the case when a method is not implemented
class NotImplementedError(Exception):
    pass


def _extract_column_components(col_name: str) -> list[str]:
    """
    Split a column name into lowercase components on ``_``, ``-``, and ``.`` separators.

    Parameters:
        col_name (str): The column name to split.

    Returns:
        list[str]: List of lowercase string components.
    """
    parts = re.split(r"[_\-\.]", col_name.lower())
    return parts


def find_best_string_match(reference: str, candidates: list[str]) -> str:
    """
    Find the best matching candidate string to a reference using component and string similarity.

    Scores each candidate by a weighted combination of component overlap (60 %) and
    sequence similarity (40 %), returning the candidate with the highest score.
    Useful for matching QC column names to data column names.

    Parameters:
        reference (str): The reference string to match against.
        candidates (list[str]): List of candidate strings to evaluate.

    Returns:
        str: The best matching candidate string.
    """
    data_parts = _extract_column_components(reference)

    best_match = None
    best_score = 0

    for candidate in candidates:
        parts = _extract_column_components(candidate)

        matching_parts = sum(
            1 for part in data_parts if part in parts and len(part) > 2
        )
        max_parts = max(len(data_parts), len(parts))

        part_score = matching_parts / max_parts if max_parts > 0 else 0

        string_score = SequenceMatcher(None, reference, candidate).ratio()

        # Weighted combination
        final_score = 0.6 * part_score + 0.4 * string_score

        if final_score > best_score:
            best_score = final_score
            best_match = candidate

    return best_match
