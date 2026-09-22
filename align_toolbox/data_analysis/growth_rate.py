import numpy as np
from scipy.signal import medfilt, savgol_filter

from align_toolbox.data_analysis.time_series import (
    correct_series_with_classification,
    smooth_series,
    smooth_series_classified,
)
from align_toolbox.foundation.utils import interpolate_nans_infs


def _extract_ecdysis_events(ecdysis: dict) -> list[float]:
    """
    Read the five ecdysis events from an ecdysis dict as floats.

    Accepts both the ``"HatchTime"`` key used by the filemaps and the
    ``"hatch_time"`` key returned by :func:`align_toolbox.foundation.detect_molts.find_molts`.

    Parameters:
        ecdysis (dict): The ecdysis events of the worm.

    Returns:
        list[float]: ``[hatch_time, M1, M2, M3, M4]``; missing events are NaN.

    Raises:
        KeyError: If no hatch time key is present.
    """
    if "HatchTime" in ecdysis:
        hatch_time = ecdysis["HatchTime"]
    else:
        hatch_time = ecdysis["hatch_time"]
    events = [hatch_time, ecdysis["M1"], ecdysis["M2"], ecdysis["M3"], ecdysis["M4"]]
    return [float(event) if event is not None else np.nan for event in events]


def compute_growth_rate_linear(
    series: np.ndarray,
    time: np.ndarray,
    ignore_start_fraction: float = 0.0,
    ignore_end_fraction: float = 0.0,
    savgol_filter_window: int = 5,
    savgol_filter_order: int = 3,
) -> float:
    """
    Compute the growth rate of a time series using linear regression.

    Parameters:
        series (np.ndarray): The series of values to compute the growth rate of.
        time (np.ndarray): The time data corresponding to the series.
        ignore_start_fraction (float, optional): Fraction of points to ignore at
            the start. (default: 0.0)
        ignore_end_fraction (float, optional): Fraction of points to ignore at
            the end. (default: 0.0)
        savgol_filter_window (int, optional): Window size of the Savitzky-Golay
            filter. (default: 5)
        savgol_filter_order (int, optional): Polynomial order of the
            Savitzky-Golay filter. (default: 3)

    Returns:
        float: The linear growth rate of the time series.
    """

    assert len(series) == len(time), "The series and time must have the same length."

    num_points = len(series)
    num_ignore_start = int(ignore_start_fraction * num_points)
    num_ignore_end = int(ignore_end_fraction * num_points)

    assert (
        num_ignore_start + num_ignore_end < num_points
    ), "The fraction of points to ignore is too large."

    # Correctly handle slicing when ignore fractions are 0
    if num_ignore_end == 0:
        time = time[num_ignore_start:]
        series = series[num_ignore_start:]
    else:
        time = time[num_ignore_start:-num_ignore_end]
        series = series[num_ignore_start:-num_ignore_end]

    # Remove extreme outliers with a small median filter
    series = medfilt(series, 3)

    series = interpolate_nans_infs(series)

    series = savgol_filter(series, savgol_filter_window, savgol_filter_order)

    # Compute linear regression
    slope, intercept = np.polyfit(time, series, 1)

    return slope


def compute_growth_rate_exponential(
    series: np.ndarray,
    time: np.ndarray,
    ignore_start_fraction: float = 0.0,
    ignore_end_fraction: float = 0.0,
    savgol_filter_window: int = 5,
    savgol_filter_order: int = 3,
) -> float:
    """
    Compute the growth rate of a time series using exponential regression.

    Parameters:
        series (np.ndarray): The series of values to compute the growth rate of.
        time (np.ndarray): The time data corresponding to the series.
        ignore_start_fraction (float, optional): Fraction of points to ignore at
            the start. (default: 0.0)
        ignore_end_fraction (float, optional): Fraction of points to ignore at
            the end. (default: 0.0)
        savgol_filter_window (int, optional): Window size of the Savitzky-Golay
            filter. (default: 5)
            ol_filter_order (int, optional): Polynomial order of the
            Savitzky-Golay filter. (default: 3)

    Returns:
        float: The exponential growth rate of the time series.
    """

    assert len(series) == len(time), "The series and time must have the same length."

    num_points = len(series)
    num_ignore_start = int(ignore_start_fraction * num_points)
    num_ignore_end = int(ignore_end_fraction * num_points)

    assert (
        num_ignore_start + num_ignore_end < num_points
    ), "The fraction of points to ignore is too large."

    # Correctly handle slicing when ignore fractions are 0
    if num_ignore_end == 0:
        time = time[num_ignore_start:]
        series = series[num_ignore_start:]
    else:
        time = time[num_ignore_start:-num_ignore_end]
        series = series[num_ignore_start:-num_ignore_end]

    # Remove extreme outliers with a small median filter
    series = medfilt(series, 3)

    series = interpolate_nans_infs(series)

    series = savgol_filter(series, savgol_filter_window, savgol_filter_order)

    # Compute exponential regression
    slope, intercept = np.polyfit(time, np.log(series), 1)

    return slope


def compute_growth_rate_classified(
    series: np.ndarray,
    time: np.ndarray,
    qc: np.ndarray,
    method: str = "exponential",
    ignore_start_fraction: float = 0.0,
    ignore_end_fraction: float = 0.0,
    savgol_filter_window: int = 5,
    savgol_filter_order: int = 3,
) -> float:
    """
    Compute the growth rate of a time series after correcting non-worm timepoints.

    Non-worm points (classified as ``"egg"`` or ``"error"``) are removed and
    interpolated before fitting.

    Parameters:
        series (np.ndarray): The time series of values.
        time (np.ndarray): The time data corresponding to the series.
        qc (np.ndarray): Per-point classification (``"worm"``, ``"egg"``, or
            ``"error"``).
        method (str, optional): Regression method; ``"exponential"`` or
            ``"linear"``. (default: ``"exponential"``)
        ignore_start_fraction (float, optional): Fraction of points to ignore at
            the start. (default: 0.0)
        ignore_end_fraction (float, optional): Fraction of points to ignore at
            the end. (default: 0.0)
        savgol_filter_window (int, optional): Window size of the Savitzky-Golay
            filter. (default: 5)
        savgol_filter_order (int, optional): Polynomial order of the
            Savitzky-Golay filter. (default: 3)

    Returns:
        float: The growth rate of the time series.
    """

    assert (
        len(series) == len(time) == len(qc)
    ), "The series, time, and qc must have the same length."

    series_worms = correct_series_with_classification(series, qc)

    if method == "exponential":
        growth_rate = compute_growth_rate_exponential(
            series_worms,
            time,
            ignore_start_fraction,
            ignore_end_fraction,
            savgol_filter_window,
            savgol_filter_order,
        )
    elif method == "linear":
        growth_rate = compute_growth_rate_linear(
            series_worms,
            time,
            ignore_start_fraction,
            ignore_end_fraction,
            savgol_filter_window,
            savgol_filter_order,
        )

    return growth_rate


def compute_instantaneous_growth_rate(
    series: np.ndarray,
    time: np.ndarray,
    lmbda: float = 0.0075,
    order: int = 2,
    medfilt_window: int = 5,
) -> np.ndarray:
    """
    Compute the instantaneous growth rate of a time series.

    Parameters:
        series (np.ndarray): The series of values to compute the growth rate of.
        time (np.ndarray): The time data corresponding to the series.
        lmbda (float, optional): The smoothing parameter for the Whittaker-Eilers smoothing. Default provides good results for our volume curves when series_time is in hours. (default: 0.0075)
        order (int, optional): The order of the penalty of the Whittaker-Eilers smoother. (default: 2)
        medfilt_window (int, optional): The window size for the median filter. (default: 5)

    Returns:
        np.ndarray: The instantaneous growth rate of the time series.
    """

    assert len(series) == len(time), "The series and time must have the same length."

    series = smooth_series(series, time, lmbda, order, medfilt_window)

    # Compute the instantaneous growth rate
    growth_rate = np.gradient(series, time)

    return growth_rate


def compute_instantaneous_growth_rate_classified(
    series: np.ndarray,
    time: np.ndarray,
    qc: np.ndarray,
    lmbda: float = 0.0075,
    order: int = 2,
    medfilt_window: int = 5,
) -> np.ndarray:
    """
    Compute the instantaneous growth rate of a time series after correcting the non-worm points by removing them and interpolating them back.

    Parameters:
        series (np.ndarray): The time series of values.
        time (np.ndarray): The time data corresponding to the series.
        qc (np.ndarray): The classification of the points as either 'worm' or 'egg' or 'error'.
        lmbda (float, optional): The smoothing parameter for the Whittaker-Eilers smoothing. Default provides good results for our volume curves when series_time is in hours. (default: 0.0075)
        order (int, optional): The order of the penalty of the Whittaker-Eilers smoother. (default: 2)
        medfilt_window (int, optional): The window size for the median filter. (default: 5)

    Returns:
        np.ndarray: The instantaneous growth rate of the time series.
    """

    assert (
        len(series) == len(time) == len(qc)
    ), "The series, time, and qc must have the same length."

    # Correct and smooth the time series
    series = smooth_series_classified(series, time, qc, lmbda, order, medfilt_window)
    # Compute the instantaneous growth rate
    growth_rate = np.gradient(series, time)

    return growth_rate


def compute_growth_rate_per_larval_stage(
    series: np.ndarray,
    time: np.ndarray,
    qc: np.ndarray,
    ecdysis: dict,
    method: str = "exponential",
    ignore_start_fraction: float = 0.0,
    ignore_end_fraction: float = 0.0,
    savgol_filter_window: int = 5,
    savgol_filter_order: int = 3,
) -> dict:
    """
    Compute the growth rate of a time series for each larval stage.

    Non-worm timepoints are removed and interpolated before fitting. Growth
    rates are computed for L1–L4 using the ecdysis boundary indices.

    Parameters:
        series (np.ndarray): The time series of values.
        time (np.ndarray): The time data corresponding to the series.
        qc (np.ndarray): Per-point classification (``"worm"``, ``"egg"``, or
            ``"error"``).
        ecdysis (dict): Ecdysis boundary dict with keys ``"HatchTime"``,
            ``"M1"``, ``"M2"``, ``"M3"``, ``"M4"`` as integer indices.
        method (str, optional): Regression method; ``"exponential"`` or
            ``"linear"``. (default: ``"exponential"``)
        ignore_start_fraction (float, optional): Fraction of points to ignore
            at the start of each stage. (default: 0.0)
        ignore_end_fraction (float, optional): Fraction of points to ignore at
            the end of each stage. (default: 0.0)
        savgol_filter_window (int, optional): Window size of the Savitzky-Golay
            filter. (default: 5)
        savgol_filter_order (int, optional): Polynomial order of the
            Savitzky-Golay filter. (default: 3)

    Returns:
        dict: Growth rate for each larval stage, keyed ``"L1"``–``"L4"``.
            Stages with NaN ecdysis boundaries return ``np.nan``.
    """

    assert (
        len(series) == len(time) == len(qc)
    ), "The series, time, and qc, must have the same length."

    # Correct the time series
    series_worms = correct_series_with_classification(series, qc)

    hatch_time, M1, M2, M3, M4 = _extract_ecdysis_events(ecdysis)

    growth_rates = {}

    # Compute the growth rate per larval stage
    for i, (start, end) in enumerate(zip([hatch_time, M1, M2, M3], [M1, M2, M3, M4])):
        # check if start or end is NaN
        if np.isnan(start) or np.isnan(end):
            growth_rates[f"L{i+1}"] = np.nan

        else:
            start, end = int(start), int(end)
            series_worms_stage = series_worms[start:end]
            time_stage = time[start:end]
            qc_stage = qc[start:end]

            growth_rate_stage = compute_growth_rate_classified(
                series_worms_stage,
                time_stage,
                qc_stage,
                method,
                ignore_start_fraction,
                ignore_end_fraction,
                savgol_filter_window,
                savgol_filter_order,
            )
            growth_rates[f"L{i+1}"] = growth_rate_stage

    return growth_rates


def compute_larval_stage_duration(ecdysis: dict) -> dict:
    """
    Compute the duration of each larval stage.

    Parameters:
        ecdysis (dict): The ecdysis events of the worm.

    Returns:
        dict: The duration of each larval stage.
    """

    hatch_time, M1, M2, M3, M4 = _extract_ecdysis_events(ecdysis)

    ls_durations = {}

    for i, (start, end) in enumerate(zip([hatch_time, M1, M2, M3], [M1, M2, M3, M4])):
        # check if start or end is NaN
        if np.isnan(start) or np.isnan(end):
            ls_durations[f"L{i+1}"] = np.nan
        else:
            ls_durations[f"L{i+1}"] = end - start

    return ls_durations
