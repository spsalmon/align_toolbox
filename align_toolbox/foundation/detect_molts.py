import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, medfilt, savgol_filter
from scipy.stats import linregress

from .utils import interpolate_nans


def interpolate_peaks(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Interpolate over prominent peaks in a given signal.

    Identifies prominent peaks in the signal and replaces their values with NaN.
    The NaN values are then interpolated to provide a smoother representation.
    If there's a ValueError, likely due to no peaks being found, the original signal is returned.

    Parameters:
        signal (np.ndarray): The input signal array.

    Returns:
        np.ndarray: The signal with interpolated values over detected peaks.
    """

    try:
        peaks = find_peaks(signal, prominence=0.1, wlen=7, distance=1)[0]
        interpolated_signal = signal.copy()
        interpolated_signal[peaks] = np.nan
        interpolated_signal = interpolate_nans(interpolated_signal)
        return interpolated_signal
    except ValueError:
        return signal


def find_mid_molts(
    volume: np.ndarray,
    molt_size_range: np.ndarray,
) -> np.ndarray:
    """
    Identify mid-molts in a given volume time series based on expected sizes at molt.

    Processes the volume data by performing logarithmic transformations,
    smoothing, and calculating the derivative. Peaks in the derivative represent potential
    mid-molts. The best mid-molts are then selected based on closeness to expected molt sizes
    and linear regression slopes.

    Parameters:
        volume (np.ndarray): A time series representing volume.
        molt_size_range (np.ndarray): An array of expected sizes at molt.

    Returns:
        np.ndarray: Indices of the identified mid-moults in the input volume.
    """

    log_moult_size_range = np.log(molt_size_range)
    # Smooth volume time series
    log_volume = np.log(volume.astype(float))
    log_volume = interpolate_peaks(log_volume)
    medfilt_log_volume = medfilt(log_volume, 3)

    try:
        savgol_log_volume = savgol_filter(medfilt_log_volume, 9, 3)
    except Exception as e:
        print(f"Caught an exception while running SavGol filter: {e}")
        savgol_log_volume = medfilt_log_volume

    log_diff = np.diff(savgol_log_volume)
    # Smooth derivative
    smoothed_log_diff = uniform_filter1d(uniform_filter1d(log_diff, size=25), size=10)

    # Find peaks in the smoothed derivative
    linear_regression_fit_range = 5
    peaks = find_peaks(-smoothed_log_diff, prominence=1e-5)[0]  # type: ignore
    peaks = peaks[peaks > linear_regression_fit_range]

    log_volume_at_peaks = savgol_log_volume[peaks]
    selected_peaks = np.full((4,), np.nan)

    # Find the best peak for each molt
    for i, log_moult_size in enumerate(log_moult_size_range):
        # Find peaks that are close to the expected size
        possible_peaks = peaks[abs(log_volume_at_peaks - log_moult_size) < np.log(1.5)]
        slopes = np.empty((len(possible_peaks),))

        # Find the best one by computing the slope of the linear regression
        if len(possible_peaks) > 0:
            for j, peak in enumerate(possible_peaks):
                fit_range = np.arange(
                    peak - linear_regression_fit_range,
                    peak + linear_regression_fit_range + 1,
                )
                fit_range = fit_range[fit_range > 1]
                fit_range = fit_range[fit_range < len(savgol_log_volume)]

                p = linregress(
                    fit_range[np.isfinite(savgol_log_volume[fit_range])],
                    medfilt_log_volume[
                        fit_range[np.isfinite(savgol_log_volume[fit_range])]
                    ],
                )
                slopes[j] = p.slope  # type: ignore

            # Select the peak with the smallest slope
            best_peak = possible_peaks[np.argmin(slopes)]
            selected_peaks[i] = best_peak

    # Remove peaks that are too close to each other
    for i in range(3, 0, -1):
        if np.isfinite(selected_peaks[i]) and np.isfinite(selected_peaks[i - 1]):
            if (
                selected_peaks[i] - selected_peaks[i - 1] < 6
                or selected_peaks[i - 1] > selected_peaks[i]
            ):
                selected_peaks[i - 1] = np.nan

    midmoults = selected_peaks

    return midmoults


def find_end_molts(
    volume: np.ndarray,
    midmolts: np.ndarray,
    search_width: int = 20,
    fit_width: int = 5,
) -> np.ndarray:
    """
    Identify the end of molts in a given volume time series based on mid-molt locations.

    Processes the volume data by performing logarithmic transformations and
    smoothing. Searches for the end of each molt around the provided mid-molt locations
    using the provided search width. The end of each molt is determined by analyzing the
    second derivative of the smoothed volume.

    Parameters:
        volume (np.ndarray): A time series representing volume.
        midmolts (np.ndarray): An array of identified mid-molt locations.
        search_width (int, optional): Width around the mid-molt to search for the end molt. Default is 20.
        fit_width (int, optional): Width for the linear regression fit used in determining the end molt. Default is 5.

    Returns:
        np.ndarray: Indices of the identified end-molts in the input volume.
    """
    # Smooth volume time series
    log_volume = np.log(volume.astype(float))
    log_volume = interpolate_peaks(log_volume)
    log_volume = -interpolate_peaks(-log_volume)

    medfilt_log_volume = medfilt(log_volume, 3)

    endmolts = np.full((4,), np.nan)
    for i, midmolt in enumerate(midmolts):
        if np.isfinite(midmolt):
            search_window = range(
                int(max(midmolt - search_width, 0)),
                int(min(midmolt + search_width + 1, len(medfilt_log_volume))),
            )

            slope1 = np.full_like(medfilt_log_volume, np.nan)
            slope2 = np.full_like(medfilt_log_volume, np.nan)

            for h in search_window:
                # split search window into two parts
                fit_range1 = np.arange(max(0, h - fit_width), h + 1, dtype=int)
                fit_range2 = np.arange(
                    h,
                    min(h + 1 + fit_width, len(medfilt_log_volume)),
                    dtype=int,
                )
                # fit linear regression to each part
                p1 = linregress(
                    fit_range1[np.isfinite(medfilt_log_volume[fit_range1])],
                    medfilt_log_volume[
                        fit_range1[np.isfinite(medfilt_log_volume[fit_range1])]
                    ],
                )
                p2 = linregress(
                    fit_range2[np.isfinite(medfilt_log_volume[fit_range2])],
                    medfilt_log_volume[
                        fit_range2[np.isfinite(medfilt_log_volume[fit_range2])]
                    ],
                )

                slope1[h] = p1.slope  # type: ignore
                slope2[h] = p2.slope  # type: ignore
            second_derivative = slope2 - slope1

            # a = np.max(second_derivative[search_window])
            b = np.argmax(second_derivative[search_window])
            # if np.isfinite(a):
            #     fit_range = np.arange(max(b - 4, 0), min(b + 4 + 1, len(medfilt_log_volume)))
            #     p = np.polyfit(fit_range, second_derivative[search_window][fit_range], 3)
            #     b = np.argmax(np.polyval(p, fit_range))
            #     b = fit_range[b]

            endmolts[i] = search_window[b]

    return endmolts


def find_hatch_time(
    worm_types: np.ndarray,
) -> float:
    """
    Determine the hatch time based on the classified worm types.

    The hatch time is defined as the index immediately after the last egg
    found before the first worm in the sequence. If no worms or eggs are found
    before the first worm, it returns NaN.

    Parameters:
        worm_types (np.ndarray): An array of strings representing the sequence
                                 of worm-related types, e.g., ["egg", "egg", "worm", ...].

    Returns:
        float: The index representing the hatch time.
             Returns NaN if the conditions for hatch time aren't met.
    """

    worm_index = np.argwhere(worm_types == "worm")
    if worm_index.size == 0:
        return np.nan

    first_worm = worm_index[0][0]
    eggs_before_first_worm = np.argwhere(worm_types[:first_worm] == "egg")

    if eggs_before_first_worm.size == 0:
        return np.nan

    last_egg = eggs_before_first_worm[-1][0]

    hatch_time = last_egg + 1
    return hatch_time


def find_molts(
    volume: np.ndarray,
    worm_types: np.ndarray,
    molt_size_range: list = [6.6e4, 15e4, 36e4, 102e4],
    search_width: int = 20,
    fit_width: int = 5,
) -> dict:
    """
    Identify molt events.

    Integrated approach that uses a combination of several utility functions to:
    1. Identify hatch time of the worm.
    2. Identify mid-molts based on specified size ranges.
    3. Identify the end-molts based on mid-molts and given search and fit widths.
    4. Compute the volume at hatch and each molt event.

    Parameters:
        volume (np.ndarray): A time series representing volume.
        worm_types (np.ndarray): An array indicating the type of each entry in the volume time series. Expected values are "worm", "egg", "error", etc.
        molt_size_range (list, optional): Expected size ranges for mid-molts. Default values are provided.
        search_width (int, optional): Width for searching the end-molts. Default is 20.
        fit_width (int, optional): Width for the linear regression fit used in computing the volume. Default is 5.

    Returns:
        dict: Dictionary containing the hatch time and end-molt times.
    """
    volume = volume.astype(float)

    errors = np.where(worm_types == "error")
    volume_for_finding_molts = volume.copy()
    volume_for_finding_molts[errors] = np.nan

    hatch_time = find_hatch_time(worm_types)
    midmolts = find_mid_molts(volume_for_finding_molts, molt_size_range)  # type: ignore
    endmolts = find_end_molts(
        volume_for_finding_molts, midmolts, search_width, fit_width
    )

    ecdysis = {
        "hatch_time": hatch_time,
        "M1": endmolts[0],
        "M2": endmolts[1],
        "M3": endmolts[2],
        "M4": endmolts[3],
    }

    return ecdysis
