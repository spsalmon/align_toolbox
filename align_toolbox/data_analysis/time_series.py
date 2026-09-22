import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from whittaker_eilers import WhittakerSmoother

from align_toolbox.foundation.utils import interpolate_nans_infs


def resize_series_to_length(
    series: np.ndarray, new_length: int, kind: str = "linear"
) -> np.ndarray:
    """
    Resize a 1D or 2D series to a new length using interpolation.

    Parameters:
        series (np.ndarray): The time series to resize.
        new_length (int): The desired length of the resized series.
        kind (str, optional): The type of interpolation to use. (default: "linear")

    Returns:
        np.ndarray: The resized series of the specified length.

    Raises:
        ValueError: If the input series is not 1D or 2D.
    """
    series = np.asarray(series)
    length = series.shape[-1]
    if length == new_length:
        return series
    x_old = np.linspace(0, 1, length)
    x_new = np.linspace(0, 1, new_length)
    if series.ndim == 1:
        f = interp1d(x_old, series, kind=kind, fill_value="extrapolate")
        return f(x_new)
    elif series.ndim == 2:
        f = interp1d(x_old, series, axis=1, kind=kind, fill_value="extrapolate")
        return f(x_new)
    else:
        raise ValueError("Input series must be 1D or 2D array.")


def pad_series_to_length(
    series: np.ndarray, new_length: int, pad_value: float = 0
) -> np.ndarray:
    """
    Pad a 1D or 2D series to a new length with a specified pad value.

    Parameters:
        series (np.ndarray): The time series to pad.
        new_length (int): The desired length of the padded series.
        pad_value (int, optional): The value to use for padding. Default is 0

    Returns:
        np.ndarray: The padded series of the specified length.

    Raises:
        ValueError: If the input series is not 1D or 2D.
    """
    series = np.asarray(series)
    length = series.shape[-1]
    if length == new_length:
        return series
    if series.ndim == 1:
        return np.pad(
            series, (0, new_length - length), mode="constant", constant_values=pad_value
        )
    elif series.ndim == 2:
        return np.pad(
            series,
            ((0, 0), (0, new_length - length)),
            mode="constant",
            constant_values=pad_value,
        )
    else:
        raise ValueError("Input series must be 1D or 2D array.")


def crop_series_to_length(series: np.ndarray, new_length: int) -> np.ndarray:
    """
    Crop a 1D or 2D series to a new length.

    Parameters:
        series (np.ndarray): The time series to crop.
        new_length (int): The desired length of the cropped series.

    Returns:
        np.ndarray: The cropped series of the specified length.

    Raises:
        ValueError: If the input series is not 1D or 2D.
    """
    series = np.asarray(series)
    length = series.shape[-1]
    if length == new_length:
        return series
    if series.ndim == 1:
        return series[0:new_length]
    elif series.ndim == 2:
        return series[:, 0:new_length]
    else:
        raise ValueError("Input series must be 1D or 2D array.")


def random_crop_series_to_length(series: np.ndarray, new_length: int) -> np.ndarray:
    """
    Randomly crop a 1D or 2D series to a new length.

    Parameters:
        series (np.ndarray): The time series to crop.
        new_length (int): The desired length of the cropped series.

    Returns:
        np.ndarray: The cropped series of the specified length.

    Raises:
        ValueError: If the input series is not 1D or 2D.
    """
    series = np.asarray(series)
    length = series.shape[-1]

    if length == new_length:
        return series
    if series.ndim == 1:
        start = np.random.randint(0, length - new_length + 1)
        return series[start : start + new_length]
    elif series.ndim == 2:
        start = np.random.randint(0, length - new_length + 1)
        return series[:, start : start + new_length]
    else:
        raise ValueError("Input series must be 1D or 2D array.")


def correct_series_with_classification(
    series: np.ndarray, qc: np.ndarray
) -> np.ndarray:
    """
    Remove the points of non-worms from the time series and interpolate them back.

    Parameters:
        series (np.ndarray): The time series of values.
        qc (np.ndarray): The classification of the points as either 'worm' or 'egg' or 'error'.

    Returns:
        np.ndarray: The corrected time series.
    """

    # Set the series of non worms to NaN
    non_worms_indices = qc != "worm"
    series_worms = series.copy()
    series_worms[non_worms_indices] = np.nan

    try:
        # Interpolate the NaNs
        series_worms = interpolate_nans_infs(series_worms)
    except ValueError:
        return np.full(series.shape, np.nan)

    return series_worms


def filter_series_with_classification(series: np.ndarray, qc: np.ndarray) -> np.ndarray:
    """Remove the points of non-worms from the time series.

    Parameters:
        series (np.ndarray): The time series of values.
        qc (np.ndarray): The classification of the points as either 'worm' or 'egg' or 'error'.

    Returns:
        np.ndarray: The filtered time series."""

    # Set the series of non worms to NaN
    non_worms_indices = qc != "worm"
    series_worms = series.copy()
    series_worms[non_worms_indices] = np.nan

    return series_worms


def interpolate_larval_stage(
    series: np.ndarray,
    time: np.ndarray,
    ecdysis: np.ndarray,
    larval_stage: int,
    qc: np.ndarray | None = None,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate a time series over a single larval stage to a fixed number of points.

    Uses a linear b-spline to resample the series between the ecdysis events bounding
    the requested larval stage. Non-worm time points are excluded when ``qc`` is
    provided.

    Parameters:
        series (np.ndarray): 1D array of measured values.
        time (np.ndarray): 1D array of time points corresponding to ``series``.
        ecdysis (array-like): 5-element sequence
            ``[hatch_time, M1, M2, M3, M4]`` of ecdysis indices.
        larval_stage (int): Larval stage to interpolate (1–4).
        qc (np.ndarray, optional): Classification array with values ``"worm"``,
            ``"egg"``, or ``"error"``; non-worm points are masked before
            interpolation. (default: None)
        n_points (int, optional): Number of evenly-spaced output points.
            (default: 100)

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(interpolated_time, interpolated_series)``
            each of length ``n_points``. Both arrays are filled with NaN when the
            bounding ecdysis events are invalid.

    Raises:
        ValueError: If ``larval_stage`` is not in the range [1, 4].
    """
    if larval_stage < 1 or larval_stage > 4:
        raise ValueError("The larval stage must be between 1 and 4.")

    previous_ecdys = ecdysis[larval_stage - 1]
    ecdys = ecdysis[larval_stage]

    # check that the molts are correct
    if (np.isnan(previous_ecdys) or np.isnan(ecdys)) or previous_ecdys > ecdys:
        return np.full(n_points, np.nan), np.full(n_points, np.nan)

    # convert ecdysis times to int to get the index
    previous_ecdys = int(previous_ecdys)
    ecdys = int(ecdys)

    interpolated_time = np.linspace(time[previous_ecdys], time[ecdys], n_points)

    if qc is not None:
        filtered_series = filter_series_with_classification(series, qc)
        filtered_time = filter_series_with_classification(time, qc)
    else:
        filtered_series = series
        filtered_time = time

    # remove nan values for interpolation
    valid_indices = ~np.isnan(filtered_time) & ~np.isnan(filtered_series)
    filtered_time = filtered_time[valid_indices]
    filtered_series = filtered_series[valid_indices]

    if filtered_time.size == 0 or filtered_series.size == 0:
        return np.full(n_points, np.nan), np.full(n_points, np.nan)

    interpolated_series = interpolate.make_interp_spline(
        filtered_time, filtered_series, k=1
    )(interpolated_time)

    return interpolated_time, interpolated_series


def interpolate_entire_development(
    series: np.ndarray,
    time: np.ndarray,
    ecdysis: np.ndarray,
    qc: np.ndarray | None = None,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate a time series over all four larval stages.

    Calls :func:`interpolate_larval_stage` for each stage and stacks the results.

    Parameters:
        series (np.ndarray): 1D array of measured values.
        time (np.ndarray): 1D array of time points corresponding to ``series``.
        ecdysis (array-like): 5-element sequence
            ``[hatch_time, M1, M2, M3, M4]`` of ecdysis indices.
        qc (np.ndarray, optional): Classification array with values ``"worm"``,
            ``"egg"``, or ``"error"``; non-worm points are masked before
            interpolation. (default: None)
        n_points (int, optional): Number of evenly-spaced output points per stage.
            (default: 100)

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(interpolated_time, interpolated_series)``
            both of shape ``(4, n_points)``. Rows correspond to larval stages L1–L4;
            stages with invalid ecdysis events are filled with NaN.
    """
    interpolated_time = np.full((4, n_points), np.nan)
    interpolated_series = np.full((4, n_points), np.nan)
    for larval_stage in range(1, 5):
        (
            interpolated_time_stage,
            interpolated_series_stage,
        ) = interpolate_larval_stage(
            series, time, ecdysis, larval_stage, qc=qc, n_points=n_points
        )

        interpolated_time[larval_stage - 1, :] = interpolated_time_stage
        interpolated_series[larval_stage - 1, :] = interpolated_series_stage

    return interpolated_time, interpolated_series


def compute_exponential_series_at_time_classified(
    series: np.ndarray,
    time: np.ndarray,
    qc: np.ndarray,
    fit_width: int = 10,
) -> np.ndarray:
    """
    Evaluate a time series at given time indices using exponential (log-linear) regression.

    For each requested time index, fits a linear regression to the log-transformed
    series within a window of ``fit_width`` points on each side, using only
    points classified as ``"worm"``. Returns the back-transformed (exponential)
    predicted value at each requested time.

    Parameters:
        series (np.ndarray): 1D array of measured values (e.g. volume).
        time (np.ndarray): 1D array of time indices at which to evaluate the series.
        qc (np.ndarray): Classification array with values ``"worm"``, ``"egg"``, or
            ``"error"``; only ``"worm"`` points are used for fitting.
        fit_width (int, optional): Half-width of the fitting window around each
            requested time index. (default: 10)

    Returns:
        np.ndarray: Predicted series values at each element of ``time``. NaN is
            returned for time indices that are non-finite or where no ``"worm"``
            points fall within the fitting window.
    """

    def compute_single_time(time: float) -> float:
        if np.isfinite(time):
            fit_x = np.arange(
                max(0, int(time - fit_width)),
                min(len(series), int(time + fit_width)),
                dtype=int,
            )

            filtered_fit_x = fit_x[np.where(qc[fit_x] == "worm")]
            if filtered_fit_x.size != 0:
                fit_y = np.log(series[filtered_fit_x])
                try:
                    p = np.polyfit(filtered_fit_x, fit_y, 1)
                except Exception as e:
                    print(
                        f"Caught an exception while interpolating volume at time {time}, returning nan : {e}"
                    )
                    return np.nan
                series_at_time = np.exp(np.polyval(p, time))
            else:
                series_at_time = np.nan
        else:
            series_at_time = np.nan

        return float(series_at_time)

    result = np.array([compute_single_time(t) for t in time])

    return result


def smooth_series_classified(
    series: np.ndarray,
    series_time: np.ndarray | None,
    qc: np.ndarray,
    lmbda: float = 0.0075,
    order: int = 2,
    medfilt_window: int = 5,
) -> np.ndarray:
    """
    Smooth a time series after masking non-worm points.

    Non-worm points (according to ``qc``) are replaced by interpolated values,
    then a median filter is applied followed by Whittaker-Eilers smoothing.

    Parameters:
        series (np.ndarray): 1D array of measured values.
        series_time (np.ndarray): 1D array of time points corresponding to
            ``series``. Pass ``None`` to use integer indices.
        qc (np.ndarray): Classification array with values ``"worm"``, ``"egg"``,
            or ``"error"``; non-worm points are interpolated before smoothing.
        lmbda (float, optional): Smoothing parameter for the Whittaker-Eilers
            smoother. (default: 0.0075)
        order (int, optional): Penalty order for the Whittaker-Eilers smoother.
            (default: 2)
        medfilt_window (int, optional): Kernel size for the median filter applied
            before Whittaker-Eilers smoothing. (default: 5)

    Returns:
        np.ndarray: Smoothed series of the same length as ``series``. Returns an
            all-NaN array when all input values are NaN.
    """

    if np.all(np.isnan(series)):
        return np.full(series.shape, np.nan)

    series = interpolate_nans_infs(series)

    # Remove the points of non-worms from the time series and interpolate them back
    series = correct_series_with_classification(series, qc)

    smoothed_series = _smooth_series(
        series,
        series_time,
        lmbda=lmbda,
        order=order,
        medfilt_window=medfilt_window,
    )
    return smoothed_series


def smooth_series(
    series: np.ndarray,
    series_time: np.ndarray | None,
    lmbda: float = 0.0075,
    order: int = 2,
    medfilt_window: int = 5,
) -> np.ndarray:
    """
    Smooth a time series without classification-based masking.

    Applies a median filter followed by Whittaker-Eilers smoothing. Unlike
    :func:`smooth_series_classified`, non-worm points are not masked first.

    Parameters:
        series (np.ndarray): 1D array of measured values.
        series_time (np.ndarray): 1D array of time points corresponding to
            ``series``. Pass ``None`` to use integer indices.
        lmbda (float, optional): Smoothing parameter for the Whittaker-Eilers
            smoother. (default: 0.0075)
        order (int, optional): Penalty order for the Whittaker-Eilers smoother.
            (default: 2)
        medfilt_window (int, optional): Kernel size for the median filter applied
            before Whittaker-Eilers smoothing. (default: 5)

    Returns:
        np.ndarray: Smoothed series of the same length as ``series``. Returns an
            all-NaN array when all input values are NaN.
    """

    if np.all(np.isnan(series)):
        return np.full(series.shape, np.nan)

    series = interpolate_nans_infs(series)

    smoothed_series = _smooth_series(
        series,
        series_time,
        lmbda=lmbda,
        order=order,
        medfilt_window=medfilt_window,
    )
    return smoothed_series


def _smooth_series(
    series: np.ndarray,
    series_time: np.ndarray,
    lmbda: float = 0.0075,
    order: int = 2,
    medfilt_window: int = 5,
) -> np.ndarray:
    """
    Apply median filter then Whittaker-Eilers smoothing to a series.

    Parameters:
        series (np.ndarray): 1D array of measured values (NaN-free expected).
        series_time (np.ndarray): 1D array of time points. Pass ``None`` to use
            integer indices.
        lmbda (float, optional): Smoothing parameter for the Whittaker-Eilers
            smoother. (default: 0.0075)
        order (int, optional): Penalty order for the Whittaker-Eilers smoother.
            (default: 2)
        medfilt_window (int, optional): Kernel size for the median filter.
            (default: 5)

    Returns:
        np.ndarray: Smoothed series of the same length as ``series``.
    """
    # Median filter to remove outliers
    series = medfilt(series, medfilt_window)

    if series_time is not None:
        nan_series_time = np.isnan(series_time)
        non_nan_series = series[~nan_series_time]
        non_nan_series_time = series_time[~nan_series_time]
        whittaker_smoother = WhittakerSmoother(
            lmbda=lmbda,
            order=order,
            data_length=len(non_nan_series),
            x_input=non_nan_series_time,
        )

        smoothed_series = whittaker_smoother.smooth(non_nan_series)
    else:
        whittaker_smoother = WhittakerSmoother(
            lmbda=lmbda, order=order, data_length=len(series)
        )

        smoothed_series = whittaker_smoother.smooth(series)

    smoothed_series = np.array(smoothed_series)

    # pad the smoothed series to the original shape
    smoothed_series = np.pad(
        smoothed_series,
        (0, series.shape[0] - smoothed_series.shape[0]),
        mode="constant",
        constant_values=np.nan,
    )

    # Interpolate the nans again
    smoothed_series = interpolate_nans_infs(smoothed_series)

    return smoothed_series


def compute_series_at_time_classified(
    series: np.ndarray,
    time: np.ndarray,
    series_time: np.ndarray | None,
    qc: np.ndarray,
    lmbda: float = 0.0075,
    medfilt_window: int = 5,
    bspline_order: int = 3,
) -> np.ndarray:
    """
    Evaluate a time series at arbitrary time points after smoothing and classification-based masking.

    Non-worm points are masked and interpolated, then the series is smoothed with a
    median filter and Whittaker-Eilers smoothing, and finally evaluated at the
    requested ``time`` values using a b-spline interpolant.

    Parameters:
        series (np.ndarray): 1D array of measured values.
        time (np.ndarray): Time points at which the smoothed series is to be evaluated.
        series_time (np.ndarray): 1D array of time points corresponding to ``series``.
            Pass ``None`` to use integer indices.
        qc (np.ndarray): Classification array with values ``"worm"``, ``"egg"``,
            or ``"error"``; non-worm points are masked before smoothing.
        lmbda (float, optional): Smoothing parameter for the Whittaker-Eilers
            smoother. (default: 0.0075)
        medfilt_window (int, optional): Kernel size for the median filter. (default: 5)
        bspline_order (int, optional): The order of the b-spline interpolation. (default: 3)

    Returns:
        np.ndarray: The series at the given time(s).
    """

    if np.all(np.isnan(series)):
        # handle time being a single value
        if np.isscalar(time):
            return np.full(1, np.nan)
        else:
            return np.full(time.shape, np.nan)

    series = smooth_series_classified(
        series,
        series_time,
        qc,
        medfilt_window=medfilt_window,
        lmbda=lmbda,
    )

    if series_time is None:
        series_time = np.arange(len(series))

    try:
        valid_indices = ~np.isnan(series_time) & ~np.isnan(series)
        filtered_time = series_time[valid_indices]
        filtered_series = series[valid_indices]
        interpolated_series = interpolate.make_interp_spline(
            filtered_time, filtered_series, k=bspline_order
        )
    except Exception as e:
        print(f"Caught an exception while interpolating series: {e}")
        return np.full(time.shape, np.nan)

    return interpolated_series(time)


def rescale_series(
    series: np.ndarray,
    time: np.ndarray,
    ecdysis: np.ndarray,
    qc: np.ndarray,
    points: list[int] | None = None,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolates one or multiple series and the time to have n_points points in total per larval stages.

    Parameters:
        series (np.ndarray) : Array of shape (nb_of_worms / points, length of series) containing the series to interpolate.
        time (np.ndarray) : Array of shape (nb_of_worms / points, length of series) containing the time information.
        ecdysis (np.ndarray) : Array of shape (nb_of_worms / points, 5) containing the ecdysis times.
        qc (np.ndarray) : Array of shape (nb_of_worms / points, length of series) containing the worm type classification
        points (list) : List of points to consider for the interpolation. If None, all points are considered.
        n_points (int) : Number of points to interpolate the series to.

    Returns:
        all_points_interpolated_time (np.ndarray) : Array of shape (nb_of_worms / points, 4, n_points) containing the interpolated time information.
        all_points_interpolated_series (np.ndarray) : Array of shape (nb_of_worms / points, 4, n_points) containing the interpolated series.
    """
    if points is not None:
        series = series[points]
        time = time[points]
        qc = qc[points]
        ecdysis = ecdysis[points]

    # Interpolate the time and the series

    all_points_interpolated_time = []
    all_points_interpolated_series = []

    for point in range(series.shape[0]):
        series_point, time_point, ecdysis_point, qc_point = (
            series[point],
            time[point],
            ecdysis[point],
            qc[point],
        )
        (
            interpolated_time,
            interpolated_series,
        ) = interpolate_entire_development(
            series_point,
            time_point,
            ecdysis_point,
            qc=qc_point,
            n_points=n_points,
        )

        all_points_interpolated_time.append(interpolated_time)
        all_points_interpolated_series.append(interpolated_series)

    all_points_interpolated_time = np.array(all_points_interpolated_time)
    all_points_interpolated_series = np.array(all_points_interpolated_series)

    return all_points_interpolated_time, all_points_interpolated_series


def aggregate_interpolated_series(
    all_points_interpolated_series: np.ndarray,
    all_points_interpolated_time: np.ndarray,
    larval_stage_durations: np.ndarray,
    aggregation: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregates the interpolated series and time information.

    Parameters:
        all_points_interpolated_series (np.ndarray) : Array of shape (nb_of_worms / points, 4, n_points) containing the interpolated series.
        all_points_interpolated_time (np.ndarray) : Array of shape (nb_of_worms / points, 4, n_points) containing the interpolated time information.
        larval_stage_durations (np.ndarray) : Array of shape (nb_of_worms / points, 4) containing the larval stage durations.
        aggregation (str) : Aggregation method to use. Can be 'mean' or 'median'.

    Returns:
        rescaled_time (np.ndarray) : Array of shape (4*n_points) containing the rescaled time information.
        aggregated_series (np.ndarray) : Array of shape (4*n_points) containing the aggregated series.
        std_series (np.ndarray) : Array of shape (4*n_points) containing the standard deviation of the series.
        ste_series (np.ndarray) : Array of shape (4*n_points) containing the standard error of the series.
    """
    if aggregation == "mean":
        aggregation_function = np.nanmean
    elif aggregation == "median":
        aggregation_function = np.nanmedian

    n_points = all_points_interpolated_time.shape[-1]

    aggregated_series = np.full((4, n_points), np.nan)
    std_series = np.full((4, n_points), np.nan)
    ste_series = np.full((4, n_points), np.nan)
    rescaled_time = np.full((4, n_points), np.nan)

    aggregated_larval_stage_durations = aggregation_function(
        larval_stage_durations, axis=0
    )

    for i in range(4):
        aggregated_series[i, :] = aggregation_function(
            all_points_interpolated_series[:, i, :], axis=0
        )
        std_series[i, :] = np.nanstd(all_points_interpolated_series[:, i, :], axis=0)
        # standard error uses the number of worms contributing to each time
        # point, not the number of finite values in the whole larval stage
        n_worms = np.sum(np.isfinite(all_points_interpolated_series[:, i, :]), axis=0)
        ste_series[i, :] = np.nanstd(
            all_points_interpolated_series[:, i, :], axis=0
        ) / np.sqrt(n_worms)

        beginning = (
            np.nansum(aggregated_larval_stage_durations[: i + 1])
            - aggregated_larval_stage_durations[i]
        )
        end = np.nansum(aggregated_larval_stage_durations[: i + 1])
        rescaled_time[i, :] = np.linspace(beginning, end, n_points)

    # flatten the arrays
    aggregated_series = aggregated_series.flatten()
    std_series = std_series.flatten()
    ste_series = ste_series.flatten()
    rescaled_time = rescaled_time.flatten()

    return rescaled_time, aggregated_series, std_series, ste_series


def rescale_and_aggregate(
    series: np.ndarray,
    time: np.ndarray,
    ecdysis: np.ndarray,
    larval_stage_durations: np.ndarray,
    qc: np.ndarray,
    points: list[int] | None = None,
    aggregation: str = "mean",
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rescales and aggregates the series and time information.

    Parameters:
        series (np.ndarray) : Array of shape (nb_of_worms, length of series) containing the series to interpolate.
        time (np.ndarray) : Array of shape (nb_of_worms, length of series) containing the time information.
        ecdysis (np.ndarray) : Array of shape (nb_of_worms, 5) containing the ecdysis times.
        qc (np.ndarray) : Array of shape (nb_of_worms, length of series) containing the worm type classification
        points (list) : List of points to consider for the interpolation. If None, all points are considered.
        aggregation (str) : Aggregation method to use. Can be 'mean' or 'median'.
        n_points (int) : Number of points to interpolate the series to.

    Returns:
        rescaled_time (np.ndarray) : Array of shape (4*n_points) containing the rescaled time information.
        aggregated_series (np.ndarray) : Array of shape (4*n_points) containing the aggregated series.
        std_series (np.ndarray) : Array of shape (4*n_points) containing the standard deviation of the series.
        ste_series (np.ndarray) : Array of shape (4*n_points) containing the standard error of the series.
    """

    (
        all_points_interpolated_time,
        all_points_interpolated_series,
    ) = rescale_series(series, time, ecdysis, qc, points=points, n_points=n_points)
    (
        rescaled_time,
        aggregated_series,
        std_series,
        ste_series,
    ) = aggregate_interpolated_series(
        all_points_interpolated_series,
        all_points_interpolated_time,
        larval_stage_durations,
        aggregation=aggregation,
    )

    return rescaled_time, aggregated_series, std_series, ste_series
