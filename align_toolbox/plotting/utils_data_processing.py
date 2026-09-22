import bottleneck as bn
import numpy as np

from align_toolbox.data_analysis import rescale_series
from align_toolbox.data_analysis.growth_rate import (
    compute_instantaneous_growth_rate_classified,
)
from align_toolbox.data_analysis.time_series import smooth_series_classified
from align_toolbox.foundation.utils import find_best_string_match


def combine_series(
    conditions_struct: list,
    series_one: str,
    series_two: str,
    operation: str,
    new_series_name: str,
) -> list:
    """
    Combine two series in every condition dict via an elementwise arithmetic operation.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        series_one (str) : Key of the first operand series.
        series_two (str) : Key of the second operand series.
        operation (str) : Arithmetic operation to apply.
            Supported values: ``"add"``, ``"subtract"``, ``"multiply"``, ``"divide"``.
            Division adds a small epsilon (1e-8) to the denominator to avoid division by zero.
        new_series_name (str) : Key under which the result is stored in each condition dict.

    Returns:
        list : The modified ``conditions_struct`` with the new series added in place.
    """
    for condition in conditions_struct:
        series_one_values = condition[series_one]
        series_two_values = condition[series_two]

        if operation == "add":
            new_series_values = np.add(series_one_values, series_two_values)
        elif operation == "subtract":
            new_series_values = series_one_values - series_two_values
        elif operation == "multiply":
            new_series_values = series_one_values * series_two_values
        elif operation == "divide":
            new_series_values = np.divide(series_one_values, series_two_values + 1e-8)
        condition[new_series_name] = new_series_values
    return conditions_struct


def transform_series(
    conditions_struct: list,
    series: str,
    operation: str,
    new_series_name: str,
) -> list:
    """
    Apply a pointwise mathematical transformation to a series in every condition dict.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        series (str) : Key of the series to transform.
        operation (str) : Transformation to apply.
            Supported values: ``"log"`` (natural log), ``"exp"``, ``"sqrt"``.
        new_series_name (str) : Key under which the transformed series is stored.

    Returns:
        list : The modified ``conditions_struct`` with the transformed series added in place.
    """
    for conditions in conditions_struct:
        series_values = conditions[series]

        if operation == "log":
            new_series_values = np.log(series_values)
        elif operation == "exp":
            new_series_values = np.exp(series_values)
        elif operation == "sqrt":
            new_series_values = np.sqrt(series_values)
        conditions[new_series_name] = new_series_values

    return conditions_struct


def compute_growth_rate(
    conditions_struct: list,
    series_name: str,
    gr_series_name: str,
    experiment_time: bool = True,
    lmbda: float = 0.0075,
    order: int = 2,
    medfilt_window: int = 5,
) -> list:
    """
    Compute the instantaneous growth rate for each worm in every condition.

    The quality-control column is detected automatically by matching ``series_name``
    against all keys containing ``"qc"``.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        series_name (str) : Key of the measurement series (e.g. body length).
        gr_series_name (str) : Key under which the growth-rate series is stored.
        experiment_time (bool) : If ``True``, use ``"experiment_time_hours"``; otherwise
            use ``"time"``.  Defaults to ``True``.
        lmbda (float) : Regularisation parameter for the Whittaker smoother used
            internally.  Defaults to ``0.0075``.
        order (int) : Polynomial order for smoothing.  Defaults to ``2``.
        medfilt_window (int) : Median-filter window size applied before differentiation.
            Defaults to ``5``.

    Returns:
        list : The modified ``conditions_struct`` with the growth-rate series added in place.
    """
    for condition in conditions_struct:
        series_values = condition[series_name]
        qc_keys = [key for key in condition.keys() if "qc" in key]
        if len(qc_keys) == 1:
            qc_key = qc_keys[0]
        else:
            qc_key = find_best_string_match(series_name, qc_keys)
        qc = condition[qc_key]

        if experiment_time:
            time = condition["experiment_time_hours"]
        else:
            time = condition["time"]

        growth_rate = []
        for i in range(series_values.shape[0]):
            gr = compute_instantaneous_growth_rate_classified(
                series_values[i],
                time[i],
                qc[i],
                lmbda=lmbda,
                order=order,
                medfilt_window=medfilt_window,
            )
            growth_rate.append(gr)

        growth_rate = np.array(growth_rate)

        condition[gr_series_name] = growth_rate

    return conditions_struct


def rescale(
    conditions_struct: list,
    series_name: str,
    rescaled_series_name: str,
    experiment_time: bool = True,
    n_points: int = 100,
) -> list:
    """
    Rescale a series to a fixed number of points per larval stage and flatten the stage axis.

    Each worm's series is resampled to ``n_points`` per larval stage via
    ``rescale_series``, yielding shape ``(n_worms, 4, n_points)``, which is then
    reshaped to ``(n_worms, 4 * n_points)``.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        series_name (str) : Key of the raw series to rescale.
        rescaled_series_name (str) : Key under which the rescaled series is stored.
        experiment_time (bool) : If ``True``, use ``"experiment_time"``; otherwise
            use ``"time"``.  Defaults to ``True``.
        n_points (int) : Number of resampled points per larval stage.  Defaults to ``100``.

    Returns:
        list : The modified ``conditions_struct`` with the rescaled series added in place.
    """
    for condition in conditions_struct:
        series_values = condition[series_name]
        qc_keys = [key for key in condition.keys() if "qc" in key]
        if len(qc_keys) == 1:
            qc_key = qc_keys[0]
        else:
            qc_key = find_best_string_match(series_name, qc_keys)
        qc = condition[qc_key]
        ecdysis = condition["ecdysis_index"]

        if experiment_time:
            time = condition["experiment_time"]
        else:
            time = condition["time"]

        _, rescaled_series = rescale_series(
            series_values, time, ecdysis, qc, n_points=n_points
        )  # shape (n_worms, 4, n_points)

        # reshape into (n_worms, 4*n_points)

        rescaled_series = rescaled_series.reshape(rescaled_series.shape[0], -1)

        condition[rescaled_series_name] = rescaled_series

    return conditions_struct


def rescale_without_flattening(
    conditions_struct: list,
    series_name: str,
    rescaled_series_name: str,
    experiment_time: bool = True,
    n_points: int = 100,
) -> list:
    """
    Rescale a series to a fixed number of points per larval stage, retaining the stage axis.

    Like ``rescale``, but the output shape is ``(n_worms, 4, n_points)`` rather than
    being flattened.  Useful when per-stage statistics are required downstream.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        series_name (str) : Key of the raw series to rescale.
        rescaled_series_name (str) : Key under which the rescaled series is stored.
        experiment_time (bool) : If ``True``, use ``"experiment_time"``; otherwise
            use ``"time"``.  Defaults to ``True``.
        n_points (int) : Number of resampled points per larval stage.  Defaults to ``100``.

    Returns:
        list : The modified ``conditions_struct`` with the rescaled series added in place.
    """
    for condition in conditions_struct:
        series_values = condition[series_name]
        qc_keys = [key for key in condition.keys() if "qc" in key]
        if len(qc_keys) == 1:
            qc_key = qc_keys[0]
        else:
            qc_key = find_best_string_match(series_name, qc_keys)
        qc = condition[qc_key]
        ecdysis = condition["ecdysis_index"]

        if experiment_time:
            time = condition["experiment_time"]
        else:
            time = condition["time"]

        _, rescaled_series = rescale_series(
            series_values, time, ecdysis, qc, n_points=n_points
        )  # shape (n_worms, 4, n_points)

        condition[rescaled_series_name] = rescaled_series

    return conditions_struct


def exclude_arrests_from_series_at_ecdysis(series_at_ecdysis: np.ndarray) -> np.ndarray:
    """
    Remove arrested worms from a per-molt measurement array.

    A worm is considered arrested at molt *i* if its value at molt *i+1* is NaN.
    The value at that molt is set to NaN in the output.  The last molt event is
    always kept regardless.

    Parameters:
        series_at_ecdysis (np.ndarray) : Per-worm, per-molt values.
            Shape ``(n_molts,)`` for a single worm or ``(n_worms, n_molts)`` for
            multiple worms.

    Returns:
        np.ndarray : Copy of ``series_at_ecdysis`` with arrested molt values replaced by NaN.
    """
    filtered_series = np.full(series_at_ecdysis.shape, np.nan)
    # keep only a value at one ecdys event if the next one is not nan
    if series_at_ecdysis.shape[0] == 1 or len(series_at_ecdysis.shape) == 1:
        for i in range(len(series_at_ecdysis)):
            if i == len(series_at_ecdysis) - 1:
                filtered_series[i] = series_at_ecdysis[i]
            elif not np.isnan(series_at_ecdysis[i + 1]):
                filtered_series[i] = series_at_ecdysis[i]
        return filtered_series
    else:
        for i in range(series_at_ecdysis.shape[0]):
            for j in range(series_at_ecdysis.shape[1]):
                if j == series_at_ecdysis.shape[1] - 1:
                    filtered_series[i, j] = series_at_ecdysis[i, j]
                elif not np.isnan(series_at_ecdysis[i, j + 1]):
                    filtered_series[i, j] = series_at_ecdysis[i, j]
        return filtered_series


def smooth_series(
    conditions_struct: list,
    series_name: str,
    smoothed_series_name: str,
    experiment_time: bool = True,
    lmbda: float = 0.0075,
    order: int = 2,
    medfilt_window: int = 5,
) -> list:
    """
    Smooth a measurement series for each worm in every condition using a classified smoother.

    Short smoothed series are right-padded with NaN to match the original length.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        series_name (str) : Key of the raw series to smooth.
        smoothed_series_name (str) : Key under which the smoothed series is stored.
        experiment_time (bool) : If ``True``, use ``"experiment_time_hours"``; otherwise
            use ``"time"``.  Defaults to ``True``.
        lmbda (float) : Regularisation parameter for the Whittaker smoother.  Defaults to ``0.0075``.
        order (int) : Polynomial order for smoothing.  Defaults to ``2``.
        medfilt_window (int) : Median-filter window size applied before the smoother.
            Defaults to ``5``.

    Returns:
        list : The modified ``conditions_struct`` with the smoothed series added in place.
    """
    for condition in conditions_struct:
        series_values = condition[series_name]
        qc_keys = [key for key in condition.keys() if "qc" in key]
        if len(qc_keys) == 1:
            qc_key = qc_keys[0]
        else:
            qc_key = find_best_string_match(series_name, qc_keys)
        qc = condition[qc_key]
        if experiment_time:
            time = condition["experiment_time_hours"]
        else:
            time = condition["time"]

        smoothed_series = []
        for i in range(len(series_values)):
            values = series_values[i]
            smoothed = smooth_series_classified(
                values,
                time[i],
                qc[i],
                lmbda=lmbda,
                order=order,
                medfilt_window=medfilt_window,
            )

            # pad with 0 until it's the same length as the original series
            if len(smoothed) < len(values):
                smoothed = np.pad(
                    smoothed,
                    (0, len(values) - len(smoothed)),
                    "constant",
                    constant_values=(np.nan, np.nan),
                )

            smoothed_series.append(smoothed)

        smoothed_series = np.array(smoothed_series)
        condition[smoothed_series_name] = smoothed_series

    return conditions_struct


def smooth_and_rescale_series(
    conditions_struct: list,
    series_name: str,
    smoothed_series_name: str,
    experiment_time: bool = True,
    lmbda: float = 0.0075,
    order: int = 2,
    medfilt_window: int = 5,
    n_points: int = 100,
    flatten: bool = True,
) -> list:
    """
    Smooth a series and rescale it to a fixed number of points per larval stage.

    After smoothing, the QC classification is discarded (all frames are treated as
    ``"worm"``) before resampling.  Two new keys are written to each condition dict:
    ``smoothed_series_name`` (the smoothed but not rescaled series) and
    ``f"{smoothed_series_name}_rescaled"`` (the rescaled result).

    Parameters:
        conditions_struct (list) : List of condition dicts.
        series_name (str) : Key of the raw series.
        smoothed_series_name (str) : Base key for the output series.
        experiment_time (bool) : If ``True``, use ``"experiment_time_hours"``; otherwise
            use ``"time"``.  Defaults to ``True``.
        lmbda (float) : Regularisation parameter for the Whittaker smoother.  Defaults to ``0.0075``.
        order (int) : Polynomial order for smoothing.  Defaults to ``2``.
        medfilt_window (int) : Median-filter window size applied before the smoother.
            Defaults to ``5``.
        n_points (int) : Number of resampled points per larval stage.  Defaults to ``100``.
        flatten (bool) : If ``True``, reshape the rescaled array from
            ``(n_worms, 4, n_points)`` to ``(n_worms, 4 * n_points)``.
            Defaults to ``True``.

    Returns:
        list : The modified ``conditions_struct`` with the smoothed and rescaled series added in place.
    """
    for condition in conditions_struct:
        series_values = condition[series_name]
        qc_keys = [key for key in condition.keys() if "qc" in key]
        if len(qc_keys) == 1:
            qc_key = qc_keys[0]
        else:
            qc_key = find_best_string_match(series_name, qc_keys)
        qc = condition[qc_key]
        ecdysis = condition["ecdysis_index"]
        if experiment_time:
            time = condition["experiment_time_hours"]
        else:
            time = condition["time"]

        smoothed_series = []
        for i in range(len(series_values)):
            values = series_values[i]
            smoothed = smooth_series_classified(
                values,
                time[i],
                qc[i],
                lmbda=lmbda,
                order=order,
                medfilt_window=medfilt_window,
            )

            # pad with 0 until it's the same length as the original series
            if len(smoothed) < len(values):
                smoothed = np.pad(
                    smoothed,
                    (0, len(values) - len(smoothed)),
                    "constant",
                    constant_values=(np.nan, np.nan),
                )

            smoothed_series.append(smoothed)

        smoothed_series = np.array(smoothed_series)

        # we don't need the classification anymore
        qc = np.full_like(smoothed_series, "worm", dtype=object)

        # rescale the smoothed series
        _, rescaled_series = rescale_series(
            smoothed_series, time, ecdysis, qc, n_points=n_points
        )  # shape (n_worms, 4, n_points)

        # reshape into (n_worms, 4*n_points)
        if flatten:
            rescaled_series = rescaled_series.reshape(rescaled_series.shape[0], -1)

        condition[smoothed_series_name] = smoothed_series
        condition[f"{smoothed_series_name}_rescaled"] = rescaled_series
    return conditions_struct


def detrend_rescaled_series_population_mean(
    conditions_struct: list,
    rescaled_series: str,
) -> list:
    """
    Remove the population mean trend from a rescaled series in every condition.

    The population mean is computed across worms within each condition independently.
    The detrended result is stored under ``f"{rescaled_series}_detrended"``.

    Parameters:
        conditions_struct (list) : List of condition dicts.
        rescaled_series (str) : Key of the rescaled series to detrend.

    Returns:
        list : The modified ``conditions_struct`` with the detrended series added in place.
    """
    for condition in conditions_struct:
        series = condition[rescaled_series]
        detrended_series = _detrend_rescaled_series_population_mean(series)
        new_name = f"{rescaled_series}_detrended"
        condition[new_name] = detrended_series

    return conditions_struct


def _detrend_rescaled_series_population_mean(series: np.ndarray) -> np.ndarray:
    """
    Subtract the column-wise population mean from a 2-D series array.

    Parameters:
        series (np.ndarray) : Array of shape ``(n_worms, n_points)``.

    Returns:
        np.ndarray : Detrended array of the same shape.
    """
    # detrend the series by subtracting the population mean
    population_mean = bn.nanmean(series, axis=0)
    detrended_series = series - population_mean

    return detrended_series
