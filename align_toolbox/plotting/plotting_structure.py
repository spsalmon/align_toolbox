import os
from collections import defaultdict

import numpy as np
import polars as pl
import yaml

from align_toolbox.data_analysis import compute_series_at_time_classified
from align_toolbox.foundation.file_handling import read_filemap
from align_toolbox.foundation.utils import find_best_string_match
from align_toolbox.foundation.worm_features import get_features_to_compute_at_molt

FEATURES_TO_COMPUTE_AT_MOLT = get_features_to_compute_at_molt()

# THIS PART HANDLES THE PROCESSING OF THE EXPERIMENT FILEMAP AND THE CREATION OF THE PLOTTING STRUCTURE


def build_conditions(conditions_yaml: str | dict) -> list:
    """
    Process a conditions YAML file structured in a factorized way. Refer to the documentation for the expected format.
    The function will return a list of dictionaries, each representing a condition with its parameters.

    Parameters:
        conditions_yaml (str or dict): Path to the YAML file or a dictionary containing the conditions.
    Returns:
        list: A list of dictionaries, each representing a condition with its parameters.
    Raises:
        ValueError: If the conditions YAML file is not structured correctly.
    """
    if isinstance(conditions_yaml, str):
        with open(conditions_yaml) as file:
            conditions_dict = yaml.safe_load(file)
            file.close()
    elif isinstance(conditions_yaml, dict):
        conditions_dict = conditions_yaml

    conditions = []
    condition_id = 0

    for condition in conditions_dict["conditions"]:
        condition = {
            key: [val] if not isinstance(val, list) else val
            for key, val in condition.items()
        }

        lengths = {len(val) for val in condition.values()}
        if len(lengths) > 2 or (len(lengths) == 2 and 1 not in lengths):
            raise ValueError(
                "All lists in the condition must have the same length or be of length 1."
            )

        max_length = max(lengths)
        for i in range(max_length):
            condition_dict = {
                key: val[0] if len(val) == 1 else val[i]
                for key, val in condition.items()
            }
            condition_dict["condition_id"] = condition_id
            conditions.append(condition_dict)
            condition_id += 1

    return conditions


def _add_conditions_to_filemap(
    experiment_filemap: pl.DataFrame,
    conditions: list[dict],
) -> pl.DataFrame:
    """
    Add condition metadata columns to the experiment filemap.

    Rows are matched to conditions via ``"point_range"`` (inclusive integer range)
    or ``"pad"`` (exact pad string) keys in each condition dict.  Unmatched rows
    receive ``None`` for the added columns.

    Parameters:
        experiment_filemap (polars.DataFrame) : Filemap with at least ``"Point"``
            and optionally ``"Pad"`` columns.
        conditions (list[dict]) : List of condition dicts as returned by
            ``build_conditions``; each must contain ``"point_range"`` or ``"pad"``.

    Returns:
        polars.DataFrame : Filemap with one new column per condition metadata key.
    """
    if "Pad" in experiment_filemap.columns:
        new_columns = experiment_filemap.select(pl.col("Point"), pl.col("Pad"))
    else:
        new_columns = experiment_filemap.select(pl.col("Point"))
    for condition in conditions:
        # Get condition rows mask
        if "point_range" in condition:
            point_range = condition["point_range"]
            # handle both single range and list of ranges
            if isinstance(point_range[0], list):
                # multiple point ranges - build expressions list
                mask_exprs = [
                    (pl.col("Point") >= pr[0]) & (pl.col("Point") <= pr[1])
                    for pr in point_range
                ]
                # Combine with or operator
                condition_mask = mask_exprs[0]
                for expr in mask_exprs[1:]:
                    condition_mask = condition_mask | expr
            else:
                # single point range
                condition_mask = (pl.col("Point") >= point_range[0]) & (
                    pl.col("Point") <= point_range[1]
                )
            filter_key = "point_range"
        elif "pad" in condition:
            pad = condition["pad"]
            condition_mask = pl.col("Pad") == pad
            filter_key = "pad"
        else:
            print(
                "Condition does not contain 'point_range' or 'pad' key, impossible to add condition to filemap, skipping."
            )
            continue

        # Extract the condition attributes to add (excluding the filter key)
        conditions_to_add = {k: v for k, v in condition.items() if k != filter_key}

        column_ops = []
        for key, val in conditions_to_add.items():
            # For new columns, add with null values first if needed
            if key not in new_columns.columns:
                new_columns = new_columns.with_columns(pl.lit(None).alias(key))

            # Build the column operation but don't apply it yet
            column_ops.append(
                pl.when(condition_mask)
                .then(pl.lit(val))
                .otherwise(pl.col(key))
                .alias(key)
            )

        # Add the new columns to the DataFrame
        if column_ops:
            new_columns = new_columns.with_columns(column_ops)

    if "Pad" in new_columns.columns:
        new_columns = new_columns.drop(["Point", "Pad"])
    else:
        new_columns = new_columns.drop(["Point"])

    # Apply the new columns to the experiment filemap
    if not new_columns.is_empty():
        experiment_filemap = experiment_filemap.with_columns(new_columns)

    return experiment_filemap


def _get_custom_columns(filemap: pl.DataFrame) -> list[str]:
    """
    Identify non-standard columns in the filemap for pass-through to the plotting structure.

    Standard columns (time, ecdysis events, QC, analysis, and known worm features)
    are excluded; anything else is considered a custom column.

    Parameters:
        filemap (polars.DataFrame) : Experiment filemap.

    Returns:
        list[str] : Names of custom columns not covered by the standard schema.
    """
    usual_columns = [
        "Time",
        "ExperimentTime",
        "Point",
        "raw",
        "HatchTime",
        "M1",
        "M2",
        "M3",
        "M4",
        "Arrest",
        "Ignore",
        "Death",
        "Dead",
    ]

    usual_columns.extend([column for column in filemap.columns if "qc" in column])

    for feature in FEATURES_TO_COMPUTE_AT_MOLT:
        usual_columns.extend(
            [column for column in filemap.columns if feature in column]
        )

    usual_columns.extend([column for column in filemap.columns if "analysis" in column])

    feature_columns = []
    for feature in FEATURES_TO_COMPUTE_AT_MOLT:
        feature_columns.extend(
            [
                column
                for column in filemap.columns
                if feature in column and "_at_" not in column
            ]
        )

    custom_columns = [
        column for column in filemap.columns if column not in usual_columns
    ]

    return custom_columns


# TODO: Instead of doing it for each condition, do it for all conditions at once, then split
def _process_condition_id_plotting_structure(
    experiment_dir: str,
    experiment_filemap: pl.DataFrame,
    filemap_path: str,
    organ_channels: dict[str, str],
    conditions_keys: list[str],
    condition_id: int,
    custom_columns: list[str] | None = None,
    recompute_values_at_molt: bool = False,
    rescale_n_points: int = 100,
) -> dict:
    """
    Build the condition dict for a single condition ID.

    Extracts time series, ecdysis events, QC classifications, worm feature arrays,
    and optional custom columns for all points belonging to ``condition_id``.

    Parameters:
        experiment_dir (str) : Path to the experiment directory.
        experiment_filemap (polars.DataFrame) : Full filemap with condition metadata
            already merged in.
        filemap_path (str) : Path to the CSV filemap file (stored in the output dict).
        organ_channels (dict[str, str]) : Mapping of organ name → filemap channel prefix
            (e.g. ``{"body": "ch2", "pharynx": "ch1"}``).
        conditions_keys (list[str]) : Metadata keys from the conditions YAML to copy
            into the condition dict.
        condition_id (int) : Condition identifier to process.
        custom_columns (list[str] or None) : Additional filemap columns to include.
            Defaults to ``None``.
        recompute_values_at_molt (bool) : If ``True``, recompute at-molt values even
            when they are already present in the filemap.  Defaults to ``False``.
        rescale_n_points (int) : Unused; reserved for future use.  Defaults to ``100``.

    Returns:
        dict : Condition dict containing time arrays, ecdysis indices, QC, feature
            series, and all metadata keys.
    """
    condition_df = experiment_filemap.filter(pl.col("condition_id") == condition_id)
    condition_dict = {}

    for key in conditions_keys:
        condition_dict[key] = condition_df.select(pl.col(key))[0].item()

    (
        time,
        experiment_time,
        ecdysis_index,
        ecdysis_time_step,
        ecdysis_experiment_time,
        larval_stage_durations_time_step,
        larval_stage_durations_experiment_time,
    ) = _get_time_ecdysis_and_durations(condition_df)
    death, arrest = _get_death_and_arrest(condition_df)

    n_points = condition_df.select(pl.col("Point")).n_unique()

    condition_dict.update(
        {
            "condition_id": int(condition_dict["condition_id"]),
            "ecdysis_index": ecdysis_index,
            "ecdysis_time_step": ecdysis_time_step,
            "larval_stage_durations_time_step": larval_stage_durations_time_step,
            "ecdysis_experiment_time": ecdysis_experiment_time,
            "ecdysis_experiment_time_hours": ecdysis_experiment_time / 3600,
            "larval_stage_durations_experiment_time": larval_stage_durations_experiment_time,
            "larval_stage_durations_experiment_time_hours": larval_stage_durations_experiment_time
            / 3600,
            "experiment": np.full((n_points, 1), experiment_dir),
            "filemap_path": np.full((n_points, 1), filemap_path),
            "point": condition_df.select(
                pl.col("Point").unique(maintain_order=True)
            ).to_numpy(),
            "time": time,
            "experiment_time": experiment_time,
            "experiment_time_hours": experiment_time / 3600,
            "death": death,
            "arrest": arrest,
        }
    )

    qc_columns = [
        col for col in condition_df.columns if "qc" in col or "worm_type" in col
    ]

    for organ in organ_channels.keys():
        organ_channel = organ_channels[organ]
        organ_columns = [
            col for col in condition_df.columns if col.startswith(organ_channel)
        ]

        # remove any column with _at_ in it
        organ_columns = [col for col in organ_columns if "_at_" not in col]

        organ_qc_columns = [
            col for col in organ_columns if "qc" in col or "worm_type" in col
        ]
        if len(organ_qc_columns) == 0:
            organ_qc_columns.append(qc_columns[0])

        renamed_organ_qc_columns = [
            col.replace(organ_channel, organ) for col in organ_qc_columns
        ]
        renamed_organ_qc_columns = [
            col.replace("worm_type", "qc") for col in renamed_organ_qc_columns
        ]

        for column, renamed_column in zip(organ_qc_columns, renamed_organ_qc_columns):
            qc_values = separate_column_by_point(condition_df, column)
            condition_dict[renamed_column] = qc_values
            if column in organ_columns:
                organ_columns.remove(column)

        # get the columns that contain the interesting features
        organ_feature_columns = []
        for feature in FEATURES_TO_COMPUTE_AT_MOLT:
            organ_feature_columns.extend(
                [col for col in organ_columns if feature in col]
            )

        for organ_feature_column in organ_feature_columns:
            # rename the column to remove the organ channel
            renamed_feature_organ_column = organ_feature_column.replace(
                organ_channel, organ
            )

            qc_key = find_best_string_match(
                renamed_feature_organ_column, renamed_organ_qc_columns
            )
            qc = condition_dict[qc_key]

            condition_dict[renamed_feature_organ_column] = separate_column_by_point(
                condition_df, organ_feature_column
            )
            condition_dict[f"{renamed_feature_organ_column}_at_ecdysis"] = (
                _get_values_at_molt(
                    condition_df, organ_feature_column, ecdysis_time_step
                )
            )

            condition_dict = _compute_values_at_molt(
                condition_dict,
                renamed_feature_organ_column,
                qc,
                recompute_values_at_molt=recompute_values_at_molt,
            )

    # Add custom columns if they exist
    if custom_columns is not None:
        for custom_column in custom_columns:
            if custom_column in condition_df.columns:
                condition_dict[custom_column] = separate_column_by_point(
                    condition_df, custom_column
                )

    return condition_dict


def build_plotting_struct(
    experiment_dir: str,
    filemap_path: str,
    conditions_yaml_path: str | dict,
    organ_channels: dict[str, str] = {"body": "ch2", "pharynx": "ch1"},
    recompute_values_at_molt: bool = False,
    rescale_n_points: int = 100,
) -> tuple[list[dict], list[dict]]:
    """
    Build the plotting structure for a single experiment.

    Reads the filemap CSV and the conditions YAML, merges condition metadata into
    the filemap, and produces a list of condition dicts ready for plotting.

    Parameters:
        experiment_dir (str) : Path to the experiment directory.
        filemap_path (str) : Path to the CSV filemap file.
        conditions_yaml_path (str or dict) : Path to the conditions YAML file or
            an already-parsed conditions dict.
        organ_channels (dict[str, str]) : Mapping of organ name → filemap channel
            prefix.  Defaults to ``{"body": "ch2", "pharynx": "ch1"}``.
        recompute_values_at_molt (bool) : If ``True``, recompute at-molt values even
            when they are already stored in the filemap.  Defaults to ``False``.
        rescale_n_points (int) : Unused; reserved for future use.  Defaults to ``100``.

    Returns:
        tuple[list[dict], list[dict]] : ``(conditions_struct, conditions_info)`` where
            ``conditions_struct`` is the full list of condition dicts sorted by
            ``condition_id``, and ``conditions_info`` contains only the metadata keys
            from the YAML, also sorted by ``condition_id``.
    """
    experiment_filemap = read_filemap(filemap_path)

    custom_columns = _get_custom_columns(experiment_filemap)
    if not custom_columns:
        custom_columns = None

    conditions = build_conditions(conditions_yaml_path)
    conditions_keys = list(conditions[0].keys())

    # remove 'point_range' and 'pad' from the conditions keys if they are present
    if "point_range" in conditions_keys:
        conditions_keys.remove("point_range")
    if "pad" in conditions_keys:
        conditions_keys.remove("pad")

    experiment_filemap = _add_conditions_to_filemap(
        experiment_filemap,
        conditions,
    )

    # if ExperimentTime is not present in the filemap, add it
    if "ExperimentTime" not in experiment_filemap.columns:
        experiment_filemap = experiment_filemap.with_columns(
            pl.lit(np.nan).alias("ExperimentTime")
        )

    # remove rows where condition_id is null
    experiment_filemap = experiment_filemap.filter(~pl.col("condition_id").is_null())
    # set molts that should be ignored to NaN
    if "Ignore" in experiment_filemap.columns:
        experiment_filemap = remove_ignored_molts(experiment_filemap)

    # remove rows where Ignore is True
    if "Ignore" in experiment_filemap.columns:
        experiment_filemap = experiment_filemap.filter(~pl.col("Ignore"))

    conditions_struct = []

    for condition_id in (
        experiment_filemap.select(pl.col("condition_id"))
        .unique(maintain_order=True)
        .to_numpy()
        .squeeze()
    ):
        condition_dict = _process_condition_id_plotting_structure(
            experiment_dir,
            experiment_filemap,
            filemap_path,
            organ_channels,
            conditions_keys,
            condition_id,
            custom_columns=custom_columns,
            recompute_values_at_molt=recompute_values_at_molt,
            rescale_n_points=rescale_n_points,
        )

        conditions_struct.append(condition_dict)

    conditions_info = [
        {key: condition[key] for key in conditions_keys}
        for condition in conditions_struct
    ]

    # sort the conditions and conditions_info by condition_id
    conditions_struct = sorted(conditions_struct, key=lambda x: x["condition_id"])
    conditions_info = sorted(conditions_info, key=lambda x: x["condition_id"])

    return conditions_struct, conditions_info


def _compute_larval_stage_duration(ecdysis_array: np.ndarray) -> np.ndarray:
    """
    Compute the duration of each larval stage from consecutive ecdysis times.

    Parameters:
        ecdysis_array (array-like) : Ecdysis time values for one worm
            (length 5: Hatch, M1, M2, M3, M4).

    Returns:
        np.ndarray : Durations of shape ``(4,)`` (L1–L4); NaN where either
            boundary event is NaN.
    """
    durations = np.full(len(ecdysis_array) - 1, np.nan)
    for i, (start, end) in enumerate(zip(ecdysis_array[:-1], ecdysis_array[1:])):
        # check if start or end is NaN
        if np.isnan(start) or np.isnan(end):
            durations[i] = np.nan
        else:
            durations[i] = end - start
    return durations


def _get_time_ecdysis_and_durations(filemap: pl.DataFrame) -> tuple:
    """
    Extract per-point time arrays, ecdysis indices, and larval stage durations.

    Parameters:
        filemap (polars.DataFrame) : Condition filemap containing ``"Point"``,
            ``"Time"``, ``"ExperimentTime"``, and the five ecdysis columns
            (``"HatchTime"``, ``"M1"``–``"M4"``).

    Returns:
        tuple :
            - time (np.ndarray) : Shape ``(n_points, n_frames)`` time-step values.
            - experiment_time (np.ndarray) : Shape ``(n_points, n_frames)`` experiment
              time in seconds.
            - ecdysis_index (np.ndarray) : Shape ``(n_points, 5)`` frame indices of
              ecdysis events.
            - ecdysis_time_step (np.ndarray) : Shape ``(n_points, 5)`` ecdysis times
              in raw time steps.
            - ecdysis_experiment_time (np.ndarray) : Shape ``(n_points, 5)`` ecdysis
              times in seconds.
            - larval_stage_durations_time_step (np.ndarray) : Shape ``(n_points, 4)``
              stage durations in raw time steps.
            - larval_stage_durations_experiment_time (np.ndarray) : Shape
              ``(n_points, 4)`` stage durations in seconds.
    """
    all_ecdysis_time_step = []
    all_ecdysis_index = []
    all_durations_time_step = []

    all_ecdysis_experiment_time = []
    all_durations_experiment_time = []

    ecdysis_columns = ["HatchTime", "M1", "M2", "M3", "M4"]
    column_list = ["Point", "Time", "ExperimentTime"] + ecdysis_columns
    filemap = filemap.select(pl.col(column_list))

    ecdysis_values = (
        filemap.group_by("Point", maintain_order=True)
        .agg(pl.col(ecdysis_columns).first())
        .drop("Point")
        .cast(pl.Float64)
    )

    time = separate_column_by_point(filemap, "Time").astype(float)
    experiment_time = separate_column_by_point(filemap, "ExperimentTime").astype(float)

    for i in range(len(ecdysis_values)):
        ecdysis = ecdysis_values[i].to_numpy().squeeze()
        time_of_point = time[i]
        experiment_time_of_point = experiment_time[i]

        ecdysis_index = [
            (
                float(np.where(time_of_point == ecdysis)[0][0])
                if ecdysis in time_of_point
                else np.nan
            )
            for ecdysis in ecdysis
        ]
        ecdysis_experiment_time = [
            experiment_time_of_point[int(index)] if not np.isnan(index) else np.nan
            for index in ecdysis_index
        ]
        all_ecdysis_time_step.append(ecdysis)
        all_ecdysis_index.append(ecdysis_index)
        all_ecdysis_experiment_time.append(ecdysis_experiment_time)

        larval_stage_durations = _compute_larval_stage_duration(ecdysis)
        larval_stage_durations_experiment_time = _compute_larval_stage_duration(
            ecdysis_experiment_time
        )
        all_durations_time_step.append(larval_stage_durations)
        all_durations_experiment_time.append(larval_stage_durations_experiment_time)

    return (
        time,
        experiment_time,
        np.array(all_ecdysis_index),
        np.array(all_ecdysis_time_step),
        np.array(all_ecdysis_experiment_time),
        np.array(all_durations_time_step),
        np.array(all_durations_experiment_time),
    )


def _get_values_at_molt(
    filemap: pl.DataFrame,
    column: str,
    ecdysis_time_step: np.ndarray,
) -> np.ndarray:
    """
    Retrieve precomputed at-ecdysis values for a column from the filemap.

    Reads the ``{column}_at_{event}`` columns for each of the five ecdysis events.
    Entries corresponding to NaN ecdysis times are set to NaN.

    Parameters:
        filemap (polars.DataFrame) : Condition filemap with the at-ecdysis columns.
        column (str) : Base column name (without the ``_at_*`` suffix).
        ecdysis_time_step (np.ndarray) : Ecdysis time array of shape
            ``(n_points, 5)`` used to mask NaN events.

    Returns:
        np.ndarray : At-ecdysis values of shape ``(n_points, 5)``; NaN where
            the corresponding ecdysis event is NaN.
    """
    ecdysis = ["HatchTime", "M1", "M2", "M3", "M4"]
    columns_at_ecdysis = [f"{column}_at_{e}" for e in ecdysis]
    column_list = ["Point"] + columns_at_ecdysis

    # if the column_at_ecdysis does not exist, create it
    for col in columns_at_ecdysis:
        if col not in filemap.columns:
            filemap = filemap.with_columns(
                pl.lit(np.nan).alias(col),
            )

    filemap = filemap.select(pl.col(column_list))

    values_at_ecdysis = (
        (
            filemap.group_by("Point", maintain_order=True)
            .agg(pl.col(columns_at_ecdysis).first())
            .drop("Point")
            .cast(pl.Float64)
        )
        .to_numpy()
        .squeeze()
    )

    # handle a edge case where there is only one point in the filemap
    if values_at_ecdysis.ndim == 1:
        values_at_ecdysis = values_at_ecdysis[np.newaxis, :]

    # Set all values at molt at the same index as nan ecdysis to nan
    nan_mask = np.isnan(ecdysis_time_step)
    values_at_ecdysis[nan_mask] = np.nan

    return values_at_ecdysis


def _get_death_and_arrest(filemap: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract per-point death time and arrest flag from the filemap.

    If ``"Death"`` or ``"Arrest"`` columns are absent, they are filled with NaN
    and False respectively.

    Parameters:
        filemap (polars.DataFrame) : Condition filemap.

    Returns:
        tuple[np.ndarray, np.ndarray] :
            - death (np.ndarray) : Shape ``(n_points, 1)`` death times (float, NaN if alive).
            - arrest (np.ndarray) : Shape ``(n_points, 1)`` arrest flags (bool).
    """
    column_list = ["Point", "Death", "Arrest"]
    if "Death" not in filemap.columns:
        filemap = filemap.with_columns(
            pl.lit(np.nan).alias("Death"),
        )
    if "Arrest" not in filemap.columns:
        filemap = filemap.with_columns(
            pl.lit(False).alias("Arrest"),
        )

    filemap = filemap.select(pl.col(column_list))

    death_and_arrest = (
        (
            filemap.group_by("Point", maintain_order=True)
            .agg(pl.col(["Death", "Arrest"]).first())
            .drop("Point")
        )
        .to_numpy()
        .squeeze()
    )

    # handle a edge case where there is only one point in the filemap
    if death_and_arrest.ndim == 1:
        death_and_arrest = death_and_arrest[np.newaxis, :]

    death = death_and_arrest[:, 0].astype(float)
    arrest = death_and_arrest[:, 1].astype(bool)
    return death[:, np.newaxis], arrest[:, np.newaxis]


def _compute_values_at_molt(
    condition_dict: dict,
    column: str,
    worm_types: np.ndarray,
    recompute_values_at_molt: bool = False,
) -> dict:
    """
    Fill missing (or recompute all) at-molt values for a column using the time series.

    Missing values are identified as NaN entries in ``{column}_at_ecdysis`` where the
    corresponding ecdysis time is not NaN.  ``compute_series_at_time_classified`` is
    called to interpolate the value from the raw time series.

    Parameters:
        condition_dict (dict) : Condition dict containing the column, its at-ecdysis
            counterpart, time arrays, and ecdysis times.
        column (str) : Base column name.
        worm_types (np.ndarray) : QC classification array of shape
            ``(n_points, n_frames)`` used by the interpolation.
        recompute_values_at_molt (bool) : If ``True``, recompute all at-molt values
            regardless of existing values.  Defaults to ``False``.

    Returns:
        dict : The updated ``condition_dict`` with ``{column}_at_ecdysis`` filled in.
    """
    column_at_molt = f"{column}_at_ecdysis"

    values_at_molt = condition_dict[column_at_molt]
    updated_values_at_molt = values_at_molt.copy()

    nan_indexes_values_mask = np.isnan(values_at_molt)
    experiment_time = condition_dict["experiment_time_hours"]

    if (~np.isnan(experiment_time)).any():
        time = condition_dict["experiment_time_hours"]
        ecdysis = condition_dict["ecdysis_experiment_time_hours"]
    else:
        time = condition_dict["time"]
        ecdysis = condition_dict["ecdysis_index"]

    non_nan_indexes_ecdysis_mask = np.invert(np.isnan(ecdysis))

    if recompute_values_at_molt:
        values_to_recompute_mask = non_nan_indexes_ecdysis_mask
    else:
        values_to_recompute_mask = (
            nan_indexes_values_mask & non_nan_indexes_ecdysis_mask
        )

    for i in range(len(values_to_recompute_mask)):
        mask = values_to_recompute_mask[i]
        idx_values_to_recompute = np.where(mask)[0]

        if len(idx_values_to_recompute) == 0:
            continue

        ecdys = ecdysis[i][idx_values_to_recompute]

        recomputed_values = compute_series_at_time_classified(
            condition_dict[column][i],
            ecdys,
            time[i],
            worm_types[i],
        )

        updated_values_at_molt[i][idx_values_to_recompute] = recomputed_values

    condition_dict[column_at_molt] = updated_values_at_molt
    return condition_dict


def separate_column_by_point(filemap: pl.DataFrame, column: str) -> np.ndarray:
    """
    Pivot a long-format filemap column into a 2-D array indexed by point.

    Each row of the output corresponds to one imaging point, in order of first
    appearance in the filemap; columns correspond to time frames. Numeric, boolean
    and entirely-null columns are returned as floats with nulls and padding set to
    NaN. String-like columns are returned as objects, with nulls kept as ``None``
    and shorter points right-padded with ``"error"``. Any other dtype is returned
    as objects padded with ``None``.

    Parameters:
        filemap (polars.DataFrame): Filemap with at least ``"Point"`` and
            ``column`` columns.
        column (str): Column to pivot.

    Returns:
        np.ndarray: Array of shape ``(n_points, max_n_frames)``.

    Raises:
        ValueError: If ``"Point"`` or ``column`` is missing from the filemap.
    """
    for required_column in ("Point", column):
        if required_column not in filemap.columns:
            raise ValueError(
                f"Column '{required_column}' not found in filemap, available "
                f"columns are: {filemap.columns}"
            )

    filemap_points = filemap.select(pl.col("Point"), pl.col(column))
    if filemap_points.height == 0:
        return np.full((0, 0), np.nan)

    dtype = filemap_points.schema[column]
    all_null = filemap_points.get_column(column).null_count() == filemap_points.height

    if all_null or dtype.is_numeric() or dtype == pl.Boolean:
        filemap_points = filemap_points.with_columns(
            pl.col(column).cast(pl.Float64, strict=False)
        )
        fill_value, result_dtype = np.nan, float
    elif dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum):
        filemap_points = filemap_points.with_columns(pl.col(column).cast(pl.String))
        fill_value, result_dtype = "error", object
    else:
        fill_value, result_dtype = None, object

    point_dataframes = filemap_points.partition_by("Point", maintain_order=True)
    max_height = max(point_df.height for point_df in point_dataframes)
    result = np.full(
        (len(point_dataframes), max_height), fill_value, dtype=result_dtype
    )

    for i, point_df in enumerate(point_dataframes):
        point_series = point_df.get_column(column)
        point_column = (
            point_series.to_numpy() if result_dtype is float else point_series.to_list()
        )
        result[i, : len(point_column)] = point_column
    return result


def remove_ignored_molts(filemap: pl.DataFrame) -> pl.DataFrame:
    """
    Set molt-time columns to null for time points flagged with ``Ignore=True``.

    For each ecdysis column, any molt time that coincides with an ignored
    ``(Point, Time)`` pair is replaced with ``None``.

    Parameters:
        filemap (polars.DataFrame) : Filemap with ecdysis columns and an optional
            ``"Ignore"`` column.

    Returns:
        polars.DataFrame : Updated filemap; unchanged if ``"Ignore"`` is absent.
    """
    molt_columns = ["HatchTime", "M1", "M2", "M3", "M4"]

    # Only process if "Ignore" column exists
    if "Ignore" not in filemap.columns:
        return filemap

    # Get all rows where Ignore is True
    ignored_rows = filemap.filter(pl.col("Ignore"))

    # Build a set of (Point, Time) pairs to ignore
    ignored_points = ignored_rows.select(pl.col("Point")).to_numpy().flatten()
    ignored_times = ignored_rows.select(pl.col("Time")).to_numpy().flatten()
    ignored_pairs = set(zip(ignored_points, ignored_times))

    # For each molt column, set to None where (Point, molt_time) is in ignored_pairs
    for col in molt_columns:
        molt_times = filemap.select(pl.col(col)).to_numpy().flatten()
        points = filemap.select(pl.col("Point")).to_numpy().flatten()
        mask = np.array(
            [
                (
                    (p, mt) in ignored_pairs
                    if mt is not None and not (isinstance(mt, float) and np.isnan(mt))
                    else False
                )
                for p, mt in zip(points, molt_times)
            ]
        )
        if mask.any():
            filemap = filemap.with_columns(
                pl.when(pl.Series(mask)).then(None).otherwise(pl.col(col)).alias(col)
            )

    return filemap


def remove_unwanted_info(conditions_info: list[dict]) -> list[dict]:
    """
    Remove ``"description"`` and ``"condition_id"`` from each entry in conditions_info.

    Parameters:
        conditions_info (list[dict]) : List of condition metadata dicts.

    Returns:
        list[dict] : The modified list with the two keys removed in place.
    """
    for condition in conditions_info:
        if "description" in condition.keys():
            condition.pop("description")
        if "condition_id" in condition.keys():
            condition.pop("condition_id")
    return conditions_info


def combine_experiments(
    filemap_paths: list[str],
    config_paths: list[str],
    experiment_dirs: list[str] | None = None,
    organ_channels: list[dict] | dict = [{"body": "ch2", "pharynx": "ch1"}],
    recompute_values_at_molt: bool = False,
    rescale_n_points: int = 100,
) -> list[dict]:
    """
    Build and merge the plotting structure from multiple experiments.

    Each experiment is processed independently via ``build_plotting_struct``.
    Conditions with identical metadata (after removing ``description`` and
    ``condition_id``) are merged by concatenating their numpy arrays along axis 0.
    Condition IDs in the merged structure are reassigned sequentially.

    Parameters:
        filemap_paths (list[str]) : Paths to the CSV filemap files, one per experiment.
        config_paths (list[str]) : Paths to the conditions YAML files, one per experiment.
        experiment_dirs (list[str] or None) : Experiment directories; defaults to the
            parent directory of each filemap when ``None``.
        organ_channels (list[dict] or dict) : Organ-to-channel mapping(s).  A single
            dict is broadcast to all experiments; otherwise one dict per experiment is
            expected.  Defaults to ``[{"body": "ch2", "pharynx": "ch1"}]``.
        recompute_values_at_molt (bool) : Passed to ``build_plotting_struct``.
            Defaults to ``False``.
        rescale_n_points (int) : Passed to ``build_plotting_struct``.
            Defaults to ``100``.

    Returns:
        list[dict] : Merged conditions_struct with sequential condition IDs.

    Raises:
        ValueError : If the length of ``organ_channels`` is neither 1 nor equal to
            the number of experiments.
    """
    all_conditions_struct = []
    condition_info_merge_list = []
    conditions_info_keys = set()
    condition_id_counter = 0

    if isinstance(organ_channels, dict):
        organ_channels = [organ_channels]

    if len(organ_channels) == 1:
        organ_channels = organ_channels * len(filemap_paths)
    elif len(organ_channels) != len(filemap_paths):
        raise ValueError(
            "Number of organ channels must be equal to the number of experiments."
        )

    # Process each experiment
    for i, (filemap_path, config_path, organ_channel) in enumerate(
        zip(filemap_paths, config_paths, organ_channels)
    ):
        experiment_dir = (
            experiment_dirs[i] if experiment_dirs else os.path.dirname(filemap_path)
        )

        conditions_struct, conditions_info = build_plotting_struct(
            experiment_dir,
            filemap_path,
            config_path,
            organ_channels=organ_channel,
            recompute_values_at_molt=recompute_values_at_molt,
            rescale_n_points=rescale_n_points,
        )

        # Process conditions for this experiment
        for condition in conditions_struct:
            condition["condition_id"] = condition_id_counter
            condition_id_counter += 1
            all_conditions_struct.append(condition)

        # Process condition info
        experiment_conditions_info = remove_unwanted_info(conditions_info)
        condition_info_merge_list.extend(experiment_conditions_info)
        conditions_info_keys.update(
            *[condition.keys() for condition in experiment_conditions_info]
        )

    # Merge conditions based on their info
    condition_dict = defaultdict(list)
    for i, condition_info in enumerate(condition_info_merge_list):
        key = frozenset(condition_info.items())
        condition_dict[key].append(i)

    merged_conditions_struct = []
    for indices in condition_dict.values():
        base_condition = all_conditions_struct[indices[0]]
        for idx in indices[1:]:
            for key, value in all_conditions_struct[idx].items():
                if key not in conditions_info_keys:
                    if isinstance(value, np.ndarray):
                        if key not in base_condition:
                            base_condition[key] = value
                            continue
                        if value.shape[1] > base_condition[key].shape[1]:
                            base_condition[key] = np.pad(
                                base_condition[key],
                                (
                                    (0, 0),
                                    (0, value.shape[1] - base_condition[key].shape[1]),
                                ),
                                mode="constant",
                                constant_values=np.nan,
                            )
                        elif value.shape[1] < base_condition[key].shape[1]:
                            value = np.pad(
                                value,
                                (
                                    (0, 0),
                                    (0, base_condition[key].shape[1] - value.shape[1]),
                                ),
                                mode="constant",
                                constant_values=np.nan,
                            )
                    try:
                        base_condition[key] = np.concatenate(
                            (base_condition[key], value), axis=0
                        )
                    except ValueError as e:
                        print(f"Could not concatenate {key}: {e}")

        merged_conditions_struct.append(base_condition)

    # # Sort and reassign condition IDs
    # merged_conditions_struct.sort(key=lambda x: x['condition_id'])
    for i, condition in enumerate(merged_conditions_struct):
        condition["condition_id"] = i

    return merged_conditions_struct
