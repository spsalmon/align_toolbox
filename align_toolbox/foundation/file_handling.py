import os
import re
from pathlib import Path

import numpy as np
import polars as pl


def extract_time_point(
    path: str,
    time_regex: str = r"Time(\d+)",
    point_regex: str = r"Point(\d+)",
) -> tuple[int, int]:
    r"""
    Extract time and point information from a file name using regular expressions.

    Parameters:
        path (str): The file path or name to extract information from.
        time_regex (str): Regular expression pattern to extract time information. (default: r"Time(\d+)")
        point_regex (str): Regular expression pattern to extract point information. (default: r"Point(\d+)")

    Returns:
        tuple: A tuple containing the extracted time and point as integers.
    """

    time_pattern = re.compile(time_regex)
    point_pattern = re.compile(point_regex)

    time_match = time_pattern.search(path)
    point_match = point_pattern.search(path)
    if time_match and point_match:
        time = int(time_match.group(1))
        point = int(point_match.group(1))
        return time, point
    else:
        raise ValueError("Could not extract time and point from file name.")


def get_all_timepoints_from_dir(
    dir_path: str,
    time_regex: str = r"Time(\d+)",
    point_regex: str = r"Point(\d+)",
) -> list[dict]:
    r"""
    Retrieve all time points and corresponding image paths from a directory.

    Parameters:
        dir_path (str): The path to the directory containing the images.
        time_regex (str): Regular expression pattern to extract time information from file names. (default: r"Time(\d+)")
        point_regex (str): Regular expression pattern to extract point information from file names. (default: r"Point(\d+)")

    Returns:
        list: A list of dictionaries, each containing the time, point, and image path.
    """

    time_pattern = re.compile(time_regex)
    point_pattern = re.compile(point_regex)

    timepoint_list = []

    # Get a list of file paths in the directory (excluding subdirectories)
    image_paths = [
        os.path.join(dir_path, x)
        for x in os.listdir(dir_path)
        if not os.path.isdir(os.path.join(dir_path, x))
    ]

    for image_path in image_paths:
        time_match = time_pattern.search(image_path)
        point_match = point_pattern.search(image_path)

        # Only add to list if both time and point are found
        if time_match and point_match:
            time = int(time_match.group(1))
            point = int(point_match.group(1))

            timepoint_list.append(
                {"Time": time, "Point": point, "ImagePath": image_path}
            )

    return timepoint_list


def fill_empty_timepoints(
    filemap: pl.DataFrame,
) -> pl.DataFrame:
    """
    Fill in missing time points in a filemap dataframe with null image paths.

    Parameters:
        filemap (pl.DataFrame): The filemap dataframe containing 'Time', 'Point', and 'ImagePath' columns.

    Returns:
        pl.DataFrame: The filled filemap dataframe with missing time points included.
    """
    all_points = (
        filemap.select(pl.col("Point")).unique(maintain_order=True).to_numpy().squeeze()
    )
    if all_points.ndim == 0:
        all_points = np.array([all_points])

    all_times = (
        filemap.select(pl.col("Time")).unique(maintain_order=True).to_numpy().squeeze()
    )

    if all_times.ndim == 0:
        all_times = np.array([all_times])

    missing_times = []

    for point in all_points:
        # Get the unique times associated with the current point.
        times_of_point = np.atleast_1d(
            filemap.filter(pl.col("Point") == point)
            .select(pl.col("Time"))
            .to_numpy()
            .squeeze()
        )

        missing = set(all_times) - set(times_of_point)
        missing_times.extend(
            [{"Time": time, "Point": point, "ImagePath": None} for time in missing]
        )

    if missing_times:
        filemap_extended = pl.DataFrame(missing_times)
        filled_filemap = pl.concat([filemap, filemap_extended]).sort(["Point", "Time"])
    else:
        filled_filemap = filemap.sort(["Point", "Time"])

    return filled_filemap


def get_dir_filemap(
    dir_path: str,
    time_regex: str = r"Time(\d+)",
    point_regex: str = r"Point(\d+)",
) -> pl.DataFrame:
    r"""
    Get the filemap dataframe for a directory by retrieving all time points and filling in missing time points.

    Parameters:
        dir_path (str): The path to the directory containing the images.
        time_regex (str): Regular expression pattern to extract time information from file names. (default: r"Time(\d+)")
        point_regex (str): Regular expression pattern to extract point information from file names. (default: r"Point(\d+)")

    Returns:
        pl.DataFrame: The filemap dataframe with 'Time', 'Point', and 'ImagePath' columns.
    """
    timepoint_list = get_all_timepoints_from_dir(dir_path, time_regex, point_regex)
    filemap = pl.DataFrame(timepoint_list)
    filled_filemap = fill_empty_timepoints(filemap)

    return filled_filemap


def get_experiment_dir_filemap(
    dir_path: str,
    raw_dir: str = "raw",
    analysis_dir: str = "analysis",
    time_regex: str = r"Time(\d+)",
    point_regex: str = r"Point(\d+)",
) -> pl.DataFrame:
    r"""
    Get the filemap dataframe for an experiment directory.

    Retrieves time points from the 'raw' directory and fills in missing ones then adds paths
    from the 'analysis' subdirectories.

    Parameters:
        dir_path (str): Base directory path for the experiment.
        raw_dir (str): Subdirectory name for raw images. (default: "raw")
        analysis_dir (str): Subdirectory name for analysis output. (default: "analysis")
        time_regex (str): Regular expression pattern to extract time information from file names. (default: r"Time(\d+)")
        point_regex (str): Regular expression pattern to extract point information from file names. (default: r"Point(\d+)")

    Returns:
        pl.DataFrame: Extended filemap dataframe including both raw and analysis image paths.
    """
    raw_timepoint_list = get_all_timepoints_from_dir(
        os.path.join(dir_path, raw_dir), time_regex, point_regex
    )
    raw_filemap = pl.DataFrame(raw_timepoint_list)
    experiment_filemap = fill_empty_timepoints(raw_filemap)
    experiment_filemap = experiment_filemap.rename({"ImagePath": raw_dir})

    analysis_dir = os.path.join(dir_path, analysis_dir)
    if os.path.exists(analysis_dir):
        subdir_list = [x[0] for x in os.walk(analysis_dir)]
        for subdir in subdir_list:
            if subdir != analysis_dir:
                timepoint_list = get_all_timepoints_from_dir(
                    subdir, time_regex, point_regex
                )
                filemap = pl.DataFrame(timepoint_list)
                filemap = fill_empty_timepoints(filemap)
                filemap = filemap.rename(
                    {"ImagePath": os.path.join(analysis_dir, os.path.basename(subdir))},
                )
                experiment_filemap = experiment_filemap.join(
                    filemap, on=["Time", "Point"], how="left"
                )
    return experiment_filemap


def add_dir_to_experiment_filemap(
    experiment_filemap: pl.DataFrame,
    dir_path: str,
    subdir_name: str,
    time_regex: str = r"Time(\d+)",
    point_regex: str = r"Point(\d+)",
) -> pl.DataFrame:
    r"""
    Add the images contained in a directory to an existing filemap as a new column.

    Parameters:
        experiment_filemap (pl.DataFrame): Existing filemap dataframe with at least
            ``"Time"`` and ``"Point"`` columns.
        dir_path (str): The path to the directory containing the images.
        subdir_name (str): The name of the new column to be added to the filemap.
        time_regex (str): Regular expression pattern to extract time information from
            file names. (default: r"Time(\d+)")
        point_regex (str): Regular expression pattern to extract point information from
            file names. (default: r"Point(\d+)")

    Returns:
        pl.DataFrame: Updated filemap dataframe with the new column added, missing
            entries left null.
    """
    subdir_filemap = get_dir_filemap(dir_path, time_regex, point_regex)
    subdir_filemap = subdir_filemap.rename({"ImagePath": subdir_name})
    # check if column already exists
    if subdir_name in experiment_filemap.columns:
        experiment_filemap = experiment_filemap.drop(subdir_name)
    experiment_filemap = experiment_filemap.join(
        subdir_filemap, on=["Time", "Point"], how="left"
    )
    return experiment_filemap


def normalize_filemap_dtypes(
    filemap: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Enforce the filemap convention that missing values are null.

    Empty strings and float NaNs become null. For a ``DataFrame``, String columns
    whose non-null values all parse as numbers are also cast to ``Int64`` or
    ``Float64``, healing filemaps whose numeric columns were persisted with ``""``
    fills. This inference needs the data, so it is skipped for a ``LazyFrame``.

    Parameters:
        filemap (pl.DataFrame | pl.LazyFrame): The filemap to normalize.

    Returns:
        pl.DataFrame | pl.LazyFrame: The normalized filemap.
    """
    filemap = filemap.with_columns(pl.col(pl.String).replace("", None))

    if isinstance(filemap, pl.DataFrame):
        numeric_columns = []
        for column in filemap.select(pl.col(pl.String)).columns:
            values = filemap[column]
            if values.null_count() == len(values):
                continue
            for dtype in (pl.Int64, pl.Float64):
                parsed = values.str.strip_chars().cast(dtype, strict=False)
                if parsed.null_count() == values.null_count():
                    numeric_columns.append(parsed)
                    break
        filemap = filemap.with_columns(numeric_columns)

    return filemap.with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))


def read_filemap(
    filemap_path: str, lazy_loading: bool = False
) -> pl.DataFrame | pl.LazyFrame:
    """
    Read a filemap from a CSV or Parquet file using Polars.

    The format is taken from the file extension (``.parquet`` → Parquet, anything
    else → CSV). If the file is missing, the sibling path with the other
    extension is tried instead. The result goes through
    ``normalize_filemap_dtypes``.

    Parameters:
        filemap_path (str): Path to the filemap file (``.csv`` or ``.parquet``).
        lazy_loading (bool, optional): If ``True``, return a ``LazyFrame``
            instead of a ``DataFrame``. (default: False)

    Returns:
        pl.DataFrame | pl.LazyFrame: The filemap.
    """
    null_values = ["np.nan", "[nan]", "", "NaN", "nan", "NA", "N/A"]

    def load(path: Path) -> pl.DataFrame | pl.LazyFrame:
        if path.suffix == ".parquet":
            return pl.scan_parquet(path) if lazy_loading else pl.read_parquet(path)
        read_csv = pl.scan_csv if lazy_loading else pl.read_csv
        return read_csv(path, infer_schema_length=10000, null_values=null_values)

    path = Path(filemap_path)
    fallback = path.with_suffix(".csv" if path.suffix == ".parquet" else ".parquet")

    try:
        filemap = load(path)
    except FileNotFoundError:
        print(f"File not found: {path}, trying {fallback} instead.")
        filemap = load(fallback)
    return normalize_filemap_dtypes(filemap)


def write_filemap(filemap: pl.DataFrame, filemap_path: str) -> None:
    """
    Write a filemap to a CSV or Parquet file using Polars.

    Detects the output format from the ``.parquet`` extension; otherwise writes CSV.
    The filemap goes through ``normalize_filemap_dtypes`` first, so empty strings
    and NaNs are never persisted.

    Parameters:
        filemap (pl.DataFrame): The filemap dataframe to write.
        filemap_path (str): Destination path for the filemap file (``.csv`` or ``.parquet``).

    Returns:
        None
    """
    filemap = normalize_filemap_dtypes(filemap)
    if filemap_path.endswith(".parquet"):
        filemap.write_parquet(filemap_path)
    else:
        filemap.write_csv(filemap_path)
