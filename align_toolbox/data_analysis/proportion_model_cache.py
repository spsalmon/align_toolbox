import json
import os
import pickle
import tempfile
from hashlib import sha256
from inspect import signature
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd

from align_toolbox.data_analysis.proportion_model import (
    ProportionModelResult,
    build_proportion_table,
    fit_per_experiment,
    fit_proportion_model,
    fit_provenance,
    table_data_hash,
    to_json_compatible,
)

__all__ = [
    "PerExperimentFits",
    "fit_or_load_per_experiment",
    "fit_or_load_proportion_model",
    "list_cached_models",
]

# bump when the key, the pickled objects or the sidecar change incompatibly
CACHE_FORMAT_VERSION = 2
# arguments that change how sampling runs, not what it produces
UNKEYED_FIT_ARGUMENTS = ("progressbar", "cores")
KINDS = {"joint": "proportion_model", "per_experiment": "per_experiment"}
KEY_LENGTH = 24
LISTED_COLUMNS = [
    "file",
    "kind",
    "column_x",
    "column_y",
    "sampling",
    "percentages",
    "conditions",
    "likelihood",
    "reference_shape",
    "reference_shape_used",
    "created",
]

_FIT_SIGNATURE = signature(fit_proportion_model)


class PerExperimentFits(dict):
    """
    Fits of ``fit_per_experiment`` keyed by experiment label, with the provenance of the request.

    Attributes:
        provenance (dict or None): Record of the data and settings, as for
            ``ProportionModelResult.provenance``, with the cache entry.
    """

    def __init__(
        self,
        fits: dict[str, ProportionModelResult] | None = None,
        provenance: dict | None = None,
    ) -> None:
        super().__init__(fits or {})
        self.provenance = provenance


def _fit_arguments(fit_kwargs: dict, reference_shape: str | None = None) -> dict:
    """
    Every argument of ``fit_proportion_model`` but the table, defaults filled in.

    Parameters:
        fit_kwargs (dict): Keyword arguments for ``fit_proportion_model``.
        reference_shape (str or None): Reference shape passed separately, for
            per-experiment fits. (default: None)

    Returns:
        dict: Argument name to value.

    Raises:
        TypeError: If ``fit_kwargs`` holds an argument ``fit_proportion_model``
            does not take, or lacks ``random_seed``.
    """
    if reference_shape is not None:
        fit_kwargs = {**fit_kwargs, "reference_shape": reference_shape}
    bound = _FIT_SIGNATURE.bind(None, **fit_kwargs)
    bound.apply_defaults()
    return {name: value for name, value in bound.arguments.items() if name != "table"}


def _request(table: pd.DataFrame, kind: str, fit_arguments: dict) -> dict:
    """JSON-compatible description of what an entry must have been fitted on and with."""
    attrs = table.attrs
    return to_json_compatible(
        {
            "format_version": CACHE_FORMAT_VERSION,
            "kind": kind,
            "data_hash": table_data_hash(table),
            "sampling": attrs["sampling"],
            "percentages": attrs["percentages"],
            "column_x": attrs["column_x"],
            "column_y": attrs["column_y"],
            "conditions": attrs["conditions"],
            "remove_hatch": attrs["remove_hatch"],
            "lmbda": attrs["lmbda"],
            "medfilt_window": attrs["medfilt_window"],
            "bspline_order": attrs["bspline_order"],
            "fit_arguments": {
                name: value
                for name, value in fit_arguments.items()
                if name not in UNKEYED_FIT_ARGUMENTS
            },
        }
    )


def _cache_key(request: dict) -> str:
    """Hexadecimal SHA-256 digest of a request serialized with sorted keys."""
    return sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def _entry_paths(cache_dir: Path, kind: str, key: str) -> tuple[Path, Path]:
    stem = f"{KINDS[kind]}_{key[:KEY_LENGTH]}"
    return cache_dir / f"{stem}.pkl", cache_dir / f"{stem}.json"


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to a temporary file next to ``path``, then rename it over ``path``."""
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_entry(
    pickle_path: Path, key: str, request: dict, expected_type: type
) -> object | None:
    """
    Load a cache entry if it is readable and was produced for ``request``.

    Returns:
        object or None: The cached object, or ``None`` after a warning if the
            entry cannot be read or does not match.
    """
    try:
        with open(pickle_path, "rb") as file:
            cached = pickle.load(file)
    except Exception as error:
        warn(
            f"The cached model {pickle_path.name} could not be read ({error!r}); "
            "refitting and overwriting it.",
            stacklevel=3,
        )
        return None
    provenance = getattr(cached, "provenance", None)
    entry = provenance.get("cache") if isinstance(provenance, dict) else None
    if (
        not isinstance(cached, expected_type)
        or not isinstance(entry, dict)
        or entry.get("key") != key
        or entry.get("request") != request
    ):
        warn(
            f"The cached model {pickle_path.name} does not record the requested data "
            "and settings; refitting and overwriting it.",
            stacklevel=3,
        )
        return None
    return cached


def _save_entry(
    pickle_path: Path, sidecar_path: Path, cached: object, provenance: dict
) -> None:
    """Write the pickle, then its JSON sidecar, each atomically."""
    _atomic_write(pickle_path, pickle.dumps(cached, protocol=pickle.HIGHEST_PROTOCOL))
    _atomic_write(
        sidecar_path, json.dumps(provenance, indent=2, sort_keys=True).encode()
    )


def _cache_section(kind: str, key: str, request: dict, pickle_path: Path) -> dict:
    return {
        "kind": kind,
        "key": key,
        "format_version": CACHE_FORMAT_VERSION,
        "file": pickle_path.name,
        "request": request,
    }


def fit_or_load_proportion_model(
    cache_dir: str | Path,
    conditions_struct: list,
    column_x: str,
    column_y: str,
    conditions: list[int],
    percentages: np.ndarray | None = None,
    refit: bool = False,
    remove_hatch: bool = True,
    lmbda: float = 0.0075,
    medfilt_window: int = 5,
    bspline_order: int = 3,
    **fit_kwargs,
) -> ProportionModelResult:
    """
    Fit the joint proportion model, or load the cached fit made on the same data with the same settings.

    The table is built with ``build_proportion_table``. The cache key hashes the
    table's data (``table_data_hash``), the canonical percentages (sorted and
    unique after rounding to 6 decimals, so the same set in another order or
    with repeats hits the same entry), the column names, conditions,
    ``remove_hatch`` and, with percentages, the smoothing settings, every argument of ``fit_proportion_model`` with defaults
    filled in, except ``progressbar`` and ``cores``, which do not change the
    draws, and a cache format version. Each entry is a pickle and a JSON
    sidecar holding the result's provenance, both written atomically. A cached
    result is returned only if its stored provenance records exactly this
    request; an unreadable or mismatching entry is refitted and overwritten
    with a warning. Files are never deleted.

    Parameters:
        cache_dir (str or Path): Directory of the cache; created if missing.
        conditions_struct (list): Conditions, as for ``build_proportion_table``.
        column_x (str): Key of the X measurement.
        column_y (str): Key of the Y measurement.
        conditions (list[int]): Condition ids to include.
        percentages (np.ndarray or None): Development percentages at which to
            sample the raw series; ``None`` uses values at ecdysis.
            (default: None)
        refit (bool): Fit and overwrite the entry even if a matching one
            exists. (default: False)
        remove_hatch (bool): Passed to ``build_proportion_table``; ignored
            with ``percentages``. (default: True)
        lmbda (float): Passed to ``build_proportion_table``. (default: 0.0075)
        medfilt_window (int): Passed to ``build_proportion_table``. (default: 5)
        bspline_order (int): Passed to ``build_proportion_table``. (default: 3)
        **fit_kwargs: Keyword arguments of ``fit_proportion_model``, including
            ``random_seed`` and the genotype coding.

    Returns:
        ProportionModelResult: The fit, whose ``provenance`` has a ``"cache"``
            entry with the key, the request and the pickle's file name.

    Raises:
        TypeError: If ``fit_kwargs`` holds an argument ``fit_proportion_model``
            does not take.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    table = build_proportion_table(
        conditions_struct,
        column_x,
        column_y,
        conditions,
        remove_hatch=remove_hatch,
        percentages=percentages,
        lmbda=lmbda,
        medfilt_window=medfilt_window,
        bspline_order=bspline_order,
    )
    request = _request(table, "joint", _fit_arguments(fit_kwargs))
    key = _cache_key(request)
    pickle_path, sidecar_path = _entry_paths(cache_dir, "joint", key)

    if not refit and pickle_path.exists():
        cached = _load_entry(pickle_path, key, request, ProportionModelResult)
        if cached is not None:
            return cached

    result = fit_proportion_model(table, **fit_kwargs)
    result.provenance = {
        **(result.provenance or {}),
        "cache": _cache_section("joint", key, request, pickle_path),
    }
    _save_entry(pickle_path, sidecar_path, result, result.provenance)
    return result


def fit_or_load_per_experiment(
    cache_dir: str | Path,
    conditions_struct: list,
    column_x: str,
    column_y: str,
    conditions: list[int],
    reference_shape: str,
    percentages: np.ndarray | None = None,
    refit: bool = False,
    remove_hatch: bool = True,
    lmbda: float = 0.0075,
    medfilt_window: int = 5,
    bspline_order: int = 3,
    **fit_kwargs,
) -> PerExperimentFits:
    """
    Fit the proportion model per experiment, or load the cached fits made on the same data with the same settings.

    Keyed, stored and verified like ``fit_or_load_proportion_model``; the
    cached object is the dict of ``fit_per_experiment`` with a ``provenance``
    attribute.

    Parameters:
        cache_dir (str or Path): Directory of the cache; created if missing.
        conditions_struct (list): Conditions, as for ``build_proportion_table``.
        column_x (str): Key of the X measurement.
        column_y (str): Key of the Y measurement.
        conditions (list[int]): Condition ids to include.
        reference_shape (str): ``"linear"`` or ``"spline"``.
        percentages (np.ndarray or None): Development percentages at which to
            sample the raw series; ``None`` uses values at ecdysis.
            (default: None)
        refit (bool): Fit and overwrite the entry even if a matching one
            exists. (default: False)
        remove_hatch (bool): Passed to ``build_proportion_table``; ignored
            with ``percentages``. (default: True)
        lmbda (float): Passed to ``build_proportion_table``. (default: 0.0075)
        medfilt_window (int): Passed to ``build_proportion_table``. (default: 5)
        bspline_order (int): Passed to ``build_proportion_table``. (default: 3)
        **fit_kwargs: Keyword arguments of ``fit_proportion_model``, including
            ``random_seed`` and the genotype coding.

    Returns:
        PerExperimentFits: Fit of each experiment keyed by its label, with
            ``provenance``.

    Raises:
        TypeError: If ``fit_kwargs`` holds an argument ``fit_proportion_model``
            does not take.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    table = build_proportion_table(
        conditions_struct,
        column_x,
        column_y,
        conditions,
        remove_hatch=remove_hatch,
        percentages=percentages,
        lmbda=lmbda,
        medfilt_window=medfilt_window,
        bspline_order=bspline_order,
    )
    fit_arguments = _fit_arguments(fit_kwargs, reference_shape)
    request = _request(table, "per_experiment", fit_arguments)
    key = _cache_key(request)
    pickle_path, sidecar_path = _entry_paths(cache_dir, "per_experiment", key)

    if not refit and pickle_path.exists():
        cached = _load_entry(pickle_path, key, request, PerExperimentFits)
        if cached is not None:
            return cached

    fits = fit_per_experiment(table, reference_shape, **fit_kwargs)
    provenance = {
        **fit_provenance(table, fit_arguments),
        "cache": _cache_section("per_experiment", key, request, pickle_path),
    }
    cached = PerExperimentFits(fits, provenance)
    _save_entry(pickle_path, sidecar_path, cached, provenance)
    return cached


def list_cached_models(cache_dir: str | Path) -> pd.DataFrame:
    """
    List the entries of a proportion model cache from their JSON sidecars.

    Sidecars that cannot be read are skipped with a warning.

    Parameters:
        cache_dir (str or Path): Directory of the cache.

    Returns:
        pd.DataFrame: One row per entry, oldest first, with ``file`` (pickle
            file name), ``kind`` (``"joint"`` or ``"per_experiment"``),
            ``column_x``, ``column_y``, ``sampling``, ``percentages``,
            ``conditions``, ``likelihood``, ``reference_shape`` (requested),
            ``reference_shape_used`` (``None`` for per-experiment entries) and
            ``created``.
    """
    rows = []
    for sidecar in sorted(Path(cache_dir).glob("*.json")):
        try:
            provenance = json.loads(sidecar.read_text())
            entry = provenance["cache"]
            request = entry["request"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            warn(f"Skipping unreadable cache sidecar {sidecar.name} ({error!r}).")
            continue
        fit_arguments = request["fit_arguments"]
        rows.append(
            {
                "file": entry["file"],
                "kind": entry["kind"],
                "column_x": request["column_x"],
                "column_y": request["column_y"],
                "sampling": request["sampling"],
                "percentages": request["percentages"],
                "conditions": request["conditions"],
                "likelihood": fit_arguments.get("likelihood"),
                "reference_shape": fit_arguments.get("reference_shape"),
                "reference_shape_used": provenance.get("reference_shape_used"),
                "created": provenance.get("created"),
            }
        )
    listing = pd.DataFrame(rows, columns=LISTED_COLUMNS)
    return listing.sort_values("created", kind="stable").reset_index(drop=True)
