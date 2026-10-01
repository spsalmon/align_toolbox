from dataclasses import dataclass
from itertools import combinations, product
from typing import Literal
from warnings import warn

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from pytensor import tensor as pt
from xarray import DataTree

from align_toolbox.plotting.proportions import log_ratio_to_percentage

__all__ = [
    "GenotypeCoding",
    "ProportionModelResult",
    "build_genotype_coding",
    "build_proportion_table",
    "fit_proportion_model",
]

LIKELIHOODS = ("normal", "student_t")
SUPPORT_LABELS = ("in", "gap", "out")
SINGLE_FACTOR_NAME = "condition"

MAX_RHAT = 1.01
MIN_ESS = 400

# priors, in natural-log units
INTERCEPT_PRIOR_SD = 1.0
SLOPE_PRIOR_MEAN = 1.0
SLOPE_PRIOR_SD = 1.0
EFFECT_PRIOR_SD = 0.5
EXPERIMENT_PRIOR_SD = 0.5
SCALE_PRIOR_SD = 0.2
NU_PRIOR_ALPHA = 2.0
NU_PRIOR_BETA = 0.1


# DATA PREPARATION


def build_proportion_table(
    conditions_struct: list,
    column_x: str,
    column_y: str,
    conditions: list[int],
    remove_hatch: bool = True,
) -> pd.DataFrame:
    """
    Flatten per-molt measurements of two columns into one row per worm × molt in log space.

    Rows where either value is NaN or non-positive are dropped. Each experiment path
    is mapped to a short label (``"E1"``, ``"E2"``, ... in sorted path order) and
    worms are identified by condition, experiment label and point number, since
    point numbers repeat between experiments. The attrition per condition × molt is
    stored in ``table.attrs["counts"]`` and the label mapping in
    ``table.attrs["experiment_labels"]``.

    Parameters:
        conditions_struct (list): List of condition dicts, each holding
            ``"condition_id"``, ``"point"`` and ``"experiment"`` arrays of shape
            ``(n_worms, 1)`` and the two measurement columns.
        column_x (str): Key of the X measurement, of shape ``(n_worms, n_events)``.
        column_y (str): Key of the Y measurement, of shape ``(n_worms, n_events)``.
        conditions (list[int]): Condition ids to include.
        remove_hatch (bool): If ``True``, drop event index 0 (hatch). (default: True)

    Returns:
        pd.DataFrame: Columns ``condition_id``, ``worm_id``, ``experiment``,
            ``molt`` (original event index), ``log_x`` and ``log_y``. ``attrs``
            holds ``"counts"`` (DataFrame with ``condition_id``, ``molt``,
            ``n_worms``, ``n_nan``, ``n_non_positive`` and ``n_kept``) and
            ``"experiment_labels"`` (dict mapping label to experiment path).

    Raises:
        ValueError: If a requested condition is missing from ``conditions_struct``,
            a condition lacks the ``"experiment"`` key, or the two columns differ
            in shape.
    """
    conditions_by_id = {int(c["condition_id"]): c for c in conditions_struct}
    missing = [c for c in conditions if c not in conditions_by_id]
    if missing:
        raise ValueError(f"Conditions {missing} are not in conditions_struct.")
    for condition_id in conditions:
        if "experiment" not in conditions_by_id[condition_id]:
            raise ValueError(
                f"Condition {condition_id} has no 'experiment' key; it is needed to "
                "tell apart worms with the same point number."
            )

    experiment_paths = sorted(
        {
            str(path)
            for condition_id in conditions
            for path in np.ravel(conditions_by_id[condition_id]["experiment"])
        }
    )
    label_of_path = {path: f"E{i + 1}" for i, path in enumerate(experiment_paths)}

    first_event = 1 if remove_hatch else 0
    blocks = []
    counts = []
    for condition_id in conditions:
        condition = conditions_by_id[condition_id]
        x = np.asarray(condition[column_x], dtype=float)
        y = np.asarray(condition[column_y], dtype=float)
        if x.shape != y.shape:
            raise ValueError(
                f"Condition {condition_id}: {column_x} has shape {x.shape} but "
                f"{column_y} has shape {y.shape}."
            )
        experiment_labels = np.array(
            [label_of_path[str(p)] for p in np.ravel(condition["experiment"])]
        )
        worm_ids = np.array(
            [
                f"c{condition_id}_{label}_p{point}"
                for label, point in zip(experiment_labels, np.ravel(condition["point"]))
            ]
        )

        for molt in range(first_event, x.shape[1]):
            x_molt, y_molt = x[:, molt], y[:, molt]
            is_nan = np.isnan(x_molt) | np.isnan(y_molt)
            is_non_positive = ~is_nan & ((x_molt <= 0) | (y_molt <= 0))
            keep = ~is_nan & ~is_non_positive
            counts.append(
                {
                    "condition_id": condition_id,
                    "molt": molt,
                    "n_worms": len(x_molt),
                    "n_nan": int(is_nan.sum()),
                    "n_non_positive": int(is_non_positive.sum()),
                    "n_kept": int(keep.sum()),
                }
            )
            blocks.append(
                pd.DataFrame(
                    {
                        "condition_id": condition_id,
                        "worm_id": worm_ids[keep],
                        "experiment": experiment_labels[keep],
                        "molt": molt,
                        "log_x": np.log(x_molt[keep]),
                        "log_y": np.log(y_molt[keep]),
                    }
                )
            )

    table = pd.concat(blocks, ignore_index=True)
    table.attrs["counts"] = pd.DataFrame(counts)
    table.attrs["experiment_labels"] = {
        label: path for path, label in label_of_path.items()
    }
    return table


# GENOTYPE CODING


@dataclass
class GenotypeCoding:
    """
    Treatment coding of genotype factors with all interactions.

    Attributes:
        factor_names (list[str]): Factor names, in the order of ``reference``.
        reference (dict[str, str]): Reference level of each factor.
        levels (dict[str, list[str]]): Levels of each factor, reference first.
        condition_levels (dict[int, dict[str, str]]): Factor levels of each
            coded condition.
        effect_names (list[str]): Names of the design columns, e.g.
            ``"yap1[abt7]"`` or ``"yap1[abt7]:tir[col-10]"``.
        cells (list[str]): Labels of the observed genotype cells, reference first.
    """

    factor_names: list[str]
    reference: dict[str, str]
    levels: dict[str, list[str]]
    condition_levels: dict[int, dict[str, str]]
    effect_names: list[str]
    cells: list[str]

    @property
    def reference_cell(self) -> str:
        """Label of the cell where every factor is at its reference level."""
        return self.cells[0]

    def cell_label(self, levels: dict[str, str]) -> str:
        """
        Label a genotype cell from its factor levels.

        Parameters:
            levels (dict[str, str]): Level of each factor.

        Returns:
            str: Label such as ``"yap1=abt7, tir=col-10"``.
        """
        return ", ".join(f"{name}={levels[name]}" for name in self.factor_names)

    def cell_of_condition(self, condition_id: int) -> str:
        """
        Return the genotype cell label of a coded condition.

        Parameters:
            condition_id (int): Condition id.

        Returns:
            str: Cell label.
        """
        return self.cell_label(self.condition_levels[condition_id])

    def effect_vector(self, levels: dict[str, str]) -> np.ndarray:
        """
        Return the design row of a genotype cell.

        Parameters:
            levels (dict[str, str]): Level of each factor.

        Returns:
            np.ndarray: 0/1 vector of shape ``(n_effects,)``.
        """
        active = {f"{name}[{levels[name]}]" for name in self.factor_names}
        return np.array(
            [
                float(all(term in active for term in e.split(":")))
                for e in self.effect_names
            ]
        )

    def design_matrix(self, condition_ids: list[int]) -> pd.DataFrame:
        """
        Return the design rows of the given conditions.

        Parameters:
            condition_ids (list[int]): Coded condition ids.

        Returns:
            pd.DataFrame: One row per condition (indexed by ``condition_id``) and
                one 0/1 column per effect.
        """
        return pd.DataFrame(
            [self.effect_vector(self.condition_levels[c]) for c in condition_ids],
            index=pd.Index(condition_ids, name="condition_id"),
            columns=self.effect_names,
        )

    def resolve_cell(self, cell: int | str | dict[str, str]) -> dict[str, str]:
        """
        Return the factor levels of a fitted genotype cell.

        Parameters:
            cell (int, str or dict[str, str]): A condition id, a cell label, or the
                factor levels of the cell.

        Returns:
            dict[str, str]: Level of each factor.

        Raises:
            ValueError: If ``cell`` does not designate a fitted cell.
        """
        if isinstance(cell, dict):
            levels = {name: str(cell.get(name)) for name in self.factor_names}
        elif isinstance(cell, str):
            levels = next(
                (
                    lv
                    for lv in self.condition_levels.values()
                    if self.cell_label(lv) == cell
                ),
                None,
            )
        else:
            levels = self.condition_levels.get(int(cell))
        if levels is None or self.cell_label(levels) not in self.cells:
            raise ValueError(f"{cell!r} is not one of the fitted cells {self.cells}.")
        return levels


def build_genotype_coding(
    condition_ids: list[int],
    factors: dict[int, dict[str, str]] | None = None,
    reference: dict[str, str] | None = None,
    reference_condition: int | None = None,
) -> GenotypeCoding:
    """
    Build treatment-coded main effects and all their interactions for the given conditions.

    Without ``factors``, the condition id is used as a single factor named
    ``"condition"`` with ``reference_condition`` as its baseline. Interaction
    columns whose cell combination does not occur among ``condition_ids`` are
    dropped, since no data inform them.

    Parameters:
        condition_ids (list[int]): Conditions that will be modelled.
        factors (dict[int, dict[str, str]] or None): Factor levels of each
            condition, e.g. ``{4: {"yap1": "abt7", "tir": "none"}}``.
            (default: None)
        reference (dict[str, str] or None): Reference level of each factor;
            required with ``factors``. (default: None)
        reference_condition (int or None): Baseline condition when ``factors`` is
            ``None``. (default: None)

    Returns:
        GenotypeCoding: The coding.

    Raises:
        ValueError: If the reference is missing, a condition is not coded or
            lacks a factor, or no condition sits in the reference cell.
    """
    condition_ids = [int(c) for c in condition_ids]
    if factors is None:
        if reference_condition is None:
            raise ValueError(
                "Pass either `factors` and `reference`, or `reference_condition`."
            )
        if reference_condition not in condition_ids:
            raise ValueError(
                f"Reference condition {reference_condition} is not among the "
                f"modelled conditions {condition_ids}."
            )
        factors = {c: {SINGLE_FACTOR_NAME: str(c)} for c in condition_ids}
        reference = {SINGLE_FACTOR_NAME: str(reference_condition)}
    elif reference is None:
        raise ValueError("`reference` must name the reference level of each factor.")

    factor_names = list(reference)
    uncoded = [c for c in condition_ids if c not in factors]
    if uncoded:
        raise ValueError(f"Conditions {uncoded} have no entry in `factors`.")
    condition_levels = {}
    for c in condition_ids:
        if set(factors[c]) != set(factor_names):
            raise ValueError(
                f"Condition {c} is coded with factors {sorted(factors[c])}, "
                f"expected {sorted(factor_names)}."
            )
        condition_levels[c] = {name: str(factors[c][name]) for name in factor_names}
    reference = {name: str(level) for name, level in reference.items()}

    if not any(lv == reference for lv in condition_levels.values()):
        raise ValueError(
            f"No modelled condition is in the reference cell {reference}; "
            f"conditions are coded as {condition_levels}."
        )

    levels = {
        name: [reference[name]]
        + sorted({lv[name] for lv in condition_levels.values()} - {reference[name]})
        for name in factor_names
    }

    candidate_effects = [
        ":".join(f"{name}[{level}]" for name, level in zip(combo, combo_levels))
        for order in range(1, len(factor_names) + 1)
        for combo in combinations(factor_names, order)
        for combo_levels in product(*(levels[name][1:] for name in combo))
    ]
    cells = []
    for c in sorted(condition_levels, key=lambda c: condition_levels[c] != reference):
        label = ", ".join(f"{n}={condition_levels[c][n]}" for n in factor_names)
        if label not in cells:
            cells.append(label)

    coding = GenotypeCoding(
        factor_names=factor_names,
        reference=reference,
        levels=levels,
        condition_levels=condition_levels,
        effect_names=candidate_effects,
        cells=cells,
    )
    design = coding.design_matrix(condition_ids)
    coding.effect_names = [e for e in candidate_effects if design[e].any()]

    cell_design = coding.design_matrix(condition_ids).drop_duplicates().to_numpy()
    cell_design = np.column_stack([np.ones(len(cell_design)), cell_design])
    if np.linalg.matrix_rank(cell_design) < cell_design.shape[1]:
        warn(
            "The observed genotype cells do not identify every effect; some "
            "effects are informed by their prior only.",
            stacklevel=2,
        )
    return coding


# MODEL


@dataclass
class ProportionModelResult:
    """
    Posterior of the joint proportion model and the data it was fitted on.

    Attributes:
        idata (DataTree): ArviZ inference data returned by ``pymc.sample``.
        table (pd.DataFrame): Data table from ``build_proportion_table``.
        coding (GenotypeCoding): Genotype coding of the conditions.
        x_ref (float): Centering point of ``log_x``; genotype intercept effects
            are offsets at this body size.
        experiment_labels (dict[str, str]): Experiment label to path mapping.
        counts (pd.DataFrame or None): Attrition per condition × molt.
        diagnostics (dict[str, float]): ``max_rhat``, ``min_ess_bulk``,
            ``min_ess_tail`` and ``n_divergences``.
        likelihood (str): ``"normal"`` or ``"student_t"``.
        group_variances (bool): Whether ``tau`` and ``sigma`` vary by cell.
        random_seed (int): Seed used for sampling.
    """

    idata: DataTree
    table: pd.DataFrame
    coding: GenotypeCoding
    x_ref: float
    experiment_labels: dict[str, str]
    counts: pd.DataFrame | None
    diagnostics: dict[str, float]
    likelihood: str
    group_variances: bool
    random_seed: int

    def _draws(self, name: str) -> np.ndarray:
        """Posterior draws of ``name`` with chains and draws flattened to axis 0."""
        values = self.idata["posterior"][name].values
        return values.reshape(-1, *values.shape[2:])

    def _has(self, name: str) -> bool:
        return name in self.idata["posterior"].data_vars

    def _scale_draws(self, name: str) -> np.ndarray:
        """Draws of ``tau`` or ``sigma`` of shape ``(n_draws, n_cells)``."""
        draws = self._draws(name)
        if draws.ndim == 1:
            draws = np.repeat(draws[:, np.newaxis], len(self.coding.cells), axis=1)
        return draws

    def summary(self) -> pd.DataFrame:
        """
        Summarize the posterior with means and 95% equal-tailed intervals.

        Rows cover ``a``, ``b``, each genotype intercept effect ``beta`` (also
        back-transformed to percent with ``log_ratio_to_percentage``), each slope
        effect ``gamma``, the experiment effects, ``nu`` for a Student-t
        likelihood, ``tau`` and ``sigma``, and, with group variances, the ratios
        of each cell's ``tau`` and ``sigma`` to the reference cell's.

        Returns:
            pd.DataFrame: Columns ``parameter``, ``mean``, ``lower``, ``upper``,
                ``mean_percent``, ``lower_percent`` and ``upper_percent``; the
                percent columns are NaN except for ``beta`` rows.
        """
        rows = []

        def add(parameter, draws, percent=False):
            row = {
                "parameter": parameter,
                "mean": draws.mean(),
                "lower": np.quantile(draws, 0.025),
                "upper": np.quantile(draws, 0.975),
            }
            for key in ("mean", "lower", "upper"):
                row[f"{key}_percent"] = (
                    log_ratio_to_percentage(row[key]) if percent else np.nan
                )
            rows.append(row)

        add("a", self._draws("a"))
        add("b", self._draws("b"))
        if self.coding.effect_names:
            beta, gamma = self._draws("beta"), self._draws("gamma")
            for k, effect in enumerate(self.coding.effect_names):
                add(f"beta[{effect}]", beta[:, k], percent=True)
            for k, effect in enumerate(self.coding.effect_names):
                add(f"gamma[{effect}]", gamma[:, k])
        if self._has("experiment_effect"):
            effects = self._draws("experiment_effect")
            experiments = self.idata["posterior"]["experiment_effect"].coords[
                "experiment"
            ]
            for k, experiment in enumerate(experiments.values):
                add(f"experiment_effect[{experiment}]", effects[:, k])
        if self._has("nu"):
            add("nu", self._draws("nu"))
        for name in ("tau", "sigma"):
            if self.group_variances:
                draws = self._draws(name)
                for k, cell in enumerate(self.coding.cells):
                    add(f"{name}[{cell}]", draws[:, k])
                for k, cell in enumerate(self.coding.cells[1:], start=1):
                    add(f"{name}_ratio[{cell}]", draws[:, k] / draws[:, 0])
            else:
                add(name, self._draws(name))
        return pd.DataFrame(rows)

    def group_offset_curve(
        self, cell: int | str | dict[str, str], log_x_grid: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute the size-matched log-ratio offset of a genotype cell relative to the reference cell.

        At each ``log_x``, the offset is the sum over the cell's active effects
        (main effects and interactions) of ``beta_k + gamma_k * (log_x - x_ref)``.

        Parameters:
            cell (int, str or dict[str, str]): A condition id, a cell label, or
                the factor levels of the cell.
            log_x_grid (np.ndarray): Natural-log X values of shape ``(n_points,)``.

        Returns:
            pd.DataFrame: Columns ``log_x``, ``mean``, ``lower`` and ``upper``
                (95% equal-tailed band), in natural-log ratio units.

        Raises:
            ValueError: If ``cell`` does not designate a fitted cell.
        """
        log_x_grid = np.asarray(log_x_grid, dtype=float)
        d = self.coding.effect_vector(self.coding.resolve_cell(cell))
        n_draws = len(self._draws("a"))
        if self.coding.effect_names:
            intercept = self._draws("beta") @ d
            slope = self._draws("gamma") @ d
        else:
            intercept = slope = np.zeros(n_draws)
        offsets = (
            intercept[:, np.newaxis]
            + slope[:, np.newaxis] * (log_x_grid - self.x_ref)[np.newaxis, :]
        )
        return pd.DataFrame(
            {
                "log_x": log_x_grid,
                "mean": offsets.mean(axis=0),
                "lower": np.quantile(offsets, 0.025, axis=0),
                "upper": np.quantile(offsets, 0.975, axis=0),
            }
        )

    def _reference_mask(self) -> np.ndarray:
        cells = self.table["condition_id"].map(self.coding.cell_of_condition)
        return (cells == self.coding.reference_cell).to_numpy()

    def support_flags(self) -> pd.Series:
        """
        Label each observation by where its ``log_x`` falls relative to the reference cell's data.

        The reference support is the union of the reference cell's per-molt
        min–max ``log_x`` ranges. Values inside a range are ``"in"``, values
        between the lowest and highest range but outside all of them are
        ``"gap"``, and values below or above every range are ``"out"``.

        Returns:
            pd.Series: Labels aligned with ``table.index``, named ``"support"``.
        """
        reference = self.table[self._reference_mask()]
        ranges = reference.groupby("molt")["log_x"].agg(["min", "max"]).to_numpy()
        log_x = self.table["log_x"].to_numpy()[:, np.newaxis]
        inside = ((log_x >= ranges[:, 0]) & (log_x <= ranges[:, 1])).any(axis=1)
        log_x = log_x[:, 0]
        outside = (log_x < ranges[:, 0].min()) | (log_x > ranges[:, 1].max())
        labels = np.where(inside, "in", np.where(outside, "out", "gap"))
        return pd.Series(labels, index=self.table.index, name="support")

    def _index_of(self, variable: str, dim: str, values: pd.Series) -> np.ndarray:
        coords = pd.Index(self.idata["posterior"][variable].coords[dim].values)
        return coords.get_indexer(values)

    def reference_residuals_by_molt(self) -> pd.DataFrame:
        """
        Compute the posterior mean residual of the reference cell at each molt, in percent.

        Residuals are ``log_y`` minus the fitted value including experiment and
        worm effects, averaged over the reference observations of each molt, then
        back-transformed with ``log_ratio_to_percentage``. Values near 0 at every
        molt mean the line describes the reference cell.

        Returns:
            pd.DataFrame: Columns ``molt``, ``n_observations``, ``mean_percent``,
                ``lower_percent`` and ``upper_percent`` (95% equal-tailed interval).
        """
        reference = self.table[self._reference_mask()]
        x_centered = reference["log_x"].to_numpy() - self.x_ref
        fitted = (
            self._draws("a")[:, np.newaxis]
            + self._draws("b")[:, np.newaxis] * x_centered[np.newaxis, :]
        )
        if self._has("experiment_effect"):
            experiment_index = self._index_of(
                "experiment_effect", "experiment", reference["experiment"]
            )
            fitted += self._draws("experiment_effect")[:, experiment_index]
        worm_index = self._index_of("worm_effect", "worm", reference["worm_id"])
        fitted += self._draws("worm_effect")[:, worm_index]
        residuals = reference["log_y"].to_numpy()[np.newaxis, :] - fitted

        rows = []
        for molt in sorted(reference["molt"].unique()):
            in_molt = (reference["molt"] == molt).to_numpy()
            mean_residual = residuals[:, in_molt].mean(axis=1)
            rows.append(
                {
                    "molt": molt,
                    "n_observations": int(in_molt.sum()),
                    "mean_percent": log_ratio_to_percentage(mean_residual.mean()),
                    "lower_percent": log_ratio_to_percentage(
                        np.quantile(mean_residual, 0.025)
                    ),
                    "upper_percent": log_ratio_to_percentage(
                        np.quantile(mean_residual, 0.975)
                    ),
                }
            )
        return pd.DataFrame(rows)

    def worm_effects(self) -> pd.DataFrame:
        """
        Summarize the worm intercepts ``u_i`` with posterior means and 95% equal-tailed intervals.

        Returns:
            pd.DataFrame: Columns ``worm_id``, ``condition_id``, ``experiment``,
                ``mean``, ``lower`` and ``upper``, in natural-log units.
        """
        worms = self.table.drop_duplicates("worm_id")[
            ["worm_id", "condition_id", "experiment"]
        ].reset_index(drop=True)
        draws = self._draws("worm_effect")[
            :, self._index_of("worm_effect", "worm", worms["worm_id"])
        ]
        return worms.assign(
            mean=draws.mean(axis=0),
            lower=np.quantile(draws, 0.025, axis=0),
            upper=np.quantile(draws, 0.975, axis=0),
        )

    def reference_predictive_interval(
        self,
        log_x_grid: np.ndarray,
        prob: float = 0.95,
        random_seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Compute the posterior predictive interval of ``log_y`` for a new reference-cell worm.

        Each posterior draw adds a new worm intercept ``Normal(0, tau)`` and a
        residual ``Normal(0, sigma)`` (or ``StudentT(nu, 0, sigma)``) to the line
        of the average experiment, using the reference cell's ``tau`` and
        ``sigma``.

        Parameters:
            log_x_grid (np.ndarray): Natural-log X values of shape ``(n_points,)``.
            prob (float): Probability mass of the equal-tailed interval.
                (default: 0.95)
            random_seed (int or None): Seed for the predictive draws; ``None``
                reuses the sampling seed. (default: None)

        Returns:
            pd.DataFrame: Columns ``log_x``, ``mean`` (posterior mean of the
                line), ``lower`` and ``upper``.

        Raises:
            ValueError: If ``prob`` is not strictly between 0 and 1.
        """
        if not 0 < prob < 1:
            raise ValueError(f"`prob` must be between 0 and 1, got {prob}.")
        rng = np.random.default_rng(
            self.random_seed if random_seed is None else random_seed
        )
        log_x_grid = np.asarray(log_x_grid, dtype=float)
        line = (
            self._draws("a")[:, np.newaxis]
            + self._draws("b")[:, np.newaxis] * (log_x_grid - self.x_ref)[np.newaxis, :]
        )
        n_draws, n_points = line.shape
        tau = self._scale_draws("tau")[:, 0, np.newaxis]
        sigma = self._scale_draws("sigma")[:, 0, np.newaxis]
        worm = tau * rng.standard_normal((n_draws, 1))
        if self._has("nu"):
            noise = rng.standard_t(
                self._draws("nu")[:, np.newaxis], (n_draws, n_points)
            )
        else:
            noise = rng.standard_normal((n_draws, n_points))
        predicted = line + worm + sigma * noise
        tail = (1 - prob) / 2
        return pd.DataFrame(
            {
                "log_x": log_x_grid,
                "mean": line.mean(axis=0),
                "lower": np.quantile(predicted, tail, axis=0),
                "upper": np.quantile(predicted, 1 - tail, axis=0),
            }
        )


def _compute_diagnostics(idata: DataTree, var_names: list[str]) -> dict[str, float]:
    """
    Compute convergence diagnostics over the free parameters.

    Parameters:
        idata (DataTree): Inference data with ``posterior`` and ``sample_stats``.
        var_names (list[str]): Free parameters to check.

    Returns:
        dict[str, float]: ``max_rhat``, ``min_ess_bulk``, ``min_ess_tail`` and
            ``n_divergences``.
    """

    def extreme(statistic, reduce):
        values = [
            reduce(np.asarray(statistic[name].values, dtype=float))
            for name in var_names
        ]
        return float(reduce(np.array(values)))

    return {
        "max_rhat": extreme(az.rhat(idata, var_names=var_names), np.nanmax),
        "min_ess_bulk": extreme(
            az.ess(idata, var_names=var_names, method="bulk"), np.nanmin
        ),
        "min_ess_tail": extreme(
            az.ess(idata, var_names=var_names, method="tail"), np.nanmin
        ),
        "n_divergences": int(idata["sample_stats"]["diverging"].values.sum()),
    }


def fit_proportion_model(
    table: pd.DataFrame,
    factors: dict[int, dict[str, str]] | None = None,
    reference: dict[str, str] | None = None,
    reference_condition: int | None = None,
    likelihood: Literal["normal", "student_t"] = "normal",
    group_variances: bool = False,
    x_ref: float | None = None,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    *,
    random_seed: int,
    progressbar: bool = False,
) -> ProportionModelResult:
    """
    Fit a hierarchical log-log proportion model jointly to all genotypes with NUTS.

    The model is ``log_y = a + b * xc + sum_k D_k * (beta_k + gamma_k * xc) + e_exp
    + u_worm + eps`` with ``xc = log_x - x_ref``, treatment-coded genotype effects
    ``D_k`` (main effects and interactions), sum-to-zero experiment effects,
    non-centered worm intercepts ``u ~ Normal(0, tau)`` and residuals
    ``eps ~ Normal(0, sigma)`` or ``StudentT(nu, 0, sigma)``. ``a`` and ``beta``
    therefore describe the average experiment at body size ``x_ref``.

    Parameters:
        table (pd.DataFrame): Output of ``build_proportion_table``.
        factors (dict[int, dict[str, str]] or None): Factor levels of each
            condition; see ``build_genotype_coding``. (default: None)
        reference (dict[str, str] or None): Reference level of each factor.
            (default: None)
        reference_condition (int or None): Baseline condition when ``factors`` is
            ``None``. (default: None)
        likelihood (str): ``"normal"`` or ``"student_t"``; the latter estimates
            ``nu`` with a ``Gamma(2, 0.1)`` prior. (default: "normal")
        group_variances (bool): If ``True``, ``tau`` and ``sigma`` are estimated
            per genotype cell; otherwise they are shared. (default: False)
        x_ref (float or None): Centering point of ``log_x``; ``None`` uses the
            median ``log_x`` of the reference cell. (default: None)
        draws (int): Posterior draws per chain. (default: 1000)
        tune (int): Tuning steps per chain. (default: 1000)
        chains (int): Number of chains. (default: 4)
        target_accept (float): NUTS target acceptance rate. (default: 0.9)
        random_seed (int): Seed for sampling; required for reproducibility.
        progressbar (bool): Show the sampling progress bar. (default: False)

    Returns:
        ProportionModelResult: Posterior, data, coding and diagnostics. A warning
            is emitted if R-hat exceeds 1.01, an ESS is below 400, or any
            transition diverged.

    Raises:
        ValueError: If ``likelihood`` is unknown, ``random_seed`` is not an int,
            the table is empty, or the genotype coding is invalid.
    """
    if likelihood not in LIKELIHOODS:
        raise ValueError(
            f"Invalid likelihood {likelihood!r}; expected one of {LIKELIHOODS}."
        )
    if not isinstance(random_seed, (int, np.integer)) or isinstance(random_seed, bool):
        raise ValueError(f"`random_seed` must be an int, got {random_seed!r}.")
    if len(table) == 0:
        raise ValueError("The table has no observations.")

    condition_ids = sorted(int(c) for c in table["condition_id"].unique())
    coding = build_genotype_coding(
        condition_ids, factors, reference, reference_condition
    )

    cell_of_row = table["condition_id"].map(coding.cell_of_condition)
    cell_index = pd.Index(coding.cells)
    obs_cell = cell_index.get_indexer(cell_of_row)
    is_reference = obs_cell == 0

    log_x = table["log_x"].to_numpy(dtype=float)
    log_y = table["log_y"].to_numpy(dtype=float)
    if x_ref is None:
        x_ref = float(np.median(log_x[is_reference]))
    x_centered = log_x - x_ref
    design = coding.design_matrix(condition_ids).loc[table["condition_id"]].to_numpy()

    experiments = sorted(table["experiment"].unique())
    experiment_index = pd.Index(experiments).get_indexer(table["experiment"])
    worms = table.drop_duplicates("worm_id")
    worm_index = pd.Index(worms["worm_id"]).get_indexer(table["worm_id"])
    worm_cell = cell_index.get_indexer(
        worms["condition_id"].map(coding.cell_of_condition)
    )

    coords = {
        "effect": coding.effect_names,
        "experiment": experiments,
        "worm": worms["worm_id"].to_numpy(),
        "cell": coding.cells,
        "obs": np.arange(len(table)),
    }
    with pm.Model(coords=coords) as model:
        a = pm.Normal("a", mu=log_y[is_reference].mean(), sigma=INTERCEPT_PRIOR_SD)
        b = pm.Normal("b", mu=SLOPE_PRIOR_MEAN, sigma=SLOPE_PRIOR_SD)
        mu = a + b * x_centered
        if coding.effect_names:
            beta = pm.Normal("beta", mu=0, sigma=EFFECT_PRIOR_SD, dims="effect")
            gamma = pm.Normal("gamma", mu=0, sigma=EFFECT_PRIOR_SD, dims="effect")
            mu = mu + pt.dot(design, beta) + pt.dot(design * x_centered[:, None], gamma)
        if len(experiments) > 1:
            experiment_effect = pm.ZeroSumNormal(
                "experiment_effect", sigma=EXPERIMENT_PRIOR_SD, dims="experiment"
            )
            mu = mu + experiment_effect[experiment_index]

        if group_variances:
            tau = pm.HalfNormal("tau", sigma=SCALE_PRIOR_SD, dims="cell")
            sigma = pm.HalfNormal("sigma", sigma=SCALE_PRIOR_SD, dims="cell")
            worm_tau, obs_sigma = tau[worm_cell], sigma[obs_cell]
        else:
            tau = pm.HalfNormal("tau", sigma=SCALE_PRIOR_SD)
            sigma = pm.HalfNormal("sigma", sigma=SCALE_PRIOR_SD)
            worm_tau, obs_sigma = tau, sigma

        worm_z = pm.Normal("worm_z", mu=0, sigma=1, dims="worm")
        worm_effect = pm.Deterministic("worm_effect", worm_tau * worm_z, dims="worm")
        mu = mu + worm_effect[worm_index]

        if likelihood == "student_t":
            nu = pm.Gamma("nu", alpha=NU_PRIOR_ALPHA, beta=NU_PRIOR_BETA)
            pm.StudentT(
                "log_y", nu=nu, mu=mu, sigma=obs_sigma, observed=log_y, dims="obs"
            )
        else:
            pm.Normal("log_y", mu=mu, sigma=obs_sigma, observed=log_y, dims="obs")

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=int(random_seed),
            progressbar=progressbar,
            compute_convergence_checks=False,
        )

    diagnostics = _compute_diagnostics(idata, [rv.name for rv in model.free_RVs])
    problems = []
    if diagnostics["max_rhat"] > MAX_RHAT:
        problems.append(f"max R-hat {diagnostics['max_rhat']:.3f} > {MAX_RHAT}")
    for key in ("min_ess_bulk", "min_ess_tail"):
        if diagnostics[key] < MIN_ESS:
            problems.append(f"{key} {diagnostics[key]:.0f} < {MIN_ESS}")
    if diagnostics["n_divergences"] > 0:
        problems.append(f"{diagnostics['n_divergences']} divergent transitions")
    if problems:
        warn(
            "Proportion model sampling may be unreliable: " + "; ".join(problems) + ".",
            stacklevel=2,
        )

    all_labels = table.attrs.get("experiment_labels", {})
    return ProportionModelResult(
        idata=idata,
        table=table,
        coding=coding,
        x_ref=x_ref,
        experiment_labels={e: all_labels.get(e, e) for e in experiments},
        counts=table.attrs.get("counts"),
        diagnostics=diagnostics,
        likelihood=likelihood,
        group_variances=group_variances,
        random_seed=int(random_seed),
    )
