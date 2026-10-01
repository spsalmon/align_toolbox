import logging
from dataclasses import dataclass, replace
from itertools import combinations, product
from typing import Literal
from warnings import warn

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from pytensor import tensor as pt
from scipy.interpolate import BSpline
from xarray import Dataset, DataTree

from align_toolbox.plotting.proportions import log_ratio_to_percentage

__all__ = [
    "GenotypeCoding",
    "ProportionModelResult",
    "SplineBasis",
    "build_genotype_coding",
    "build_proportion_table",
    "fit_proportion_model",
]

logger = logging.getLogger(__name__)

LIKELIHOODS = ("normal", "student_t")
REFERENCE_SHAPES = ("linear", "spline", "auto")
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

SPLINE_DEGREE = 3

LINEAR_TARGET_ACCEPT = 0.9
SPLINE_TARGET_ACCEPT = 0.99


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


# REFERENCE SPLINE


@dataclass
class SplineBasis:
    """
    Mixed-model basis of a Bayesian P-spline for the curvature of the reference curve.

    Cubic B-splines on equally spaced knots span the reference cell's min–max
    ``log_x``. Beyond that range each basis row is extended linearly at the
    boundary derivative, so ``s`` continues as a straight line. The columns are
    ``B · Dᵀ(DDᵀ)⁻¹`` for the second-order difference matrix ``D``, projected to be
    orthogonal to ``[1, xc]`` over the reference-cell observations and rescaled so
    that ``s = Z · (sd_s · z)`` with ``z ~ Normal(0, 1)`` has an RMS of about
    ``sd_s`` over those observations.

    Attributes:
        knots (np.ndarray): Full knot vector, including the 3 extra knots on each
            side, of shape ``(n_knots + 2 + 6,)``.
        lower (float): Smallest reference ``log_x``.
        upper (float): Largest reference ``log_x``.
        x_ref (float): Centering point of ``log_x``.
        penalty_map (np.ndarray): ``Dᵀ(DDᵀ)⁻¹`` of shape ``(n_basis, n_basis - 2)``.
        projection (np.ndarray): Coefficients of ``[1, xc]`` removed from each
            column, of shape ``(2, n_basis - 2)``.
        scale (float): Factor applied to the projected columns.
        rotation (np.ndarray or None): Orthogonal matrix applied last, of shape
            ``(n_basis - 2, n_basis - 2)``; it leaves the ``Normal(0, 1)`` prior
            of ``z`` unchanged. ``None`` means the identity. (default: None)
    """

    knots: np.ndarray
    lower: float
    upper: float
    x_ref: float
    penalty_map: np.ndarray
    projection: np.ndarray
    scale: float
    rotation: np.ndarray | None = None

    def _penalized_columns(self, log_x: np.ndarray) -> np.ndarray:
        """Columns ``B · Dᵀ(DDᵀ)⁻¹`` with linear tails, before projection."""
        n_basis = len(self.knots) - SPLINE_DEGREE - 1
        splines = BSpline(self.knots, np.eye(n_basis), SPLINE_DEGREE)
        clipped = np.clip(log_x, self.lower, self.upper)
        basis = splines(clipped) + (log_x - clipped)[
            :, np.newaxis
        ] * splines.derivative()(clipped)
        return basis @ self.penalty_map

    def design(self, log_x: np.ndarray) -> np.ndarray:
        """
        Evaluate the curvature basis at the given ``log_x`` values.

        Parameters:
            log_x (np.ndarray): Natural-log X values of shape ``(n_points,)``.

        Returns:
            np.ndarray: Basis of shape ``(n_points, n_basis - 2)``; linear in
                ``log_x`` outside ``[lower, upper]``.
        """
        log_x = np.asarray(log_x, dtype=float)
        linear = np.column_stack([np.ones_like(log_x), log_x - self.x_ref])
        columns = (
            self._penalized_columns(log_x) - linear @ self.projection
        ) * self.scale
        return columns if self.rotation is None else columns @ self.rotation


def _build_spline_basis(
    reference_log_x: np.ndarray, x_ref: float, n_knots: int = 10
) -> SplineBasis:
    """
    Build the P-spline curvature basis from the reference cell's ``log_x`` values.

    Parameters:
        reference_log_x (np.ndarray): ``log_x`` of the reference-cell
            observations, of shape ``(n_reference,)``.
        x_ref (float): Centering point of ``log_x``.
        n_knots (int): Number of interior knots. (default: 10)

    Returns:
        SplineBasis: The basis.

    Raises:
        ValueError: If ``n_knots`` is below 1 or the reference ``log_x`` values
            span no range.
    """
    if n_knots < 1:
        raise ValueError(f"`n_knots` must be at least 1, got {n_knots}.")
    reference_log_x = np.asarray(reference_log_x, dtype=float)
    lower, upper = float(reference_log_x.min()), float(reference_log_x.max())
    if not upper > lower:
        raise ValueError("The reference cell's log_x values span no range.")

    spacing = (upper - lower) / (n_knots + 1)
    knots = lower + spacing * np.arange(-SPLINE_DEGREE, n_knots + 2 + SPLINE_DEGREE)
    n_basis = len(knots) - SPLINE_DEGREE - 1
    difference = np.diff(np.eye(n_basis), n=2, axis=0)
    penalty_map = difference.T @ np.linalg.inv(difference @ difference.T)

    basis = SplineBasis(
        knots=knots,
        lower=lower,
        upper=upper,
        x_ref=float(x_ref),
        penalty_map=penalty_map,
        projection=np.zeros((2, n_basis - 2)),
        scale=1.0,
    )
    columns = basis._penalized_columns(reference_log_x)
    linear = np.column_stack(
        [np.ones_like(reference_log_x), reference_log_x - basis.x_ref]
    )
    basis.projection = np.linalg.lstsq(linear, columns, rcond=None)[0]
    projected = columns - linear @ basis.projection
    # E[mean(s²)] = sd_s² · ||Z||²_F / n over the reference rows
    basis.scale = float(np.sqrt(len(reference_log_x) / np.sum(projected**2)))
    return basis


def _rotate_spline_basis(
    basis: SplineBasis, log_x: np.ndarray, fixed_design: np.ndarray
) -> SplineBasis:
    """
    Rotate the curvature basis onto the principal directions of what the data can inform.

    The basis evaluated at ``log_x`` is residualized on the fixed effects and
    decomposed by SVD, and its right singular vectors become the rotation. The
    rotated coefficients keep their ``Normal(0, 1)`` prior and are uncorrelated
    in the likelihood given the fixed effects, which suits the diagonal mass
    matrix of NUTS.

    Parameters:
        basis (SplineBasis): Basis to rotate.
        log_x (np.ndarray): ``log_x`` of every observation, of shape ``(n_obs,)``.
        fixed_design (np.ndarray): Fixed-effect columns of shape
            ``(n_obs, n_fixed)``.

    Returns:
        SplineBasis: The rotated basis.
    """
    columns = basis.design(log_x)
    coefficients = np.linalg.lstsq(fixed_design, columns, rcond=None)[0]
    _, _, right = np.linalg.svd(
        columns - fixed_design @ coefficients, full_matrices=False
    )
    rotation = right.T if basis.rotation is None else basis.rotation @ right.T
    return replace(basis, rotation=rotation)


def _smoothness_prior_rate(upper: float, prob: float) -> float:
    """
    Return the rate of the exponential prior on ``sd_s`` with ``P(sd_s > upper) = prob``.

    Parameters:
        upper (float): Curvature RMS, in natural-log units, that is a priori
            unlikely to be exceeded.
        prob (float): Prior probability of exceeding ``upper``.

    Returns:
        float: Rate ``-log(prob) / upper``.

    Raises:
        ValueError: If ``upper`` is not positive or ``prob`` is not strictly
            between 0 and 1.
    """
    if not upper > 0:
        raise ValueError(f"`smooth_sd_prior_upper` must be positive, got {upper}.")
    if not 0 < prob < 1:
        raise ValueError(f"`smooth_sd_prior_prob` must be between 0 and 1, got {prob}.")
    return float(-np.log(prob) / upper)


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
        reference_shape_used (str): ``"linear"`` or ``"spline"``. (default: "linear")
        spline_basis (SplineBasis or None): Curvature basis of the spline
            reference curve; ``None`` for a line. (default: None)
        linear_misfit (pd.DataFrame or None): ``reference_residuals_by_molt`` of
            the linear fit, when one was made. (default: None)
        spline_misfit (pd.DataFrame or None): ``reference_residuals_by_molt`` of
            the spline fit, when one was made. (default: None)
        linearity_threshold (float or None): Misfit threshold, as a fraction,
            used by ``reference_shape="auto"``. (default: None)
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
    reference_shape_used: str = "linear"
    spline_basis: SplineBasis | None = None
    linear_misfit: pd.DataFrame | None = None
    spline_misfit: pd.DataFrame | None = None
    linearity_threshold: float | None = None

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
        of each cell's ``tau`` and ``sigma`` to the reference cell's. A spline
        fit adds ``sd_s``.

        With a Student-t likelihood ``sigma`` is a scale, not a standard
        deviation, so ``sigma_sd`` rows give ``sigma * sqrt(nu / (nu - 2))``
        computed per draw. Draws with ``nu <= 2`` have no finite SD and are
        excluded; their fraction is the ``mean`` of the ``sigma_sd_undefined_fraction``
        row. The ``sigma`` ratios stay valid as SD ratios because ``nu`` is
        shared by all cells.

        Returns:
            pd.DataFrame: Columns ``parameter``, ``mean``, ``lower``, ``upper``,
                ``mean_percent``, ``lower_percent`` and ``upper_percent``; the
                percent columns are NaN except for ``beta`` rows.
                ``attrs["reference_shape"]`` is ``"linear"`` or ``"spline"``.
        """
        rows = []

        def add(parameter, draws, percent=False):
            row = {
                "parameter": parameter,
                "mean": np.nanmean(draws),
                "lower": np.nanquantile(draws, 0.025),
                "upper": np.nanquantile(draws, 0.975),
            }
            for key in ("mean", "lower", "upper"):
                row[f"{key}_percent"] = (
                    log_ratio_to_percentage(row[key]) if percent else np.nan
                )
            rows.append(row)

        add("a", self._draws("a"))
        add("b", self._draws("b"))
        if self._has("sd_s"):
            add("sd_s", self._draws("sd_s"))
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
        if self._has("nu"):
            nu = self._draws("nu")
            defined = nu > 2
            sd_factor = np.full_like(nu, np.nan)
            sd_factor[defined] = np.sqrt(nu[defined] / (nu[defined] - 2))
            sigma = self._draws("sigma")
            if self.group_variances:
                for k, cell in enumerate(self.coding.cells):
                    add(f"sigma_sd[{cell}]", sigma[:, k] * sd_factor)
            else:
                add("sigma_sd", sigma * sd_factor)
            rows.append(
                {"parameter": "sigma_sd_undefined_fraction", "mean": 1 - defined.mean()}
            )
        summary = pd.DataFrame(rows, columns=list(rows[0]))
        summary.attrs["reference_shape"] = self.reference_shape_used
        return summary

    def group_offset_curve(
        self, cell: int | str | dict[str, str], log_x_grid: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute the size-matched log-ratio offset of a genotype cell relative to the reference cell.

        At each ``log_x``, the offset is the sum over the cell's active effects
        (main effects and interactions) of ``beta_k + gamma_k * (log_x - x_ref)``.
        Genotype effects are linear even when the reference curve is a spline.

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

    def _reference_curve_draws(self, log_x: np.ndarray) -> np.ndarray:
        """Draws of ``f(log_x)`` for the average experiment, of shape ``(n_draws, n_points)``."""
        x_centered = np.asarray(log_x, dtype=float) - self.x_ref
        curve = (
            self._draws("a")[:, np.newaxis]
            + self._draws("b")[:, np.newaxis] * x_centered[np.newaxis, :]
        )
        if self.spline_basis is not None:
            weights = self._draws("sd_s")[:, np.newaxis] * self._draws("spline_z")
            curve = curve + weights @ self.spline_basis.design(log_x).T
        return curve

    def _marginal_fitted_draws(
        self, rows: pd.DataFrame, genotype_effects: bool
    ) -> np.ndarray:
        """Draws of the fitted value of ``rows`` without worm effects, of shape ``(n_draws, n_rows)``."""
        log_x = rows["log_x"].to_numpy(dtype=float)
        fitted = self._reference_curve_draws(log_x)
        if genotype_effects and self.coding.effect_names:
            design = self.coding.design_matrix(
                sorted(rows["condition_id"].unique())
            ).loc[rows["condition_id"]]
            design = design.to_numpy()
            x_centered = log_x - self.x_ref
            fitted = (
                fitted
                + self._draws("beta") @ design.T
                + self._draws("gamma") @ (design * x_centered[:, np.newaxis]).T
            )
        if self._has("experiment_effect"):
            experiment_index = self._index_of(
                "experiment_effect", "experiment", rows["experiment"]
            )
            fitted = fitted + self._draws("experiment_effect")[:, experiment_index]
        return fitted

    def reference_residuals_by_molt(self) -> pd.DataFrame:
        """
        Compute the posterior mean residual of the reference cell at each molt, in percent.

        Residuals are marginal: ``log_y`` minus the fitted reference curve
        (``a + b * xc``, plus ``s(xc)`` for a spline) and the experiment effect,
        without the worm intercepts, which could otherwise absorb part of the
        misfit, especially for worms with missing molts. They are averaged over
        the reference observations of each molt, then back-transformed with
        ``log_ratio_to_percentage``. Values near 0 at every molt mean the curve
        describes the reference cell.

        Returns:
            pd.DataFrame: Columns ``molt``, ``n_observations``, ``mean_percent``,
                ``lower_percent`` and ``upper_percent`` (95% equal-tailed interval).
        """
        reference = self.table[self._reference_mask()]
        fitted = self._marginal_fitted_draws(reference, genotype_effects=False)
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

    def residuals(self) -> pd.DataFrame:
        """
        Compute the posterior mean residual of every observation around its genotype's curve.

        The fitted value is the reference curve plus the genotype's
        ``beta + gamma * xc`` terms and the experiment effect, without the worm
        intercepts.

        Returns:
            pd.DataFrame: Columns ``condition_id``, ``worm_id``, ``molt``,
                ``experiment`` and ``residual`` (natural-log units), aligned with
                ``table.index``.
        """
        fitted = self._marginal_fitted_draws(self.table, genotype_effects=True)
        residual = self.table["log_y"].to_numpy() - fitted.mean(axis=0)
        return self.table[["condition_id", "worm_id", "molt", "experiment"]].assign(
            residual=residual
        )

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

    def reference_curve(self, log_x_grid: np.ndarray) -> pd.DataFrame:
        """
        Compute the fitted reference curve of the average experiment with a 95% band.

        Outside the reference cell's min–max ``log_x`` the curve is a linear
        extrapolation: a spline continues at its boundary slope.

        Parameters:
            log_x_grid (np.ndarray): Natural-log X values of shape ``(n_points,)``.

        Returns:
            pd.DataFrame: Columns ``log_x``, ``mean``, ``lower``, ``upper`` (95%
                equal-tailed band, natural-log units) and ``extrapolated``
                (``True`` outside the reference range).
        """
        log_x_grid = np.asarray(log_x_grid, dtype=float)
        curve = self._reference_curve_draws(log_x_grid)
        reference_log_x = self.table.loc[self._reference_mask(), "log_x"]
        return pd.DataFrame(
            {
                "log_x": log_x_grid,
                "mean": curve.mean(axis=0),
                "lower": np.quantile(curve, 0.025, axis=0),
                "upper": np.quantile(curve, 0.975, axis=0),
                "extrapolated": (log_x_grid < reference_log_x.min())
                | (log_x_grid > reference_log_x.max()),
            }
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
        residual ``Normal(0, sigma)`` (or ``StudentT(nu, 0, sigma)``) to the
        reference curve of the average experiment, using the reference cell's
        ``tau`` and ``sigma``.

        Parameters:
            log_x_grid (np.ndarray): Natural-log X values of shape ``(n_points,)``.
            prob (float): Probability mass of the equal-tailed interval.
                (default: 0.95)
            random_seed (int or None): Seed for the predictive draws; ``None``
                reuses the sampling seed. (default: None)

        Returns:
            pd.DataFrame: Columns ``log_x``, ``mean`` (posterior mean of the
                curve), ``lower`` and ``upper``.

        Raises:
            ValueError: If ``prob`` is not strictly between 0 and 1.
        """
        if not 0 < prob < 1:
            raise ValueError(f"`prob` must be between 0 and 1, got {prob}.")
        rng = np.random.default_rng(
            self.random_seed if random_seed is None else random_seed
        )
        log_x_grid = np.asarray(log_x_grid, dtype=float)
        curve = self._reference_curve_draws(log_x_grid)
        n_draws, n_points = curve.shape
        tau = self._scale_draws("tau")[:, 0, np.newaxis]
        sigma = self._scale_draws("sigma")[:, 0, np.newaxis]
        worm = tau * rng.standard_normal((n_draws, 1))
        if self._has("nu"):
            noise = rng.standard_t(
                self._draws("nu")[:, np.newaxis], (n_draws, n_points)
            )
        else:
            noise = rng.standard_normal((n_draws, n_points))
        predicted = curve + worm + sigma * noise
        tail = (1 - prob) / 2
        return pd.DataFrame(
            {
                "log_x": log_x_grid,
                "mean": curve.mean(axis=0),
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

    # a plain Dataset is accepted by both arviz 0.x (Python < 3.12) and arviz 1.x
    posterior = Dataset({name: idata["posterior"][name] for name in var_names})
    return {
        "max_rhat": extreme(az.rhat(posterior), np.nanmax),
        "min_ess_bulk": extreme(az.ess(posterior, method="bulk"), np.nanmin),
        "min_ess_tail": extreme(az.ess(posterior, method="tail"), np.nanmin),
        "n_divergences": int(idata["sample_stats"]["diverging"].values.sum()),
    }


def _sample_proportion_model(
    table: pd.DataFrame,
    coding: GenotypeCoding,
    x_ref: float,
    likelihood: str,
    group_variances: bool,
    spline_basis: SplineBasis | None,
    smooth_sd_rate: float,
    sampler_kwargs: dict,
) -> ProportionModelResult:
    """
    Build the proportion model with a linear or spline reference curve and sample it.

    Parameters:
        table (pd.DataFrame): Output of ``build_proportion_table``.
        coding (GenotypeCoding): Genotype coding of the conditions in ``table``.
        x_ref (float): Centering point of ``log_x``.
        likelihood (str): ``"normal"`` or ``"student_t"``.
        group_variances (bool): Whether ``tau`` and ``sigma`` vary by cell.
        spline_basis (SplineBasis or None): Curvature basis; ``None`` fits a line.
        smooth_sd_rate (float): Rate of the exponential prior on ``sd_s``.
        sampler_kwargs (dict): Keyword arguments of ``pymc.sample``; must include
            ``random_seed``.

    Returns:
        ProportionModelResult: Posterior, data, coding and diagnostics. A warning
            is emitted if R-hat exceeds 1.01, an ESS is below 400, or any
            transition diverged.
    """
    condition_ids = sorted(int(c) for c in table["condition_id"].unique())
    cell_of_row = table["condition_id"].map(coding.cell_of_condition)
    cell_index = pd.Index(coding.cells)
    obs_cell = cell_index.get_indexer(cell_of_row)
    is_reference = obs_cell == 0

    log_x = table["log_x"].to_numpy(dtype=float)
    log_y = table["log_y"].to_numpy(dtype=float)
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
    if spline_basis is not None:
        fixed_design = np.column_stack(
            [
                np.ones_like(x_centered),
                x_centered,
                design,
                design * x_centered[:, None],
                np.eye(len(experiments))[experiment_index],
            ]
        )
        spline_basis = _rotate_spline_basis(spline_basis, log_x, fixed_design)
        spline_design = spline_basis.design(log_x)
        coords["spline_basis"] = np.arange(spline_design.shape[1])
    with pm.Model(coords=coords) as model:
        a = pm.Normal("a", mu=log_y[is_reference].mean(), sigma=INTERCEPT_PRIOR_SD)
        b = pm.Normal("b", mu=SLOPE_PRIOR_MEAN, sigma=SLOPE_PRIOR_SD)
        mu = a + b * x_centered
        if spline_basis is not None:
            sd_s = pm.Exponential("sd_s", lam=smooth_sd_rate)
            spline_z = pm.Normal("spline_z", mu=0, sigma=1, dims="spline_basis")
            mu = mu + pt.dot(spline_design, sd_s * spline_z)
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

        idata = pm.sample(**sampler_kwargs, compute_convergence_checks=False)

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
        shape = "spline" if spline_basis is not None else "linear"
        warn(
            f"Proportion model sampling ({shape} reference) may be unreliable: "
            + "; ".join(problems)
            + ".",
            stacklevel=3,
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
        random_seed=int(sampler_kwargs["random_seed"]),
        reference_shape_used="linear" if spline_basis is None else "spline",
        spline_basis=spline_basis,
    )


def _max_abs_log_misfit(misfit: pd.DataFrame) -> tuple[float, int]:
    """Largest absolute posterior-mean per-molt residual in log units, and its molt."""
    log_misfit = np.abs(np.log1p(misfit["mean_percent"].to_numpy() / 100))
    worst = int(np.argmax(log_misfit))
    return float(log_misfit[worst]), misfit["molt"].iloc[worst]


def fit_proportion_model(
    table: pd.DataFrame,
    factors: dict[int, dict[str, str]] | None = None,
    reference: dict[str, str] | None = None,
    reference_condition: int | None = None,
    likelihood: Literal["normal", "student_t"] = "normal",
    group_variances: bool = False,
    x_ref: float | None = None,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float | None = None,
    *,
    random_seed: int,
    progressbar: bool = False,
    cores: int | None = None,
    reference_shape: Literal["linear", "spline", "auto"] = "auto",
    linearity_threshold: float = 0.03,
    n_knots: int = 10,
    smooth_sd_prior_upper: float = 0.05,
    smooth_sd_prior_prob: float = 0.05,
) -> ProportionModelResult:
    """
    Fit a hierarchical log-log proportion model jointly to all genotypes with NUTS.

    The model is ``log_y = f(xc) + sum_k D_k * (beta_k + gamma_k * xc) + e_exp
    + u_worm + eps`` with ``xc = log_x - x_ref``, treatment-coded genotype effects
    ``D_k`` (main effects and interactions), sum-to-zero experiment effects,
    non-centered worm intercepts ``u ~ Normal(0, tau)`` and residuals
    ``eps ~ Normal(0, sigma)`` or ``StudentT(nu, 0, sigma)``. For a Student-t
    likelihood ``sigma`` is a scale; its SD is ``sigma * sqrt(nu / (nu - 2))``,
    reported as ``sigma_sd`` by ``summary``. ``a`` and ``beta`` describe the
    average experiment at body size ``x_ref``.

    The reference curve ``f`` is shared by all genotype cells. It is the line
    ``a + b * xc``, or ``a + b * xc + s(xc)`` with a Bayesian P-spline ``s``
    (Lang & Brezger 2004): cubic B-splines on ``n_knots`` equally spaced interior
    knots over the reference cell's min–max ``log_x``, a second-order difference
    penalty in mixed-model form, and ``s`` orthogonal to ``[1, xc]`` over the
    reference observations so that ``b`` stays the overall exponent. Its size
    ``sd_s`` (about the RMS of ``s`` in log units) has an exponential prior with
    ``P(sd_s > smooth_sd_prior_upper) = smooth_sd_prior_prob``, which shrinks
    ``f`` toward the line (a penalized-complexity prior; Simpson et al. 2017,
    Stat Sci 32:1). Observations of all genotypes inform ``s``, so every genotype
    curve inherits the reference curvature, with linear ``beta``/``gamma`` terms
    on top. Beyond the reference range ``f`` is a linear extrapolation at the
    boundary slope.

    With ``reference_shape="auto"``, the line is fitted first; if the largest
    absolute posterior-mean marginal reference residual of a molt (see
    ``ProportionModelResult.reference_residuals_by_molt``) exceeds
    ``log(1 + linearity_threshold)``, the model is refitted with the spline. The
    decision and the maximum misfit are logged at INFO level.

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
        draws (int): Posterior draws per chain. (default: 2000)
        tune (int): Tuning steps per chain. (default: 2000)
        chains (int): Number of chains. (default: 4)
        target_accept (float or None): NUTS target acceptance rate; ``None``
            uses 0.9 for the line and 0.99 for the spline, whose smoothness
            scale ``sd_s`` otherwise causes divergent transitions. (default: None)
        random_seed (int): Seed for sampling; required for reproducibility.
        progressbar (bool): Show the sampling progress bar. (default: False)
        cores (int or None): Chains sampled in parallel; ``None`` keeps PyMC's
            default. (default: None)
        reference_shape (str): ``"linear"``, ``"spline"`` or ``"auto"``.
            (default: "auto")
        linearity_threshold (float): Per-molt misfit of the line, as a fraction,
            above which ``"auto"`` switches to the spline. (default: 0.03)
        n_knots (int): Number of interior spline knots. (default: 10)
        smooth_sd_prior_upper (float): Upper value of the ``sd_s`` prior, in
            natural-log units. (default: 0.05)
        smooth_sd_prior_prob (float): Prior probability that ``sd_s`` exceeds
            ``smooth_sd_prior_upper``. (default: 0.05)

    Returns:
        ProportionModelResult: Posterior, data, coding and diagnostics of the
            selected fit, with the reference shape used and the per-molt
            misfit tables of the fits that were made. A warning is emitted if
            R-hat exceeds 1.01, an ESS is below 400, or any transition diverged.

    Raises:
        ValueError: If ``likelihood`` or ``reference_shape`` is unknown,
            ``random_seed`` is not an int, the table is empty, the spline
            settings are invalid, or the genotype coding is invalid.
    """
    if likelihood not in LIKELIHOODS:
        raise ValueError(
            f"Invalid likelihood {likelihood!r}; expected one of {LIKELIHOODS}."
        )
    if reference_shape not in REFERENCE_SHAPES:
        raise ValueError(
            f"Invalid reference_shape {reference_shape!r}; expected one of "
            f"{REFERENCE_SHAPES}."
        )
    if not linearity_threshold > 0:
        raise ValueError(
            f"`linearity_threshold` must be positive, got {linearity_threshold}."
        )
    if not isinstance(random_seed, (int, np.integer)) or isinstance(random_seed, bool):
        raise ValueError(f"`random_seed` must be an int, got {random_seed!r}.")
    if len(table) == 0:
        raise ValueError("The table has no observations.")
    smooth_sd_rate = _smoothness_prior_rate(smooth_sd_prior_upper, smooth_sd_prior_prob)

    condition_ids = sorted(int(c) for c in table["condition_id"].unique())
    coding = build_genotype_coding(
        condition_ids, factors, reference, reference_condition
    )
    is_reference = (
        table["condition_id"].map(coding.cell_of_condition) == coding.reference_cell
    ).to_numpy()
    reference_log_x = table["log_x"].to_numpy(dtype=float)[is_reference]
    if x_ref is None:
        x_ref = float(np.median(reference_log_x))
    spline_basis = None
    if reference_shape != "linear":
        spline_basis = _build_spline_basis(reference_log_x, x_ref, n_knots)

    sampler_kwargs = {
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "cores": cores,
        "random_seed": int(random_seed),
        "progressbar": progressbar,
    }

    def sample(basis):
        default_accept = LINEAR_TARGET_ACCEPT if basis is None else SPLINE_TARGET_ACCEPT
        return _sample_proportion_model(
            table,
            coding,
            x_ref,
            likelihood,
            group_variances,
            basis,
            smooth_sd_rate,
            {
                **sampler_kwargs,
                "target_accept": (
                    default_accept if target_accept is None else target_accept
                ),
            },
        )

    linear_misfit = None
    if reference_shape != "spline":
        result = sample(None)
        linear_misfit = result.reference_residuals_by_molt()
        result.linear_misfit = linear_misfit
        if reference_shape == "linear":
            return result
        max_misfit, worst_molt = _max_abs_log_misfit(linear_misfit)
        use_spline = max_misfit > np.log1p(linearity_threshold)
        logger.info(
            "Proportion model reference shape: the line's largest per-molt misfit "
            "is %.2f%% (log-ratio %.4f, molt %s) against a threshold of %.2f%%; %s.",
            100 * np.expm1(max_misfit),
            max_misfit,
            worst_molt,
            100 * linearity_threshold,
            "refitting with a spline" if use_spline else "keeping the line",
        )
        result.linearity_threshold = linearity_threshold
        if not use_spline:
            return result

    result = sample(spline_basis)
    result.linear_misfit = linear_misfit
    result.spline_misfit = result.reference_residuals_by_molt()
    if reference_shape == "auto":
        result.linearity_threshold = linearity_threshold
    return result
