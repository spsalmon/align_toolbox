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
from scipy.sparse import csr_matrix
from scipy.stats import kurtosis, levene, norm
from statsmodels.stats.multitest import multipletests
from xarray import Dataset, DataTree

from align_toolbox.plotting.proportions import log_ratio_to_percentage

__all__ = [
    "GenotypeCoding",
    "ProportionModelResult",
    "SplineBasis",
    "build_genotype_coding",
    "build_proportion_table",
    "compare_experiments",
    "fit_per_experiment",
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

# robust SD = MAD_SCALE * median absolute deviation, for normal data
MAD_SCALE = 1.4826
OUTLIER_ROBUST_SDS = 3.0

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

    def _draws(self, name: str, index: np.ndarray | None = None) -> np.ndarray:
        """Posterior draws of ``name`` with chains and draws flattened to axis 0, optionally subset by ``index``."""
        values = self.idata["posterior"][name].values
        values = values.reshape(-1, *values.shape[2:])
        return values if index is None else values[index]

    def _has(self, name: str) -> bool:
        return name in self.idata["posterior"].data_vars

    def _scale_draws(self, name: str, index: np.ndarray | None = None) -> np.ndarray:
        """Draws of ``tau`` or ``sigma`` of shape ``(n_draws, n_cells)``."""
        draws = self._draws(name, index)
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

    def _reference_curve_draws(
        self, log_x: np.ndarray, index: np.ndarray | None = None
    ) -> np.ndarray:
        """Draws of ``f(log_x)`` for the average experiment, of shape ``(n_draws, n_points)``."""
        x_centered = np.asarray(log_x, dtype=float) - self.x_ref
        curve = (
            self._draws("a", index)[:, np.newaxis]
            + self._draws("b", index)[:, np.newaxis] * x_centered[np.newaxis, :]
        )
        if self.spline_basis is not None:
            weights = self._draws("sd_s", index)[:, np.newaxis] * self._draws(
                "spline_z", index
            )
            curve = curve + weights @ self.spline_basis.design(log_x).T
        return curve

    def _marginal_fitted_draws(
        self,
        rows: pd.DataFrame,
        genotype_effects: bool,
        index: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draws of the fitted value of ``rows`` without worm effects, of shape ``(n_draws, n_rows)``."""
        log_x = rows["log_x"].to_numpy(dtype=float)
        fitted = self._reference_curve_draws(log_x, index)
        if genotype_effects and self.coding.effect_names:
            design = self.coding.design_matrix(
                sorted(rows["condition_id"].unique())
            ).loc[rows["condition_id"]]
            design = design.to_numpy()
            x_centered = log_x - self.x_ref
            fitted = (
                fitted
                + self._draws("beta", index) @ design.T
                + self._draws("gamma", index) @ (design * x_centered[:, np.newaxis]).T
            )
        if self._has("experiment_effect"):
            experiment_index = self._index_of(
                "experiment_effect", "experiment", rows["experiment"]
            )
            fitted = (
                fitted + self._draws("experiment_effect", index)[:, experiment_index]
            )
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

    # PUBLICATION ANALYSES

    def _draw_index(self, n_draws: int | None) -> np.ndarray | None:
        """Evenly spaced flattened draw indices, or ``None`` for all draws."""
        total = len(self._draws("a"))
        if n_draws is None or n_draws >= total:
            return None
        if n_draws < 1:
            raise ValueError(f"`n_draws` must be at least 1, got {n_draws}.")
        return np.linspace(0, total - 1, n_draws).round().astype(int)

    def _rng(self, random_seed: int | None) -> np.random.Generator:
        return np.random.default_rng(
            self.random_seed if random_seed is None else random_seed
        )

    def _row_cells(self) -> pd.Series:
        """Genotype cell label of each table row."""
        return self.table["condition_id"].map(self.coding.cell_of_condition)

    def _reference_range(self) -> tuple[float, float]:
        """Min and max ``log_x`` of the reference cell."""
        reference_log_x = self.table.loc[self._reference_mask(), "log_x"]
        return float(reference_log_x.min()), float(reference_log_x.max())

    def _offset_draws(
        self,
        effect_vector: np.ndarray,
        log_x: np.ndarray,
        index: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draws of ``sum_k d_k * (beta_k + gamma_k * xc)``, of shape ``(n_draws, n_points)``."""
        x_centered = np.asarray(log_x, dtype=float) - self.x_ref
        if not self.coding.effect_names:
            return np.zeros((len(self._draws("a", index)), len(x_centered)))
        intercept = self._draws("beta", index) @ effect_vector
        slope = self._draws("gamma", index) @ effect_vector
        return intercept[:, np.newaxis] + slope[:, np.newaxis] * x_centered

    def cell_offsets(
        self,
        log_x: float | None = None,
        prob: float = 0.95,
        n_draws: int | None = None,
    ) -> pd.DataFrame:
        """
        Compute each genotype cell's size-matched offset from the reference cell, split into main effects and interaction.

        ``offset`` is ``sum_k D_k * (beta_k + gamma_k * xc)`` over all the cell's
        active effects at ``log_x``; ``expected`` keeps only the main effects, i.e.
        the offset an additive model would predict from the single-factor cells,
        and ``interaction`` is ``offset - expected``. All three are computed per
        posterior draw. ``interaction`` is NaN for cells with no interaction term
        (the reference, single-factor cells, and every cell under single-factor
        coding, where ``expected`` equals ``offset``).

        Parameters:
            log_x (float or None): Natural-log X at which to evaluate the offsets;
                ``None`` uses ``x_ref``. (default: None)
            prob (float): Probability mass of the equal-tailed intervals.
                (default: 0.95)
            n_draws (int or None): Number of evenly spaced posterior draws to use;
                ``None`` uses all. (default: None)

        Returns:
            pd.DataFrame: One row per cell with ``cell``, ``log_x``, the
                ``mean``, ``lower`` and ``upper`` of ``offset``, ``expected`` and
                ``interaction`` (natural-log units, e.g. ``offset_mean``) and their
                ``_percent`` versions, ``interaction_prob_positive`` (posterior
                probability that the interaction is positive),
                ``within_cell_range`` and ``within_reference_range`` (whether
                ``log_x`` lies inside the cell's or the reference cell's observed
                min–max ``log_x``; ``False`` means the offset is extrapolated).

        Raises:
            ValueError: If ``prob`` is not strictly between 0 and 1.
        """
        _check_prob(prob)
        log_x = self.x_ref if log_x is None else float(log_x)
        index = self._draw_index(n_draws)
        row_cells = self._row_cells().to_numpy()
        reference_lower, reference_upper = self._reference_range()
        is_main = np.array(
            [":" not in e for e in self.coding.effect_names], dtype=float
        )

        rows = []
        for cell in self.coding.cells:
            d = self.coding.effect_vector(self.coding.resolve_cell(cell))
            offset = self._offset_draws(d, [log_x], index)[:, 0]
            expected = self._offset_draws(d * is_main, [log_x], index)[:, 0]
            has_interaction = bool(np.any(d * (1 - is_main)))
            interaction = (
                offset - expected if has_interaction else np.full_like(offset, np.nan)
            )
            cell_log_x = self.table["log_x"].to_numpy()[row_cells == cell]
            row = {"cell": cell, "log_x": log_x}
            row.update(_summarize_draws(offset, prob, "offset", percent=True))
            row.update(_summarize_draws(expected, prob, "expected", percent=True))
            row.update(_summarize_draws(interaction, prob, "interaction", percent=True))
            row["interaction_prob_positive"] = (
                float(np.mean(interaction > 0)) if has_interaction else np.nan
            )
            row["within_cell_range"] = bool(
                cell_log_x.min() <= log_x <= cell_log_x.max()
            )
            row["within_reference_range"] = bool(
                reference_lower <= log_x <= reference_upper
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def molt_positions(
        self, prob: float = 0.95, n_draws: int | None = None
    ) -> pd.DataFrame:
        """
        Summarize where each genotype cell sits at each molt, in the data and in the model.

        ``model_offset`` is the cell's offset (see ``cell_offsets``) at the cell's
        mean ``log_x`` for that molt. ``observed_offset`` is the mean over the
        cell × molt observations of ``log_y - f(log_x) - e_exp``, the data's own
        deviation from the reference curve; its interval comes from the posterior
        draws of ``f`` and the experiment effects. Because genotype effects are
        linear in ``log_x``, the two agree when the model fits, up to the
        average worm intercept and residual of the group.

        Parameters:
            prob (float): Probability mass of the equal-tailed intervals.
                (default: 0.95)
            n_draws (int or None): Number of evenly spaced posterior draws to use;
                ``None`` uses all. (default: None)

        Returns:
            pd.DataFrame: One row per cell × molt with ``cell``, ``molt``, ``n``,
                ``log_x_mean``, ``log_x_sd``, ``log_y_mean``, ``log_y_sd`` (raw data,
                SD with ``ddof=1``), the ``mean``, ``lower`` and ``upper`` of
                ``model_offset`` and ``observed_offset`` (natural-log units) with
                their ``_percent`` versions, and ``extrapolated`` (``True`` if the
                mean ``log_x`` is below or above the reference cell's overall
                ``log_x`` range; gaps between reference molts do not count).

        Raises:
            ValueError: If ``prob`` is not strictly between 0 and 1.
        """
        _check_prob(prob)
        index = self._draw_index(n_draws)
        row_cells = self._row_cells().to_numpy()
        molts = self.table["molt"].to_numpy()
        log_x, log_y = (self.table[c].to_numpy(dtype=float) for c in ("log_x", "log_y"))
        deviation = log_y - self._marginal_fitted_draws(
            self.table, genotype_effects=False, index=index
        )
        reference_lower, reference_upper = self._reference_range()

        rows = []
        for cell in self.coding.cells:
            d = self.coding.effect_vector(self.coding.resolve_cell(cell))
            for molt in sorted(np.unique(molts[row_cells == cell])):
                in_group = (row_cells == cell) & (molts == molt)
                mean_log_x = float(log_x[in_group].mean())
                row = {
                    "cell": cell,
                    "molt": molt,
                    "n": int(in_group.sum()),
                    "log_x_mean": mean_log_x,
                    "log_x_sd": _sd(log_x[in_group]),
                    "log_y_mean": float(log_y[in_group].mean()),
                    "log_y_sd": _sd(log_y[in_group]),
                }
                model = self._offset_draws(d, [mean_log_x], index)[:, 0]
                observed = deviation[:, in_group].mean(axis=1)
                row.update(_summarize_draws(model, prob, "model_offset", percent=True))
                row.update(
                    _summarize_draws(observed, prob, "observed_offset", percent=True)
                )
                row["extrapolated"] = bool(
                    mean_log_x < reference_lower or mean_log_x > reference_upper
                )
                rows.append(row)
        return pd.DataFrame(rows)

    def _residual_sd_factor(self, index: np.ndarray | None) -> np.ndarray:
        """Per-draw factor from ``sigma`` to the residual SD; NaN where ``nu <= 2``."""
        if not self._has("nu"):
            return np.ones(len(self._draws("a", index)))
        nu = self._draws("nu", index)
        factor = np.full_like(nu, np.nan)
        defined = nu > 2
        factor[defined] = np.sqrt(nu[defined] / (nu[defined] - 2))
        return factor

    def repeatability(
        self, prob: float = 0.95, n_draws: int | None = None
    ) -> pd.DataFrame:
        """
        Compute the repeatability of each genotype cell, the share of variance between worms.

        ``R = tau² / (tau² + sigma_sd²)`` per posterior draw, where ``sigma_sd`` is
        the SD of the residual distribution: ``sigma`` for a normal likelihood and
        ``sigma * sqrt(nu / (nu - 2))`` for a Student-t one. Student-t draws with
        ``nu <= 2`` have no finite SD and are excluded; their fraction is
        reported. The ratio to the reference cell is also taken per draw. With
        shared variances (``group_variances=False``) every cell has the same
        ``R``, so a single row labelled ``"shared"`` is returned and the ratio
        columns are NaN.

        Parameters:
            prob (float): Probability mass of the equal-tailed intervals.
                (default: 0.95)
            n_draws (int or None): Number of evenly spaced posterior draws to use;
                ``None`` uses all. (default: None)

        Returns:
            pd.DataFrame: One row per cell with ``cell``, ``repeatability_mean``,
                ``repeatability_lower``, ``repeatability_upper``, the same for
                ``repeatability_ratio`` (relative to the reference cell), and
                ``sd_undefined_fraction`` (fraction of draws with ``nu <= 2``).

        Raises:
            ValueError: If ``prob`` is not strictly between 0 and 1.
        """
        _check_prob(prob)
        index = self._draw_index(n_draws)
        factor = self._residual_sd_factor(index)
        tau = self._scale_draws("tau", index)
        residual_sd = self._scale_draws("sigma", index) * factor[:, np.newaxis]
        ratio = tau**2 / (tau**2 + residual_sd**2)
        undefined_fraction = float(np.mean(np.isnan(factor)))

        cells = self.coding.cells if self.group_variances else ["shared"]
        rows = []
        for k, cell in enumerate(cells):
            row = {"cell": cell}
            row.update(_summarize_draws(ratio[:, k], prob, "repeatability"))
            relative = (
                ratio[:, k] / ratio[:, 0]
                if self.group_variances
                else np.full(len(ratio), np.nan)
            )
            row.update(_summarize_draws(relative, prob, "repeatability_ratio"))
            row["sd_undefined_fraction"] = undefined_fraction
            rows.append(row)
        return pd.DataFrame(rows)

    def worm_deviations(
        self,
        relative_to: Literal["own", "reference"] = "own",
        n_draws: int | None = None,
    ) -> pd.DataFrame:
        """
        Tabulate the posterior mean marginal deviation of every worm at every molt.

        Deviations exclude the worm intercepts. With ``relative_to="own"`` they are
        residuals around the worm's genotype curve (``f`` plus its
        ``beta + gamma * xc`` and the experiment effect), as in ``residuals``; with
        ``"reference"`` they are deviations from the reference curve ``f`` plus
        the experiment effect, i.e. they include the genotype's offset.

        Parameters:
            relative_to (str): ``"own"`` or ``"reference"``. (default: "own")
            n_draws (int or None): Number of evenly spaced posterior draws to use;
                ``None`` uses all. (default: None)

        Returns:
            pd.DataFrame: One row per worm with ``worm_id``, ``condition_id``,
                ``experiment`` and one column per molt (named by the molt index)
                holding the deviation in natural-log units, NaN where the worm has
                no observation at that molt.

        Raises:
            ValueError: If ``relative_to`` is not ``"own"`` or ``"reference"``.
        """
        if relative_to not in ("own", "reference"):
            raise ValueError(
                f"`relative_to` must be 'own' or 'reference', got {relative_to!r}."
            )
        fitted = self._marginal_fitted_draws(
            self.table,
            genotype_effects=relative_to == "own",
            index=self._draw_index(n_draws),
        )
        deviation = self.table.assign(
            deviation=self.table["log_y"].to_numpy() - fitted.mean(axis=0)
        )
        wide = deviation.pivot(index="worm_id", columns="molt", values="deviation")
        wide.columns.name = None
        worms = self.table.drop_duplicates("worm_id")[
            ["worm_id", "condition_id", "experiment"]
        ]
        return worms.merge(wide, left_on="worm_id", right_index=True).reset_index(
            drop=True
        )

    def penetrance(
        self,
        prob: float = 0.95,
        random_seed: int | None = None,
        n_draws: int | None = None,
        n_simulations: int = 2000,
        interval_prob: float = 0.95,
    ) -> pd.DataFrame:
        """
        Estimate the fraction of each cell's worms that lie outside the reference cell's range of worms.

        For each worm and posterior draw, the worm's deviation is the mean over its
        ``n_i`` observed molts of ``log_y - f(log_x) - e_exp``. It is compared with
        the central ``prob`` interval of the same statistic for a new
        reference-cell worm observed ``n_i`` times: a worm intercept
        ``Normal(0, tau_ref)`` plus the mean of ``n_i`` residuals
        ``Normal(0, sigma_ref)`` (analytic) or ``StudentT(nu, 0, sigma_ref)``
        (``n_simulations`` simulations per draw). Each worm is classified per draw
        as below, within or above that interval, and the fractions of the cell's
        worms below, above and outside are summarized over draws. For the
        reference cell the outside fraction should be close to ``1 - prob``, which
        checks the calibration.

        Deviations at ``log_x`` beyond the reference cell's range rely on the
        extrapolated reference curve; ``extrapolated_fraction`` reports how many
        of the cell's observations are affected.

        Parameters:
            prob (float): Probability mass of the reference interval.
                (default: 0.95)
            random_seed (int or None): Seed for the Student-t simulations;
                ``None`` reuses the sampling seed. (default: None)
            n_draws (int or None): Number of evenly spaced posterior draws to use;
                ``None`` uses all. (default: None)
            n_simulations (int): Simulated reference worms per draw and number of
                molts, for a Student-t likelihood. (default: 2000)
            interval_prob (float): Probability mass of the equal-tailed
                intervals of the fractions over draws. (default: 0.95)

        Returns:
            pd.DataFrame: One row per cell with ``cell``, ``n_worms``,
                ``n_observations``, the ``mean``, ``lower`` and ``upper`` (over
                draws) of ``below``, ``above`` and ``outside``
                (fractions of worms, e.g. ``outside_mean``), and
                ``extrapolated_fraction`` (fraction of the cell's observations
                whose ``log_x`` is outside the reference cell's range).

        Raises:
            ValueError: If ``prob`` or ``interval_prob`` is not strictly between 0
                and 1.
        """
        _check_prob(prob)
        _check_prob(interval_prob)
        rng = self._rng(random_seed)
        index = self._draw_index(n_draws)
        deviation = self.table["log_y"].to_numpy() - self._marginal_fitted_draws(
            self.table, genotype_effects=False, index=index
        )
        n_samples = len(deviation)

        worms = self.table.drop_duplicates("worm_id")
        worm_of_row = pd.Index(worms["worm_id"]).get_indexer(self.table["worm_id"])
        n_molts = np.bincount(worm_of_row, minlength=len(worms))
        averaging = csr_matrix(
            (
                1 / n_molts[worm_of_row],
                (np.arange(len(worm_of_row)), worm_of_row),
            ),
            shape=(len(worm_of_row), len(worms)),
        )
        worm_means = np.asarray(averaging.T @ deviation.T).T

        tail = (1 - prob) / 2
        tau = self._scale_draws("tau", index)[:, 0]
        sigma = self._scale_draws("sigma", index)[:, 0]
        molt_counts, count_of_worm = np.unique(n_molts, return_inverse=True)
        lower = np.empty((n_samples, len(molt_counts)))
        upper = np.empty_like(lower)
        if self._has("nu"):
            nu = self._draws("nu", index)
            chunk = max(1, 2_000_000 // (n_simulations * molt_counts.max()))
            for j, n in enumerate(molt_counts):
                for start in range(0, n_samples, chunk):
                    rows = slice(start, start + chunk)
                    size = (len(nu[rows]), n_simulations)
                    noise = rng.standard_t(
                        nu[rows, np.newaxis, np.newaxis], (*size, n)
                    ).mean(axis=2)
                    simulated = (
                        tau[rows, np.newaxis] * rng.standard_normal(size)
                        + sigma[rows, np.newaxis] * noise
                    )
                    lower[rows, j] = np.quantile(simulated, tail, axis=1)
                    upper[rows, j] = np.quantile(simulated, 1 - tail, axis=1)
        else:
            sd = np.sqrt(
                tau[:, np.newaxis] ** 2 + sigma[:, np.newaxis] ** 2 / molt_counts
            )
            upper = norm.ppf(1 - tail) * sd
            lower = -upper
        below = worm_means < lower[:, count_of_worm]
        above = worm_means > upper[:, count_of_worm]

        worm_cells = worms["condition_id"].map(self.coding.cell_of_condition).to_numpy()
        row_cells = self._row_cells().to_numpy()
        is_out = (self.support_flags() == "out").to_numpy()
        rows = []
        for cell in self.coding.cells:
            in_cell = worm_cells == cell
            row = {
                "cell": cell,
                "n_worms": int(in_cell.sum()),
                "n_observations": int((row_cells == cell).sum()),
            }
            for name, flags in (
                ("below", below),
                ("above", above),
                ("outside", below | above),
            ):
                row.update(
                    _summarize_draws(
                        flags[:, in_cell].mean(axis=1), interval_prob, name
                    )
                )
            row["extrapolated_fraction"] = float(is_out[row_cells == cell].mean())
            rows.append(row)
        return pd.DataFrame(rows)

    def posterior_predictive_residuals(
        self, n_draws: int | None = 200, random_seed: int | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulate marginal residuals of replicated data on the observed design, alongside the observed ones.

        For each selected posterior draw, every worm gets a new intercept from
        ``Normal(0, tau)`` of its cell and every observation a new residual from
        ``Normal(0, sigma)`` (or ``StudentT(nu, 0, sigma)``) of its cell, keeping
        the observed worms, molts and ``log_x``. The replicated marginal residual
        around the cell's own curve is then intercept plus residual; the
        observed one is ``log_y`` minus the same draw's fitted value without worm
        intercepts. Pass both to ``ppc_summary``.

        Parameters:
            n_draws (int or None): Number of evenly spaced posterior draws to use;
                ``None`` uses all. (default: 200)
            random_seed (int or None): Seed of the replicates; ``None`` reuses the
                sampling seed. (default: None)

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: ``(replicated, observed)``, each
                with columns ``draw`` (flattened posterior draw index), ``cell``,
                ``molt`` and ``residual`` (natural-log units), one row per draw ×
                observation in draw-major order.
        """
        rng = self._rng(random_seed)
        index = self._draw_index(n_draws)
        fitted = self._marginal_fitted_draws(
            self.table, genotype_effects=True, index=index
        )
        n_samples, n_rows = fitted.shape
        cell_index = pd.Index(self.coding.cells)
        row_cells = self._row_cells()
        obs_cell = cell_index.get_indexer(row_cells)
        worms = self.table.drop_duplicates("worm_id")
        worm_of_row = pd.Index(worms["worm_id"]).get_indexer(self.table["worm_id"])
        worm_cell = cell_index.get_indexer(
            worms["condition_id"].map(self.coding.cell_of_condition)
        )

        tau = self._scale_draws("tau", index)
        sigma = self._scale_draws("sigma", index)
        worm_effect = tau[:, worm_cell] * rng.standard_normal((n_samples, len(worms)))
        if self._has("nu"):
            noise = rng.standard_t(
                self._draws("nu", index)[:, np.newaxis], (n_samples, n_rows)
            )
        else:
            noise = rng.standard_normal((n_samples, n_rows))
        replicated = worm_effect[:, worm_of_row] + sigma[:, obs_cell] * noise
        observed = self.table["log_y"].to_numpy() - fitted

        draw_ids = np.arange(n_samples) if index is None else index
        design = {
            "draw": np.repeat(draw_ids, n_rows),
            "cell": np.tile(row_cells.to_numpy(), n_samples),
            "molt": np.tile(self.table["molt"].to_numpy(), n_samples),
        }
        return (
            pd.DataFrame({**design, "residual": replicated.ravel()}),
            pd.DataFrame({**design, "residual": observed.ravel()}),
        )

    def ppc_summary(
        self,
        replicated: pd.DataFrame | None = None,
        observed: pd.DataFrame | None = None,
        prob: float = 0.95,
        n_draws: int | None = 200,
        random_seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Compare the spread and tails of each cell's marginal residuals with posterior predictive replicates.

        Per cell and draw, three statistics are computed on the observed and the
        replicated residuals: the SD, the excess kurtosis (Fisher) and the
        fraction of residuals whose absolute value exceeds 3 robust SDs
        (``1.4826 * MAD``). The posterior predictive p-value of each is the
        fraction of draws whose replicated statistic is at least the observed
        one; values near 0 or 1 flag a feature of the data the model does not
        reproduce, e.g. heavier tails than a normal likelihood allows.

        Parameters:
            replicated (pd.DataFrame or None): Replicated residuals from
                ``posterior_predictive_residuals``; ``None`` computes them.
                (default: None)
            observed (pd.DataFrame or None): Observed residuals from the same
                call. (default: None)
            prob (float): Probability mass of the equal-tailed intervals.
                (default: 0.95)
            n_draws (int or None): Draws used when the residuals are computed
                here. (default: 200)
            random_seed (int or None): Seed used when the residuals are computed
                here. (default: None)

        Returns:
            pd.DataFrame: One row per cell × statistic with ``cell``,
                ``statistic`` (``"sd"``, ``"excess_kurtosis"`` or
                ``"outlier_fraction"``), ``observed`` (mean over draws),
                ``replicated_mean``, ``replicated_lower``, ``replicated_upper`` and
                ``p_value``.

        Raises:
            ValueError: If ``prob`` is not strictly between 0 and 1.
        """
        _check_prob(prob)
        if replicated is None or observed is None:
            replicated, observed = self.posterior_predictive_residuals(
                n_draws, random_seed
            )

        def by_draw(residuals, cell):
            in_cell = residuals[residuals["cell"] == cell].sort_values(
                "draw", kind="stable"
            )
            n_samples = in_cell["draw"].nunique()
            return in_cell["residual"].to_numpy().reshape(n_samples, -1)

        rows = []
        for cell in self.coding.cells:
            observed_statistics = _residual_statistics(by_draw(observed, cell))
            replicated_statistics = _residual_statistics(by_draw(replicated, cell))
            for name, values in observed_statistics.items():
                row = {"cell": cell, "statistic": name, "observed": values.mean()}
                row.update(
                    _summarize_draws(replicated_statistics[name], prob, "replicated")
                )
                row["p_value"] = float(np.mean(replicated_statistics[name] >= values))
                rows.append(row)
        return pd.DataFrame(rows)

    def molt_dispersion_tests(
        self,
        comparisons: (
            list[tuple[int | str | dict[str, str], int | str | dict[str, str]]] | None
        ) = None,
        family: Literal["all", "comparison"] = "all",
        n_draws: int | None = None,
    ) -> pd.DataFrame:
        """
        Test, molt by molt, whether two genotype cells differ in the spread of their residuals.

        Each test is a Brown–Forsythe test (``scipy.stats.levene`` with
        ``center="median"``) on the posterior mean marginal residuals around each
        cell's own curve (see ``residuals``), so differences in mean are removed
        and only spread is compared. The p-values are adjusted with Holm's
        step-down procedure (Holm 1979, Scand J Stat 6:65), which controls the
        family-wise error rate under any dependence between the tests. The
        per-molt tests of a comparison share worms and are therefore positively
        correlated, which makes Holm conservative here. Tests with fewer than 2
        observations in a cell are not run (NaN) and do not count toward the
        family.

        Parameters:
            comparisons (list[tuple] or None): Pairs ``(cell_a, cell_b)``, each a
                condition id, cell label or dict of factor levels; ``None``
                compares each non-reference cell with the reference cell.
                (default: None)
            family (str): ``"all"`` adjusts over every comparison × molt of the
                call; ``"comparison"`` adjusts over the molts within each
                comparison. (default: "all")
            n_draws (int or None): Number of evenly spaced posterior draws used for
                the residuals; ``None`` uses all. (default: None)

        Returns:
            pd.DataFrame: One row per comparison × molt with ``comparison``
                (``"cell_a vs cell_b"``), ``cell_a``, ``cell_b``, ``molt``,
                ``n_a``, ``n_b``, ``statistic``, ``p_raw``, ``p_holm`` and
                ``family_size``.

        Raises:
            ValueError: If ``family`` is unknown or a cell is not fitted.
        """
        if family not in ("all", "comparison"):
            raise ValueError(f"`family` must be 'all' or 'comparison', got {family!r}.")
        if comparisons is None:
            pairs = [(c, self.coding.reference_cell) for c in self.coding.cells[1:]]
        else:
            pairs = [
                tuple(self.coding.cell_label(self.coding.resolve_cell(c)) for c in pair)
                for pair in comparisons
            ]
        fitted = self._marginal_fitted_draws(
            self.table, genotype_effects=True, index=self._draw_index(n_draws)
        )
        residual = self.table["log_y"].to_numpy() - fitted.mean(axis=0)
        row_cells = self._row_cells().to_numpy()
        molts = self.table["molt"].to_numpy()

        rows = []
        for cell_a, cell_b in pairs:
            in_pair = np.isin(row_cells, [cell_a, cell_b])
            for molt in sorted(np.unique(molts[in_pair])):
                a = residual[(row_cells == cell_a) & (molts == molt)]
                b = residual[(row_cells == cell_b) & (molts == molt)]
                statistic = p_raw = np.nan
                if len(a) >= 2 and len(b) >= 2:
                    statistic, p_raw = levene(a, b, center="median")
                rows.append(
                    {
                        "comparison": f"{cell_a} vs {cell_b}",
                        "cell_a": cell_a,
                        "cell_b": cell_b,
                        "molt": molt,
                        "n_a": len(a),
                        "n_b": len(b),
                        "statistic": float(statistic),
                        "p_raw": float(p_raw),
                    }
                )
        tests = pd.DataFrame(
            rows,
            columns=[
                "comparison",
                "cell_a",
                "cell_b",
                "molt",
                "n_a",
                "n_b",
                "statistic",
                "p_raw",
            ],
        )
        tests["p_holm"] = np.nan
        tests["family_size"] = 0
        families = (
            [tests.index]
            if family == "all"
            else [group.index for _, group in tests.groupby("comparison", sort=False)]
        )
        for members in families:
            tested = members[tests.loc[members, "p_raw"].notna().to_numpy()]
            if len(tested):
                tests.loc[tested, "p_holm"] = multipletests(
                    tests.loc[tested, "p_raw"], method="holm"
                )[1]
            tests.loc[members, "family_size"] = len(tested)
        return tests


def _check_prob(prob: float) -> None:
    if not 0 < prob < 1:
        raise ValueError(f"`prob` must be between 0 and 1, got {prob}.")


def _summarize_draws(
    draws: np.ndarray, prob: float, prefix: str = "", percent: bool = False
) -> dict[str, float]:
    """
    Summarize draws with their mean and central ``prob`` interval, ignoring NaN.

    Parameters:
        draws (np.ndarray): Draws of shape ``(n_draws,)``.
        prob (float): Probability mass of the equal-tailed interval.
        prefix (str): Prefix of the keys, e.g. ``"offset"`` gives
            ``"offset_mean"``. (default: "")
        percent (bool): Also return the values converted with
            ``log_ratio_to_percentage`` under ``..._percent`` keys.
            (default: False)

    Returns:
        dict[str, float]: ``mean``, ``lower`` and ``upper`` (NaN if every draw is
            NaN), with the prefix and optional percent keys.
    """
    draws = np.asarray(draws, dtype=float)
    tail = (1 - prob) / 2
    if np.isnan(draws).all():
        values = (np.nan, np.nan, np.nan)
    else:
        values = (
            float(np.nanmean(draws)),
            float(np.nanquantile(draws, tail)),
            float(np.nanquantile(draws, 1 - tail)),
        )
    keys = [f"{prefix}_{k}" if prefix else k for k in ("mean", "lower", "upper")]
    summary = dict(zip(keys, values))
    if percent:
        summary.update(
            {f"{k}_percent": log_ratio_to_percentage(v) for k, v in zip(keys, values)}
        )
    return summary


def _sd(values: np.ndarray) -> float:
    """Sample SD with ``ddof=1``; NaN for fewer than 2 values."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else np.nan


def _residual_statistics(residuals: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute the SD, excess kurtosis and outlier fraction of each row of residuals.

    Parameters:
        residuals (np.ndarray): Residuals of shape ``(n_draws, n_residuals)``.

    Returns:
        dict[str, np.ndarray]: ``"sd"``, ``"excess_kurtosis"`` and
            ``"outlier_fraction"`` (fraction of absolute residuals above
            ``OUTLIER_ROBUST_SDS`` robust SDs), each of shape ``(n_draws,)``.
    """
    median = np.median(residuals, axis=1, keepdims=True)
    robust_sd = MAD_SCALE * np.median(np.abs(residuals - median), axis=1, keepdims=True)
    return {
        "sd": residuals.std(axis=1, ddof=1),
        "excess_kurtosis": kurtosis(residuals, axis=1),
        "outlier_fraction": np.mean(
            np.abs(residuals) > OUTLIER_ROBUST_SDS * robust_sd, axis=1
        ),
    }


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


# PER-EXPERIMENT FITS


def fit_per_experiment(
    table: pd.DataFrame,
    reference_shape: Literal["linear", "spline"],
    **fit_kwargs,
) -> dict[str, ProportionModelResult]:
    """
    Fit the proportion model separately to each experiment, to check that effects replicate.

    Every experiment is fitted with the same reference shape and, unless
    ``x_ref`` is given, the same centering point (the median reference ``log_x``
    of the whole table), so that ``beta`` means the same body size in every fit.
    Genotype cells absent from an experiment are dropped from its fit, with a
    warning naming them. A single-experiment fit has no experiment effect.

    Parameters:
        table (pd.DataFrame): Output of ``build_proportion_table``.
        reference_shape (str): ``"linear"`` or ``"spline"``; ``"auto"`` is
            refused because it could pick different shapes per experiment.
        **fit_kwargs: Keyword arguments of ``fit_proportion_model``, which must
            include ``random_seed`` and the genotype coding (``factors`` and
            ``reference``, or ``reference_condition``).

    Returns:
        dict[str, ProportionModelResult]: Fit of each experiment, keyed by its
            label, in sorted order.

    Raises:
        ValueError: If ``reference_shape`` is not ``"linear"`` or ``"spline"``,
            or an experiment has no reference-cell observations.
    """
    if reference_shape not in ("linear", "spline"):
        raise ValueError(
            f"`reference_shape` must be 'linear' or 'spline', got {reference_shape!r}; "
            "'auto' could choose different shapes for different experiments."
        )
    coding = build_genotype_coding(
        sorted(int(c) for c in table["condition_id"].unique()),
        fit_kwargs.get("factors"),
        fit_kwargs.get("reference"),
        fit_kwargs.get("reference_condition"),
    )
    row_cells = table["condition_id"].map(coding.cell_of_condition).to_numpy()
    is_reference = row_cells == coding.reference_cell
    if fit_kwargs.get("x_ref") is None:
        fit_kwargs["x_ref"] = float(np.median(table["log_x"].to_numpy()[is_reference]))

    results = {}
    for experiment in sorted(table["experiment"].unique()):
        in_experiment = (table["experiment"] == experiment).to_numpy()
        present = set(row_cells[in_experiment])
        if coding.reference_cell not in present:
            raise ValueError(
                f"Experiment {experiment} has no observations of the reference cell "
                f"{coding.reference_cell!r}."
            )
        missing = [cell for cell in coding.cells if cell not in present]
        if missing:
            warn(
                f"Experiment {experiment} has no observations of cells {missing}; "
                "they are dropped from its fit.",
                stacklevel=2,
            )
        results[experiment] = fit_proportion_model(
            table[in_experiment].reset_index(drop=True),
            reference_shape=reference_shape,
            **fit_kwargs,
        )
    return results


def compare_experiments(
    results: dict[str, ProportionModelResult], prob: float = 0.95
) -> pd.DataFrame:
    """
    Tabulate the genotype effects and variance ratios of per-experiment fits side by side.

    Parameters:
        results (dict[str, ProportionModelResult]): Fits keyed by experiment,
            e.g. from ``fit_per_experiment``.
        prob (float): Probability mass of the equal-tailed intervals.
            (default: 0.95)

    Returns:
        pd.DataFrame: One row per experiment × parameter with ``experiment``,
            ``parameter`` (``beta[...]``, ``gamma[...]`` and, for fits with group
            variances, ``tau_ratio[...]`` and ``sigma_ratio[...]`` relative to the
            reference cell), ``mean``, ``lower``, ``upper`` and the ``_percent``
            versions, which are NaN except for ``beta`` rows.

    Raises:
        ValueError: If ``prob`` is not strictly between 0 and 1.
    """
    _check_prob(prob)
    rows = []
    for experiment, result in results.items():

        def add(parameter, draws, percent=False):
            row = {"experiment": experiment, "parameter": parameter}
            row.update(_summarize_draws(draws, prob, percent=percent))
            rows.append(row)

        if result.coding.effect_names:
            beta, gamma = result._draws("beta"), result._draws("gamma")
            for k, effect in enumerate(result.coding.effect_names):
                add(f"beta[{effect}]", beta[:, k], percent=True)
            for k, effect in enumerate(result.coding.effect_names):
                add(f"gamma[{effect}]", gamma[:, k])
        if result.group_variances:
            for name in ("tau", "sigma"):
                draws = result._draws(name)
                for k, cell in enumerate(result.coding.cells[1:], start=1):
                    add(f"{name}_ratio[{cell}]", draws[:, k] / draws[:, 0])
    return pd.DataFrame(
        rows,
        columns=[
            "experiment",
            "parameter",
            "mean",
            "lower",
            "upper",
            "mean_percent",
            "lower_percent",
            "upper_percent",
        ],
    )
