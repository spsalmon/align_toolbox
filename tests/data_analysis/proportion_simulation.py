"""Simulated proportion data with known parameters for the proportion model tests."""

from functools import cache

import numpy as np

from align_toolbox.data_analysis import proportion_model as pmod

# log body size at molts 1-4: clusters ~0.1 wide, separated by gaps
MOLT_CENTERS = np.array([10.0, 11.0, 12.0, 13.0])
X0 = 11.5
INTERCEPT = 8.0
SLOPE = 0.5
EXPERIMENTS = {"/data/2025_exp_a": -0.04, "/data/2025_exp_b": 0.04}
CELLS = {
    0: {"n": 150, "beta": 0.0, "gamma": 0.0, "tau": 0.03, "sigma": 0.03, "nu": None},
    1: {"n": 100, "beta": 0.10, "gamma": 0.0, "tau": 0.03, "sigma": 0.03, "nu": None},
    2: {"n": 80, "beta": -0.08, "gamma": -0.04, "tau": 0.03, "sigma": 0.03, "nu": None},
    3: {"n": 60, "beta": 0.05, "gamma": 0.0, "tau": 0.05, "sigma": 0.06, "nu": 3},
}


def simulate_proportions(seed, cells=CELLS):
    """
    Conditions shaped like ``build_plotting_struct`` output, with hatch + 4 molts.

    Worms of each condition are split between two experiments whose point
    numbers both start at 0. ``log_y = INTERCEPT + SLOPE * (log_x - X0)`` plus the
    cell's ``beta + gamma * (log_x - X0)``, the experiment effect, a worm
    intercept ``Normal(0, tau)`` and a residual ``Normal(0, sigma)`` (or
    ``sigma * t(nu)``). A cell's optional ``x_shift`` moves its molt clusters along
    ``log_x``. Each worm survives to the next molt with probability 0.9.
    """
    rng = np.random.default_rng(seed)
    paths = list(EXPERIMENTS)
    conditions_struct = []
    for condition_id, cell in cells.items():
        n = cell["n"]
        experiment = np.array([paths[i % 2] for i in range(n)])
        point = np.array([i // 2 for i in range(n)])
        worm_size = rng.normal(0, 0.03, (n, 1))
        log_x = (
            MOLT_CENTERS
            + cell.get("x_shift", 0.0)
            + worm_size
            + rng.normal(0, 0.02, (n, 4))
        )
        x_centered = log_x - X0
        if cell["nu"] is None:
            noise = rng.standard_normal((n, 4))
        else:
            noise = rng.standard_t(cell["nu"], (n, 4))
        log_y = (
            INTERCEPT
            + SLOPE * x_centered
            + cell["beta"]
            + cell["gamma"] * x_centered
            + np.array([EXPERIMENTS[e] for e in experiment])[:, np.newaxis]
            + rng.normal(0, cell["tau"], (n, 1))
            + cell["sigma"] * noise
        )
        last_molt = 1 + np.minimum(rng.geometric(0.1, n), 4)
        dropped = np.arange(1, 5)[np.newaxis, :] >= last_molt[:, np.newaxis]
        x, y = np.exp(log_x), np.exp(log_y)
        x[dropped] = np.nan
        y[dropped] = np.nan
        hatch = np.full((n, 1), 1.0)
        conditions_struct.append(
            {
                "condition_id": condition_id,
                "point": point[:, np.newaxis],
                "experiment": experiment[:, np.newaxis],
                "volume_at_ecdysis": np.hstack([hatch, x]),
                "pharynx_volume_at_ecdysis": np.hstack([hatch, y]),
            }
        )
    return conditions_struct


# body volume ~ body length: log length at molts 1-4, with gaps between the clusters
ALLOMETRY_MOLT_CENTERS = np.log([350.0, 490.0, 690.0, 970.0])
ALLOMETRY_INTERCEPT = 12.0
ALLOMETRY_CELLS = {
    0: {"n": 100, "beta": 0.0, "length_shift": np.zeros(4)},
    # molts 1-3 fall halfway into the reference cell's gaps
    1: {"n": 100, "beta": 0.10, "length_shift": np.array([0.17, 0.17, 0.17, 0.0])},
}
# molts 2-4 of the mutant fall halfway into the gaps below the reference clusters;
# a line then mostly biases its gamma rather than its beta
ALLOMETRY_LOW_GAP_CELLS = {
    0: ALLOMETRY_CELLS[0],
    1: {"n": 100, "beta": 0.10, "length_shift": np.array([0.0, -0.17, -0.17, -0.17])},
}


def allometric_log_volume(log_length, exponent_range):
    """
    Log volume whose local exponent rises linearly in ``log_length`` across the molts.

    The exponent goes from ``exponent_range[0]`` at the first molt center to
    ``exponent_range[1]`` at the last, so the curve is quadratic in ``log_length``.
    """
    start, end = exponent_range
    span = ALLOMETRY_MOLT_CENTERS[-1] - ALLOMETRY_MOLT_CENTERS[0]
    t = log_length - ALLOMETRY_MOLT_CENTERS[0]
    return ALLOMETRY_INTERCEPT + start * t + (end - start) / (2 * span) * t**2


def simulate_allometry(seed, exponent_range=(2.4, 3.3), cells=ALLOMETRY_CELLS):
    """
    Body length (x) and volume (y) per molt, shaped like ``build_plotting_struct`` output.

    Each cell's ``log_y`` is ``allometric_log_volume(log_x) + beta`` plus an
    experiment effect, a worm intercept ``Normal(0, 0.03)`` and a residual
    ``Normal(0, 0.03)``; its molt clusters are shifted by ``length_shift``. Each
    worm survives to the next molt with probability 0.9.
    """
    rng = np.random.default_rng(seed)
    paths = list(EXPERIMENTS)
    conditions_struct = []
    for condition_id, cell in cells.items():
        n = cell["n"]
        experiment = np.array([paths[i % 2] for i in range(n)])
        point = np.array([i // 2 for i in range(n)])
        log_x = (
            ALLOMETRY_MOLT_CENTERS
            + cell["length_shift"]
            + rng.normal(0, 0.03, (n, 1))
            + rng.normal(0, 0.02, (n, 4))
        )
        log_y = (
            allometric_log_volume(log_x, exponent_range)
            + cell["beta"]
            + np.array([EXPERIMENTS[e] for e in experiment])[:, np.newaxis]
            + rng.normal(0, 0.03, (n, 1))
            + rng.normal(0, 0.03, (n, 4))
        )
        last_molt = 1 + np.minimum(rng.geometric(0.1, n), 4)
        dropped = np.arange(1, 5)[np.newaxis, :] >= last_molt[:, np.newaxis]
        x, y = np.exp(log_x), np.exp(log_y)
        x[dropped] = np.nan
        y[dropped] = np.nan
        hatch = np.full((n, 1), 1.0)
        conditions_struct.append(
            {
                "condition_id": condition_id,
                "point": point[:, np.newaxis],
                "experiment": experiment[:, np.newaxis],
                "length_at_ecdysis": np.hstack([hatch, x]),
                "volume_at_ecdysis": np.hstack([hatch, y]),
            }
        )
    return conditions_struct


# log body size grows by SERIES_GROWTH over development, from SERIES_START
SERIES_START = 10.0
SERIES_GROWTH = 3.0
SERIES_CELLS = {
    0: {"n": 40, "beta": 0.0},
    1: {"n": 40, "beta": 0.10},
}
# frames every 10 minutes, development from about 2 h to about 46 h
SERIES_HOURS = np.arange(0.0, 50.0, 1 / 6)
STAGE_HOURS = np.array([12.0, 9.0, 10.0, 13.0])


def development_fraction(time, ecdysis):
    """Fraction of development (L1–L4) at ``time``, linear within each stage, 0 before hatch and 1 after molt 4."""
    return np.interp(time, ecdysis, np.linspace(0, 1, 5))


def simulate_series_proportions(seed, cells=SERIES_CELLS):
    """
    Raw time series of shape ``(n_worms, n_frames)``, shaped like ``build_plotting_struct`` output.

    Each worm hatches at ``Uniform(1, 3)`` h and its stages last ``STAGE_HOURS``
    scaled by ``1 + Normal(0, 0.05)``. ``log_x`` rises linearly with the
    development fraction, plus a worm size ``Normal(0, 0.03)`` and a frame
    residual ``Normal(0, 0.02)``; ``log_y = INTERCEPT + SLOPE * (log_x - X0) +
    beta`` on the noiseless ``log_x``, plus an experiment effect, a worm
    intercept ``Normal(0, 0.03)`` and a frame residual ``Normal(0, 0.02)``.
    Worms of each condition alternate between two experiments, and every frame
    passes QC.
    """
    rng = np.random.default_rng(seed)
    paths = list(EXPERIMENTS)
    n_frames = len(SERIES_HOURS)
    conditions_struct = []
    for condition_id, cell in cells.items():
        n = cell["n"]
        experiment = np.array([paths[i % 2] for i in range(n)])
        durations = STAGE_HOURS * (1 + rng.normal(0, 0.05, (n, 4)))
        ecdysis = np.cumsum(np.hstack([rng.uniform(1, 3, (n, 1)), durations]), axis=1)
        fraction = np.array(
            [development_fraction(SERIES_HOURS, worm) for worm in ecdysis]
        )
        log_x = SERIES_START + SERIES_GROWTH * fraction + rng.normal(0, 0.03, (n, 1))
        log_y = (
            INTERCEPT
            + SLOPE * (log_x - X0)
            + cell["beta"]
            + np.array([EXPERIMENTS[e] for e in experiment])[:, np.newaxis]
            + rng.normal(0, 0.03, (n, 1))
            + rng.normal(0, 0.02, (n, n_frames))
        )
        log_x += rng.normal(0, 0.02, (n, n_frames))
        conditions_struct.append(
            {
                "condition_id": condition_id,
                "point": np.array([i // 2 for i in range(n)])[:, np.newaxis],
                "experiment": experiment[:, np.newaxis],
                "time": np.tile(np.arange(n_frames, dtype=float), (n, 1)),
                "experiment_time_hours": np.tile(SERIES_HOURS, (n, 1)),
                "ecdysis_index": ecdysis * 6,
                "ecdysis_experiment_time_hours": ecdysis,
                "body_qc": np.full((n, n_frames), "worm"),
                "volume": np.exp(log_x),
                "pharynx_volume": np.exp(log_y),
            }
        )
    return conditions_struct


# several timepoints per larval stage, molts included
FIT_PERCENTAGES = (0.1, 0.2, 0.25, 0.4, 0.6, 0.75, 0.9, 1.0)


@cache
def fit_series_proportions():
    """
    Linear, group-variance fit at ``FIT_PERCENTAGES`` of ``simulate_series_proportions``.

    Sampled briefly and shared by every test of the session that uses it.
    """
    table = pmod.build_proportion_table(
        simulate_series_proportions(seed=3),
        "volume",
        "pharynx_volume",
        list(SERIES_CELLS),
        percentages=list(FIT_PERCENTAGES),
    )
    return pmod.fit_proportion_model(
        table,
        reference_condition=0,
        group_variances=True,
        reference_shape="linear",
        draws=300,
        tune=300,
        chains=2,
        random_seed=5,
    )
