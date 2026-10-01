"""Simulated proportion data with known parameters for the proportion model tests."""

import numpy as np

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
    ``sigma * t(nu)``). Each worm survives to the next molt with probability 0.9.
    """
    rng = np.random.default_rng(seed)
    paths = list(EXPERIMENTS)
    conditions_struct = []
    for condition_id, cell in cells.items():
        n = cell["n"]
        experiment = np.array([paths[i % 2] for i in range(n)])
        point = np.array([i // 2 for i in range(n)])
        worm_size = rng.normal(0, 0.03, (n, 1))
        log_x = MOLT_CENTERS + worm_size + rng.normal(0, 0.02, (n, 4))
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
