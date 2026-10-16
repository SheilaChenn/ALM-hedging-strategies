from __future__ import annotations

import numpy as np
import pytest

from alm_hedging.market import MarketState
from alm_hedging.pricing import black_scholes_price, simulate_gbm_paths


def _mc_call(market: MarketState, n_paths: int, seed: int) -> float:
    _, paths = simulate_gbm_paths(
        market, horizon=1.0, n_paths=n_paths, steps_per_year=1, seed=seed
    )
    payoff = np.maximum(paths[:, -1] - 100.0, 0.0)
    return float(market.rate_curve.discount(1.0) * payoff.mean())


def test_monte_carlo_converges_to_black_scholes(market: MarketState) -> None:
    analytic = black_scholes_price(100.0, 100.0, 1.0, 0.03, 0.01, 0.20)
    seeds = range(4)
    coarse_error = np.mean(
        [abs(_mc_call(market, 2_000, seed) - analytic) for seed in seeds]
    )
    fine_values = np.array(
        [_mc_call(market, 40_000, seed) for seed in seeds], dtype=float
    )
    fine_error = np.mean(np.abs(fine_values - analytic))

    assert fine_values.mean() == pytest.approx(analytic, abs=0.12)
    assert fine_error < coarse_error


def test_antithetic_paths_are_seed_reproducible(market: MarketState) -> None:
    times_a, paths_a = simulate_gbm_paths(market, 1.0, 100, 12, seed=73)
    times_b, paths_b = simulate_gbm_paths(market, 1.0, 100, 12, seed=73)
    np.testing.assert_array_equal(times_a, times_b)
    np.testing.assert_array_equal(paths_a, paths_b)
