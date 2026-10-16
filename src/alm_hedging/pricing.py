"""Analytic option pricing, static replication, and seeded Monte Carlo engines."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.stats import norm

from .contracts import (
    Cohort,
    FIAAnnualPointToPoint,
    IULAccountValue,
    LiabilityContract,
    RILABufferedSegment,
    RILAFlooredSegment,
)
from .market import MarketState

OptionType = Literal["call", "put"]


def black_scholes_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    volatility: float,
    option_type: OptionType = "call",
) -> float:
    """Price a European option under Black-Scholes with continuous carry."""
    if spot <= 0.0 or strike <= 0.0 or maturity < 0.0 or volatility < 0.0:
        raise ValueError(
            "spot/strike must be positive; maturity/volatility non-negative"
        )
    sign = 1.0 if option_type == "call" else -1.0
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if maturity == 0.0:
        return float(max(sign * (spot - strike), 0.0))
    discount_r = np.exp(-rate * maturity)
    discount_q = np.exp(-dividend * maturity)
    if volatility == 0.0:
        return float(max(sign * (spot * discount_q - strike * discount_r), 0.0))
    root_t = np.sqrt(maturity)
    d1 = (
        np.log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * maturity
    ) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    return float(
        sign
        * (
            spot * discount_q * norm.cdf(sign * d1)
            - strike * discount_r * norm.cdf(sign * d2)
        )
    )


def market_option_price(
    market: MarketState,
    strike: float,
    maturity: float,
    option_type: OptionType = "call",
) -> float:
    """Price an option from an interpolated market state."""
    tau = max(maturity - market.valuation_time, 0.0)
    if tau == 0.0:
        sign = 1.0 if option_type == "call" else -1.0
        return float(max(sign * (market.spot - strike), 0.0))
    rate = float(market.rate_curve.zero_rate(tau))
    dividend = float(market.dividend_curve.zero_rate(tau))
    return black_scholes_price(
        market.spot,
        strike,
        tau,
        rate,
        dividend,
        market.volatility(strike, tau),
        option_type,
    )


@dataclass(frozen=True)
class ReplicationLeg:
    """One component of a static payoff replication."""

    kind: Literal["cash", "underlying", "call", "put"]
    quantity: float
    strike: float | None = None


def piecewise_linear_replication(
    strikes: Sequence[float],
    values: Sequence[float],
    *,
    left_slope: float = 0.0,
    right_slope: float = 0.0,
) -> tuple[ReplicationLeg, ...]:
    """Replicate a continuous piecewise-linear terminal payoff.

    Slopes outside the supplied knot range default to zero.  The result comprises
    cash, underlying, and calls at payoff knots.
    """
    x = np.asarray(strikes, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2 or y.shape != x.shape or np.any(np.diff(x) <= 0):
        raise ValueError("strikes/values require matching increasing grids")
    slopes = np.diff(y) / np.diff(x)
    legs: list[ReplicationLeg] = [
        ReplicationLeg("cash", float(y[0] - left_slope * x[0])),
        ReplicationLeg("underlying", float(left_slope)),
        ReplicationLeg("call", float(slopes[0] - left_slope), float(x[0])),
    ]
    for i in range(1, x.size):
        next_slope = slopes[i] if i < slopes.size else right_slope
        change = next_slope - slopes[i - 1]
        if change != 0.0:
            legs.append(ReplicationLeg("call", float(change), float(x[i])))
    return tuple(legs)


def replication_value(
    legs: Sequence[ReplicationLeg], market: MarketState, maturity: float
) -> float:
    """Present value a static replication portfolio."""
    tau = max(maturity - market.valuation_time, 0.0)
    value = 0.0
    for leg in legs:
        if leg.kind == "cash":
            value += leg.quantity * float(market.rate_curve.discount(tau))
        elif leg.kind == "underlying":
            value += (
                leg.quantity * market.spot * float(market.dividend_curve.discount(tau))
            )
        else:
            assert leg.strike is not None
            value += leg.quantity * market_option_price(
                market, leg.strike, maturity, leg.kind
            )
    return float(value)


def analytic_liability_value(
    contract: LiabilityContract | Cohort, market: MarketState
) -> float:
    """Analytically value terminal piecewise-linear or one-year FIA liabilities."""
    multiplier = contract.policy_count if isinstance(contract, Cohort) else 1
    liability = contract.contract if isinstance(contract, Cohort) else contract
    n = liability.notional

    if isinstance(liability, RILABufferedSegment):
        s0 = liability.index_start
        knots = np.unique(
            np.array(
                [
                    max(s0 * (1.0 + liability.floor - liability.buffer), 1e-8),
                    s0 * (1.0 - liability.buffer),
                    s0,
                    s0 * (1.0 + liability.cap / max(liability.participation, 1e-15)),
                ]
            )
        )
        returns = knots / s0 - 1.0
        values = n * (1.0 + liability.credited_return(returns))
    elif isinstance(liability, RILAFlooredSegment):
        s0 = liability.index_start
        p = max(liability.participation, 1e-15)
        knots = np.unique(
            np.array(
                [
                    max(s0 * (1.0 + liability.floor / p), 1e-8),
                    s0,
                    s0 * (1.0 + liability.cap / p),
                ]
            )
        )
        values = n * (1.0 + liability.credited_return(knots / s0 - 1.0))
    elif isinstance(liability, FIAAnnualPointToPoint) and liability.maturity <= 1.0:
        s0 = liability.index_start
        p = max(liability.participation, 1e-15)
        lower = (liability.annual_floor + liability.spread) / p
        upper = (liability.annual_cap + liability.spread) / p
        knots = np.unique(np.array([max(s0 * (1.0 + lower), 1e-8), s0 * (1.0 + upper)]))
        credit = np.clip(
            liability.participation * (knots / s0 - 1.0) - liability.spread,
            liability.annual_floor,
            liability.annual_cap,
        )
        values = n * (1.0 + credit)
    else:
        raise TypeError("contract requires Monte Carlo valuation")
    return multiplier * replication_value(
        piecewise_linear_replication(knots, values), market, liability.maturity
    )


def simulate_gbm_paths(
    market: MarketState,
    horizon: float,
    n_paths: int = 20_000,
    steps_per_year: int = 12,
    seed: int = 0,
    *,
    antithetic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate deterministic-seed GBM paths consistent with the market term structure.

    Each step uses the forward rate, forward dividend yield, and forward
    variance implied by the zero curves and the at-the-money volatility term
    structure, so that ``E[S_T]`` equals the market forward and the terminal
    log-variance equals ``sigma_atm(T)^2 T`` for every grid maturity.  Using
    spot zero rates and spot implied volatilities as instantaneous parameters
    (the previous convention) biased Monte Carlo values against the analytic
    benchmark whenever the curves were not flat.
    """
    if horizon <= 0.0 or n_paths <= 0 or steps_per_year <= 0:
        raise ValueError("horizon, n_paths, and steps_per_year must be positive")
    n_steps = max(1, int(np.ceil(horizon * steps_per_year)))
    times = np.linspace(0.0, horizon, n_steps + 1)
    rng = np.random.default_rng(seed)
    n_base = (n_paths + 1) // 2 if antithetic else n_paths
    z = rng.standard_normal((n_base, n_steps))
    if antithetic:
        z = np.vstack((z, -z))[:n_paths]
    grid = np.maximum(times, 1e-12)
    rate_integral = np.asarray(market.rate_curve.zero_rate(grid)) * grid
    dividend_integral = np.asarray(market.dividend_curve.zero_rate(grid)) * grid
    atm_vol = np.asarray(market.vol_surface.volatility(grid, 0.0))
    total_variance = atm_vol**2 * grid
    # A non-increasing total variance is a calendar arbitrage in the input
    # surface; the step is then floored at a tiny positive variance.
    step_variance = np.maximum(np.diff(total_variance), 1e-16)
    drift = np.diff(rate_integral) - np.diff(dividend_integral) - 0.5 * step_variance
    log_increments = drift[None, :] + np.sqrt(step_variance)[None, :] * z
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = market.spot
    paths[:, 1:] = market.spot * np.exp(np.cumsum(log_increments, axis=1))
    return times, paths


def monte_carlo_value(
    contract: LiabilityContract | Cohort,
    market: MarketState,
    *,
    n_paths: int = 20_000,
    steps_per_year: int = 12,
    seed: int = 0,
) -> float:
    """Value a path-dependent liability with deterministic seeded Monte Carlo."""
    multiplier = contract.policy_count if isinstance(contract, Cohort) else 1
    liability = contract.contract if isinstance(contract, Cohort) else contract
    if isinstance(contract, Cohort) and isinstance(liability, IULAccountValue):
        liability = replace(
            liability,
            mortality_rates=tuple(
                min(rate * contract.mortality_multiplier, 1.0)
                for rate in liability.mortality_rates
            ),
            lapse_rates=tuple(
                min(rate * contract.lapse_multiplier, 1.0)
                for rate in liability.lapse_rates
            ),
        )
    horizon = max(liability.maturity - market.valuation_time, 1e-12)
    times, paths = simulate_gbm_paths(market, horizon, n_paths, steps_per_year, seed)
    if isinstance(liability, IULAccountValue):
        uniforms = np.random.default_rng(seed + 1).random((n_paths, times.size - 1))
        payoff, payment_times = liability.cashflows(paths, times, uniforms)
        discounted_payoff = payoff * np.asarray(
            market.rate_curve.discount(payment_times)
        )
    else:
        payoff = liability.payoff(paths, times)
        discounted_payoff = payoff * float(market.rate_curve.discount(horizon))
    return float(multiplier * np.mean(discounted_payoff))


def portfolio_value(
    liabilities: Sequence[LiabilityContract | Cohort],
    market: MarketState,
    *,
    method: Literal["auto", "analytic", "monte_carlo"] = "auto",
    n_paths: int = 20_000,
    seed: int = 0,
) -> float:
    """Aggregate liability values using analytic pricing where available."""
    total = 0.0
    for i, liability in enumerate(liabilities):
        if method != "monte_carlo":
            try:
                total += analytic_liability_value(liability, market)
                continue
            except TypeError:
                if method == "analytic":
                    raise
        total += monte_carlo_value(
            liability, market, n_paths=n_paths, seed=seed + 10_007 * i
        )
    return float(total)


Pricer = Callable[[MarketState], float]
