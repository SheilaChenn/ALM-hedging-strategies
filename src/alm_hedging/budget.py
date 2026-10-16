"""Option-budget solvers for participation, cap, and strike design."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .market import MarketState
from .pricing import market_option_price


@dataclass(frozen=True)
class BudgetSolution:
    """Result of a one-dimensional option-budget solve."""

    parameter: float
    premium: float
    target_budget: float
    residual: float
    converged: bool


def solve_budget_parameter(
    premium_function: Callable[[float], float],
    target_budget: float,
    lower: float,
    upper: float,
    *,
    tolerance: float = 1e-10,
) -> BudgetSolution:
    """Solve ``premium_function(parameter) == target_budget`` deterministically.

    Brent root finding is used when endpoints bracket the target.  Otherwise
    bounded scalar minimization returns the closest feasible parameter and marks
    the result unconverged when the residual exceeds tolerance.
    """
    if target_budget < 0.0 or lower >= upper:
        raise ValueError("budget must be non-negative and lower < upper")

    def residual(x: float) -> float:
        value = float(premium_function(x))
        if not np.isfinite(value):
            raise ValueError("premium function returned a non-finite value")
        return value - target_budget

    f_lower, f_upper = residual(lower), residual(upper)
    if f_lower == 0.0:
        root = lower
    elif f_upper == 0.0:
        root = upper
    elif f_lower * f_upper < 0.0:
        root = float(brentq(residual, lower, upper, xtol=tolerance, rtol=1e-14))
    else:
        result = minimize_scalar(
            lambda x: residual(float(x)) ** 2,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": tolerance},
        )
        root = float(result.x)
    premium = float(premium_function(root))
    error = premium - target_budget
    return BudgetSolution(root, premium, target_budget, error, abs(error) <= tolerance)


def solve_call_spread_cap(
    market: MarketState,
    maturity: float,
    budget: float,
    *,
    participation: float = 1.0,
    notional: float = 1.0,
    minimum_cap: float = 0.0,
    maximum_cap: float = 5.0,
) -> BudgetSolution:
    """Solve the return cap funded by an ATM call-spread budget."""
    if participation <= 0.0 or notional < 0.0:
        raise ValueError("participation must be positive and notional non-negative")
    atm = market.spot

    def premium(cap: float) -> float:
        upper = atm * (1.0 + cap / participation)
        spread = market_option_price(market, atm, maturity) - market_option_price(
            market, upper, maturity
        )
        return notional * participation * spread / atm

    return solve_budget_parameter(premium, budget, minimum_cap, maximum_cap)


def solve_participation_rate(
    market: MarketState,
    maturity: float,
    budget: float,
    *,
    cap: float | None = None,
    notional: float = 1.0,
    minimum_participation: float = 1e-6,
    maximum_participation: float = 5.0,
) -> BudgetSolution:
    """Solve participation for an ATM call or capped call spread."""
    if notional < 0.0:
        raise ValueError("notional must be non-negative")
    atm = market.spot

    def premium(participation: float) -> float:
        value = participation * market_option_price(market, atm, maturity)
        if cap is not None:
            upper = atm * (1.0 + cap / participation)
            value -= participation * market_option_price(market, upper, maturity)
        return notional * value / atm

    return solve_budget_parameter(
        premium,
        budget,
        minimum_participation,
        maximum_participation,
    )


def solve_option_budget(
    market: MarketState,
    maturity: float,
    budget: float,
    solve_for: Literal["cap", "participation"],
    **kwargs: float,
) -> BudgetSolution:
    """Dispatch to the standard call-spread budget solver."""
    if solve_for == "cap":
        return solve_call_spread_cap(market, maturity, budget, **kwargs)
    if solve_for == "participation":
        return solve_participation_rate(market, maturity, budget, **kwargs)
    raise ValueError("solve_for must be 'cap' or 'participation'")
