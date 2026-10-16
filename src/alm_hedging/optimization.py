"""Constrained hedge optimization across futures, options, and swaps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import minimize

InstrumentKind = Literal["future", "option", "swap"]


@dataclass(frozen=True)
class HedgeInstrument:
    """Tradable hedge represented by unit risk exposures and implementation data."""

    name: str
    kind: InstrumentKind
    tenor: float
    price: float
    exposures: Mapping[str, float]
    premium: float = 0.0
    max_position: float = np.inf
    max_trade: float = np.inf
    lot_size: float = 1.0
    transaction_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("future", "option", "swap"):
            raise ValueError("kind must be future, option, or swap")
        if self.tenor <= 0.0 or self.price < 0.0:
            raise ValueError("tenor must be positive and price non-negative")
        if self.max_position <= 0.0 or self.max_trade <= 0.0:
            raise ValueError("position and trade limits must be positive")
        if self.lot_size <= 0.0 or self.transaction_cost < 0.0:
            raise ValueError("lot size must be positive and cost non-negative")
        if self.premium < 0.0:
            raise ValueError("premium must be non-negative")
        if not all(np.isfinite(float(x)) for x in self.exposures.values()):
            raise ValueError("exposures must be finite")


@dataclass(frozen=True)
class HedgeConstraints:
    """Portfolio implementation constraints."""

    premium_budget: float = np.inf
    turnover_budget: float = np.inf
    min_tenor: float = 0.0
    max_tenor: float = np.inf
    max_concentration: float = 1.0

    def __post_init__(self) -> None:
        if self.premium_budget < 0.0 or self.turnover_budget < 0.0:
            raise ValueError("budgets must be non-negative")
        if self.min_tenor < 0.0 or self.max_tenor < self.min_tenor:
            raise ValueError("invalid tenor range")
        if not 0.0 < self.max_concentration <= 1.0:
            raise ValueError("max_concentration must lie in (0,1]")


@dataclass(frozen=True)
class HedgeSolution:
    """Optimized positions and post-trade risk diagnostics."""

    positions: Mapping[str, float]
    trades: Mapping[str, float]
    residual_exposures: Mapping[str, float]
    objective: float
    premium_used: float
    turnover_used: float
    success: bool
    message: str
    constraint_slacks: Mapping[str, float] = field(default_factory=dict)
    stress_cvar: float = 0.0


def optimize_hedge(
    target_exposures: Mapping[str, float],
    instruments: Sequence[HedgeInstrument],
    *,
    current_positions: Mapping[str, float] | None = None,
    exposure_weights: Mapping[str, float] | None = None,
    constraints: HedgeConstraints = HedgeConstraints(),
    cost_penalty: float = 1e-6,
    scenario_liability_losses: Sequence[float] | None = None,
    instrument_scenario_pnls: Mapping[str, Sequence[float]] | None = None,
    cvar_weight: float = 0.0,
    cvar_alpha: float = 0.95,
) -> HedgeSolution:
    """Minimize weighted residual risk subject to trading constraints.

    ``target_exposures`` are the unhedged liability/asset exposures.  Instrument
    unit exposures are added to them, so a successful hedge drives their sum
    toward zero.  The continuous SLSQP solution is rounded to instrument lots and
    improved with a deterministic one-lot coordinate search.
    """
    if not instruments:
        raise ValueError("at least one hedge instrument is required")
    if cost_penalty < 0.0 or cvar_weight < 0.0:
        raise ValueError("cost and CVaR penalties must be non-negative")
    if not 0.0 < cvar_alpha < 1.0:
        raise ValueError("cvar_alpha must lie in (0, 1)")
    names = [item.name for item in instruments]
    if len(set(names)) != len(names):
        raise ValueError("instrument names must be unique")
    risk_names = tuple(target_exposures)
    if not risk_names:
        raise ValueError("target_exposures cannot be empty")

    target = np.array([float(target_exposures[k]) for k in risk_names])
    weights = np.array(
        [float((exposure_weights or {}).get(k, 1.0)) for k in risk_names]
    )
    if np.any(weights <= 0.0) or not np.all(np.isfinite(target)):
        raise ValueError("targets must be finite and weights positive")
    matrix = np.array(
        [
            [float(item.exposures.get(k, 0.0)) for item in instruments]
            for k in risk_names
        ]
    )
    current = np.array(
        [float((current_positions or {}).get(item.name, 0.0)) for item in instruments]
    )
    prices = np.array([item.price for item in instruments])
    premiums = np.array([item.premium for item in instruments])
    costs = np.array([item.transaction_cost for item in instruments])
    lots = np.array([item.lot_size for item in instruments])
    scenario_losses = np.asarray(
        scenario_liability_losses if scenario_liability_losses is not None else (),
        dtype=float,
    )
    if scenario_losses.ndim != 1 or not np.all(np.isfinite(scenario_losses)):
        raise ValueError("scenario_liability_losses must be finite and one-dimensional")
    scenario_pnls = np.zeros((scenario_losses.size, len(instruments)))
    if scenario_losses.size:
        supplied = instrument_scenario_pnls or {}
        for i, item in enumerate(instruments):
            values = np.asarray(supplied.get(item.name, ()), dtype=float)
            if values.shape != scenario_losses.shape or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"scenario P&L for {item.name!r} must match liability losses"
                )
            scenario_pnls[:, i] = values

    bounds: list[tuple[float, float]] = []
    for i, item in enumerate(instruments):
        eligible = constraints.min_tenor <= item.tenor <= constraints.max_tenor
        if not eligible:
            bounds.append((current[i], current[i]))
            continue
        lower = max(-item.max_position, current[i] - item.max_trade)
        upper = min(item.max_position, current[i] + item.max_trade)
        bounds.append((lower, upper))

    def premium_used(q: np.ndarray) -> float:
        return float(np.dot(np.abs(q), premiums))

    def turnover_used(q: np.ndarray) -> float:
        return float(np.dot(np.abs(q - current), prices))

    def gross_notionals(q: np.ndarray) -> np.ndarray:
        return np.abs(q) * np.maximum(prices, 1e-12)

    def concentration_slack(q: np.ndarray) -> float:
        notionals = gross_notionals(q)
        gross = float(np.sum(notionals))
        if gross <= 1e-12:
            return 0.0
        return float(constraints.max_concentration * gross - np.max(notionals))

    def objective(q: np.ndarray) -> float:
        residual = target + matrix @ q
        risk = np.dot(weights * residual, weights * residual)
        implementation = np.dot(np.abs(q - current), costs)
        return float(
            risk + cost_penalty * implementation + cvar_weight * stress_cvar(q) ** 2
        )

    def stress_cvar(q: np.ndarray) -> float:
        if not scenario_losses.size:
            return 0.0
        net_losses = scenario_losses - scenario_pnls @ q
        tail_count = max(1, int(np.ceil((1.0 - cvar_alpha) * net_losses.size)))
        return float(np.mean(np.sort(net_losses)[-tail_count:]))

    scipy_constraints = []
    if np.isfinite(constraints.premium_budget):
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda q: constraints.premium_budget - premium_used(q),
            }
        )
    if np.isfinite(constraints.turnover_budget):
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda q: constraints.turnover_budget - turnover_used(q),
            }
        )
    if constraints.max_concentration < 1.0:
        scipy_constraints.append({"type": "ineq", "fun": concentration_slack})

    x0 = np.array([np.clip(current[i], *bounds[i]) for i in range(len(instruments))])
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=scipy_constraints,
        options={"ftol": 1e-12, "maxiter": 2_000, "disp": False},
    )

    rounded = np.round(np.asarray(result.x) / lots) * lots
    rounded = np.array(
        [np.clip(rounded[i], bounds[i][0], bounds[i][1]) for i in range(len(rounded))]
    )

    def slacks(q: np.ndarray) -> dict[str, float]:
        values = {
            "premium": constraints.premium_budget - premium_used(q),
            "turnover": constraints.turnover_budget - turnover_used(q),
        }
        if constraints.max_concentration < 1.0:
            values["concentration"] = concentration_slack(q)
        for i, item in enumerate(instruments):
            values[f"position:{item.name}"] = min(
                q[i] - bounds[i][0], bounds[i][1] - q[i]
            )
        return values

    def feasible(q: np.ndarray, tolerance: float = 1e-8) -> bool:
        return bool(min(slacks(q).values(), default=0.0) >= -tolerance)

    # Rounding can cross a budget. Move each trade one lot toward current until
    # feasible, then use a deterministic local search to recover hedge quality.
    for _ in range(10_000):
        if feasible(rounded):
            break
        trade = rounded - current
        candidates = np.flatnonzero(np.abs(trade) > 1e-12)
        if candidates.size == 0:
            break
        i = int(candidates[np.argmax(np.abs(trade[candidates]) * prices[candidates])])
        rounded[i] -= np.sign(trade[i]) * lots[i]
        rounded[i] = np.clip(rounded[i], *bounds[i])

    best = rounded
    best_objective = objective(best)
    for _ in range(10):
        improved = False
        for i in range(len(instruments)):
            for direction in (-1.0, 1.0):
                candidate = best.copy()
                candidate[i] = np.clip(candidate[i] + direction * lots[i], *bounds[i])
                if (
                    feasible(candidate)
                    and objective(candidate) + 1e-14 < best_objective
                ):
                    best, best_objective = candidate, objective(candidate)
                    improved = True
        if not improved:
            break

    residual = target + matrix @ best
    final_slacks = {name: float(value) for name, value in slacks(best).items()}
    success = feasible(best)
    message = (
        "optimal rounded feasible hedge"
        if success
        else f"no feasible rounded hedge: {result.message}"
    )
    return HedgeSolution(
        positions=dict(zip(names, map(float, best), strict=False)),
        trades=dict(zip(names, map(float, best - current), strict=False)),
        residual_exposures=dict(zip(risk_names, map(float, residual), strict=False)),
        objective=float(best_objective),
        premium_used=premium_used(best),
        turnover_used=turnover_used(best),
        success=success,
        message=message,
        constraint_slacks=final_slacks,
        stress_cvar=stress_cvar(best),
    )
