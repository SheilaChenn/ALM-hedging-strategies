"""Validated liability contracts and policy cohorts for ALM applications."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class LiabilityContract(Protocol):
    """Structural interface shared by liabilities accepted by pricing engines."""

    notional: float
    maturity: float

    def payoff(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Return one payoff for each simulated path."""


def _validate_common(notional: float, maturity: float) -> None:
    if not np.isfinite(notional) or notional < 0.0:
        raise ValueError("notional must be finite and non-negative")
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive")


def _terminal_return(paths: np.ndarray) -> np.ndarray:
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("paths must have shape (n_paths, n_times>=2)")
    return paths[:, -1] / paths[:, 0] - 1.0


@dataclass(frozen=True)
class RILABufferedSegment:
    """Registered index-linked annuity segment with downside buffer and cap."""

    notional: float
    maturity: float
    buffer: float
    cap: float
    participation: float = 1.0
    floor: float = -1.0
    index_start: float = 100.0

    def __post_init__(self) -> None:
        _validate_common(self.notional, self.maturity)
        if not (0.0 <= self.buffer <= 1.0):
            raise ValueError("buffer must lie in [0, 1]")
        if self.cap < 0.0 or self.participation < 0.0:
            raise ValueError("cap and participation must be non-negative")
        if self.floor > 0.0 or self.floor < -1.0:
            raise ValueError("floor must lie in [-1, 0]")
        if self.index_start <= 0.0:
            raise ValueError("index_start must be positive")

    def credited_return(self, index_return: np.ndarray | float) -> np.ndarray:
        """Calculate contract return before multiplication by notional."""
        r = np.asarray(index_return, dtype=float)
        upside = np.minimum(self.participation * np.maximum(r, 0.0), self.cap)
        downside = np.minimum(r + self.buffer, 0.0)
        return np.maximum(upside + downside, self.floor)

    def payoff(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        del times
        index_return = paths[:, -1] / self.index_start - 1.0
        return self.notional * (1.0 + self.credited_return(index_return))


@dataclass(frozen=True)
class RILAFlooredSegment:
    """RILA segment with participation, cap, and explicit return floor."""

    notional: float
    maturity: float
    floor: float
    cap: float
    participation: float = 1.0
    index_start: float = 100.0

    def __post_init__(self) -> None:
        _validate_common(self.notional, self.maturity)
        if not (-1.0 <= self.floor <= self.cap):
            raise ValueError("require -1 <= floor <= cap")
        if self.participation < 0.0:
            raise ValueError("participation must be non-negative")
        if self.index_start <= 0.0:
            raise ValueError("index_start must be positive")

    def credited_return(self, index_return: np.ndarray | float) -> np.ndarray:
        return np.clip(
            self.participation * np.asarray(index_return), self.floor, self.cap
        )

    def payoff(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        del times
        index_return = paths[:, -1] / self.index_start - 1.0
        return self.notional * (1.0 + self.credited_return(index_return))


@dataclass(frozen=True)
class FIAAnnualPointToPoint:
    """FIA annual point-to-point crediting with annual caps and optional spread."""

    notional: float
    maturity: float
    annual_cap: float
    participation: float = 1.0
    spread: float = 0.0
    annual_floor: float = 0.0
    index_start: float = 100.0

    def __post_init__(self) -> None:
        _validate_common(self.notional, self.maturity)
        if self.annual_cap < self.annual_floor:
            raise ValueError("annual_cap must be at least annual_floor")
        if self.participation < 0.0 or self.spread < 0.0:
            raise ValueError("participation and spread must be non-negative")
        if self.index_start <= 0.0:
            raise ValueError("index_start must be positive")

    def payoff(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        if paths.ndim != 2 or times.ndim != 1 or paths.shape[1] != times.size:
            raise ValueError("path columns must match times")
        horizon = min(self.maturity, float(times[-1]))
        anniversaries = np.minimum(np.arange(0.0, horizon + 1.0, 1.0), horizon)
        if anniversaries[-1] < horizon:
            anniversaries = np.append(anniversaries, horizon)
        indices = np.array([int(np.argmin(np.abs(times - t))) for t in anniversaries])
        starts = paths[:, indices[:-1]].copy()
        starts[:, 0] = self.index_start
        period_returns = paths[:, indices[1:]] / starts - 1.0
        credits = np.clip(
            self.participation * period_returns - self.spread,
            self.annual_floor,
            self.annual_cap,
        )
        return self.notional * np.prod(1.0 + credits, axis=1)


@dataclass(frozen=True)
class IULAccountValue:
    """Indexed universal life account with monthly charges and mortality/lapse.

    Mortality and lapse rates are annual probabilities.  A path that terminates
    receives the account value at the first monthly decrement; otherwise the
    terminal account value is paid.
    """

    notional: float
    maturity: float
    annual_cap: float
    participation: float = 1.0
    annual_floor: float = 0.0
    annual_policy_charge: float = 0.0
    monthly_expense_charge: float = 0.0
    mortality_rates: Sequence[float] = field(default_factory=lambda: (0.0,))
    lapse_rates: Sequence[float] = field(default_factory=lambda: (0.0,))
    death_benefit_multiple: float = 1.0
    index_start: float = 100.0

    def __post_init__(self) -> None:
        _validate_common(self.notional, self.maturity)
        if self.annual_cap < self.annual_floor or self.participation < 0.0:
            raise ValueError("invalid crediting terms")
        if self.annual_policy_charge < 0.0 or self.monthly_expense_charge < 0.0:
            raise ValueError("charges must be non-negative")
        for name, values in (
            ("mortality_rates", self.mortality_rates),
            ("lapse_rates", self.lapse_rates),
        ):
            x = np.asarray(values, dtype=float)
            if x.ndim != 1 or x.size == 0 or np.any((x < 0.0) | (x > 1.0)):
                raise ValueError(f"{name} must contain annual probabilities in [0,1]")
        if self.death_benefit_multiple < 0.0:
            raise ValueError("death_benefit_multiple must be non-negative")
        if self.index_start <= 0.0:
            raise ValueError("index_start must be positive")

    @staticmethod
    def _rate_for_year(rates: Sequence[float], year: int) -> float:
        return float(rates[min(year, len(rates) - 1)])

    def payoff(
        self,
        paths: np.ndarray,
        times: np.ndarray,
        uniform_draws: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return projected payment amounts; optional draws model decrements."""
        amounts, _ = self.cashflows(paths, times, uniform_draws)
        return amounts

    def cashflows(
        self,
        paths: np.ndarray,
        times: np.ndarray,
        uniform_draws: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return pathwise payment amounts and payment times."""
        if paths.ndim != 2 or paths.shape[1] != times.size:
            raise ValueError("path columns must match times")
        n_paths = paths.shape[0]
        account = np.full(n_paths, self.notional, dtype=float)
        alive = np.ones(n_paths, dtype=bool)
        paid = np.zeros(n_paths)
        payment_times = np.full(n_paths, float(times[-1]))
        if uniform_draws is not None and uniform_draws.shape != (
            n_paths,
            times.size - 1,
        ):
            raise ValueError("uniform_draws shape must be (n_paths, n_steps)")

        last_anniversary = 0
        for step in range(1, times.size):
            dt = float(times[step] - times[step - 1])
            account[alive] = np.maximum(
                account[alive] * (1.0 - self.annual_policy_charge * dt)
                - self.monthly_expense_charge * 12.0 * dt,
                0.0,
            )
            current_anniversary = int(np.floor(times[step] + 1e-10))
            if current_anniversary > last_anniversary or step == times.size - 1:
                base_step = int(np.argmin(np.abs(times - last_anniversary)))
                index_base = (
                    self.index_start if last_anniversary == 0 else paths[:, base_step]
                )
                r = paths[:, step] / index_base - 1.0
                credit = np.clip(
                    self.participation * r, self.annual_floor, self.annual_cap
                )
                account[alive] *= 1.0 + credit[alive]
                last_anniversary = current_anniversary
            if uniform_draws is not None:
                year = min(int(times[step - 1]), int(np.ceil(self.maturity)) - 1)
                qx = self._rate_for_year(self.mortality_rates, year)
                lx = self._rate_for_year(self.lapse_rates, year)
                mortality = 1.0 - (1.0 - qx) ** dt
                lapse = 1.0 - (1.0 - lx) ** dt
                u = uniform_draws[:, step - 1]
                deaths = alive & (u < mortality)
                lapses = alive & ~deaths & (u < mortality + (1.0 - mortality) * lapse)
                paid[deaths] = account[deaths] * self.death_benefit_multiple
                paid[lapses] = account[lapses]
                payment_times[deaths | lapses] = times[step]
                alive[deaths | lapses] = False
        paid[alive] = account[alive]
        return paid, payment_times


@dataclass(frozen=True)
class Cohort:
    """Homogeneous block of contracts with policy count and behavior multipliers."""

    contract: LiabilityContract
    policy_count: int = 1
    mortality_multiplier: float = 1.0
    lapse_multiplier: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.policy_count < 0:
            raise ValueError("policy_count must be non-negative")
        if self.mortality_multiplier < 0.0 or self.lapse_multiplier < 0.0:
            raise ValueError("behavior multipliers must be non-negative")

    @property
    def notional(self) -> float:
        """Aggregate cohort notional."""
        return self.policy_count * self.contract.notional

    @property
    def maturity(self) -> float:
        """Contract maturity in years."""
        return self.contract.maturity
