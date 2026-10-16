"""Central finite-difference bump-and-revalue risk measures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .market import MarketState
from .pricing import Pricer


@dataclass(frozen=True)
class BumpSizes:
    """Finite-difference bump conventions."""

    spot_relative: float = 0.01
    vol_absolute: float = 0.01
    rate_bp: float = 1.0
    time_years: float = 1.0 / 365.0

    def __post_init__(self) -> None:
        if (
            min(self.spot_relative, self.vol_absolute, self.rate_bp, self.time_years)
            <= 0.0
        ):
            raise ValueError("all bump sizes must be positive")


@dataclass(frozen=True)
class Greeks:
    """PV and finite-difference sensitivities.

    Delta and gamma are per spot unit and squared spot unit. Vega is per one
    absolute volatility unit (multiply by 0.01 for a vol point), rho is per one
    absolute rate unit, DV01 is the direct PV change for +1 bp, and theta is the
    annualized PV change as valuation time advances.
    """

    pv: float
    delta: float
    gamma: float
    vega: float
    rho: float
    dv01: float
    theta: float

    def as_dict(self) -> dict[str, float]:
        """Return measures in stable field order."""
        return {
            "pv": self.pv,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "rho": self.rho,
            "dv01": self.dv01,
            "theta": self.theta,
        }


def bump_and_revalue(
    pricer: Pricer,
    market: MarketState,
    bumps: BumpSizes = BumpSizes(),
) -> Greeks:
    """Calculate PV, delta, gamma, vega, rho/DV01, and theta."""
    pv = float(pricer(market))

    spot_h = market.spot * bumps.spot_relative
    up_s = float(pricer(market.bumped(spot_relative=bumps.spot_relative)))
    dn_s = float(pricer(market.bumped(spot_relative=-bumps.spot_relative)))
    delta = (up_s - dn_s) / (2.0 * spot_h)
    gamma = (up_s - 2.0 * pv + dn_s) / spot_h**2

    up_v = float(pricer(market.bumped(vol_absolute=bumps.vol_absolute)))
    dn_v = float(pricer(market.bumped(vol_absolute=-bumps.vol_absolute)))
    vega = (up_v - dn_v) / (2.0 * bumps.vol_absolute)

    up_r = float(pricer(market.bumped(rate_bp=bumps.rate_bp)))
    dn_r = float(pricer(market.bumped(rate_bp=-bumps.rate_bp)))
    rate_h = bumps.rate_bp * 1e-4
    rho = (up_r - dn_r) / (2.0 * rate_h)
    dv01 = (up_r - pv) / bumps.rate_bp

    future = float(pricer(market.bumped(time_years=bumps.time_years)))
    theta = (future - pv) / bumps.time_years
    return Greeks(pv, delta, gamma, vega, rho, dv01, theta)


def aggregate_greeks(
    positions: Mapping[str, tuple[float, Greeks]],
) -> Greeks:
    """Quantity-weight and aggregate named position Greeks."""
    totals = {name: 0.0 for name in Greeks.__dataclass_fields__}
    for quantity, measures in positions.values():
        for name, value in measures.as_dict().items():
            totals[name] += quantity * value
    return Greeks(**totals)
