"""Dynamic hedge simulation with transaction costs and P&L attribution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .market import MarketState
from .optimization import HedgeConstraints, HedgeInstrument, optimize_hedge
from .pricing import Pricer
from .risk import Greeks


@dataclass(frozen=True)
class DynamicHedgeInstrument:
    """Market-aware instrument callbacks used by the dynamic simulator."""

    name: str
    kind: str
    tenor: float
    value: Callable[[MarketState], float]
    risks: Callable[[MarketState], Mapping[str, float]]
    lot_size: float = 1.0
    max_position: float = np.inf
    max_trade: float = np.inf
    transaction_cost: float = 0.0
    premium: Callable[[MarketState], float] | None = None


@dataclass(frozen=True)
class DynamicHedgeResult:
    """Time series and terminal summary from a dynamic hedge simulation."""

    history: pd.DataFrame
    final_positions: Mapping[str, float]

    @property
    def cumulative_pnl(self) -> float:
        """Total realized mark-to-market hedge P&L net of liability change."""
        return float(self.history["total_pnl"].sum())

    @property
    def cumulative_transaction_cost(self) -> float:
        """Total transaction costs paid."""
        return float(self.history["transaction_cost"].sum())


def _greeks_mapping(value: Greeks | Mapping[str, float]) -> Mapping[str, float]:
    return value.as_dict() if isinstance(value, Greeks) else value


def simulate_dynamic_hedge(
    markets: Sequence[MarketState],
    liability_pricer: Pricer,
    liability_risk: Callable[[MarketState], Greeks | Mapping[str, float]],
    instruments: Sequence[DynamicHedgeInstrument],
    *,
    rebalance_every: int = 1,
    risk_targets: Sequence[str] = ("delta",),
    exposure_weights: Mapping[str, float] | None = None,
    constraints: HedgeConstraints = HedgeConstraints(),
    initial_positions: Mapping[str, float] | None = None,
) -> DynamicHedgeResult:
    """Simulate periodic constrained rebalancing over supplied market states.

    P&L is reported from the insurer perspective: hedge gains minus increases in
    liability value.  First-order attribution uses beginning-of-period net risk;
    unexplained nonlinear/basis effects are collected in ``residual_pnl``.
    """
    if len(markets) < 2:
        raise ValueError("at least two market states are required")
    if rebalance_every <= 0 or not risk_targets:
        raise ValueError("rebalance_every and risk_targets must be non-empty/positive")
    if not instruments:
        raise ValueError("at least one dynamic hedge instrument is required")

    positions = {
        item.name: float((initial_positions or {}).get(item.name, 0.0))
        for item in instruments
    }
    rows: list[dict[str, float]] = []

    def net_risks(
        market: MarketState, liability: Mapping[str, float]
    ) -> dict[str, float]:
        result = {name: -float(liability.get(name, 0.0)) for name in risk_targets}
        for item in instruments:
            item_risks = item.risks(market)
            for name in risk_targets:
                result[name] += positions[item.name] * float(item_risks.get(name, 0.0))
        return result

    for step in range(len(markets) - 1):
        market, next_market = markets[step], markets[step + 1]
        liability_before = float(liability_pricer(market))
        liability_risks = _greeks_mapping(liability_risk(market))
        transaction_cost = 0.0

        if step % rebalance_every == 0:
            hedge_universe = []
            for item in instruments:
                price = float(item.value(market))
                hedge_universe.append(
                    HedgeInstrument(
                        name=item.name,
                        kind=item.kind,
                        tenor=item.tenor,
                        price=abs(price),
                        exposures={
                            name: float(item.risks(market).get(name, 0.0))
                            for name in risk_targets
                        },
                        premium=(
                            float(item.premium(market))
                            if item.premium is not None
                            else (max(price, 0.0) if item.kind == "option" else 0.0)
                        ),
                        max_position=item.max_position,
                        max_trade=item.max_trade,
                        lot_size=item.lot_size,
                        transaction_cost=item.transaction_cost,
                    )
                )
            solution = optimize_hedge(
                {name: -float(liability_risks.get(name, 0.0)) for name in risk_targets},
                hedge_universe,
                current_positions=positions,
                exposure_weights=exposure_weights,
                constraints=constraints,
            )
            transaction_cost = sum(
                abs(solution.trades[item.name]) * item.transaction_cost
                for item in instruments
            )
            positions = dict(solution.positions)

        beginning_net = net_risks(market, liability_risks)
        liability_after = float(liability_pricer(next_market))
        liability_pnl = -(liability_after - liability_before)
        hedge_pnl = sum(
            positions[item.name]
            * (float(item.value(next_market)) - float(item.value(market)))
            for item in instruments
        )
        total_pnl = liability_pnl + hedge_pnl - transaction_cost

        d_spot = next_market.spot - market.spot
        d_vol = float(np.mean(next_market.vol_surface.vols - market.vol_surface.vols))
        d_rate = float(
            np.mean(next_market.rate_curve.zero_rates - market.rate_curve.zero_rates)
        )
        dt = next_market.valuation_time - market.valuation_time
        delta_pnl = beginning_net.get("delta", 0.0) * d_spot
        gamma_pnl = 0.5 * beginning_net.get("gamma", 0.0) * d_spot**2
        vega_pnl = beginning_net.get("vega", 0.0) * d_vol
        rho_pnl = beginning_net.get("rho", 0.0) * d_rate
        theta_pnl = beginning_net.get("theta", 0.0) * dt
        explained = delta_pnl + gamma_pnl + vega_pnl + rho_pnl + theta_pnl

        row: dict[str, float] = {
            "step": float(step),
            "time": next_market.valuation_time,
            "spot": next_market.spot,
            "liability_value": liability_after,
            "liability_pnl": liability_pnl,
            "hedge_pnl": hedge_pnl,
            "transaction_cost": transaction_cost,
            "delta_pnl": delta_pnl,
            "gamma_pnl": gamma_pnl,
            "vega_pnl": vega_pnl,
            "rho_pnl": rho_pnl,
            "theta_pnl": theta_pnl,
            "residual_pnl": total_pnl - explained + transaction_cost,
            "total_pnl": total_pnl,
        }
        row.update({f"position_{name}": value for name, value in positions.items()})
        rows.append(row)

    return DynamicHedgeResult(pd.DataFrame(rows), positions)
