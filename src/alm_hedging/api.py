"""Dashboard-facing facade backed entirely by the quantitative engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    Cohort,
    FIAAnnualPointToPoint,
    IULAccountValue,
    RILABufferedSegment,
)
from .market import MarketState, SyntheticMarketModel
from .optimization import HedgeConstraints, HedgeInstrument
from .optimization import optimize_hedge as solve
from .pricing import portfolio_value
from .risk import bump_and_revalue
from .simulation import (
    DynamicHedgeInstrument,
    simulate_dynamic_hedge,
)


def _market(values: Mapping[str, Any]) -> MarketState:
    return SyntheticMarketModel(
        base_rate=float(values.get("rate", 0.035)),
        dividend_yield=float(values.get("dividend_yield", 0.015)),
        atm_vol=float(values.get("volatility", 0.20)),
    ).build(spot=float(values.get("spot", 100.0)))


def _liabilities(
    cohorts: Sequence[Mapping[str, Any]], index_start: float = 100.0
) -> list[Cohort]:
    """Build cohorts whose index strikes are set at the valuation spot.

    Dashboard cohorts are treated as newly issued segments, so every contract
    is struck at ``index_start`` (the valuation-market spot).  Leaving the
    contract default of 100 while valuing at an index level of several thousand
    put every option leg deep in the money and reported zero delta.
    """
    result: list[Cohort] = []
    for item in cohorts:
        product = str(item["product"]).upper()
        notional = float(item["account_value"])
        maturity = max(float(item["duration"]), 0.25)
        participation = float(item["participation"])
        cap = float(item["cap"])
        floor = float(item["floor"])
        if product == "RILA":
            contract = RILABufferedSegment(
                notional,
                maturity,
                buffer=max(-floor, 0.0),
                cap=cap,
                participation=participation,
                floor=max(floor, -1.0),
                index_start=index_start,
            )
        elif product == "FIA":
            contract = FIAAnnualPointToPoint(
                notional,
                maturity,
                annual_cap=cap,
                participation=participation,
                annual_floor=max(floor, 0.0),
                index_start=index_start,
            )
        elif product == "IUL":
            contract = IULAccountValue(
                notional,
                maturity,
                annual_cap=cap,
                participation=participation,
                annual_floor=max(floor, 0.0),
                annual_policy_charge=0.012,
                monthly_expense_charge=8.0,
                mortality_rates=(0.0020, 0.0022, 0.0024, 0.0027, 0.0030),
                lapse_rates=(0.08, 0.07, 0.06, 0.05, 0.04),
                index_start=index_start,
            )
        else:
            raise ValueError(f"unsupported product {product!r}")
        result.append(Cohort(contract, label=product))
    return result


def _value(
    cohorts: Sequence[Mapping[str, Any]],
    market: Mapping[str, Any],
    *,
    n_paths: int = 2_000,
) -> float:
    state = _market(market)
    return portfolio_value(
        _liabilities(cohorts, state.spot), state, n_paths=n_paths, seed=46_011
    )


def get_market_snapshot(
    as_of: str,
    spot: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
) -> dict[str, Any]:
    """Create a reproducible synthetic market snapshot for the dashboard."""
    state = SyntheticMarketModel(
        base_rate=rate,
        dividend_yield=dividend_yield,
        atm_vol=volatility,
    ).build(spot=spot)
    rng = np.random.default_rng(46_011)
    dates = pd.bdate_range(end=pd.Timestamp(as_of), periods=126)
    returns = rng.normal(
        (rate - dividend_yield - volatility**2 / 2) / 252,
        volatility / np.sqrt(252),
        len(dates),
    )
    history = spot * np.exp(np.cumsum(returns) - np.sum(returns))
    return {
        "spot": spot,
        "rate": rate,
        "volatility": volatility,
        "dividend_yield": dividend_yield,
        "history": pd.DataFrame({"date": dates, "spot": history}),
        "curve": pd.DataFrame(
            {"tenor": state.rate_curve.tenors, "zero_rate": state.rate_curve.zero_rates}
        ),
        "as_of": as_of,
    }


def price_portfolio(
    cohorts: Sequence[Mapping[str, Any]],
    market: Mapping[str, Any],
) -> dict[str, Any]:
    """Price dashboard cohorts with deterministic analytic/Monte Carlo valuation."""
    rows = []
    for item in cohorts:
        pv = _value([item], market)
        account = float(item["account_value"])
        guarantee = account * float(item.get("guarantee", 0.0))
        rows.append(
            {
                "product": item["product"],
                "option_budget": max(pv - account, 0.0),
                "guarantee": guarantee,
                "floor_value": max(account - pv, 0.0),
                "liability": pv,
            }
        )
    components = pd.DataFrame(rows)
    return {
        "components": components,
        "liability": float(components["liability"].sum()),
        "account_value": float(sum(float(x["account_value"]) for x in cohorts)),
        "spot": float(market["spot"]),
    }


def calculate_greeks(
    cohorts: Sequence[Mapping[str, Any]],
    market: Mapping[str, Any],
) -> pd.DataFrame:
    """Return portfolio PV and bump-and-revalue market sensitivities."""
    state = _market(market)
    liabilities = _liabilities(cohorts, state.spot)
    measures = bump_and_revalue(
        lambda bumped: portfolio_value(liabilities, bumped, n_paths=2_000, seed=46_011),
        state,
    )
    return pd.DataFrame(
        {
            "Greek": ["Delta", "Gamma", "Vega", "Rho", "DV01", "Theta"],
            "Exposure": [
                measures.delta,
                measures.gamma,
                measures.vega,
                measures.rho,
                measures.dv01,
                measures.theta / 365.0,
            ],
            "Unit": [
                "$ / index",
                "$ / index²",
                "$ / 1.00 vol",
                "$ / 1.00 rate",
                "$ / bp",
                "$ / day",
            ],
        }
    )


def run_stress_test(
    cohorts: Sequence[Mapping[str, Any]],
    market: Mapping[str, Any],
) -> pd.DataFrame:
    """Full-revaluation equity/volatility stress matrix."""
    state = _market(market)
    liabilities = _liabilities(cohorts, state.spot)
    base = portfolio_value(liabilities, state, n_paths=1_000, seed=46_011)
    spot_shocks = (-0.25, -0.15, -0.05, 0.0, 0.05, 0.15, 0.25)
    vol_shocks = (-0.10, -0.05, 0.0, 0.05, 0.10)
    values = [
        [
            portfolio_value(
                liabilities,
                state.bumped(spot_relative=spot, vol_absolute=vol),
                n_paths=1_000,
                seed=46_011,
            )
            - base
            for spot in spot_shocks
        ]
        for vol in vol_shocks
    ]
    return pd.DataFrame(
        values,
        index=[f"{x:+.0%}" for x in vol_shocks],
        columns=[f"{x:+.0%}" for x in spot_shocks],
    )


def optimize_hedge(
    cohorts: Sequence[Mapping[str, Any]],
    market: Mapping[str, Any],
    premium_budget: float,
    turnover_limit: float,
    instruments: Sequence[str],
) -> dict[str, Any]:
    """Recommend a constrained liquid hedge against delta, vega, and rho.

    Liability Greeks are reported as changes in liability value.  The insurer
    is short the liability, so its net exposure is the negative of each
    liability Greek and the hedge assets must carry the *same-signed* Greeks as
    the liability (a block of index-linked crediting is hedged by buying, not
    selling, index calls).  ``effectiveness`` is the reduction in a scaled norm
    of first-order Greek exposures evaluated at the valuation market; it is not
    a P&L-based or out-of-sample effectiveness measure.
    """
    greek_frame = calculate_greeks(cohorts, market)
    risks = dict(
        zip(greek_frame["Greek"].str.lower(), greek_frame["Exposure"], strict=True)
    )
    spot = float(market["spot"])
    scale = max(sum(float(x["account_value"]) for x in cohorts), 1.0)
    # Exposure that moves value by 0.1% of account value for a 1% spot move,
    # a one-point vol move, or a 10 bp rate move; used to floor the weights so
    # that creating exposure on an initially flat Greek is penalized.
    reference_exposure = {
        "delta": 0.1 * scale / spot,
        "vega": 0.1 * scale,
        "rho": 1.0 * scale,
    }
    templates = {
        "Equity futures": ("future", {"delta": 50.0}, 0.0, spot * 50),
        "ATM calls": ("option", {"delta": 55.0, "vega": 3_500.0}, 800.0, 800.0),
        "Put spread": ("option", {"delta": -25.0, "vega": 1_800.0}, 400.0, 400.0),
        "Variance swap": ("option", {"vega": 10_000.0}, 500.0, 500.0),
        "10Y swap": ("swap", {"rho": 1_000.0}, 0.0, 100.0),
    }
    universe = [
        HedgeInstrument(
            name=name,
            kind=templates[name][0],
            tenor=10.0 if "10Y" in name else 1.0,
            price=templates[name][3],
            premium=templates[name][2],
            exposures=templates[name][1],
            max_position=max(scale / max(templates[name][3], 1.0), 1.0),
            max_trade=max(turnover_limit * scale / max(templates[name][3], 1.0), 1.0),
            lot_size=1.0,
            transaction_cost=0.05,
        )
        for name in instruments
    ]
    # Every Greek any eligible instrument can move is part of the residual, so
    # that a hedge cannot report high effectiveness while silently creating
    # exposure on an untargeted Greek.
    target_names = tuple(
        name
        for name in ("delta", "vega", "rho")
        if any(abs(item.exposures.get(name, 0.0)) > 0.0 for item in universe)
    )
    if not target_names:
        raise ValueError("selected instruments carry no hedgeable exposure")
    normalized_weights = {
        name: 1.0 / max(abs(float(risks.get(name, 0.0))), reference_exposure[name])
        for name in target_names
    }
    net_exposure = {name: -float(risks.get(name, 0.0)) for name in target_names}
    solution = solve(
        net_exposure,
        universe,
        constraints=HedgeConstraints(
            premium_budget=premium_budget,
            turnover_budget=max(turnover_limit * scale, 1.0),
            max_concentration=1.0,
        ),
        exposure_weights=normalized_weights,
    )
    allocation = pd.DataFrame(
        [
            {
                "instrument": item.name,
                "units": solution.trades[item.name],
                "delta_offset": solution.trades[item.name]
                * item.exposures.get("delta", 0.0),
                "vega_offset": solution.trades[item.name]
                * item.exposures.get("vega", 0.0),
                "premium": abs(solution.positions[item.name]) * item.premium,
            }
            for item in universe
        ]
    )
    initial_norm = np.linalg.norm(
        [net_exposure[x] * normalized_weights[x] for x in target_names]
    )
    residual_norm = np.linalg.norm(
        [solution.residual_exposures[x] * normalized_weights[x] for x in target_names]
    )
    # Residuals are the insurer's net exposure after hedging; report them in
    # liability-Greek sign so they are comparable with ``calculate_greeks``.
    return {
        "allocation": allocation,
        "premium": solution.premium_used,
        "residual_delta": -solution.residual_exposures.get("delta", 0.0),
        "residual_vega": -solution.residual_exposures.get("vega", 0.0),
        "residual_rho": -solution.residual_exposures.get("rho", 0.0),
        "effectiveness": float(
            np.clip(1.0 - residual_norm / max(initial_norm, 1e-12), 0.0, 1.0)
        ),
        "effectiveness_basis": (
            "1 - ||scaled residual Greeks|| / ||scaled net Greeks|| over "
            + ", ".join(target_names)
            + " at the valuation market (first-order, in-sample)"
        ),
        "status": "Optimal" if solution.success else solution.message,
    }


def backtest_hedge(
    pricing: Mapping[str, Any],
    market: Mapping[str, Any],
    frequency: str,
    transaction_cost_bps: float,
    lookback_years: int,
    cohorts: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Run a deterministic real-world dynamic delta-hedge simulation.

    The liability is represented by a delta-gamma Taylor proxy around the
    valuation spot rather than by full revaluation along the path.  When
    ``cohorts`` are supplied the proxy coefficients are the engine's own
    bump-and-revalue delta and gamma; otherwise stylized coefficients are used.
    The simulation is therefore an illustration of delta-hedging mechanics and
    costs, not an out-of-sample test of the priced liabilities.
    """
    periods_per_year = {"Daily": 252, "Weekly": 52, "Monthly": 12}[frequency]
    periods = periods_per_year * lookback_years
    rng = np.random.default_rng(46_011 + periods)
    dt = 1.0 / periods_per_year
    spot0 = float(market["spot"])
    rate = float(market["rate"])
    vol = float(market["volatility"])
    shocks = rng.normal((0.055 - vol**2 / 2) * dt, vol * np.sqrt(dt), periods)
    spots = spot0 * np.exp(np.r_[0.0, np.cumsum(shocks)])
    base_model = SyntheticMarketModel(
        base_rate=rate,
        dividend_yield=float(market["dividend_yield"]),
        atm_vol=vol,
    )
    states = [
        base_model.build(spot=float(spot)).bumped(time_years=i * dt)
        for i, spot in enumerate(spots)
    ]
    liability0 = float(pricing["liability"])
    if cohorts:
        greek_frame = calculate_greeks(cohorts, market)
        measured = dict(
            zip(greek_frame["Greek"].str.lower(), greek_frame["Exposure"], strict=True)
        )
        delta = float(measured["delta"])
        gamma = float(measured["gamma"])
    else:
        delta = 0.43 * float(pricing["account_value"]) / spot0
        gamma = 0.001 * float(pricing["account_value"]) / spot0**2

    def liability_value(state: MarketState) -> float:
        change = state.spot - spot0
        return liability0 + delta * change + 0.5 * gamma * change**2

    future = DynamicHedgeInstrument(
        "Equity future",
        "future",
        1.0,
        value=lambda state: state.spot,
        risks=lambda state: {"delta": 1.0},
        max_position=10_000_000.0,
        max_trade=10_000_000.0,
        transaction_cost=transaction_cost_bps * spot0 / 10_000.0,
    )
    result = simulate_dynamic_hedge(
        states,
        liability_value,
        lambda state: {"delta": delta + gamma * (state.spot - spot0)},
        [future],
        risk_targets=("delta",),
        constraints=HedgeConstraints(turnover_budget=np.inf),
    )
    frame = result.history.rename(
        columns={"transaction_cost": "cost", "total_pnl": "net_pnl"}
    )
    end = pd.Timestamp(str(market.get("as_of", date.today().isoformat())))
    calendar = {"Daily": "B", "Weekly": "W-FRI", "Monthly": "ME"}[frequency]
    frame["date"] = pd.date_range(end=end, periods=len(frame), freq=calendar)
    return frame[["date", "liability_pnl", "hedge_pnl", "cost", "net_pnl"]]
