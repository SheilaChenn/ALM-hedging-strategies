from __future__ import annotations

import numpy as np
import pytest

from alm_hedging import (
    DynamicHedgeInstrument,
    HedgeConstraints,
    HedgeInstrument,
    SyntheticMarketModel,
    optimize_hedge,
    simulate_dynamic_hedge,
    solve_call_spread_cap,
)


def test_option_budget_solver_reprices_target() -> None:
    market = SyntheticMarketModel(atm_vol=0.20).build()
    solution = solve_call_spread_cap(
        market,
        maturity=1.0,
        budget=0.04,
        participation=1.0,
        notional=1.0,
    )
    assert solution.converged
    assert 0.0 < solution.parameter < 5.0
    assert solution.premium == pytest.approx(0.04, abs=1e-8)


def test_dynamic_hedge_replay_is_deterministic() -> None:
    model = SyntheticMarketModel()
    spots = (100.0, 98.0, 103.0, 101.0)
    markets = [
        model.build(spot=spot).bumped(time_years=i / 252)
        for i, spot in enumerate(spots)
    ]
    future = DynamicHedgeInstrument(
        name="future",
        kind="future",
        tenor=1.0,
        value=lambda state: state.spot,
        risks=lambda state: {"delta": 1.0},
        max_position=1_000.0,
        max_trade=1_000.0,
    )

    def run() -> object:
        return simulate_dynamic_hedge(
            markets,
            liability_pricer=lambda state: 1_000.0 + 5.0 * (state.spot - 100.0),
            liability_risk=lambda state: {"delta": 5.0},
            instruments=[future],
            risk_targets=("delta",),
            constraints=HedgeConstraints(turnover_budget=1_000_000.0),
        )

    first = run()
    second = run()
    assert first.history.equals(second.history)
    assert np.isfinite(first.cumulative_pnl)
    assert abs(first.final_positions["future"] - 5.0) < 1e-8


def test_stress_cvar_enters_hedge_objective() -> None:
    instrument = HedgeInstrument(
        "crash_put",
        "option",
        tenor=1.0,
        price=5.0,
        premium=5.0,
        exposures={"delta": 0.0},
        max_position=100.0,
        max_trade=100.0,
    )
    solution = optimize_hedge(
        {"delta": 0.0},
        [instrument],
        constraints=HedgeConstraints(
            premium_budget=500.0,
            turnover_budget=500.0,
        ),
        scenario_liability_losses=[10.0, 20.0, 100.0],
        instrument_scenario_pnls={"crash_put": [0.0, 1.0, 10.0]},
        cvar_weight=1.0,
        cvar_alpha=0.66,
    )
    assert solution.positions["crash_put"] > 0.0
    assert solution.stress_cvar < 100.0
