from __future__ import annotations

import pytest

from alm_hedging.contracts import Cohort, RILAFlooredSegment
from alm_hedging.market import MarketState
from alm_hedging.pricing import analytic_liability_value, portfolio_value
from alm_hedging.stress import StressScenario, run_stress_scenarios


def _liabilities() -> tuple[Cohort, Cohort]:
    return (
        Cohort(
            RILAFlooredSegment(100.0, 1.0, floor=-0.10, cap=0.15),
            policy_count=10,
            label="A",
        ),
        Cohort(
            RILAFlooredSegment(80.0, 1.0, floor=0.0, cap=0.08, participation=0.75),
            policy_count=20,
            label="B",
        ),
    )


def test_portfolio_value_reconciles_to_constituents(market: MarketState) -> None:
    liabilities = _liabilities()
    expected = sum(analytic_liability_value(item, market) for item in liabilities)
    assert portfolio_value(liabilities, market) == pytest.approx(expected)


def test_equity_stress_directionality(market: MarketState) -> None:
    liabilities = _liabilities()

    def value(state: MarketState, items: tuple[Cohort, ...] | list[Cohort]) -> float:
        return portfolio_value(items, state)

    down, up = run_stress_scenarios(
        market,
        (
            StressScenario("down", equity_relative=-0.20),
            StressScenario("up", equity_relative=0.20),
        ),
        value,  # type: ignore[arg-type]
        liabilities,
    )
    assert down.pnl < 0.0
    assert up.pnl > 0.0
