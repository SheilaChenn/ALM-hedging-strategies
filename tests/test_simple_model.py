"""Tests for the self-contained simple model in ``simple/simple_model.py``."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simple"))

import simple_model as sm  # noqa: E402

MARKET = sm.Market()


def test_put_call_parity() -> None:
    spot, strike, t, r, q, vol = 100.0, 105.0, 1.0, 0.04, 0.015, 0.19
    call = sm.bs_call(spot, strike, t, r, q, vol)
    put = sm.bs_put(spot, strike, t, r, q, vol)
    expected = spot * math.exp(-q * t) - strike * math.exp(-r * t)
    assert call - put == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("index_return", "expected"),
    [
        (0.25, 0.18),  # above the cap: credited the cap
        (0.18, 0.18),  # exactly at the cap
        (0.12, 0.12),  # between zero and the cap: full participation
        (0.0, 0.0),  # flat index
        (-0.05, 0.0),  # inside the 10% buffer: no loss
        (-0.10, 0.0),  # exactly at the buffer edge
        (-0.25, -0.15),  # beyond the buffer: loss net of the buffer
    ],
)
def test_rila_payoff_edge_cases(index_return: float, expected: float) -> None:
    product = sm.rila()
    assert product.credited_return(index_return) == pytest.approx(expected, abs=1e-12)


def test_fia_and_iul_payoffs_floor_and_cap() -> None:
    assert sm.fia().credited_return(-0.30) == 0.0
    assert sm.fia().credited_return(0.05) == pytest.approx(0.05)
    assert sm.fia().credited_return(0.30) == pytest.approx(0.095)
    assert sm.iul().credited_return(0.30) == pytest.approx(0.11)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [90.0, 100.0, 118.0])
def test_analytic_greeks_match_finite_differences(kind: str, strike: float) -> None:
    s, t, r, q, vol = 100.0, 1.0, 0.04, 0.015, 0.19

    def price(s_: float = s, t_: float = t, r_: float = r, vol_: float = vol) -> float:
        return sm.bs_greeks(s_, strike, t_, r_, q, vol_, kind).price

    g = sm.bs_greeks(s, strike, t, r, q, vol, kind)
    h = 1e-3
    assert g.delta == pytest.approx(
        (price(s_=s + h) - price(s_=s - h)) / (2 * h), rel=1e-6
    )
    assert g.gamma == pytest.approx(
        (price(s_=s + h) - 2 * price() + price(s_=s - h)) / (h * h), rel=1e-4
    )
    assert g.vega == pytest.approx(
        (price(vol_=vol + 1e-5) - price(vol_=vol - 1e-5)) / 2e-5, rel=1e-6
    )
    assert g.rho == pytest.approx(
        (price(r_=r + 1e-5) - price(r_=r - 1e-5)) / 2e-5, rel=1e-6
    )
    # Theta is dV/dt with t calendar time, i.e. minus the derivative in time to maturity.
    assert g.theta == pytest.approx(
        -(price(t_=t + 1e-5) - price(t_=t - 1e-5)) / 2e-5, rel=1e-6
    )


def test_product_greeks_sum_legs() -> None:
    val = sm.price_product(sm.rila(), MARKET)
    assert len(val.legs) == 3
    assert val.pv == pytest.approx(sum(leg.greeks.price for leg in val.legs))
    assert val.greeks.delta == pytest.approx(sum(leg.greeks.delta for leg in val.legs))


def test_cap_solver_round_trip() -> None:
    for builder, cap in (
        (sm.fia, 0.095),
        (sm.iul, 0.11),
        (lambda c: sm.rila(cap=c), 0.18),
    ):
        budget = sm.price_product(builder(cap), MARKET).pv
        solved = sm.solve_cap_from_budget(builder, budget, MARKET)
        assert solved == pytest.approx(cap, abs=1e-7)
        assert sm.price_product(builder(solved), MARKET).pv == pytest.approx(
            budget, abs=1e-8
        )


def test_cap_solver_rejects_unaffordable_budget() -> None:
    with pytest.raises(ValueError):
        sm.solve_cap_from_budget(sm.fia, 50.0, MARKET)


def test_delta_hedge_neutralises_delta_and_reduces_stress_range() -> None:
    products = (sm.rila(), sm.fia(), sm.iul())
    hedge = sm.delta_hedge(products, MARKET)
    liability_delta = sm.portfolio_greeks(products, MARKET).delta
    assert hedge.futures_units * MARKET.forward_factor() == pytest.approx(
        liability_delta
    )
    before = sm.stress_table(products, MARKET)
    after = sm.stress_table(products, MARKET, hedge)
    assert all(row.hedge_pnl == 0.0 for row in before)
    assert sm.pnl_range(after, hedged=True) < 0.25 * sm.pnl_range(before, hedged=False)
    for row in after:
        if row.scenario.startswith("Index"):
            assert abs(row.hedged_pnl) < abs(row.unhedged_pnl)


def test_run_simple_model_is_consistent() -> None:
    results = sm.run_simple_model()
    assert results.portfolio.price == pytest.approx(
        sum(v.pv for v in results.valuations)
    )
    assert results.hedged_range < results.unhedged_range
    assert set(results.solved_caps) == {"RILA", "FIA", "IUL"}
    text = sm.recommendation(results)
    assert "futures" in text and "Illustrative" in text
