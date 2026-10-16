"""Regression tests derived from the independent model validation.

Each test pins a benchmark or a corrected behaviour documented in
``MODEL_VALIDATION_REPORT.md`` using the reference implementations in
``validation/reference.py`` (which do not import engine pricing code).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

import alm_hedging as ah
from alm_hedging import api
from alm_hedging.market import MarketState

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation import reference as ref  # noqa: E402

FLAT = (0.03, 0.01, 0.20)


def test_black_scholes_matches_quadrature_and_black76() -> None:
    for spot, strike, t, r, q, v, kind in [
        (100.0, 100.0, 1.0, 0.03, 0.01, 0.20, "call"),
        (100.0, 80.0, 0.25, 0.05, 0.0, 0.35, "put"),
        (5250.0, 5250.0, 1.0, 0.0425, 0.015, 0.185, "call"),
    ]:
        price = ah.black_scholes_price(spot, strike, t, r, q, v, kind)  # type: ignore[arg-type]
        assert price == pytest.approx(
            ref.bs_quadrature(spot, strike, t, r, q, v, kind), rel=1e-9
        )
        assert price == pytest.approx(
            ref.bs_via_black76(spot, strike, t, r, q, v, kind), rel=1e-12
        )


def test_rila_replication_matches_quadrature_and_decomposition(
    market: MarketState,
) -> None:
    r, q, v = FLAT
    contract = ah.RILABufferedSegment(100.0, 4.5, 0.10, 0.15, 1.08, -0.10)
    actual = ah.analytic_liability_value(contract, market)
    expected = ref.rila_buffered_call_put_decomposition(
        100.0, 100.0, 100.0, 4.5, r, q, v, 0.10, 0.15, 1.08, -0.10
    )
    assert actual == pytest.approx(expected, abs=1e-10)
    quad = ref.terminal_liability_quad(
        lambda x: ref.rila_buffered_credit(x, 0.10, 0.15, 1.08, -0.10),
        100.0,
        100.0,
        100.0,
        4.5,
        r,
        q,
        v,
        [80.0, 90.0, 100.0, 100.0 * (1.0 + 0.15 / 1.08)],
    )
    assert actual == pytest.approx(quad, abs=1e-9)


def test_monte_carlo_reprices_vanilla_on_sloped_term_structure() -> None:
    """Forward variance/forward rates make MC consistent with the analytic surface."""
    sloped = ah.SyntheticMarketModel(skew=0.0).build()
    for t in (1.0, 5.0):
        analytic = ah.market_option_price(sloped, 100.0, t, "call")
        _, paths = ah.simulate_gbm_paths(sloped, t, 200_000, 12, seed=1)
        mc = (
            float(sloped.rate_curve.discount(t))
            * np.maximum(paths[:, -1] - 100.0, 0.0).mean()
        )
        assert mc == pytest.approx(analytic, rel=5e-3)
        log_var = float(np.var(np.log(paths[:, -1] / 100.0)))
        assert log_var == pytest.approx(sloped.volatility(100.0, t) ** 2 * t, rel=0.02)
        forward = 100.0 * float(
            sloped.dividend_curve.discount(t) / sloped.rate_curve.discount(t)
        )
        assert float(np.mean(paths[:, -1])) == pytest.approx(forward, rel=2e-3)


def test_fia_ratchet_matches_independent_closed_form(market: MarketState) -> None:
    r, q, v = FLAT
    contract = ah.FIAAnnualPointToPoint(100.0, 5.0, 0.095, 0.85)
    closed = ref.cliquet_closed_form(
        100.0, 5, 0.095, 0.0, 0.85, 0.0, 100.0, 100.0, r, q, v
    )
    mc = np.mean(
        [
            ah.monte_carlo_value(contract, market, n_paths=50_000, seed=s)
            for s in range(3)
        ]
    )
    assert mc == pytest.approx(closed, rel=5e-4)


def test_iul_matches_deterministic_expected_value_recursion(
    market: MarketState,
) -> None:
    r, q, v = FLAT
    mortality, lapse = (0.003, 0.0032, 0.0035), (0.035, 0.03, 0.025)
    contract = ah.IULAccountValue(
        85_000.0, 5.0, 0.11, 1.0, 0.0, 0.0125, 8.0, mortality, lapse, 1.5
    )
    expected = ref.iul_expected_pv(
        85_000.0,
        5,
        0.11,
        0.0,
        1.0,
        0.0125,
        8.0,
        mortality,
        lapse,
        1.5,
        100.0,
        100.0,
        r,
        q,
        v,
    )
    mc = np.mean(
        [
            ah.monte_carlo_value(contract, market, n_paths=50_000, seed=s)
            for s in range(3)
        ]
    )
    assert mc == pytest.approx(expected, rel=1e-3)


def test_finite_difference_greeks_match_analytic(market: MarketState) -> None:
    r, q, v = FLAT
    greeks = ah.bump_and_revalue(
        lambda s: ah.market_option_price(s, 100.0, 1.0), market
    )
    exact = ref.bs_analytic_greeks(100.0, 100.0, 1.0, r, q, v)
    assert greeks.delta == pytest.approx(exact["delta"], abs=2e-4)
    assert greeks.gamma == pytest.approx(exact["gamma"], abs=1e-5)
    assert greeks.vega == pytest.approx(exact["vega"], abs=2e-3)
    assert greeks.rho == pytest.approx(exact["rho"], abs=1e-5)
    assert greeks.theta == pytest.approx(exact["theta"], abs=5e-3)
    assert greeks.dv01 == pytest.approx(exact["rho"] * 1e-4, abs=1e-6)


def test_skewed_surface_delta_follows_sticky_moneyness_convention() -> None:
    skewed = ah.SyntheticMarketModel(term_vol=0.0).build()
    vol = skewed.volatility(100.0, 1.0)
    exact = ref.bs_analytic_greeks(
        100.0,
        100.0,
        1.0,
        float(skewed.rate_curve.zero_rate(1.0)),
        float(skewed.dividend_curve.zero_rate(1.0)),
        vol,
    )
    greeks = ah.bump_and_revalue(
        lambda s: ah.market_option_price(s, 100.0, 1.0), skewed
    )
    assert greeks.delta == pytest.approx(
        exact["delta"] + exact["vega"] * 0.08 / 100.0, abs=5e-4
    )


def test_lot_rounding_with_binding_premium_budget_matches_brute_force() -> None:
    instruments = [
        ah.HedgeInstrument(
            "fut",
            "future",
            1.0,
            100.0,
            {"delta": 1.0},
            max_position=200,
            max_trade=200,
            lot_size=5,
            transaction_cost=0.5,
        ),
        ah.HedgeInstrument(
            "call",
            "option",
            1.0,
            8.0,
            {"delta": 0.6, "vega": 40.0},
            premium=8.0,
            max_position=60,
            max_trade=60,
            lot_size=2,
            transaction_cost=0.2,
        ),
        ah.HedgeInstrument(
            "put",
            "option",
            1.0,
            6.0,
            {"delta": -0.4, "vega": 35.0},
            premium=6.0,
            max_position=60,
            max_trade=60,
            lot_size=2,
            transaction_cost=0.2,
        ),
    ]
    solution = ah.optimize_hedge(
        {"delta": -80.0, "vega": -1500.0},
        instruments,
        exposure_weights={"delta": 1.0, "vega": 0.02},
        constraints=ah.HedgeConstraints(premium_budget=250.0, turnover_budget=1e9),
        cost_penalty=1e-3,
    )
    m = np.array([[1.0, 0.6, -0.4], [0.0, 40.0, 35.0]])
    t = np.array([-80.0, -1500.0])
    w = np.array([1.0, 0.02])
    premiums = np.array([0.0, 8.0, 6.0])
    costs = np.array([0.5, 0.2, 0.2])
    _, best = ref.brute_force_lot_hedge(
        lambda qty: float(
            np.sum((w * (t + m @ qty)) ** 2) + 1e-3 * np.dot(np.abs(qty), costs)
        ),
        lambda qty: bool(np.dot(np.abs(qty), premiums) <= 250.0 + 1e-9),
        [(-200, 200), (-60, 60), (-60, 60)],
        [5, 2, 2],
    )
    assert solution.success
    assert solution.premium_used <= 250.0 + 1e-9
    assert solution.objective == pytest.approx(best, abs=1e-9)


def test_dynamic_attribution_explains_gamma_and_theta_on_delta_hedge() -> None:
    def flat(spot: float) -> MarketState:
        tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
        k = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
        return MarketState(
            spot,
            ah.ZeroCurve(tenors, np.full_like(tenors, 0.03)),
            ah.ZeroCurve(tenors, np.full_like(tenors, 0.01)),
            ah.VolSurface(tenors, k, np.full((tenors.size, k.size), 0.20)),
        )

    rng = np.random.RandomState(7)
    dt = 1.0 / 250.0
    spots = np.r_[
        100.0,
        100.0
        * np.exp(
            np.cumsum(
                (0.05 - 0.02) * dt + 0.2 * math.sqrt(dt) * rng.standard_normal(60)
            )
        ),
    ]
    markets = [flat(float(s)).bumped(time_years=i * dt) for i, s in enumerate(spots)]

    def liability(state: MarketState) -> float:
        return 1000.0 * ah.market_option_price(state, 100.0, 1.0, "call")

    future = ah.DynamicHedgeInstrument(
        "fut",
        "future",
        1.0,
        value=lambda s: s.spot,
        risks=lambda s: {"delta": 1.0},
        transaction_cost=0.01,
    )
    result = ah.simulate_dynamic_hedge(
        markets,
        liability,
        lambda s: ah.bump_and_revalue(liability, s),
        [future],
        risk_targets=("delta",),
    )
    h = result.history
    assert h.iloc[0]["position_fut"] > 0.0
    components = (
        h["delta_pnl"]
        + h["gamma_pnl"]
        + h["vega_pnl"]
        + h["rho_pnl"]
        + h["theta_pnl"]
        + h["residual_pnl"]
    )
    np.testing.assert_allclose(
        components - h["transaction_cost"], h["total_pnl"], atol=1e-9
    )
    assert h["theta_pnl"].abs().sum() > 0.0 and h["gamma_pnl"].abs().sum() > 0.0
    assert h["residual_pnl"].abs().sum() < 0.05 * h["liability_pnl"].abs().sum()
    assert h["transaction_cost"].sum() == pytest.approx(
        0.01 * np.abs(np.diff(np.r_[0.0, h["position_fut"].values])).sum()
    )


def _dashboard_inputs() -> tuple[list[dict[str, object]], dict[str, object]]:
    cohorts = [
        {
            "product": "RILA",
            "account_value": 200e6,
            "policies": 10_000,
            "duration": 4.5,
            "participation": 1.08,
            "cap": 0.15,
            "floor": -0.10,
            "guarantee": 0.02,
        },
        {
            "product": "FIA",
            "account_value": 175e6,
            "policies": 8_750,
            "duration": 5.1,
            "participation": 0.72,
            "cap": 0.09,
            "floor": 0.0,
            "guarantee": 0.025,
        },
        {
            "product": "IUL",
            "account_value": 125e6,
            "policies": 6_250,
            "duration": 5.7,
            "participation": 0.92,
            "cap": 0.11,
            "floor": 0.0,
            "guarantee": 0.03,
        },
    ]
    market = {
        "spot": 5250.0,
        "rate": 0.0425,
        "volatility": 0.185,
        "dividend_yield": 0.015,
        "as_of": "2026-10-16",
    }
    return cohorts, market


def test_dashboard_contracts_are_struck_at_valuation_spot() -> None:
    cohorts, market = _dashboard_inputs()
    state = api._market(market)
    for cohort in api._liabilities(cohorts, state.spot):
        assert cohort.contract.index_start == pytest.approx(5250.0)  # type: ignore[attr-defined]
    greeks = api.calculate_greeks(cohorts, market)
    delta = float(greeks.loc[greeks["Greek"] == "Delta", "Exposure"].iloc[0])
    assert abs(delta) > 1_000.0  # was ~1e-8 with the index_start=100 defect


def test_dashboard_hedge_has_same_sign_greeks_as_liability() -> None:
    cohorts, market = _dashboard_inputs()
    greeks = api.calculate_greeks(cohorts, market)
    risks = dict(zip(greeks["Greek"].str.lower(), greeks["Exposure"], strict=True))
    hedge = api.optimize_hedge(
        cohorts,
        market,
        0.0075 * 500e6,
        0.15,
        ["Equity futures", "ATM calls", "Put spread"],
    )
    templates = {
        "Equity futures": {"delta": 50.0},
        "ATM calls": {"delta": 55.0, "vega": 3500.0},
        "Put spread": {"delta": -25.0, "vega": 1800.0},
    }
    units = dict(
        zip(
            hedge["allocation"]["instrument"], hedge["allocation"]["units"], strict=True
        )
    )
    hedge_delta = sum(units[n] * templates[n].get("delta", 0.0) for n in units)
    hedge_vega = sum(units[n] * templates[n].get("vega", 0.0) for n in units)
    assert np.sign(hedge_delta) == np.sign(risks["delta"])
    assert np.sign(hedge_vega) == np.sign(risks["vega"])
    assert hedge["residual_vega"] == pytest.approx(risks["vega"] - hedge_vega, abs=1.0)
    assert hedge["premium"] <= 0.0075 * 500e6 + 1e-6
    assert 0.0 <= hedge["effectiveness"] <= 1.0
    assert "first-order" in hedge["effectiveness_basis"]


def test_dashboard_effectiveness_penalizes_untargeted_exposure() -> None:
    """A vega-only universe cannot report high effectiveness while creating delta."""
    cohorts, market = _dashboard_inputs()
    hedge = api.optimize_hedge(cohorts, market, 0.0075 * 500e6, 0.15, ["Variance swap"])
    assert (
        "delta" not in hedge["effectiveness_basis"]
    )  # variance swap has no delta template
    hedge = api.optimize_hedge(cohorts, market, 0.0075 * 500e6, 0.15, ["ATM calls"])
    assert (
        "delta" in hedge["effectiveness_basis"]
        and "vega" in hedge["effectiveness_basis"]
    )


def test_backtest_uses_engine_greeks_when_cohorts_supplied() -> None:
    cohorts, market = _dashboard_inputs()
    pricing = api.price_portfolio(cohorts, market)
    frame = api.backtest_hedge(pricing, market, "Monthly", 3.0, 1, cohorts=cohorts)
    assert len(frame) == 12
    assert frame["date"].is_monotonic_increasing
    assert np.isfinite(frame["net_pnl"]).all()
