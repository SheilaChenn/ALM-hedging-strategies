"""October 2026 market shock case study.

"Hedging Index-Linked Insurance Liabilities Amid Equity Selloffs and Treasury
Yield Volatility" is a scenario-driven case study built on the validated
pricing, risk, and hedge-optimization engine.  Every number shown in the PDF
(``scripts/build_case_study.py``) and in the dashboard's case-study section
comes from :func:`run_case_study` in this module, serialized once to JSON by
the build script and loaded by both consumers, so the two presentations
reconcile exactly.

Everything in this module is illustrative.  The documented market events in
:data:`DOCUMENTED_EVENTS` are sourced and dated; the shock sizes, the synthetic
market, the synthetic $500M portfolio, the backing asset ladder, the hedge
instruments, and the model-generated hedges are assumptions for a research
exercise.  They are not Pacific Life positions, hedges, or trading decisions.

Sign conventions follow the engine: liability Greeks are reported as changes in
liability value (positive = liability rises).  The insurer's *net exposure* is
asset Greeks minus liability Greeks, i.e. the P&L sensitivity of the surplus,
and the optimizer is asked to drive ``net + hedge`` to zero.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .contracts import (
    Cohort,
    FIAAnnualPointToPoint,
    IULAccountValue,
    RILABufferedSegment,
)
from .market import MarketState, SyntheticMarketModel
from .optimization import HedgeConstraints, HedgeInstrument, HedgeSolution
from .optimization import optimize_hedge as solve
from .pricing import market_option_price, portfolio_value
from .risk import Greeks, bump_and_revalue
from .stress import StressScenario

CASE_STUDY_TITLE = (
    "Hedging Index-Linked Insurance Liabilities Amid Equity Selloffs "
    "and Treasury Yield Volatility"
)
CASE_STUDY_SHORT_TITLE = "October 2026 Market Shock Case Study"
RESULTS_FILENAME = "october-2026-market-shock-results.json"
PDF_FILENAME = "october-2026-market-shock-case-study.pdf"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "docs" / "case_study"

ILLUSTRATIVE_NOTICE = (
    "Illustrative research exercise. The simulated shocks, the synthetic $500M "
    "portfolio, the backing assets, the hedge instruments, and every "
    "model-generated hedge are assumptions, not Pacific Life positions, hedges, "
    "or trading decisions. Documented market events are listed separately with "
    "their sources."
)

# --------------------------------------------------------------------------- #
# Documented market backdrop (sourced) versus illustrative assumptions
# --------------------------------------------------------------------------- #

DOCUMENTED_EVENTS: tuple[dict[str, str], ...] = (
    {
        "date": "2026-09-16",
        "summary": "FOMC hikes 25 bp to 3.75%-4.00%, first increase in over three years",
        "event": (
            "FOMC raises the federal funds target range by 25 bp to 3.75%-4.00%, "
            "the first increase in more than three years, citing elevated inflation."
        ),
        "source": "Federal Reserve press release",
        "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260916a.htm",
    },
    {
        "date": "2026-10-07",
        "summary": "10y yield hits 5.364% (24-year high), 30y also a 24-year high; oil above $100",
        "event": (
            "10-year Treasury yield touches 5.364%, a 24-year high, and the 30-year "
            "also sets a 24-year high as oil trades above $100; a strong $39bn "
            "10-year auction pulls yields off the highs."
        ),
        "source": "Reuters",
        "url": "https://www.reuters.com/business/us-30-year-bond-yield-hits-fresh-24-year-high-2026-10-07/",
    },
    {
        "date": "2026-10-14",
        "summary": "Sept CPI +0.6% m/m, 3.7% y/y (core 2.6%); Oct-28 hike odds rise to ~35-54%",
        "event": (
            "September CPI +0.6% m/m and 3.7% y/y (consensus 3.6%); core CPI 2.6% "
            "y/y. CME FedWatch odds of a 28 October hike rise to roughly 35%-54%."
        ),
        "source": "BLS via CNBC; Reuters",
        "url": "https://www.cnbc.com/2026/10/14/cpi-inflation-report-september-2026.html",
    },
    {
        "date": "2026-10-14",
        "summary": "S&P 500 -1.1%, Nasdaq -1.9%: worst day since July; AI names lead",
        "event": (
            "S&P 500 -1.1% and Nasdaq -1.9%, the worst day since July and the fifth "
            "loss in six sessions since the record high; AI-linked names lead."
        ),
        "source": "AP News",
        "url": "https://apnews.com/article/stock-markets-earnings-bonds-artificial-intelligence-704d3190f0fba2908df07c85f26346ec",
    },
    {
        "date": "2026-10-14",
        "summary": "Semis index -4.3% on the day, ~-12% in a week; NVDA -3.2%, ORCL -8.8%, DELL -11.3%",
        "event": (
            "Semiconductor index -4.3% on the day and about -12% over the past "
            "week; Nvidia -3.2%, Oracle -8.8%, Dell -11.3%, Marvell -8.7%."
        ),
        "source": "CNN Business",
        "url": "https://www.cnn.com/2026/10/14/investing/us-stocks-ai-chipmakers",
    },
    {
        "date": "2026-10-14",
        "summary": "10y yield above 5.3% intraday after CPI, closes near 5.19%; 2y ~4.81%",
        "event": (
            "10-year yield crests above 5.3% intraday after the CPI release and "
            "closes near 5.19%; the 2-year trades near 4.81%."
        ),
        "source": "CNBC",
        "url": "https://www.cnbc.com/2026/10/14/us-treasury-yields-cpi-inflation-report.html",
    },
    {
        "date": "2026-10-15",
        "summary": "Nasdaq near one-month low (>4% off record); VIX 17.74 two-week high; SOXX 8th straight loss",
        "event": (
            "Nasdaq near a one-month low, more than 4% below the prior week's "
            "record; VIX hits a two-week high of 17.74; the iShares Semiconductor "
            "ETF falls for an eighth straight session (about -12% over seven)."
        ),
        "source": "Reuters via Kitco; MarketWatch via Morningstar",
        "url": "https://www.kitco.com/news/off-the-wire/2026-10-15/nasdaq-slides-near-one-month-low-ai-rout-deepens",
    },
    {
        "date": "2026-10-16",
        "summary": "CoreWeave 5y CDS >1,140 bp, bonds >15%; >$1tn erased from 20 AI/semis stocks; PPI +6.3% y/y",
        "event": (
            "CoreWeave five-year CDS briefly above 1,140 bp with bonds yielding "
            "above 15%; more than $1 trillion of market value erased from 20 "
            "large AI and semiconductor stocks over the week; September PPI "
            "+6.3% y/y."
        ),
        "source": "CNBC Daily Open",
        "url": "https://www.cnbc.com/2026/10/16/cnbc-daily-open-ai-inflation-coreweave-.html",
    },
    {
        "date": "2026-10-16",
        "summary": "S&P 500 heads for worst week since June, Nasdaq >-4%; 10y quoted 5.07%-5.26%",
        "event": (
            "S&P 500 heads for its steepest weekly loss since early June and the "
            "Nasdaq for a weekly loss above 4%; the 10-year yield is quoted "
            "between about 5.07% and 5.26% across sources on 15-16 October, "
            "20-30 bp below the 7 October peak."
        ),
        "source": "Reuters via StreetInsider; MarketWatch; Trading Economics (FRED DGS10 is the primary series)",
        "url": "https://www.streetinsider.com/Market+Check/Wall+Street+tumbles+as+worries+about+AI+trade+and+high+oil+prices+weigh/27195576.html",
    },
)

ILLUSTRATIVE_ASSUMPTIONS: tuple[str, ...] = (
    "Synthetic broad U.S. equity index at 7,750 on the illustrative issue date "
    "(6 October 2026). The level is in the neighbourhood of the documented S&P 500 "
    "prints of ~7,700-7,750 on 15-16 October but the index is synthetic.",
    "Pre-shock market: 1y ATM implied volatility 19% (the documented VIX, a 30-day "
    "measure, was 16-18); zero curve 4.75% + 0.70% x (1 - exp(-t/5)), anchoring the "
    "10y near the documented 5.36% peak (2y 4.98%, 10y 5.36%); dividend yield 1.2%.",
    "Illustrative shock applied instantaneously to the block: equity -12% (the "
    "documented seven-session semiconductor-index drawdown applied to the broad "
    "index as an escalation scenario; the documented broad-market drawdown was "
    "roughly 2-4%), ATM implied volatility +8 points to 27% (the documented VIX "
    "move was about +2 points), parallel zero-curve shift -25 bp (inside the "
    "documented 5.36% -> 5.07%-5.19% move).",
    "Synthetic $500M block of newly issued RILA, FIA, and IUL segments struck at "
    "7,750, backed by a Treasury strip ladder whose market value equalled the "
    "liability PV at issue. The index-linked credits are not yet hedged at the "
    "valuation date; the case study is the hedge-design decision at the stressed "
    "market.",
    "Forward-looking scenario distribution for hedge design and evaluation "
    "(roughly a one-month horizon in the stressed regime): Student-t(4) index "
    "shocks with 7.8% standard deviation, implied-volatility shocks of -0.30 x "
    "index return plus N(0, 2.5 pts) noise (clipped to [-15, +40] points), and "
    "independent N(0, 35 bp) parallel rate shocks. In-sample and out-of-sample "
    "sets use different seeds.",
    "Hedge budget 0.75% of account value ($3.75M) of gross option premium "
    "(premium received on sold options is counted as consumed, the engine's "
    "conservative convention); liquidity budget $10M of cash for initial margin "
    "and premium (equity futures 10% of notional, Treasury futures 2.5% of face, "
    "options at premium); position caps per instrument; transaction costs "
    "(futures $15, Treasury futures $25, options 1.5% of premium per contract).",
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CaseStudyConfig:
    """All inputs of the case study. Defaults reproduce the published results."""

    as_of: str = "2026-10-16"
    issue_date: str = "2026-10-06"
    issue_index: float = 7_750.0
    base_rate: float = 0.0475
    rate_slope: float = 0.0070
    dividend_yield: float = 0.012
    atm_vol: float = 0.18
    skew: float = -0.08
    term_vol: float = 0.01
    equity_shock: float = -0.12
    vol_shock: float = 0.08
    rate_shock_bp: float = -25.0
    ladder_tenors: tuple[float, ...] = (1.0, 3.0, 5.0, 7.0, 9.0)
    premium_budget_pct: float = 0.0075
    liquidity_budget: float = 10_000_000.0
    futures_margin_pct: float = 0.10
    treasury_margin_pct: float = 0.025
    option_cost_pct: float = 0.015
    futures_cost: float = 15.0
    treasury_cost: float = 25.0
    max_futures: float = 600.0
    max_options: float = 300.0
    max_put_spreads: float = 600.0
    max_treasury: float = 3_500.0
    n_paths: int = 3_000
    pricing_seed: int = 20_261_016
    in_sample_scenarios: int = 300
    out_of_sample_scenarios: int = 800
    in_sample_seed: int = 101
    out_of_sample_seed: int = 202
    spot_scale: float = 0.055
    spot_df: float = 4.0
    vol_beta: float = -0.30
    vol_noise: float = 0.025
    rate_sd_bp: float = 35.0
    rate_spot_corr: float = 0.0
    robustness_scenarios: int = 200
    robustness_seed: int = 303
    cvar_weight_scale: float = 0.5
    cvar_alpha: float = 0.95
    cost_penalty: float = 1.0e-8
    carry_horizon_years: float = 1.0 / 12.0

    @property
    def account_value(self) -> float:
        return sum(c["policies"] * c["account_value_per_policy"] for c in COHORT_SPECS)

    @property
    def premium_budget(self) -> float:
        return self.premium_budget_pct * self.account_value


COHORT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "product": "RILA",
        "description": "6y buffered segment: 10% buffer, 18% cap, 100% participation, -10% floor",
        "policies": 2_000,
        "account_value_per_policy": 100_000.0,
        "term_years": 6.0,
        "cap": 0.18,
        "participation": 1.0,
        "buffer": 0.10,
        "floor": -0.10,
    },
    {
        "product": "FIA",
        "description": "10y annual point-to-point: 9.5% annual cap, 85% participation, 0% floor",
        "policies": 2_500,
        "account_value_per_policy": 80_000.0,
        "term_years": 10.0,
        "cap": 0.095,
        "participation": 0.85,
        "buffer": 0.0,
        "floor": 0.0,
    },
    {
        "product": "IUL",
        "description": "20y indexed account: 11% annual cap, 100% participation, 0% floor, 1.25% charge, 3.5% lapse, 0.3% mortality",
        "policies": 1_250,
        "account_value_per_policy": 80_000.0,
        "term_years": 20.0,
        "cap": 0.11,
        "participation": 1.0,
        "buffer": 0.0,
        "floor": 0.0,
    },
)

NAMED_STRESSES: tuple[StressScenario, ...] = (
    StressScenario("Equity -20%", equity_relative=-0.20),
    StressScenario("Equity +20%", equity_relative=0.20),
    StressScenario("Implied vol +10 pts", vol_absolute=0.10),
    StressScenario("Rates -100 bp", rate_bp=-100.0),
    StressScenario("Rates +100 bp", rate_bp=100.0),
    StressScenario(
        "AI selloff escalation (-15%, +10 vol, -40 bp)",
        equity_relative=-0.15,
        vol_absolute=0.10,
        rate_bp=-40.0,
    ),
    StressScenario(
        "Inflation scare (-8%, +5 vol, +50 bp)",
        equity_relative=-0.08,
        vol_absolute=0.05,
        rate_bp=50.0,
    ),
    StressScenario(
        "Relief rally (+10%, -6 vol, +25 bp)",
        equity_relative=0.10,
        vol_absolute=-0.06,
        rate_bp=25.0,
    ),
    StressScenario(
        "Crash (-40%, +25 vol, -75 bp)",
        equity_relative=-0.40,
        vol_absolute=0.25,
        rate_bp=-75.0,
    ),
)

STRATEGY_ORDER = ("delta_only", "multi_greek", "stress_aware")
STRATEGY_LABELS = {
    "delta_only": "Delta-only (index futures)",
    "multi_greek": "Multi-Greek (delta, vega, rho)",
    "stress_aware": "Stress-aware constrained (scenario P&L + CVaR)",
}


# --------------------------------------------------------------------------- #
# Market, portfolio, and assets
# --------------------------------------------------------------------------- #


def build_markets(config: CaseStudyConfig) -> tuple[MarketState, MarketState]:
    """Return the pre-shock market (issue date) and the stressed valuation market."""
    pre = SyntheticMarketModel(
        base_rate=config.base_rate,
        rate_slope=config.rate_slope,
        dividend_yield=config.dividend_yield,
        atm_vol=config.atm_vol,
        skew=config.skew,
        term_vol=config.term_vol,
    ).build(spot=config.issue_index)
    post = pre.bumped(
        spot_relative=config.equity_shock,
        vol_absolute=config.vol_shock,
        rate_bp=config.rate_shock_bp,
    )
    return pre, post


def build_cohorts(config: CaseStudyConfig) -> list[Cohort]:
    """Synthetic RILA, FIA, and IUL cohorts struck at the issue index level."""
    cohorts: list[Cohort] = []
    for spec in COHORT_SPECS:
        notional = float(spec["account_value_per_policy"])
        term = float(spec["term_years"])
        if spec["product"] == "RILA":
            contract: Any = RILABufferedSegment(
                notional,
                term,
                buffer=float(spec["buffer"]),
                cap=float(spec["cap"]),
                participation=float(spec["participation"]),
                floor=float(spec["floor"]),
                index_start=config.issue_index,
            )
        elif spec["product"] == "FIA":
            contract = FIAAnnualPointToPoint(
                notional,
                term,
                annual_cap=float(spec["cap"]),
                participation=float(spec["participation"]),
                annual_floor=float(spec["floor"]),
                index_start=config.issue_index,
            )
        else:
            contract = IULAccountValue(
                notional,
                term,
                annual_cap=float(spec["cap"]),
                participation=float(spec["participation"]),
                annual_floor=float(spec["floor"]),
                annual_policy_charge=0.0125,
                monthly_expense_charge=8.0,
                mortality_rates=(0.003,),
                lapse_rates=(0.035,),
                index_start=config.issue_index,
            )
        cohorts.append(Cohort(contract, int(spec["policies"]), label=spec["product"]))
    return cohorts


@dataclass(frozen=True)
class StripLadder:
    """Zero-coupon Treasury strip ladder backing the block."""

    tenors: tuple[float, ...]
    faces: tuple[float, ...]

    def value(self, market: MarketState) -> float:
        remaining = np.maximum(np.asarray(self.tenors) - market.valuation_time, 0.0)
        return float(
            np.sum(
                np.asarray(self.faces)
                * np.asarray(market.rate_curve.discount(remaining))
            )
        )

    def duration(self, market: MarketState) -> float:
        remaining = np.maximum(np.asarray(self.tenors) - market.valuation_time, 0.0)
        weights = np.asarray(self.faces) * np.asarray(
            market.rate_curve.discount(remaining)
        )
        return float(np.sum(weights * remaining) / np.sum(weights))


def build_ladder(
    config: CaseStudyConfig, pre_shock: MarketState, liability_pv_at_issue: float
) -> StripLadder:
    """Equal-market-value strips whose total equals the liability PV at issue."""
    tenors = np.asarray(config.ladder_tenors)
    per_bucket = liability_pv_at_issue / tenors.size
    faces = per_bucket / np.asarray(pre_shock.rate_curve.discount(tenors))
    return StripLadder(tuple(float(t) for t in tenors), tuple(float(f) for f in faces))


# --------------------------------------------------------------------------- #
# Hedge instruments
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CaseInstrument:
    """Market-priced hedge instrument with full-revaluation value function."""

    key: str
    name: str
    kind: str
    description: str
    multiplier: float
    value: Callable[[MarketState], float]
    liquidity_per_contract: float
    transaction_cost: float
    max_position: float
    premium_per_contract: float
    strike: float | None = None
    tenor: float = 1.0


def build_instruments(
    config: CaseStudyConfig, market: MarketState
) -> list[CaseInstrument]:
    """Liquid instruments priced from the stressed valuation market."""
    spot = market.spot
    issue = config.issue_index
    atm_strike = round(spot / 25.0) * 25.0
    ps_upper = issue * (1.0 + COHORT_SPECS[0]["floor"])
    ps_lower = issue * (1.0 + COHORT_SPECS[0]["floor"] - COHORT_SPECS[0]["buffer"])

    def future(state: MarketState) -> float:
        tau = max(1.0 - state.valuation_time, 0.0)
        carry = float(state.dividend_curve.discount(tau)) / float(
            state.rate_curve.discount(tau)
        )
        return 50.0 * state.spot * carry

    def atm_call(state: MarketState) -> float:
        return 100.0 * market_option_price(state, atm_strike, 1.0, "call")

    def issue_call(state: MarketState) -> float:
        return 100.0 * market_option_price(state, issue, 1.0, "call")

    def put_spread(state: MarketState) -> float:
        return 100.0 * (
            market_option_price(state, ps_upper, 1.0, "put")
            - market_option_price(state, ps_lower, 1.0, "put")
        )

    def treasury(state: MarketState) -> float:
        return 100_000.0 * float(
            state.rate_curve.discount(max(10.0 - state.valuation_time, 0.0))
        )

    future_value = future(market)
    atm_value = atm_call(market)
    issue_value = issue_call(market)
    ps_value = put_spread(market)
    return [
        CaseInstrument(
            "future",
            "Index futures (1y, 50x)",
            "future",
            "One-year index future, 50 index multiplier; margin-only, no premium",
            50.0,
            future,
            config.futures_margin_pct * future_value,
            config.futures_cost,
            config.max_futures,
            0.0,
        ),
        CaseInstrument(
            "atm_call",
            f"ATM call K={atm_strike:,.0f} (1y, 100x)",
            "option",
            "One-year listed index call struck at the stressed spot",
            100.0,
            atm_call,
            atm_value,
            config.option_cost_pct * atm_value,
            config.max_options,
            atm_value,
            atm_strike,
        ),
        CaseInstrument(
            "issue_call",
            f"OTM call K={issue:,.0f} (1y, 100x)",
            "option",
            "One-year index call struck at the issue level (the liability's crediting strike)",
            100.0,
            issue_call,
            issue_value,
            config.option_cost_pct * issue_value,
            config.max_options,
            issue_value,
            issue,
        ),
        CaseInstrument(
            "put_spread",
            f"Put spread {ps_upper:,.0f}/{ps_lower:,.0f} (1y, 100x)",
            "option",
            "One-year put spread at the RILA buffer and floor strikes",
            100.0,
            put_spread,
            ps_value,
            config.option_cost_pct * ps_value * 4.0 / 3.0,
            config.max_put_spreads,
            ps_value,
            ps_upper,
        ),
        CaseInstrument(
            "treasury",
            "10y Treasury futures proxy ($100k face)",
            "swap",
            "Ten-year zero-coupon Treasury exposure per $100k face; margin-only",
            100_000.0,
            treasury,
            config.treasury_margin_pct * 100_000.0,
            config.treasury_cost,
            config.max_treasury,
            0.0,
            None,
            10.0,
        ),
    ]


# --------------------------------------------------------------------------- #
# Scenario generation and effectiveness metrics
# --------------------------------------------------------------------------- #


def generate_scenarios(
    config: CaseStudyConfig,
    n: int,
    seed: int,
    *,
    spot_scale: float | None = None,
    vol_beta: float | None = None,
    vol_noise: float | None = None,
    rate_sd_bp: float | None = None,
    rate_spot_corr: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint (spot return, vol shock, rate shock bp) draws for the stressed regime.

    Keyword overrides let the robustness analysis re-draw from alternative
    distributions while keeping the base configuration intact.
    """
    scale = config.spot_scale if spot_scale is None else spot_scale
    beta = config.vol_beta if vol_beta is None else vol_beta
    noise = config.vol_noise if vol_noise is None else vol_noise
    rate_sd = config.rate_sd_bp if rate_sd_bp is None else rate_sd_bp
    corr = config.rate_spot_corr if rate_spot_corr is None else rate_spot_corr
    rng = np.random.default_rng(seed)
    spot = rng.standard_t(config.spot_df, n) * scale
    spot = np.clip(spot, -0.60, 1.00)
    vol = beta * spot + rng.normal(0.0, noise, n)
    vol = np.clip(vol, -0.15, 0.40)
    standardized_spot = (spot - np.mean(spot)) / max(float(np.std(spot)), 1e-12)
    rate = rate_sd * (
        corr * standardized_spot + math.sqrt(1.0 - corr**2) * rng.standard_normal(n)
    )
    return spot, vol, rate


ROBUSTNESS_SETS: tuple[dict[str, Any], ...] = (
    {
        "key": "no_vol_correlation",
        "name": "No vol-spot correlation",
        "description": "Implied-vol shocks independent of the index (beta 0, noise 3.5 pts)",
        "vol_beta": 0.0,
        "vol_noise": 0.035,
    },
    {
        "key": "strong_vol_correlation",
        "name": "Strong vol-spot correlation",
        "description": "Implied-vol beta -0.60 to the index return (noise 2.0 pts)",
        "vol_beta": -0.60,
        "vol_noise": 0.020,
    },
    {
        "key": "flight_to_quality",
        "name": "Flight-to-quality rates",
        "description": "Rate shocks 60 bp sd, +0.5 correlation with the index (yields fall when equities fall)",
        "rate_sd_bp": 60.0,
        "rate_spot_corr": 0.5,
    },
    {
        "key": "inflation_regime",
        "name": "Inflation-scare rates",
        "description": "Rate shocks 60 bp sd, -0.5 correlation with the index (yields rise when equities fall)",
        "rate_sd_bp": 60.0,
        "rate_spot_corr": -0.5,
    },
)


def scenario_metrics(
    unhedged_pnl: np.ndarray, hedge_pnl: np.ndarray
) -> dict[str, float]:
    """Scenario-based effectiveness in the insurer's P&L sign.

    ``unhedged_pnl`` is the surplus P&L before hedging and ``hedge_pnl`` the
    hedge asset gain in the same scenarios.  Variance reduction is
    1 - Var(unhedged + hedge) / Var(unhedged); the dollar-offset slope is the
    regression slope of hedge P&L on the insurer's loss (-unhedged); ES95 is
    the mean of the worst 5% of losses.
    """
    loss = -np.asarray(unhedged_pnl, dtype=float)
    gain = np.asarray(hedge_pnl, dtype=float)
    net_loss = loss - gain
    var_l = float(np.var(loss))
    var_n = float(np.var(net_loss))
    slope = float(np.cov(gain, loss)[0, 1] / var_l) if var_l > 0.0 else 0.0
    return {
        "variance_reduction": 1.0 - var_n / var_l if var_l > 0.0 else 0.0,
        "dollar_offset_slope": slope,
        "unhedged_sd": math.sqrt(var_l),
        "hedged_sd": math.sqrt(var_n),
        "es95_unhedged": expected_shortfall(loss, 0.95),
        "es95_hedged": expected_shortfall(net_loss, 0.95),
        "es95_reduction": (
            1.0 - expected_shortfall(net_loss, 0.95) / expected_shortfall(loss, 0.95)
            if expected_shortfall(loss, 0.95) != 0.0
            else 0.0
        ),
        "worst_unhedged_loss": float(np.max(loss)),
        "worst_hedged_loss": float(np.max(net_loss)),
    }


def expected_shortfall(losses: np.ndarray, alpha: float) -> float:
    """Mean of the worst ceil((1 - alpha) n) losses."""
    k = max(1, math.ceil((1.0 - alpha) * losses.size))
    return float(np.mean(np.sort(losses)[-k:]))


# --------------------------------------------------------------------------- #
# Results container
# --------------------------------------------------------------------------- #


@dataclass
class StrategyResult:
    """Positions, costs, residual risk, and effectiveness of one hedge strategy."""

    key: str
    label: str
    objective: str
    status: str
    trades: dict[str, float]
    notional: dict[str, float]
    premium_used: float
    liquidity_used: float
    transaction_cost: float
    hedge_greeks: dict[str, float]
    residual_greeks: dict[str, float]
    greek_effectiveness: float
    in_sample: dict[str, float]
    out_of_sample: dict[str, float]
    stress_pnl: dict[str, float]
    oos_hedged_pnl: list[float]
    cost_per_sd_removed: float


@dataclass
class CaseStudyResult:
    """Complete, serializable output of :func:`run_case_study`."""

    title: str
    short_title: str
    as_of: str
    issue_date: str
    engine_version: str
    notice: str
    config: dict[str, Any]
    market: dict[str, Any]
    portfolio: list[dict[str, Any]]
    assets: dict[str, Any]
    shock_pnl: dict[str, float]
    greeks: dict[str, dict[str, float]]
    instruments: list[dict[str, Any]]
    scenarios: dict[str, Any]
    unhedged_stress_pnl: dict[str, float]
    strategies: dict[str, StrategyResult]
    recommendation: dict[str, Any]
    robustness: list[dict[str, Any]]
    documented_events: list[dict[str, str]]
    illustrative_assumptions: list[str]
    limitations: list[str]
    oos_unhedged_pnl: list[float]
    oos_shocks: dict[str, list[float]]
    runtime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return json.loads(json.dumps(payload, default=_jsonable))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CaseStudyResult:
        data = dict(payload)
        data["strategies"] = {
            key: StrategyResult(**value) for key, value in data["strategies"].items()
        }
        return cls(**data)

    def strategy_frame(self) -> pd.DataFrame:
        """Three-way comparison table shared by the PDF and the dashboard."""
        return comparison_table(self)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


# --------------------------------------------------------------------------- #
# Core computation
# --------------------------------------------------------------------------- #


def _greeks_dict(greeks: Greeks) -> dict[str, float]:
    return {
        "pv": greeks.pv,
        "delta": greeks.delta,
        "gamma": greeks.gamma,
        "vega": greeks.vega,
        "rho": greeks.rho,
        "dv01": greeks.dv01,
    }


def _combine(
    a: Mapping[str, float], b: Mapping[str, float], sign: float
) -> dict[str, float]:
    return {k: float(a.get(k, 0.0)) + sign * float(b.get(k, 0.0)) for k in a}


def run_case_study(config: CaseStudyConfig = CaseStudyConfig()) -> CaseStudyResult:
    """Run the complete case study and return all numbers used downstream."""
    import time

    started = time.perf_counter()
    pre, post = build_markets(config)
    cohorts = build_cohorts(config)
    av = config.account_value

    def cohort_value(cohort: Cohort, state: MarketState) -> float:
        return portfolio_value(
            [cohort], state, n_paths=config.n_paths, seed=config.pricing_seed
        )

    def liability_value(state: MarketState) -> float:
        return sum(cohort_value(c, state) for c in cohorts)

    liability_pre = liability_value(pre)
    ladder = build_ladder(config, pre, liability_pre)

    def surplus_value(state: MarketState) -> float:
        return ladder.value(state) - liability_value(state)

    # ---- portfolio table and shock P&L (full revaluation pre vs post) -------
    portfolio_rows: list[dict[str, Any]] = []
    for spec, cohort in zip(COHORT_SPECS, cohorts, strict=True):
        pv_pre = cohort_value(cohort, pre)
        pv_post = cohort_value(cohort, post)
        portfolio_rows.append(
            {
                **{k: v for k, v in spec.items()},
                "account_value": spec["policies"] * spec["account_value_per_policy"],
                "pv_pre_shock": pv_pre,
                "pv_post_shock": pv_post,
                "shock_change": pv_post - pv_pre,
            }
        )
    liability_post = sum(r["pv_post_shock"] for r in portfolio_rows)
    assets_pre = ladder.value(pre)
    assets_post = ladder.value(post)
    shock_pnl = {
        "liability_change": liability_post - liability_pre,
        "asset_change": assets_post - assets_pre,
        "surplus_change": (assets_post - liability_post) - (assets_pre - liability_pre),
    }
    # Decompose the shock by applying one factor at a time (order-dependent
    # cross terms are reported explicitly).
    equity_only = pre.bumped(spot_relative=config.equity_shock)
    equity_vol = equity_only.bumped(vol_absolute=config.vol_shock)
    liability_eq = liability_value(equity_only)
    liability_eqvol = liability_value(equity_vol)
    shock_pnl.update(
        {
            "liability_equity_leg": liability_eq - liability_pre,
            "liability_vol_leg": liability_eqvol - liability_eq,
            "liability_rate_leg": liability_post - liability_eqvol,
        }
    )

    # ---- Greeks at the stressed market ---------------------------------------
    greeks: dict[str, dict[str, float]] = {}
    for cohort in cohorts:

        def cohort_pricer(state: MarketState, c: Cohort = cohort) -> float:
            return cohort_value(c, state)

        greeks[cohort.label] = _greeks_dict(bump_and_revalue(cohort_pricer, post))
    liability_greeks = _greeks_dict(bump_and_revalue(liability_value, post))
    asset_greeks = _greeks_dict(bump_and_revalue(ladder.value, post))
    net_greeks = _combine(asset_greeks, liability_greeks, -1.0)
    greeks["Liabilities"] = liability_greeks
    greeks["Assets"] = asset_greeks
    greeks["Net (insurer)"] = net_greeks
    carry = {
        "liability": liability_value(post.bumped(time_years=config.carry_horizon_years))
        - liability_post,
        "assets": ladder.value(post.bumped(time_years=config.carry_horizon_years))
        - assets_post,
    }
    carry["surplus"] = carry["assets"] - carry["liability"]

    # ---- instruments -----------------------------------------------------------
    instruments = build_instruments(config, post)
    instrument_greeks = {
        item.key: _greeks_dict(bump_and_revalue(item.value, post))
        for item in instruments
    }
    instrument_rows = [
        {
            "key": item.key,
            "name": item.name,
            "kind": item.kind,
            "description": item.description,
            "strike": item.strike,
            "unit_value": item.value(post),
            "premium_per_contract": item.premium_per_contract,
            "liquidity_per_contract": item.liquidity_per_contract,
            "transaction_cost": item.transaction_cost,
            "max_position": item.max_position,
            "delta": instrument_greeks[item.key]["delta"],
            "vega": instrument_greeks[item.key]["vega"],
            "rho": instrument_greeks[item.key]["rho"],
            "dv01": instrument_greeks[item.key]["dv01"],
        }
        for item in instruments
    ]

    # ---- scenario sets (full revaluation of surplus and instruments) ---------
    def revalue(
        spot: np.ndarray, vol: np.ndarray, rate: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        base_surplus = surplus_value(post)
        base_values = {item.key: item.value(post) for item in instruments}
        pnl = np.empty(spot.size)
        inst = {item.key: np.empty(spot.size) for item in instruments}
        for i in range(spot.size):
            state = post.bumped(
                spot_relative=float(spot[i]),
                vol_absolute=float(vol[i]),
                rate_bp=float(rate[i]),
            )
            pnl[i] = surplus_value(state) - base_surplus
            for item in instruments:
                inst[item.key][i] = item.value(state) - base_values[item.key]
        return pnl, inst

    is_spot, is_vol, is_rate = generate_scenarios(
        config, config.in_sample_scenarios, config.in_sample_seed
    )
    oos_spot, oos_vol, oos_rate = generate_scenarios(
        config, config.out_of_sample_scenarios, config.out_of_sample_seed
    )
    is_pnl, is_inst = revalue(is_spot, is_vol, is_rate)
    oos_pnl, oos_inst = revalue(oos_spot, oos_vol, oos_rate)

    # ---- named stresses (full revaluation) -----------------------------------
    base_surplus = surplus_value(post)
    base_values = {item.key: item.value(post) for item in instruments}
    stress_unhedged: dict[str, float] = {}
    stress_inst: dict[str, dict[str, float]] = {}
    for scenario in NAMED_STRESSES:
        state = scenario.apply_market(post)
        stress_unhedged[scenario.name] = surplus_value(state) - base_surplus
        stress_inst[scenario.name] = {
            item.key: item.value(state) - base_values[item.key] for item in instruments
        }

    # ---- hedge universe for the optimizer --------------------------------------
    def universe(
        keys: Sequence[str], exposures_for: Callable[[CaseInstrument], dict[str, float]]
    ) -> list[HedgeInstrument]:
        return [
            HedgeInstrument(
                name=item.key,
                kind=item.kind,
                tenor=item.tenor,
                price=item.liquidity_per_contract,
                exposures=exposures_for(item),
                premium=item.premium_per_contract,
                max_position=item.max_position,
                max_trade=item.max_position,
                lot_size=1.0,
                transaction_cost=item.transaction_cost,
            )
            for item in instruments
            if item.key in keys
        ]

    constraints = HedgeConstraints(
        premium_budget=config.premium_budget,
        turnover_budget=config.liquidity_budget,
    )
    reference_exposure = {
        "delta": 0.1 * av / post.spot,
        "vega": 0.1 * av,
        "rho": 1.0 * av,
    }
    greek_weights = {
        k: 1.0 / max(abs(net_greeks[k]), reference_exposure[k])
        for k in reference_exposure
    }
    all_keys = tuple(item.key for item in instruments)

    def greek_exposures(
        names: Sequence[str],
    ) -> Callable[[CaseInstrument], dict[str, float]]:
        return lambda item: {k: instrument_greeks[item.key][k] for k in names}

    strategies: dict[str, StrategyResult] = {}

    def evaluate(key: str, objective: str, solution: HedgeSolution) -> StrategyResult:
        q = {
            item.key: float(solution.positions.get(item.key, 0.0)) + 0.0
            for item in instruments
        }
        hedge_greeks = {
            g: sum(q[k] * instrument_greeks[k][g] for k in q)
            for g in ("delta", "gamma", "vega", "rho", "dv01")
        }
        residual = _combine(net_greeks, hedge_greeks, 1.0)
        residual_norm = math.sqrt(
            sum((residual[g] * greek_weights[g]) ** 2 for g in greek_weights)
        )
        initial_norm = math.sqrt(
            sum((net_greeks[g] * greek_weights[g]) ** 2 for g in greek_weights)
        )
        greek_eff = float(
            np.clip(1.0 - residual_norm / max(initial_norm, 1e-12), 0.0, 1.0)
        )
        is_hedge = sum(q[k] * is_inst[k] for k in q)
        oos_hedge = sum(q[k] * oos_inst[k] for k in q)
        oos = scenario_metrics(oos_pnl, oos_hedge)
        cost = sum(
            abs(q[k]) * item.transaction_cost
            for k, item in zip(q, instruments, strict=True)
        )
        sd_removed = oos["unhedged_sd"] - oos["hedged_sd"]
        return StrategyResult(
            key=key,
            label=STRATEGY_LABELS[key],
            objective=objective,
            status="Optimal" if solution.success else solution.message,
            trades=q,
            notional={
                k: q[k] * item.value(post)
                for k, item in zip(q, instruments, strict=True)
            },
            premium_used=float(solution.premium_used),
            liquidity_used=float(solution.turnover_used),
            transaction_cost=float(cost),
            hedge_greeks=hedge_greeks,
            residual_greeks=residual,
            greek_effectiveness=greek_eff,
            in_sample=scenario_metrics(is_pnl, is_hedge),
            out_of_sample=oos,
            stress_pnl={
                name: stress_unhedged[name]
                + sum(q[k] * stress_inst[name][k] for k in q)
                for name in stress_unhedged
            },
            oos_hedged_pnl=[float(x) for x in (oos_pnl + oos_hedge)],
            cost_per_sd_removed=(
                (float(solution.premium_used) + float(cost)) / sd_removed
                if sd_removed > 0.0
                else math.inf
            ),
        )

    # Strategy A: delta-only with index futures.
    sol_a = solve(
        {"delta": net_greeks["delta"]},
        universe(("future",), greek_exposures(("delta",))),
        exposure_weights={"delta": greek_weights["delta"]},
        constraints=constraints,
        cost_penalty=config.cost_penalty,
    )
    strategies["delta_only"] = evaluate(
        "delta_only",
        "Zero the net delta with futures; vega and rates left unhedged",
        sol_a,
    )

    # Strategy B: multi-Greek matching (delta, vega, rho) over the full universe.
    targets_b = {k: net_greeks[k] for k in ("delta", "vega", "rho")}
    sol_b = solve(
        targets_b,
        universe(all_keys, greek_exposures(("delta", "vega", "rho"))),
        exposure_weights=greek_weights,
        constraints=constraints,
        cost_penalty=config.cost_penalty,
    )
    strategies["multi_greek"] = evaluate(
        "multi_greek",
        "Minimize the scaled first-order delta/vega/rho residual at the stressed market",
        sol_b,
    )

    # Strategy C: stress-aware. Each in-sample scenario is a risk factor whose
    # "exposure" is the full-revaluation P&L, so the Greek objective becomes a
    # root-mean-square hedged P&L; a CVaR(95%) penalty adds tail awareness.
    n_is = is_pnl.size
    scale = float(np.sqrt(np.mean(is_pnl**2))) * math.sqrt(n_is)
    scenario_names = tuple(f"s{i}" for i in range(n_is))
    es_unhedged = expected_shortfall(-is_pnl, config.cvar_alpha)
    sol_c = solve(
        dict(zip(scenario_names, is_pnl.tolist(), strict=True)),
        universe(
            all_keys,
            lambda item: dict(
                zip(scenario_names, is_inst[item.key].tolist(), strict=True)
            ),
        ),
        exposure_weights=dict.fromkeys(scenario_names, 1.0 / scale),
        constraints=constraints,
        cost_penalty=config.cost_penalty,
        scenario_liability_losses=(-is_pnl).tolist(),
        instrument_scenario_pnls={k: is_inst[k].tolist() for k in all_keys},
        cvar_weight=config.cvar_weight_scale / max(es_unhedged, 1.0) ** 2,
        cvar_alpha=config.cvar_alpha,
    )
    strategies["stress_aware"] = evaluate(
        "stress_aware",
        "Minimize RMS full-revaluation surplus P&L over in-sample joint shocks plus a CVaR(95%) penalty",
        sol_c,
    )

    # ---- robustness: fixed positions, alternative shock distributions ---------
    robustness: list[dict[str, Any]] = []
    for offset, spec in enumerate(ROBUSTNESS_SETS):
        alt_spot, alt_vol, alt_rate = generate_scenarios(
            config,
            config.robustness_scenarios,
            config.robustness_seed + offset,
            spot_scale=spec.get("spot_scale"),
            vol_beta=spec.get("vol_beta"),
            vol_noise=spec.get("vol_noise"),
            rate_sd_bp=spec.get("rate_sd_bp"),
            rate_spot_corr=spec.get("rate_spot_corr"),
        )
        alt_pnl, alt_inst = revalue(alt_spot, alt_vol, alt_rate)
        entry: dict[str, Any] = {
            "key": spec["key"],
            "name": spec["name"],
            "description": spec["description"],
            "count": int(alt_pnl.size),
            "unhedged_sd": float(np.std(alt_pnl)),
            "strategies": {},
        }
        for key, strategy in strategies.items():
            alt_hedge = sum(strategy.trades[k] * alt_inst[k] for k in strategy.trades)
            metrics = scenario_metrics(alt_pnl, alt_hedge)
            entry["strategies"][key] = {
                "variance_reduction": metrics["variance_reduction"],
                "es95_reduction": metrics["es95_reduction"],
                "dollar_offset_slope": metrics["dollar_offset_slope"],
                "hedged_sd": metrics["hedged_sd"],
            }
        robustness.append(entry)

    recommendation = select_recommendation(strategies, robustness)
    unhedged_oos = scenario_metrics(oos_pnl, np.zeros_like(oos_pnl))

    result = CaseStudyResult(
        title=CASE_STUDY_TITLE,
        short_title=CASE_STUDY_SHORT_TITLE,
        as_of=config.as_of,
        issue_date=config.issue_date,
        engine_version=__version__,
        notice=ILLUSTRATIVE_NOTICE,
        config=asdict(config),
        market={
            "issue_index": config.issue_index,
            "stressed_index": post.spot,
            "atm_vol_1y_pre": pre.volatility(pre.spot, 1.0),
            "atm_vol_1y_post": post.volatility(post.spot, 1.0),
            "zero_2y_pre": float(pre.rate_curve.zero_rate(2.0)),
            "zero_10y_pre": float(pre.rate_curve.zero_rate(10.0)),
            "zero_2y_post": float(post.rate_curve.zero_rate(2.0)),
            "zero_10y_post": float(post.rate_curve.zero_rate(10.0)),
            "dividend_yield": config.dividend_yield,
            "equity_shock": config.equity_shock,
            "vol_shock": config.vol_shock,
            "rate_shock_bp": config.rate_shock_bp,
            "curve_tenors": [float(t) for t in pre.rate_curve.tenors],
            "curve_pre": [float(r) for r in pre.rate_curve.zero_rates],
            "curve_post": [float(r) for r in post.rate_curve.zero_rates],
        },
        portfolio=portfolio_rows,
        assets={
            "tenors": list(ladder.tenors),
            "faces": list(ladder.faces),
            "market_value_pre_shock": assets_pre,
            "market_value_post_shock": assets_post,
            "duration_post_shock": ladder.duration(post),
            "liability_duration_post_shock": -liability_greeks["dv01"]
            * 1e4
            / liability_post,
            "dv01_post_shock": asset_greeks["dv01"],
        },
        shock_pnl=shock_pnl,
        greeks={**greeks, "Carry (1 month)": carry},
        instruments=instrument_rows,
        scenarios={
            "in_sample_count": int(n_is),
            "out_of_sample_count": int(oos_pnl.size),
            "in_sample_unhedged": scenario_metrics(is_pnl, np.zeros_like(is_pnl)),
            "out_of_sample_unhedged": unhedged_oos,
            "description": ILLUSTRATIVE_ASSUMPTIONS[4],
        },
        unhedged_stress_pnl=stress_unhedged,
        strategies=strategies,
        recommendation=recommendation,
        robustness=robustness,
        documented_events=[dict(e) for e in DOCUMENTED_EVENTS],
        illustrative_assumptions=list(ILLUSTRATIVE_ASSUMPTIONS),
        limitations=list(LIMITATIONS),
        oos_unhedged_pnl=[float(x) for x in oos_pnl],
        oos_shocks={
            "spot": [float(x) for x in oos_spot],
            "vol": [float(x) for x in oos_vol],
            "rate_bp": [float(x) for x in oos_rate],
        },
        runtime_seconds=time.perf_counter() - started,
    )
    return result


LIMITATIONS: tuple[str, ...] = (
    "Model-consistent test: hedges are designed and evaluated with the same "
    "engine (Black-Scholes replication, GBM Monte Carlo without skew for FIA/IUL). "
    "Out-of-sample means different scenario draws, not a different model; basis, "
    "skew dynamics, and jump risk are not represented.",
    "Shocks are instantaneous; one-month carry is reported separately. Rebalancing "
    "between shocks, margin calls over time, and path-dependent gap risk are not "
    "simulated.",
    "FIA and IUL values use 3,000 antithetic paths with common random numbers, so "
    "scenario differences are smooth but absolute PVs carry Monte Carlo error of "
    "roughly 0.05%; their finite-difference gamma is statistically noise (see the "
    "validation report).",
    "One-day finite-difference theta on the 20-year IUL is dominated by a monthly "
    "time-grid re-discretization artefact in the engine; carry is therefore "
    "measured with a one-month step and theta is not used in the hedge design.",
    "The premium budget counts gross option premium (sold options consume budget), "
    "which penalizes the natural short-put-spread RILA hedge; margin on short "
    "options is approximated by premium.",
    "Treasury exposure is a zero-coupon proxy valued off the engine's risk-free "
    "curve; futures basis, repo, cheapest-to-deliver, and credit spreads on the "
    "backing assets are ignored. Behaviour (lapse, mortality) is applied only to "
    "IUL, and RILA/FIA are valued to maturity without decrements.",
    "Scenario correlations (vol beta -0.30 to the index, rates independent) are "
    "assumptions. The documented October 2026 pattern mixed yields-up/equities-down "
    "(CPI day) and yields-down/equities-down (flight to quality) sessions.",
    "Hedge-effectiveness figures are variance and ES95 reductions over the stated "
    "shock distribution; they are not regulatory, accounting, or realized "
    "effectiveness measures.",
)


def select_recommendation(
    strategies: Mapping[str, StrategyResult],
    robustness: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Transparent decision rule applied to the model outputs.

    Primary criterion: out-of-sample variance reduction.  A strategy within 2
    percentage points of the best is treated as equivalent and the tie is
    broken by the lower total cost (premium plus transaction cost) and then by
    ES95 reduction.  Any strategy whose solver status is not optimal or whose
    variance reduction is negative is excluded.  Robustness results do not
    change the choice but are summarized in the rationale.
    """
    eligible = [
        s
        for s in strategies.values()
        if s.status == "Optimal" and s.out_of_sample["variance_reduction"] > 0.0
    ]
    if not eligible:
        return {"key": None, "label": "No eligible strategy", "rationale": []}
    best_vr = max(s.out_of_sample["variance_reduction"] for s in eligible)
    near_best = [
        s for s in eligible if s.out_of_sample["variance_reduction"] >= best_vr - 0.02
    ]
    chosen = sorted(
        near_best,
        key=lambda s: (
            s.premium_used + s.transaction_cost,
            -s.out_of_sample["es95_reduction"],
        ),
    )[0]
    rationale = [
        (
            f"Out-of-sample variance reduction {chosen.out_of_sample['variance_reduction']:.1%} "
            f"(best among the three is {best_vr:.1%}); dollar-offset slope "
            f"{chosen.out_of_sample['dollar_offset_slope']:.2f}; ES95 reduction "
            f"{chosen.out_of_sample['es95_reduction']:.1%}."
        ),
        (
            f"Total cost {chosen.premium_used + chosen.transaction_cost:,.0f} "
            f"(premium {chosen.premium_used:,.0f}, transaction {chosen.transaction_cost:,.0f}); "
            f"liquidity used {chosen.liquidity_used:,.0f}."
        ),
        "Decision rule: highest out-of-sample variance reduction, with strategies "
        "within two points treated as equivalent and the tie broken by lower cost, "
        "then by ES95 reduction.",
    ]
    if robustness:
        alt = [
            (entry["name"], entry["strategies"][chosen.key]["variance_reduction"])
            for entry in robustness
        ]
        worst_name, worst_vr = min(alt, key=lambda x: x[1])
        best_name, best_alt = max(alt, key=lambda x: x[1])
        wins = 0
        for entry in robustness:
            ranked = sorted(
                entry["strategies"].items(),
                key=lambda kv: -float(kv[1]["variance_reduction"]),
            )
            wins += int(ranked[0][0] == chosen.key)
        rationale.append(
            f"Robustness: with positions fixed, variance reduction ranges from "
            f"{worst_vr:.1%} ({worst_name}) to {best_alt:.1%} ({best_name}) across "
            f"{len(robustness)} alternative shock distributions; the chosen strategy "
            f"ranks first in {wins} of {len(robustness)}."
        )
    return {
        "key": chosen.key,
        "label": chosen.label,
        "best_variance_reduction": best_vr,
        "rationale": rationale,
        "ranking": [
            {
                "key": s.key,
                "variance_reduction": s.out_of_sample["variance_reduction"],
                "total_cost": s.premium_used + s.transaction_cost,
                "es95_reduction": s.out_of_sample["es95_reduction"],
            }
            for s in sorted(
                strategies.values(),
                key=lambda s: -s.out_of_sample["variance_reduction"],
            )
        ],
    }


# --------------------------------------------------------------------------- #
# Shared presentation helpers (used by both the PDF and the dashboard)
# --------------------------------------------------------------------------- #


def comparison_table(result: CaseStudyResult) -> pd.DataFrame:
    """Three-way strategy comparison with the same rows in the PDF and dashboard."""
    rows = []
    for key in STRATEGY_ORDER:
        s = result.strategies[key]
        rows.append(
            {
                "Strategy": s.label,
                "Premium paid": s.premium_used,
                "Transaction cost": s.transaction_cost,
                "Liquidity used": s.liquidity_used,
                "Greek-metric effectiveness": s.greek_effectiveness,
                "OOS variance reduction": s.out_of_sample["variance_reduction"],
                "Dollar-offset slope": s.out_of_sample["dollar_offset_slope"],
                "ES95 reduction": s.out_of_sample["es95_reduction"],
                "Hedged P&L sd": s.out_of_sample["hedged_sd"],
                "Residual delta": s.residual_greeks["delta"],
                "Residual vega": s.residual_greeks["vega"],
                "Residual DV01": s.residual_greeks["dv01"],
                "Recommended": key == result.recommendation.get("key"),
            }
        )
    return pd.DataFrame(rows)


def trade_table(result: CaseStudyResult) -> pd.DataFrame:
    """Contracts per instrument for each strategy."""
    rows = []
    for item in result.instruments:
        row: dict[str, Any] = {"Instrument": item["name"]}
        for key in STRATEGY_ORDER:
            row[STRATEGY_LABELS[key]] = result.strategies[key].trades[item["key"]]
        rows.append(row)
    return pd.DataFrame(rows)


def stress_table(result: CaseStudyResult) -> pd.DataFrame:
    """Named-stress surplus P&L, unhedged and under each strategy."""
    rows = []
    for name, unhedged in result.unhedged_stress_pnl.items():
        row: dict[str, Any] = {"Scenario": name, "Unhedged": unhedged}
        for key in STRATEGY_ORDER:
            row[STRATEGY_LABELS[key]] = result.strategies[key].stress_pnl[name]
        rows.append(row)
    return pd.DataFrame(rows)


def robustness_table(result: CaseStudyResult) -> pd.DataFrame:
    """Variance reduction of the fixed hedges under alternative shock distributions."""
    rows = []
    for entry in result.robustness:
        row: dict[str, Any] = {"Shock distribution": entry["name"]}
        for key in STRATEGY_ORDER:
            row[STRATEGY_LABELS[key]] = entry["strategies"][key]["variance_reduction"]
        rows.append(row)
    return pd.DataFrame(rows)


def headline_metrics(result: CaseStudyResult) -> dict[str, float]:
    """Headline figures quoted in both the PDF and the dashboard."""
    rec = result.strategies[result.recommendation["key"]]
    liabilities = result.greeks["Liabilities"]
    net = result.greeks["Net (insurer)"]
    return {
        "account_value": float(sum(r["account_value"] for r in result.portfolio)),
        "liability_pv_pre": float(sum(r["pv_pre_shock"] for r in result.portfolio)),
        "liability_pv_post": float(sum(r["pv_post_shock"] for r in result.portfolio)),
        "surplus_shock_pnl": result.shock_pnl["surplus_change"],
        "liability_delta": liabilities["delta"],
        "liability_vega": liabilities["vega"],
        "liability_dv01": liabilities["dv01"],
        "net_delta": net["delta"],
        "net_vega": net["vega"],
        "net_dv01": net["dv01"],
        "unhedged_oos_sd": result.scenarios["out_of_sample_unhedged"]["unhedged_sd"],
        "unhedged_oos_es95": result.scenarios["out_of_sample_unhedged"][
            "es95_unhedged"
        ],
        "recommended_variance_reduction": rec.out_of_sample["variance_reduction"],
        "recommended_dollar_offset": rec.out_of_sample["dollar_offset_slope"],
        "recommended_es95_reduction": rec.out_of_sample["es95_reduction"],
        "recommended_premium": rec.premium_used,
        "recommended_transaction_cost": rec.transaction_cost,
        "recommended_liquidity": rec.liquidity_used,
        "recommended_hedged_sd": rec.out_of_sample["hedged_sd"],
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def default_results_path() -> Path:
    return DEFAULT_RESULTS_DIR / RESULTS_FILENAME


def save_results(result: CaseStudyResult, path: Path | None = None) -> Path:
    """Serialize results to JSON (the single source for the PDF and dashboard)."""
    target = path or default_results_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=1, sort_keys=True))
    return target


def load_results(path: Path | None = None) -> CaseStudyResult:
    """Load the committed case-study results."""
    target = path or default_results_path()
    if not target.exists():
        raise FileNotFoundError(
            f"case-study results not found at {target}; run "
            "`python scripts/build_case_study.py` to generate them"
        )
    return CaseStudyResult.from_dict(json.loads(target.read_text()))


__all__ = [
    "CASE_STUDY_SHORT_TITLE",
    "CASE_STUDY_TITLE",
    "COHORT_SPECS",
    "DOCUMENTED_EVENTS",
    "ILLUSTRATIVE_ASSUMPTIONS",
    "ILLUSTRATIVE_NOTICE",
    "LIMITATIONS",
    "NAMED_STRESSES",
    "PDF_FILENAME",
    "RESULTS_FILENAME",
    "ROBUSTNESS_SETS",
    "STRATEGY_LABELS",
    "STRATEGY_ORDER",
    "CaseInstrument",
    "CaseStudyConfig",
    "CaseStudyResult",
    "StrategyResult",
    "StripLadder",
    "build_cohorts",
    "build_instruments",
    "build_ladder",
    "build_markets",
    "comparison_table",
    "default_results_path",
    "expected_shortfall",
    "generate_scenarios",
    "headline_metrics",
    "load_results",
    "robustness_table",
    "run_case_study",
    "save_results",
    "scenario_metrics",
    "select_recommendation",
    "stress_table",
    "trade_table",
]
