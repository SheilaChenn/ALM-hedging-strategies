"""ALM HedgeLab — interactive asset-liability hedging dashboard.

The UI is intentionally usable before the ``alm_hedging`` package is installed:
an adapter calls a small, plausible public API when available and otherwise
uses deterministic illustrative analytics. The active data source is always
shown in the interface.
"""

from __future__ import annotations

import importlib
import inspect
import io
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="ALM HedgeLab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "ink": "#14213D",
    "navy": "#102A43",
    "blue": "#2F6BFF",
    "teal": "#12B8A6",
    "gold": "#F3B61F",
    "red": "#E25555",
    "slate": "#68778D",
    "pale": "#EAF0F8",
}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');
:root { --ink:#14213D; --blue:#2F6BFF; --teal:#12B8A6; --paper:#F5F7FB; }
html, body, [class*="css"] { font-family:"DM Sans",sans-serif; }
.stApp { background:linear-gradient(145deg,#F8FAFD 0%,#F2F5FA 55%,#F7F9FC 100%); color:var(--ink); }
[data-testid="stSidebar"] { background:#102A43; }
[data-testid="stSidebar"] * { color:#F7FAFC; }
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
[data-testid="stSidebar"] input { color:#14213D; background:#fff; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color:#fff; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.14); }
.block-container { max-width:1540px; padding-top:1.4rem; padding-bottom:3rem; }
h1,h2,h3 { letter-spacing:-.025em; color:#14213D; }
.eyebrow { font-family:"IBM Plex Mono",monospace; color:#2F6BFF; font-size:.72rem;
  letter-spacing:.14em; text-transform:uppercase; font-weight:600; }
.hero { background:linear-gradient(120deg,#102A43,#173F63 70%,#126E78); border-radius:18px;
  padding:1.5rem 1.7rem; margin-bottom:1rem; box-shadow:0 10px 30px rgba(16,42,67,.13); }
.hero h1 { color:white; font-size:2rem; margin:.25rem 0; }
.hero p { color:#D9E7F2; margin:0; max-width:850px; }
.hero .eyebrow { color:#65E2D3; }
.status-pill { display:inline-flex; align-items:center; gap:.4rem; background:rgba(255,255,255,.1);
  color:#EAF8F6; border:1px solid rgba(255,255,255,.16); border-radius:99px;
  padding:.32rem .65rem; font-size:.75rem; float:right; }
.metric-card { background:#fff; border:1px solid #E2E8F0; border-radius:14px;
  padding:1rem 1.05rem; min-height:112px; box-shadow:0 3px 14px rgba(20,33,61,.045); }
.metric-label { color:#68778D; font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }
.metric-value { font-family:"IBM Plex Mono",monospace; color:#14213D; font-size:1.45rem; font-weight:500; margin:.35rem 0 .15rem; }
.metric-help { color:#77869A; font-size:.76rem; }
.section-note { border-left:3px solid #12B8A6; background:#EDF9F7; border-radius:0 8px 8px 0;
  padding:.7rem .9rem; color:#315A5C; font-size:.83rem; margin:.5rem 0 1rem; }
.source-live { color:#087F72; background:#E5F8F4; border-radius:99px; padding:.22rem .55rem; font-size:.72rem; }
.source-demo { color:#8A6200; background:#FFF5D6; border-radius:99px; padding:.22rem .55rem; font-size:.72rem; }
[data-testid="stDataFrame"] { border:1px solid #E2E8F0; border-radius:12px; overflow:hidden; }
.stTabs [data-baseweb="tab-list"] { gap:.3rem; background:#EAF0F8; padding:.3rem; border-radius:12px; }
.stTabs [data-baseweb="tab"] { height:42px; border-radius:9px; padding:0 .85rem; }
.stTabs [aria-selected="true"] { background:white; box-shadow:0 1px 5px rgba(20,33,61,.09); }
.disclaimer { color:#718096; font-size:.72rem; line-height:1.45; border-top:1px solid #E2E8F0; padding-top:1rem; margin-top:2rem; }
div[data-testid="stDownloadButton"] button { width:100%; border-color:#2F6BFF; color:#2455CD; }
@media(max-width:768px) {
  .block-container { padding:.8rem .7rem 2rem; }
  .hero { padding:1.1rem; border-radius:14px; }
  .hero h1 { font-size:1.55rem; }
  .status-pill { float:none; margin-bottom:.6rem; }
  .metric-card { min-height:96px; margin-bottom:.35rem; }
  .stTabs [data-baseweb="tab"] { padding:0 .5rem; font-size:.78rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


@dataclass(frozen=True)
class Cohort:
    product: str
    account_value: float
    policies: int
    duration: float
    participation: float
    cap: float
    floor: float
    guarantee: float
    weight: float = 1.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value


def _as_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, Mapping):
        if all(np.isscalar(v) for v in value.values()):
            return pd.DataFrame([value])
        return pd.DataFrame(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return pd.DataFrame(value)
    return pd.DataFrame()


class AnalyticsBridge:
    """Thin compatibility layer around likely ``alm_hedging`` API shapes."""

    candidates = ("alm_hedging", "alm_hedging.api", "alm_hedging.analytics")
    public_methods = {
        "get_market_snapshot",
        "market_snapshot",
        "build_market",
        "price_portfolio",
        "price_liabilities",
        "price",
        "calculate_greeks",
        "portfolio_greeks",
        "greeks",
        "run_stress_test",
        "stress_matrix",
        "stress",
        "optimize_hedge",
        "recommend_hedge",
        "solve_hedge",
        "backtest_hedge",
        "run_backtest",
        "backtest",
    }

    def __init__(self) -> None:
        self.module: Any | None = None
        self.error: str | None = None
        for name in self.candidates:
            try:
                candidate = importlib.import_module(name)
                if any(
                    callable(getattr(candidate, method, None))
                    for method in self.public_methods
                ):
                    self.module = candidate
                    break
            except ModuleNotFoundError as exc:
                if exc.name and not exc.name.startswith("alm_hedging"):
                    self.error = f"Package dependency unavailable: {exc.name}"
                    break
            except Exception as exc:  # package initialization errors belong in UI
                self.error = f"{type(exc).__name__}: {exc}"
                break

    @property
    def live(self) -> bool:
        return self.module is not None

    @property
    def version(self) -> str:
        return (
            str(getattr(self.module, "__version__", "development"))
            if self.live
            else "illustrative-1.0"
        )

    def call(
        self, names: tuple[str, ...], fallback: Callable[..., Any], **kwargs: Any
    ) -> Any:
        if self.live:
            for name in names:
                fn = getattr(self.module, name, None)
                if callable(fn):
                    try:
                        params = inspect.signature(fn).parameters
                        accepted = (
                            kwargs
                            if any(p.kind == p.VAR_KEYWORD for p in params.values())
                            else {
                                key: val for key, val in kwargs.items() if key in params
                            }
                        )
                        return fn(**accepted)
                    except Exception as exc:
                        st.warning(
                            f"`alm_hedging.{name}` could not complete; showing illustrative result. {exc}"
                        )
                        break
        return fallback(**kwargs)


bridge = AnalyticsBridge()


@st.cache_data(ttl=300, show_spinner=False)
def demo_market(
    as_of: str, spot: float, rate: float, vol: float, dividend: float, **_: Any
) -> dict[str, Any]:
    rng = np.random.default_rng(abs(hash(as_of)) % (2**32))
    dates = pd.bdate_range(end=pd.Timestamp(as_of), periods=126)
    rets = rng.normal((rate - dividend) / 252, vol / math.sqrt(252), len(dates))
    path = spot * np.exp(np.cumsum(rets) - np.cumsum(rets)[-1])
    curve_t = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10])
    curve = rate + 0.004 * np.log1p(curve_t) - 0.0025 * curve_t / 10
    return {
        "spot": spot,
        "rate": rate,
        "volatility": vol,
        "dividend_yield": dividend,
        "history": pd.DataFrame({"date": dates, "spot": path}),
        "curve": pd.DataFrame({"tenor": curve_t, "zero_rate": curve}),
        "as_of": as_of,
    }


def demo_price(
    cohorts: list[dict[str, Any]], market: Mapping[str, Any], **_: Any
) -> dict[str, Any]:
    spot, rate, vol = market["spot"], market["rate"], market["volatility"]
    rows = []
    for c in cohorts:
        t, av = c["duration"], c["account_value"]
        d1 = (math.log(1.0) + (rate + 0.5 * vol**2) * t) / (vol * math.sqrt(t))
        option_factor = vol * math.sqrt(t) * math.exp(-0.4 * abs(d1)) * 0.38
        option = av * c["participation"] * min(option_factor, c["cap"] + 0.02)
        guarantee = av * max(c["guarantee"] - rate, 0) * t * 0.43
        floor = av * max(c["floor"], 0) * math.exp(-rate * t)
        total = option + guarantee + floor
        rows.append(
            {
                "product": c["product"],
                "option_budget": option,
                "guarantee": guarantee,
                "floor_value": floor,
                "liability": total,
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "components": frame,
        "liability": frame["liability"].sum(),
        "account_value": sum(c["account_value"] for c in cohorts),
        "spot": spot,
    }


def demo_greeks(
    pricing: Mapping[str, Any], market: Mapping[str, Any], **_: Any
) -> pd.DataFrame:
    notional = pricing["account_value"]
    vol, spot = market["volatility"], market["spot"]
    return pd.DataFrame(
        {
            "Greek": ["Delta", "Gamma", "Vega", "Rho", "Theta"],
            "Exposure": [
                notional * 0.43,
                notional * 0.0011 / spot,
                notional * vol * 0.62,
                -notional * 1.7,
                -notional * vol / 365,
            ],
            "Unit": [
                "$ / 1.00 index",
                "$ / index²",
                "$ / 1.00 vol",
                "$ / 1.00 rate",
                "$ / day",
            ],
        }
    )


def demo_stress(
    pricing: Mapping[str, Any], market: Mapping[str, Any], **_: Any
) -> pd.DataFrame:
    spot_shocks = [-0.25, -0.15, -0.05, 0.0, 0.05, 0.15, 0.25]
    vol_shocks = [-0.10, -0.05, 0.0, 0.05, 0.10]
    n, base_vol = pricing["account_value"], market["volatility"]
    values = []
    for vs in vol_shocks:
        row = []
        for ss in spot_shocks:
            convex = n * (0.43 * ss + 0.19 * ss**2)
            vega = n * 0.62 * base_vol * vs
            cross = n * 0.21 * ss * vs
            row.append(convex + vega + cross)
        values.append(row)
    return pd.DataFrame(
        values,
        index=[f"{x:+.0%}" for x in vol_shocks],
        columns=[f"{x:+.0%}" for x in spot_shocks],
    )


def demo_hedge(
    pricing: Mapping[str, Any],
    market: Mapping[str, Any],
    budget: float,
    turnover_limit: float,
    instruments: list[str],
    **_: Any,
) -> dict[str, Any]:
    n = pricing["account_value"]
    specs = {
        "Equity futures": (0.34 * n / market["spot"], 0.0, 0.08),
        "ATM calls": (0.11 * n / market["spot"], n * market["volatility"] * 0.37, 0.58),
        "Put spread": (
            -0.025 * n / market["spot"],
            n * market["volatility"] * 0.16,
            0.21,
        ),
        "Variance swap": (0.0, n * market["volatility"] * 0.09, 0.07),
        "10Y swap": (0.0, 0.0, -0.06),
    }
    rows = []
    for name in instruments:
        delta, vega, weight = specs[name]
        units = min(1.0, turnover_limit / 0.25) * weight * n / max(market["spot"], 1)
        cost = abs(weight) * budget * 0.82
        rows.append(
            {
                "instrument": name,
                "units": units,
                "delta_offset": delta,
                "vega_offset": vega,
                "premium": cost,
            }
        )
    alloc = pd.DataFrame(rows)
    return {
        "allocation": alloc,
        "premium": alloc["premium"].sum(),
        "residual_delta": 0.055 * n,
        "residual_vega": 0.10 * n * market["volatility"],
        "effectiveness": 0.84,
        "status": "Optimal",
    }


def demo_backtest(
    pricing: Mapping[str, Any],
    market: Mapping[str, Any],
    frequency: str,
    transaction_cost_bps: float,
    lookback_years: int,
    **_: Any,
) -> pd.DataFrame:
    periods = {"Daily": 252, "Weekly": 52, "Monthly": 12}[frequency] * lookback_years
    freq = {"Daily": "B", "Weekly": "W-FRI", "Monthly": "ME"}[frequency]
    dates = pd.date_range(end=pd.Timestamp(market["as_of"]), periods=periods, freq=freq)
    rng = np.random.default_rng(42 + periods)
    raw = rng.normal(0, pricing["account_value"] * 0.005, periods)
    tail = rng.choice([0, 1], periods, p=[0.97, 0.03]) * rng.normal(
        -pricing["account_value"] * 0.014, pricing["account_value"] * 0.005, periods
    )
    liability = raw + tail
    hedge = -0.81 * liability + rng.normal(
        0, pricing["account_value"] * 0.0017, periods
    )
    costs = np.abs(np.diff(np.r_[0, hedge])) * transaction_cost_bps / 10_000
    return pd.DataFrame(
        {
            "date": dates,
            "liability_pnl": liability,
            "hedge_pnl": hedge,
            "cost": costs,
            "net_pnl": liability + hedge - costs,
        }
    )


def money(value: float, short: bool = True) -> str:
    sign = "−" if value < 0 else ""
    value = abs(value)
    if short and value >= 1e9:
        return f"{sign}${value / 1e9:,.2f}B"
    if short and value >= 1e6:
        return f"{sign}${value / 1e6:,.1f}M"
    if short and value >= 1e3:
        return f"{sign}${value / 1e3:,.1f}K"
    return f"{sign}${value:,.0f}"


def metric_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-help">{help_text}</div></div>',
        unsafe_allow_html=True,
    )


def style_figure(fig: go.Figure, height: int = 350) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=40, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="DM Sans", color=COLORS["ink"]),
        hoverlabel=dict(bgcolor="white"),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#DDE5EF")
    fig.update_yaxes(gridcolor="#E8EDF4", zerolinecolor="#CBD5E1")
    return fig


def md(text: str) -> str:
    """Escape dollar signs so Streamlit markdown does not render them as LaTeX."""
    return text.replace("$", "\\$")


def case_figure(fig: go.Figure, height: int = 350) -> go.Figure:
    """Dashboard styling with the legend kept clear of the chart title."""
    fig = style_figure(fig, height)
    fig.update_layout(
        margin=dict(l=12, r=12, t=64, b=15),
        legend=dict(orientation="h", y=1.14, x=0),
        title=dict(y=0.97),
    )
    return fig


@st.cache_resource(show_spinner=False)
def load_case_study() -> tuple[Any | None, Any | None, str]:
    """Load the committed October 2026 case-study results.

    Returns ``(module, result, message)``; ``result`` is ``None`` when the
    package or the committed JSON is unavailable so the tab can explain why.
    The dashboard never recomputes the case study: it reads the same JSON the
    PDF is built from, which is what keeps the two presentations reconciled.
    """
    try:
        module = importlib.import_module("alm_hedging.case_study")
    except ImportError as exc:
        return None, None, f"alm_hedging.case_study is not importable ({exc})."
    try:
        return module, module.load_results(), ""
    except FileNotFoundError as exc:
        return module, None, str(exc)


today = date.today()
with st.sidebar:
    st.markdown("## ◈ ALM HedgeLab")
    st.caption("Liability-aware hedge design")
    st.divider()
    portfolio_type = st.selectbox(
        "Portfolio cohort", ["Mixed portfolio", "RILA", "FIA", "IUL"]
    )
    total_av = st.number_input(
        "Account value ($M)", min_value=10.0, max_value=25_000.0, value=500.0, step=10.0
    )
    duration = st.slider("Liability duration (years)", 0.5, 12.0, 5.0, 0.5)
    policies = st.number_input(
        "In-force policies",
        min_value=100,
        max_value=2_000_000,
        value=25_000,
        step=1_000,
    )
    st.markdown("##### Product economics")
    participation = st.slider("Participation rate", 0.20, 2.00, 1.00, 0.05)
    cap = st.slider("Annual cap", 0.02, 0.30, 0.12, 0.01)
    floor = st.slider("Annual floor", -0.20, 0.05, 0.00, 0.01)
    guarantee = st.slider("Guarantee rate", 0.00, 0.08, 0.02, 0.0025)
    st.markdown("##### Valuation market")
    as_of = st.date_input("As of", today)
    spot = st.number_input("Equity index", min_value=100.0, value=5_250.0, step=25.0)
    rate = st.slider("Risk-free rate", 0.0, 0.10, 0.0425, 0.0025)
    vol = st.slider("Implied volatility", 0.05, 0.60, 0.185, 0.005)
    dividend = st.slider("Dividend yield", 0.0, 0.08, 0.015, 0.0025)
    run = st.button("Run analysis", type="primary", width="stretch")
    st.caption("Inputs are scenario assumptions—not live market data.")


def make_cohorts() -> list[Cohort]:
    av = total_av * 1e6
    if portfolio_type != "Mixed portfolio":
        return [
            Cohort(
                portfolio_type,
                av,
                int(policies),
                duration,
                participation,
                cap,
                floor,
                guarantee,
            )
        ]
    mix = [
        ("RILA", 0.40, 1.08, 0.15, -0.10, 0.0),
        ("FIA", 0.35, 0.72, 0.09, 0.0, 0.025),
        ("IUL", 0.25, 0.92, 0.11, 0.0, 0.03),
    ]
    return [
        Cohort(
            name,
            av * weight,
            round(policies * weight),
            duration * (0.9 + i * 0.12),
            part,
            prod_cap,
            prod_floor,
            max(guarantee, guar),
            weight,
        )
        for i, (name, weight, part, prod_cap, prod_floor, guar) in enumerate(mix)
    ]


if run or "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = True

source_badge = (
    '<span class="source-live">● PACKAGE CONNECTED</span>'
    if bridge.live
    else '<span class="source-demo">● ILLUSTRATIVE ENGINE</span>'
)
st.markdown(
    f"""
<div class="hero">
  <div class="status-pill">Model {bridge.version} · {as_of:%d %b %Y}</div>
  <div class="eyebrow">Asset–liability management / decision support</div>
  <h1>See the liability. Shape the hedge.</h1>
  <p>Price embedded guarantees, map nonlinear risk, and test a capital-aware hedge program across market regimes.</p>
</div>
<div style="margin-bottom:.7rem">{source_badge}</div>
""",
    unsafe_allow_html=True,
)

if bridge.error:
    st.error(
        f"The analytics package could not initialize. Illustrative analytics remain available. Technical detail: {bridge.error}"
    )

try:
    with st.spinner("Revaluing liabilities and solving the hedge program…"):
        cohorts = make_cohorts()
        cohort_dicts = [asdict(c) for c in cohorts]
        market = bridge.call(
            ("get_market_snapshot", "market_snapshot", "build_market"),
            demo_market,
            as_of=as_of.isoformat(),
            spot=spot,
            rate=rate,
            volatility=vol,
            vol=vol,
            dividend_yield=dividend,
            dividend=dividend,
        )
        market = (
            dict(market)
            if isinstance(market, Mapping)
            else demo_market(as_of.isoformat(), spot, rate, vol, dividend)
        )
        market.setdefault("spot", spot)
        market.setdefault("rate", rate)
        market.setdefault("volatility", vol)
        market.setdefault("as_of", as_of.isoformat())
        pricing = bridge.call(
            ("price_portfolio", "price_liabilities", "price"),
            demo_price,
            cohorts=cohort_dicts,
            portfolio=cohort_dicts,
            market=market,
        )
        pricing = (
            dict(pricing)
            if isinstance(pricing, Mapping)
            else demo_price(cohort_dicts, market)
        )
        pricing.setdefault("account_value", sum(c.account_value for c in cohorts))
        pricing.setdefault("liability", 0.0)
        greeks = _as_frame(
            bridge.call(
                ("calculate_greeks", "portfolio_greeks", "greeks"),
                demo_greeks,
                pricing=pricing,
                portfolio=cohort_dicts,
                cohorts=cohort_dicts,
                market=market,
            )
        )
except Exception as exc:
    st.error(f"Analysis could not be completed: {type(exc).__name__}: {exc}")
    st.info(
        "Review the market and cohort inputs, then select **Run analysis**. No trades or files were created."
    )
    st.stop()

if not cohorts or pricing["account_value"] <= 0:
    st.info(
        "No in-force exposure is available for analysis. Add a cohort or increase account value to continue."
    )
    st.stop()

greek_map = dict(zip(greeks.get("Greek", []), greeks.get("Exposure", []), strict=False))
mcols = st.columns(5)
with mcols[0]:
    metric_card(
        "Account value",
        money(pricing["account_value"]),
        f"{sum(c.policies for c in cohorts):,} policies",
    )
with mcols[1]:
    metric_card(
        "Liability value",
        money(float(pricing["liability"])),
        "Embedded option + guarantees",
    )
with mcols[2]:
    metric_card(
        "Net delta",
        money(float(greek_map.get("Delta", 0))),
        "$ exposure per index point",
    )
with mcols[3]:
    metric_card(
        "Vega", money(float(greek_map.get("Vega", 0))), "$ exposure per 100 vol points"
    )
with mcols[4]:
    liability_ratio = float(pricing["liability"]) / max(
        float(pricing["account_value"]), 1
    )
    metric_card("Option budget", f"{liability_ratio:.2%}", "Liability / account value")

tabs = st.tabs(
    [
        "Overview",
        "Pricing & Greeks",
        "Stress lab",
        "Hedge design",
        "Backtest",
        "Attribution",
        "Export",
        "Oct 2026 case study",
    ]
)

with tabs[0]:
    st.markdown("### Market and cohort snapshot")
    left, right = st.columns([1.45, 1])
    with left:
        history = _as_frame(market.get("history", []))
        if not history.empty and {"date", "spot"}.issubset(history.columns):
            fig = px.area(
                history, x="date", y="spot", color_discrete_sequence=[COLORS["blue"]]
            )
            fig.update_traces(line_width=2, fillcolor="rgba(47,107,255,.10)")
            fig.update_layout(
                title="Equity index · trailing six months",
                yaxis_title="Index",
                xaxis_title=None,
            )
            st.plotly_chart(style_figure(fig), width="stretch")
        else:
            st.info("Market history is unavailable for this valuation date.")
    with right:
        curve = _as_frame(market.get("curve", []))
        if not curve.empty and {"tenor", "zero_rate"}.issubset(curve.columns):
            fig = px.line(
                curve,
                x="tenor",
                y="zero_rate",
                markers=True,
                color_discrete_sequence=[COLORS["teal"]],
            )
            fig.update_layout(
                title="Risk-free zero curve",
                xaxis_title="Tenor (years)",
                yaxis_tickformat=".1%",
            )
            st.plotly_chart(style_figure(fig), width="stretch")
        else:
            st.info("Yield-curve points are unavailable.")
    cohort_frame = pd.DataFrame(cohort_dicts).rename(
        columns={
            "product": "Product",
            "account_value": "Account value",
            "policies": "Policies",
            "duration": "Duration",
            "participation": "Participation",
            "cap": "Cap",
            "floor": "Floor",
            "guarantee": "Guarantee",
            "weight": "Portfolio weight",
        }
    )
    st.dataframe(
        cohort_frame,
        width="stretch",
        hide_index=True,
        column_config={
            "Account value": st.column_config.NumberColumn(format="$%.0f"),
            "Participation": st.column_config.NumberColumn(format="%.1%%"),
            "Cap": st.column_config.NumberColumn(format="%.1%%"),
            "Floor": st.column_config.NumberColumn(format="%.1%%"),
            "Guarantee": st.column_config.NumberColumn(format="%.2%%"),
            "Portfolio weight": st.column_config.ProgressColumn(
                format="%.0%%", min_value=0, max_value=1
            ),
        },
    )

with tabs[1]:
    st.markdown("### Payoff decomposition")
    st.markdown(
        '<div class="section-note">Values are risk-neutral estimates at the selected valuation market. Attribution separates embedded index optionality from contractual floors and guarantees.</div>',
        unsafe_allow_html=True,
    )
    components = _as_frame(pricing.get("components", []))
    c1, c2 = st.columns([1.2, 1])
    with c1:
        if not components.empty and "product" in components.columns:
            value_cols = [
                c
                for c in ["option_budget", "guarantee", "floor_value"]
                if c in components
            ]
            long = components.melt(
                id_vars="product",
                value_vars=value_cols,
                var_name="Component",
                value_name="Value",
            )
            long["Component"] = long["Component"].str.replace("_", " ").str.title()
            fig = px.bar(
                long,
                x="product",
                y="Value",
                color="Component",
                barmode="stack",
                color_discrete_sequence=[
                    COLORS["blue"],
                    COLORS["gold"],
                    COLORS["teal"],
                ],
            )
            fig.update_layout(
                title="Liability value by source",
                xaxis_title=None,
                yaxis_tickprefix="$",
                yaxis_tickformat="~s",
            )
            st.plotly_chart(style_figure(fig), width="stretch")
        else:
            st.info("Payoff components were not returned by the pricing engine.")
    with c2:
        if not greeks.empty and {"Greek", "Exposure"}.issubset(greeks.columns):
            colors = [
                COLORS["red"] if x < 0 else COLORS["teal"] for x in greeks["Exposure"]
            ]
            fig = go.Figure(
                go.Bar(
                    x=greeks["Exposure"],
                    y=greeks["Greek"],
                    orientation="h",
                    marker_color=colors,
                )
            )
            fig.update_layout(
                title="First- and second-order sensitivities",
                xaxis_title="Exposure (native units)",
                yaxis_title=None,
            )
            st.plotly_chart(style_figure(fig), width="stretch")
    st.dataframe(
        greeks,
        width="stretch",
        hide_index=True,
        column_config={"Exposure": st.column_config.NumberColumn(format="$%.2f")},
    )

with tabs[2]:
    st.markdown("### Joint equity / volatility stress")
    stress = _as_frame(
        bridge.call(
            ("run_stress_test", "stress_matrix", "stress"),
            demo_stress,
            pricing=pricing,
            portfolio=cohort_dicts,
            cohorts=cohort_dicts,
            market=market,
        )
    )
    if stress.empty:
        st.info(
            "No stress results were returned. Confirm that the scenario grid is supported by the analytics engine."
        )
    else:
        numeric = stress.select_dtypes(include="number")
        if numeric.empty:
            st.warning("Stress output contains no numeric P&L values.")
        else:
            fig = go.Figure(
                go.Heatmap(
                    z=numeric.values,
                    x=numeric.columns.astype(str),
                    y=stress.index.astype(str),
                    colorscale=[[0, "#B83B5E"], [0.5, "#F7F9FC"], [1, "#0A9F8E"]],
                    zmid=0,
                    colorbar=dict(title="P&L", tickprefix="$", tickformat="~s"),
                    hovertemplate="Equity shock %{x}<br>Vol shock %{y}<br>P&L $%{z:,.0f}<extra></extra>",
                )
            )
            fig.update_layout(
                title="Unhedged liability P&L",
                xaxis_title="Equity shock",
                yaxis_title="Volatility shock",
            )
            st.plotly_chart(style_figure(fig, 430), width="stretch")
            worst = float(numeric.min().min())
            st.caption(
                f"Worst modeled cell: {money(worst)}. Parallel rate, basis, lapse, and behavior stresses should be reviewed separately."
            )

with tabs[3]:
    st.markdown("### Constrained hedge recommendation")
    a, b, c = st.columns(3)
    with a:
        budget_pct = (
            st.slider("Premium budget (% AV)", 0.05, 3.0, 0.75, 0.05, key="budget")
            / 100
        )
    with b:
        turnover_limit = st.slider("Turnover limit (% AV)", 1.0, 40.0, 15.0, 1.0) / 100
    with c:
        instruments = st.multiselect(
            "Eligible instruments",
            ["Equity futures", "ATM calls", "Put spread", "Variance swap", "10Y swap"],
            default=["Equity futures", "ATM calls", "Put spread"],
        )
    if not instruments:
        st.info("Select at least one eligible instrument to generate a recommendation.")
    else:
        hedge = bridge.call(
            ("recommend_hedge", "optimize_hedge", "solve_hedge"),
            demo_hedge,
            pricing=pricing,
            portfolio=cohort_dicts,
            cohorts=cohort_dicts,
            market=market,
            budget=budget_pct * pricing["account_value"],
            premium_budget=budget_pct * pricing["account_value"],
            turnover_limit=turnover_limit,
            instruments=instruments,
        )
        hedge = dict(hedge) if isinstance(hedge, Mapping) else {}
        hc = st.columns(4)
        with hc[0]:
            metric_card(
                "Solver status",
                str(hedge.get("status", "Unknown")),
                "Constrained recommendation",
            )
        with hc[1]:
            metric_card(
                "Premium",
                money(float(hedge.get("premium", 0))),
                f"{float(hedge.get('premium', 0)) / pricing['account_value']:.2%} of AV",
            )
        with hc[2]:
            metric_card(
                "Residual delta",
                money(float(hedge.get("residual_delta", 0))),
                "After recommended trades",
            )
        with hc[3]:
            metric_card(
                "Effectiveness",
                f"{float(hedge.get('effectiveness', 0)):.0%}",
                "Modeled risk reduction",
            )
        allocation = _as_frame(hedge.get("allocation", []))
        st.dataframe(
            allocation,
            width="stretch",
            hide_index=True,
            column_config={
                "units": st.column_config.NumberColumn(format="%.1f"),
                "delta_offset": st.column_config.NumberColumn(format="$%.0f"),
                "vega_offset": st.column_config.NumberColumn(format="$%.0f"),
                "premium": st.column_config.NumberColumn(format="$%.0f"),
            },
        )
        st.warning(
            "Recommendation is a model output, not an execution instruction. Validate liquidity, accounting, counterparty, collateral, and governance constraints before use."
        )

with tabs[4]:
    st.markdown("### Dynamic hedge backtest")
    x, y, z = st.columns(3)
    with x:
        frequency = st.selectbox(
            "Rebalance frequency", ["Daily", "Weekly", "Monthly"], index=1
        )
    with y:
        transaction_cost = st.number_input(
            "Transaction cost (bps)", 0.0, 100.0, 3.0, 0.5
        )
    with z:
        lookback = st.selectbox(
            "History",
            [1, 3, 5, 10],
            index=1,
            format_func=lambda n: f"{n} year{'s' if n > 1 else ''}",
        )
    bt = _as_frame(
        bridge.call(
            ("backtest_hedge", "run_backtest", "backtest"),
            demo_backtest,
            pricing=pricing,
            portfolio=cohort_dicts,
            cohorts=cohort_dicts,
            market=market,
            frequency=frequency,
            rebalance_frequency=frequency.lower(),
            transaction_cost_bps=transaction_cost,
            lookback_years=lookback,
        )
    )
    required = {"date", "liability_pnl", "hedge_pnl", "cost", "net_pnl"}
    if not required.issubset(bt.columns):
        st.info("Backtest output is empty or missing expected P&L series.")
    else:
        bt["date"] = pd.to_datetime(bt["date"])
        bt["cumulative_unhedged"] = bt["liability_pnl"].cumsum()
        bt["cumulative_hedged"] = bt["net_pnl"].cumsum()
        fig = go.Figure()
        fig.add_scatter(
            x=bt["date"],
            y=bt["cumulative_unhedged"],
            name="Unhedged",
            line=dict(color=COLORS["slate"], width=2),
        )
        fig.add_scatter(
            x=bt["date"],
            y=bt["cumulative_hedged"],
            name="Hedged, net of cost",
            line=dict(color=COLORS["teal"], width=2.5),
        )
        fig.update_layout(
            title="Cumulative liability P&L",
            yaxis_tickprefix="$",
            yaxis_tickformat="~s",
            xaxis_title=None,
        )
        st.plotly_chart(style_figure(fig, 400), width="stretch")
        vol_before = bt["liability_pnl"].std() * math.sqrt(
            {"Daily": 252, "Weekly": 52, "Monthly": 12}[frequency]
        )
        vol_after = bt["net_pnl"].std() * math.sqrt(
            {"Daily": 252, "Weekly": 52, "Monthly": 12}[frequency]
        )
        tail_before = bt["liability_pnl"].quantile(0.05)
        tail_after = bt["net_pnl"].quantile(0.05)
        k = st.columns(4)
        with k[0]:
            metric_card("Ann. risk before", money(vol_before), "Standard deviation")
        with k[1]:
            metric_card(
                "Ann. risk after",
                money(vol_after),
                f"{1 - vol_after / max(vol_before, 1):.0%} reduction",
            )
        with k[2]:
            metric_card("5% tail before", money(tail_before), "Historical period P&L")
        with k[3]:
            metric_card("5% tail after", money(tail_after), "Net of transaction cost")

with tabs[5]:
    st.markdown("### Before / after P&L attribution")
    if "bt" not in locals() or not required.issubset(bt.columns):
        st.info("Run a valid backtest to populate P&L attribution.")
    else:
        sums = {
            "Liability movement": bt["liability_pnl"].sum(),
            "Hedge instruments": bt["hedge_pnl"].sum(),
            "Transaction cost": -bt["cost"].sum(),
        }
        total = sum(sums.values())
        fig = go.Figure(
            go.Waterfall(
                x=list(sums) + ["Net hedged P&L"],
                y=list(sums.values()) + [total],
                measure=["relative", "relative", "relative", "total"],
                connector={"line": {"color": "#9AA8B9"}},
                increasing={"marker": {"color": COLORS["teal"]}},
                decreasing={"marker": {"color": COLORS["red"]}},
                totals={"marker": {"color": COLORS["navy"]}},
                text=[money(v) for v in list(sums.values()) + [total]],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Cumulative economic attribution",
            yaxis_tickprefix="$",
            yaxis_tickformat="~s",
        )
        st.plotly_chart(style_figure(fig, 420), width="stretch")
        monthly = (
            bt.set_index("date")[["liability_pnl", "hedge_pnl", "cost", "net_pnl"]]
            .resample("ME")
            .sum()
            .reset_index()
        )
        st.dataframe(
            monthly.tail(18),
            width="stretch",
            hide_index=True,
            column_config={
                "date": st.column_config.DateColumn("Month", format="MMM YYYY"),
                "liability_pnl": st.column_config.NumberColumn(
                    "Liability P&L", format="$%.0f"
                ),
                "hedge_pnl": st.column_config.NumberColumn("Hedge P&L", format="$%.0f"),
                "cost": st.column_config.NumberColumn("Cost", format="$%.0f"),
                "net_pnl": st.column_config.NumberColumn("Net P&L", format="$%.0f"),
            },
        )

with tabs[6]:
    st.markdown("### Export analysis package")
    st.write(
        "Download reproducible inputs, summary analytics, and backtest observations for independent review."
    )
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_version": bridge.version,
        "engine": "alm_hedging" if bridge.live else "illustrative",
        "as_of": as_of.isoformat(),
        "cohorts": cohort_dicts,
        "market_inputs": {
            "spot": spot,
            "rate": rate,
            "volatility": vol,
            "dividend_yield": dividend,
        },
        "pricing": _jsonable(pricing),
        "greeks": _jsonable(greeks),
    }
    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button(
            "Download analysis JSON",
            json.dumps(manifest, default=_jsonable, indent=2),
            f"alm-analysis-{as_of}.json",
            "application/json",
        )
    with e2:
        greek_csv = greeks.to_csv(index=False).encode()
        st.download_button(
            "Download Greeks CSV", greek_csv, f"alm-greeks-{as_of}.csv", "text/csv"
        )
    with e3:
        if "bt" in locals() and not bt.empty:
            output = io.BytesIO()
            bt.to_csv(output, index=False)
            st.download_button(
                "Download backtest CSV",
                output.getvalue(),
                f"alm-backtest-{as_of}.csv",
                "text/csv",
            )
        else:
            st.button("Backtest unavailable", disabled=True, width="stretch")
    st.code(
        json.dumps(
            {
                "model_version": bridge.version,
                "valuation_date": as_of.isoformat(),
                "engine": manifest["engine"],
            },
            indent=2,
        ),
        language="json",
    )

with tabs[7]:
    st.markdown("### October 2026 Market Shock Case Study")
    cs_module, cs_result, cs_message = load_case_study()
    if cs_result is None or cs_module is None:
        st.warning(
            "The case study results are not available in this environment. "
            f"{cs_message} Generate them with `python scripts/build_case_study.py`."
        )
    else:
        cs = cs_module
        hm = cs.headline_metrics(cs_result)
        rec = cs_result.strategies[cs_result.recommendation["key"]]
        mkt = cs_result.market
        net = cs_result.greeks["Net (insurer)"]
        cs_cfg = cs_result.config
        strategy_colors = {
            "delta_only": COLORS["slate"],
            "multi_greek": COLORS["gold"],
            "stress_aware": COLORS["teal"],
        }
        short_labels = {k: v.split(" (")[0] for k, v in cs.STRATEGY_LABELS.items()}
        st.markdown(md(f"**{cs_result.title}**"))
        st.markdown(
            f'<div class="section-note"><b>Illustrative only.</b> {cs_result.notice} '
            "Every number on this tab is read from the committed results JSON that "
            "also feeds the three-page PDF, so the two reconcile exactly "
            "(regression-tested in <code>tests/test_case_study.py</code>).</div>",
            unsafe_allow_html=True,
        )

        cards = st.columns(5)
        with cards[0]:
            metric_card(
                "Unhedged shock P&L",
                money(hm["surplus_shock_pnl"]),
                "Surplus change, issue → stressed market (full revaluation)",
            )
        with cards[1]:
            metric_card(
                "Net delta",
                f"{hm['net_delta']:+,.0f} $/pt",
                "Insurer sign at the stressed market",
            )
        with cards[2]:
            metric_card(
                "Net vega",
                f"{money(hm['net_vega'] / 100)} /vol pt",
                "Long vega: liability credits fall when vol rises",
            )
        with cards[3]:
            metric_card(
                "Net DV01",
                f"{hm['net_dv01']:+,.0f} $/bp",
                "Liability duration exceeds the asset ladder",
            )
        with cards[4]:
            metric_card(
                "Recommended hedge",
                f"{hm['recommended_variance_reduction']:.1%}",
                f"OOS variance reduction · {short_labels[rec.key]}",
            )

        st.markdown("#### 1 · Documented backdrop vs. illustrative shock")
        left, right = st.columns([1.35, 1])
        with left:
            st.caption(
                "Documented market events (dated, sourced). These are facts about "
                "October 2026 markets; they are not model inputs."
            )
            events = pd.DataFrame(
                [
                    {
                        "Date": e["date"],
                        "Event": e["summary"],
                        "Outlet": e["source"],
                        "Source": e["url"],
                    }
                    for e in cs_result.documented_events
                ]
            )
            st.dataframe(
                events,
                width="stretch",
                hide_index=True,
                column_config={
                    "Date": st.column_config.TextColumn(width="small"),
                    "Event": st.column_config.TextColumn(width="large"),
                    "Outlet": st.column_config.TextColumn(width="small"),
                    "Source": st.column_config.LinkColumn(
                        "Source", display_text="open", width="small"
                    ),
                },
            )
        with right:
            st.caption(
                "Illustrative shock applied to the synthetic block (assumption, "
                "sized as an escalation of the documented moves)."
            )
            shock_rows = pd.DataFrame(
                [
                    {
                        "Factor": "Equity index",
                        "Issue (6 Oct)": f"{mkt['issue_index']:,.0f}",
                        "Stressed (16 Oct)": f"{mkt['stressed_index']:,.0f}",
                        "Shock": f"{mkt['equity_shock']:+.0%}",
                    },
                    {
                        "Factor": "1y ATM implied vol",
                        "Issue (6 Oct)": f"{mkt['atm_vol_1y_pre']:.0%}",
                        "Stressed (16 Oct)": f"{mkt['atm_vol_1y_post']:.0%}",
                        "Shock": f"{mkt['vol_shock'] * 100:+.0f} pts",
                    },
                    {
                        "Factor": "10y zero rate",
                        "Issue (6 Oct)": f"{mkt['zero_10y_pre']:.2%}",
                        "Stressed (16 Oct)": f"{mkt['zero_10y_post']:.2%}",
                        "Shock": f"{mkt['rate_shock_bp']:+.0f} bp (parallel)",
                    },
                ]
            )
            st.dataframe(shock_rows, width="stretch", hide_index=True)
            curve = pd.DataFrame(
                {
                    "Tenor (years)": list(mkt["curve_tenors"]) * 2,
                    "Zero rate": list(mkt["curve_pre"]) + list(mkt["curve_post"]),
                    "State": ["Issue date"] * len(mkt["curve_tenors"])
                    + ["Stressed"] * len(mkt["curve_tenors"]),
                }
            )
            fig = px.line(
                curve,
                x="Tenor (years)",
                y="Zero rate",
                color="State",
                markers=True,
                log_x=True,
                color_discrete_sequence=[COLORS["slate"], COLORS["blue"]],
            )
            fig.update_layout(
                title="Synthetic zero curve (illustrative)",
                yaxis_tickformat=".2%",
                legend_title_text="",
            )
            st.plotly_chart(case_figure(fig, 260), width="stretch")

        st.markdown("#### 2 · Synthetic $500M block and what the shock did")
        port = pd.DataFrame(
            [
                {
                    "Product": r["product"],
                    "Policies": f"{r['policies']:,}",
                    "Account value": money(r["account_value"]),
                    "Term": f"{r['term_years']:.0f}y",
                    "Crediting terms": r["description"],
                    "PV at issue": money(r["pv_pre_shock"], short=False),
                    "PV stressed": money(r["pv_post_shock"], short=False),
                    "Shock ΔPV": money(r["shock_change"], short=False),
                }
                for r in cs_result.portfolio
            ]
        )
        st.dataframe(
            port,
            width="stretch",
            hide_index=True,
            column_config={
                "Crediting terms": st.column_config.TextColumn(width="large"),
            },
        )
        p1, p2 = st.columns([1, 1.3])
        with p1:
            legs = [
                ("Liability equity leg", -cs_result.shock_pnl["liability_equity_leg"]),
                ("Liability vol leg", -cs_result.shock_pnl["liability_vol_leg"]),
                ("Liability rates leg", -cs_result.shock_pnl["liability_rate_leg"]),
                ("Asset ladder (rates)", cs_result.shock_pnl["asset_change"]),
                ("Surplus change", cs_result.shock_pnl["surplus_change"]),
            ]
            fig = go.Figure(
                go.Bar(
                    x=[name for name, _ in legs],
                    y=[v for _, v in legs],
                    marker_color=[
                        COLORS["teal"] if v >= 0 else COLORS["red"] for _, v in legs
                    ][:-1]
                    + [COLORS["navy"]],
                    text=[money(v) for _, v in legs],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Shock P&L decomposition (insurer sign, full revaluation)",
                yaxis_title="$",
                showlegend=False,
                yaxis_range=[
                    min(0.0, min(v for _, v in legs)) * 1.35,
                    max(v for _, v in legs) * 1.45,
                ],
            )
            fig.update_xaxes(tickangle=-20, tickfont=dict(size=11))
            st.plotly_chart(case_figure(fig, 360), width="stretch")
        with p2:
            greek_rows = pd.DataFrame(
                [
                    {
                        "Position": label,
                        "PV": money(g["pv"], short=False),
                        "Delta ($/pt)": f"{g['delta']:,.0f}",
                        "Vega ($/vol pt)": money(g["vega"] / 100.0, short=False),
                        "DV01 ($/bp)": f"{g['dv01']:,.0f}",
                    }
                    for label in (
                        "RILA",
                        "FIA",
                        "IUL",
                        "Liabilities",
                        "Assets",
                        "Net (insurer)",
                    )
                    for g in [cs_result.greeks[label]]
                ]
            )
            st.caption(
                "Bump-and-revalue exposures at the stressed market (engine sign: "
                "positive = value rises). The Net row is what the hedge must offset."
            )
            st.dataframe(greek_rows, width="stretch", hide_index=True)
            st.caption(
                md(
                    f"Forward-looking scenario set: {cs_result.scenarios['description']} "
                    f"Unhedged out-of-sample P&L sd {money(hm['unhedged_oos_sd'])}, "
                    f"ES95 {money(hm['unhedged_oos_es95'])}."
                )
            )

        st.markdown("#### 3 · Three hedging strategies, one optimizer")
        comp = cs.comparison_table(cs_result)
        trades = cs.trade_table(cs_result)
        st.caption(
            md(
                "All three hedges are solved by the engine's constrained optimizer from "
                "the same instrument universe, premium budget "
                f"({money(cs_cfg['premium_budget_pct'] * hm['account_value'])}), "
                f"liquidity budget ({money(cs_cfg['liquidity_budget'])}) and "
                "position caps; they differ only in the objective. Effectiveness is "
                "measured by full revaluation on "
                f"{cs_result.scenarios['out_of_sample_count']} out-of-sample joint "
                "shocks, as the validation report recommends."
            )
        )
        comp_display = pd.DataFrame(
            {
                "Strategy": comp["Strategy"],
                "Recommended": comp["Recommended"],
                "Premium paid": comp["Premium paid"].map(
                    lambda v: money(v, short=False)
                ),
                "Transaction cost": comp["Transaction cost"].map(
                    lambda v: money(v, short=False)
                ),
                "Liquidity used": comp["Liquidity used"].map(money),
                "Greek-metric effectiveness": comp["Greek-metric effectiveness"].map(
                    lambda v: f"{v:.1%}"
                ),
                "OOS variance reduction": comp["OOS variance reduction"],
                "Dollar-offset slope": comp["Dollar-offset slope"].map(
                    lambda v: f"{v:.2f}"
                ),
                "ES95 reduction": comp["ES95 reduction"].map(lambda v: f"{v:.1%}"),
                "Hedged P&L sd": comp["Hedged P&L sd"].map(money),
                "Residual delta ($/pt)": comp["Residual delta"].map(
                    lambda v: f"{v:+,.0f}"
                ),
                "Residual vega ($/vol pt)": (comp["Residual vega"] / 100).map(
                    lambda v: f"{v:+,.0f}"
                ),
                "Residual DV01 ($/bp)": comp["Residual DV01"].map(
                    lambda v: f"{v:+,.0f}"
                ),
            }
        )
        st.dataframe(
            comp_display,
            width="stretch",
            hide_index=True,
            column_config={
                "OOS variance reduction": st.column_config.ProgressColumn(
                    format="percent", min_value=0, max_value=1
                ),
                "Recommended": st.column_config.CheckboxColumn(),
            },
        )
        t1, t2 = st.columns([1, 1.2])
        with t1:
            st.caption("Trades (contracts; positive = long)")
            trades_display = trades.rename(
                columns={v: short_labels[k] for k, v in cs.STRATEGY_LABELS.items()}
            )
            for col in short_labels.values():
                trades_display[col] = trades_display[col].map(lambda v: f"{v:+,.0f}")
            st.dataframe(trades_display, width="stretch", hide_index=True)
            st.caption(
                md(
                    f"Recommended: {short_labels[rec.key]} — premium "
                    f"{money(rec.premium_used)}, transaction cost "
                    f"{money(rec.transaction_cost)}, liquidity {money(rec.liquidity_used)} "
                    f"of {money(cs_cfg['liquidity_budget'])}."
                )
            )
        with t2:
            fig = go.Figure()
            unhedged = np.asarray(cs_result.oos_unhedged_pnl) / 1e6
            fig.add_trace(
                go.Histogram(
                    x=unhedged,
                    name="Unhedged",
                    marker_color="#CBD5E1",
                    opacity=0.85,
                    xbins=dict(size=1.0),
                )
            )
            for key in cs.STRATEGY_ORDER:
                s = cs_result.strategies[key]
                fig.add_trace(
                    go.Histogram(
                        x=np.asarray(s.oos_hedged_pnl) / 1e6,
                        name=short_labels[key],
                        marker_color=strategy_colors[key],
                        opacity=0.55,
                        xbins=dict(size=1.0),
                    )
                )
            fig.update_layout(
                barmode="overlay",
                title="Out-of-sample surplus P&L ($M, insurer sign)",
                xaxis_title="P&L ($M)",
            )
            st.plotly_chart(case_figure(fig, 360), width="stretch")

        s1, s2 = st.columns(2)
        with s1:
            stress = cs.stress_table(cs_result)
            stress_names = [n.split(" (")[0] for n in stress["Scenario"]]
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=stress_names,
                    y=stress["Unhedged"],
                    name="Unhedged",
                    marker_color="#CBD5E1",
                )
            )
            for key in cs.STRATEGY_ORDER:
                fig.add_trace(
                    go.Bar(
                        x=stress_names,
                        y=stress[cs.STRATEGY_LABELS[key]],
                        name=short_labels[key],
                        marker_color=strategy_colors[key],
                    )
                )
            fig.update_layout(
                barmode="group",
                title="Named full-revaluation stresses: surplus P&L ($)",
                yaxis_title="$",
            )
            fig.update_xaxes(tickangle=-30, tickfont=dict(size=10))
            st.plotly_chart(case_figure(fig, 400), width="stretch")
            with st.expander("Stress definitions and P&L table"):
                stress_display = stress.copy()
                for col in stress_display.columns[1:]:
                    stress_display[col] = stress_display[col].map(money)
                st.dataframe(
                    stress_display.rename(
                        columns={
                            v: short_labels[k] for k, v in cs.STRATEGY_LABELS.items()
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
        with s2:
            rob = cs.robustness_table(cs_result)
            base_row: dict[str, Any] = {"Shock distribution": "Base out-of-sample set"}
            for key in cs.STRATEGY_ORDER:
                base_row[cs.STRATEGY_LABELS[key]] = cs_result.strategies[
                    key
                ].out_of_sample["variance_reduction"]
            rob = pd.concat([pd.DataFrame([base_row]), rob], ignore_index=True)
            rob = rob.rename(
                columns={v: short_labels[k] for k, v in cs.STRATEGY_LABELS.items()}
            )
            for col in short_labels.values():
                rob[col] = rob[col].map(lambda v: f"{v:.1%}")
            st.caption(
                "Robustness: positions held fixed, out-of-sample variance reduction "
                "re-measured under alternative shock distributions."
            )
            st.dataframe(rob, width="stretch", hide_index=True)
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Shock distribution": e["name"],
                            "Description": e["description"],
                        }
                        for e in cs_result.robustness
                    ]
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "Shock distribution": st.column_config.TextColumn(width="small"),
                    "Description": st.column_config.TextColumn(width="large"),
                },
            )

        st.markdown(md(f"#### 4 · Recommended strategy: {rec.label}"))
        trade_lines = [
            f"{rec.trades[i['key']]:+,.0f} × {i['name']} "
            f"(notional {money(rec.notional[i['key']])})"
            for i in cs_result.instruments
            if abs(rec.trades[i["key"]]) >= 0.5
        ]
        r1, r2 = st.columns([1, 1])
        with r1:
            st.markdown("**Trades**")
            for line in trade_lines:
                st.markdown(md(f"- {line}"))
            st.markdown(
                md(
                    f"- Option premium {money(rec.premium_used)}; transaction cost "
                    f"{money(rec.transaction_cost)}; liquidity used "
                    f"{money(rec.liquidity_used)} of {money(cs_cfg['liquidity_budget'])}"
                )
            )
            st.markdown("**Effectiveness (out-of-sample, full revaluation)**")
            st.markdown(
                md(
                    f"- Variance reduction {rec.out_of_sample['variance_reduction']:.1%}; "
                    f"dollar-offset slope {rec.out_of_sample['dollar_offset_slope']:.2f}\n"
                    f"- P&L sd {money(hm['unhedged_oos_sd'])} → "
                    f"{money(rec.out_of_sample['hedged_sd'])}; ES95 "
                    f"{money(hm['unhedged_oos_es95'])} → "
                    f"{money(rec.out_of_sample['es95_hedged'])} "
                    f"({rec.out_of_sample['es95_reduction']:.1%} reduction)\n"
                    f"- Residual exposures: delta {rec.residual_greeks['delta']:+,.0f} $/pt, "
                    f"vega {money(rec.residual_greeks['vega'] / 100)}/pt, DV01 "
                    f"{rec.residual_greeks['dv01']:+,.0f} $/bp "
                    f"(unhedged {net['delta']:+,.0f} / {money(net['vega'] / 100)} / "
                    f"{net['dv01']:+,.0f})"
                )
            )
        with r2:
            st.markdown("**Why this one (decision rule applied to model outputs)**")
            for line in cs_result.recommendation["rationale"]:
                st.markdown(md(f"- {line}"))
            st.markdown("**Ranking**")
            ranking = pd.DataFrame(cs_result.recommendation["ranking"])
            st.dataframe(
                pd.DataFrame(
                    {
                        "Strategy": ranking["key"].map(short_labels),
                        "OOS variance reduction": ranking["variance_reduction"].map(
                            lambda v: f"{v:.1%}"
                        ),
                        "Premium + cost": ranking["total_cost"].map(
                            lambda v: money(v, short=False)
                        ),
                        "ES95 reduction": ranking["es95_reduction"].map(
                            lambda v: f"{v:.1%}"
                        ),
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        with st.expander("Key assumptions (illustrative)"):
            for line in cs_result.illustrative_assumptions:
                st.markdown(md(f"- {line}"))
        with st.expander("Limitations and what would change the answer"):
            for line in cs_result.limitations:
                st.markdown(md(f"- {line}"))
        with st.expander("Sources for the documented backdrop"):
            for e in cs_result.documented_events:
                st.markdown(md(f"- {e['date']} — {e['source']}: {e['url']}"))
        pdf_path = cs.DEFAULT_RESULTS_DIR / cs.PDF_FILENAME
        d1, d2 = st.columns(2)
        with d1:
            if pdf_path.exists():
                st.download_button(
                    "Download the three-page case study PDF",
                    pdf_path.read_bytes(),
                    pdf_path.name,
                    "application/pdf",
                )
            else:
                st.button("PDF not built yet", disabled=True, width="stretch")
        with d2:
            st.download_button(
                "Download the results JSON (single source for PDF and dashboard)",
                json.dumps(cs_result.to_dict(), indent=1, sort_keys=True),
                cs.RESULTS_FILENAME,
                "application/json",
            )
        st.caption(
            f"Results as of {cs_result.as_of} · engine v{cs_result.engine_version} · "
            f"run_case_study() runtime {cs_result.runtime_seconds:.0f}s · rebuild with "
            "`python scripts/build_case_study.py`."
        )

st.markdown(
    """
<div class="disclaimer">
<b>Important:</b> This application is for institutional analysis and scenario exploration only. It is not investment,
actuarial, accounting, tax, or legal advice and does not constitute an offer or recommendation to transact.
Illustrative results are synthetic, omit material risks, and must not be used for production decisions. Model outputs
depend on assumptions and may differ materially from realized outcomes. Independently validate models, inputs, market
data, policyholder behavior, liquidity, collateral, counterparty exposure, and regulatory treatment before relying on results.
</div>
""",
    unsafe_allow_html=True,
)
