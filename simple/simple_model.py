"""Simple ALM hedging model for RILA, FIA, and IUL embedded options.

Illustrative only. Every input is synthetic. One year, one index, one
representative $100 of account value per product, closed-form Black-Scholes
with a flat rate, flat dividend yield, and flat volatility. No Monte Carlo,
no optimizer, no policyholder behavior. This module is deliberately
self-contained and does not import from ``src/alm_hedging``.

Conventions
-----------
* Each product is a list of vanilla option legs on the index. Quantity +1 is
  a long leg from the policyholder's point of view, so the product PV is the
  value of the benefit the insurer owes (a liability).
* Greeks are in standard units: delta per index point, gamma per index point
  squared, vega per 1.00 of volatility, rho per 1.00 of rate, theta per year.
  ``scaled_greeks`` converts to per-vol-point, per-basis-point, per-day.
* Stress P&L is from the insurer's point of view: a rise in the liability PV
  is a loss (negative number).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

OptionKind = str  # "call" or "put"


@dataclass(frozen=True)
class Market:
    """Flat synthetic market. Defaults: index 100, rate 4%, dividend 1.5%, vol 19%."""

    spot: float = 100.0
    rate: float = 0.04
    dividend: float = 0.015
    vol: float = 0.19
    maturity: float = 1.0

    def forward_factor(self) -> float:
        return math.exp((self.rate - self.dividend) * self.maturity)


@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float
    rho: float
    theta: float

    def __add__(self, other: Greeks) -> Greeks:
        return Greeks(
            *(a + b for a, b in zip(self.as_tuple(), other.as_tuple(), strict=True))
        )

    def scale(self, k: float) -> Greeks:
        return Greeks(*(k * a for a in self.as_tuple()))

    def as_tuple(self) -> tuple[float, ...]:
        return (self.price, self.delta, self.gamma, self.vega, self.rho, self.theta)


ZERO_GREEKS = Greeks(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_greeks(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    vol: float,
    kind: OptionKind,
) -> Greeks:
    """Black-Scholes price and analytic Greeks for a European call or put."""
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    sign = 1.0 if kind == "call" else -1.0
    if maturity <= 0.0 or vol <= 0.0:
        # Expired or deterministic: discounted forward intrinsic value, no convexity.
        fwd = spot * math.exp(-dividend * maturity)
        k_pv = strike * math.exp(-rate * maturity)
        intrinsic = max(sign * (fwd - k_pv), 0.0)
        in_money = sign * (fwd - k_pv) > 0.0
        return Greeks(
            intrinsic,
            sign * math.exp(-dividend * maturity) if in_money else 0.0,
            0.0,
            0.0,
            sign * strike * maturity * math.exp(-rate * maturity) if in_money else 0.0,
            0.0,
        )
    sqrt_t = math.sqrt(maturity)
    disc_q = math.exp(-dividend * maturity)
    disc_r = math.exp(-rate * maturity)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * maturity) / (
        vol * sqrt_t
    )
    d2 = d1 - vol * sqrt_t
    n_d1 = _norm_pdf(d1)
    price = sign * (
        spot * disc_q * _norm_cdf(sign * d1) - strike * disc_r * _norm_cdf(sign * d2)
    )
    delta = sign * disc_q * _norm_cdf(sign * d1)
    gamma = disc_q * n_d1 / (spot * vol * sqrt_t)
    vega = spot * disc_q * n_d1 * sqrt_t
    rho = sign * strike * maturity * disc_r * _norm_cdf(sign * d2)
    theta = (
        -spot * disc_q * n_d1 * vol / (2.0 * sqrt_t)
        - sign * rate * strike * disc_r * _norm_cdf(sign * d2)
        + sign * dividend * spot * disc_q * _norm_cdf(sign * d1)
    )
    return Greeks(price, delta, gamma, vega, rho, theta)


def bs_call(
    spot: float, strike: float, t: float, r: float, q: float, vol: float
) -> float:
    return bs_greeks(spot, strike, t, r, q, vol, "call").price


def bs_put(
    spot: float, strike: float, t: float, r: float, q: float, vol: float
) -> float:
    return bs_greeks(spot, strike, t, r, q, vol, "put").price


@dataclass(frozen=True)
class Leg:
    """One vanilla option leg. ``quantity`` is in option units per index unit."""

    kind: OptionKind
    strike: float
    quantity: float
    label: str

    def payoff(self, index_at_maturity: float) -> float:
        intrinsic = (
            max(index_at_maturity - self.strike, 0.0)
            if self.kind == "call"
            else max(self.strike - index_at_maturity, 0.0)
        )
        return self.quantity * intrinsic

    def greeks(self, market: Market) -> Greeks:
        raw = bs_greeks(
            market.spot,
            self.strike,
            market.maturity,
            market.rate,
            market.dividend,
            market.vol,
            self.kind,
        )
        return raw.scale(self.quantity)


@dataclass(frozen=True)
class Product:
    """An indexed crediting design expressed as option legs on $100 of account value.

    ``units`` converts index-unit legs to the account value: one index unit at
    the inception spot corresponds to ``account_value`` dollars.
    """

    name: str
    description: str
    legs: Sequence[Leg]
    account_value: float = 100.0
    inception_spot: float = 100.0

    @property
    def units(self) -> float:
        return self.account_value / self.inception_spot

    def payoff(self, index_at_maturity: float) -> float:
        """Credited amount (or loss, if negative) at maturity in dollars."""
        return self.units * sum(leg.payoff(index_at_maturity) for leg in self.legs)

    def credited_return(self, index_return: float) -> float:
        """Credited return for an index return, e.g. 0.12 -> 0.12 for FIA below cap."""
        return (
            self.payoff(self.inception_spot * (1.0 + index_return)) / self.account_value
        )


def rila(
    cap: float = 0.18,
    buffer: float = 0.10,
    spot: float = 100.0,
    account_value: float = 100.0,
) -> Product:
    """RILA: long ATM call, short call at the cap, short put at the buffer."""
    return Product(
        name="RILA",
        description=f"{buffer:.0%} buffer, {cap:.1%} cap: long ATM call, short {1 + cap:.2f} call, short {1 - buffer:.2f} put",
        legs=(
            Leg("call", spot, 1.0, "Long ATM call"),
            Leg("call", spot * (1.0 + cap), -1.0, f"Short {1 + cap:.3f} call (cap)"),
            Leg(
                "put",
                spot * (1.0 - buffer),
                -1.0,
                f"Short {1 - buffer:.2f} put (buffer)",
            ),
        ),
        account_value=account_value,
        inception_spot=spot,
    )


def fia(
    cap: float = 0.095, spot: float = 100.0, account_value: float = 100.0
) -> Product:
    """FIA: 0% floor and a cap, i.e. a long ATM call minus a short call at the cap."""
    return Product(
        name="FIA",
        description=f"0% floor, {cap:.1%} cap: long ATM call, short {1 + cap:.3f} call",
        legs=(
            Leg("call", spot, 1.0, "Long ATM call"),
            Leg("call", spot * (1.0 + cap), -1.0, f"Short {1 + cap:.3f} call (cap)"),
        ),
        account_value=account_value,
        inception_spot=spot,
    )


def iul(
    cap: float = 0.11, spot: float = 100.0, account_value: float = 100.0
) -> Product:
    """IUL: the same call spread as FIA on one year of indexed account value."""
    return Product(
        name="IUL",
        description=f"0% floor, {cap:.1%} cap on one year of account value: long ATM call, short {1 + cap:.2f} call",
        legs=(
            Leg("call", spot, 1.0, "Long ATM call"),
            Leg("call", spot * (1.0 + cap), -1.0, f"Short {1 + cap:.2f} call (cap)"),
        ),
        account_value=account_value,
        inception_spot=spot,
    )


@dataclass(frozen=True)
class LegValuation:
    label: str
    kind: OptionKind
    strike: float
    quantity: float
    greeks: Greeks


@dataclass(frozen=True)
class ProductValuation:
    product: Product
    greeks: Greeks
    legs: Sequence[LegValuation]

    @property
    def pv(self) -> float:
        return self.greeks.price


def price_product(product: Product, market: Market) -> ProductValuation:
    """PV and Greeks of the embedded option package on ``product.account_value``."""
    leg_vals = []
    total = ZERO_GREEKS
    for leg in product.legs:
        g = leg.greeks(market).scale(product.units)
        leg_vals.append(LegValuation(leg.label, leg.kind, leg.strike, leg.quantity, g))
        total = total + g
    return ProductValuation(product, total, tuple(leg_vals))


def scaled_greeks(g: Greeks) -> dict[str, float]:
    """Greeks in reporting units: per vol point, per basis point, per calendar day."""
    return {
        "PV": g.price,
        "Delta (per index pt)": g.delta,
        "Gamma (per index pt^2)": g.gamma,
        "Vega (per vol pt)": g.vega * 0.01,
        "Rho (per bp)": g.rho * 0.0001,
        "Theta (per day)": g.theta / 365.0,
    }


def solve_cap_from_budget(
    builder: Callable[[float], Product],
    budget: float,
    market: Market,
    lo: float = 0.0005,
    hi: float = 2.0,
    tol: float = 1e-10,
) -> float:
    """Find the cap whose option package costs ``budget`` dollars, by bisection.

    PV is increasing in the cap for every product here (a higher cap means a
    cheaper short call). Raises if the budget is outside the attainable range.
    """
    f_lo = price_product(builder(lo), market).pv - budget
    f_hi = price_product(builder(hi), market).pv - budget
    if f_lo > 0.0 or f_hi < 0.0:
        raise ValueError(
            f"budget {budget:.4f} is outside the attainable PV range [{f_lo + budget:.4f}, {f_hi + budget:.4f}]"
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = price_product(builder(mid), market).pv - budget
        if abs(f_mid) < tol or hi - lo < tol:
            return mid
        if f_mid < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class Scenario:
    name: str
    index_shock: float = 0.0  # relative, e.g. -0.10
    vol_shock: float = 0.0  # absolute, e.g. 0.05
    rate_shock: float = 0.0  # absolute, e.g. 0.005

    def apply(self, market: Market) -> Market:
        return replace(
            market,
            spot=market.spot * (1.0 + self.index_shock),
            vol=market.vol + self.vol_shock,
            rate=market.rate + self.rate_shock,
        )


DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("Index -20%", index_shock=-0.20),
    Scenario("Index -10%", index_shock=-0.10),
    Scenario("Index +10%", index_shock=0.10),
    Scenario("Index +20%", index_shock=0.20),
    Scenario("Vol +5 pts", vol_shock=0.05),
    Scenario("Rates +50 bp", rate_shock=0.005),
    Scenario("Rates -50 bp", rate_shock=-0.005),
)


@dataclass(frozen=True)
class Hedge:
    """Long index futures position. ``futures_units`` is in index units."""

    futures_units: float
    liability_delta: float
    residual_gamma: float
    residual_vega: float
    residual_rho: float

    def pnl(self, base: Market, shocked: Market) -> float:
        forward_base = base.spot * base.forward_factor()
        forward_shocked = shocked.spot * shocked.forward_factor()
        return self.futures_units * (forward_shocked - forward_base)


def portfolio_greeks(products: Sequence[Product], market: Market) -> Greeks:
    total = ZERO_GREEKS
    for product in products:
        total = total + price_product(product, market).greeks
    return total


def delta_hedge(products: Sequence[Product], market: Market) -> Hedge:
    """Size index futures to offset the liability delta; report what is left.

    The insurer is short the option package, so its delta is minus the
    liability delta. Buying ``delta / forward_factor`` futures units offsets
    it (a futures contract has delta ``forward_factor`` per index unit).
    Futures carry no gamma, vega, or rho, so those residuals are unchanged.
    """
    g = portfolio_greeks(products, market)
    return Hedge(
        futures_units=g.delta / market.forward_factor(),
        liability_delta=g.delta,
        residual_gamma=-g.gamma,
        residual_vega=-g.vega,
        residual_rho=-g.rho,
    )


@dataclass(frozen=True)
class StressRow:
    scenario: str
    liability_pv: float
    unhedged_pnl: float
    hedge_pnl: float
    hedged_pnl: float


def stress_table(
    products: Sequence[Product],
    market: Market,
    hedge: Hedge | None = None,
    scenarios: Sequence[Scenario] = DEFAULT_SCENARIOS,
) -> list[StressRow]:
    """Full revaluation of the liability under each scenario, before and after the hedge."""
    base_pv = sum(price_product(p, market).pv for p in products)
    rows = []
    for scenario in scenarios:
        shocked = scenario.apply(market)
        pv = sum(price_product(p, shocked).pv for p in products)
        unhedged = -(pv - base_pv)
        hedge_pnl = hedge.pnl(market, shocked) if hedge is not None else 0.0
        rows.append(
            StressRow(scenario.name, pv, unhedged, hedge_pnl, unhedged + hedge_pnl)
        )
    return rows


def pnl_range(rows: Sequence[StressRow], hedged: bool) -> float:
    values = [r.hedged_pnl if hedged else r.unhedged_pnl for r in rows]
    return max(values) - min(values)


@dataclass(frozen=True)
class SimpleModelResults:
    market: Market
    products: Sequence[Product]
    valuations: Sequence[ProductValuation]
    portfolio: Greeks
    option_budget: float
    solved_caps: dict[str, float]
    budget_curve: dict[str, list[tuple[float, float]]]
    hedge: Hedge
    stress: Sequence[StressRow]
    payoff_grid: list[float] = field(default_factory=list)
    payoffs: dict[str, list[float]] = field(default_factory=dict)

    @property
    def unhedged_range(self) -> float:
        return pnl_range(self.stress, hedged=False)

    @property
    def hedged_range(self) -> float:
        return pnl_range(self.stress, hedged=True)


DEFAULT_CAPS_FOR_CURVE: tuple[float, ...] = tuple(
    round(0.03 + 0.005 * step, 4) for step in range(35)
)


def run_simple_model(
    market: Market = Market(),
    option_budget: float = 3.50,
    caps_for_curve: Sequence[float] = DEFAULT_CAPS_FOR_CURVE,
) -> SimpleModelResults:
    """Run every calculation once so the notebook and the report reconcile exactly."""
    products = (rila(spot=market.spot), fia(spot=market.spot), iul(spot=market.spot))
    valuations = tuple(price_product(p, market) for p in products)
    builders: dict[str, Callable[[float], Product]] = {
        "RILA": lambda c: rila(cap=c, spot=market.spot),
        "FIA": lambda c: fia(cap=c, spot=market.spot),
        "IUL": lambda c: iul(cap=c, spot=market.spot),
    }
    solved = {
        name: solve_cap_from_budget(b, option_budget, market)
        for name, b in builders.items()
    }
    curve = {
        name: [(cap, price_product(b(cap), market).pv) for cap in caps_for_curve]
        for name, b in builders.items()
    }
    hedge = delta_hedge(products, market)
    grid = [0.60 + 0.01 * i for i in range(81)]
    payoffs = {p.name: [p.credited_return(x - 1.0) for x in grid] for p in products}
    return SimpleModelResults(
        market=market,
        products=products,
        valuations=valuations,
        portfolio=portfolio_greeks(products, market),
        option_budget=option_budget,
        solved_caps=solved,
        budget_curve=curve,
        hedge=hedge,
        stress=stress_table(products, market, hedge),
        payoff_grid=grid,
        payoffs=payoffs,
    )


def recommendation(results: SimpleModelResults) -> str:
    """One-paragraph hedge recommendation with the residual risks spelled out."""
    h, s = results.hedge, results.stress
    vol_row = next(r for r in s if r.scenario.startswith("Vol"))
    worst_after = min(s, key=lambda r: r.hedged_pnl)
    gamma_side = "long" if h.residual_gamma > 0 else "short"
    gamma_effect = "gains" if h.residual_gamma > 0 else "loses"
    vega_side = "long" if h.residual_vega > 0 else "short"
    vol_effect = "adds" if vol_row.hedged_pnl > 0 else "costs"
    return (
        f"Buy {h.futures_units:.3f} index futures units (notional about ${h.futures_units * results.market.spot:,.0f} "
        f"against ${sum(p.account_value for p in results.products):,.0f} of account value) to offset the liability delta "
        f"of {h.liability_delta:.3f} per index point. Full revaluation shows the stress P&L range shrinking from "
        f"${results.unhedged_range:,.2f} to ${results.hedged_range:,.2f}. The hedge is linear, so residual risks remain: "
        f"the insurer is {gamma_side} gamma ({h.residual_gamma:+.4f} per index point squared, so the hedged book {gamma_effect} "
        f"on large moves in either direction; worst hedged scenario is {worst_after.scenario} at ${worst_after.hedged_pnl:,.2f}), "
        f"{vega_side} vega ({h.residual_vega * 0.01:+.3f} per vol point; a +5 pt vol shock {vol_effect} ${abs(vol_row.hedged_pnl):,.2f}), "
        f"and rho of {h.residual_rho * 1e-4:+.4f} per basis point, which the futures roll partly offsets. "
        "Rebalance the futures as delta drifts with the index and with time, and revisit the hedge if the gamma, vega, "
        "or rate residuals move outside the risk budget. Illustrative model with synthetic inputs."
    )
