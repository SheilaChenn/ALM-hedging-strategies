"""Independent reference implementations used for model validation.

Nothing in this module imports pricing, risk, or hedging logic from
``alm_hedging``. Every function re-derives its result from first principles
(closed forms, numerical quadrature of the lognormal density, deterministic
recursions, brute-force search, or an independently seeded Monte Carlo stream)
so that agreement with the production engine is evidence rather than
tautology.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

# --------------------------------------------------------------------------- #
# Vanilla options
# --------------------------------------------------------------------------- #


def black76(
    forward: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
) -> float:
    """Black-76 price written on the forward (independent of carry split)."""
    if maturity <= 0.0 or volatility <= 0.0:
        intrinsic = forward - strike if option_type == "call" else strike - forward
        return math.exp(-rate * maturity) * max(intrinsic, 0.0)
    sd = volatility * math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    df = math.exp(-rate * maturity)
    if option_type == "call":
        return df * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    return df * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def bs_via_black76(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    volatility: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes obtained by routing through the forward and Black-76."""
    forward = spot * math.exp((rate - dividend) * maturity)
    return black76(forward, strike, maturity, rate, volatility, option_type)


def lognormal_expectation(
    payoff: Callable[[np.ndarray], np.ndarray],
    spot: float,
    maturity: float,
    rate: float,
    dividend: float,
    volatility: float,
    *,
    breakpoints: Sequence[float] = (),
    n_sd: float = 12.0,
) -> float:
    """E[payoff(S_T)] under risk-neutral GBM by adaptive quadrature in log-space."""
    mu = math.log(spot) + (rate - dividend - 0.5 * volatility**2) * maturity
    sd = volatility * math.sqrt(maturity)

    def integrand(x: float) -> float:
        value = payoff(np.asarray([math.exp(x)]))[0]
        return (
            float(value)
            * math.exp(-0.5 * ((x - mu) / sd) ** 2)
            / (sd * math.sqrt(2.0 * math.pi))
        )

    lo, hi = mu - n_sd * sd, mu + n_sd * sd
    points = sorted(
        {math.log(p) for p in breakpoints if p > 0.0 and lo < math.log(p) < hi}
    )
    edges = [lo, *points, hi]
    total = 0.0
    for a, b in itertools.pairwise(edges):
        value, _ = quad(integrand, a, b, limit=400, epsabs=1e-13, epsrel=1e-12)
        total += value
    return total


def bs_quadrature(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    volatility: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes price by direct integration of the terminal payoff."""
    sign = 1.0 if option_type == "call" else -1.0
    expectation = lognormal_expectation(
        lambda s: np.maximum(sign * (s - strike), 0.0),
        spot,
        maturity,
        rate,
        dividend,
        volatility,
        breakpoints=(strike,),
    )
    return math.exp(-rate * maturity) * expectation


def bs_analytic_greeks(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    volatility: float,
    option_type: str = "call",
) -> dict[str, float]:
    """Closed-form Black-Scholes sensitivities (theta is dV/dt, calendar time)."""
    sd = volatility * math.sqrt(maturity)
    d1 = (
        math.log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * maturity
    ) / sd
    d2 = d1 - sd
    dq = math.exp(-dividend * maturity)
    dr = math.exp(-rate * maturity)
    pdf = norm.pdf(d1)
    sign = 1.0 if option_type == "call" else -1.0
    delta = sign * dq * norm.cdf(sign * d1)
    gamma = dq * pdf / (spot * sd)
    vega = spot * dq * pdf * math.sqrt(maturity)
    rho = sign * strike * maturity * dr * norm.cdf(sign * d2)
    theta = (
        -spot * dq * pdf * volatility / (2.0 * math.sqrt(maturity))
        - sign * rate * strike * dr * norm.cdf(sign * d2)
        + sign * dividend * spot * dq * norm.cdf(sign * d1)
    )
    price = sign * (spot * dq * norm.cdf(sign * d1) - strike * dr * norm.cdf(sign * d2))
    return {
        "pv": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "rho": rho,
        "theta": theta,
    }


def richardson_derivative(
    f: Callable[[float], float], x: float, h: float, order: int = 1
) -> float:
    """Richardson-extrapolated central difference (first or second derivative)."""

    def d(step: float) -> float:
        if order == 1:
            return (f(x + step) - f(x - step)) / (2.0 * step)
        return (f(x + step) - 2.0 * f(x) + f(x - step)) / step**2

    coarse, fine = d(h), d(h / 2.0)
    return (4.0 * fine - coarse) / 3.0


# --------------------------------------------------------------------------- #
# Indexed-annuity payoffs written independently of ``alm_hedging.contracts``
# --------------------------------------------------------------------------- #


def rila_buffered_credit(
    r: np.ndarray, buffer: float, cap: float, participation: float, floor: float
) -> np.ndarray:
    up = np.minimum(participation * np.maximum(r, 0.0), cap)
    down = np.where(r < -buffer, r + buffer, 0.0)
    return np.maximum(up + down, floor)


def rila_floored_credit(
    r: np.ndarray, floor: float, cap: float, participation: float
) -> np.ndarray:
    return np.minimum(np.maximum(participation * r, floor), cap)


def fia_annual_credit(
    r: np.ndarray, cap: float, floor: float, participation: float, spread: float
) -> np.ndarray:
    return np.minimum(np.maximum(participation * r - spread, floor), cap)


def terminal_liability_quad(
    credit: Callable[[np.ndarray], np.ndarray],
    notional: float,
    index_start: float,
    spot: float,
    maturity: float,
    rate: float,
    dividend: float,
    volatility: float,
    breakpoints: Sequence[float],
) -> float:
    """PV of notional * (1 + credit(S_T / index_start - 1)) by quadrature."""
    expectation = lognormal_expectation(
        lambda s: notional * (1.0 + credit(s / index_start - 1.0)),
        spot,
        maturity,
        rate,
        dividend,
        volatility,
        breakpoints=breakpoints,
    )
    return math.exp(-rate * maturity) * expectation


def rila_buffered_call_put_decomposition(
    notional: float,
    index_start: float,
    spot: float,
    maturity: float,
    rate: float,
    dividend: float,
    volatility: float,
    buffer: float,
    cap: float,
    participation: float,
    floor: float,
) -> float:
    """Textbook decomposition: bond + p * call spread - put spread (buffer, floor).

    credit(r) = min(p * max(r, 0), cap) + min(r + b, 0), floored at ``floor``.
    In price terms with S0 = index_start, the floor binds at S0 (1 + floor - b).
    """
    s0 = index_start

    def call(k: float) -> float:
        return bs_via_black76(spot, k, maturity, rate, dividend, volatility, "call")

    def put(k: float) -> float:
        return bs_via_black76(spot, k, maturity, rate, dividend, volatility, "put")

    df = math.exp(-rate * maturity)
    k_cap = s0 * (1.0 + cap / participation)
    k_buf = s0 * (1.0 - buffer)
    k_floor = s0 * (1.0 + floor - buffer)
    upside = participation * (call(s0) - call(k_cap)) / s0
    downside = -(put(k_buf) - (put(k_floor) if k_floor > 0.0 else 0.0)) / s0
    return notional * (df + upside + downside)


def cliquet_closed_form(
    notional: float,
    n_years: int,
    cap: float,
    floor: float,
    participation: float,
    spread: float,
    index_start: float,
    spot: float,
    rate: float,
    dividend: float,
    volatility: float,
) -> float:
    """Annual point-to-point ratchet under flat GBM parameters.

    Independent increments make the expected compounded credit a product of
    one-year expectations.  Year one is struck off ``index_start``; later years
    are forward starting and spot-independent.
    """

    def credit(r: np.ndarray) -> np.ndarray:
        return fia_annual_credit(r, cap, floor, participation, spread)

    knots = (
        index_start * (1.0 + (floor + spread) / participation),
        index_start * (1.0 + (cap + spread) / participation),
    )
    year_one = lognormal_expectation(
        lambda s: 1.0 + credit(s / index_start - 1.0),
        spot,
        1.0,
        rate,
        dividend,
        volatility,
        breakpoints=knots,
    )
    forward_start = lognormal_expectation(
        lambda s: 1.0 + credit(s - 1.0),
        1.0,
        1.0,
        rate,
        dividend,
        volatility,
        breakpoints=(
            1.0 + (floor + spread) / participation,
            1.0 + (cap + spread) / participation,
        ),
    )
    return (
        notional * math.exp(-rate * n_years) * year_one * forward_start ** (n_years - 1)
    )


def iul_expected_pv(
    notional: float,
    n_years: int,
    cap: float,
    floor: float,
    participation: float,
    annual_policy_charge: float,
    monthly_expense_charge: float,
    mortality_rates: Sequence[float],
    lapse_rates: Sequence[float],
    death_benefit_multiple: float,
    index_start: float,
    spot: float,
    rate: float,
    dividend: float,
    volatility: float,
    *,
    steps_per_year: int = 12,
) -> float:
    """Deterministic expected-PV recursion for the IUL account value.

    Mirrors the documented event order within each monthly step - charges,
    anniversary credit, then independent decrements - but replaces path
    simulation with expectations.  Market and decrement independence makes
    E[AV * 1{exit}] = E[AV] * P(exit).
    """

    def credit(r: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(participation * r, floor), cap)

    e_year_one = lognormal_expectation(
        lambda s: 1.0 + credit(s / index_start - 1.0),
        spot,
        1.0,
        rate,
        dividend,
        volatility,
        breakpoints=(
            index_start * (1.0 + floor / participation),
            index_start * (1.0 + cap / participation),
        ),
    )
    e_forward = lognormal_expectation(
        lambda s: 1.0 + credit(s - 1.0),
        1.0,
        1.0,
        rate,
        dividend,
        volatility,
        breakpoints=(1.0 + floor / participation, 1.0 + cap / participation),
    )
    dt = 1.0 / steps_per_year
    expected_av = notional
    survival = 1.0
    pv = 0.0
    n_steps = n_years * steps_per_year
    for step in range(1, n_steps + 1):
        t = step * dt
        expected_av = max(
            expected_av * (1.0 - annual_policy_charge * dt)
            - monthly_expense_charge * 12.0 * dt,
            0.0,
        )
        if step % steps_per_year == 0:
            expected_av *= e_year_one if step == steps_per_year else e_forward
        year = min(int((step - 1) * dt), n_years - 1)
        qx = mortality_rates[min(year, len(mortality_rates) - 1)]
        lx = lapse_rates[min(year, len(lapse_rates) - 1)]
        p_death = 1.0 - (1.0 - qx) ** dt
        p_lapse = (1.0 - p_death) * (1.0 - (1.0 - lx) ** dt)
        df = math.exp(-rate * t)
        pv += survival * expected_av * df * (p_death * death_benefit_multiple + p_lapse)
        survival *= 1.0 - p_death - p_lapse
    pv += survival * expected_av * math.exp(-rate * n_years)
    return pv


# --------------------------------------------------------------------------- #
# Independent Monte Carlo
# --------------------------------------------------------------------------- #


def gbm_terminal_sample(
    spot: float,
    maturity: float,
    rate: float,
    dividend: float,
    volatility: float,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Terminal GBM sample from an independent legacy RNG stream."""
    rng = np.random.RandomState(seed)
    z = rng.standard_normal(n_paths)
    return spot * np.exp(
        (rate - dividend - 0.5 * volatility**2) * maturity
        + volatility * math.sqrt(maturity) * z
    )


def mc_mean_and_se(values: np.ndarray) -> tuple[float, float]:
    """Sample mean and standard error."""
    return float(np.mean(values)), float(
        np.std(values, ddof=1) / math.sqrt(values.size)
    )


# --------------------------------------------------------------------------- #
# Hedge optimization references
# --------------------------------------------------------------------------- #


def weighted_least_squares_hedge(
    target: np.ndarray, matrix: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Unconstrained minimizer of ||w * (target + M q)||^2 via normal equations."""
    a = matrix * weights[:, None]
    b = -target * weights
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    return solution


def brute_force_lot_hedge(
    objective: Callable[[np.ndarray], float],
    feasible: Callable[[np.ndarray], bool],
    bounds: Sequence[tuple[float, float]],
    lots: Sequence[float],
) -> tuple[np.ndarray, float]:
    """Exhaustive lot-grid search for small problems (ground truth)."""
    grids = [
        np.arange(math.ceil(lo / lot) * lot, math.floor(hi / lot) * lot + lot / 2, lot)
        for (lo, hi), lot in zip(bounds, lots, strict=True)
    ]
    best, best_value = None, math.inf
    for point in itertools.product(*grids):
        q = np.array(point, dtype=float)
        if not feasible(q):
            continue
        value = objective(q)
        if value < best_value:
            best, best_value = q, value
    if best is None:
        raise ValueError("no feasible lattice point")
    return best, best_value


def discrete_cvar(losses: np.ndarray, alpha: float) -> float:
    """Average of the worst ceil((1 - alpha) n) losses."""
    k = max(1, math.ceil((1.0 - alpha) * losses.size))
    return float(np.mean(np.sort(losses)[-k:]))


def scenario_effectiveness(
    liability_loss: np.ndarray, hedge_pnl: np.ndarray
) -> dict[str, float]:
    """Variance, dollar-offset, and tail measures of hedge effectiveness.

    ``liability_loss`` is the insurer's loss (increase in liability value) and
    ``hedge_pnl`` the asset gain in the same scenarios.  Variance reduction is
    1 - Var(loss - hedge) / Var(loss); the dollar-offset ratio is the slope of
    hedge P&L on liability loss; the tail ratio compares 95% expected shortfall.
    """
    net = liability_loss - hedge_pnl
    var_l = float(np.var(liability_loss))
    var_n = float(np.var(net))
    slope = float(np.cov(hedge_pnl, liability_loss)[0, 1] / var_l) if var_l else 0.0
    es_l = discrete_cvar(liability_loss, 0.95)
    es_n = discrete_cvar(net, 0.95)
    return {
        "variance_reduction": 1.0 - var_n / var_l if var_l else 0.0,
        "dollar_offset_slope": slope,
        "es95_unhedged": es_l,
        "es95_hedged": es_n,
        "es95_reduction": 1.0 - es_n / es_l if es_l else 0.0,
    }
