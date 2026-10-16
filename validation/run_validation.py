"""Independent model validation runner for the ALM hedging framework.

Runs every benchmark described in ``MODEL_VALIDATION_REPORT.md`` and prints
Markdown tables of expected versus actual values, errors, tolerances, and
pass/fail status.  Usage::

    python validation/run_validation.py            # full run (~1-2 minutes)
    python validation/run_validation.py --quick    # smaller Monte Carlo sizes
    python validation/run_validation.py --out results.md

Only ``validation/reference.py`` is used for benchmarks; the production engine
is exercised exclusively through the public ``alm_hedging`` API.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import alm_hedging as ah  # noqa: E402
from alm_hedging import api  # noqa: E402
from validation import reference as ref  # noqa: E402


@dataclass
class Check:
    section: str
    name: str
    expected: float
    actual: float
    tolerance: float
    note: str = ""

    @property
    def error(self) -> float:
        return self.actual - self.expected

    @property
    def informational(self) -> bool:
        return not math.isfinite(self.tolerance)

    @property
    def passed(self) -> bool:
        return self.informational or abs(self.error) <= self.tolerance


CHECKS: list[Check] = []


def record(
    section: str,
    name: str,
    expected: float,
    actual: float,
    tolerance: float,
    note: str = "",
) -> Check:
    check = Check(section, name, float(expected), float(actual), float(tolerance), note)
    CHECKS.append(check)
    return check


def flat_market(
    spot: float = 100.0, rate: float = 0.03, dividend: float = 0.01, vol: float = 0.20
) -> ah.MarketState:
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    moneyness = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    return ah.MarketState(
        spot,
        ah.ZeroCurve(tenors, np.full_like(tenors, rate)),
        ah.ZeroCurve(tenors, np.full_like(tenors, dividend)),
        ah.VolSurface(tenors, moneyness, np.full((tenors.size, moneyness.size), vol)),
    )


FLAT = (0.03, 0.01, 0.20)


# --------------------------------------------------------------------------- #
# A. Vanilla option pricing
# --------------------------------------------------------------------------- #


def section_vanilla() -> None:
    cases = [
        (100.0, 100.0, 1.0, 0.03, 0.01, 0.20, "call"),
        (100.0, 80.0, 0.25, 0.05, 0.00, 0.35, "put"),
        (100.0, 130.0, 5.0, 0.04, 0.02, 0.15, "call"),
        (5250.0, 5250.0, 1.0, 0.0425, 0.015, 0.185, "call"),
        (100.0, 100.0, 0.01, 0.03, 0.01, 0.20, "put"),
        (100.0, 60.0, 2.0, 0.03, 0.01, 0.50, "call"),
    ]
    for spot, strike, t, r, q, v, kind in cases:
        actual = ah.black_scholes_price(spot, strike, t, r, q, v, kind)
        label = f"{kind} S={spot:g} K={strike:g} T={t:g}"
        record(
            "A. Vanilla pricing",
            f"BS vs lognormal quadrature: {label}",
            ref.bs_quadrature(spot, strike, t, r, q, v, kind),
            actual,
            1e-9 * max(actual, 1.0),
        )
        record(
            "A. Vanilla pricing",
            f"BS vs Black-76 on forward: {label}",
            ref.bs_via_black76(spot, strike, t, r, q, v, kind),
            actual,
            1e-12 * max(actual, 1.0),
        )
    call = ah.black_scholes_price(100.0, 105.0, 2.0, 0.035, 0.012, 0.24, "call")
    put = ah.black_scholes_price(100.0, 105.0, 2.0, 0.035, 0.012, 0.24, "put")
    record(
        "A. Vanilla pricing",
        "Put-call parity C-P = S e^-qT - K e^-rT",
        100.0 * math.exp(-0.012 * 2.0) - 105.0 * math.exp(-0.035 * 2.0),
        call - put,
        1e-11,
    )
    record(
        "A. Vanilla pricing",
        "Zero-vol limit equals discounted forward intrinsic",
        max(100.0 * math.exp(-0.01) - 95.0 * math.exp(-0.03), 0.0),
        ah.black_scholes_price(100.0, 95.0, 1.0, 0.03, 0.01, 0.0),
        1e-12,
    )
    record(
        "A. Vanilla pricing",
        "Expiry returns intrinsic",
        10.0,
        ah.black_scholes_price(110.0, 100.0, 0.0, 0.03, 0.01, 0.2),
        0.0,
    )
    market = flat_market()
    record(
        "A. Vanilla pricing",
        "market_option_price equals BS at interpolated (r,q,sigma)",
        ref.bs_via_black76(100.0, 110.0, 3.0, *FLAT, "put"),
        ah.market_option_price(market, 110.0, 3.0, "put"),
        1e-10,
    )


# --------------------------------------------------------------------------- #
# B. Indexed liabilities
# --------------------------------------------------------------------------- #


def section_liabilities(quick: bool) -> None:
    market = flat_market()
    r, q, v = FLAT
    sec = "B. Liability payoffs & replication"
    buffered = [
        (0.10, 0.18, 1.0, -1.0, 1.0),
        (0.10, 0.15, 1.08, -0.10, 4.5),
        (0.20, 0.25, 0.9, -0.5, 6.0),
        (0.0, 0.12, 1.0, -0.1, 1.0),
    ]
    for buffer, cap, part, floor, t in buffered:
        contract = ah.RILABufferedSegment(100.0, t, buffer, cap, part, floor)
        actual = ah.analytic_liability_value(contract, market)
        knots = [
            100.0 * (1.0 + floor - buffer),
            100.0 * (1.0 - buffer),
            100.0,
            100.0 * (1.0 + cap / part),
        ]
        label = f"buffer={buffer:g} cap={cap:g} p={part:g} floor={floor:g} T={t:g}"
        record(
            sec,
            f"RILA buffered analytic vs quadrature: {label}",
            ref.terminal_liability_quad(
                lambda x, b=buffer, c=cap, pr=part, f=floor: ref.rila_buffered_credit(
                    x, b, c, pr, f
                ),
                100.0,
                100.0,
                100.0,
                t,
                r,
                q,
                v,
                knots,
            ),
            actual,
            1e-9,
        )
        record(
            sec,
            f"RILA buffered analytic vs bond+call-spread-put-spread: {label}",
            ref.rila_buffered_call_put_decomposition(
                100.0, 100.0, 100.0, t, r, q, v, buffer, cap, part, floor
            ),
            actual,
            1e-10,
        )
    for floor, cap, part, t in [(-0.10, 0.15, 1.0, 1.0), (0.0, 0.08, 0.75, 1.0)]:
        contract = ah.RILAFlooredSegment(100.0, t, floor, cap, part)
        record(
            sec,
            f"RILA floored analytic vs quadrature: floor={floor:g} cap={cap:g} p={part:g}",
            ref.terminal_liability_quad(
                lambda x, f=floor, c=cap, pr=part: ref.rila_floored_credit(x, f, c, pr),
                100.0,
                100.0,
                100.0,
                t,
                r,
                q,
                v,
                [100.0 * (1.0 + floor / part), 100.0 * (1.0 + cap / part)],
            ),
            ah.analytic_liability_value(contract, market),
            1e-9,
        )
    for cap, part, spread, floor in [(0.095, 0.85, 0.0, 0.0), (0.10, 1.0, 0.02, 0.0)]:
        contract = ah.FIAAnnualPointToPoint(100.0, 1.0, cap, part, spread, floor)
        record(
            sec,
            f"FIA 1y analytic vs quadrature: cap={cap:g} p={part:g} spread={spread:g}",
            ref.terminal_liability_quad(
                lambda x, c=cap, f=floor, pr=part, sp=spread: ref.fia_annual_credit(
                    x, c, f, pr, sp
                ),
                100.0,
                100.0,
                100.0,
                1.0,
                r,
                q,
                v,
                [
                    100.0 * (1.0 + (floor + spread) / part),
                    100.0 * (1.0 + (cap + spread) / part),
                ],
            ),
            ah.analytic_liability_value(contract, market),
            1e-9,
        )
    # Static replication identity on a dense terminal grid.
    contract = ah.RILABufferedSegment(100.0, 1.0, 0.10, 0.18, 1.0, -0.5)
    knots = np.array([40.0, 90.0, 100.0, 118.0])
    legs = ah.piecewise_linear_replication(
        knots, 100.0 * (1.0 + contract.credited_return(knots / 100.0 - 1.0))
    )
    terminal = np.linspace(1.0, 300.0, 3000)
    replicated = np.zeros_like(terminal)
    for leg in legs:
        if leg.kind == "cash":
            replicated += leg.quantity
        elif leg.kind == "underlying":
            replicated += leg.quantity * terminal
        else:
            assert leg.strike is not None
            replicated += leg.quantity * np.maximum(terminal - leg.strike, 0.0)
    payoff = 100.0 * (
        1.0 + ref.rila_buffered_credit(terminal / 100.0 - 1.0, 0.10, 0.18, 1.0, -0.5)
    )
    record(
        sec,
        "Replication legs reproduce payoff on [1, 300] (max abs error)",
        0.0,
        float(np.max(np.abs(replicated - payoff))),
        1e-10,
    )

    # Monte Carlo consistency with analytic pricing on the sloped synthetic market.
    sec_mc = "B. Monte Carlo consistency & convergence"
    sloped = ah.SyntheticMarketModel(skew=0.0).build()
    n_paths = 50_000 if quick else 200_000
    seeds = range(2 if quick else 6)
    for t in (1.0, 2.0, 5.0):
        analytic = ah.market_option_price(sloped, 100.0, t, "call")
        estimates = []
        for seed in seeds:
            _, paths = ah.simulate_gbm_paths(sloped, t, n_paths, 12, seed=seed)
            estimates.append(
                float(sloped.rate_curve.discount(t))
                * np.maximum(paths[:, -1] - 100.0, 0.0).mean()
            )
        se = float(np.std(estimates, ddof=1) / math.sqrt(len(estimates)))
        record(
            sec_mc,
            f"ATM call T={t:g}: MC (forward variance/rates) vs analytic on sloped curves",
            analytic,
            float(np.mean(estimates)),
            max(3.0 * se, 1e-3 * analytic),
            f"se={se:.4f}",
        )
        log_var = float(np.var(np.log(paths[:, -1] / 100.0)))
        record(
            sec_mc,
            f"Terminal log-variance equals sigma_atm(T)^2 T, T={t:g}",
            sloped.volatility(100.0, t) ** 2 * t,
            log_var,
            0.01 * sloped.volatility(100.0, t) ** 2 * t,
        )
        forward = (
            100.0
            * float(sloped.dividend_curve.discount(t))
            / float(sloped.rate_curve.discount(t))
        )
        record(
            sec_mc,
            f"E[S_T] equals market forward, T={t:g}",
            forward,
            float(np.mean(paths[:, -1])),
            0.002 * forward,
        )
    # Multi-year FIA against the cliquet closed form.
    for t, cap, part, spread in [
        (2, 0.10, 1.0, 0.0),
        (5, 0.095, 0.85, 0.0),
        (10, 0.09, 0.72, 0.0),
    ]:
        contract = ah.FIAAnnualPointToPoint(100.0, float(t), cap, part, spread, 0.0)
        closed = ref.cliquet_closed_form(
            100.0, t, cap, 0.0, part, spread, 100.0, 100.0, r, q, v
        )
        estimates = [
            ah.monte_carlo_value(
                contract, market, n_paths=40_000 if quick else 100_000, seed=s
            )
            for s in range(4 if quick else 8)
        ]
        se = float(np.std(estimates, ddof=1) / math.sqrt(len(estimates)))
        record(
            sec_mc,
            f"FIA annual ratchet {t}y: MC vs independent closed form",
            closed,
            float(np.mean(estimates)),
            max(3.0 * se, 5e-4 * closed),
            f"se={se:.4f} (tolerance floor 5e-4 relative)",
        )
    # IUL against the deterministic expected-value recursion.
    mortality = (0.003, 0.0032, 0.0035)
    lapse = (0.035, 0.03, 0.025)
    for t, multiple in [(5, 1.0), (10, 1.5), (20, 1.0)]:
        contract = ah.IULAccountValue(
            85_000.0, float(t), 0.11, 1.0, 0.0, 0.0125, 8.0, mortality, lapse, multiple
        )
        expected = ref.iul_expected_pv(
            85_000.0,
            t,
            0.11,
            0.0,
            1.0,
            0.0125,
            8.0,
            mortality,
            lapse,
            multiple,
            100.0,
            100.0,
            r,
            q,
            v,
        )
        estimates = [
            ah.monte_carlo_value(
                contract, market, n_paths=40_000 if quick else 100_000, seed=s
            )
            for s in range(4 if quick else 6)
        ]
        se = float(np.std(estimates, ddof=1) / math.sqrt(len(estimates)))
        record(
            sec_mc,
            f"IUL {t}y (DB multiple {multiple:g}): MC vs deterministic recursion",
            expected,
            float(np.mean(estimates)),
            max(3.0 * se, 1e-3 * expected),
            f"se={se:.1f} (tolerance floor 1e-3 relative)",
        )
    cohort = ah.Cohort(contract, 10, mortality_multiplier=2.0, lapse_multiplier=0.5)
    expected = 10.0 * ref.iul_expected_pv(
        85_000.0,
        20,
        0.11,
        0.0,
        1.0,
        0.0125,
        8.0,
        tuple(2.0 * x for x in mortality),
        tuple(0.5 * x for x in lapse),
        1.0,
        100.0,
        100.0,
        r,
        q,
        v,
    )
    actual = ah.monte_carlo_value(
        cohort, market, n_paths=20_000 if quick else 100_000, seed=0
    )
    record(
        sec_mc,
        "IUL cohort behavior multipliers applied (x2 mortality, x0.5 lapse)",
        expected,
        actual,
        2e-3 * expected,
    )
    # Convergence: RMSE should scale like n^-1/2; CI coverage near 95%.
    analytic = ah.black_scholes_price(100.0, 100.0, 1.0, r, q, v)
    rmse = {}
    for n in (1_000, 16_000, 256_000 if not quick else 64_000):
        est = []
        for seed in range(20):
            _, paths = ah.simulate_gbm_paths(market, 1.0, n, 1, seed=seed)
            est.append(math.exp(-r) * np.maximum(paths[:, -1] - 100.0, 0.0).mean())
        rmse[n] = float(np.sqrt(np.mean((np.array(est) - analytic) ** 2)))
    sizes = sorted(rmse)
    slope = math.log(rmse[sizes[-1]] / rmse[sizes[0]]) / math.log(sizes[-1] / sizes[0])
    record(
        sec_mc,
        "RMSE convergence order (log-log slope vs paths, expect -0.5)",
        -0.5,
        slope,
        0.15,
        " ".join(f"n={n}:rmse={rmse[n]:.4f}" for n in sizes),
    )
    covered = 0
    trials = 100 if quick else 200
    for seed in range(trials):
        _, paths = ah.simulate_gbm_paths(market, 1.0, 20_000, 1, seed=seed)
        pay = math.exp(-r) * np.maximum(paths[:, -1] - 100.0, 0.0)
        pairs = 0.5 * (pay[:10_000] + pay[10_000:])
        mu, se = ref.mc_mean_and_se(pairs)
        covered += abs(mu - analytic) < 1.96 * se
    record(
        sec_mc,
        f"95% CI coverage over {trials} seeds (antithetic-pair SE)",
        0.95,
        covered / trials,
        0.05,
    )
    reproducible = ah.monte_carlo_value(
        contract, market, n_paths=2_000, seed=3
    ) - ah.monte_carlo_value(contract, market, n_paths=2_000, seed=3)
    record(sec_mc, "Seeded replay is bitwise reproducible", 0.0, reproducible, 0.0)


# --------------------------------------------------------------------------- #
# C. Option budget solvers
# --------------------------------------------------------------------------- #


def section_budget() -> None:
    sec = "C. Option budget solvers"
    market = flat_market()
    r, q, v = FLAT
    solution = ah.solve_call_spread_cap(
        market, 1.0, 0.04, participation=1.0, notional=1.0
    )
    cap = solution.parameter
    repriced = (
        ref.bs_via_black76(100.0, 100.0, 1.0, r, q, v)
        - ref.bs_via_black76(100.0, 100.0 * (1.0 + cap), 1.0, r, q, v)
    ) / 100.0
    record(
        sec,
        "Cap solved for 4% budget reprices to budget (independent BS)",
        0.04,
        repriced,
        1e-9,
        f"cap={cap:.6f}",
    )
    liability = ah.FIAAnnualPointToPoint(1.0, 1.0, cap, 1.0, 0.0, 0.0)
    record(
        sec,
        "Solved cap: FIA 1y analytic value = bond + budget",
        math.exp(-r) + 0.04,
        ah.analytic_liability_value(liability, market),
        1e-9,
    )
    participation = ah.solve_participation_rate(market, 1.0, 0.05, notional=1.0)
    record(
        sec,
        "Participation solved for 5% ATM-call budget reprices to budget",
        0.05,
        participation.parameter
        * ref.bs_via_black76(100.0, 100.0, 1.0, r, q, v)
        / 100.0,
        1e-9,
        f"p={participation.parameter:.6f}",
    )
    unreachable = ah.solve_call_spread_cap(
        market, 1.0, 0.50, participation=1.0, notional=1.0
    )
    record(
        sec,
        "Unreachable budget flagged unconverged (converged flag == 0)",
        0.0,
        float(unreachable.converged),
        0.0,
        f"residual={unreachable.residual:.4f}",
    )


# --------------------------------------------------------------------------- #
# D. Greeks and stresses
# --------------------------------------------------------------------------- #


def section_greeks(quick: bool) -> None:
    sec = "D. Greeks: finite differences vs analytic"
    market = flat_market()
    r, q, v = FLAT
    for strike, t, kind in [
        (100.0, 1.0, "call"),
        (90.0, 0.5, "put"),
        (120.0, 2.0, "call"),
    ]:
        greeks = ah.bump_and_revalue(
            lambda s, k=strike, tt=t, kd=kind: ah.market_option_price(s, k, tt, kd),
            market,
        )
        exact = ref.bs_analytic_greeks(100.0, strike, t, r, q, v, kind)
        label = f"{kind} K={strike:g} T={t:g}"
        tolerances = {
            "delta": 5e-4,
            "gamma": 1e-4,
            "vega": 2e-2,
            "rho": 1e-5,
            "theta": 3e-2,
        }
        for name, tol in tolerances.items():
            record(sec, f"{name} {label}", exact[name], getattr(greeks, name), tol)
        record(
            sec,
            f"dv01 {label} (forward difference per bp)",
            exact["rho"] * 1e-4,
            greeks.dv01,
            1e-5,
        )
    exact = ref.bs_analytic_greeks(100.0, 100.0, 1.0, r, q, v)
    errors = {}
    for h in (0.02, 0.01, 0.005):
        g = ah.bump_and_revalue(
            lambda s: ah.market_option_price(s, 100.0, 1.0),
            market,
            ah.BumpSizes(spot_relative=h, vol_absolute=h, rate_bp=h * 100),
        )
        errors[h] = {k: getattr(g, k) - exact[k] for k in ("delta", "gamma", "vega")}
    for name in ("delta", "gamma", "vega"):
        ratio = errors[0.02][name] / errors[0.01][name]
        record(
            "D. Greeks: bump-size stability",
            f"{name} error ratio when halving bump 2%->1% (expect 4 for O(h^2))",
            4.0,
            ratio,
            0.3,
        )
        ratio = errors[0.01][name] / errors[0.005][name]
        record(
            "D. Greeks: bump-size stability",
            f"{name} error ratio when halving bump 1%->0.5% (expect 4)",
            4.0,
            ratio,
            0.3,
        )
    # Sticky-moneyness convention on the skewed synthetic surface.
    skewed = ah.SyntheticMarketModel(term_vol=0.0).build()
    vol = skewed.volatility(100.0, 1.0)
    rate = float(skewed.rate_curve.zero_rate(1.0))
    dividend = float(skewed.dividend_curve.zero_rate(1.0))
    g = ah.bump_and_revalue(lambda s: ah.market_option_price(s, 100.0, 1.0), skewed)
    exact = ref.bs_analytic_greeks(100.0, 100.0, 1.0, rate, dividend, vol)
    record(
        "D. Greeks: volatility-surface convention",
        "Skewed surface delta = BS delta + vega * dsigma/dS (sticky moneyness)",
        exact["delta"] + exact["vega"] * 0.08 / 100.0,
        g.delta,
        5e-4,
        f"sticky-strike BS delta would be {exact['delta']:.4f}",
    )
    # Aggregation.
    cohorts = [
        ah.Cohort(ah.RILAFlooredSegment(100.0, 1.0, -0.10, 0.15), 10),
        ah.Cohort(ah.RILABufferedSegment(80.0, 2.0, 0.1, 0.2), 20),
    ]
    total = ah.bump_and_revalue(lambda s: ah.portfolio_value(cohorts, s), market)
    parts = ah.aggregate_greeks(
        {
            str(i): (
                1.0,
                ah.bump_and_revalue(lambda s, c=c: ah.portfolio_value([c], s), market),
            )
            for i, c in enumerate(cohorts)
        }
    )
    for name, value in total.as_dict().items():
        record(
            "D. Greeks: aggregation",
            f"portfolio {name} equals sum of cohort {name}",
            parts.as_dict()[name],
            value,
            1e-8 * max(abs(value), 1.0),
        )
    # Economic signs for a long call-spread / short put-spread RILA liability.
    rila = ah.RILABufferedSegment(100.0, 1.0, 0.10, 0.18, 1.0, -1.0)
    g = ah.bump_and_revalue(lambda s: ah.analytic_liability_value(rila, s), market)
    record(
        "D. Greeks: sign conventions",
        "1y buffered RILA liability delta > 0 (sign check: 1=pass)",
        1.0,
        float(g.delta > 0),
        0.0,
        f"delta={g.delta:.4f}",
    )
    record(
        "D. Greeks: sign conventions",
        "1y buffered RILA liability DV01 < 0 (sign check: 1=pass)",
        1.0,
        float(g.dv01 < 0),
        0.0,
        f"dv01={g.dv01:.6f}",
    )
    record(
        "D. Greeks: sign conventions",
        "DV01 equals rho * 1e-4 to first order",
        g.rho * 1e-4,
        g.dv01,
        1e-5,
    )

    # Full-revaluation stresses reproduce direct repricing.
    sec_s = "D. Stress: full revaluation"
    scenario = ah.StressScenario(
        "crash", equity_relative=-0.4, vol_absolute=0.25, rate_bp=-75.0
    )
    stressed = scenario.apply_market(market)
    direct = sum(
        c.policy_count
        * ref.terminal_liability_quad(
            lambda x, c=c: (
                ref.rila_floored_credit(
                    x, c.contract.floor, c.contract.cap, c.contract.participation
                )
                if isinstance(c.contract, ah.RILAFlooredSegment)
                else ref.rila_buffered_credit(
                    x,
                    c.contract.buffer,
                    c.contract.cap,
                    c.contract.participation,
                    c.contract.floor,
                )
            ),
            c.contract.notional,
            100.0,
            60.0,
            c.contract.maturity,
            r - 0.0075,
            q,
            v + 0.25,
            [],
        )
        for c in cohorts
    )
    results = ah.run_stress_scenarios(
        market, (scenario,), lambda s, items: ah.portfolio_value(items, s), cohorts
    )
    record(
        sec_s,
        "Crash scenario stressed value equals independent repricing",
        direct,
        results[0].stressed_value,
        1e-6 * direct,
    )
    record(sec_s, "Stressed spot = S(1-40%)", 60.0, stressed.spot, 0.0)
    record(
        sec_s,
        "Stressed vol = sigma + 0.25",
        0.45,
        float(stressed.vol_surface.vols[0, 0]),
        1e-12,
    )
    record(
        sec_s,
        "Stressed rate = r - 75bp",
        0.0225,
        float(stressed.rate_curve.zero_rates[0]),
        1e-12,
    )
    totals = ah.aggregate_stress_results({"a": results, "b": results})
    record(
        sec_s,
        "aggregate_stress_results sums base values",
        2.0 * results[0].base_value,
        totals["crash"].base_value,
        1e-9,
    )

    # Monte Carlo Greek noise on the dashboard portfolio (seed sensitivity).
    cohorts_api = dashboard_cohorts()
    market_api = dashboard_market()
    state = api._market(market_api)
    liabilities = api._liabilities(cohorts_api, state.spot)
    seeds = (46_011, 1, 2) if quick else (46_011, 1, 2, 3, 4)
    samples = [
        ah.bump_and_revalue(
            lambda s, seed=seed: ah.portfolio_value(
                liabilities, s, n_paths=2_000, seed=seed
            ),
            state,
        )
        for seed in seeds
    ]
    sec_n = "D. Greeks: Monte Carlo noise (dashboard, 2,000 paths)"
    for name in ("pv", "delta", "gamma", "vega", "dv01"):
        values = np.array([getattr(s, name) for s in samples])
        tolerance = {
            "pv": 0.005,
            "delta": 0.10,
            "gamma": math.inf,
            "vega": 0.10,
            "dv01": 0.01,
        }[name]
        record(
            sec_n,
            f"{name}: seed-to-seed coefficient of variation",
            0.0,
            float(np.std(values, ddof=1) / max(abs(np.mean(values)), 1e-12)),
            tolerance,
            f"mean={np.mean(values):.6g}"
            + (" (sign unstable; see materiality check)" if name == "gamma" else ""),
        )
    gammas = np.array([s.gamma for s in samples])
    pv = float(np.mean([s.pv for s in samples]))
    move = 0.10 * state.spot
    record(
        sec_n,
        "gamma noise materiality: 0.5 * sd(gamma) * (10% spot move)^2 / PV",
        0.0,
        0.5 * float(np.std(gammas, ddof=1)) * move**2 / pv,
        1e-3,
        "second-order term is statistically zero but economically immaterial",
    )


# --------------------------------------------------------------------------- #
# E. Hedging
# --------------------------------------------------------------------------- #


def dashboard_cohorts(
    account_value: float = 500e6, duration: float = 5.0
) -> list[dict[str, float | str]]:
    mix = [
        ("RILA", 0.40, 1.08, 0.15, -0.10, 0.0),
        ("FIA", 0.35, 0.72, 0.09, 0.0, 0.025),
        ("IUL", 0.25, 0.92, 0.11, 0.0, 0.03),
    ]
    return [
        {
            "product": name,
            "account_value": account_value * weight,
            "policies": round(25_000 * weight),
            "duration": duration * (0.9 + i * 0.12),
            "participation": part,
            "cap": cap,
            "floor": floor,
            "guarantee": max(0.02, guarantee),
            "weight": weight,
        }
        for i, (name, weight, part, cap, floor, guarantee) in enumerate(mix)
    ]


def dashboard_market() -> dict[str, float | str]:
    return {
        "spot": 5250.0,
        "rate": 0.0425,
        "volatility": 0.185,
        "dividend_yield": 0.015,
        "as_of": "2026-10-16",
    }


def section_hedging(quick: bool) -> None:
    sec = "E. Hedge optimizer"
    instruments = [
        ah.HedgeInstrument("fut", "future", 1.0, 100.0, {"delta": 1.0}, lot_size=1e-6),
        ah.HedgeInstrument(
            "call",
            "option",
            1.0,
            8.0,
            {"delta": 0.57, "vega": 38.7},
            premium=8.0,
            lot_size=1e-6,
        ),
        ah.HedgeInstrument(
            "put",
            "option",
            1.0,
            6.0,
            {"delta": -0.43, "vega": 38.7},
            premium=6.0,
            lot_size=1e-6,
        ),
    ]
    target = {"delta": -125.0, "vega": -5000.0, "rho": -300.0}
    weights = np.array([1.0, 0.01, 0.1])
    solution = ah.optimize_hedge(
        target,
        instruments,
        exposure_weights=dict(zip(target, weights, strict=True)),
        cost_penalty=0.0,
    )
    matrix = np.array([[1.0, 0.57, -0.43], [0.0, 38.7, 38.7], [0.0, 0.0, 0.0]])
    reference = ref.weighted_least_squares_hedge(
        np.array(list(target.values())), matrix, weights
    )
    for i, name in enumerate(("fut", "call", "put")):
        record(
            sec,
            f"Unconstrained solution matches weighted least squares: {name}",
            reference[i],
            solution.positions[name],
            1e-4,
        )
    record(
        sec,
        "Unhedgeable rho residual is left untouched",
        -300.0,
        solution.residual_exposures["rho"],
        1e-9,
    )

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
    target = {"delta": -80.0, "vega": -1500.0}
    weights_map = {"delta": 1.0, "vega": 0.02}
    constraints = ah.HedgeConstraints(premium_budget=250.0, turnover_budget=1e9)
    solution = ah.optimize_hedge(
        target,
        instruments,
        exposure_weights=weights_map,
        constraints=constraints,
        cost_penalty=1e-3,
    )
    m = np.array([[1.0, 0.6, -0.4], [0.0, 40.0, 35.0]])
    t = np.array([-80.0, -1500.0])
    w = np.array([1.0, 0.02])
    premiums = np.array([0.0, 8.0, 6.0])
    costs = np.array([0.5, 0.2, 0.2])

    def objective(qty: np.ndarray) -> float:
        return float(
            np.sum((w * (t + m @ qty)) ** 2) + 1e-3 * np.dot(np.abs(qty), costs)
        )

    best, best_value = ref.brute_force_lot_hedge(
        objective,
        lambda qty: bool(np.dot(np.abs(qty), premiums) <= 250.0 + 1e-9),
        [(-200, 200), (-60, 60), (-60, 60)],
        [5, 2, 2],
    )
    record(
        sec,
        "Lot-constrained objective with binding premium budget vs brute force",
        best_value,
        solution.objective,
        1e-9,
        f"brute={best.tolist()} solver={list(solution.positions.values())}",
    )
    record(
        sec,
        "Premium budget respected after rounding",
        1.0,
        float(solution.premium_used <= 250.0 + 1e-9),
        0.0,
        f"premium_used={solution.premium_used:g}",
    )
    losses = np.array([10.0, 20.0, 100.0, 5.0, 40.0, 60.0, 70.0, 15.0, 22.0, 90.0])
    record(
        sec,
        "Discrete CVaR(80%) of sample losses",
        ref.discrete_cvar(losses, 0.8),
        95.0,
        0.0,
    )
    put = ah.HedgeInstrument(
        "p",
        "option",
        1.0,
        5.0,
        {"delta": 0.0},
        premium=5.0,
        max_position=100,
        max_trade=100,
    )
    solution = ah.optimize_hedge(
        {"delta": 0.0},
        [put],
        constraints=ah.HedgeConstraints(premium_budget=500.0, turnover_budget=500.0),
        scenario_liability_losses=losses,
        instrument_scenario_pnls={"p": losses / 10.0},
        cvar_weight=1.0,
        cvar_alpha=0.8,
    )
    record(
        sec,
        "Reported stress CVaR equals independent CVaR of net losses",
        ref.discrete_cvar(losses - solution.positions["p"] * losses / 10.0, 0.8),
        solution.stress_cvar,
        1e-12,
    )

    # Dynamic hedge of a short call with futures along a GBM path.
    sec_d = "E. Dynamic hedge simulation"
    r, q, v = FLAT
    rng = np.random.RandomState(7)
    n, dt = 250, 1.0 / 250.0
    z = rng.standard_normal(n)
    spots = np.r_[
        100.0,
        100.0 * np.exp(np.cumsum((0.05 - 0.5 * v**2) * dt + v * math.sqrt(dt) * z)),
    ]
    markets = [
        flat_market(spot=float(s)).bumped(time_years=i * dt)
        for i, s in enumerate(spots)
    ]

    def liability(state: ah.MarketState) -> float:
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
    record(
        sec_d,
        "Initial futures position = +1000 x BS delta (buys futures against a liability with positive delta)",
        1000.0 * ref.bs_analytic_greeks(100.0, 100.0, 1.0, r, q, v)["delta"],
        float(h.iloc[0]["position_fut"]),
        1.0,
    )
    record(
        sec_d,
        "Transaction costs = sum |trade| x unit cost",
        0.01 * float(np.abs(np.diff(np.r_[0.0, h["position_fut"].values])).sum()),
        float(h["transaction_cost"].sum()),
        1e-9,
    )
    identity = (
        h["delta_pnl"]
        + h["gamma_pnl"]
        + h["vega_pnl"]
        + h["rho_pnl"]
        + h["theta_pnl"]
        + h["residual_pnl"]
        - h["transaction_cost"]
        - h["total_pnl"]
    )
    record(
        sec_d,
        "Attribution identity: sum of components - costs = total (max abs error)",
        0.0,
        float(np.max(np.abs(identity))),
        1e-9,
    )
    record(
        sec_d,
        "Daily delta hedge variance reduction on short call (expect > 0.99)",
        1.0,
        1.0 - float(h["total_pnl"].var() / h["liability_pnl"].var()),
        0.01,
    )
    unexplained = abs(float(h["residual_pnl"].sum())) / abs(
        float(h["liability_pnl"].abs().sum())
    )
    record(
        sec_d,
        "Unexplained residual after gamma/theta attribution (fraction of gross liability moves)",
        0.0,
        unexplained,
        0.01,
        f"theta={h['theta_pnl'].sum():.1f} gamma={h['gamma_pnl'].sum():.1f}",
    )

    # Dashboard hedge recommendation: sign convention and effectiveness.
    sec_e = "E. Dashboard hedge effectiveness"
    cohorts = dashboard_cohorts()
    market = dashboard_market()
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
    hedge_greeks = {
        k: sum(units[name] * templates[name].get(k, 0.0) for name in units)
        for k in ("delta", "vega")
    }
    record(
        sec_e,
        "Hedge delta has the same sign as liability delta (asset offsets short liability)",
        1.0,
        float(np.sign(hedge_greeks["delta"]) == np.sign(risks["delta"])),
        0.0,
        f"liability={risks['delta']:.0f} hedge={hedge_greeks['delta']:.0f}",
    )
    record(
        sec_e,
        "Hedge vega has the same sign as liability vega",
        1.0,
        float(np.sign(hedge_greeks["vega"]) == np.sign(risks["vega"])),
        0.0,
        f"liability={risks['vega']:.0f} hedge={hedge_greeks['vega']:.0f}",
    )
    record(
        sec_e,
        "Reported residual vega = liability vega - hedge vega",
        risks["vega"] - hedge_greeks["vega"],
        hedge["residual_vega"],
        1.0,
    )
    record(
        sec_e,
        "Premium used within budget",
        1.0,
        float(hedge["premium"] <= 0.0075 * 500e6 + 1e-6),
        0.0,
        f"premium={hedge['premium']:.0f}",
    )
    scale = 500e6
    reference_exposure = {"delta": 0.1 * scale / 5250.0, "vega": 0.1 * scale}
    wts = {
        k: 1.0 / max(abs(risks[k]), reference_exposure[k]) for k in ("delta", "vega")
    }
    initial = math.hypot(*(risks[k] * wts[k] for k in wts))
    residual = math.hypot(*((risks[k] - hedge_greeks[k]) * wts[k] for k in wts))
    record(
        sec_e,
        "Effectiveness equals 1 - ||scaled residual Greeks|| / ||scaled Greeks|| (independent recomputation)",
        1.0 - residual / initial,
        hedge["effectiveness"],
        1e-9,
    )
    # Linear scenario test of the recommendation (instrument Greeks are the template constants).
    rng = np.random.RandomState(1)
    n = 5_000
    ds = 5250.0 * rng.normal(0.0, 0.05, n)
    dv = rng.normal(0.0, 0.03, n) - 0.09 * ds / 5250.0
    dr = rng.normal(0.0, 0.0025, n)
    loss = (
        risks["delta"] * ds
        + 0.5 * risks["gamma"] * ds**2
        + risks["vega"] * dv
        + risks["rho"] * dr
    )
    gain = hedge_greeks["delta"] * ds + hedge_greeks["vega"] * dv
    metrics = ref.scenario_effectiveness(loss, gain)
    record(
        sec_e,
        "Scenario variance reduction of recommended hedge vs reported Greek effectiveness (linear, joint spot/vol/rate shocks)",
        hedge["effectiveness"],
        metrics["variance_reduction"],
        math.inf,
        f"dollar-offset slope={metrics['dollar_offset_slope']:.2f} ES95 reduction={metrics['es95_reduction']:.2f}",
    )

    # Fully consistent out-of-sample experiment: a RILA cohort hedged with
    # market-priced vanilla options, full revaluation of both sides.
    sec_f = "E. Out-of-sample full-revaluation effectiveness (RILA cohort, market-priced hedges)"
    base = ah.SyntheticMarketModel(base_rate=0.04, atm_vol=0.20).build(spot=100.0)
    cohort = ah.Cohort(
        ah.RILABufferedSegment(1_000.0, 1.0, 0.10, 0.15, 1.0, -0.10), 1_000
    )

    def liability_value(state: ah.MarketState) -> float:
        return ah.analytic_liability_value(cohort, state)

    hedge_defs = {
        "atm_call": lambda s: ah.market_option_price(s, 100.0, 1.0, "call"),
        "otm_call_115": lambda s: ah.market_option_price(s, 115.0, 1.0, "call"),
        "put_spread_90_80": lambda s: (
            ah.market_option_price(s, 90.0, 1.0, "put")
            - ah.market_option_price(s, 80.0, 1.0, "put")
        ),
        "future": lambda s: (
            s.spot
            * float(s.dividend_curve.discount(1.0))
            / float(s.rate_curve.discount(1.0))
        ),
    }
    liability_greeks = ah.bump_and_revalue(liability_value, base)
    instrument_greeks = {
        name: ah.bump_and_revalue(fn, base) for name, fn in hedge_defs.items()
    }
    replication_size = (
        cohort.notional / 100.0
    )  # units of each option leg in the exact static hedge

    def universe(limit: float) -> list[ah.HedgeInstrument]:
        return [
            ah.HedgeInstrument(
                name,
                "future" if name == "future" else "option",
                1.0,
                abs(fn(base)),
                {
                    "delta": instrument_greeks[name].delta,
                    "vega": instrument_greeks[name].vega,
                    "rho": instrument_greeks[name].rho,
                },
                premium=0.0 if name == "future" else max(fn(base), 0.0),
                lot_size=1.0,
                max_position=limit,
                max_trade=limit,
            )
            for name, fn in hedge_defs.items()
        ]

    net = {
        "delta": -liability_greeks.delta,
        "vega": -liability_greeks.vega,
        "rho": -liability_greeks.rho,
    }
    weights_oos = {k: 1.0 / abs(v) for k, v in net.items()}
    rng = np.random.RandomState(11)
    n = 1_000 if quick else 3_000
    spot_shocks = rng.standard_t(5, n) * 0.06
    vol_shocks = rng.normal(0.0, 0.03, n) - 0.4 * spot_shocks
    rate_shocks = rng.normal(0.0, 30.0, n)

    def evaluate(positions: dict[str, float], horizon: float) -> dict[str, float]:
        base_liability = liability_value(base)
        base_hedge = sum(positions[name] * hedge_defs[name](base) for name in positions)
        losses = np.empty(n)
        gains = np.empty(n)
        for i in range(n):
            state = base.bumped(
                spot_relative=float(spot_shocks[i]),
                vol_absolute=float(max(vol_shocks[i], -0.15)),
                rate_bp=float(rate_shocks[i]),
                time_years=horizon,
            )
            losses[i] = liability_value(state) - base_liability
            gains[i] = (
                sum(positions[name] * hedge_defs[name](state) for name in positions)
                - base_hedge
            )
        return ref.scenario_effectiveness(losses, gains)

    def greek_effectiveness(sol: ah.HedgeSolution) -> float:
        names = tuple(sol.residual_exposures)
        residual_norm = math.sqrt(
            sum((sol.residual_exposures[k] * weights_oos[k]) ** 2 for k in names)
        )
        initial_norm = math.sqrt(sum((net[k] * weights_oos[k]) ** 2 for k in names))
        return 1.0 - residual_norm / initial_norm

    static = {
        "atm_call": replication_size,
        "otm_call_115": -replication_size,
        "put_spread_90_80": -replication_size,
        "future": 0.0,
    }
    metrics = evaluate(static, 0.0)
    record(
        sec_f,
        "Exact static replication (call spread - put spread): variance reduction",
        1.0,
        metrics["variance_reduction"],
        0.02,
        f"ES95 reduction={metrics['es95_reduction']:.3f}; residual is the unhedged fixed-notional rate exposure",
    )
    capped = universe(2.0 * replication_size)
    delta_vega = {k: net[k] for k in ("delta", "vega")}
    experiments = [
        (
            "Delta-only futures hedge",
            [universe(1e7)[-1]],
            {"delta": net["delta"]},
            np.inf,
            0.0,
        ),
        (
            "Delta+vega Greek match with options and futures, positions capped at 2x replication size",
            capped,
            delta_vega,
            np.inf,
            0.0,
        ),
        (
            "Delta+vega Greek match, capped, 1-month horizon with time decay",
            capped,
            delta_vega,
            np.inf,
            1.0 / 12.0,
        ),
        (
            "Delta+vega Greek match, capped, premium budget 1% of notional",
            capped,
            delta_vega,
            0.01 * cohort.notional,
            0.0,
        ),
        (
            "Delta+vega+rho Greek match with equity-only instruments, capped (rho is unhedgeable here)",
            capped,
            net,
            np.inf,
            0.0,
        ),
        (
            "Delta+vega+rho Greek match, uncapped (collinear instruments, no regularization)",
            universe(1e7),
            net,
            np.inf,
            0.0,
        ),
    ]
    outcomes = {}
    for label, instruments_oos, targets, budget, horizon in experiments:
        sol = ah.optimize_hedge(
            targets,
            instruments_oos,
            exposure_weights={k: weights_oos[k] for k in targets},
            constraints=ah.HedgeConstraints(premium_budget=budget),
        )
        positions = {name: sol.positions.get(name, 0.0) for name in hedge_defs}
        metrics = evaluate(positions, horizon)
        outcomes[label] = metrics["variance_reduction"]
        greek_eff = greek_effectiveness(sol)
        record(
            sec_f,
            f"{label}: Greek-metric effectiveness (expected) vs scenario variance reduction (actual)",
            greek_eff,
            metrics["variance_reduction"],
            math.inf,
            f"slope={metrics['dollar_offset_slope']:.3f} ES95 red={metrics['es95_reduction']:.3f} positions={ {k: round(v) for k, v in positions.items() if v} }",
        )
    record(
        sec_f,
        "Delta-only futures hedge achieves >= 85% variance reduction",
        1.0,
        outcomes[experiments[0][0]],
        0.15,
    )
    record(
        sec_f,
        "Capped delta+vega hedge achieves >= 90% variance reduction",
        1.0,
        outcomes[experiments[1][0]],
        0.10,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def fmt(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1e5 or abs(value) < 1e-3:
        return f"{value:.4e}"
    return f"{value:.6g}"


def render() -> str:
    lines: list[str] = []
    sections = list(dict.fromkeys(c.section for c in CHECKS))
    failures = [c for c in CHECKS if not c.passed]
    info = [c for c in CHECKS if c.informational]
    lines.append(
        f"Checks: {len(CHECKS)}  Passed: {len(CHECKS) - len(failures) - len(info)}  "
        f"Failed: {len(failures)}  Informational: {len(info)}"
    )
    lines.append("")
    for section in sections:
        lines.append(f"### {section}")
        lines.append("")
        lines.append(
            "| Check | Expected | Actual | Error | Tolerance | Result | Note |"
        )
        lines.append("|---|---:|---:|---:|---:|:---:|---|")
        for c in (x for x in CHECKS if x.section == section):
            lines.append(
                f"| {c.name} | {fmt(c.expected)} | {fmt(c.actual)} | {fmt(c.error)} | "
                f"{'n/a' if c.informational else fmt(c.tolerance)} | "
                f"{'INFO' if c.informational else 'PASS' if c.passed else 'FAIL'} | {c.note} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="smaller Monte Carlo sample sizes"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="write Markdown results to this path"
    )
    args = parser.parse_args()
    section_vanilla()
    section_liabilities(args.quick)
    section_budget()
    section_greeks(args.quick)
    section_hedging(args.quick)
    text = render()
    print(text)
    if args.out is not None:
        args.out.write_text(text)
    return 0 if all(c.passed for c in CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
