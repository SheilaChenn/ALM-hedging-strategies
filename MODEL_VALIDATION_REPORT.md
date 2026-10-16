# Model Validation Report: ALM Hedging Framework

| | |
|---|---|
| Scope | Pricing (RILA/FIA/IUL payoffs, Black-Scholes/Black-76, Monte Carlo, option budgets), Greeks (delta, gamma, vega, rho/DV01, theta, aggregation, stresses), hedging (optimizer, constraints, transaction costs, dynamic P&L attribution, hedge effectiveness) |
| Code validated | `main` at `5d5882d` ("Build complete ALM hedging framework"); corrections applied in the commits listed in Section 3 |
| Method | Independent reference implementations in `validation/reference.py` (closed forms, lognormal quadrature, deterministic recursions, brute-force search, an independent GBM sampler, weighted least squares, discrete CVaR, scenario-based effectiveness). None of these import the engine's pricing, risk, or optimization code. |
| Environment | Python 3.12.3, numpy 2.5.4, scipy 1.18.1, pandas 3.0.6, Linux x86-64 |
| Result | 126 benchmark checks: 118 PASS, 0 FAIL, 8 informational. Six material defects found in the original commit and corrected (Section 3). The reported 95% hedge effectiveness in the original dashboard was an artefact of two of those defects and is not a valid measure of hedge quality (Section 4). |

Reproduce everything in this report with:

```bash
pip install -e ".[dev]"
python validation/run_validation.py            # full benchmark tables (~1 min)
python validation/run_validation.py --quick    # reduced Monte Carlo sizes (~15 s)
pytest tests/test_validation_benchmarks.py     # regression subset of the benchmarks
pytest                                         # complete suite (37 tests)
ruff check . && ruff format --check . && mypy src
```

`run_validation.py` prints the Markdown tables reproduced in Section 5 and exits non-zero if any check fails. Checks with tolerance `n/a` are informational: they quantify a modelling limitation rather than test a right/wrong answer.

---

## 1. Summary of findings

### 1.1 Validated as correct (no change required)

| Area | Evidence |
|---|---|
| Black-Scholes with continuous carry | Agrees with lognormal quadrature and Black-76 on the forward to ~1e-14 across ATM/ITM/OTM, 0.01y–5y, S=100 and S=5250 |
| Put-call parity, zero-vol and expiry limits | Residuals ≤ 1e-14 |
| Buffered and floored RILA, FIA point-to-point crediting | Analytic values match quadrature of the payoff to ≤ 2e-13; buffered RILA matches the independent bond + call-spread − put-spread decomposition exactly; replication legs reproduce the payoff on S ∈ [1, 300] to 3e-14 |
| Option budget solvers | Solved cap and participation rate reprice to the budget to ≤ 1e-12; unreachable budgets are flagged unconverged |
| Finite-difference Greeks | Match analytic BS delta/gamma/vega/rho/theta/DV01 for three contracts; errors scale as O(h²) (ratio 3.99–4.02 when halving the bump) |
| Cohort → portfolio aggregation | PV and all Greeks sum exactly (≤ 1e-11) |
| Stress testing | Stressed values equal independent full repricing at the shocked spot/vol/rate (≤ 7e-11); shocks applied exactly as specified |
| Transaction costs, CVaR | Costs = Σ\|trade\| × unit cost exactly; reported CVaR equals the independent discrete CVaR |
| Dynamic hedge attribution identity | Σ components − costs = total P&L to 9e-16 |
| Monte Carlo statistics | RMSE ∝ n^-0.55 (expected −0.5), 95% CI coverage 0.950 over 200 seeds, seeded replay bitwise reproducible |

### 1.2 Material defects found and corrected

| # | Defect in `5d5882d` | Impact | Fix commit |
|---|---|---|---|
| 1 | `api._liabilities` built every dashboard contract with `index_start=100` while the market spot was 5250 | Dashboard liabilities were valued 52× in the money: delta ≈ 0 (1.8e-8), vega +17.3M with the wrong sign, liability $488.8M instead of $457.9M | `3bb67cc` |
| 2 | `api.optimize_hedge` passed +liability Greeks to the solver as the exposure to neutralize | The optimizer **sold** 3,729 ATM calls and 1,917 put spreads against a liability that is long index optionality, i.e. doubled the insurer's risk; residual delta reported as 0 while −157,170 of delta was created | `3bb67cc` |
| 3 | Effectiveness = 1 − ‖residual‖/‖net‖ over the targeted Greeks only, with weights 1/\|risk\| | Any Greek with near-zero liability exposure received an unbounded weight, and Greeks that the hedge created but did not target were invisible; this produced the 95.4% figure | `3bb67cc` |
| 4 | `simulate_gbm_paths` used spot zero rates and spot ATM implied vols as instantaneous step parameters | On sloped curves the simulated forward and terminal variance did not reprice the inputs; ATM call bias −2.0% (1y), −3.1% (2y), −5.8% (5y) vs analytic | `1b5affb` |
| 5 | Lot-rounding repair loop unwound the trade with the largest `\|trade\| × price` | Futures (price 1, no premium) were unwound forever when the premium budget bound; objective 899.2 vs brute-force optimum 4.30 | `700329f` |
| 6 | Dynamic hedge P&L attribution only covered Greeks in `risk_targets` | A delta-only hedge left 1,131.6 of "unexplained" P&L that was pure gamma/theta; after the fix the residual is 17.6 (1.7e-4 of gross moves) | `67610cf` |

Two further non-material corrections were made in `3bb67cc`: `backtest_hedge` used a hard-coded delta of 0.43 (it now uses engine delta/gamma when cohorts are supplied) and mapped the "Weekly"/"Monthly" labels to pandas aliases that produced the wrong frequency.

### 1.3 Corrected dashboard output (default inputs)

| Quantity | Original `5d5882d` | Corrected `main` |
|---|---|---|
| Liability PV | $488.8M | $457.9M |
| Delta ($/index point) | 1.8e-8 | 14,983 |
| Vega ($/1.00 vol) | +17.3M | −25.0M (RILA −38.8M, FIA +13.3M, IUL +1.3M) |
| DV01 ($/bp) | −212k | −180k |
| Recommended trades | sell 3,729 ATM calls, sell 1,917 put spreads, 0 futures | buy futures, buy options, same-signed Greeks as the liability |
| Reported effectiveness | 95.4% | 85.1% (first-order Greek metric) |
| Scenario variance reduction of the recommended hedge (linear, joint spot/vol/rate shocks) | −59.1 (hedge adds risk) | 0.44 |

The dashboard still reports the Greek-matching metric because that is what the optimizer minimizes, but the metric's definition is now exposed via `effectiveness_basis` and documented as first-order and in-sample. Section 4 explains why it should not be read as the fraction of risk removed.

---

## 2. Validation approach

### 2.1 Independence

Reference implementations in `validation/reference.py` were written from the mathematical definitions, not from the engine:

- `black76`, `bs_via_black76`: Black-76 on the forward F = S·e^{(r−q)T}, used as the analytic cross-check for the engine's continuous-carry Black-Scholes.
- `lognormal_expectation`, `bs_quadrature`, `terminal_liability_quad`: e^{−rT}·E[payoff(S_T)] by adaptive quadrature in log-space with breakpoints at every payoff kink.
- `rila_buffered_credit`, `rila_floored_credit`, `fia_annual_credit`: payoff functions written directly from the contract descriptions.
- `rila_buffered_call_put_decomposition`: bond + participation × (call spread) − put spread, with each leg priced by Black-76.
- `cliquet_closed_form`: product of one-year expectations for the annual ratchet (year 1 struck at the index start, later years forward-starting and spot-independent).
- `iul_expected_pv`: deterministic recursion of expected account value with charges, anniversary credit, and independent decrements, using E[AV·1{exit}] = E[AV]·P(exit).
- `bs_analytic_greeks`, `richardson_derivative`: closed-form Greeks and Richardson-extrapolated derivatives.
- `gbm_terminal_sample`, `mc_mean_and_se`: an independent `RandomState` GBM sampler used to check the engine's Monte Carlo and to verify forward and variance repricing.
- `weighted_least_squares_hedge`, `brute_force_lot_hedge`: closed-form unconstrained solution and exhaustive lattice search for the lot-constrained problem.
- `discrete_cvar`, `scenario_effectiveness`: discrete CVaR and scenario-based hedge metrics (variance reduction, dollar-offset regression slope, ES95 reduction).

### 2.2 Tolerances

- Analytic-vs-analytic identities: 1e-9 to 1e-12 relative (floating-point level).
- Monte Carlo vs analytic: 3 standard errors from the independent estimate of the MC standard error, with a relative floor (5e-4 FIA, 1e-3 IUL) because the engine's antithetic-pair SE is itself an estimate.
- Finite-difference Greeks vs analytic: tolerances set at roughly ten times the expected O(h²) truncation error for the engine's default bumps (1% spot, 1% vol, 10bp rate, 1 day theta).
- Hedge optimization: exact match to the brute-force integer optimum; unconstrained solution within 1e-4 of weighted least squares.
- Out-of-sample effectiveness: informational, except that the two hedges that should work (static replication; capped delta+vega) are required to reach variance reductions ≥ 0.98 and ≥ 0.90.

---

## 3. Corrections in detail

### 3.1 Dashboard liabilities struck at `index_start=100` (fixed in `3bb67cc`)

`api._liabilities` constructed `RILABufferedSegment`, `FIAAnnualPointToPoint`, and `IULAccountValue` with the default `index_start=100.0` while `api._market` set spot to the dashboard input (5250). Every contract was therefore valued with the index up 5,150% since inception, so all caps were hit, delta was numerically zero, and the "vega" reported was the sensitivity of a deep-in-the-money capped payoff. `_liabilities` now accepts `index_start` and every API entry point passes the valuation-market spot. The dashboard cohorts are therefore treated as newly issued segments struck at the valuation market (see the assumption in `docs/assumptions.md`). Regression test: `test_dashboard_contracts_are_struck_at_valuation_spot`.

### 3.2 Hedge sign convention (fixed in `3bb67cc`)

Liability Greeks are reported with the convention "positive = liability value rises". The insurer is short the liability, so the net exposure to neutralize is −(liability Greek), and the hedge assets must carry the **same**-signed Greeks as the liability. `api.optimize_hedge` passed the +liability Greeks to `optimization.optimize_hedge`, whose objective drives `target + M·q → 0`, so the solver chose positions with opposite-signed Greeks: short calls against a liability that is long calls. The API now passes `net_exposure = {name: −risk}`, returns residuals in liability-Greek sign, and documents the convention in its docstring. Independent check: after the fix, hedge delta = +15,025 vs liability delta +14,983 and hedge vega = −16.7M vs liability vega −25.0M (same signs); before the fix the recommended trades had delta −157,170 and vega −16.5M against liability vega +17.3M. Regression test: `test_dashboard_hedge_has_same_sign_greeks_as_liability`.

### 3.3 Effectiveness metric (fixed in `3bb67cc`)

The original metric was 1 − ‖w∘residual‖/‖w∘net‖ with w = 1/\|net Greek\| over only the targeted Greeks (delta and vega for the default instruments). Two consequences:

- With delta ≈ 1.8e-8 (defect 3.1), the delta weight was ~5.7e7 and the delta residual was trivially zero, so the metric was effectively "fraction of vega offset" = 1 − 0.80M/17.3M = 95.4%.
- Greeks that the hedge created but did not target (−157,170 of delta from the sold options) were invisible.

The corrected metric includes every Greek any selected instrument can move, and weights each Greek by 1/max(\|net Greek\|, reference exposure) where the reference exposure is 10% of account value per 100% index move for delta, 10% of account value per vol point for vega, and 100% of account value per unit rate for rho. The returned dictionary carries an `effectiveness_basis` string describing the computation. The independent recomputation in `run_validation.py` reproduces the reported 0.850627 exactly. Regression test: `test_dashboard_effectiveness_penalizes_untargeted_exposure`.

### 3.4 Monte Carlo drift and variance (fixed in `1b5affb`)

`simulate_gbm_paths` previously used `rate_curve.zero_rate(t_mid)`, `dividend_curve.zero_rate(t_mid)`, and `vol_surface.volatility(t_mid, 0)` as instantaneous parameters for each step. On a sloped curve the sum of step drifts is not the zero rate × T, and the sum of step variances is not σ_atm(T)²·T, so simulated prices did not reprice the analytic forward or the ATM option. Measured bias on the synthetic dashboard market (rates 3.8%–4.6%, ATM vols 16%–21%): −2.0% at 1y, −3.1% at 2y, −5.8% at 5y with 400k paths (≥ 20 standard errors). The simulation now uses increments of r(t)·t, q(t)·t, and σ_atm(t)²·t between grid points, so E[S_T] equals the market forward and the terminal log-variance equals σ_atm(T)²·T by construction. After the fix the MC ATM call is within 1.9, 0.6, and 0.7 standard errors of analytic at 1y, 2y, and 5y. Regression test: `test_monte_carlo_reprices_vanilla_on_sloped_term_structure`.

### 3.5 Lot-rounding repair and local search (fixed in `700329f`)

When rounding the continuous SLSQP solution to lot sizes violated the premium constraint, the repair loop unwound one lot of the trade with the largest `|trade| × price`. Futures (price 1.0, zero premium) were always chosen when their position was large, so the premium violation never shrank and the loop terminated at the iteration cap with a poor solution (objective 899.2 vs brute-force optimum 4.30 in a three-instrument test with lot sizes 5/2/2 and a binding budget). The repair step now unwinds the lot that maximizes the minimum constraint slack, the continuous stage is multi-started from the current position and from the clipped least-squares solution, and the local search tries single-lot and pairwise lot moves. The solver now returns the brute-force optimum [95, 2, 38] exactly. Regression test: `test_lot_rounding_with_binding_premium_budget_matches_brute_force`.

### 3.6 Dynamic P&L attribution (fixed in `67610cf`)

`simulate_dynamic_hedge` attributed liability P&L only to the Greeks in `risk_targets`. For a delta-only hedge the gamma and theta of the liability landed in "unexplained". The attribution now always covers delta, gamma, vega, rho, and theta in addition to any requested targets. On a daily delta hedge of a 1y ATM call (1,000 units, 252 steps) the unexplained residual dropped from 1,131.6 to 17.6, which is 1.7e-4 of gross liability moves; attributed theta = 5,890.5 and gamma = −4,776.5. Regression test: `test_dynamic_attribution_explains_gamma_and_theta_on_delta_hedge`.

### 3.7 Backtest proxy (fixed in `3bb67cc`)

`backtest_hedge` priced liability P&L with a hard-coded delta of 0.43 per unit of account value and gamma of 0.001 regardless of the portfolio, and mapped "Weekly" to a daily alias and "Monthly" to an invalid one. It now uses `calculate_greeks` delta and gamma when cohorts are supplied (falling back to the stylized values only when they are not) and uses `B`/`W-FRI`/`ME`. The backtest remains a second-order Taylor proxy on simulated index paths, not a full revaluation (Section 6). Regression test: `test_backtest_uses_engine_greeks_when_cohorts_supplied`.

---

## 4. Scrutiny of the reported 95% hedge effectiveness

**Definition.** The dashboard's "Effectiveness" is `1 − ‖w∘(net + M q)‖₂ / ‖w∘net‖₂`: the fraction of the weighted first-order Greek vector that the recommended positions offset at the current market. It is computed from the same Greeks the optimizer minimizes, so it is in-sample by construction and is maximized by any position vector that zeroes the targeted Greeks, regardless of what happens to second-order terms, un-targeted Greeks, or large moves.

**Why it read 95.4% in `5d5882d`.** Both ingredients were wrong: the liability Greeks were computed on contracts struck at 100 instead of 5250 (delta ≈ 0, vega with the wrong sign), and the optimizer was handed the wrong sign of exposure. The metric then measured how much of the wrong-signed vega the sold options cancelled, ignoring the −157,170 delta they introduced. A linear scenario analysis (joint spot/vol/rate shocks through the liability and hedge Greeks) gives a variance reduction of −59.1, i.e. the recommended hedge increased P&L variance roughly sixty-fold.

**What it reads after correction and what it means.** The corrected dashboard reports 85.1%. The independent scenario analysis of the same positions gives a variance reduction of 0.44, a dollar-offset regression slope of 0.45, and an ES95 reduction of 0.25. The gap is structural, not a bug: with only equity futures, ATM calls, and put spreads the −$180k/bp rate exposure is unhedgeable, and the premium budget (0.75% of AV) caps the vega offset at 16.7M of the 25.0M needed. The Greek metric reports how much of the *hedgeable* first-order risk was offset; the scenario metric reports how much *total* risk was removed.

**Out-of-sample behaviour (full revaluation).** `run_validation.py` section E prices a 1y buffered RILA cohort (buffer 10%, cap 15%, participation 1.08) and market-priced hedge instruments under 3,000 joint spot/vol/rate scenarios with full revaluation:

| Hedge | Greek-metric effectiveness | Scenario variance reduction | Dollar-offset slope | ES95 reduction |
|---|---:|---:|---:|---:|
| Exact static replication (call spread − put spread) | — | 0.991 | — | 0.91 |
| Delta-only futures | 1.000 | 0.940 | 0.98 | 0.77 |
| Delta + vega, positions capped at 2× replication size | 1.000 | 0.946 | 1.05 | 0.74 |
| Same, one-month horizon with time decay | 1.000 | 0.950 | 1.03 | 0.73 |
| Delta + vega, capped, premium budget 1% of notional | 1.000 | 0.949 | 1.05 | 0.75 |
| Delta + vega + rho with equity-only instruments (rho unhedgeable) | 0.266 | 0.480 | 0.33 | 0.37 |
| Delta + vega + rho, uncapped, collinear instruments | 1.000 | **−9.06** | 0.40 | −0.56 |

Conclusions:

1. Greek-matching hedges reach 94–95% scenario variance reduction on the RILA block when positions are bounded; the residual is the unhedged rate exposure of the fixed-notional guarantee (the exact static replication, which hedges all equity risk, still leaves 0.9% from rates).
2. The Greek metric is systematically optimistic: 1.000 vs 0.94–0.95.
3. The Greek metric is **not a safe optimization target without position limits**. Asking the optimizer to also match rho with equity-only instruments produced 182k long / 201k short calls (collinear vega, opposite gamma) that score 1.000 on the Greek metric and multiply P&L variance by ten. The dashboard's default instrument limits (`max_position`, lot sizes, premium and turnover budgets) prevent this in the current configuration, but the protection is coming from the constraints, not the metric.
4. A hedge-effectiveness figure for governance purposes should be the scenario-based variance reduction or dollar-offset slope over a specified shock distribution and horizon, not the first-order Greek metric. The engine already produces the inputs (`run_stress_test`, `simulate_dynamic_hedge`); the dashboard figure should be relabelled or supplemented accordingly.

---

## 5. Benchmark results

All tables below are the output of `python validation/run_validation.py` on the corrected `main`. Error = Actual − Expected. Tolerance `n/a` marks informational checks.

### A. Vanilla pricing

| Check | Expected | Actual | Error | Tolerance | Result |
|---|---:|---:|---:|---:|:---:|
| BS vs lognormal quadrature: call S=100 K=100 T=1 | 8.82732 | 8.82732 | −2.66e-14 | 8.83e-09 | PASS |
| BS vs Black-76 on forward: call S=100 K=100 T=1 | 8.82732 | 8.82732 | −3.55e-15 | 8.83e-12 | PASS |
| BS vs lognormal quadrature: put S=100 K=80 T=0.25 | 0.639232 | 0.639232 | 5.00e-15 | 1.00e-09 | PASS |
| BS vs Black-76 on forward: put S=100 K=80 T=0.25 | 0.639232 | 0.639232 | 1.11e-15 | 1.00e-12 | PASS |
| BS vs lognormal quadrature: call S=100 K=130 T=5 | 6.6266 | 6.6266 | −1.87e-14 | 6.63e-09 | PASS |
| BS vs Black-76 on forward: call S=100 K=130 T=5 | 6.6266 | 6.6266 | 1.78e-15 | 6.63e-12 | PASS |
| BS vs lognormal quadrature: call S=5250 K=5250 T=1 | 450.269 | 450.269 | 1.93e-12 | 4.50e-07 | PASS |
| BS vs Black-76 on forward: call S=5250 K=5250 T=1 | 450.269 | 450.269 | 0 | 4.50e-10 | PASS |
| BS vs lognormal quadrature: put S=100 K=100 T=0.01 | 0.787754 | 0.787754 | 1.63e-14 | 1.00e-09 | PASS |
| BS vs Black-76 on forward: put S=100 K=100 T=0.01 | 0.787754 | 0.787754 | 5.33e-15 | 1.00e-12 | PASS |
| BS vs lognormal quadrature: call S=100 K=60 T=2 | 47.8427 | 47.8427 | −7.11e-14 | 4.78e-08 | PASS |
| BS vs Black-76 on forward: call S=100 K=60 T=2 | 47.8427 | 47.8427 | −7.11e-15 | 4.78e-11 | PASS |
| Put-call parity C−P = S·e^{−qT} − K·e^{−rT} | −0.27278 | −0.27278 | −7.11e-15 | 1.00e-11 | PASS |
| Zero-vol limit equals discounted forward intrinsic | 6.81266 | 6.81266 | 0 | 1.00e-12 | PASS |
| Expiry returns intrinsic | 10 | 10 | 0 | 0 | PASS |
| `market_option_price` equals BS at interpolated (r, q, σ) | 15.3981 | 15.3981 | −1.24e-14 | 1.00e-10 | PASS |

### B. Liability payoffs and replication

| Check | Expected | Actual | Error | Tolerance | Result |
|---|---:|---:|---:|---:|:---:|
| RILA buffered analytic vs quadrature: buffer 0.1, cap 0.18, p 1, floor −1, T 1 | 99.9828 | 99.9828 | −1.71e-13 | 1e-09 | PASS |
| RILA buffered analytic vs bond + call spread − put spread (same contract) | 99.9828 | 99.9828 | 0 | 1e-10 | PASS |
| RILA buffered analytic vs quadrature: buffer 0.1, cap 0.15, p 1.08, floor −0.1, T 4.5 | 90.0473 | 90.0473 | 1.42e-14 | 1e-09 | PASS |
| RILA buffered analytic vs decomposition (same contract) | 90.0473 | 90.0473 | 1.42e-14 | 1e-10 | PASS |
| RILA buffered analytic vs quadrature: buffer 0.2, cap 0.25, p 0.9, floor −0.5, T 6 | 86.4132 | 86.4132 | −8.53e-14 | 1e-09 | PASS |
| RILA buffered analytic vs decomposition (same contract) | 86.4132 | 86.4132 | −2.84e-14 | 1e-10 | PASS |
| RILA buffered analytic vs quadrature: buffer 0, cap 0.12, p 1, floor −0.1, T 1 | 97.6888 | 97.6888 | −9.95e-14 | 1e-09 | PASS |
| RILA buffered analytic vs decomposition (same contract) | 97.6888 | 97.6888 | 2.84e-14 | 1e-10 | PASS |
| RILA floored analytic vs quadrature: floor −0.1, cap 0.15, p 1 | 98.456 | 98.456 | −1.28e-13 | 1e-09 | PASS |
| RILA floored analytic vs quadrature: floor 0, cap 0.08, p 0.75 | 100.145 | 100.145 | 5.68e-14 | 1e-09 | PASS |
| FIA 1y analytic vs quadrature: cap 0.095, p 0.85, spread 0 | 100.686 | 100.686 | −7.11e-14 | 1e-09 | PASS |
| FIA 1y analytic vs quadrature: cap 0.1, p 1, spread 0.02 | 100.629 | 100.629 | 0 | 1e-09 | PASS |
| Replication legs reproduce payoff on S ∈ [1, 300] (max abs error) | 0 | 2.84e-14 | 2.84e-14 | 1e-10 | PASS |

### B. Monte Carlo consistency and convergence

| Check | Expected | Actual | Error | Tolerance | Result | Note |
|---|---:|---:|---:|---:|:---:|---|
| ATM call T=1: MC (forward variance/rates) vs analytic on sloped curves | 9.16685 | 9.1539 | −0.01295 | 0.02107 | PASS | se = 0.0070 |
| Terminal log-variance equals σ_atm(T)²·T, T=1 | 0.046225 | 0.046382 | 1.57e-04 | 4.62e-04 | PASS | |
| E[S_T] equals market forward, T=1 | 101.603 | 101.611 | 0.00792 | 0.2032 | PASS | |
| ATM call T=2: MC vs analytic on sloped curves | 13.5207 | 13.5315 | 0.01085 | 0.05376 | PASS | se = 0.0179 |
| Terminal log-variance equals σ_atm(T)²·T, T=2 | 0.097871 | 0.098131 | 2.61e-04 | 9.79e-04 | PASS | |
| E[S_T] equals market forward, T=2 | 103.386 | 103.399 | 0.01309 | 0.2068 | PASS | |
| ATM call T=5: MC vs analytic on sloped curves | 22.5769 | 22.5954 | 0.01855 | 0.07786 | PASS | se = 0.0260 |
| Terminal log-variance equals σ_atm(T)²·T, T=5 | 0.272707 | 0.272966 | 2.59e-04 | 2.73e-03 | PASS | |
| E[S_T] equals market forward, T=5 | 109.505 | 109.513 | 0.00801 | 0.2190 | PASS | |
| FIA annual ratchet 2y: MC vs independent closed form | 101.964 | 101.957 | −0.00743 | 0.05098 | PASS | se = 0.0017 |
| FIA annual ratchet 5y: MC vs independent closed form | 103.476 | 103.467 | −0.00889 | 0.05174 | PASS | se = 0.0050 |
| FIA annual ratchet 10y: MC vs independent closed form | 104.022 | 104.010 | −0.01178 | 0.05201 | PASS | se = 0.0088 |
| IUL 5y (DB multiple 1): MC vs deterministic recursion | 84410.9 | 84408.0 | −2.93 | 84.41 | PASS | se = 8.6 |
| IUL 10y (DB multiple 1.5): MC vs deterministic recursion | 85190.1 | 85205.9 | 15.83 | 85.19 | PASS | se = 15.8 |
| IUL 20y (DB multiple 1): MC vs deterministic recursion | 83365.8 | 83347.4 | −18.39 | 83.37 | PASS | se = 8.4 |
| IUL cohort behaviour multipliers applied (×2 mortality, ×0.5 lapse) | 834,800 | 834,280 | −516 | 1,670 | PASS | |
| RMSE convergence order (log-log slope vs paths; expect −0.5) | −0.5 | −0.554 | −0.054 | 0.15 | PASS | n=1k: 0.348, 16k: 0.058, 256k: 0.016 |
| 95% CI coverage over 200 seeds (antithetic-pair SE) | 0.95 | 0.95 | 0 | 0.05 | PASS | |
| Seeded replay is bitwise reproducible | 0 | 0 | 0 | 0 | PASS | |

The FIA and IUL biases of ≈ −1e-4 relative are within the tolerance floors. Larger runs (24–60 seeds × 100k paths) gave FIA 5y bias −0.0086 ± 0.0025 and IUL 10y/20y relative bias −4e-5 and +2e-5, i.e. no material bias; the small FIA deficit is consistent with the closed form ignoring the engine's time-grid discretization of the exact anniversary dates.

### C. Option budget solvers

| Check | Expected | Actual | Error | Tolerance | Result | Note |
|---|---:|---:|---:|---:|:---:|---|
| Cap solved for 4% budget reprices to budget (independent BS) | 0.04 | 0.04 | 1.17e-12 | 1e-09 | PASS | cap = 0.102203 |
| Solved cap: FIA 1y analytic value = bond + budget | 1.01045 | 1.01045 | 1.17e-12 | 1e-09 | PASS | |
| Participation solved for 5% ATM-call budget reprices to budget | 0.05 | 0.05 | 2.08e-17 | 1e-09 | PASS | p = 0.566423 |
| Unreachable budget flagged unconverged | 0 | 0 | 0 | 0 | PASS | residual = −0.4117 |

### D. Greeks: finite differences vs analytic

| Check | Expected | Actual | Error | Tolerance | Result |
|---|---:|---:|---:|---:|:---:|
| delta call K=100 T=1 | 0.573496 | 0.573431 | −6.45e-05 | 5e-04 | PASS |
| gamma call K=100 T=1 | 0.0193576 | 0.0193545 | −3.06e-06 | 1e-04 | PASS |
| vega call K=100 T=1 | 38.7152 | 38.7145 | −6.46e-04 | 0.02 | PASS |
| rho call K=100 T=1 | 48.5223 | 48.5223 | −5.64e-07 | 1e-05 | PASS |
| theta call K=100 T=1 | −4.75369 | −4.75661 | −2.92e-03 | 0.03 | PASS |
| dv01 call K=100 T=1 (forward difference per bp) | 0.00485223 | 0.00485295 | 7.25e-07 | 1e-05 | PASS |
| delta put K=90 T=0.5 | −0.186756 | −0.186986 | −2.29e-04 | 5e-04 | PASS |
| gamma put K=90 T=0.5 | 0.0189495 | 0.0189511 | 1.59e-06 | 1e-04 | PASS |
| vega put K=90 T=0.5 | 18.9495 | 18.9371 | −0.01237 | 0.02 | PASS |
| rho put K=90 T=0.5 | −10.1131 | −10.1131 | −2.91e-07 | 1e-05 | PASS |
| theta put K=90 T=0.5 | −3.36987 | −3.37291 | −3.04e-03 | 0.03 | PASS |
| dv01 put K=90 T=0.5 | −0.00101131 | −0.00101105 | 2.62e-07 | 1e-05 | PASS |
| delta call K=120 T=2 | 0.351661 | 0.351667 | 6.03e-06 | 5e-04 | PASS |
| gamma call K=120 T=2 | 0.0129497 | 0.0129484 | −1.37e-06 | 1e-04 | PASS |
| vega call K=120 T=2 | 51.7989 | 51.7833 | −0.01568 | 0.02 | PASS |
| rho call K=120 T=2 | 58.6739 | 58.6739 | 8.73e-07 | 1e-05 | PASS |
| theta call K=120 T=2 | −3.11839 | −3.11892 | −5.30e-04 | 0.03 | PASS |
| dv01 call K=120 T=2 | 0.00586739 | 0.00586939 | 2.00e-06 | 1e-05 | PASS |

Engine theta is reported per day as the value change from one day of time decay (negative for long options); vega and rho are per unit (1.00) of volatility and rate.

### D. Greeks: bump-size stability, surface convention, aggregation, signs, stress

| Check | Expected | Actual | Error | Tolerance | Result | Note |
|---|---:|---:|---:|---:|:---:|---|
| delta error ratio when halving bump 2% → 1% (expect 4 for O(h²)) | 4 | 3.99398 | −0.0060 | 0.3 | PASS | |
| delta error ratio when halving bump 1% → 0.5% | 4 | 3.99849 | −0.0015 | 0.3 | PASS | |
| gamma error ratio 2% → 1% | 4 | 3.99953 | −4.7e-04 | 0.3 | PASS | |
| gamma error ratio 1% → 0.5% | 4 | 3.99988 | −1.2e-04 | 0.3 | PASS | |
| vega error ratio 2% → 1% | 4 | 4.02256 | 0.0226 | 0.3 | PASS | |
| vega error ratio 1% → 0.5% | 4 | 4.00559 | 0.0056 | 0.3 | PASS | |
| Skewed surface delta = BS delta + vega·dσ/dS (sticky moneyness) | 0.593673 | 0.593574 | −9.86e-05 | 5e-04 | PASS | sticky-strike BS delta would be 0.5627 |
| portfolio pv equals Σ cohort pv | 2524.08 | 2524.08 | 0 | 2.5e-05 | PASS | |
| portfolio delta equals Σ cohort delta | 12.5049 | 12.5049 | −1.1e-13 | 1.3e-07 | PASS | |
| portfolio gamma equals Σ cohort gamma | −0.195816 | −0.195816 | 6.8e-13 | 1e-08 | PASS | |
| portfolio vega equals Σ cohort vega | −737.507 | −737.507 | 1.1e-11 | 7.4e-06 | PASS | |
| portfolio rho equals Σ cohort rho | −2015.14 | −2015.14 | 0 | 2.0e-05 | PASS | |
| portfolio dv01 equals Σ cohort dv01 | −0.201532 | −0.201532 | 4.5e-13 | 1e-08 | PASS | |
| portfolio theta equals Σ cohort theta | 89.8801 | 89.8801 | 4.1e-11 | 9.0e-07 | PASS | |
| 1y buffered RILA liability delta > 0 | 1 | 1 | 0 | 0 | PASS | delta = 0.5423 |
| 1y buffered RILA liability DV01 < 0 | 1 | 1 | 0 | 0 | PASS | dv01 = −0.004575 |
| DV01 equals rho × 1e-4 to first order | −0.00457487 | −0.00457524 | −3.7e-07 | 1e-05 | PASS | |
| Crash scenario stressed value equals independent repricing | 1909.4 | 1909.4 | −6.7e-11 | 1.9e-03 | PASS | |
| Stressed spot = S·(1 − 40%) | 60 | 60 | 0 | 0 | PASS | |
| Stressed vol = σ + 0.25 | 0.45 | 0.45 | 0 | 1e-12 | PASS | |
| Stressed rate = r − 75bp | 0.0225 | 0.0225 | 0 | 1e-12 | PASS | |
| `aggregate_stress_results` sums base values | 5048.15 | 5048.15 | 0 | 1e-09 | PASS | |

The sticky-moneyness result is a convention, not an error: the engine's volatility surface is indexed by log(K/S), so a spot bump moves the surface with spot and the finite-difference delta includes the skew term. This is documented in `docs/assumptions.md`; a desk that hedges on sticky-strike deltas would see a 3-point delta difference on the test contract.

### D. Greeks: Monte Carlo noise at dashboard settings (2,000 paths, 5 seeds)

| Check | Expected | Actual | Tolerance | Result | Note |
|---|---:|---:|---:|:---:|---|
| pv: seed-to-seed coefficient of variation | 0 | 4.9e-04 | 5e-03 | PASS | mean = 4.582e+08 |
| delta: seed-to-seed coefficient of variation | 0 | 0.0196 | 0.1 | PASS | mean = 14,999.6 |
| gamma: seed-to-seed coefficient of variation | 0 | 1.371 | n/a | INFO | mean = −1.34; sign unstable |
| vega: seed-to-seed coefficient of variation | 0 | 0.0293 | 0.1 | PASS | mean = −2.509e+07 |
| dv01: seed-to-seed coefficient of variation | 0 | 0.0026 | 0.01 | PASS | mean = −179,716 |
| gamma noise materiality: 0.5·sd(gamma)·(10% spot move)²/PV | 0 | 5.5e-04 | 1e-03 | PASS | statistically zero, economically immaterial |

The dashboard gamma for the Monte Carlo products (FIA, IUL) is statistically indistinguishable from zero at 2,000 paths; the second-order P&L it implies for a 10% move is 0.06% of PV, so it does not affect hedge decisions, but the displayed gamma should not be interpreted.

### E. Hedge optimizer, costs, CVaR

| Check | Expected | Actual | Error | Tolerance | Result | Note |
|---|---:|---:|---:|---:|:---:|---|
| Unconstrained solution matches weighted least squares: futures | 77.304 | 77.304 | −2.3e-07 | 1e-04 | PASS | |
| Unconstrained solution matches weighted least squares: call | 103.252 | 103.252 | −3.2e-07 | 1e-04 | PASS | |
| Unconstrained solution matches weighted least squares: put | 25.9475 | 25.9475 | −8.7e-08 | 1e-04 | PASS | |
| Unhedgeable rho residual is left untouched | −300 | −300 | 0 | 1e-09 | PASS | |
| Lot-constrained objective with binding premium budget vs brute force | 4.2955 | 4.2955 | 0 | 1e-09 | PASS | brute = solver = [95, 2, 38] |
| Premium budget respected after rounding | 1 | 1 | 0 | 0 | PASS | premium used = 244 of 250 |
| Discrete CVaR(80%) of sample losses | 95 | 95 | 0 | 0 | PASS | |
| Reported stress CVaR equals independent CVaR of net losses | 0 | 0 | 0 | 1e-12 | PASS | |

### E. Dynamic hedge simulation (1y ATM call, 1,000 units, daily rebalancing)

| Check | Expected | Actual | Error | Tolerance | Result | Note |
|---|---:|---:|---:|---:|:---:|---|
| Initial futures position = +1000 × BS delta (buys futures against a positive-delta liability) | 573.496 | 573 | −0.496 | 1 | PASS | lot rounding |
| Transaction costs = Σ\|trade\| × unit cost | 71.76 | 71.76 | 0 | 1e-09 | PASS | |
| Attribution identity: Σ components − costs = total (max abs error) | 0 | 8.9e-16 | 8.9e-16 | 1e-09 | PASS | |
| Daily delta hedge variance reduction on short call (expect > 0.99) | 1 | 0.99798 | −0.0020 | 0.01 | PASS | |
| Unexplained residual after gamma/theta attribution (fraction of gross liability moves) | 0 | 1.73e-04 | 1.73e-04 | 0.01 | PASS | theta = 5,890.5, gamma = −4,776.5 |

### E. Dashboard hedge effectiveness (default inputs, corrected engine)

| Check | Expected | Actual | Error | Tolerance | Result | Note |
|---|---:|---:|---:|---:|:---:|---|
| Hedge delta has the same sign as liability delta | 1 | 1 | 0 | 0 | PASS | liability 14,983; hedge 15,025 |
| Hedge vega has the same sign as liability vega | 1 | 1 | 0 | 0 | PASS | liability −25.03M; hedge −16.68M |
| Reported residual vega = liability vega − hedge vega | −8.351e+06 | −8.351e+06 | 0 | 1 | PASS | |
| Premium used within budget | 1 | 1 | 0 | 0 | PASS | premium = 3,750,000 |
| Effectiveness equals 1 − ‖scaled residual Greeks‖/‖scaled Greeks‖ (independent recomputation) | 0.850627 | 0.850627 | 0 | 1e-09 | PASS | |
| Scenario variance reduction of recommended hedge vs reported Greek effectiveness | 0.850627 | 0.442994 | −0.4076 | n/a | INFO | dollar-offset slope 0.45; ES95 reduction 0.25 |

### E. Out-of-sample full-revaluation effectiveness (RILA cohort, market-priced hedges)

| Check | Expected | Actual | Error | Tolerance | Result | Note |
|---|---:|---:|---:|---:|:---:|---|
| Exact static replication: variance reduction | 1 | 0.991125 | −0.0089 | 0.02 | PASS | ES95 reduction 0.912; residual is the fixed-notional rate exposure |
| Delta-only futures: Greek metric (expected) vs scenario variance reduction (actual) | 0.999949 | 0.939802 | −0.0601 | n/a | INFO | slope 0.977; ES95 red 0.766; future 3,683 |
| Delta + vega, capped at 2× replication size | 0.999964 | 0.946064 | −0.0539 | n/a | INFO | slope 1.051; ES95 red 0.736 |
| Delta + vega, capped, 1-month horizon with time decay | 0.999964 | 0.950417 | −0.0495 | n/a | INFO | slope 1.028; ES95 red 0.725 |
| Delta + vega, capped, premium budget 1% of notional | 0.999973 | 0.949413 | −0.0506 | n/a | INFO | slope 1.050; ES95 red 0.745 |
| Delta + vega + rho with equity-only instruments, capped | 0.26615 | 0.480292 | 0.2141 | n/a | INFO | slope 0.331; ES95 red 0.369 |
| Delta + vega + rho, uncapped (collinear, unregularized) | 0.999991 | −9.06226 | −10.06 | n/a | INFO | atm_call +182,116; otm_call −201,268; future −36,119 |
| Delta-only futures hedge achieves ≥ 85% variance reduction | 1 | 0.939802 | −0.0602 | 0.15 | PASS | |
| Capped delta + vega hedge achieves ≥ 90% variance reduction | 1 | 0.946064 | −0.0539 | 0.1 | PASS | |

Scenario sampling is seeded (3,000 joint shocks: Student-t(5) spot shocks with 6% scale, vol shocks correlated −0.4 with spot, 30bp rate shocks). The `--quick` run uses a 1,000-scenario sample and gave 0.991/0.938/0.919/0.925/0.922/0.494/−9.19 for the same rows, i.e. the same conclusions.

---

## 6. Material assumptions and limitations

Assumptions the validation relied on, and limitations that remain in the corrected code:

1. **Dashboard cohorts are new business.** Contracts are struck at the valuation-market spot with maturities measured from the valuation date. In-force blocks with historical strikes, partial terms, and realized credits are not represented; the dashboard would need per-cohort `index_start` and anniversary inputs.
2. **Skew is used inconsistently.** Analytic replication prices each strike at its own implied volatility (sticky moneyness), but the Monte Carlo products (FIA ratchet, IUL) use the ATM volatility term structure only. On the synthetic dashboard market the skew is material (16%–21% across strikes), so FIA/IUL values ignore the cost of the downside floor's skew. A local- or stochastic-volatility simulation, or at least a strike-dependent variance for the capped call spread, would be needed for consistency.
3. **Annual-reset products use stub periods.** Non-integer maturities (dashboard durations 5.1y and 5.7y) end with a short stub period that receives a full annual cap and floor. The cliquet closed form in `validation/reference.py` makes the same assumption, so the MC agreement does not validate the stub treatment economically.
4. **Behaviour assumptions are partial.** `data/assumptions.json` lapse and mortality tables are applied only to IUL; RILA and FIA are valued to maturity with no decrements, surrender charges, or dynamic lapse. Greeks for those products therefore overstate the hedge notional for seasoned blocks.
5. **Hedge instruments are templates.** The dashboard's instrument Greeks and premiums (`api.optimize_hedge` templates for futures, ATM calls, put spreads, variance swaps, 10Y swap) are hard-coded constants scaled by account value, not market-priced from the valuation market. The out-of-sample study in Section 4 therefore priced its own instruments; the dashboard's premium and Greek offsets should be regenerated from the market state before any real use.
6. **Premium budget counts sold options.** The budget constraint uses Σ\|q\|·premium, so premium received from sold options is treated as consumed budget. This is conservative but means short-option hedges are penalized twice (budget and turnover).
7. **Backtest is a Taylor proxy.** `backtest_hedge` evolves liability P&L as delta·ΔS + ½·gamma·ΔS² on simulated index paths with constant Greeks and no vol or rate shocks. It is a sanity check on rebalancing frequency and cost, not a hedge-effectiveness test.
8. **`price_portfolio` decomposition is illustrative.** The `option_budget`, `guarantee`, and `floor_value` columns are heuristics for the dashboard and are not reconciled to the pricing engine. Only the `liability` column is validated.
9. **Monte Carlo gamma at dashboard settings is noise.** See the materiality check in Section 5.D. Raising paths to ≥ 50,000 or using a pathwise/likelihood-ratio estimator would be needed for a usable gamma.
10. **Greek matching is unregularized.** With collinear instruments and no position caps the optimizer finds large offsetting positions that score perfectly on the Greek metric and destroy the hedge (Section 4). The dashboard relies on `max_position`, lot sizes, and premium/turnover budgets for protection. A quadratic position penalty or a scenario-based objective is the appropriate structural fix.
11. **`price_portfolio` and `calculate_greeks` use different Monte Carlo seeds/path counts**, so the PV shown on the dashboard and the PV underlying the Greeks can differ by a few basis points.
12. **`solve_participation_rate` with a cap may be non-monotone** in the participation rate when the cap binds; the solver converges on the tested cases but a bracketing check on monotonicity is not performed.
13. **Demo fallback in `app.py`.** When the analytics bridge is unavailable the dashboard shows a hard-coded 84% effectiveness; this should be labelled as illustrative in the UI.
14. **Legacy tests.** Several tests in `tests/test_hedging.py` have names that describe "offsetting" trades in the pre-fix sign convention; they still pass because they test the solver against a given target, but the names are misleading and were left unchanged to preserve history.

---

## 7. Unresolved modelling issues (recommended follow-up)

In priority order:

1. Replace or supplement the dashboard's first-order Greek "Effectiveness" with the scenario-based variance reduction / dollar-offset slope over a declared shock distribution and horizon (inputs already exist in `run_stress_test` and `simulate_dynamic_hedge`).
2. Market-price the hedge instrument templates from the valuation `MarketState` so premiums, deltas, and vegas are consistent with the liabilities they hedge.
3. Introduce skew into the Monte Carlo products or price the FIA/IUL capped call spreads with strike-dependent volatility.
4. Add regularization (position penalty) or a scenario objective to the optimizer so the solution is well-posed without relying on hard position caps.
5. Extend RILA/FIA valuation with lapse, mortality, surrender charges, and per-cohort strikes/anniversaries so in-force blocks can be represented.
6. Increase Monte Carlo path counts (or change the estimator) before displaying gamma for MC-priced products.

---

## 8. Files added or changed by this validation

| Path | Purpose |
|---|---|
| `validation/reference.py` | Independent reference implementations (no imports from engine pricing/risk/optimization) |
| `validation/run_validation.py` | Benchmark runner producing the tables in Section 5; exits 1 on any FAIL |
| `tests/test_validation_benchmarks.py` | 13 regression tests pinned to the benchmarks and to each correction |
| `src/alm_hedging/pricing.py` | Forward-rate / forward-variance GBM simulation |
| `src/alm_hedging/optimization.py` | Multi-start SLSQP, slack-maximizing repair, single and pairwise lot search |
| `src/alm_hedging/simulation.py` | Attribution over all reported Greeks |
| `src/alm_hedging/api.py` | Contracts struck at valuation spot; net-exposure sign convention; effectiveness over all movable Greeks with reference-scale weights; `effectiveness_basis`; backtest uses engine Greeks |
| `docs/assumptions.md`, `docs/model-methodology.md`, `docs/validation.md`, `README.md` | Sign convention, sticky moneyness, MC forward variance, effectiveness definition, validation commands |

Final verification on the corrected `main`: `pytest` 37 passed; `ruff check .`, `ruff format --check .`, and `mypy src` clean; `python validation/run_validation.py` 126 checks, 118 PASS, 0 FAIL, 8 INFO.
