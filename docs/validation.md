# Validation and controls

This prototype uses layered controls rather than treating one backtest as model
validation.

## Automated checks

1. Payoff boundaries: expiry, zero notional, cap/floor boundaries, deep
   in/out-of-the-money cases, and invalid inputs.
2. Analytic identities: put-call parity, option bounds, delta parity, and
   monotonicity.
3. Numerical behavior: Monte Carlo standard error and convergence to the
   analytic benchmark; finite-difference stability across bump sizes.
4. Aggregation: cohort and portfolio values and risks reconcile exactly to
   constituent results.
5. Economic signs: long calls have positive delta/vega, liability hedge trades
   oppose liability sensitivities, and adverse stresses move values in the
   documented direction.
6. Reproducibility: repeated runs with identical inputs and seeds produce equal
   outputs.
7. Integration: package workflow and Streamlit module import without network
   access.

## Acceptance criteria

- Put-call parity residual is below numerical tolerance.
- Monte Carlo confidence intervals cover the analytic value and error generally
  decreases as path count rises.
- Central-difference Greeks remain close when spot and volatility bumps are
  halved.
- Portfolio totals reconcile to the sum of positions within floating-point
  tolerance.
- Every output includes valuation date, model version, scenario/configuration,
  and seed where random sampling is used.

These checks detect implementation regressions; they do not establish fitness
for production. Before any production use, an independent team should validate
contract interpretation, source data, calibration, numerical implementation,
hedging constraints, behavior assumptions, and end-user controls.

## Independent model validation

An independent quantitative validation of pricing, Greeks, and hedging is
documented in [`MODEL_VALIDATION_REPORT.md`](../MODEL_VALIDATION_REPORT.md).
Its benchmarks live in `validation/reference.py` (closed forms, lognormal
quadrature, deterministic recursions, brute-force search, and an independent
Monte Carlo stream that import nothing from the engine's pricing code) and are
executed by:

```bash
python validation/run_validation.py           # full benchmark tables
python validation/run_validation.py --quick   # smaller Monte Carlo sizes
pytest tests/test_validation_benchmarks.py    # regression subset
```

## Reproducible review

Install the locked project environment, run `pytest`, execute the notebook from
a clean kernel, and retain package versions with exported results. Changes to
data or assumptions require a new model version and comparison against the prior
baseline.
