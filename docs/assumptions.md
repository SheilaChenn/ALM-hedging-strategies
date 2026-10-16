# Model assumptions

## Scope and conventions

The framework is an illustrative ALM research model using synthetic data dated
2025-01-02 and model version `v1.0`. Currency amounts are USD, rates and
volatilities are decimals, maturities are ACT/365 year fractions, and index
levels exclude dividends. Positive risk means the liability value increases.

## Market

- Continuously compounded zero rates are linearly interpolated in maturity with
  flat endpoint extrapolation.
- Implied volatility is bilinearly interpolated by expiry and log-moneyness,
  with flat extrapolation. Volatility is lognormal and deterministic.
- The equity index follows risk-neutral geometric Brownian motion with
  deterministic rates, dividend yield, and volatility.
- Counterparty credit, funding, collateral, transaction costs, taxes, and
  market impact are omitted from valuation.

## Contracts and policyholders

- RILA and FIA cohorts use annual point-to-point index returns. Caps,
  participation rates, buffers, and floors are applied to simple index returns.
- IUL account value is projected with deterministic premium, expense, mortality,
  and lapse assumptions. Crediting never falls below the declared floor.
- Decrements are deterministic expected decrements; no dynamic behavior,
  anti-selection, or stochastic mortality is modeled.
- Cohorts aggregate identical representative policies. Premiums and notionals
  are beginning-of-period amounts unless a contract states otherwise.

## Valuation and hedging

- Vanilla components use Black-Scholes/Black-76 replication; path-dependent
  cash flows use risk-neutral Monte Carlo.
- Monte Carlo uses explicitly supplied seeds and common random numbers for
  bumped valuations.
- Greeks are liability Greeks. A hedge trade should generally have the opposite
  sensitivity. DV01 is the value change for a one-basis-point upward parallel
  rate shift.
- Hedge instruments are liquid index futures, European options, and
  fixed-for-floating swaps. Recommendations are illustrative and constrained by
  premium, lots, turnover, tenor, liquidity, and concentration.

See [model methodology](model-methodology.md) for formulas and
[exclusions](exclusions.md) before interpreting results.
