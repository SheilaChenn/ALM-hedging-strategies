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
  bumped valuations. Paths are driven by forward rates, forward dividend
  yields, and forward variance from the at-the-money volatility term structure,
  so simulated forwards and terminal variances reprice the input curves.
  Monte Carlo ignores volatility skew; analytic replication prices each strike
  at its own implied volatility.
- Greeks are liability Greeks: positive delta means the liability value rises
  with the index. The insurer is short the liability, so its net exposure is
  the negative of each liability Greek and hedge assets must carry the
  same-signed Greeks as the liability (a block of index-linked crediting is
  hedged by buying index calls, not selling them). The optimizer is given the
  insurer's net exposure to neutralize. DV01 is the value change for a
  one-basis-point upward parallel rate shift.
- Spot bumps hold the surface fixed in log-moneyness ("sticky moneyness"), so
  on a skewed surface finite-difference delta includes the skew term
  vega × dσ/dS and differs from the sticky-strike Black-Scholes delta.
- Dashboard cohorts are treated as newly issued segments struck at the
  valuation-market index level; FIA and IUL anniversaries are measured from the
  valuation date, and a non-integer maturity ends with a stub period that
  receives a full annual cap and floor.
- Hedge instruments are liquid index futures, European options, and
  fixed-for-floating swaps. Recommendations are illustrative and constrained by
  premium, lots, turnover, tenor, liquidity, and concentration.

See [model methodology](model-methodology.md) for formulas and
[exclusions](exclusions.md) before interpreting results.
