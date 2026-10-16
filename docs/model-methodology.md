# Model methodology

## Market representation

The market state contains spot \(S\), continuously compounded zero and dividend
curves, and an implied-volatility surface \(\sigma(T,\log(K/S))\). Discount
factors are \(D(0,T)=e^{-r(T)T}\). Inputs are immutable so every valuation can
record and replay one complete state.

## Vanilla benchmark

European options use Black-Scholes with maturity-specific \(r,q,\sigma\):

\[
d_1=\frac{\log(S/K)+(r-q+\sigma^2/2)T}{\sigma\sqrt T},\qquad
d_2=d_1-\sigma\sqrt T.
\]

Call and put values are \(Se^{-qT}N(d_1)-Ke^{-rT}N(d_2)\) and
\(Ke^{-rT}N(-d_2)-Se^{-qT}N(-d_1)\). Implementations handle expiry and
zero-volatility limits explicitly. Put-call parity and analytic Greeks are
validation anchors.

## Indexed liabilities

Contract crediting is represented as cash plus vanilla-option pieces where
possible. A capped participation payoff, for example, is a long call spread
after scaling for participation. Buffers and floors add put-spread protection.
RILA, FIA, and IUL projections then combine the credited return with their
contract-specific floors, charges, premiums, and expected decrements.

Multi-period cash flows are valued under the risk-neutral measure. Antithetic
geometric-Brownian paths use an explicit random seed. Bumped calculations reuse
the same standard-normal draws to reduce noise.

## Risk

Portfolio value is the sum of cohort values. Delta, gamma, vega, rho/DV01, and
theta use central bump-and-revalue unless a stable analytic result is available.
Reports state each bump size. Stability is checked by halving bumps and comparing
the resulting sensitivity.

Scenarios perform full revaluation, not Greek approximation:

- equity up/down: relative spot shock;
- volatility: absolute volatility-point shock;
- rates: parallel basis-point shock;
- twist: maturity-dependent curve shock;
- combined crash: equity down, volatility up, and rates shocked together;
- behavior: deterministic decrement overrides.

Scenario P&L is stressed value less base value. A more valuable liability is a
loss to the insurer unless offset by hedge-asset P&L.

## Hedge recommendation and simulation

Candidate trades expose a sensitivity matrix across Greeks and scenarios.
Constrained least squares minimizes scaled residual exposure plus turnover and
cost penalties, subject to lot, premium, tenor, liquidity, and concentration
limits. Hedge quantities therefore oppose the liability exposure under the
reporting sign convention.

Dynamic simulations rebalance at the configured schedule, deduct explicit
transaction costs when provided, and attribute total P&L to liability, hedge,
carry, and costs. Historical-looking paths remain synthetic. Identical market,
contract, configuration, and seed inputs must produce identical outputs.
