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
the same standard-normal draws to reduce noise. Each simulation step over
\([t_i,t_{i+1}]\) uses the forward drift
\(\Delta(r(t)t)-\Delta(q(t)t)\) and the forward variance
\(\Delta(\sigma_{\mathrm{ATM}}(t)^2 t)\), so \(E[S_T]\) equals the market
forward and the terminal log-variance equals \(\sigma_{\mathrm{ATM}}(T)^2T\)
on every grid date. Monte Carlo therefore reprices at-the-money vanillas but
not the skew.

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
limits. The optimizer receives the insurer's net exposure, \(-\)(liability
Greeks), and drives \(\text{net} + M q\) toward zero; hedge assets therefore
carry the same-signed Greeks as the liability. The continuous SLSQP solution is
multi-started from the current book and the unconstrained least-squares hedge,
rounded to lots, repaired against any violated budget by unwinding the lot that
most improves the violated slack, and refined by single- and pair-lot local
search.

The dashboard's "effectiveness" is
\(1-\lVert w\circ \text{residual}\rVert_2/\lVert w\circ \text{net}\rVert_2\)
over every Greek the eligible instruments can move, with \(w\) the inverse of
the larger of the initial exposure and a reference exposure. It is a
first-order, in-sample measure at the valuation market; it does not account
for gamma, basis, or scenario P&L. `MODEL_VALIDATION_REPORT.md` compares it
with full-revaluation scenario effectiveness.

Dynamic simulations rebalance at the configured schedule, deduct explicit
transaction costs when provided, and attribute total P&L to delta, gamma, vega,
rho, theta, costs, and an unexplained residual using beginning-of-period net
Greeks (all reported Greeks, not only the hedged targets). Historical-looking
paths remain synthetic. Identical market, contract, configuration, and seed
inputs must produce identical outputs.
