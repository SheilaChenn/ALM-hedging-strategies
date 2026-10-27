# ALM hedging framework

An educational, reproducible asset-liability management prototype for synthetic
RILA, FIA, and IUL cohorts. It prices representative indexed benefits, reports
market sensitivities and full-revaluation stresses, and illustrates constrained
hedge recommendations. It is not investment advice or a production valuation
system.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The committed data are synthetic and versioned. No credentials or external
market-data service are required.

## Run

Launch the interactive dashboard:

```bash
streamlit run app.py --server.port 8765
```

Run the end-to-end research narrative:

```bash
jupyter nbconvert --to notebook --execute \
  notebooks/alm_hedging_research.ipynb \
  --output alm_hedging_research.executed.ipynb \
  --output-dir /tmp
```

The notebook and dashboard import the same production package under
`src/alm_hedging`; neither contains a parallel pricing implementation.

## Quality checks

```bash
pytest
pytest --cov=alm_hedging --cov-report=term-missing
ruff check .
ruff format --check .
mypy src
```

## Model validation

[`MODEL_VALIDATION_REPORT.md`](MODEL_VALIDATION_REPORT.md) documents an
independent validation of the pricing, Greeks, and hedging engine, the
material corrections it produced, and the modeling limitations that remain.
Reproduce the benchmark tables with:

```bash
python validation/run_validation.py            # ~1 minute, full Monte Carlo sizes
python validation/run_validation.py --quick    # smaller sample sizes
```

## October 2026 market shock case study

*Hedging Index-Linked Insurance Liabilities Amid Equity Selloffs and Treasury
Yield Volatility* applies the validated engine to a synthetic $500M RILA/FIA/IUL
block struck just before the October 2026 AI-related equity selloff and the
Treasury yield swings around it. Documented market events are dated and
sourced; the shock sizes, the portfolio, the hedge instruments, and every
model-generated hedge are illustrative and are not Pacific Life positions or
trading decisions.

- `src/alm_hedging/case_study.py` is the single reproducible source of every
  number: market states, cohorts, Greeks, full-revaluation scenario sets, the
  three hedge strategies (delta-only, multi-Greek, stress-aware constrained),
  the out-of-sample effectiveness metrics, the robustness sets, and the
  decision rule.
- `docs/case_study/october-2026-market-shock-results.json` is the committed
  output of `run_case_study()`; the PDF and the dashboard both read it.
- `docs/case_study/october-2026-market-shock-case-study.pdf` is the three-page
  brief (market shock and exposures; hedging alternatives and recommendation;
  assumptions, workflow, validation evidence, limitations).
- The dashboard's **Oct 2026 case study** tab presents the same results
  interactively and offers the PDF and JSON for download.
- `tests/test_case_study.py` re-runs the module and checks that the committed
  JSON reconciles, recomputes the effectiveness metrics with
  `validation/reference.py`, and rebuilds the hedged stress P&L from the trades.

Rebuild the results and the PDF (about one minute; `matplotlib` is in the dev
extras):

```bash
python scripts/build_case_study.py            # recompute JSON, then render the PDF
python scripts/build_case_study.py --reuse    # re-render the PDF from the committed JSON
```

## Simple model

`simple/` is a self-contained, illustrative version of the framework for
readers who want the core idea without the full machinery: one year, one
index, $100 of account value per product, closed-form Black-Scholes with flat
synthetic inputs, and no imports from `src/alm_hedging`.

- `simple/simple_model.py`: Black-Scholes legs with analytic Greeks; RILA,
  FIA, and IUL as lists of vanilla options; cap solver; full-revaluation stress
  table; futures delta hedge; `run_simple_model()` returns everything.
- `simple/simple_model.ipynb`: six short sections (assumptions, payoffs,
  pricing, Greeks, cap vs budget, hedge and stress) with saved outputs.
- `simple/build_report.py` renders `simple/simple-model-report.pdf`, a
  one-page summary built from the same `run_simple_model()` output.
- `tests/test_simple_model.py` covers put-call parity, RILA payoff edge cases,
  finite-difference Greeks, the cap solver, and the hedge.

```bash
jupyter nbconvert --to notebook --execute --inplace simple/simple_model.ipynb
python simple/build_report.py
```

## Repository guide

- `simple/`: the illustrative one-file model, its notebook, and its PDF.
- `src/alm_hedging/`: market data, contracts, pricing, risk, scenarios, and
  hedge simulation; `case_study.py` orchestrates the October 2026 case study.
- `scripts/`: `build_case_study.py` renders the case-study JSON and PDF.
- `data/`: synthetic curve, volatility surface, representative cohorts, and
  model assumptions, all tagged with model version and as-of date.
- `docs/`: assumptions, methodology, validation standards, exclusions, and the
  `case_study/` results JSON and PDF.
- `notebooks/`: executable research narrative using only the public package API.
- `tests/`: quantitative identities, numerical stability, replay, and smoke
  tests, plus regression tests pinned to the validation benchmarks.
- `validation/`: independent reference implementations and the benchmark
  runner behind the model validation report.

Start with [model methodology](docs/model-methodology.md) and
[assumptions](docs/assumptions.md). Limitations and intended use are explicit in
[exclusions](docs/exclusions.md).
