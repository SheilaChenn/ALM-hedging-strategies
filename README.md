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

## Repository guide

- `src/alm_hedging/`: market data, contracts, pricing, risk, scenarios, and
  hedge simulation.
- `data/`: synthetic curve, volatility surface, representative cohorts, and
  model assumptions, all tagged with model version and as-of date.
- `docs/`: assumptions, methodology, validation standards, and exclusions.
- `notebooks/`: executable research narrative using only the public package API.
- `tests/`: quantitative identities, numerical stability, replay, and smoke
  tests, plus regression tests pinned to the validation benchmarks.
- `validation/`: independent reference implementations and the benchmark
  runner behind the model validation report.

Start with [model methodology](docs/model-methodology.md) and
[assumptions](docs/assumptions.md). Limitations and intended use are explicit in
[exclusions](docs/exclusions.md).
