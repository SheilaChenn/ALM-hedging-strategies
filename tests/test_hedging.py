from __future__ import annotations

from alm_hedging.optimization import HedgeConstraints, HedgeInstrument, optimize_hedge


def test_hedge_trade_opposes_liability_exposure() -> None:
    future = HedgeInstrument(
        name="index_future",
        kind="future",
        tenor=1.0,
        price=100.0,
        exposures={"delta": 1.0},
        max_position=1_000.0,
        max_trade=1_000.0,
        lot_size=1.0,
    )
    solution = optimize_hedge(
        {"delta": 125.0},
        [future],
        constraints=HedgeConstraints(
            premium_budget=0.0,
            turnover_budget=100_000.0,
        ),
    )

    assert solution.success
    assert solution.trades["index_future"] < 0.0
    assert abs(solution.residual_exposures["delta"]) < 1.0


def test_positive_vega_liability_buys_negative_target_vega() -> None:
    option = HedgeInstrument(
        name="put_option",
        kind="option",
        tenor=1.0,
        price=8.0,
        premium=8.0,
        exposures={"vega": 2.0},
        max_position=100.0,
        max_trade=100.0,
        lot_size=1.0,
    )
    solution = optimize_hedge(
        {"vega": 20.0},
        [option],
        constraints=HedgeConstraints(premium_budget=1_000.0, turnover_budget=1_000.0),
    )
    assert solution.positions["put_option"] < 0.0
