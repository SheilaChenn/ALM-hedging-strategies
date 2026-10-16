from __future__ import annotations

import pytest

from alm_hedging.market import MarketState
from alm_hedging.pricing import market_option_price
from alm_hedging.risk import BumpSizes, Greeks, aggregate_greeks, bump_and_revalue


def test_finite_difference_greeks_are_stable(market: MarketState) -> None:
    def pricer(state: MarketState) -> float:
        return market_option_price(state, 100.0, 1.0, "call")

    wide = bump_and_revalue(
        pricer,
        market,
        BumpSizes(spot_relative=0.01, vol_absolute=0.01, rate_bp=2.0),
    )
    narrow = bump_and_revalue(
        pricer,
        market,
        BumpSizes(spot_relative=0.005, vol_absolute=0.005, rate_bp=1.0),
    )

    assert narrow.delta > 0.0
    assert narrow.gamma > 0.0
    assert narrow.vega > 0.0
    assert narrow.delta == pytest.approx(wide.delta, rel=2e-3)
    assert narrow.gamma == pytest.approx(wide.gamma, rel=1e-2)
    assert narrow.vega == pytest.approx(wide.vega, rel=2e-3)


def test_portfolio_greek_aggregation_is_quantity_weighted() -> None:
    first = Greeks(10.0, 2.0, 0.1, 4.0, -5.0, -0.0005, -1.0)
    second = Greeks(3.0, -1.0, 0.2, 1.0, 2.0, 0.0002, -0.5)
    total = aggregate_greeks({"first": (2.0, first), "second": (-3.0, second)})

    assert total.pv == pytest.approx(11.0)
    assert total.delta == pytest.approx(7.0)
    assert total.gamma == pytest.approx(-0.4)
    assert total.vega == pytest.approx(5.0)
