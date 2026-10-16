from __future__ import annotations

import numpy as np
import pytest

from alm_hedging.market import MarketState, VolSurface, ZeroCurve


@pytest.fixture
def market() -> MarketState:
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    log_moneyness = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    return MarketState(
        spot=100.0,
        rate_curve=ZeroCurve(tenors, np.full_like(tenors, 0.03)),
        dividend_curve=ZeroCurve(tenors, np.full_like(tenors, 0.01)),
        vol_surface=VolSurface(
            tenors,
            log_moneyness,
            np.full((tenors.size, log_moneyness.size), 0.20),
        ),
    )
