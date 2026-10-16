from __future__ import annotations

import numpy as np
import pytest

from alm_hedging.contracts import (
    FIAAnnualPointToPoint,
    RILABufferedSegment,
    RILAFlooredSegment,
)


def test_buffered_rila_boundaries() -> None:
    contract = RILABufferedSegment(
        notional=100.0,
        maturity=1.0,
        buffer=0.10,
        cap=0.20,
        floor=-0.50,
    )
    returns = np.array([-0.80, -0.10, 0.0, 0.20, 0.50])
    np.testing.assert_allclose(
        contract.credited_return(returns),
        [-0.50, 0.0, 0.0, 0.20, 0.20],
    )


def test_floored_rila_honors_floor_cap_and_participation() -> None:
    contract = RILAFlooredSegment(
        notional=100.0,
        maturity=1.0,
        floor=-0.10,
        cap=0.15,
        participation=0.50,
    )
    np.testing.assert_allclose(
        contract.credited_return(np.array([-1.0, 0.10, 1.0])),
        [-0.10, 0.05, 0.15],
    )


def test_fia_compounds_annual_credits_and_floor() -> None:
    contract = FIAAnnualPointToPoint(
        notional=100.0,
        maturity=2.0,
        annual_cap=0.10,
        participation=1.0,
    )
    times = np.array([0.0, 1.0, 2.0])
    paths = np.array([[100.0, 120.0, 108.0], [100.0, 80.0, 120.0]])
    np.testing.assert_allclose(contract.payoff(paths, times), [110.0, 110.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"notional": -1.0, "maturity": 1.0, "floor": -0.1, "cap": 0.1},
        {"notional": 1.0, "maturity": 0.0, "floor": -0.1, "cap": 0.1},
        {"notional": 1.0, "maturity": 1.0, "floor": 0.2, "cap": 0.1},
    ],
)
def test_invalid_contract_terms_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RILAFlooredSegment(**kwargs)
