from __future__ import annotations

import math

import pytest

from alm_hedging.pricing import black_scholes_price


def test_put_call_parity() -> None:
    spot, strike, maturity, rate, dividend, volatility = (
        100.0,
        105.0,
        2.0,
        0.035,
        0.012,
        0.24,
    )
    call = black_scholes_price(
        spot, strike, maturity, rate, dividend, volatility, "call"
    )
    put = black_scholes_price(spot, strike, maturity, rate, dividend, volatility, "put")
    expected = spot * math.exp(-dividend * maturity) - strike * math.exp(
        -rate * maturity
    )
    assert call - put == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize(
    ("option_type", "spot", "strike", "expected"),
    [("call", 110.0, 100.0, 10.0), ("put", 90.0, 100.0, 10.0)],
)
def test_expiry_is_intrinsic(
    option_type: str, spot: float, strike: float, expected: float
) -> None:
    value = black_scholes_price(
        spot,
        strike,
        0.0,
        0.03,
        0.01,
        0.20,
        option_type,  # type: ignore[arg-type]
    )
    assert value == expected


def test_zero_volatility_is_discounted_forward_intrinsic() -> None:
    value = black_scholes_price(100.0, 95.0, 1.0, 0.03, 0.01, 0.0)
    expected = max(100.0 * math.exp(-0.01) - 95.0 * math.exp(-0.03), 0.0)
    assert value == pytest.approx(expected)


def test_option_bounds() -> None:
    call = black_scholes_price(100.0, 100.0, 1.0, 0.03, 0.01, 0.20)
    assert 0.0 < call < 100.0 * math.exp(-0.01)
