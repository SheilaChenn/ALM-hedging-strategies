from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import alm_hedging as ah


def test_public_api_end_to_end() -> None:
    market = ah.SyntheticMarketModel().build()
    contract = ah.RILAFlooredSegment(
        notional=100_000.0,
        maturity=1.0,
        floor=-0.10,
        cap=0.15,
    )
    cohort = ah.Cohort(contract, policy_count=25, label="smoke")
    value = ah.portfolio_value([cohort], market)
    risk = ah.bump_and_revalue(
        lambda state: ah.portfolio_value([cohort], state), market
    )
    stresses = ah.run_stress_scenarios(
        market,
        ah.standard_stress_scenarios()[:2],
        lambda state, liabilities: ah.portfolio_value(liabilities, state),
        [cohort],
    )

    assert value > 0.0
    assert risk.pv == pytest.approx(value)
    assert len(stresses) == 2


def test_streamlit_app_smoke() -> None:
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()
    assert not app.exception
