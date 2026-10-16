"""Reconciliation tests for the October 2026 market shock case study.

The PDF (``scripts/build_case_study.py``) and the dashboard tab (``app.py``)
both read ``docs/case_study/october-2026-market-shock-results.json``.  These
tests check that the committed JSON is what ``run_case_study()`` produces,
that the effectiveness figures it quotes can be recomputed from the stored
scenario P&L with the independent validation reference, and that the hedged
stress P&L and the costs follow from the trades.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from alm_hedging import case_study as cs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation import reference as ref  # noqa: E402


@pytest.fixture(scope="module")
def committed() -> cs.CaseStudyResult:
    return cs.load_results()


@pytest.fixture(scope="module")
def recomputed() -> cs.CaseStudyResult:
    return cs.run_case_study()


def _flatten(prefix: str, value: object, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}", v, out)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _flatten(f"{prefix}[{i}]", v, out)
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, int | float):
        out[prefix] = float(value)


def test_committed_json_matches_rerun(
    committed: cs.CaseStudyResult, recomputed: cs.CaseStudyResult
) -> None:
    """Every number the PDF and dashboard consume is reproduced by the module."""
    a = committed.to_dict()
    b = recomputed.to_dict()
    for volatile in ("runtime_seconds",):
        a.pop(volatile)
        b.pop(volatile)
    flat_a: dict[str, float] = {}
    flat_b: dict[str, float] = {}
    _flatten("", a, flat_a)
    _flatten("", b, flat_b)
    assert flat_a.keys() == flat_b.keys()
    mismatches = [
        k
        for k in flat_a
        if not math.isclose(flat_a[k], flat_b[k], rel_tol=1e-9, abs_tol=1e-6)
    ]
    assert mismatches == []
    for key in cs.STRATEGY_ORDER:
        assert committed.strategies[key].trades == recomputed.strategies[key].trades
    assert committed.recommendation["key"] == recomputed.recommendation["key"]
    assert (
        committed.recommendation["rationale"] == recomputed.recommendation["rationale"]
    )


def test_effectiveness_recomputed_with_validation_reference(
    committed: cs.CaseStudyResult,
) -> None:
    unhedged = np.asarray(committed.oos_unhedged_pnl)
    assert unhedged.size == committed.scenarios["out_of_sample_count"]
    for key in cs.STRATEGY_ORDER:
        s = committed.strategies[key]
        hedged = np.asarray(s.oos_hedged_pnl)
        independent = ref.scenario_effectiveness(-unhedged, hedged - unhedged)
        for metric in (
            "variance_reduction",
            "dollar_offset_slope",
            "es95_unhedged",
            "es95_hedged",
            "es95_reduction",
        ):
            assert s.out_of_sample[metric] == pytest.approx(
                independent[metric], rel=1e-9, abs=1e-9
            )
        assert s.out_of_sample["hedged_sd"] == pytest.approx(
            float(np.std(hedged)), rel=1e-9
        )
        assert s.out_of_sample["unhedged_sd"] == pytest.approx(
            float(np.std(unhedged)), rel=1e-9
        )


def test_costs_and_budgets_follow_from_trades(committed: cs.CaseStudyResult) -> None:
    cfg = committed.config
    budget_premium = cfg["premium_budget_pct"] * sum(
        r["account_value"] for r in committed.portfolio
    )
    by_key = {i["key"]: i for i in committed.instruments}
    for key in cs.STRATEGY_ORDER:
        s = committed.strategies[key]
        assert s.status == "Optimal"
        premium = sum(
            abs(q) * by_key[k]["premium_per_contract"] for k, q in s.trades.items()
        )
        liquidity = sum(
            abs(q) * by_key[k]["liquidity_per_contract"] for k, q in s.trades.items()
        )
        cost = sum(abs(q) * by_key[k]["transaction_cost"] for k, q in s.trades.items())
        assert s.premium_used == pytest.approx(premium, rel=1e-9, abs=1e-6)
        assert s.liquidity_used == pytest.approx(liquidity, rel=1e-9, abs=1e-6)
        assert s.transaction_cost == pytest.approx(cost, rel=1e-9, abs=1e-6)
        assert s.premium_used <= budget_premium * (1 + 1e-9)
        assert s.liquidity_used <= cfg["liquidity_budget"] * (1 + 1e-9)
        for k, q in s.trades.items():
            assert abs(q) <= by_key[k]["max_position"] + 1e-9
            assert q == pytest.approx(round(q))
        for greek in ("delta", "vega", "dv01"):
            hedge = sum(q * by_key[k][greek] for k, q in s.trades.items())
            assert s.hedge_greeks[greek] == pytest.approx(hedge, rel=1e-9, abs=1e-6)
            assert s.residual_greeks[greek] == pytest.approx(
                committed.greeks["Net (insurer)"][greek] + hedge, rel=1e-9, abs=1e-6
            )


def test_hedged_stress_pnl_rebuilt_from_trades(committed: cs.CaseStudyResult) -> None:
    """Named-stress P&L = unhedged surplus P&L + Σ trades × instrument P&L."""
    config = cs.CaseStudyConfig(**committed.config)
    _, post = cs.build_markets(config)
    instruments = {i.key: i for i in cs.build_instruments(config, post)}
    base = {k: i.value(post) for k, i in instruments.items()}
    for scenario in cs.NAMED_STRESSES:
        state = scenario.apply_market(post)
        inst_pnl = {k: i.value(state) - base[k] for k, i in instruments.items()}
        unhedged = committed.unhedged_stress_pnl[scenario.name]
        for key in cs.STRATEGY_ORDER:
            s = committed.strategies[key]
            rebuilt = unhedged + sum(q * inst_pnl[k] for k, q in s.trades.items())
            assert s.stress_pnl[scenario.name] == pytest.approx(
                rebuilt, rel=1e-7, abs=1.0
            )


def test_recommendation_rule_is_reproducible(committed: cs.CaseStudyResult) -> None:
    rec = cs.select_recommendation(committed.strategies, committed.robustness)
    assert rec["key"] == committed.recommendation["key"]
    assert rec["key"] in cs.STRATEGY_ORDER
    best = max(
        s.out_of_sample["variance_reduction"] for s in committed.strategies.values()
    )
    chosen = committed.strategies[rec["key"]]
    assert chosen.out_of_sample["variance_reduction"] >= best - 0.02
    hm = cs.headline_metrics(committed)
    table = cs.comparison_table(committed)
    row = table[table["Recommended"]].iloc[0]
    assert row["OOS variance reduction"] == hm["recommended_variance_reduction"]
    assert row["Premium paid"] == hm["recommended_premium"]
    assert hm["liability_pv_post"] == pytest.approx(
        sum(r["pv_post_shock"] for r in committed.portfolio)
    )
    assert committed.shock_pnl["surplus_change"] == pytest.approx(
        committed.shock_pnl["asset_change"] - committed.shock_pnl["liability_change"],
        rel=1e-9,
    )


def test_documented_events_are_separated_and_sourced(
    committed: cs.CaseStudyResult,
) -> None:
    assert len(committed.documented_events) >= 5
    for event in committed.documented_events:
        assert re.fullmatch(r"2026-\d{2}-\d{2}", event["date"])
        assert event["url"].startswith("https://")
        assert event["source"]
    assert (
        "ILLUSTRATIVE" in committed.notice.upper() or "illustrative" in committed.notice
    )
    assert any("not Pacific Life" in text for text in [committed.notice])
    assert len(committed.illustrative_assumptions) >= 5
    assert len(committed.limitations) >= 5


def test_json_round_trip(committed: cs.CaseStudyResult, tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    cs.save_results(committed, path)
    reloaded = cs.load_results(path)
    assert reloaded.to_dict() == committed.to_dict()
    assert (
        json.loads(path.read_text())["recommendation"]["key"]
        == (committed.recommendation["key"])
    )


def test_pdf_builder_produces_three_pages(
    committed: cs.CaseStudyResult, tmp_path: Path
) -> None:
    pytest.importorskip("matplotlib")
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_case_study  # noqa: PLC0415

    pdf = build_case_study.build_pdf(committed, tmp_path / "case.pdf")
    raw = pdf.read_bytes()
    assert raw.startswith(b"%PDF")
    pages = len(re.findall(rb"/Type\s*/Page[^s]", raw))
    assert pages == 3
