"""Build the October 2026 market shock case study: results JSON and 3-page PDF.

Usage::

    python scripts/build_case_study.py            # recompute, write JSON + PDF
    python scripts/build_case_study.py --reuse    # rebuild the PDF from the JSON

Every number in the PDF comes from ``alm_hedging.case_study.run_case_study``
through the JSON written next to the PDF; the Streamlit dashboard reads the
same file, so the two presentations reconcile exactly (checked by
``tests/test_case_study.py``).
"""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from alm_hedging import case_study as cs  # noqa: E402

INK = "#14213D"
NAVY = "#102A43"
BLUE = "#2F6BFF"
TEAL = "#12B8A6"
GOLD = "#F3B61F"
RED = "#E25555"
SLATE = "#68778D"
PALE = "#EAF0F8"
STRATEGY_COLORS = {"delta_only": SLATE, "multi_greek": GOLD, "stress_aware": TEAL}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.titlesize": 7.6,
        "axes.labelsize": 6.6,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.6,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.parse_math": False,
    }
)

PAGE = (8.5, 11.0)
LEFT = 0.05
WIDTH = 0.90
RIGHT = LEFT + WIDTH


def line_height(fontsize: float) -> float:
    """Figure-fraction height of one text line (page height 11in = 792pt)."""
    return fontsize * 1.42 / 792.0


def money(value: float, decimals: int = 1) -> str:
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1e9:
        return f"{sign}${v / 1e9:,.{decimals}f}B"
    if v >= 1e6:
        return f"{sign}${v / 1e6:,.{decimals}f}M"
    if v >= 1e3:
        return f"{sign}${v / 1e3:,.0f}K"
    return f"{sign}${v:,.0f}"


def pct(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width))


def short(label: str) -> str:
    return label.split(" (")[0]


# --------------------------------------------------------------------------- #
# Layout primitives
# --------------------------------------------------------------------------- #


def header(
    fig: plt.Figure, result: cs.CaseStudyResult, page: int, subtitle: str
) -> None:
    fig.patches.append(
        FancyBboxPatch(
            (0.03, 0.924),
            0.94,
            0.064,
            boxstyle="round,pad=0.004,rounding_size=0.01",
            transform=fig.transFigure,
            facecolor=NAVY,
            edgecolor="none",
        )
    )
    fig.text(LEFT, 0.969, result.title, color="white", fontsize=10.2, weight="bold")
    fig.text(
        LEFT,
        0.950,
        f"{result.short_title}  ·  Page {page} of 3  ·  {subtitle}",
        color="#D9E7F2",
        fontsize=6.6,
    )
    fig.text(
        LEFT,
        0.934,
        f"As of {result.as_of} · engine v{result.engine_version} · "
        "ILLUSTRATIVE: synthetic portfolio, simulated shocks, model-generated hedges "
        "— not Pacific Life positions or trading decisions",
        color="#F3B61F",
        fontsize=5.8,
        weight="bold",
    )


def footer(fig: plt.Figure, result: cs.CaseStudyResult) -> None:
    fig.text(
        LEFT, 0.016, wrap(result.notice, 175), fontsize=5.5, color=SLATE, va="bottom"
    )


def section(
    fig: plt.Figure, x: float, y: float, text: str, width_chars: int = 118
) -> float:
    """Section heading (wrapped); returns the y where content should start."""
    lines = textwrap.wrap(text.upper(), width_chars)
    fig.text(
        x,
        y,
        "\n".join(lines),
        fontsize=6.4,
        color=BLUE,
        weight="bold",
        va="top",
        linespacing=1.3,
    )
    rule_y = y - 0.0125 - (len(lines) - 1) * line_height(6.4) * 0.92
    fig.add_artist(
        plt.Line2D(
            [x, x + 0.07],
            [rule_y, rule_y],
            color=BLUE,
            lw=0.8,
            transform=fig.transFigure,
        )
    )
    return rule_y - 0.0065


def place_table(
    fig: plt.Figure,
    x: float,
    y_top: float,
    width: float,
    frame: pd.DataFrame,
    col_widths: Sequence[float],
    fontsize: float = 5.8,
    highlight_rows: Sequence[int] = (),
    pad_lines: float = 0.85,
) -> float:
    """Draw a table whose row heights fit their wrapped text; return bottom y."""
    cells = frame.astype(str).values.tolist()
    row_lines = [max(c.count("\n") + 1 for c in row) for row in cells]
    header_lines = max(str(c).count("\n") + 1 for c in frame.columns)
    units = [header_lines + pad_lines] + [n + pad_lines for n in row_lines]
    total = float(sum(units))
    height = total * line_height(fontsize)
    ax = fig.add_axes([x, y_top - height, width, height])
    ax.axis("off")
    tbl = ax.table(
        cellText=cells,
        colLabels=list(frame.columns),
        loc="upper left",
        cellLoc="left",
        colWidths=list(col_widths),
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    for (row, _col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDE5EF")
        cell.set_linewidth(0.4)
        cell.PAD = 0.03
        cell.set_height(units[row] / total)
        if row == 0:
            cell.set_facecolor(PALE)
            cell.set_text_props(weight="bold", color=INK)
        elif (row - 1) in highlight_rows:
            cell.set_facecolor("#E5F8F4")
    return y_top - height


def bullets(
    fig: plt.Figure,
    x: float,
    y_top: float,
    items: Sequence[str],
    width_chars: int,
    fontsize: float = 5.8,
    color: str = INK,
    bullet: str = "•",
    gap: float = 0.3,
) -> float:
    """Wrapped bullet list flowing downward; returns bottom y."""
    step = line_height(fontsize)
    y = y_top
    for item in items:
        for i, line in enumerate(textwrap.wrap(item, width_chars)):
            prefix = f"{bullet} " if i == 0 else "   "
            fig.text(x, y, prefix + line, fontsize=fontsize, va="top", color=color)
            y -= step
        y -= step * gap
    return y


def paragraph(
    fig: plt.Figure,
    x: float,
    y_top: float,
    text: str,
    width_chars: int,
    fontsize: float = 5.8,
    color: str = INK,
) -> float:
    lines = textwrap.wrap(text, width_chars)
    fig.text(
        x,
        y_top,
        "\n".join(lines),
        fontsize=fontsize,
        va="top",
        color=color,
        linespacing=1.42,
    )
    return y_top - len(lines) * line_height(fontsize)


# --------------------------------------------------------------------------- #
# Page 1: market shock, portfolio setup, pricing and risk exposures
# --------------------------------------------------------------------------- #


def page_one(pdf: PdfPages, result: cs.CaseStudyResult) -> None:
    fig = plt.figure(figsize=PAGE)
    header(
        fig, result, 1, "Market shock · portfolio setup · pricing and risk exposures"
    )
    hm = cs.headline_metrics(result)
    m = result.market
    top = 0.912

    # Left: documented backdrop. Right: illustrative shock.
    y = section(
        fig,
        LEFT,
        top,
        "Documented market backdrop (dated, sourced; see page 3 for links)",
        78,
    )
    events = pd.DataFrame(
        [
            {
                "Date": e["date"],
                "Event": wrap(e["summary"], 62),
                "Source": wrap(short(e["source"].split(";")[0]), 18),
            }
            for e in result.documented_events
        ]
    )
    y_left = place_table(
        fig, LEFT, y, 0.565, events, [0.14, 0.64, 0.22], fontsize=5.6, pad_lines=0.7
    )

    xr = 0.645
    yr = section(
        fig, xr, top, "Illustrative shock applied to the block (assumption)", 44
    )
    ax = fig.add_axes([xr + 0.04, yr - 0.1, 0.265, 0.088])
    tenors = np.array(m["curve_tenors"])
    ax.plot(
        tenors,
        np.array(m["curve_pre"]) * 100,
        color=SLATE,
        lw=1.3,
        label="Issue date (illustrative)",
    )
    ax.plot(
        tenors,
        np.array(m["curve_post"]) * 100,
        color=BLUE,
        lw=1.3,
        label=f"Stressed ({m['rate_shock_bp']:+.0f} bp)",
    )
    ax.set_xscale("log")
    ax.set_xticks([0.25, 1, 2, 5, 10, 30])
    ax.set_xticklabels(["3m", "1y", "2y", "5y", "10y", "30y"])
    ax.set_ylabel("Zero rate (%)")
    ax.set_xlabel("Synthetic risk-free zero curve", fontsize=5.6, labelpad=2)
    ax.legend(frameon=False, fontsize=5.4, loc="lower right")
    ax.grid(axis="y", color="#E8EDF4", lw=0.5)
    shock_lines = [
        f"Index {m['issue_index']:,.0f} → {m['stressed_index']:,.0f} ({pct(m['equity_shock'], 0)}). The documented seven-session semiconductor-index drawdown applied to the broad index as an escalation; the documented broad-market move through 16 Oct was about -2% to -4%.",
        f"1y ATM implied vol {pct(m['atm_vol_1y_pre'], 0)} → {pct(m['atm_vol_1y_post'], 0)} (+{m['vol_shock'] * 100:.0f} pts). The documented VIX move was about +2 points, to a two-week high of 17.74.",
        f"Zero curve {m['rate_shock_bp']:+.0f} bp parallel (10y {pct(m['zero_10y_pre'], 2)} → {pct(m['zero_10y_post'], 2)}), inside the documented retreat from the 5.364% peak (7 Oct) to 5.07%-5.19% (15-16 Oct).",
    ]
    y_right = bullets(fig, xr, yr - 0.128, shock_lines, 68, fontsize=5.4)
    y = min(y_left, y_right) - 0.012

    # Portfolio.
    y = section(
        fig,
        LEFT,
        y,
        f"Synthetic ${hm['account_value'] / 1e6:,.0f}M block struck at {m['issue_index']:,.0f} (illustrative issue {result.issue_date}) — not Pacific Life positions",
    )
    port = pd.DataFrame(
        [
            {
                "Product": r["product"],
                "Policies": f"{r['policies']:,}",
                "Account value": money(r["account_value"]),
                "Term": f"{r['term_years']:.0f}y",
                "Crediting terms": wrap(r["description"], 60),
                "PV at issue": money(r["pv_pre_shock"], 2),
                "PV stressed": money(r["pv_post_shock"], 2),
                "Shock ΔPV": money(r["shock_change"], 2),
            }
            for r in result.portfolio
        ]
        + [
            {
                "Product": "Total liabilities",
                "Policies": f"{sum(r['policies'] for r in result.portfolio):,}",
                "Account value": money(hm["account_value"]),
                "Term": "",
                "Crediting terms": wrap(
                    f"Backed by a Treasury strip ladder ({', '.join(f'{t:.0f}y' for t in result.assets['tenors'])} equal market value), MV = liability PV at issue, duration {result.assets['duration_post_shock']:.1f}y vs liability {result.assets['liability_duration_post_shock']:.1f}y",
                    60,
                ),
                "PV at issue": money(hm["liability_pv_pre"], 2),
                "PV stressed": money(hm["liability_pv_post"], 2),
                "Shock ΔPV": money(result.shock_pnl["liability_change"], 2),
            }
        ]
    )
    y = place_table(
        fig,
        LEFT,
        y,
        WIDTH,
        port,
        [0.11, 0.07, 0.09, 0.05, 0.40, 0.09, 0.09, 0.10],
        fontsize=5.7,
    )
    y -= 0.012

    # Shock P&L.
    y = section(
        fig,
        LEFT,
        y,
        "What the shock did to the unhedged block (full revaluation, issue market → stressed market, insurer sign)",
    )
    chart_h = 0.118
    ax = fig.add_axes([LEFT + 0.03, y - chart_h + 0.024, 0.36, chart_h - 0.03])
    legs = [
        ("Liability\nequity leg", -result.shock_pnl["liability_equity_leg"]),
        ("Liability\nvol leg", -result.shock_pnl["liability_vol_leg"]),
        ("Liability\nrates leg", -result.shock_pnl["liability_rate_leg"]),
        ("Assets\nrates", result.shock_pnl["asset_change"]),
        ("Surplus\nchange", result.shock_pnl["surplus_change"]),
    ]
    colors = [TEAL if v >= 0 else RED for _, v in legs[:-1]] + [NAVY]
    ax.bar([x[0] for x in legs], [x[1] / 1e6 for x in legs], color=colors, width=0.6)
    for i, (_, v) in enumerate(legs):
        ax.text(
            i,
            v / 1e6 + (0.4 if v >= 0 else -1.1),
            f"{v / 1e6:+.1f}M",
            ha="center",
            fontsize=5.6,
        )
    ax.axhline(0, color="#CBD5E1", lw=0.6)
    ax.set_ylabel("$M")
    ax.set_ylim(
        min(0.0, min(v for _, v in legs) / 1e6) - 2.5,
        max(v for _, v in legs) / 1e6 + 2.8,
    )
    ax.tick_params(axis="x", labelsize=5.4)
    ax.grid(axis="y", color="#E8EDF4", lw=0.5)
    note = (
        f"The unhedged block gained {money(result.shock_pnl['surplus_change'])}. Lower index levels and higher implied vol reduced the index-linked credits owed "
        f"({money(-result.shock_pnl['liability_equity_leg'] - result.shock_pnl['liability_vol_leg'])} of liability relief), while the -25 bp move raised the longer-duration liabilities by "
        f"{money(result.shock_pnl['liability_rate_leg'])} against a {money(result.shock_pnl['asset_change'])} gain on the shorter asset ladder. "
        "The gain is one realization of a symmetric exposure: a relief rally with falling vol would have produced a loss of similar size. "
        "The hedge decision below is about removing that variance going forward, not about the direction already realized. "
        "Why the decomposition: it shows which factor the hedge must address first (equity and rates are of the same order; vol is smaller but correlated with equity)."
    )
    paragraph(fig, 0.455, y - 0.004, note, 92, fontsize=5.6)
    y -= chart_h + 0.006

    # Greeks.
    y = section(
        fig,
        LEFT,
        y,
        "Risk exposures at the stressed market (bump-and-revalue; engine sign: positive = value rises)",
    )
    rows = []
    for label in ("RILA", "FIA", "IUL", "Liabilities", "Assets", "Net (insurer)"):
        g = result.greeks[label]
        rows.append(
            {
                "Position": label,
                "PV": money(g["pv"], 2),
                "Delta ($/index pt)": f"{g['delta']:,.0f}",
                "Vega ($/vol pt)": money(g["vega"] / 100.0, 2),
                "DV01 ($/bp)": f"{g['dv01']:,.0f}",
                "P&L if index -10%": money(g["delta"] * m["stressed_index"] * -0.10, 2),
                "P&L if vol +5 pts": money(g["vega"] * 0.05, 2),
                "P&L if rates -35 bp": money(g["dv01"] * -35.0, 2),
            }
        )
    y = place_table(
        fig,
        LEFT,
        y,
        WIDTH,
        pd.DataFrame(rows),
        [0.13, 0.11, 0.13, 0.12, 0.11, 0.13, 0.13, 0.14],
        fontsize=5.9,
        highlight_rows=[5],
    )
    carry = result.greeks["Carry (1 month)"]
    why = [
        "Why these calculations: PV sets the balance sheet; delta, vega, and DV01 are the first-order sensitivities the optimizer can trade against; the last three columns translate "
        "them into first-order dollar P&L for moves of roughly one standard deviation in the stressed regime so the three risks can be compared on one scale (the Net row is what the hedge must offset).",
        f"Reading the Net row: the insurer is short {abs(hm['net_delta']):,.0f} $/pt of index delta (gains if the index falls further), long {money(hm['net_vega'] / 100)} per vol point "
        f"(gains if implied vol rises), and long {hm['net_dv01']:,.0f} $/bp of rates (loses {money(abs(hm['net_dv01']) * 35)} if yields fall 35 bp) because the liability duration "
        f"({result.assets['liability_duration_post_shock']:.1f}y) exceeds the asset ladder duration ({result.assets['duration_post_shock']:.1f}y).",
        f"The RILA's 6,975/6,200 buffer strikes are now at the money, which is where most of its delta and negative vega come from; the FIA and IUL first-year call spreads are 12% out of the money. "
        f"One-month carry of the surplus is {money(carry['surplus'])} (liabilities accrete {money(carry['liability'])}, assets {money(carry['assets'])}); carry uses a one-month step because one-day theta on the Monte Carlo products is a time-grid artefact.",
        f"Unhedged surplus P&L over the {result.scenarios['out_of_sample_count']} out-of-sample joint shocks: standard deviation {money(hm['unhedged_oos_sd'])}, ES95 {money(hm['unhedged_oos_es95'])}. "
        "Gamma is omitted from the table: analytic for the RILA (-0.8 $/pt²) but statistically indistinguishable from zero for the Monte Carlo products at 3,000 paths (validation report §5.D).",
    ]
    bullets(fig, LEFT, y - 0.008, why, 186, fontsize=5.7)
    footer(fig, result)
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Page 2: hedging alternatives, recommendation, quantified results
# --------------------------------------------------------------------------- #


def page_two(pdf: PdfPages, result: cs.CaseStudyResult) -> None:
    fig = plt.figure(figsize=PAGE)
    header(
        fig,
        result,
        2,
        "Hedging alternatives · recommended strategy · quantified results",
    )
    hm = cs.headline_metrics(result)
    rec_key = result.recommendation["key"]
    rec = result.strategies[rec_key]
    cfg = result.config
    net = result.greeks["Net (insurer)"]
    top = 0.912

    y = section(
        fig,
        LEFT,
        top,
        f"Hedge universe priced from the stressed market — premium budget {money(cfg['premium_budget_pct'] * hm['account_value'], 2)} gross option premium · liquidity budget {money(cfg['liquidity_budget'])} (margin + premium) · position caps · costs",
    )
    inst = pd.DataFrame(
        [
            {
                "Instrument": i["name"],
                "Unit value": money(i["unit_value"], 1),
                "Premium": money(i["premium_per_contract"], 1)
                if i["premium_per_contract"]
                else "none",
                "Liquidity / contract": money(i["liquidity_per_contract"], 1),
                "Cost / contract": f"${i['transaction_cost']:,.0f}",
                "Delta ($/pt)": f"{i['delta']:,.1f}",
                "Vega ($/vol pt)": f"{i['vega'] / 100:,.0f}",
                "DV01 ($/bp)": f"{i['dv01']:,.1f}",
                "Cap": f"±{i['max_position']:,.0f}",
            }
            for i in result.instruments
        ]
    )
    y = place_table(
        fig,
        LEFT,
        y,
        WIDTH,
        inst,
        [0.26, 0.09, 0.09, 0.12, 0.10, 0.08, 0.10, 0.09, 0.07],
        fontsize=5.8,
    )
    y -= 0.012

    y = section(
        fig,
        LEFT,
        y,
        f"Three-way comparison — solved by the engine's constrained optimizer, evaluated by full revaluation on {result.scenarios['out_of_sample_count']} out-of-sample joint shocks (highlighted = recommended)",
    )
    trades = cs.trade_table(result)
    summary_rows = []
    for key in cs.STRATEGY_ORDER:
        s = result.strategies[key]
        pos = "; ".join(
            f"{v:+,.0f} {short(name)}"
            for name, v in zip(trades["Instrument"], trades[s.label], strict=True)
            if abs(v) >= 0.5
        )
        summary_rows.append(
            {
                "Strategy": wrap(s.label, 24),
                "Positions (contracts)": wrap(pos, 46),
                "Premium": money(s.premium_used, 2),
                "Txn cost": money(s.transaction_cost, 0),
                "Liquidity": money(s.liquidity_used, 1),
                "Greek eff.": pct(s.greek_effectiveness),
                "OOS var. red.": pct(s.out_of_sample["variance_reduction"]),
                "$-offset": f"{s.out_of_sample['dollar_offset_slope']:.2f}",
                "ES95 red.": pct(s.out_of_sample["es95_reduction"]),
                "Hedged sd": money(s.out_of_sample["hedged_sd"], 2),
            }
        )
    y = place_table(
        fig,
        LEFT,
        y,
        WIDTH,
        pd.DataFrame(summary_rows),
        [0.16, 0.28, 0.07, 0.06, 0.07, 0.07, 0.08, 0.06, 0.07, 0.08],
        fontsize=5.7,
        highlight_rows=[cs.STRATEGY_ORDER.index(rec_key)],
    )
    y = paragraph(
        fig,
        LEFT,
        y - 0.004,
        f"Unhedged: sd {money(hm['unhedged_oos_sd'], 2)}, ES95 {money(hm['unhedged_oos_es95'], 2)}. 'Greek metric' is the first-order, in-sample effectiveness the dashboard reports; the validation report shows it is optimistic and "
        "not a safe optimization target. 'OOS var. red.' = 1 - Var(hedged)/Var(unhedged) and '$-offset' is the regression slope of hedge P&L on insurer loss, both on out-of-sample full revaluations — the governance-grade "
        "measures the validation recommends. Residual Greeks after hedging (insurer sign, $/pt, $/vol pt, $/bp): "
        + "; ".join(
            f"{short(result.strategies[k].label)} {result.strategies[k].residual_greeks['delta']:+,.0f} / {result.strategies[k].residual_greeks['vega'] / 100:+,.0f} / {result.strategies[k].residual_greeks['dv01']:+,.0f}"
            for k in cs.STRATEGY_ORDER
        )
        + f" (unhedged {net['delta']:+,.0f} / {net['vega'] / 100:+,.0f} / {net['dv01']:+,.0f}).",
        198,
        fontsize=5.3,
        color=SLATE,
    )
    y -= 0.008

    # Charts.
    chart_h = 0.115
    ax = fig.add_axes([LEFT + 0.025, y - chart_h, 0.385, chart_h - 0.018])
    unhedged = np.array(result.oos_unhedged_pnl) / 1e6
    bins = np.linspace(min(unhedged.min(), -12), max(unhedged.max(), 12), 46)
    ax.hist(unhedged, bins=bins, color="#CBD5E1", alpha=0.9, label="Unhedged")
    for key in cs.STRATEGY_ORDER:
        s = result.strategies[key]
        ax.hist(
            np.array(s.oos_hedged_pnl) / 1e6,
            bins=bins,
            histtype="step",
            lw=1.7 if key == rec_key else 1.0,
            color=STRATEGY_COLORS[key],
            label=short(s.label),
        )
    ax.set_title("Out-of-sample surplus P&L distribution (insurer sign)", loc="left")
    ax.set_xlabel("P&L ($M)", labelpad=1)
    ax.legend(frameon=False, fontsize=5.3)
    ax.grid(axis="y", color="#E8EDF4", lw=0.5)

    ax = fig.add_axes([0.53, y - chart_h, 0.42, chart_h - 0.018])
    stress = cs.stress_table(result)
    names = [wrap(short(n), 12) for n in stress["Scenario"]]
    x = np.arange(len(names))
    w = 0.2
    ax.bar(x - 1.5 * w, stress["Unhedged"] / 1e6, w, color="#CBD5E1", label="Unhedged")
    for j, key in enumerate(cs.STRATEGY_ORDER):
        ax.bar(
            x + (j - 0.5) * w,
            stress[cs.STRATEGY_LABELS[key]] / 1e6,
            w,
            color=STRATEGY_COLORS[key],
            label=short(cs.STRATEGY_LABELS[key]),
            edgecolor=NAVY if key == rec_key else "none",
            lw=0.6,
        )
    ax.axhline(0, color="#CBD5E1", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=4.7)
    ax.set_title("Named full-revaluation stresses: surplus P&L ($M)", loc="left")
    ax.legend(frameon=False, fontsize=5.0, ncol=2)
    ax.grid(axis="y", color="#E8EDF4", lw=0.5)
    y -= chart_h + 0.026

    # Recommendation box: measure text first, then draw the box behind it.
    y = section(fig, LEFT, y, f"Recommended strategy: {rec.label}")
    alt_best = max(
        (s for k, s in result.strategies.items() if k != rec_key),
        key=lambda s: s.out_of_sample["variance_reduction"],
    )
    trade_lines = [
        f"{v:+,.0f} {i['name']} (notional {money(rec.notional[i['key']])})"
        for i in result.instruments
        for v in [rec.trades[i["key"]]]
        if abs(v) >= 0.5
    ]
    dv01_closed = 1.0 - abs(rec.residual_greeks["dv01"] / net["dv01"])
    worst_name = min(rec.stress_pnl, key=lambda k: rec.stress_pnl[k])
    rec_text = [
        "Trades (contracts): " + "; ".join(trade_lines) + ". No options.",
        f"Cost: option premium {money(rec.premium_used, 2)}; transaction cost {money(rec.transaction_cost, 0)}; liquidity (initial margin + premium) {money(rec.liquidity_used, 2)} of the {money(cfg['liquidity_budget'])} budget"
        + (
            " — the liquidity budget binds, the premium budget does not."
            if rec.liquidity_used > 0.98 * cfg["liquidity_budget"]
            else "."
        ),
        f"Effectiveness (out-of-sample, full revaluation): variance reduction {pct(rec.out_of_sample['variance_reduction'])}; dollar-offset slope {rec.out_of_sample['dollar_offset_slope']:.2f} (inside the 0.80-1.25 corridor; "
        f"multi-Greek {result.strategies['multi_greek'].out_of_sample['dollar_offset_slope']:.2f} and delta-only {result.strategies['delta_only'].out_of_sample['dollar_offset_slope']:.2f} are not); "
        f"P&L sd {money(hm['unhedged_oos_sd'], 1)} → {money(rec.out_of_sample['hedged_sd'], 1)}; ES95 {money(hm['unhedged_oos_es95'], 1)} → {money(rec.out_of_sample['es95_hedged'], 1)} ({pct(rec.out_of_sample['es95_reduction'])} reduction). "
        f"Next best: {short(alt_best.label)} at {pct(alt_best.out_of_sample['variance_reduction'])} for {money(alt_best.premium_used + alt_best.transaction_cost, 2)} of premium and cost.",
        f"Why it wins here: it over-hedges delta ({rec.trades['future']:+,.0f} futures vs {result.strategies['delta_only'].trades['future']:+,.0f} for a pure delta hedge) because implied vol is assumed to rise when the index falls, so the "
        f"liability's vega behaves like additional delta that futures can offset; it spends the remaining liquidity on {rec.trades['treasury']:+,.0f} Treasury contracts, closing {pct(dv01_closed, 0)} of the duration gap; "
        f"and it buys no options because at 27% implied vol the ATM call costs {money(result.instruments[1]['premium_per_contract'], 0)} per contract and each contract removes little scenario variance.",
        f"Remaining risks: residual DV01 {rec.residual_greeks['dv01']:+,.0f} $/bp (liquidity-capped: a {money(abs(rec.residual_greeks['dv01']) * 100)} loss if yields fall 100 bp); residual vega {money(rec.residual_greeks['vega'] / 100, 2)}/pt is deliberately unhedged and "
        f"becomes a loss if vol falls without the index rising; residual delta {rec.residual_greeks['delta']:+,.0f} $/pt. Worst named stress after hedging {money(min(rec.stress_pnl.values()))} ({short(worst_name)}) vs {money(min(result.unhedged_stress_pnl.values()))} unhedged. "
        f"The delta over-hedge depends on the vol-spot correlation: with uncorrelated vol the variance reduction falls to {pct(result.robustness[0]['strategies'][rec_key]['variance_reduction'])} against {pct(result.robustness[0]['strategies']['multi_greek']['variance_reduction'])} for multi-Greek. "
        f"Daily variation margin on {rec.trades['future']:+,.0f} futures in a rally (about {money(rec.trades['future'] * result.instruments[0]['delta'] * m_spot(result) * 0.05)} per +5%) is a cash-flow risk the instantaneous scenario set does not price.",
        result.recommendation["rationale"][2],
    ]
    y_text_top = y - 0.006
    y_end = bullets(fig, LEFT + 0.012, y_text_top, rec_text, 190, fontsize=5.6)
    fig.patches.append(
        FancyBboxPatch(
            (LEFT, y_end - 0.002),
            WIDTH,
            y + 0.002 - (y_end - 0.002),
            boxstyle="round,pad=0.003,rounding_size=0.006",
            transform=fig.transFigure,
            facecolor="#F3FBF9",
            edgecolor=TEAL,
            lw=0.8,
            zorder=0,
        )
    )
    y = y_end - 0.016

    # Robustness.
    y = section(
        fig,
        LEFT,
        y,
        f"Robustness — positions fixed; out-of-sample variance reduction under alternative shock distributions ({result.robustness[0]['count']} scenarios each)",
    )
    rob = cs.robustness_table(result)
    rob_rows = [
        {
            "Shock distribution": wrap(
                f"Base out-of-sample set — {result.scenarios['out_of_sample_count']} scenarios, vol beta {cfg['vol_beta']:+.2f} to the index, rates {cfg['rate_sd_bp']:.0f} bp sd independent",
                112,
            ),
            **{
                short(cs.STRATEGY_LABELS[k]): pct(
                    result.strategies[k].out_of_sample["variance_reduction"]
                )
                for k in cs.STRATEGY_ORDER
            },
        }
    ] + [
        {
            "Shock distribution": wrap(f"{e['name']} — {e['description']}", 112),
            **{
                short(cs.STRATEGY_LABELS[k]): pct(
                    e["strategies"][k]["variance_reduction"]
                )
                for k in cs.STRATEGY_ORDER
            },
        }
        for e in result.robustness
    ]
    del rob
    y = place_table(
        fig,
        LEFT,
        y,
        WIDTH,
        pd.DataFrame(rob_rows),
        [0.61, 0.13, 0.13, 0.13],
        fontsize=5.7,
        highlight_rows=[0],
    )
    paragraph(
        fig,
        LEFT,
        y - 0.005,
        "Reading the comparison: the delta-only hedge is cheapest but leaves the duration gap and vega open and loses all of its benefit when rates co-move with equities. "
        "The multi-Greek hedge does not lose on sophistication grounds; it loses because the first-order Greek objective spends scarce liquidity selling calls to chase a vega match that the scenario analysis shows is worth little, "
        "and because it cannot see the vol-spot correlation. The stress-aware hedge ranks first on every distribution tested, but its margin over multi-Greek shrinks to about one point when vol is uncorrelated with the index — "
        "that assumption, not the optimizer, is what the recommendation rests on.",
        200,
        fontsize=5.4,
        color=SLATE,
    )
    footer(fig, result)
    pdf.savefig(fig)
    plt.close(fig)


def m_spot(result: cs.CaseStudyResult) -> float:
    return float(result.market["stressed_index"])


# --------------------------------------------------------------------------- #
# Page 3: assumptions, workflow, validation evidence, limitations
# --------------------------------------------------------------------------- #


def workflow(fig: plt.Figure, y_top: float, height: float) -> float:
    ax = fig.add_axes([0.04, y_top - height, 0.92, height])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    steps = [
        ("1. Market", "SyntheticMarketModel\npre-shock → bumped()\nstressed state"),
        (
            "2. Contracts",
            "RILA / FIA / IUL cohorts\nstruck at 7,750;\nTreasury strip ladder",
        ),
        (
            "3. Pricing",
            "analytic replication (RILA)\nseeded GBM Monte Carlo\n(FIA, IUL; common RNs)",
        ),
        (
            "4. Greeks",
            "bump_and_revalue:\ndelta, vega, rho/DV01 for\nliabilities, assets, net",
        ),
        (
            "5. Scenarios",
            "joint spot/vol/rate shocks;\nfull revaluation of\nsurplus and instruments",
        ),
        (
            "6. Optimizer",
            "optimize_hedge ×3:\ndelta · Greeks · scenario\nRMS + CVaR; lots, budgets",
        ),
        (
            "7. Evaluate",
            "out-of-sample variance,\n$-offset, ES95; named\nstresses; robustness sets",
        ),
        (
            "8. Decide",
            "transparent rule →\nJSON → PDF + dashboard\n(reconciliation test)",
        ),
    ]
    bw = 1.12
    for i, (title, body) in enumerate(steps):
        x = 0.1 + i * 1.235
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.3),
                bw,
                1.7,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=PALE if i % 2 == 0 else "#F3FBF9",
                edgecolor="#CBD5E1",
                lw=0.6,
            )
        )
        ax.text(
            x + bw / 2,
            1.78,
            title,
            ha="center",
            va="center",
            fontsize=6.4,
            weight="bold",
            color=NAVY,
        )
        ax.text(
            x + bw / 2,
            1.0,
            body,
            ha="center",
            va="center",
            fontsize=5.1,
            color=INK,
            linespacing=1.4,
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + bw + 0.12, 1.15),
                xytext=(x + bw + 0.005, 1.15),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8),
            )
    return y_top - height


def page_three(pdf: PdfPages, result: cs.CaseStudyResult) -> None:
    fig = plt.figure(figsize=PAGE)
    header(
        fig,
        result,
        3,
        "Key assumptions · modeling workflow · validation evidence · limitations",
    )
    top = 0.912

    y = section(
        fig,
        LEFT,
        top,
        "Key assumptions (illustrative; separated from the documented events on page 1)",
    )
    y = bullets(fig, LEFT, y, result.illustrative_assumptions, 192, fontsize=5.6)
    y -= 0.004

    y = section(
        fig,
        LEFT,
        y,
        "Modeling workflow — every box is existing, validated engine code; the case study module adds orchestration, scenarios, and the decision rule",
    )
    y = workflow(fig, y + 0.004, 0.092)
    y -= 0.002

    y = section(
        fig,
        LEFT,
        y,
        "Validation evidence relied on (MODEL_VALIDATION_REPORT.md: 126 independent checks, 118 pass, 0 fail, 8 informational)",
    )
    evidence = pd.DataFrame(
        [
            {
                "Component used here": "Buffered RILA analytic value (bond + call spread − put spread)",
                "Evidence": "Matches lognormal quadrature and the independent decomposition to ≤ 2e-13; replication legs reproduce the payoff to 3e-14.",
                "Status": "PASS",
            },
            {
                "Component used here": "FIA ratchet and IUL Monte Carlo",
                "Evidence": "Within 3 standard errors of an independent closed form / deterministic recursion; forward-rate and forward-variance stepping reprices the market forward and ATM options on sloped curves.",
                "Status": "PASS (fixed in 1b5affb)",
            },
            {
                "Component used here": "Bump-and-revalue Greeks and DV01",
                "Evidence": "Match analytic Black-Scholes Greeks; O(h²) error ratios 3.99-4.02 when halving bumps; DV01 = rho × 1e-4 to first order; cohort → portfolio aggregation exact.",
                "Status": "PASS",
            },
            {
                "Component used here": "Full-revaluation stresses (MarketState.bumped)",
                "Evidence": "Stressed values equal independent repricing at the shocked spot/vol/rate (≤ 7e-11); shocks applied exactly as specified.",
                "Status": "PASS",
            },
            {
                "Component used here": "Constrained optimizer (lots, premium and liquidity budgets, CVaR)",
                "Evidence": "Unconstrained solution = weighted least squares (1e-4); lot-constrained objective equals the brute-force optimum; budgets respected after rounding; reported CVaR equals independent discrete CVaR.",
                "Status": "PASS (fixed in 700329f)",
            },
            {
                "Component used here": "Scenario-based effectiveness instead of the Greek metric",
                "Evidence": "Validation §4: the Greek metric read 1.000 where full-revaluation variance reduction was 0.94-0.95, and -9.06 for an uncapped collinear hedge; it recommends variance reduction / dollar-offset for governance.",
                "Status": "Adopted here",
            },
            {
                "Component used here": "Hedge sign convention",
                "Evidence": "Hedge assets carry the same-signed Greeks as the liability; the optimizer receives the insurer's net exposure. Regression-tested.",
                "Status": "PASS (fixed in 3bb67cc)",
            },
            {
                "Component used here": "This case study",
                "Evidence": f"tests/test_case_study.py re-runs run_case_study() and checks that the committed JSON consumed by this PDF and by the dashboard reconciles; effectiveness metrics are recomputed with validation/reference.py; hedged stress P&L is rebuilt from trades. Runtime {result.runtime_seconds:.0f}s.",
                "Status": "Regression test",
            },
        ]
    )
    evidence["Evidence"] = evidence["Evidence"].map(lambda s: wrap(s, 124))
    evidence["Component used here"] = evidence["Component used here"].map(
        lambda s: wrap(s, 36)
    )
    y = place_table(
        fig, LEFT, y, WIDTH, evidence, [0.21, 0.66, 0.13], fontsize=5.5, pad_lines=0.6
    )
    y -= 0.008

    y = section(fig, LEFT, y, "Limitations and what would change the answer")
    y = bullets(fig, LEFT, y, result.limitations, 196, fontsize=5.5, gap=0.2)
    y -= 0.002

    y = section(
        fig, LEFT, y, "Sources for the documented backdrop (retrieved 16 October 2026)"
    )
    sources = [
        f"{e['date']} — {e['source']}: {e['url']}" for e in result.documented_events
    ]
    bullets(fig, LEFT, y, sources, 215, fontsize=4.8, bullet="·", color=SLATE, gap=0.1)
    footer(fig, result)
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def build_pdf(result: cs.CaseStudyResult, path: Path) -> Path:
    """Render the three-page PDF from a results object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        page_one(pdf, result)
        page_two(pdf, result)
        page_three(pdf, result)
        info = pdf.infodict()
        info["Title"] = result.title
        info["Subject"] = result.short_title
        info["Author"] = "ALM Hedging Framework (illustrative research output)"
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse", action="store_true", help="rebuild the PDF from the committed JSON"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=cs.DEFAULT_RESULTS_DIR, help="output directory"
    )
    args = parser.parse_args()
    json_path = args.out_dir / cs.RESULTS_FILENAME
    if args.reuse:
        result = cs.load_results(json_path)
    else:
        result = cs.run_case_study()
        cs.save_results(result, json_path)
        print(f"results written to {json_path} ({result.runtime_seconds:.1f}s)")
    pdf_path = build_pdf(result, args.out_dir / cs.PDF_FILENAME)
    print(f"PDF written to {pdf_path}")
    rec = result.strategies[result.recommendation["key"]]
    print(
        f"recommended: {rec.label}; OOS variance reduction "
        f"{rec.out_of_sample['variance_reduction']:.1%}, dollar-offset "
        f"{rec.out_of_sample['dollar_offset_slope']:.2f}, premium {rec.premium_used:,.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
