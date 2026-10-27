"""Render the one-page simple-model report to ``simple/simple-model-report.pdf``.

Usage::

    python simple/build_report.py [--output PATH]

Every number comes from one ``run_simple_model()`` call, the same function the
notebook uses, so the two reconcile exactly. Illustrative model, synthetic inputs.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.table import Table  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import simple_model as sm  # noqa: E402

INK = "#14213D"
SLATE = "#68778D"
PALE = "#EAF0F8"
COLORS = {"RILA": "#2F6BFF", "FIA": "#12B8A6", "IUL": "#F3B61F"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.edgecolor": "#CBD5E1",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.parse_math": False,
    }
)


def _style_table(table: Table, header_rows: int = 1) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(6.6)
    table.scale(1.0, 1.25)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#D6DEE8")
        cell.set_linewidth(0.5)
        if row < header_rows:
            cell.set_facecolor(PALE)
            cell.set_text_props(weight="bold")


def _payoff_panel(ax: Axes, results: sm.SimpleModelResults) -> None:
    grid = [100.0 * (x - 1.0) for x in results.payoff_grid]
    ax.plot(grid, grid, color="#BBBBBB", lw=0.8, ls="--", label="Index return")
    for name, credited in results.payoffs.items():
        ax.plot(
            grid, [100.0 * c for c in credited], color=COLORS[name], lw=1.6, label=name
        )
    ax.axhline(0.0, color="#CBD5E1", lw=0.6)
    ax.set_xlabel("Index return at maturity (%)")
    ax.set_ylabel("Credited return (%)")
    ax.set_title("Payoff at one year per $100 of account value")
    ax.legend(frameon=False, fontsize=6.4, loc="upper left")
    ax.grid(alpha=0.25)


def _pricing_panel(ax: Axes, results: sm.SimpleModelResults) -> None:
    ax.axis("off")
    ax.set_title("Pricing and Greeks (per $100 account value)", loc="left")
    headers = ["", "RILA", "FIA", "IUL", "Portfolio"]
    greeks = [sm.scaled_greeks(v.greeks) for v in results.valuations]
    total = sm.scaled_greeks(results.portfolio)
    rows = [
        [label, *[f"{g[label]:.4f}" for g in greeks], f"{total[label]:.4f}"]
        for label in total
    ]
    rows.append(
        [
            "Cap",
            *[
                f"{p.legs[1].strike / p.inception_spot - 1:.1%}"
                for p in results.products
            ],
            "",
        ]
    )
    rows.append(
        [
            f"Cap for ${results.option_budget:.2f} budget",
            *[f"{results.solved_caps[p.name]:.2%}" for p in results.products],
            "",
        ]
    )
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="upper center",
        cellLoc="right",
        colWidths=[0.30, 0.17, 0.17, 0.17, 0.19],
    )
    _style_table(table)
    for (_row, col), cell in table.get_celld().items():
        if col == 0:
            cell.set_text_props(ha="left")


def _stress_panel(ax: Axes, results: sm.SimpleModelResults) -> None:
    ax.axis("off")
    ax.set_title(
        "Full-revaluation stress P&L, insurer view ($, negative = loss)", loc="left"
    )
    headers = ["Scenario", "Liability PV", "Unhedged", "Futures", "Hedged"]
    rows = [
        [
            r.scenario,
            f"{r.liability_pv:.2f}",
            f"{r.unhedged_pnl:+.2f}",
            f"{r.hedge_pnl:+.2f}",
            f"{r.hedged_pnl:+.2f}",
        ]
        for r in results.stress
    ]
    rows.append(
        [
            "Range (max - min)",
            "",
            f"{results.unhedged_range:.2f}",
            "",
            f"{results.hedged_range:.2f}",
        ]
    )
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="upper center",
        cellLoc="right",
        colWidths=[0.30, 0.18, 0.17, 0.17, 0.18],
    )
    _style_table(table)
    for (row, col), cell in table.get_celld().items():
        if col == 0:
            cell.set_text_props(ha="left")
        if row == len(rows):
            cell.set_text_props(weight="bold")


def _text_panel(ax: Axes, title: str, body: str, width: int = 150) -> None:
    ax.axis("off")
    ax.set_title(title, loc="left")
    ax.text(
        0.0,
        0.98,
        textwrap.fill(body, width),
        va="top",
        ha="left",
        fontsize=6.9,
        wrap=True,
        linespacing=1.35,
    )


def build_report(results: sm.SimpleModelResults, output: Path) -> Path:
    m = results.market
    fig = plt.figure(figsize=(8.5, 11.0))
    fig.text(
        0.06,
        0.965,
        "Simple ALM hedging model: RILA, FIA, and IUL embedded options",
        fontsize=13,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.948,
        "Illustrative model with synthetic inputs. One year, one index, closed-form Black-Scholes, "
        "$100 of account value per product.\n"
        f"Synthetic market: index {m.spot:.0f}, 1y rate {m.rate:.1%}, dividend yield "
        f"{m.dividend:.1%}, implied vol {m.vol:.0%}, option budget ${results.option_budget:.2f} per $100.",
        fontsize=7.2,
        color=SLATE,
        va="top",
        linespacing=1.4,
    )
    gs = fig.add_gridspec(
        4,
        2,
        left=0.06,
        right=0.95,
        top=0.895,
        bottom=0.07,
        hspace=0.42,
        wspace=0.25,
        height_ratios=[1.45, 1.0, 1.0, 0.7],
    )
    _payoff_panel(fig.add_subplot(gs[0, 0]), results)
    _text_panel(
        fig.add_subplot(gs[0, 1]),
        "How to read the products",
        " ".join(f"{p.name}: {p.description}." for p in results.products)
        + " The policyholder is long these packages, so the insurer carries them as liabilities and must fund them "
        "from the general-account spread (the option budget). The RILA's short buffer put is what lets its cap sit "
        "well above the FIA and IUL caps for a similar cost.",
        width=62,
    )
    _pricing_panel(fig.add_subplot(gs[1, :]), results)
    _stress_panel(fig.add_subplot(gs[2, :]), results)
    _text_panel(
        fig.add_subplot(gs[3, :]),
        "Hedge recommendation and residual risks",
        sm.recommendation(results),
    )
    fig.text(
        0.06,
        0.02,
        "Generated by simple/build_report.py from run_simple_model(); the notebook simple/simple_model.ipynb shows the same numbers.",
        fontsize=6.2,
        color=SLATE,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "simple-model-report.pdf",
    )
    args = parser.parse_args(argv)
    path = build_report(sm.run_simple_model(), args.output)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
