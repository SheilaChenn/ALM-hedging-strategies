"""Market, crash, curve-twist, and policyholder behavior stress testing."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from .contracts import Cohort, IULAccountValue, LiabilityContract
from .market import MarketState


@dataclass(frozen=True)
class StressScenario:
    """Combined market and behavior shock."""

    name: str
    equity_relative: float = 0.0
    vol_absolute: float = 0.0
    rate_bp: float = 0.0
    twist_bp: float = 0.0
    mortality_multiplier: float = 1.0
    lapse_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.equity_relative <= -1.0:
            raise ValueError("equity shock must leave spot positive")
        if self.mortality_multiplier < 0.0 or self.lapse_multiplier < 0.0:
            raise ValueError("behavior multipliers must be non-negative")

    def apply_market(self, market: MarketState) -> MarketState:
        """Apply all market shocks."""
        return market.bumped(
            spot_relative=self.equity_relative,
            vol_absolute=self.vol_absolute,
            rate_bp=self.rate_bp,
            twist_bp=self.twist_bp,
        )

    def apply_liability(
        self, liability: LiabilityContract | Cohort
    ) -> LiabilityContract | Cohort:
        """Apply mortality and lapse shocks without mutating the contract."""
        is_cohort = isinstance(liability, Cohort)
        contract = liability.contract if is_cohort else liability
        if isinstance(contract, IULAccountValue):
            shocked = replace(
                contract,
                mortality_rates=tuple(
                    min(rate * self.mortality_multiplier, 1.0)
                    for rate in contract.mortality_rates
                ),
                lapse_rates=tuple(
                    min(rate * self.lapse_multiplier, 1.0)
                    for rate in contract.lapse_rates
                ),
            )
            return replace(liability, contract=shocked) if is_cohort else shocked
        if is_cohort:
            return replace(
                liability,
                mortality_multiplier=liability.mortality_multiplier
                * self.mortality_multiplier,
                lapse_multiplier=liability.lapse_multiplier * self.lapse_multiplier,
            )
        return liability


@dataclass(frozen=True)
class StressResult:
    """Valuation outcome for one scenario."""

    scenario: str
    base_value: float
    stressed_value: float

    @property
    def pnl(self) -> float:
        """Stressed value less base value."""
        return self.stressed_value - self.base_value

    @property
    def relative_change(self) -> float:
        """Relative value change, or zero for a zero base value."""
        return self.pnl / self.base_value if self.base_value != 0.0 else 0.0


def standard_stress_scenarios() -> tuple[StressScenario, ...]:
    """Return a compact deterministic regulatory-style stress suite."""
    return (
        StressScenario("equity_down_20", equity_relative=-0.20),
        StressScenario("equity_up_20", equity_relative=0.20),
        StressScenario("vol_up_10pt", vol_absolute=0.10),
        StressScenario("rates_up_100bp", rate_bp=100.0),
        StressScenario("rates_down_100bp", rate_bp=-100.0),
        StressScenario("curve_steepener", twist_bp=100.0),
        StressScenario("curve_flattener", twist_bp=-100.0),
        StressScenario(
            "equity_crash",
            equity_relative=-0.40,
            vol_absolute=0.25,
            rate_bp=-75.0,
        ),
        StressScenario(
            "adverse_behavior", mortality_multiplier=1.20, lapse_multiplier=1.50
        ),
    )


def run_stress_scenarios(
    market: MarketState,
    scenarios: Sequence[StressScenario],
    value_function: Callable[
        [MarketState, Sequence[LiabilityContract | Cohort]], float
    ],
    liabilities: Sequence[LiabilityContract | Cohort],
) -> tuple[StressResult, ...]:
    """Run stresses using a caller-supplied portfolio valuation function."""
    base_value = float(value_function(market, liabilities))
    results = []
    for scenario in scenarios:
        stressed_liabilities = tuple(
            scenario.apply_liability(item) for item in liabilities
        )
        stressed_value = float(
            value_function(scenario.apply_market(market), stressed_liabilities)
        )
        results.append(StressResult(scenario.name, base_value, stressed_value))
    return tuple(results)


def aggregate_stress_results(
    portfolios: Mapping[str, Sequence[StressResult]],
) -> dict[str, StressResult]:
    """Aggregate scenario results across portfolios by scenario name."""
    totals: dict[str, list[float]] = {}
    for results in portfolios.values():
        for result in results:
            values = totals.setdefault(result.scenario, [0.0, 0.0])
            values[0] += result.base_value
            values[1] += result.stressed_value
    return {
        name: StressResult(name, values[0], values[1])
        for name, values in totals.items()
    }
