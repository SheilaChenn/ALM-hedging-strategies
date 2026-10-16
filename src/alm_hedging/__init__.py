"""Asset-liability management toolkit for equity-linked insurance hedging.

The package exposes immutable market and contract models, analytic and Monte
Carlo valuation, risk/stress engines, hedge optimization, and dynamic simulation.
Rates and volatilities are decimals, curve shocks are basis points, and time is
measured in years.
"""

__version__ = "0.1.0"

from .api import (
    backtest_hedge,
    calculate_greeks,
    get_market_snapshot,
    price_portfolio,
    run_stress_test,
)
from .api import (
    optimize_hedge as recommend_hedge,
)
from .budget import (
    BudgetSolution,
    solve_budget_parameter,
    solve_call_spread_cap,
    solve_option_budget,
    solve_participation_rate,
)
from .contracts import (
    Cohort,
    FIAAnnualPointToPoint,
    IULAccountValue,
    LiabilityContract,
    RILABufferedSegment,
    RILAFlooredSegment,
)
from .market import (
    MarketState,
    SyntheticMarketModel,
    VolSurface,
    ZeroCurve,
    calibrate_synthetic_market,
)
from .optimization import (
    HedgeConstraints,
    HedgeInstrument,
    HedgeSolution,
    optimize_hedge,
)
from .pricing import (
    Pricer,
    ReplicationLeg,
    analytic_liability_value,
    black_scholes_price,
    market_option_price,
    monte_carlo_value,
    piecewise_linear_replication,
    portfolio_value,
    replication_value,
    simulate_gbm_paths,
)
from .risk import BumpSizes, Greeks, aggregate_greeks, bump_and_revalue
from .simulation import (
    DynamicHedgeInstrument,
    DynamicHedgeResult,
    simulate_dynamic_hedge,
)
from .stress import (
    StressResult,
    StressScenario,
    aggregate_stress_results,
    run_stress_scenarios,
    standard_stress_scenarios,
)

__all__ = [
    "__version__",
    "BudgetSolution",
    "BumpSizes",
    "Cohort",
    "DynamicHedgeInstrument",
    "DynamicHedgeResult",
    "FIAAnnualPointToPoint",
    "Greeks",
    "HedgeConstraints",
    "HedgeInstrument",
    "HedgeSolution",
    "IULAccountValue",
    "LiabilityContract",
    "MarketState",
    "Pricer",
    "RILABufferedSegment",
    "RILAFlooredSegment",
    "ReplicationLeg",
    "StressResult",
    "StressScenario",
    "SyntheticMarketModel",
    "VolSurface",
    "ZeroCurve",
    "aggregate_greeks",
    "aggregate_stress_results",
    "analytic_liability_value",
    "backtest_hedge",
    "black_scholes_price",
    "bump_and_revalue",
    "calculate_greeks",
    "calibrate_synthetic_market",
    "get_market_snapshot",
    "market_option_price",
    "monte_carlo_value",
    "optimize_hedge",
    "piecewise_linear_replication",
    "portfolio_value",
    "price_portfolio",
    "recommend_hedge",
    "replication_value",
    "run_stress_test",
    "run_stress_scenarios",
    "simulate_dynamic_hedge",
    "simulate_gbm_paths",
    "solve_budget_parameter",
    "solve_call_spread_cap",
    "solve_option_budget",
    "solve_participation_rate",
    "standard_stress_scenarios",
]
