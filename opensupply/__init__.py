"""OpenSupplyBench: a scenario-based benchmark for small-business
inventory replenishment agents.

Public surface kept intentionally small; import submodules directly for
the rest (opensupply.policies, opensupply.evaluation, ...).
"""

from opensupply.scenario import Scenario
from opensupply.simulator import Simulator, SimulationResult
from opensupply.demand import DemandGenerator, DEMAND_TYPES
from opensupply.leadtime import LeadTime, LEAD_TIME_TYPES
from opensupply.forecast import forecast_tool, Forecast

__all__ = [
    "Scenario",
    "Simulator",
    "SimulationResult",
    "DemandGenerator",
    "DEMAND_TYPES",
    "LeadTime",
    "LEAD_TIME_TYPES",
    "forecast_tool",
    "Forecast",
]

__version__ = "0.1.0"
