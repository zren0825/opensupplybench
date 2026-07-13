"""Policy interface.

A policy sees an Observation each day and returns a desired order quantity
(a float; the simulator applies MOQ / case-pack / budget rounding). Keeping
the interface this thin means non-LLM baselines and the Phase-4 hybrid agent
are interchangeable inside the simulator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from opensupply.simulator import Observation


class Policy(ABC):
    name: str = "policy"

    def reset(self) -> None:
        """Called once at the start of each simulation run."""

    @abstractmethod
    def decide(self, obs: Observation) -> float:
        """Return desired order quantity for `obs.day` (0 = do not order)."""
        raise NotImplementedError
