"""Supplier lead-time models (Phase 2.3).

A LeadTime turns an order placed on `today` into an arrival day. The three
types capture the small-business reality that "I ordered it but it showed up
late, so I stocked out" — the exact failure mode a good policy must hedge.
"""

from __future__ import annotations

import numpy as np

LEAD_TIME_TYPES = ["fixed", "random", "delayed"]


class LeadTime:
    """Sample supplier lead times.

    Uses its own RNG seeded off the scenario seed so lead-time noise is
    reproducible and independent from demand noise.
    """

    def __init__(self, lead_time_type: str = "fixed", mean_lead: int = 5, seed: int = 0):
        if lead_time_type not in LEAD_TIME_TYPES:
            raise ValueError(
                f"unknown lead_time_type {lead_time_type!r}; expected {LEAD_TIME_TYPES}"
            )
        self.type = lead_time_type
        self.mean_lead = int(mean_lead)
        self._rng = np.random.default_rng(seed + 987_654)

    def sample(self) -> int:
        if self.type == "fixed":
            return self.mean_lead
        if self.type == "random":
            # symmetric jitter around the mean, floored at 1 day
            lead = self._rng.normal(self.mean_lead, max(1.0, 0.4 * self.mean_lead))
            return int(max(1, round(lead)))
        if self.type == "delayed":
            # usually on time, but a fat right tail of late deliveries
            if self._rng.random() < 0.35:
                lead = self.mean_lead + self._rng.integers(3, 10)
            else:
                lead = self.mean_lead
            return int(max(1, lead))
        raise AssertionError("unreachable")

    @property
    def expected_lead(self) -> float:
        """Point estimate a policy may use for planning."""
        if self.type == "delayed":
            return self.mean_lead + 0.35 * 6.0
        return float(self.mean_lead)
