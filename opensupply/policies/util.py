"""Shared helpers for safety-stock policies."""

from __future__ import annotations

from statistics import NormalDist


def z_for(service_level: float) -> float:
    """Safety factor z = Φ⁻¹(service_level). Uses the stdlib normal quantile
    (no scipy dependency). Clamped to a sensible range."""
    sl = min(max(service_level, 0.5), 0.999)
    return NormalDist().inv_cdf(sl)
