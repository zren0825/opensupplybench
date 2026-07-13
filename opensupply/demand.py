"""Synthetic demand generation (Phase 2.2, hardened in v0.2).

Demand is **count data** drawn from a Negative-Binomial distribution around a
time-varying mean path. NB→Poisson as the dispersion parameter r→∞, so a single
`dispersion` knob spans the Poisson-to-overdispersed regimes retail exhibits
(`CV² = 1/µ + 1/r`). The seven demand types are aligned with the LLM scenario
classifier's taxonomy. Every trajectory is fully determined by
(demand_type, dispersion, base_level, seed).
"""

from __future__ import annotations

import numpy as np

DEMAND_TYPES = [
    "stable",
    "seasonal",
    "promotion_spike",
    "viral_spike",
    "declining",
    "new_product",
    "intermittent",
]

# Negative-Binomial "size" r: larger r = closer to Poisson (less overdispersion).
DISPERSION_R = {"low": 25.0, "medium": 8.0, "high": 3.0}


class DemandGenerator:
    """Generate reproducible Negative-Binomial daily demand trajectories."""

    def __init__(self, base_level: float = 20.0):
        self.base_level = float(base_level)

    def generate(
        self,
        demand_type: str,
        horizon_days: int,
        seed: int,
        dispersion: str = "medium",
        base_level: float | None = None,
    ) -> np.ndarray:
        if demand_type not in DEMAND_TYPES:
            raise ValueError(
                f"unknown demand_type {demand_type!r}; expected one of {DEMAND_TYPES}"
            )
        if dispersion not in DISPERSION_R:
            raise ValueError(f"unknown dispersion {dispersion!r}")
        b = float(base_level) if base_level is not None else self.base_level
        rng = np.random.default_rng(seed)

        mu = self._mean_path(demand_type, horizon_days, b, rng)
        r = DISPERSION_R[dispersion]
        draws = self._nb(rng, mu, r)

        if demand_type == "intermittent":
            # zero-inflation on top of NB counts: many no-sale days, lumpy bursts
            active = rng.random(horizon_days) > 0.6
            draws = draws * active

        return np.clip(draws, 0, None).astype(int)

    # --- Negative-Binomial draw around a mean path -------------------------

    @staticmethod
    def _nb(rng, mu: np.ndarray, r: float) -> np.ndarray:
        mu = np.maximum(mu, 1e-6)
        p = r / (r + mu)  # numpy parameterization: mean = r*(1-p)/p = mu
        return rng.negative_binomial(r, p).astype(float)

    # --- mean paths (µ_t); the count draw adds the noise -------------------

    def _mean_path(self, demand_type, n, b, rng) -> np.ndarray:
        t = np.arange(n)
        fn = getattr(self, f"_mp_{demand_type}")
        return np.maximum(fn(n, t, b, rng), 0.0)

    def _mp_stable(self, n, t, b, rng):
        return np.full(n, b)

    def _mp_seasonal(self, n, t, b, rng):
        weekly = 1.0 + 0.35 * np.sin(2 * np.pi * t / 7.0)
        return b * weekly

    def _mp_promotion_spike(self, n, t, b, rng):
        mu = np.full(n, b)
        start = rng.integers(int(0.3 * n), int(0.6 * n))
        dur = rng.integers(3, 8)
        mu[start : start + dur] *= rng.uniform(2.5, 4.0)
        return mu

    def _mp_viral_spike(self, n, t, b, rng):
        mu = np.full(n, b)
        start = rng.integers(int(0.2 * n), int(0.6 * n))
        peak = rng.uniform(5.0, 10.0) * b
        decay = rng.uniform(0.6, 0.8)
        length = min(n - start, 20)
        mu[start : start + length] += peak * (decay ** np.arange(length))
        return mu

    def _mp_declining(self, n, t, b, rng):
        # gradual exponential decay from b to ~0.2*b over the horizon
        k = np.log(5.0) / max(1, n - 1)
        return b * np.exp(-k * t)

    def _mp_new_product(self, n, t, b, rng):
        # logistic growth ramp from near-zero to ~1.3*b
        cap = 1.3 * b
        t0 = rng.uniform(0.2 * n, 0.4 * n)
        g = 8.0 / n
        return cap / (1.0 + np.exp(-g * (t - t0)))

    def _mp_intermittent(self, n, t, b, rng):
        # low baseline mean; zero-inflation applied by generate()
        return np.full(n, max(1.0, 0.5 * b))
