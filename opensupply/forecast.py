"""Minimal forecasting baselines (Phase 1.2).

Deliberately simple: moving average, exponential smoothing, seasonal naive.
The paper's claim is NOT "we forecast better" — the forecast is a *tool* the
policies and the hybrid agent call. It returns a point estimate plus an
uncertainty band so downstream code can size safety stock.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Forecast:
    mean_demand: float      # expected demand per day over the horizon
    low_demand: float       # ~10th percentile per-day estimate
    high_demand: float      # ~90th percentile per-day estimate
    uncertainty: float      # std of per-day demand used for safety stock

    def over(self, days: float) -> float:
        """Expected total demand over `days` days."""
        return self.mean_demand * days


def _moving_average(history: np.ndarray, window: int) -> float:
    if len(history) == 0:
        return 0.0
    w = min(window, len(history))
    return float(np.mean(history[-w:]))


def _exp_smoothing(history: np.ndarray, alpha: float = 0.3) -> float:
    if len(history) == 0:
        return 0.0
    level = float(history[0])
    for x in history[1:]:
        level = alpha * float(x) + (1 - alpha) * level
    return level


def _seasonal_naive(history: np.ndarray, period: int = 7) -> float:
    if len(history) < period:
        return _moving_average(history, len(history))
    return float(np.mean(history[-period:]))


def forecast_tool(
    sales_history,
    horizon_days: int,
    method: str = "exp_smoothing",
    window: int = 14,
) -> Forecast:
    """forecast_tool(sales_history, horizon_days) -> Forecast.

    `sales_history` is the observed (uncensored where possible) demand series.
    Returns mean/low/high/uncertainty per-day estimates. `horizon_days` is
    accepted for API completeness / future multi-step methods.
    """
    history = np.asarray(list(sales_history), dtype=float)
    if len(history) == 0:
        return Forecast(0.0, 0.0, 0.0, 0.0)

    if method == "moving_average":
        mean = _moving_average(history, window)
    elif method == "exp_smoothing":
        mean = _exp_smoothing(history)
    elif method == "seasonal_naive":
        mean = _seasonal_naive(history)
    else:
        raise ValueError(f"unknown forecast method {method!r}")

    recent = history[-window:] if len(history) >= 2 else history
    sigma = float(np.std(recent)) if len(recent) > 1 else max(1.0, 0.2 * mean)
    # 10th/90th ~ +/- 1.28 sigma under a normal approximation
    low = max(0.0, mean - 1.28 * sigma)
    high = mean + 1.28 * sigma
    return Forecast(mean_demand=mean, low_demand=low, high_demand=high, uncertainty=sigma)
