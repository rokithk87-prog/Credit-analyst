"""
utils/calculations.py
──────────────────────
Pure-function math helpers shared across modules.
No side-effects, no I/O — just arithmetic.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np

from utils.validators import safe_divide


def growth_rate(current: float, prior: float) -> float | None:
    """Year-over-year growth rate as a decimal (e.g. 0.15 → 15%).
    Returns None when prior is zero or missing."""
    if not prior or prior == 0:
        return None
    return (current - prior) / abs(prior)


def cagr(start: float, end: float, periods: int) -> float | None:
    """Compound Annual Growth Rate over *periods* years.
    Returns None for invalid inputs."""
    if start <= 0 or end < 0 or periods <= 0:
        return None
    return (end / start) ** (1 / periods) - 1


def moving_average(values: Sequence[float], window: int = 3) -> list[float | None]:
    """Simple moving average; pads leading positions with None."""
    result: list[float | None] = []
    for i, _ in enumerate(values):
        if i < window - 1:
            result.append(None)
        else:
            window_vals = [v for v in values[i - window + 1: i + 1]
                           if v is not None]
            result.append(np.mean(window_vals) if window_vals else None)
    return result


def percentage_of(part: float, total: float) -> float:
    """Return part / total as a percentage (0–100 scale)."""
    return safe_divide(part, total, 0.0) * 100


def normalise_to_millions(value: float, unit: str = "auto") -> tuple[float, str]:
    """Scale a raw financial value to a readable unit.

    Returns (scaled_value, unit_label).
    unit='auto' infers the best scale; otherwise pass 'thousands',
    'millions', or 'billions'.
    """
    abs_val = abs(value)
    if unit == "auto":
        if abs_val >= 1_000_000_000:
            unit = "billions"
        elif abs_val >= 1_000_000:
            unit = "millions"
        elif abs_val >= 1_000:
            unit = "thousands"
        else:
            unit = "units"

    divisors = {"billions": 1e9, "millions": 1e6, "thousands": 1e3, "units": 1}
    labels   = {"billions": "B",  "millions": "M",  "thousands": "K", "units": ""}
    divisor  = divisors.get(unit, 1)
    label    = labels.get(unit, "")
    return value / divisor, label


def score_from_benchmark(value: float,
                          good: float,
                          acceptable: float,
                          poor: float,
                          higher_is_better: bool = True) -> int:
    """Map a ratio value to a 0–100 credit-quality score.

    Uses the three benchmark thresholds from settings.RATIO_BENCHMARKS.
    higher_is_better=False inverts the comparison (e.g. leverage ratios).
    """
    if higher_is_better:
        if value >= good:        return 100
        if value >= acceptable:  return 70
        if value >= poor:        return 40
        return 10
    else:   # lower is better
        if value <= good:        return 100
        if value <= acceptable:  return 70
        if value <= poor:        return 40
        return 10
