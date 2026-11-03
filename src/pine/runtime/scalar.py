"""Pine scalar semantics (PINE_PLAN.md §11 E1, §5.10).

Each helper implements the **expected** hypothesis of the corresponding probe in
``tools/capture_semantics.pine``; all are **unverified** until a capture export decides
(``python -m pine.semantics``). `na` is NaN.
"""

from __future__ import annotations

import math

__all__ = ["NA", "na", "nz", "round_", "mod", "div", "to_int", "max_", "min_", "eq"]

NA = math.nan


def _f(x) -> float:
    return NA if x is None else float(x)


def na(x) -> bool:
    """``na(x)``."""
    return math.isnan(_f(x))


def nz(x, replacement: float = 0.0) -> float:
    """``nz(x, replacement)``."""
    v = _f(x)
    return replacement if math.isnan(v) else v


def round_(x, precision: int | None = None) -> float:
    """``math.round``; expected: ties round up, ``floor(x + 0.5)`` (Java-like), so
    ``round(-2.5) = -2``. Alternatives probed: half away from zero, half to even."""
    v = _f(x)
    if math.isnan(v):
        return NA
    if precision is None:
        return float(math.floor(v + 0.5))
    k = 10.0**precision
    return math.floor(v * k + 0.5) / k


def mod(a, b) -> float:
    """``a % b``; expected: truncated remainder (sign of ``a``), ``-7 % 3 = -1``."""
    x, y = _f(a), _f(b)
    if math.isnan(x) or math.isnan(y) or y == 0:
        return NA
    return math.fmod(x, y)


def div(a, b) -> float:
    """``a / b``; expected: fractional result even for two ints (``7 / 2 = 3.5``); `na` on
    division by zero."""
    x, y = _f(a), _f(b)
    if math.isnan(x) or math.isnan(y) or y == 0:
        return NA
    return x / y


def to_int(x) -> float:
    """``int(x)``; expected: truncation toward zero, ``int(-2.7) = -2``."""
    v = _f(x)
    return NA if math.isnan(v) else float(math.trunc(v))


def max_(*xs) -> float:
    """``math.max``: `na` if any argument is `na`."""
    vals = [_f(x) for x in xs]
    return NA if any(math.isnan(v) for v in vals) else max(vals)


def min_(*xs) -> float:
    """``math.min``: `na` if any argument is `na`."""
    vals = [_f(x) for x in xs]
    return NA if any(math.isnan(v) for v in vals) else min(vals)


def eq(a, b) -> bool:
    """``a == b``; expected: false whenever either side is `na` (so ``na == na`` is false)."""
    x, y = _f(a), _f(b)
    return not (math.isnan(x) or math.isnan(y)) and x == y
