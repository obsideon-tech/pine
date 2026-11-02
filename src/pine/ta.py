"""Pine `ta.*` built-ins with Pine's `na`, warm-up and seeding semantics.

Each function is a literal bar-by-bar port of the Pine reference implementation that the
TradingView reference manual gives for the built-in. **Unverified** (PINE_PLAN.md §5.10):
the manual's code is recalled, not yet cited, and the built-in may differ from the manual's
code. Until a TradingView export confirms each one, these are verified only against
closed-form cases and property tests (oracles (a) and (b) of §5.8).

`na` is represented as NaN. Pine arithmetic with `na` yields `na`, which IEEE NaN
propagation reproduces; comparisons and `math.max` are handled explicitly.

Loops are deliberate: they mirror Pine's execution model and summation order (§5.6).
Speed is not a goal yet.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["sma", "ema", "rma", "rsi"]


def _length(length: int) -> int:
    if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 1:
        raise ValueError(f"length must be a positive integer, got {length!r}")
    return int(length)


def _source(source: ArrayLike) -> np.ndarray:
    x = np.array(source, dtype=np.float64, copy=True)
    if x.ndim != 1:
        raise ValueError(f"source must be one-dimensional, got shape {x.shape}")
    return x


def sma(source: ArrayLike, length: int) -> np.ndarray:
    """``ta.sma``. Manual's code::

        pine_sma(x, y) =>
            sum = 0.0
            for i = 0 to y - 1
                sum := sum + x[i] / y
            sum

    Newest bar first, each term divided before adding. `na` for the first ``length - 1``
    bars and for any window containing `na`.
    """
    n_len = _length(length)
    x = _source(source)
    out = np.full(x.size, np.nan)
    for t in range(n_len - 1, x.size):
        s = 0.0
        for i in range(n_len):
            s += x[t - i] / n_len
        out[t] = s
    return out


def _seeded_recursive(x: np.ndarray, n_len: int, alpha: float) -> np.ndarray:
    """``sum := na(sum[1]) ? ta.sma(src, length) : alpha * src + (1 - alpha) * nz(sum[1])``.

    Re-seeds from the SMA whenever the previous value is `na`, as the Pine code does.
    """
    seed = sma(x, n_len)
    out = np.full(x.size, np.nan)
    prev = math.nan
    for t in range(x.size):
        cur = seed[t] if math.isnan(prev) else alpha * x[t] + (1.0 - alpha) * prev
        out[t] = cur
        prev = cur
    return out


def ema(source: ArrayLike, length: int) -> np.ndarray:
    """``ta.ema``: ``alpha = 2 / (length + 1)``, seeded with ``ta.sma(src, length)``."""
    n_len = _length(length)
    return _seeded_recursive(_source(source), n_len, 2.0 / (n_len + 1))


def rma(source: ArrayLike, length: int) -> np.ndarray:
    """``ta.rma`` (Wilder): ``alpha = 1 / length``, seeded with ``ta.sma(src, length)``."""
    n_len = _length(length)
    return _seeded_recursive(_source(source), n_len, 1.0 / n_len)


def rsi(source: ArrayLike, length: int) -> np.ndarray:
    """``ta.rsi``. Manual's code::

        u = math.max(x - x[1], 0)
        d = math.max(x[1] - x, 0)
        up = ta.rma(u, y)
        down = ta.rma(d, y)
        down == 0 ? 100 : up == 0 ? 0 : 100 - 100 / (1 + up / down)

    The first change is `na`, so the first value is at bar ``length`` (0-based).
    """
    n_len = _length(length)
    x = _source(source)
    u = np.full(x.size, np.nan)
    d = np.full(x.size, np.nan)
    for t in range(1, x.size):
        ch = x[t] - x[t - 1]
        if not math.isnan(ch):
            u[t] = max(ch, 0.0)
            d[t] = max(-ch, 0.0)
    up = rma(u, n_len)
    down = rma(d, n_len)
    out = np.full(x.size, np.nan)
    for t in range(x.size):
        if math.isnan(up[t]) or math.isnan(down[t]):
            continue
        if down[t] == 0.0:
            out[t] = 100.0
        elif up[t] == 0.0:
            out[t] = 0.0
        else:
            out[t] = 100.0 - 100.0 / (1.0 + up[t] / down[t])
    return out
