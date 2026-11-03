"""Convergence bounds for composite indicators (PINE_PLAN.md §5.3, §10 B3).

Use when Python's history starts *later* than Pine's (a truncated export, §5.4): Python's
filters seed from different data, and the difference decays. These functions bound that
difference per bar and give the number of leading bars to exclude in
``compare_series(mode="convergence", exclude=n)``. As with any convergence-mode check, a pass
says nothing about seeding (§5.3).

All functions take **Python's** (truncated) arrays and return indices into them.

**Assumption.** The bounds need a magnitude for the unseen history: ``e0`` (ATR) or
``max_change`` (RSI). When not given, they are estimated from the data supplied, which
assumes the unseen bars had no larger true ranges / bar-to-bar moves than the seen ones. Pass
an explicit, larger value if that is doubtful.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from pine import ta
from pine.parity import convergence_bars

__all__ = [
    "atr_truncation_error",
    "atr_convergence_bars",
    "rsi_truncation_error",
    "rsi_convergence_bars",
]


def _first_valid(x: np.ndarray) -> int:
    idx = np.flatnonzero(~np.isnan(x))
    if not idx.size:
        raise ValueError("series has no valid values; history too short for this length")
    return int(idx[0])


def _bars_until_below(bound: np.ndarray, w: int, tol: float) -> int:
    """Smallest n >= w such that bound[t] <= tol for every t >= n."""
    bad = np.flatnonzero(bound[w:] > tol)
    return w if not bad.size else w + int(bad[-1]) + 1


def atr_truncation_error(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, length: int, e0: float | None = None
) -> np.ndarray:
    """Per-bar bound on ``|ATR_py - ATR_pine|`` when Python's history is a suffix of Pine's.

    From Python's second bar on, both sides compute identical true ranges, so after Python's
    seed bar ``w`` the two RMAs differ only by their state at ``w``, which decays exactly as
    ``e0 (1 - 1/length)^(t - w)``. Both ATRs are convex combinations of non-negative true
    ranges, so ``e0 <= max TR`` over the whole history; by default the maximum of Python's
    true ranges (see the module assumption). ``nan`` before ``w``.
    """
    h, lo, c = (np.asarray(a, dtype=np.float64) for a in (high, low, close))
    atr = ta.atr(h, lo, c, length)
    w = _first_valid(atr)
    if e0 is None:
        e0 = float(np.nanmax(ta.tr(h, lo, c, handle_na=True)))
    alpha = 1.0 / length
    out = np.full(atr.size, np.nan)
    k = np.arange(atr.size - w, dtype=float)
    out[w:] = e0 * (1 - alpha) ** k
    return out


def atr_convergence_bars(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    length: int,
    tol: float,
    e0: float | None = None,
) -> int:
    """Bars to exclude so the truncation error of ATR is ``<= tol`` everywhere after."""
    h = np.asarray(high, dtype=np.float64)
    if e0 is None:
        e0 = float(np.nanmax(ta.tr(h, low, close, handle_na=True)))
    w = length - 1
    if w >= h.size:
        raise ValueError("history too short for this length")
    return w + convergence_bars(1.0 / length, tol, e0)


def rsi_truncation_error(
    source: ArrayLike,
    length: int,
    max_change: float | None = None,
    na_policy: ta.NaPolicy = "propagate",
) -> np.ndarray:
    """Per-bar bound on ``|RSI_py - RSI_pine|`` when Python's history is a suffix of Pine's.

    ``u`` and ``d`` lie in ``[0, M]`` with ``M`` the largest absolute bar-to-bar change, so the
    up/down RMAs on both sides lie in ``[0, M]`` and differ at Python's seed bar ``w`` by at
    most ``M``; after ``w`` they see identical inputs, so each difference decays as
    ``(1 - 1/length)^(t - w)``. With ``RSI = 100 U/(U+D)``::

        |ΔRSI| = 100 |ΔU·D - U·ΔD| / ((U'+D')(U+D)) <= 100 max(|ΔU|, |ΔD|) / (U'+D')

    giving ``min(100, 100 M (1-α)^(t-w) / (U'_t + D'_t))`` from Python's own ``U', D'``.
    ``M`` defaults to the largest change in ``source`` (module assumption).
    """
    x = np.asarray(source, dtype=np.float64)
    if max_change is None:
        max_change = float(np.nanmax(np.abs(np.diff(x))))
    up, down = ta.rsi_components(x, length, na_policy)
    total = up + down
    w = _first_valid(total)
    alpha = 1.0 / length
    out = np.full(x.size, np.nan)
    k = np.arange(x.size - w, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = 100.0 * max_change * (1 - alpha) ** k / total[w:]
    b = np.where(total[w:] > 0, b, 100.0)
    out[w:] = np.minimum(b, 100.0)
    out[w:][np.isnan(total[w:])] = np.nan
    return out


def rsi_convergence_bars(
    source: ArrayLike,
    length: int,
    tol: float,
    max_change: float | None = None,
    na_policy: ta.NaPolicy = "propagate",
) -> int:
    """Bars to exclude so the RSI truncation bound is ``<= tol`` for every later bar.

    Unlike EMA/ATR this depends on the data: the bound is large wherever ``U'+D'`` is small
    (quiet markets), so the answer is computed per bar rather than from a formula.
    """
    if tol <= 0:
        raise ValueError("tol must be positive")
    bound = rsi_truncation_error(source, length, max_change, na_policy)
    w = _first_valid(bound)
    filled = np.where(np.isnan(bound), math.inf, bound)
    return _bars_until_below(filled, w, tol)
