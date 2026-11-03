"""Checking exports that start part-way through the chart's history (PINE_PLAN.md §5.4).

TradingView exports can begin long after the chart's first loaded bar. The first real export
(NFLX daily) started at ``bar_index`` 3116 of 6151. Pine's state at the first exported row then
depends on bars Python never sees. What can still be established, and how:

- **Offset.** The exported ``bar_index`` column gives the true bar number of each row, so
  patterns keyed to it (probe schedules, gap patterns) are reproduced exactly.
- **Windowed built-ins** (sma, stdev, highest, lowest, tr): exact once Python's own window is
  full; compare after that warm-up.
- **Seeded recursive filters** (ema, rma, atr): after Python's seed bar ``w`` both sides apply
  the same linear recursion to the same inputs, so the residual ``py - pine`` must be exactly
  ``r_w (1-α)^(t-w)``. `decay_adjusted` removes that predicted residual; comparing the result
  from ``w`` verifies the recursion and ``α`` on every bar. Only the seed stays unverified.
- **Other history-dependent values** (RSI, conditional EMAs): convergence mode with
  `pine.convergence` / `pine.parity.convergence_bars` exclusions.
- **Stateful ports** (Supertrend, trailing stops, state machines): their state is reset by
  events, so they are compared from the observed synchronisation row (`sync_point`), which is
  reported and must fall in the first half of the export.

None of these certifies seeding (§5.3); every report says so.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["bar_offset", "decay_adjusted", "e0_bound", "sync_point"]


def bar_offset(columns: dict[str, np.ndarray], name: str = "bar_index") -> int | None:
    """True ``bar_index`` of the first exported row, from the ``name`` column; ``None`` if
    absent. Raises, naming the column and the first offending row, if the column is not
    consecutive non-negative integers (rows missing, repeated or reordered, or `na`)."""
    if name not in columns:
        return None
    b = np.asarray(columns[name], dtype=np.float64)
    if b.size == 0:
        return 0
    nan = np.flatnonzero(np.isnan(b))
    if nan.size:
        raise ValueError(f"{name}: na at row {int(nan[0])} ({nan.size} na rows in total)")
    breaks = np.flatnonzero(np.diff(b) != 1)
    if breaks.size:
        i = int(breaks[0])
        raise ValueError(
            f"{name}: not consecutive at row {i + 1} ({b[i]:g} then {b[i + 1]:g}; "
            f"{breaks.size} breaks in total)"
        )
    if b[0] != int(b[0]) or b[0] < 0:
        raise ValueError(f"{name}: first value {b[0]:g} is not a non-negative integer")
    return int(b[0])


def decay_adjusted(py: ArrayLike, pine: ArrayLike, alpha: float, w: int) -> np.ndarray:
    """``py`` minus the residual a pure seed difference at bar ``w`` would leave:
    ``(py[w] - pine[w]) (1-α)^(t-w)`` for ``t >= w``."""
    p = np.asarray(py, dtype=np.float64)
    r = np.asarray(pine, dtype=np.float64)
    out = p.copy()
    if w >= p.size or np.isnan(p[w]) or np.isnan(r[w]):
        return out
    k = np.arange(p.size - w, dtype=float)
    out[w:] = p[w:] - (p[w] - r[w]) * (1 - alpha) ** k
    return out


def e0_bound(source: ArrayLike) -> float:
    """Bound on a seed difference for filters that are convex combinations of non-negative
    inputs: twice the largest magnitude seen (assumes unseen history was no larger)."""
    return 2.0 * float(np.nanmax(np.abs(np.asarray(source, dtype=np.float64))))


def sync_point(py: ArrayLike, pine: ArrayLike, bound: ArrayLike, start: int = 0) -> int:
    """First row ``r >= start`` from which ``py`` and ``pine`` agree (same ``na`` mask, values
    within ``bound``) on every later row. Equals ``len`` if they never synchronise."""
    p = np.asarray(py, dtype=np.float64)
    r = np.asarray(pine, dtype=np.float64)
    b = np.broadcast_to(np.asarray(bound, dtype=np.float64), p.shape)
    na_p, na_r = np.isnan(p), np.isnan(r)
    both = ~na_p & ~na_r
    err = np.where(both, np.abs(np.where(both, p, 0) - np.where(both, r, 0)), 0.0)
    fail = (na_p != na_r) | (both & (err > b))
    bad = np.flatnonzero(fail[start:])
    return start if not bad.size else start + int(bad[-1]) + 1
