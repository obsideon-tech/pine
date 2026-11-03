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

**`na` inside windows is unsettled** (PINE_PLAN.md §5.10, §10 A2). The window functions
(`sma`, `ema`, `rma`, `rsi`, `stdev`, `highest`, `lowest`) take ``na_policy``:

- ``"propagate"`` (default): the manual's code as recalled; any `na` in the window gives
  `na`, and recursive filters re-seed after a gap.
- ``"skip"``: `na` values are ignored and the window extends back over ``length`` non-`na`
  values; a bar whose own source is `na` still gets a value (recursive filters carry their
  state unchanged across it).
- ``"skip_na_out"``: as ``"skip"``, but the output is `na` wherever the source is `na`.

**Defaults are the observed TradingView behaviour** (`DEFAULT_NA_POLICY`; BATS:NFLX 1D export
of ``tools/reference_capture.pine``, 2026-11-03, PINE_PLAN.md §12): the windowed built-ins
(``sma``, ``stdev``, ``highest``, ``lowest``) follow ``"skip"``; the recursive ones (``ema``,
``rma``, ``rsi``) follow ``"skip_na_out"``. In that export each column matched exactly one
policy. ``na_policy`` stays as an explicit override. On gap-free input all three are identical.
``atr`` inherits ``rma``'s default (the manual's code is ``ta.rma(trueRange, length)``); that
inheritance itself is not yet observed.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

NaPolicy = Literal["propagate", "skip", "skip_na_out"]
NA_POLICIES: tuple[str, ...] = ("propagate", "skip", "skip_na_out")
DEFAULT_NA_POLICY: dict[str, str] = {
    "sma": "skip",
    "stdev": "skip",
    "highest": "skip",
    "lowest": "skip",
    "ema": "skip_na_out",
    "rma": "skip_na_out",
    "rsi": "skip_na_out",
}

__all__ = [
    "NA_POLICIES",
    "DEFAULT_NA_POLICY",
    "NaPolicy",
    "sma",
    "ema",
    "rma",
    "rsi",
    "rsi_components",
    "stdev",
    "highest",
    "lowest",
    "tr",
    "atr",
    "crossover",
    "crossunder",
    "change",
    "barssince",
    "valuewhen",
    "supertrend",
]


def _length(length: int) -> int:
    if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 1:
        raise ValueError(f"length must be a positive integer, got {length!r}")
    return int(length)


def _policy(na_policy: str | None, name: str | None = None) -> str:
    if na_policy is None and name is not None:
        return DEFAULT_NA_POLICY[name]
    if na_policy not in NA_POLICIES:
        raise ValueError(f"na_policy must be one of {NA_POLICIES}, got {na_policy!r}")
    return na_policy


def _skip_windows(x: np.ndarray, n_len: int, policy: str) -> Iterator[tuple[int, np.ndarray]]:
    """For the skip policies: ``(t, window)`` with the last ``n_len`` non-`na` values up to
    bar ``t``, oldest first. Under ``skip_na_out`` bars whose own value is `na` are omitted."""
    valid: list[int] = []
    for t in range(x.size):
        own_na = math.isnan(x[t])
        if not own_na:
            valid.append(t)
        if len(valid) >= n_len and not (own_na and policy == "skip_na_out"):
            yield t, x[valid[-n_len:]]


def _source(source: ArrayLike) -> np.ndarray:
    x = np.array(source, dtype=np.float64, copy=True)
    if x.ndim != 1:
        raise ValueError(f"source must be one-dimensional, got shape {x.shape}")
    return x


def sma(source: ArrayLike, length: int, na_policy: NaPolicy | None = None) -> np.ndarray:
    """``ta.sma``. Manual's code::

        pine_sma(x, y) =>
            sum = 0.0
            for i = 0 to y - 1
                sum := sum + x[i] / y
            sum

    Newest bar first, each term divided before adding. `na` for the first ``length - 1``
    bars and, under ``"propagate"``, for any window containing `na`.
    """
    n_len = _length(length)
    x = _source(source)
    out = np.full(x.size, np.nan)
    policy = _policy(na_policy, "sma")
    if policy != "propagate":
        for t, w in _skip_windows(x, n_len, policy):
            s = 0.0
            for v in w[::-1]:
                s += v / n_len
            out[t] = s
        return out
    for t in range(n_len - 1, x.size):
        s = 0.0
        for i in range(n_len):
            s += x[t - i] / n_len
        out[t] = s
    return out


def _seeded_recursive(x: np.ndarray, n_len: int, alpha: float, policy: str) -> np.ndarray:
    """``sum := na(sum[1]) ? ta.sma(src, length) : alpha * src + (1 - alpha) * nz(sum[1])``.

    Re-seeds from the SMA whenever the previous value is `na`, as the Pine code does. Under
    the skip policies the state is seeded once from the skip-SMA and carried unchanged
    across `na` source bars.
    """
    if policy != "propagate":
        seed = sma(x, n_len, "skip")
        out = np.full(x.size, np.nan)
        state = math.nan
        for t in range(x.size):
            if math.isnan(x[t]):
                if policy == "skip":
                    out[t] = state
                continue
            state = seed[t] if math.isnan(state) else alpha * x[t] + (1.0 - alpha) * state
            out[t] = state
        return out
    seed = sma(x, n_len, "propagate")
    out = np.full(x.size, np.nan)
    prev = math.nan
    for t in range(x.size):
        cur = seed[t] if math.isnan(prev) else alpha * x[t] + (1.0 - alpha) * prev
        out[t] = cur
        prev = cur
    return out


def ema(source: ArrayLike, length: int, na_policy: NaPolicy | None = None) -> np.ndarray:
    """``ta.ema``: ``alpha = 2 / (length + 1)``, seeded with ``ta.sma(src, length)``."""
    n_len = _length(length)
    return _seeded_recursive(_source(source), n_len, 2.0 / (n_len + 1), _policy(na_policy, "ema"))


def rma(source: ArrayLike, length: int, na_policy: NaPolicy | None = None) -> np.ndarray:
    """``ta.rma`` (Wilder): ``alpha = 1 / length``, seeded with ``ta.sma(src, length)``."""
    n_len = _length(length)
    return _seeded_recursive(_source(source), n_len, 1.0 / n_len, _policy(na_policy, "rma"))


def rsi_components(
    source: ArrayLike, length: int, na_policy: NaPolicy | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(up, down)``: the two RMAs inside ``ta.rsi``. Exposed so the RSI input-rounding bound
    (`pine.sensitivity.rsi_input_error`) uses exactly the port's values."""
    n_len = _length(length)
    x = _source(source)
    u = np.full(x.size, np.nan)
    d = np.full(x.size, np.nan)
    for t in range(1, x.size):
        ch = x[t] - x[t - 1]
        if not math.isnan(ch):
            u[t] = max(ch, 0.0)
            d[t] = max(-ch, 0.0)
    policy = _policy(na_policy, "rsi")
    return rma(u, n_len, policy), rma(d, n_len, policy)


def rsi(source: ArrayLike, length: int, na_policy: NaPolicy | None = None) -> np.ndarray:
    """``ta.rsi``. Manual's code::

        u = math.max(x - x[1], 0)
        d = math.max(x[1] - x, 0)
        up = ta.rma(u, y)
        down = ta.rma(d, y)
        down == 0 ? 100 : up == 0 ? 0 : 100 - 100 / (1 + up / down)

    The first change is `na`, so the first value is at bar ``length`` (0-based). A change
    is `na` when either bar is `na`; ``na_policy`` applies to the two RMAs.
    """
    up, down = rsi_components(source, length, na_policy)
    out = np.full(up.size, np.nan)
    for t in range(up.size):
        if math.isnan(up[t]) or math.isnan(down[t]):
            continue
        if down[t] == 0.0:
            out[t] = 100.0
        elif up[t] == 0.0:
            out[t] = 0.0
        else:
            out[t] = 100.0 - 100.0 / (1.0 + up[t] / down[t])
    return out


_STDEV_EPS = 1e-10


def stdev(
    source: ArrayLike, length: int, biased: bool = True, na_policy: NaPolicy | None = None
) -> np.ndarray:
    """``ta.stdev``. Manual's code, in outline::

        avg = ta.sma(src, length)
        for i = 0 to length - 1
            sum = SUM(src[i], -avg)        // |sum| <= 1e-10 -> 0
            sumOfSquareDeviations += sum * sum
        stdev = math.sqrt(sumOfSquareDeviations / length)

    ``biased=True`` (the default, **unverified**) divides by ``length`` (population);
    ``False`` divides by ``length - 1`` (sample).

    The manual's ``SUM`` helper, as recalled, also returns the literal ``15`` for deviations
    with ``1e-10 < |d| <= 1e-4``. That reads as a bug in the documentation example, not the
    built-in's behaviour, and is **not** ported; only the zeroing of ``|d| <= 1e-10`` is.
    """
    n_len = _length(length)
    if not biased and n_len < 2:
        raise ValueError("unbiased stdev needs length >= 2")
    x = _source(source)
    denom = n_len if biased else n_len - 1
    out = np.full(x.size, np.nan)
    policy = _policy(na_policy, "stdev")
    if policy != "propagate":
        avg = sma(x, n_len, policy)
        for t, w in _skip_windows(x, n_len, policy):
            ss = 0.0
            for v in w[::-1]:
                d = v - avg[t]
                if abs(d) <= _STDEV_EPS:
                    d = 0.0
                ss += d * d
            out[t] = math.sqrt(ss / denom)
        return out
    avg = sma(x, n_len, "propagate")
    for t in range(n_len - 1, x.size):
        if math.isnan(avg[t]):
            continue
        ss = 0.0
        for i in range(n_len):
            d = x[t - i] - avg[t]
            if abs(d) <= _STDEV_EPS:
                d = 0.0
            ss += d * d
        out[t] = math.sqrt(ss / denom)
    return out


def _window_extreme(x: np.ndarray, n_len: int, pick, policy: str) -> np.ndarray:
    out = np.full(x.size, np.nan)
    if policy != "propagate":
        for t, w in _skip_windows(x, n_len, policy):
            out[t] = pick(w)
        return out
    for t in range(n_len - 1, x.size):
        w = x[t - n_len + 1 : t + 1]
        if not np.isnan(w).any():
            out[t] = pick(w)
    return out


def highest(source: ArrayLike, length: int, na_policy: NaPolicy | None = None) -> np.ndarray:
    """``ta.highest(source, length)``: maximum over the current and previous ``length - 1``
    bars. `na` for the first ``length - 1`` bars; `na` inside the window per ``na_policy``
    (**unverified** which policy the built-in uses; see PINE_PLAN.md §10 A2)."""
    return _window_extreme(_source(source), _length(length), np.max, _policy(na_policy, "highest"))


def lowest(source: ArrayLike, length: int, na_policy: NaPolicy | None = None) -> np.ndarray:
    """``ta.lowest(source, length)``; same window and `na` rules as `highest`."""
    return _window_extreme(_source(source), _length(length), np.min, _policy(na_policy, "lowest"))


def tr(high: ArrayLike, low: ArrayLike, close: ArrayLike, handle_na: bool = False) -> np.ndarray:
    """``ta.tr(handle_na)``: ``max(high - low, |high - close[1]|, |low - close[1]|)``.

    Where ``close[1]`` is `na` (the first bar), returns ``high - low`` if ``handle_na`` else
    `na`. Pine's built-in variable ``ta.tr`` behaves like ``handle_na=False`` (**unverified**).
    """
    h, lo, c = _source(high), _source(low), _source(close)
    if not (h.size == lo.size == c.size):
        raise ValueError("high, low and close must have the same length")
    out = np.full(h.size, np.nan)
    for t in range(h.size):
        prev = c[t - 1] if t > 0 else math.nan
        if math.isnan(prev):
            out[t] = h[t] - lo[t] if handle_na else math.nan
        else:
            hl, hc, lc = h[t] - lo[t], abs(h[t] - prev), abs(lo[t] - prev)
            # Pine's math.max returns na if any argument is na.
            out[t] = math.nan if math.isnan(hl + hc + lc) else max(hl, hc, lc)
    return out


def atr(high: ArrayLike, low: ArrayLike, close: ArrayLike, length: int) -> np.ndarray:
    """``ta.atr(length)``. Manual's code::

        trueRange = na(high[1]) ? high - low
                  : math.max(math.max(high - low, math.abs(high - close[1])),
                             math.abs(low - close[1]))
        ta.rma(trueRange, length)

    The first bar's true range is ``high - low``, so ATR's first value is at bar
    ``length - 1``. The ``na(high[1])`` test is on ``high``, not ``close``: a `na` close later
    in the series makes the next true range `na` (via ``math.max``), not ``high - low``.
    """
    n_len = _length(length)
    h, lo, c = _source(high), _source(low), _source(close)
    if not (h.size == lo.size == c.size):
        raise ValueError("high, low and close must have the same length")
    true_range = np.full(h.size, np.nan)
    for t in range(h.size):
        if t == 0 or math.isnan(h[t - 1]):
            true_range[t] = h[t] - lo[t]
        else:
            hl, hc, lc = h[t] - lo[t], abs(h[t] - c[t - 1]), abs(lo[t] - c[t - 1])
            true_range[t] = math.nan if math.isnan(hl + hc + lc) else max(hl, hc, lc)
    return rma(true_range, n_len)


def _pair(a: ArrayLike, b: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    a_ = _source(a)
    b_ = np.broadcast_to(np.asarray(b, dtype=np.float64), a_.shape).copy()
    return a_, b_


def crossover(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """``ta.crossover(a, b)``: ``a > b and a[1] <= b[1]``.

    ``b`` may be a scalar level. Comparisons involving `na` are false, so the result is
    ``False`` (never `na`) on the first bar and wherever an input is `na`, matching v6, where
    ``bool`` cannot be `na`.
    """
    a_, b_ = _pair(a, b)
    out = np.zeros(a_.size, dtype=bool)
    with np.errstate(invalid="ignore"):
        out[1:] = (a_[1:] > b_[1:]) & (a_[:-1] <= b_[:-1])
    return out


def crossunder(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """``ta.crossunder(a, b)``: ``a < b and a[1] >= b[1]``."""
    a_, b_ = _pair(a, b)
    out = np.zeros(a_.size, dtype=bool)
    with np.errstate(invalid="ignore"):
        out[1:] = (a_[1:] < b_[1:]) & (a_[:-1] >= b_[:-1])
    return out


def change(source: ArrayLike, length: int = 1) -> np.ndarray:
    """``ta.change(source, length)``: ``source - source[length]``; `na` if either is `na`
    (**unverified**: the alternative "last valid value" is probed by
    ``tools/capture_semantics.pine``)."""
    n_len = _length(length)
    x = _source(source)
    out = np.full(x.size, np.nan)
    out[n_len:] = x[n_len:] - x[:-n_len]
    return out


def barssince(condition: ArrayLike) -> np.ndarray:
    """``ta.barssince(condition)``: bars since ``condition`` was last true (0 on a true bar);
    `na` before the first true bar (**unverified**, probed)."""
    c = np.asarray(condition, dtype=bool)
    out = np.full(c.size, np.nan)
    count = math.nan
    for t in range(c.size):
        if c[t]:
            count = 0.0
        elif not math.isnan(count):
            count += 1.0
        out[t] = count
    return out


def valuewhen(condition: ArrayLike, source: ArrayLike, occurrence: int = 0) -> np.ndarray:
    """``ta.valuewhen(condition, source, occurrence)``: ``source`` on the ``occurrence``-th most
    recent bar where ``condition`` was true (0 = latest); `na` before enough events."""
    if occurrence < 0:
        raise ValueError("occurrence must be >= 0")
    c = np.asarray(condition, dtype=bool)
    x = _source(source)
    if c.size != x.size:
        raise ValueError("condition and source must have the same length")
    out = np.full(x.size, np.nan)
    hits: list[float] = []
    for t in range(x.size):
        if c[t]:
            hits.append(x[t])
        if len(hits) > occurrence:
            out[t] = hits[-1 - occurrence]
    return out


def supertrend(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, factor: float, atr_period: int
) -> tuple[np.ndarray, np.ndarray]:
    """``ta.supertrend(factor, atrPeriod)`` -> ``(supertrend, direction)``. Manual's code::

        src = hl2
        atr = ta.atr(atrPeriod)
        upperBand = src + factor * atr
        lowerBand = src - factor * atr
        prevLowerBand = nz(lowerBand[1])
        prevUpperBand = nz(upperBand[1])
        lowerBand := lowerBand > prevLowerBand or close[1] < prevLowerBand ?
             lowerBand : prevLowerBand
        upperBand := upperBand < prevUpperBand or close[1] > prevUpperBand ?
             upperBand : prevUpperBand
        int _direction = na
        float superTrend = na
        prevSuperTrend = superTrend[1]
        if na(atr[1])
            _direction := 1
        else if prevSuperTrend == prevUpperBand
            _direction := close > upperBand ? -1 : 1
        else
            _direction := close < lowerBand ? 1 : -1
        superTrend := _direction == -1 ? lowerBand : upperBand

    Direction ``-1`` is an uptrend (line below price), ``1`` a downtrend. ``lowerBand[1]`` is the
    *final* (reassigned) value of the previous bar. Comparisons with `na` are false, so during
    ATR warm-up the lower band holds ``nz(na) = 0``, exactly as the Pine code does. A plain loop,
    independent of `pine.runtime`, used as its second implementation (**unverified** against
    TradingView until a capture export exists).
    """
    h, lo, c = _source(high), _source(low), _source(close)
    a = atr(h, lo, c, atr_period)
    src = (h + lo) / 2
    n = c.size
    st = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    lower_f = np.full(n, np.nan)
    upper_f = np.full(n, np.nan)

    def nz(v: float) -> float:
        return 0.0 if math.isnan(v) else v

    for t in range(n):
        lb = src[t] - factor * a[t]
        ub = src[t] + factor * a[t]
        prev_lb = nz(lower_f[t - 1]) if t > 0 else 0.0
        prev_ub = nz(upper_f[t - 1]) if t > 0 else 0.0
        prev_close = c[t - 1] if t > 0 else math.nan
        lb = lb if (lb > prev_lb or prev_close < prev_lb) else prev_lb
        ub = ub if (ub < prev_ub or prev_close > prev_ub) else prev_ub
        lower_f[t], upper_f[t] = lb, ub
        prev_st = st[t - 1] if t > 0 else math.nan
        if t == 0 or math.isnan(a[t - 1]):
            d = 1.0
        elif prev_st == prev_ub:
            d = -1.0 if c[t] > ub else 1.0
        else:
            d = 1.0 if c[t] < lb else -1.0
        direction[t] = d
        st[t] = lb if d == -1.0 else ub
    return st, direction
