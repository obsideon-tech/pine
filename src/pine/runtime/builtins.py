"""Incremental, call-site-keyed ``ta.*`` built-ins for `pine.runtime` (PINE_PLAN.md §11 E2).

Each object holds the state of one call site and advances **only when called**, which is how
Pine evaluates built-ins inside conditional code (expected hypothesis ``per_call``, probed by
``tools/capture_semantics.pine``). Called on every bar, each one reproduces the vectorised
`pine.ta` function bit for bit (same arithmetic order); ``tests/test_runtime.py`` enforces it.

`na` handling follows `pine.ta.DEFAULT_NA_POLICY` (observed TradingView behaviour,
PINE_PLAN.md §12): windowed built-ins ``skip`` `na` (window over the last ``length`` valid
values, value also on `na` bars); recursive ones carry state across `na` and output `na` on
`na` bars (``skip_na_out``). ``propagate`` remains available as an override.
"""

from __future__ import annotations

import math

import numpy as np

from pine.ta import NA_POLICIES

NAN = math.nan


def _check(policy: str) -> str:
    if policy not in NA_POLICIES:
        raise ValueError(f"na_policy must be one of {NA_POLICIES}, got {policy!r}")
    return policy


def _f(x) -> float:
    return NAN if x is None else float(x)


class _Window:
    def __init__(self, n: int):
        self.n = n
        self.h: list[float] = []

    def push(self, x: float) -> None:
        self.h.append(x)
        if len(self.h) > self.n:
            del self.h[0]

    @property
    def full(self) -> bool:
        return len(self.h) == self.n


class _ValidWindow:
    """Last ``n`` *valid* values (skip policies); ``ready(own_na)`` says whether to output."""

    def __init__(self, n: int, policy: str):
        self.w = _Window(n)
        self.policy = policy

    def push(self, x: float) -> bool:
        own_na = math.isnan(x)
        if not own_na:
            self.w.push(x)
        return self.w.full and not (own_na and self.policy == "skip_na_out")


class Sma:
    def __init__(self, length: int, policy: str = "skip"):
        self.policy = _check(policy)
        self.w = _Window(length) if policy == "propagate" else None
        self.vw = None if policy == "propagate" else _ValidWindow(length, policy)

    def _mean(self, h: list[float], n: int) -> float:
        s = 0.0
        for v in reversed(h):  # newest first, as pine.ta.sma
            s += v / n
        return s

    def update(self, x) -> float:
        x = _f(x)
        if self.vw is not None:
            if not self.vw.push(x):
                return NAN
            return self._mean(self.vw.w.h, self.vw.w.n)
        self.w.push(x)
        if not self.w.full:
            return NAN
        return self._mean(self.w.h, self.w.n)


class _Seeded:
    def __init__(self, length: int, alpha: float, policy: str):
        self.policy = _check(policy)
        self.sma = Sma(length, "propagate" if policy == "propagate" else "skip")
        self.alpha = alpha
        self.prev = NAN

    def update(self, x) -> float:
        x = _f(x)
        seed = self.sma.update(x)
        a = self.alpha
        if self.policy != "propagate":
            # State is carried across na bars (pine.ta._seeded_recursive, skip policies).
            if math.isnan(x):
                return self.prev if self.policy == "skip" else NAN
            self.prev = seed if math.isnan(self.prev) else a * x + (1.0 - a) * self.prev
            return self.prev
        cur = seed if math.isnan(self.prev) else a * x + (1.0 - a) * self.prev
        self.prev = cur
        return cur


class Ema(_Seeded):
    def __init__(self, length: int, policy: str = "skip_na_out"):
        super().__init__(length, 2.0 / (length + 1), policy)


class Rma(_Seeded):
    def __init__(self, length: int, policy: str = "skip_na_out"):
        super().__init__(length, 1.0 / length, policy)


class Rsi:
    def __init__(self, length: int, policy: str = "skip_na_out"):
        self.up, self.down = Rma(length, policy), Rma(length, policy)
        self.prev = NAN

    def update(self, x) -> float:
        x = _f(x)
        ch = x - self.prev
        self.prev = x
        u = NAN if math.isnan(ch) else max(ch, 0.0)
        d = NAN if math.isnan(ch) else max(-ch, 0.0)
        up, down = self.up.update(u), self.down.update(d)
        if math.isnan(up) or math.isnan(down):
            return NAN
        if down == 0.0:
            return 100.0
        if up == 0.0:
            return 0.0
        return 100.0 - 100.0 / (1.0 + up / down)


class Stdev:
    def __init__(self, length: int, biased: bool = True, policy: str = "skip"):
        self.policy = _check(policy)
        self.sma = Sma(length, policy)
        self.w = _Window(length)
        self.vw = None if policy == "propagate" else _ValidWindow(length, policy)
        self.denom = length if biased else length - 1

    def update(self, x) -> float:
        x = _f(x)
        avg = self.sma.update(x)
        if self.vw is not None:
            if not self.vw.push(x):
                return NAN
            window = self.vw.w.h
        else:
            self.w.push(x)
            if not self.w.full or math.isnan(avg):
                return NAN
            window = self.w.h
        ss = 0.0
        for v in reversed(window):
            d = v - avg
            if abs(d) <= 1e-10:
                d = 0.0
            ss += d * d
        return math.sqrt(ss / self.denom)


class _Extreme:
    pick = staticmethod(np.max)

    def __init__(self, length: int, policy: str = "skip"):
        self.policy = _check(policy)
        self.w = _Window(length)
        self.vw = None if policy == "propagate" else _ValidWindow(length, policy)

    def update(self, x) -> float:
        x = _f(x)
        if self.vw is not None:
            return float(self.pick(np.array(self.vw.w.h))) if self.vw.push(x) else NAN
        self.w.push(x)
        if not self.w.full:
            return NAN
        arr = np.array(self.w.h)
        return NAN if np.isnan(arr).any() else float(self.pick(arr))


class Highest(_Extreme):
    pick = staticmethod(np.max)


class Lowest(_Extreme):
    pick = staticmethod(np.min)


class Atr:
    """``ta.atr``: true range from the *bar* series (the Pine code reads the global ``high``,
    ``low``, ``close``); RMA state per call."""

    def __init__(self, length: int):
        self.rma = Rma(length)  # rma's default policy, as pine.ta.atr

    def update(self, high, low, prev_high, prev_close) -> float:
        h, lo = _f(high), _f(low)
        if math.isnan(_f(prev_high)):
            tr = h - lo
        else:
            pc = _f(prev_close)
            hl, hc, lc = h - lo, abs(h - pc), abs(lo - pc)
            tr = NAN if math.isnan(hl + hc + lc) else max(hl, hc, lc)
        return self.rma.update(tr)


class _Cross:
    def __init__(self):
        self.pa = NAN
        self.pb = NAN

    def update(self, a, b) -> bool:
        a, b = _f(a), _f(b)
        res = self._test(a, b, self.pa, self.pb)
        self.pa, self.pb = a, b
        return bool(res)


class Crossover(_Cross):
    @staticmethod
    def _test(a, b, pa, pb):
        return a > b and pa <= pb


class Crossunder(_Cross):
    @staticmethod
    def _test(a, b, pa, pb):
        return a < b and pa >= pb


class Change:
    def __init__(self, length: int = 1):
        self.w = _Window(length + 1)
        self.n = length

    def update(self, x) -> float:
        self.w.push(_f(x))
        if not self.w.full:
            return NAN
        return self.w.h[-1] - self.w.h[0]


class Barssince:
    def __init__(self):
        self.count = NAN

    def update(self, cond) -> float:
        if cond:
            self.count = 0.0
        elif not math.isnan(self.count):
            self.count += 1.0
        return self.count


class Valuewhen:
    def __init__(self, occurrence: int = 0):
        if occurrence < 0:
            raise ValueError("occurrence must be >= 0")
        self.occ = occurrence
        self.hits: list[float] = []

    def update(self, cond, source) -> float:
        if cond:
            self.hits.append(_f(source))
            if len(self.hits) > self.occ + 1:
                del self.hits[0]
        return self.hits[-1 - self.occ] if len(self.hits) > self.occ else NAN


# ---------------------------------------------------------------------------- batch F
# The built-ins below are defined once, here, as incremental objects; the vectorised
# `pine.ta` functions of the same name stream them over an array, so the two agree bit for bit
# by construction. Their TradingView behaviour is checked by ``tools/capture_builtins.pine``
# (`pine.builtin_capture`); until an export is checked each is **unverified**.


class _Windowed:
    """Window of the last ``length`` values under an ``na`` policy; subclasses reduce it.

    ``propagate``: any `na` in the window gives `na`. ``skip``/``skip_na_out``: the window
    holds the last ``length`` valid values (see `_ValidWindow`)."""

    def __init__(self, length: int, policy: str):
        self.policy = _check(policy)
        self.n = length
        self.w = _Window(length)
        self.vw = None if policy == "propagate" else _ValidWindow(length, policy)

    def _window(self, x: float) -> list[float] | None:
        if self.vw is not None:
            return self.vw.w.h if self.vw.push(x) else None
        self.w.push(x)
        if not self.w.full or any(math.isnan(v) for v in self.w.h):
            return None
        return self.w.h

    def update(self, x) -> float:
        h = self._window(_f(x))
        return NAN if h is None else self._reduce(h)


class Sum(_Windowed):
    """``math.sum(source, length)``: newest value first, as `Sma` without the division."""

    def _reduce(self, h: list[float]) -> float:
        s = 0.0
        for v in reversed(h):
            s += v
        return s


class Wma(_Windowed):
    """``ta.wma``. Manual's code::

    norm = 0.0, sum = 0.0
    for i = 0 to y - 1
        weight = (y - i) * y
        norm := norm + weight
        sum := sum + x[i] * weight
    sum / norm
    """

    def _reduce(self, h: list[float]) -> float:
        n = self.n
        norm = s = 0.0
        for i, v in enumerate(reversed(h)):
            weight = float((n - i) * n)
            norm += weight
            s += v * weight
        return s / norm


class Hma:
    """``ta.hma``: ``wma(2 * wma(src, length / 2) - wma(src, length), floor(sqrt(length)))``
    with ``length / 2`` truncated to an int (**unverified**; the capture probes floor vs
    round for both lengths)."""

    def __init__(self, length: int, policy: str = "skip"):
        if length < 2:
            raise ValueError("hma length must be >= 2")
        self.half = Wma(length // 2, policy)
        self.full = Wma(length, policy)
        self.outer = Wma(max(1, math.floor(math.sqrt(length))), policy)

    def update(self, x) -> float:
        x = _f(x)
        return self.outer.update(2.0 * self.half.update(x) - self.full.update(x))


class Variance(_Windowed):
    """``ta.variance(source, length, biased)``: `Stdev` without the square root."""

    def __init__(self, length: int, biased: bool = True, policy: str = "skip"):
        super().__init__(length, policy)
        self.sma = Sma(length, policy)
        self.denom = length if biased else length - 1

    def update(self, x) -> float:
        x = _f(x)
        avg = self.sma.update(x)
        h = self._window(x)
        if h is None or math.isnan(avg):
            return NAN
        ss = 0.0
        for v in reversed(h):
            d = v - avg
            if abs(d) <= 1e-10:
                d = 0.0
            ss += d * d
        return ss / self.denom


class PercentileLinear(_Windowed):
    """``ta.percentile_linear_interpolation(source, length, percentage)``: linear interpolation
    between the sorted window values at the Hazen rank ``p/100 · n − ½`` (0-based, clamped to
    the window). **Observed** on BATS:NFLX 1D (2026-11-05): the Hazen rank matched and the
    ``p(n − 1)`` (numpy default) and ``p(n + 1) − 1`` ranks did not."""

    def __init__(self, length: int, percentage: float, policy: str = "skip"):
        if not 0 <= percentage <= 100:
            raise ValueError("percentage must be in [0, 100]")
        super().__init__(length, policy)
        self.p = percentage

    def _reduce(self, h: list[float]) -> float:
        s = sorted(h)
        r = min(max(self.p / 100.0 * len(s) - 0.5, 0.0), len(s) - 1.0)
        lo = math.floor(r)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (r - lo) * (s[hi] - s[lo])


class _Pivot:
    """``ta.pivothigh/pivotlow(source, leftbars, rightbars)``: on the bar ``rightbars`` after a
    pivot, the pivot's value; otherwise `na`. Tie rule, **observed** on BATS:NFLX 1D
    (2026-11-05): at least as extreme as every left bar and strictly beyond every right bar,
    so a flat top marks its *rightmost* bar. Any `na` in the window gives `na`."""

    sign = 1.0

    def __init__(self, left: int, right: int):
        if left < 1 or right < 1:
            raise ValueError("leftbars and rightbars must be >= 1")
        self.left, self.right = left, right
        self.w = _Window(left + right + 1)

    def update(self, x) -> float:
        self.w.push(_f(x))
        if not self.w.full:
            return NAN
        h = [self.sign * v for v in self.w.h]
        if any(math.isnan(v) for v in h):
            return NAN
        c = h[self.left]
        if all(c >= v for v in h[: self.left]) and all(c > v for v in h[self.left + 1 :]):
            return self.w.h[self.left]
        return NAN


class Pivothigh(_Pivot):
    sign = 1.0


class Pivotlow(_Pivot):
    sign = -1.0


class Fixnan:
    """``fixnan(source)``: the last non-`na` value (`na` before the first)."""

    def __init__(self):
        self.last = NAN

    def update(self, x) -> float:
        x = _f(x)
        if not math.isnan(x):
            self.last = x
        return self.last


class Cum:
    """``ta.cum(source)``: running total. `na` handling per policy: ``skip_na_out`` (default,
    **observed** on BATS:NFLX 1D 2026-11-05) returns `na` on `na` bars and carries the total
    across them; ``skip`` returns the total so far on an `na` bar; ``propagate`` stays `na`
    from the first `na` on."""

    def __init__(self, policy: str = "skip_na_out"):
        self.policy = _check(policy)
        self.total = NAN
        self.dead = False

    def update(self, x) -> float:
        x = _f(x)
        if math.isnan(x):
            if self.policy == "propagate":
                self.dead = True
            return NAN if self.policy != "skip" or self.dead else self.total
        if self.dead:
            return NAN
        self.total = x if math.isnan(self.total) else self.total + x
        return self.total


class Obv:
    """``ta.obv``: ``ta.cum(math.sign(ta.change(close)) * volume)``."""

    def __init__(self, policy: str = "skip"):
        self.cum = Cum(policy)
        self.prev = NAN

    def update(self, close, volume) -> float:
        c = _f(close)
        ch = c - self.prev
        self.prev = c
        sign = NAN if math.isnan(ch) else float((ch > 0) - (ch < 0))
        return self.cum.update(sign * _f(volume))


class Pvt:
    """``ta.pvt``: ``ta.cum(ta.change(close) / close[1] * volume)``."""

    def __init__(self, policy: str = "skip"):
        self.cum = Cum(policy)
        self.prev = NAN

    def update(self, close, volume) -> float:
        c = _f(close)
        v = (c - self.prev) / self.prev * _f(volume)
        self.prev = c
        return self.cum.update(v)
