"""Incremental, call-site-keyed ``ta.*`` built-ins for `pine.runtime` (PINE_PLAN.md §11 E2).

Each object holds the state of one call site and advances **only when called**, which is how
Pine evaluates built-ins inside conditional code (expected hypothesis ``per_call``, probed by
``tools/capture_semantics.pine``). Called on every bar, each one reproduces the vectorised
`pine.ta` function bit for bit (same arithmetic order); ``tests/test_runtime.py`` enforces it.

Only the ``propagate`` na policy is implemented (§10 A2 undecided).
"""

from __future__ import annotations

import math

import numpy as np

NAN = math.nan


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


class Sma:
    def __init__(self, length: int):
        self.w = _Window(length)

    def update(self, x) -> float:
        self.w.push(_f(x))
        if not self.w.full:
            return NAN
        n, s = self.w.n, 0.0
        for v in reversed(self.w.h):  # newest first, as pine.ta.sma
            s += v / n
        return s


class _Seeded:
    def __init__(self, length: int, alpha: float):
        self.sma = Sma(length)
        self.alpha = alpha
        self.prev = NAN

    def update(self, x) -> float:
        x = _f(x)
        seed = self.sma.update(x)
        a = self.alpha
        cur = seed if math.isnan(self.prev) else a * x + (1.0 - a) * self.prev
        self.prev = cur
        return cur


class Ema(_Seeded):
    def __init__(self, length: int):
        super().__init__(length, 2.0 / (length + 1))


class Rma(_Seeded):
    def __init__(self, length: int):
        super().__init__(length, 1.0 / length)


class Rsi:
    def __init__(self, length: int):
        self.up, self.down = Rma(length), Rma(length)
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
    def __init__(self, length: int, biased: bool = True):
        self.sma = Sma(length)
        self.w = _Window(length)
        self.denom = length if biased else length - 1

    def update(self, x) -> float:
        x = _f(x)
        avg = self.sma.update(x)
        self.w.push(x)
        if not self.w.full or math.isnan(avg):
            return NAN
        ss = 0.0
        for v in reversed(self.w.h):
            d = v - avg
            if abs(d) <= 1e-10:
                d = 0.0
            ss += d * d
        return math.sqrt(ss / self.denom)


class _Extreme:
    pick = staticmethod(np.max)

    def __init__(self, length: int):
        self.w = _Window(length)

    def update(self, x) -> float:
        self.w.push(_f(x))
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
        self.rma = Rma(length)

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
