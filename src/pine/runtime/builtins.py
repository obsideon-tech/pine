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
