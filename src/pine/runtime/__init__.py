"""Bar-by-bar Pine execution model (PINE_PLAN.md §11 E2).

A *script* is a Python function ``script(ctx)`` that `run` calls once per bar, as Pine does.
Ports written in Pine's shape inherit Pine's execution semantics:

- **History per execution.** ``ctx.series(key, init)`` declares a non-``var`` variable: each
  execution of the declaration starts a new value (``init``), ``s.set(v)`` is ``:=``, and
  ``s[n]`` is the value ``n`` executions ago. At top level that means ``n`` bars ago; inside
  a branch or function it means ``n`` executions ago — Pine's rule for local scopes.
- **``var``.** ``ctx.var(key, init)`` initialises once and carries its value into each
  execution.
- **Call sites.** ``ctx.call(key, fn, *args)`` runs ``fn(ctx, *args)`` in its own namespace, so
  every call site keeps separate state and history.
- **Built-ins advance only when called.** ``ctx.ta.ema(key, src, length)`` etc. are incremental
  objects keyed by call site (`pine.runtime.builtins`); inside an ``if`` they update only on
  bars where the branch runs.
- **Scalar rules** in `pine.runtime.scalar`.

These are the *expected* Pine semantics; each is probed by ``tools/capture_semantics.pine`` and
**unverified** until an export is checked with ``python -m pine.semantics``.

Keys are explicit strings rather than inferred from source positions: deterministic and
readable, at the cost of the port author choosing unique keys. Declaring the same key twice in
one bar raises `DuplicateKeyError` (catches copy-paste); inside Python loops use plain floats.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pine.bars import Bars
from pine.runtime import builtins as _b
from pine.runtime.scalar import NA
from pine.ta import DEFAULT_NA_POLICY

__all__ = ["NA", "Context", "DuplicateKeyError", "RunResult", "Series", "run"]


class DuplicateKeyError(RuntimeError):
    """A series, var, call or built-in key was used twice in the same scope on one bar."""


def _f(x) -> float:
    return NA if x is None else float(x)


class Series:
    """Values of one declaration, one per execution; ``s[0]`` is the current one."""

    __slots__ = ("_h", "_max")

    def __init__(self, max_bars_back: int):
        self._h: list[float] = []
        self._max = max_bars_back

    def _push(self, v) -> None:
        self._h.append(_f(v))
        if len(self._h) > 2 * self._max + 2:
            del self._h[: len(self._h) - self._max - 1]

    @property
    def value(self) -> float:
        return self._h[-1] if self._h else NA

    def set(self, v) -> Series:
        if not self._h:
            raise RuntimeError("series assigned before its first declaration")
        self._h[-1] = _f(v)
        return self

    def __getitem__(self, n: int) -> float:
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise IndexError(f"history offset must be a non-negative int, got {n!r}")
        if n > self._max:
            raise IndexError(f"offset {n} exceeds max_bars_back={self._max}")
        return self._h[-1 - n] if n < len(self._h) else NA

    def __float__(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"Series(value={self.value!r}, executions={len(self._h)})"


def _pol(na_policy, name: str) -> str:
    return DEFAULT_NA_POLICY[name] if na_policy is None else na_policy


class _TA:
    """``ctx.ta``: key-addressed incremental built-ins."""

    def __init__(self, ctx: Context):
        self._ctx = ctx

    def _state(self, key: str, cls, *args):
        full = self._ctx._claim("ta", key)
        node = self._ctx._nodes.get(full)
        if node is None:
            node = cls(*args)
            self._ctx._nodes[full] = node
        elif type(node) is not cls:
            raise DuplicateKeyError(
                f"key {key!r} used for {type(node).__name__} and {cls.__name__}"
            )
        return node

    # ``na_policy=None`` means the observed default (`pine.ta.DEFAULT_NA_POLICY`).
    def sma(self, key, source, length, na_policy=None):
        return self._state(key, _b.Sma, length, _pol(na_policy, "sma")).update(source)

    def ema(self, key, source, length, na_policy=None):
        return self._state(key, _b.Ema, length, _pol(na_policy, "ema")).update(source)

    def rma(self, key, source, length, na_policy=None):
        return self._state(key, _b.Rma, length, _pol(na_policy, "rma")).update(source)

    def rsi(self, key, source, length, na_policy=None):
        return self._state(key, _b.Rsi, length, _pol(na_policy, "rsi")).update(source)

    def stdev(self, key, source, length, biased=True, na_policy=None):
        return self._state(key, _b.Stdev, length, biased, _pol(na_policy, "stdev")).update(source)

    def highest(self, key, source, length, na_policy=None):
        return self._state(key, _b.Highest, length, _pol(na_policy, "highest")).update(source)

    def lowest(self, key, source, length, na_policy=None):
        return self._state(key, _b.Lowest, length, _pol(na_policy, "lowest")).update(source)

    def tr(self, handle_na: bool = False) -> float:
        """``ta.tr`` / ``ta.tr(handle_na)``: stateless, from the bar series."""
        c = self._ctx
        pc = c.close[1]
        h, lo = c.high.value, c.low.value
        if math.isnan(pc):
            return h - lo if handle_na else NA
        hl, hc, lc = h - lo, abs(h - pc), abs(lo - pc)
        return NA if math.isnan(hl + hc + lc) else max(hl, hc, lc)

    def atr(self, key, length):
        c = self._ctx
        return self._state(key, _b.Atr, length).update(
            c.high.value, c.low.value, c.high[1], c.close[1]
        )

    def crossover(self, key, a, b) -> bool:
        return self._state(key, _b.Crossover).update(a, b)

    def crossunder(self, key, a, b) -> bool:
        return self._state(key, _b.Crossunder).update(a, b)

    def change(self, key, source, length=1):
        return self._state(key, _b.Change, length).update(source)

    def barssince(self, key, condition):
        return self._state(key, _b.Barssince).update(condition)

    def valuewhen(self, key, condition, source, occurrence=0):
        return self._state(key, _b.Valuewhen, occurrence).update(condition, source)


class Context:
    def __init__(self, bars: Bars, max_bars_back: int = 5000):
        if max_bars_back < 1:
            raise ValueError("max_bars_back must be >= 1")
        self.bars = bars
        self.max_bars_back = max_bars_back
        mk = lambda: Series(max_bars_back)  # noqa: E731
        self.open, self.high, self.low, self.close = mk(), mk(), mk(), mk()
        self.volume, self.time, self.bar_index = mk(), mk(), mk()
        self.i = -1
        self.ta = _TA(self)
        self._nodes: dict[str, object] = {}
        self._used: set[str] = set()
        self._prefix = ""
        self._plots: dict[str, np.ndarray] = {}

    # -- bookkeeping ----------------------------------------------------------------------
    def _claim(self, kind: str, key: str) -> str:
        if not isinstance(key, str) or not key:
            raise TypeError("keys must be non-empty strings")
        full = f"{self._prefix}{kind}:{key}"
        if full in self._used:
            raise DuplicateKeyError(
                f"{kind} key {key!r} used twice in one bar (scope {self._prefix!r})"
            )
        self._used.add(full)
        return full

    def _advance(self, i: int) -> None:
        b = self.bars
        self.i = i
        self._used.clear()
        self._prefix = ""
        self.open._push(b.open[i])
        self.high._push(b.high[i])
        self.low._push(b.low[i])
        self.close._push(b.close[i])
        self.volume._push(b.volume[i])
        self.time._push(b.ts[i])
        self.bar_index._push(i)

    # -- declarations -----------------------------------------------------------------------
    def series(self, key: str, init=NA) -> Series:
        """Non-``var`` variable: a new value each execution, starting at ``init``."""
        full = self._claim("s", key)
        s = self._nodes.get(full)
        if s is None:
            s = Series(self.max_bars_back)
            self._nodes[full] = s
        s._push(init)
        return s

    def var(self, key: str, init=NA) -> Series:
        """``var``: ``init`` on the first execution, then the previous value carried over."""
        full = self._claim("v", key)
        s = self._nodes.get(full)
        if s is None:
            s = Series(self.max_bars_back)
            self._nodes[full] = s
            s._push(init)
        else:
            s._push(s.value)
        return s

    def call(self, key: str, fn: Callable, *args, **kwargs):
        """Run ``fn(ctx, *args, **kwargs)`` as its own call site (separate state/history)."""
        full = self._claim("c", key)
        outer = self._prefix
        self._prefix = full + "/"
        try:
            return fn(self, *args, **kwargs)
        finally:
            self._prefix = outer

    def plot(self, name: str, value) -> None:
        self._claim("p", name)
        col = self._plots.get(name)
        if col is None:
            col = np.full(len(self.bars), NA)
            self._plots[name] = col
        col[self.i] = _f(value)


@dataclass(frozen=True)
class RunResult:
    columns: dict[str, np.ndarray]
    bars: Bars

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]


def run(script: Callable[[Context], None], bars: Bars, max_bars_back: int = 5000) -> RunResult:
    """Execute ``script`` once per bar; return the plotted columns."""
    ctx = Context(bars, max_bars_back)
    for i in range(len(bars)):
        ctx._advance(i)
        script(ctx)
    for col in ctx._plots.values():
        col.setflags(write=False)
    return RunResult(dict(ctx._plots), bars)
