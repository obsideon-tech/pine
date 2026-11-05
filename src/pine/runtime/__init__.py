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

import copy
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from pine import chart as _chart
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
        ctx = self._ctx
        node = ctx._nodes.get(full)
        if node is None:
            node = cls(*args)
            ctx._nodes[full] = node
        elif type(node) is not cls:
            raise DuplicateKeyError(
                f"key {key!r} used for {type(node).__name__} and {cls.__name__}"
            )
        if ctx._loop_depth:
            # Inside a loop the site runs several times per bar; its history keeps only the
            # bar's last call (observed). Each call therefore starts from the state the bar
            # began with: restore that snapshot on every call after the first.
            seen = ctx._touched.get(full)
            if seen is not None and seen[0] == ctx.i:
                node.__dict__.clear()
                node.__dict__.update(copy.deepcopy(seen[1]))
            else:
                ctx._touched[full] = (ctx.i, copy.deepcopy(node.__dict__))
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

    # -- batch F (unverified until a tools/capture_builtins.pine export is checked) ----------
    def sum(self, key, source, length, na_policy=None):
        """``math.sum`` (stateful like the ``ta.*`` built-ins, so it lives here)."""
        return self._state(key, _b.Sum, length, _pol(na_policy, "sum")).update(source)

    def wma(self, key, source, length, na_policy=None):
        return self._state(key, _b.Wma, length, _pol(na_policy, "wma")).update(source)

    def hma(self, key, source, length, na_policy=None):
        return self._state(key, _b.Hma, length, _pol(na_policy, "hma")).update(source)

    def variance(self, key, source, length, biased=True, na_policy=None):
        pol = _pol(na_policy, "variance")
        return self._state(key, _b.Variance, length, biased, pol).update(source)

    def percentile_linear_interpolation(self, key, source, length, percentage, na_policy=None):
        pol = _pol(na_policy, "percentile_linear_interpolation")
        return self._state(key, _b.PercentileLinear, length, percentage, pol).update(source)

    def pivothigh(self, key, source, leftbars, rightbars):
        return self._state(key, _b.Pivothigh, leftbars, rightbars).update(source)

    def pivotlow(self, key, source, leftbars, rightbars):
        return self._state(key, _b.Pivotlow, leftbars, rightbars).update(source)

    def fixnan(self, key, source):
        return self._state(key, _b.Fixnan).update(source)

    def cum(self, key, source, na_policy=None):
        return self._state(key, _b.Cum, _pol(na_policy, "cum")).update(source)

    def obv(self, key):
        c = self._ctx
        return self._state(key, _b.Obv, DEFAULT_NA_POLICY["cum"]).update(
            c.close.value, c.volume.value
        )

    def pvt(self, key):
        c = self._ctx
        return self._state(key, _b.Pvt, DEFAULT_NA_POLICY["cum"]).update(
            c.close.value, c.volume.value
        )


class _TimeframeView:
    """``ctx.timeframe``: the chart's `pine.chart.Timeframe` constants plus ``change``."""

    def __init__(self, ctx: Context):
        self._ctx = ctx
        self._tf = _chart.Timeframe.of(ctx.bars.timeframe)
        self._ids: dict[str, list] = {}

    def __getattr__(self, name):
        return getattr(self._tf, name)

    def in_seconds(self, tf: str | None = None) -> int:
        return self._tf.in_seconds(tf)

    @staticmethod
    def from_seconds(seconds: int) -> str:
        return _chart.from_seconds(seconds)

    def change(self, tf: str) -> bool:
        """``timeframe.change(tf)`` on the current bar (stateless: it compares bar times)."""
        ids = self._ids.get(tf)
        if ids is None:
            ids = self._ids[tf] = _chart.period_ids(self._ctx.bars, tf)
        i = self._ctx.i
        # False on the first bar (observed); on the first row of a truncated export the
        # previous bar is unknown, and False is returned there too.
        return i > 0 and ids[i] != ids[i - 1]


class Context:
    def __init__(
        self,
        bars: Bars,
        max_bars_back: int = 5000,
        bar_offset: int = 0,
        live_last_bar: bool = False,
    ):
        if max_bars_back < 1:
            raise ValueError("max_bars_back must be >= 1")
        if bar_offset < 0:
            raise ValueError("bar_offset must be >= 0")
        self.bars = bars
        self.bar_offset = bar_offset
        self.strategy = None  # a pine.strategy.Broker when run(..., strategy=...)
        self.max_bars_back = max_bars_back
        mk = lambda: Series(max_bars_back)  # noqa: E731
        self.open, self.high, self.low, self.close = mk(), mk(), mk(), mk()
        self.volume, self.time, self.bar_index = mk(), mk(), mk()
        self.i = -1
        self.ta = _TA(self)
        self.timeframe = _TimeframeView(self)
        self.live_last_bar = live_last_bar
        self._nodes: dict[str, object] = {}
        self._used: set[str] = set()
        self._prefix = ""
        self._loop_depth = 0
        self._touched: dict[str, tuple[int, dict]] = {}  # loop sites: bar, start-of-bar state
        self._declared: dict[str, int] = {}  # series/var key -> last bar declared
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
        self.bar_index._push(self.bar_offset + i)

    @property
    def barstate(self) -> _chart.BarState:
        """``barstate.*`` of the current bar (`pine.chart.BarState`)."""
        return _chart.BarState(self.i, len(self.bars), self.bar_offset, self.live_last_bar)

    # -- declarations -----------------------------------------------------------------------
    def series(self, key: str, init=NA) -> Series:
        """Non-``var`` variable: a new value each execution, starting at ``init``."""
        full = self._claim("s", key)
        s = self._nodes.get(full)
        if s is None:
            s = Series(self.max_bars_back)
            self._nodes[full] = s
        if self._loop_depth and self._declared.get(full) == self.i:
            s._h[-1] = _f(init)  # repeated in a loop: one history entry per bar (the last)
        else:
            s._push(init)
        self._declared[full] = self.i
        return s

    def var(self, key: str, init=NA) -> Series:
        """``var``: ``init`` on the first execution, then the previous value carried over."""
        full = self._claim("v", key)
        s = self._nodes.get(full)
        if s is None:
            s = Series(self.max_bars_back)
            self._nodes[full] = s
            s._push(init)
        elif not (self._loop_depth and self._declared.get(full) == self.i):
            s._push(s.value)
        self._declared[full] = self.i
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

    def loop(self, key: str, iterable: Iterable):
        """Pine ``for`` loop body: ``for i in ctx.loop("k", range(1, 4)): ...``.

        Inside the body, keys may repeat from one iteration to the next. Every call site in
        the body has one state, and its history keeps **one entry per bar: the value from the
        bar's last iteration**. So each call sees the history as of the previous bar plus its
        own current input: ``math.sum(x, 2)`` in the loop is ``x + x_last[1]``, and ``s[1]``
        of a series declared in the body is the previous bar's last value, whatever the
        iteration. **Observed** (hypothesis ``per_bar_last`` of the ``loop_*`` groups in
        `pine.semantics`, BATS:NFLX 1W, 2026-11-05); ``per_call`` and ``per_iteration`` did
        not match. Leaving the loop with ``break`` is fine (the generator is closed)."""
        full = self._claim("l", key)
        outer = self._prefix
        inner = full + "/"
        self._loop_depth += 1
        try:
            for item in iterable:
                self._used = {u for u in self._used if not u.startswith(inner)}
                self._prefix = inner
                yield item
                self._prefix = outer
        finally:
            self._prefix = outer
            self._loop_depth -= 1

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
    broker: object | None = None
    """The `pine.strategy.Broker` (trades, final state) when run with ``strategy=``."""

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]


def run(
    script: Callable[[Context], None],
    bars: Bars,
    max_bars_back: int = 5000,
    *,
    strategy=None,
    bar_offset: int = 0,
    live_last_bar: bool = False,
) -> RunResult:
    """Execute ``script`` once per bar; return the plotted columns.

    ``strategy``: a `pine.strategy.StrategyConfig` to give the script ``ctx.strategy`` (a
    broker emulator; orders fill around each script execution per the config).
    ``bar_offset``: true ``bar_index`` of the first bar (exports that start mid-history).
    ``live_last_bar``: the last bar was live when exported (``barstate.isrealtime`` there)."""
    ctx = Context(bars, max_bars_back, bar_offset, live_last_bar)
    broker = None
    if strategy is not None:
        from pine.strategy import Broker

        broker = Broker(strategy)
        ctx.strategy = broker
    o, h, lo, c = bars.open, bars.high, bars.low, bars.close
    for i in range(len(bars)):
        ctx._advance(i)
        if broker is not None:
            broker._before_script(i, o[i], h[i], lo[i], c[i])
        script(ctx)
        if broker is not None:
            broker._after_script(i, c[i])
    for col in ctx._plots.values():
        col.setflags(write=False)
    return RunResult(dict(ctx._plots), bars, broker)
