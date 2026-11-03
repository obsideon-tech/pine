"""Strategy broker emulator for `pine.runtime` (PINE_PLAN.md §13).

Emulates the part of TradingView's strategy engine needed for the first parity increment:
market entries, closes and reversals, and ``strategy.exit`` stop/limit brackets, with fixed
quantity, no commission, no slippage, ``pyramiding = 1``. Per-bar outputs mirror Pine's
``strategy.position_size``, ``position_avg_price``, ``netprofit``, ``openprofit``,
``equity`` and ``closedtrades``.

Execution model (each rule is a probed hypothesis; the defaults are the *expected*
TradingView behaviour and are **unverified** until ``tools/capture_strategy.pine`` is
exported and checked with ``python -m pine.capture check``):

- ``fill_timing="next_open"``: orders placed while the script runs on bar ``t`` (at its close)
  fill at the open of bar ``t + 1`` (``process_orders_on_close = false``). Alternative:
  ``"same_close"``.
- ``path="nearest_extreme_first"``: within a bar, price is assumed to move
  open → the extreme nearer the open → the other extreme → close (ties: high first).
  Alternative: ``"high_first"`` (always open → high → low → close).
- ``gap_fill="open"``: a stop or limit already beyond its price at the open fills at the
  open. Alternative: ``"order_price"``.
- ``exit_same_bar=True``: a bracket can fill on the bar where its entry filled (after the
  open). Alternative: ``False`` (active from the next bar).
- ``price_rounding="tick"`` (with ``mintick`` set): the emulator works on OHLC rounded to the
  nearest tick, so market orders fill at the rounded open. Alternative: ``"none"``.
  **Observed** on BINANCE:DOGEUSDT 1D (2026-11-03): fills at 0.00387 for an open of 0.0038741
  with mintick 0.00001; an exit only reachable through the rounded high.
- ``order_rounding="conservative"`` (with ``mintick``): limit and stop prices are rounded to
  the tick away from a better fill — sell limit / buy stop up, buy limit / sell stop down.
  Alternative: ``"nearest"``. Observed once (a sell limit 0.0029514 filled at 0.00296); the
  stop direction is inferred by symmetry, **unverified**.

A ``strategy.entry`` in the direction opposite to the open position closes it and opens the
new one at the same fill (a reversal). ``P&L = (exit - entry) * qty`` for longs (point value 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

__all__ = ["Broker", "StrategyConfig", "Trade"]

NAN = math.nan
Direction = Literal["long", "short"]


@dataclass(frozen=True)
class StrategyConfig:
    initial_capital: float = 100_000.0
    qty: float = 1.0
    fill_timing: Literal["next_open", "same_close"] = "next_open"
    path: Literal["nearest_extreme_first", "high_first"] = "nearest_extreme_first"
    gap_fill: Literal["open", "order_price"] = "open"
    exit_same_bar: bool = True
    mintick: float | None = None
    price_rounding: Literal["tick", "none"] = "tick"
    order_rounding: Literal["conservative", "nearest"] = "conservative"

    def __post_init__(self) -> None:
        if self.mintick is not None and not self.mintick > 0:
            raise ValueError("mintick must be positive")


@dataclass(frozen=True)
class Trade:
    entry_id: str
    direction: Direction
    qty: float
    entry_bar: int
    entry_price: float
    exit_bar: int
    exit_price: float

    @property
    def profit(self) -> float:
        sign = 1.0 if self.direction == "long" else -1.0
        return sign * (self.exit_price - self.entry_price) * self.qty


@dataclass
class _Bracket:
    exit_id: str
    from_entry: str
    stop: float
    limit: float


@dataclass
class Broker:
    config: StrategyConfig = field(default_factory=StrategyConfig)
    size: float = 0.0  # signed: + long, - short
    avg_price: float = NAN
    entry_id: str | None = None
    entry_bar: int = -1
    netprofit: float = 0.0
    trades: list[Trade] = field(default_factory=list)
    _pending: list[tuple] = field(default_factory=list)
    _brackets: dict[str, _Bracket] = field(default_factory=dict)
    _close: float = NAN
    _bar: int = -1

    # ---------------------------------------------------------------- script-facing API
    def entry(self, entry_id: str, direction: Direction) -> None:
        if direction not in ("long", "short"):
            raise ValueError("direction must be 'long' or 'short'")
        self._pending.append(("entry", entry_id, direction))

    def close(self, entry_id: str) -> None:
        self._pending.append(("close", entry_id))

    def close_all(self) -> None:
        self._pending.append(("close_all",))

    def exit(self, exit_id: str, from_entry: str, stop: float = NAN, limit: float = NAN) -> None:
        """Stop/limit bracket for the position opened by ``from_entry``; it persists until it
        fills or that position closes."""
        self._brackets[exit_id] = _Bracket(exit_id, from_entry, float(stop), float(limit))

    @property
    def position_size(self) -> float:
        return self.size

    @property
    def position_avg_price(self) -> float:
        return self.avg_price

    @property
    def openprofit(self) -> float:
        if self.size == 0:
            return 0.0
        return (self._close - self.avg_price) * self.size

    @property
    def equity(self) -> float:
        return self.config.initial_capital + self.netprofit + self.openprofit

    @property
    def closedtrades(self) -> int:
        return len(self.trades)

    # ---------------------------------------------------------------- engine
    def _realize(self, bar: int, price: float) -> None:
        direction: Direction = "long" if self.size > 0 else "short"
        t = Trade(self.entry_id or "", direction, abs(self.size), self.entry_bar,
                  self.avg_price, bar, price)  # fmt: skip
        self.trades.append(t)
        self.netprofit += t.profit
        self.size, self.avg_price, self.entry_id = 0.0, NAN, None
        self._brackets.clear()

    def _fill(self, order: tuple, bar: int, price: float) -> None:
        kind = order[0]
        if kind == "entry":
            _, eid, direction = order
            sign = 1.0 if direction == "long" else -1.0
            if self.size * sign > 0:
                return  # pyramiding = 1: already in that direction
            if self.size != 0:
                self._realize(bar, price)  # reversal
            self.size = sign * self.config.qty
            self.avg_price, self.entry_id, self.entry_bar = price, eid, bar
        elif kind == "close":
            if self.size != 0 and self.entry_id == order[1]:
                self._realize(bar, price)
        elif kind == "close_all":
            if self.size != 0:
                self._realize(bar, price)

    def _fill_pending(self, bar: int, price: float) -> None:
        orders, self._pending = self._pending, []
        for o in orders:
            self._fill(o, bar, price)

    def _round(self, x: float, mode: str = "nearest") -> float:
        t = self.config.mintick
        if t is None or math.isnan(x):
            return x
        k = x / t
        if mode == "nearest":
            n = math.floor(k + 0.5 + 1e-9)
        elif mode == "up":
            n = math.ceil(k - 1e-9)
        else:
            n = math.floor(k + 1e-9)
        inv = 1.0 / t
        # Divide by an exact integer scale when the tick is 10^-d, so 0.00387 prints as such.
        return n / round(inv) if abs(inv - round(inv)) < 1e-9 * inv else n * t

    def _price(self, x: float) -> float:
        return self._round(x) if self.config.price_rounding == "tick" else x

    def _order_level(self, level: float, kind: str, long_position: bool) -> float:
        """Rounded limit/stop price for an exit of a long (sell) or short (buy) position."""
        if self.config.mintick is None or math.isnan(level):
            return level
        if self.config.order_rounding == "nearest":
            return self._round(level)
        sell = long_position
        up = (kind == "limit" and sell) or (kind == "stop" and not sell)
        return self._round(level, "up" if up else "down")

    def _path(self, o: float, h: float, lo: float, c: float) -> list[float]:
        if self.config.path == "high_first" or abs(h - o) <= abs(o - lo):
            return [o, h, lo, c]
        return [o, lo, h, c]

    def _check_brackets(self, bar: int, o: float, h: float, lo: float, c: float) -> None:
        if self.size == 0:
            self._brackets.clear()
            return
        if not self.config.exit_same_bar and bar == self.entry_bar:
            return
        long = self.size > 0
        live = [
            _Bracket(
                b.exit_id,
                b.from_entry,
                self._order_level(b.stop, "stop", long),
                self._order_level(b.limit, "limit", long),
            )
            for b in self._brackets.values()
            if b.from_entry == self.entry_id
        ]
        if not live:
            return
        gap = self.config.gap_fill == "open"

        def stop_hit(p: float, s: float) -> bool:
            return p <= s if long else p >= s

        def limit_hit(p: float, lim: float) -> bool:
            return p >= lim if long else p <= lim

        # Already beyond a level at the open.
        for b in live:
            if not math.isnan(b.stop) and stop_hit(o, b.stop):
                self._realize(bar, o if gap else b.stop)
                return
            if not math.isnan(b.limit) and limit_hit(o, b.limit):
                self._realize(bar, o if gap else b.limit)
                return
        pts = self._path(o, h, lo, c)
        for p0, p1 in zip(pts, pts[1:], strict=False):
            hits = []
            for b in live:
                for level in (b.stop, b.limit):
                    if math.isnan(level):
                        continue
                    if min(p0, p1) <= level <= max(p0, p1) and level != p0:
                        hits.append(level)
            if hits:
                # The first level reached moving from p0 towards p1.
                first = min(hits, key=lambda v: abs(v - p0))
                self._realize(bar, first)
                return

    # Called by pine.runtime.run around each script execution.
    def _before_script(self, bar: int, o: float, h: float, lo: float, c: float) -> None:
        o, h, lo, c = (self._price(v) for v in (o, h, lo, c))
        self._bar, self._close = bar, c
        if self.config.fill_timing == "next_open":
            self._fill_pending(bar, o)
        self._check_brackets(bar, o, h, lo, c)

    def _after_script(self, bar: int, c: float) -> None:
        if self.config.fill_timing == "same_close":
            self._fill_pending(bar, self._price(c))
