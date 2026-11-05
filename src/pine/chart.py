"""Pine's ``timeframe.*`` and ``barstate.*`` (PINE_PLAN.md §11 F).

Both describe the chart rather than compute from prices, so they live here rather than in
`pine.ta`; `pine.runtime` exposes them as ``ctx.timeframe`` and ``ctx.barstate``.

**Unverified until a ``tools/capture_builtins.pine`` export is checked** (each point below is a
probe group in `pine.builtin_capture`):

- ``timeframe.in_seconds("1M")``: a month counted as 2 628 003 s (30.4167 days, the value the
  Pine manual's examples show, as recalled); alternatives 30 days and the mean Gregorian month.
- ``timeframe.from_seconds``: **observed** (NFLX 1D, 2026-11-05) that 90 000 s gives "2D":
  a day or more is rounded *up* to whole days rather than spelled in minutes. Exact weeks
  and months, and counts under a day, are by inference.
- ``timeframe.change(tf)``: true on the first bar of each new ``tf`` period (``D``, ``W``,
  ``M``; weeks start on Monday), dated in the exchange time zone, and true on the chart's
  first bar. Intraday targets (``"60"``) need session anchoring (§11 G1) and are refused.
- ``barstate.*`` on the last bar depends on whether it was live when the export was taken
  (`BarState` ``live_last``): a live bar is realtime, not confirmed, not new (later ticks), and
  the bar before it is the last confirmed historical one.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np

from pine.bars import Bars, validate_timeframe

__all__ = [
    "MONTH_SECONDS",
    "Timeframe",
    "BarState",
    "in_seconds",
    "from_seconds",
    "period_ids",
    "change",
    "barstate",
]

MONTH_SECONDS = 2_628_003
_UNIT = {"S": 1, "": 60, "D": 86_400, "W": 604_800, "M": MONTH_SECONDS}
_RE = re.compile(r"^([1-9]\d*)?([SDWM]?)$")


def _parse(tf: str) -> tuple[int, str]:
    validate_timeframe(tf)
    m = _RE.match(tf)
    assert m is not None  # validate_timeframe accepted it
    mult, unit = m.groups()
    return int(mult or 1), unit


def in_seconds(tf: str, month_seconds: int = MONTH_SECONDS) -> int:
    """``timeframe.in_seconds(tf)``."""
    mult, unit = _parse(tf)
    return mult * (month_seconds if unit == "M" else _UNIT[unit])


def from_seconds(seconds: int) -> str:
    """``timeframe.from_seconds``. From a day up: whole months or weeks if exact, otherwise
    days rounded **up** (90 000 s -> "2D", observed). Under a day: minutes if exact, else
    seconds."""
    if isinstance(seconds, bool) or int(seconds) != seconds or seconds <= 0:
        raise ValueError("seconds must be a positive integer")
    s = int(seconds)
    if s >= _UNIT["D"]:
        for unit in ("M", "W"):
            if s % _UNIT[unit] == 0:
                return f"{s // _UNIT[unit]}{unit}"
        return f"{-(-s // _UNIT['D'])}D"
    return str(s // 60) if s % 60 == 0 else f"{s}S"


@dataclass(frozen=True)
class Timeframe:
    """The chart's ``timeframe.*`` constants, from a Pine timeframe string."""

    period: str
    """``timeframe.period`` in v6 spelling (``"1D"``, not v5's ``"D"``)."""
    multiplier: int
    isseconds: bool
    isminutes: bool
    isdaily: bool
    isweekly: bool
    ismonthly: bool

    @classmethod
    def of(cls, tf: str) -> Timeframe:
        mult, unit = _parse(tf)
        return cls(f"{mult}{unit}", mult, unit == "S", unit == "", unit == "D", unit == "W",
                   unit == "M")  # fmt: skip

    @property
    def isintraday(self) -> bool:
        return self.isseconds or self.isminutes

    @property
    def isdwm(self) -> bool:
        return self.isdaily or self.isweekly or self.ismonthly

    @property
    def isticks(self) -> bool:
        return False

    def in_seconds(self, tf: str | None = None) -> int:
        return in_seconds(self.period if tf is None else tf)


def _dates(bars: Bars) -> list[dt.date]:
    """Calendar date of each bar in the exchange time zone. Daily-or-higher bars stamped at
    exactly 00:00 UTC (a common export convention) are dated in UTC, since converting midnight
    UTC to a western zone would put them on the previous day."""
    ts = np.asarray(bars.ts)
    tf = Timeframe.of(bars.timeframe)
    if tf.isdwm and ts.size and not (ts % 86_400).any():
        return [dt.datetime.fromtimestamp(int(t), dt.UTC).date() for t in ts]
    tz = ZoneInfo(bars.session_tz)
    return [dt.datetime.fromtimestamp(int(t), tz).date() for t in ts]


def period_ids(bars: Bars, tf: str) -> list[tuple[int, ...]]:
    """An id per bar for the ``tf`` period containing it (``D``, ``W``, ``M`` only)."""
    mult, unit = _parse(tf)
    if unit not in ("D", "W", "M") or mult != 1:
        raise NotImplementedError(
            f"timeframe.change({tf!r}): only 1D, 1W and 1M are supported; intraday and multiple "
            "periods need session anchoring (PINE_PLAN.md §11 G1)"
        )
    dates = _dates(bars)
    if unit == "D":
        return [(d.toordinal(),) for d in dates]
    if unit == "W":
        return [tuple(d.isocalendar()[:2]) for d in dates]
    return [(d.year, d.month) for d in dates]


def change(bars: Bars, tf: str, first_bar: bool = True) -> np.ndarray:
    """``timeframe.change(tf)`` on every bar: the bar starts a new ``tf`` period."""
    ids = period_ids(bars, tf)
    out = np.array([i > 0 and ids[i] != ids[i - 1] for i in range(len(ids))], dtype=bool)
    if len(ids):
        out[0] = first_bar
    return out


@dataclass(frozen=True)
class BarState:
    """``barstate.*`` for bar ``i`` of ``n`` exported bars whose first has true ``bar_index``
    ``offset``. ``live_last``: the last bar was the live (realtime) bar at export time."""

    i: int
    n: int
    offset: int = 0
    live_last: bool = False

    @property
    def isfirst(self) -> bool:
        return self.offset + self.i == 0

    @property
    def islast(self) -> bool:
        return self.i == self.n - 1

    @property
    def isrealtime(self) -> bool:
        return self.live_last and self.islast

    @property
    def ishistory(self) -> bool:
        return not self.isrealtime

    @property
    def isconfirmed(self) -> bool:
        return not self.isrealtime

    @property
    def isnew(self) -> bool:
        return not self.isrealtime

    @property
    def islastconfirmedhistory(self) -> bool:
        return self.i == self.n - (2 if self.live_last else 1)


def barstate(n: int, offset: int = 0, live_last: bool = False) -> dict[str, np.ndarray]:
    """Every ``barstate.*`` flag over ``n`` bars, as boolean arrays."""
    names = ("isfirst", "islast", "ishistory", "isrealtime", "isnew", "isconfirmed",
             "islastconfirmedhistory")  # fmt: skip
    states = [BarState(i, n, offset, live_last) for i in range(n)]
    return {k: np.array([getattr(s, k) for s in states], dtype=bool) for k in names}
