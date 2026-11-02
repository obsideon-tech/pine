"""The `Bars` container: one instrument, one timeframe, aligned OHLCV arrays.

Conventions (PINE_PLAN.md §7.1):

- ``ts`` is int64 Unix seconds, UTC, marking the bar's **open** time, as Pine's ``time`` does.
- Prices and volume are float64. Volume may be NaN (some symbols have none); OHLC may not.
- ``timeframe`` uses Pine's ``timeframe.period`` spelling ("1", "60", "1D", "1W", "1M", "30S").
- ``session_tz`` is an IANA zone name (Pine's ``syminfo.timezone``), used for session and
  calendar boundaries. It does not change ``ts``, which is always UTC.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

__all__ = ["Bars", "BarsError", "validate_timeframe", "validate_timezone"]

_TIMEFRAME_RE = re.compile(r"^(?:[1-9]\d*)?[SDWM]$|^[1-9]\d*$")


class BarsError(ValueError):
    """Raised when bar data violates the `Bars` invariants."""


def validate_timeframe(timeframe: str) -> str:
    if not isinstance(timeframe, str) or not _TIMEFRAME_RE.match(timeframe):
        raise BarsError(
            f"timeframe {timeframe!r} is not a Pine timeframe string "
            "(e.g. '1', '60', '1D', 'D', '1W', '1M', '30S')"
        )
    return timeframe


def validate_timezone(tz: str) -> str:
    try:
        ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as exc:
        raise BarsError(f"session_tz {tz!r} is not an IANA time zone") from exc
    return tz


def _as_float(name: str, values: object) -> np.ndarray:
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.ndim != 1:
        raise BarsError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return arr


@dataclass(frozen=True, eq=False)
class Bars:
    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    timeframe: str
    session_tz: str = "UTC"
    symbol: str | None = field(default=None)

    def __post_init__(self) -> None:
        ts = np.asarray(self.ts)
        if ts.ndim != 1:
            raise BarsError(f"ts must be one-dimensional, got shape {ts.shape}")
        if ts.size and not np.issubdtype(ts.dtype, np.integer):
            raise BarsError(f"ts must be integer Unix seconds, got dtype {ts.dtype}")
        ts = ts.astype(np.int64, copy=True)

        cols = {
            name: _as_float(name, getattr(self, name))
            for name in ("open", "high", "low", "close", "volume")
        }
        n = ts.size
        for name, arr in cols.items():
            if arr.size != n:
                raise BarsError(f"{name} has {arr.size} values, ts has {n}")

        if n > 1 and not np.all(np.diff(ts) > 0):
            bad = int(np.argmax(np.diff(ts) <= 0)) + 1
            raise BarsError(f"ts must be strictly increasing; violated at index {bad}")

        for name in ("open", "high", "low", "close"):
            if not np.all(np.isfinite(cols[name])):
                bad = int(np.argmax(~np.isfinite(cols[name])))
                raise BarsError(f"{name} must be finite; non-finite at index {bad}")

        o, h, lo, c = cols["open"], cols["high"], cols["low"], cols["close"]
        bad_hl = (h < np.maximum(o, c)) | (lo > np.minimum(o, c)) | (lo > h)
        if np.any(bad_hl):
            raise BarsError(
                f"high/low inconsistent with open/close at index {int(np.argmax(bad_hl))}"
            )
        v = cols["volume"]
        if np.any(v[~np.isnan(v)] < 0):
            raise BarsError("volume must be non-negative or NaN")

        validate_timeframe(self.timeframe)
        validate_timezone(self.session_tz)

        for arr in (ts, *cols.values()):
            arr.setflags(write=False)
        object.__setattr__(self, "ts", ts)
        for name, arr in cols.items():
            object.__setattr__(self, name, arr)

    def __len__(self) -> int:
        return int(self.ts.size)

    def equals(self, other: Bars) -> bool:
        """Exact equality, NaN == NaN in volume. Used by round-trip tests."""
        return (
            self.timeframe == other.timeframe
            and self.session_tz == other.session_tz
            and self.symbol == other.symbol
            and np.array_equal(self.ts, other.ts)
            and all(
                np.array_equal(getattr(self, n), getattr(other, n), equal_nan=True)
                for n in ("open", "high", "low", "close", "volume")
            )
        )
