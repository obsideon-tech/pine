"""Read and write TradingView chart-export CSV files.

Assumed layout (from "Export chart data…"; **unverified against a current export**, see
PINE_PLAN.md §8): a header row, then one row per bar. The first five columns are
``time,open,high,low,close`` (matched case-insensitively). An optional ``Volume`` column and
any number of plotted-indicator columns follow. Plot titles may repeat; repeats are
disambiguated as ``"Title #2"``, ``"Title #3"``, … in `TradingViewExport.columns`, while the
original header is kept for writing back.

``time`` is either integer Unix seconds or an ISO 8601 timestamp with a UTC offset; both are
accepted. Missing values (``""``, ``NaN``, ``nan``, ``na``) load as NaN.

The CSV carries no timeframe or time zone, so the caller supplies them (normally from the
fixture manifest).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np

from pine.bars import Bars

__all__ = [
    "TradingViewExport",
    "TradingViewFormatError",
    "load_tradingview_csv",
    "parse_tradingview_csv",
    "dump_tradingview_csv",
    "write_tradingview_csv",
]

_NA_TOKENS = frozenset({"", "nan", "na", "n/a"})
_REQUIRED = ("time", "open", "high", "low", "close")

TimeFormat = Literal["unix", "iso"]


class TradingViewFormatError(ValueError):
    """Raised when a file does not look like a TradingView chart export."""


@dataclass(frozen=True, eq=False)
class TradingViewExport:
    bars: Bars
    columns: dict[str, np.ndarray]
    """Plotted-indicator columns by disambiguated title, in file order."""
    header: tuple[str, ...]
    """Header exactly as read, used when writing back."""
    time_format: TimeFormat
    na_token: str = "NaN"
    observed_decimals: dict[str, int] = field(default_factory=dict)
    """Maximum digits after the decimal point seen per column (excluding time).

    A lower bound on export precision; feeds the ``atol`` floor in PINE_PLAN.md §5.1.
    """


def _is_na(token: str) -> bool:
    return token.strip().lower() in _NA_TOKENS


def _parse_float(token: str, row: int, col: str) -> float:
    if _is_na(token):
        return float("nan")
    try:
        return float(token)
    except ValueError as exc:
        raise TradingViewFormatError(
            f"row {row}, column {col!r}: {token!r} is not a number"
        ) from exc


def _decimals(token: str) -> int:
    t = token.strip()
    if _is_na(t) or "e" in t.lower() or "." not in t:
        return 0
    return len(t.split(".", 1)[1])


def _parse_time(token: str, row: int) -> tuple[int, TimeFormat]:
    t = token.strip()
    if t.lstrip("-").isdigit():
        return int(t), "unix"
    try:
        dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TradingViewFormatError(f"row {row}: time {token!r} is neither Unix nor ISO") from exc
    if dt.tzinfo is None:
        raise TradingViewFormatError(
            f"row {row}: ISO time {token!r} has no UTC offset; refusing to guess a zone"
        )
    return int(dt.timestamp()), "iso"


def _disambiguate(names: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name} #{seen[name]}")
    return out


def parse_tradingview_csv(
    text: str,
    *,
    timeframe: str,
    session_tz: str = "UTC",
    symbol: str | None = None,
) -> TradingViewExport:
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if r]  # tolerate a trailing blank line
    if not rows:
        raise TradingViewFormatError("empty file")
    header = tuple(rows[0])
    lower = [h.strip().lower() for h in header]
    if tuple(lower[:5]) != _REQUIRED:
        raise TradingViewFormatError(
            f"header must start with {', '.join(_REQUIRED)}; got {', '.join(header[:5])}"
        )
    has_volume = len(lower) > 5 and lower[5] == "volume"
    plot_start = 6 if has_volume else 5
    plot_names = _disambiguate(h.strip() for h in header[plot_start:])

    body = rows[1:]
    width = len(header)
    ts = np.empty(len(body), dtype=np.int64)
    values = np.empty((len(body), width - 1), dtype=np.float64)
    decimals = [0] * (width - 1)
    formats: set[TimeFormat] = set()
    na_token = None

    for i, r in enumerate(body, start=2):
        if len(r) != width:
            raise TradingViewFormatError(f"row {i}: {len(r)} fields, header has {width}")
        t, fmt = _parse_time(r[0], i)
        formats.add(fmt)
        ts[i - 2] = t
        for j, tok in enumerate(r[1:]):
            name = header[j + 1]
            values[i - 2, j] = _parse_float(tok, i, name)
            decimals[j] = max(decimals[j], _decimals(tok))
            if na_token is None and _is_na(tok):
                na_token = tok.strip()

    if len(formats) > 1:
        raise TradingViewFormatError("time column mixes Unix and ISO formats")
    time_format: TimeFormat = formats.pop() if formats else "unix"

    volume = values[:, 4] if has_volume else np.full(len(body), np.nan)
    bars = Bars(
        ts=ts,
        open=values[:, 0],
        high=values[:, 1],
        low=values[:, 2],
        close=values[:, 3],
        volume=volume,
        timeframe=timeframe,
        session_tz=session_tz,
        symbol=symbol,
    )
    columns = {}
    for k, name in enumerate(plot_names):
        arr = values[:, plot_start - 1 + k].copy()
        arr.setflags(write=False)
        columns[name] = arr
    observed = {name: decimals[j] for j, name in enumerate([*lower[1:plot_start], *plot_names])}
    return TradingViewExport(
        bars=bars,
        columns=columns,
        header=header,
        time_format=time_format,
        na_token="NaN" if na_token is None else na_token,
        observed_decimals=observed,
    )


def load_tradingview_csv(
    path: str | Path,
    *,
    timeframe: str,
    session_tz: str = "UTC",
    symbol: str | None = None,
) -> TradingViewExport:
    text = Path(path).read_text(encoding="utf-8-sig")  # tolerate a BOM
    return parse_tradingview_csv(text, timeframe=timeframe, session_tz=session_tz, symbol=symbol)


def _fmt_float(x: float, na_token: str) -> str:
    if np.isnan(x):
        return na_token
    # Shortest string that round-trips to the same float64; no exponent, no trailing ".0".
    return np.format_float_positional(x, unique=True, trim="-")


def _fmt_time(t: int, time_format: TimeFormat, tz: str) -> str:
    if time_format == "unix":
        return str(int(t))
    return datetime.fromtimestamp(int(t), tz=UTC).astimezone(ZoneInfo(tz)).isoformat()


def dump_tradingview_csv(export: TradingViewExport) -> str:
    b = export.bars
    has_volume = len(export.header) > 5 and export.header[5].strip().lower() == "volume"
    cols = [b.open, b.high, b.low, b.close]
    if has_volume:
        cols.append(b.volume)
    cols.extend(export.columns.values())
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(export.header)
    for i in range(len(b)):
        w.writerow(
            [_fmt_time(b.ts[i], export.time_format, b.session_tz)]
            + [_fmt_float(float(c[i]), export.na_token) for c in cols]
        )
    return buf.getvalue()


def write_tradingview_csv(export: TradingViewExport, path: str | Path) -> None:
    Path(path).write_text(dump_tradingview_csv(export), encoding="utf-8")
