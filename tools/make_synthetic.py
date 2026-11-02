"""Regenerate the synthetic fixtures in tests/fixtures/synthetic/.

Deterministic: a seeded random walk rounded to 2 decimals (prices) and integers (volume),
written in TradingView export layout. No real market data is used.

    uv run python tools/make_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from pine.bars import Bars
from pine.io import TradingViewExport, write_tradingview_csv

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "synthetic"


def random_walk_bars(n: int, seed: int, timeframe: str, start: int, step: int) -> Bars:
    rng = np.random.default_rng(seed)
    close = np.round(100.0 + np.cumsum(rng.normal(0.0, 1.0, n)), 2)
    open_ = np.round(np.concatenate([[100.0], close[:-1]]), 2)
    wick_hi = np.round(np.abs(rng.normal(0.0, 0.5, n)), 2)
    wick_lo = np.round(np.abs(rng.normal(0.0, 0.5, n)), 2)
    high = np.round(np.maximum(open_, close) + wick_hi, 2)
    low = np.round(np.minimum(open_, close) - wick_lo, 2)
    volume = rng.integers(1_000, 10_000, n).astype(np.float64)
    ts = start + step * np.arange(n, dtype=np.int64)
    return Bars(ts, open_, high, low, close, volume, timeframe=timeframe, session_tz="UTC")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bars = random_walk_bars(n=300, seed=20261103, timeframe="60", start=1_767_225_600, step=3600)
    # One placeholder plot column with leading na, to exercise na handling in the loader.
    plot = np.full(len(bars), np.nan)
    plot[13:] = bars.close[13:]
    exp = TradingViewExport(
        bars=bars,
        columns={"Plot": plot},
        header=("time", "open", "high", "low", "close", "Volume", "Plot"),
        time_format="unix",
    )
    write_tradingview_csv(exp, OUT / "random_walk_60.csv")

    # Opens gap away from the previous close, so true range differs from high - low.
    # Without gaps an ATR port that ignores close[1] is indistinguishable from a correct one.
    rng = np.random.default_rng(42)
    g = random_walk_bars(n=300, seed=99, timeframe="60", start=1_767_225_600, step=3600)
    gap_open = np.round(np.concatenate([[100.0], g.close[:-1]]) + rng.normal(0.0, 1.5, len(g)), 2)
    wick = np.round(np.abs(rng.normal(0.0, 0.3, (2, len(g)))), 2)
    gap_high = np.round(np.maximum(gap_open, g.close) + wick[0], 2)
    gap_low = np.round(np.minimum(gap_open, g.close) - wick[1], 2)
    write_tradingview_csv(
        TradingViewExport(
            bars=Bars(g.ts, gap_open, gap_high, gap_low, g.close, g.volume, timeframe="60"),
            columns={},
            header=("time", "open", "high", "low", "close", "Volume"),
            time_format="unix",
        ),
        OUT / "random_walk_gaps_60.csv",
    )

    daily = random_walk_bars(n=60, seed=7, timeframe="1D", start=1_767_243_600, step=86_400)
    # 2026-01-01 00:00 America/New_York; 60 days stays clear of the March DST change.
    exp_iso = TradingViewExport(
        bars=Bars(
            daily.ts,
            daily.open,
            daily.high,
            daily.low,
            daily.close,
            daily.volume,
            timeframe="1D",
            session_tz="America/New_York",
        ),
        columns={},
        header=("time", "open", "high", "low", "close", "Volume"),
        time_format="iso",
    )
    write_tradingview_csv(exp_iso, OUT / "random_walk_1d_iso.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
