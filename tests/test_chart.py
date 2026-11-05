"""timeframe.* and barstate.* (pine.chart, ctx.timeframe, ctx.barstate)."""

import datetime as dt

import numpy as np
import pytest

from pine import Bars, chart
from pine.builtin_capture import decide, runtime_capture
from pine.io import TradingViewExport, load_tradingview_csv, write_tradingview_csv
from pine.runtime import run


def _bars(ts, tf="1D", tz="UTC"):
    n = len(ts)
    x = np.full(n, 10.0)
    return Bars(np.asarray(ts, dtype=np.int64), x, x + 1, x - 1, x, np.ones(n), tf, tz)


def _ts(*dates):
    """Midnight UTC of each date (the daily-export convention)."""
    return [int(dt.datetime(*d, tzinfo=dt.UTC).timestamp()) for d in dates]


@pytest.mark.parametrize(
    "tf, secs",
    [("1", 60), ("60", 3600), ("240", 14400), ("30S", 30), ("D", 86400), ("1D", 86400),
     ("3D", 259200), ("1W", 604800), ("1M", 2628003), ("12M", 12 * 2628003)],
)  # fmt: skip
def test_in_seconds(tf, secs):
    assert chart.in_seconds(tf) == secs


@pytest.mark.parametrize(
    "secs, tf",
    [(60, "1"), (5400, "90"), (86400, "1D"), (604800, "1W"), (2628003, "1M"), (45, "45S")],
)  # fmt: skip
def test_from_seconds_round_trips(secs, tf):
    assert chart.from_seconds(secs) == tf
    assert chart.in_seconds(chart.from_seconds(secs)) == secs


def test_from_seconds_rounds_up_to_days():
    # Observed on TradingView (NFLX 1D, 2026-11-05): 25 hours gives "2D", not "1500".
    assert chart.from_seconds(90_000) == "2D"
    assert chart.from_seconds(86_401) == "2D" and chart.from_seconds(8 * 86_400) == "8D"


def test_from_seconds_and_timeframe_validation():
    for bad in (0, -60, 1.5, True):
        with pytest.raises(ValueError):
            chart.from_seconds(bad)
    with pytest.raises(ValueError):
        chart.in_seconds("1h")


@pytest.mark.parametrize(
    "tf, flags",
    [
        ("30S", {"isseconds", "isintraday"}),
        ("60", {"isminutes", "isintraday"}),
        ("D", {"isdaily", "isdwm"}),
        ("1W", {"isweekly", "isdwm"}),
        ("3M", {"ismonthly", "isdwm"}),
    ],
)
def test_timeframe_flags(tf, flags):
    t = chart.Timeframe.of(tf)
    names = ("isseconds", "isminutes", "isintraday", "isdaily", "isweekly", "ismonthly", "isdwm")
    assert {n for n in names if getattr(t, n)} == flags
    assert not t.isticks
    assert t.period == ("1D" if tf == "D" else tf)  # v6 spelling
    assert t.in_seconds() == chart.in_seconds(tf) and t.in_seconds("1W") == 604800


def test_change_weeks_and_months_daily_midnight_utc():
    # Fri 2026-01-30, Mon 02-02 (new week and month), Tue 02-03, Mon 02-09 (new week).
    b = _bars(_ts((2026, 1, 30), (2026, 2, 2), (2026, 2, 3), (2026, 2, 9)), tz="America/New_York")
    # False on the chart's first bar (observed, NFLX 1W 2026-11-05).
    assert chart.change(b, "1W").tolist() == [False, True, False, True]
    assert chart.change(b, "1M").tolist() == [False, True, False, False]
    assert chart.change(b, "1W", first_bar=True)[0] == np.True_


def test_change_intraday_uses_exchange_dates():
    # Sunday 2026-03-01 19:00 and 21:00 New York are Monday 00:00 and 02:00 UTC; Monday
    # 2026-03-02 10:00 New York follows. Dated in New York the first two fall on Sunday (old
    # week) and the third starts a new day and week; dated in UTC all three would be Monday.
    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")
    local = [(2026, 3, 1, 19), (2026, 3, 1, 21), (2026, 3, 2, 10)]
    ts = [int(dt.datetime(*t, tzinfo=ny).timestamp()) for t in local]
    assert dt.datetime.fromtimestamp(ts[0], dt.UTC).weekday() == 0  # Monday in UTC
    b = _bars(ts, tf="60", tz="America/New_York")
    assert chart.change(b, "1W").tolist() == [False, False, True]
    assert chart.change(b, "1D").tolist() == [False, False, True]


def test_change_refuses_intraday_targets():
    b = _bars(_ts((2026, 1, 5), (2026, 1, 6)))
    for tf in ("60", "2W"):
        with pytest.raises(NotImplementedError, match="G1"):
            chart.change(b, tf)


@pytest.mark.parametrize("live", [False, True])
def test_barstate_arrays(live):
    s = chart.barstate(5, offset=0, live_last=live)
    assert s["isfirst"].tolist() == [True, False, False, False, False]
    assert s["islast"].tolist() == [False] * 4 + [True]
    assert s["isrealtime"][-1] == live and not s["isrealtime"][:-1].any()
    assert (s["ishistory"] == ~s["isrealtime"]).all()
    assert (s["isconfirmed"] == ~s["isrealtime"]).all() and (s["isnew"] == ~s["isrealtime"]).all()
    assert np.flatnonzero(s["islastconfirmedhistory"]).tolist() == [3 if live else 4]
    assert not chart.barstate(3, offset=100)["isfirst"].any()  # truncated export


def test_runtime_views():
    b = _bars(_ts((2026, 1, 30), (2026, 2, 2), (2026, 2, 3)))

    def script(ctx):
        ctx.plot("week", ctx.timeframe.change("1W"))
        ctx.plot("daily", ctx.timeframe.isdaily)
        ctx.plot("secs", ctx.timeframe.in_seconds("60"))
        ctx.plot("from", ctx.timeframe.in_seconds(ctx.timeframe.from_seconds(5400)))
        ctx.plot("first", ctx.barstate.isfirst)
        ctx.plot("rt", ctx.barstate.isrealtime)

    r = run(script, b, live_last_bar=True, bar_offset=0)
    assert r["week"].tolist() == [0, 1, 0] and (r["daily"] == 1).all()
    assert (r["secs"] == 3600).all() and (r["from"] == 5400).all()
    assert r["first"].tolist() == [1, 0, 0] and r["rt"].tolist() == [0, 0, 1]


def _export_of(bars, cols, path):
    exp = TradingViewExport(bars, cols, ("time", "open", "high", "low", "close", "Volume", *cols),
                            "unix")  # fmt: skip
    write_tradingview_csv(exp, path)
    return load_tradingview_csv(path, timeframe=bars.timeframe, session_tz=bars.session_tz)


@pytest.mark.parametrize("live", [False, True])
def test_capture_reports_the_last_bar_state(tmp_path, live):
    rng = np.random.default_rng(1)
    n = 200
    c = np.round(50 + np.cumsum(rng.normal(0, 0.3, n)), 2)
    ts = np.arange(n) * 86400 + 1_767_225_600  # daily, midnight UTC, from 2026-01-01
    b = Bars(ts, c, c + 0.5, c - 0.5, c, rng.integers(1, 9, n).astype(float), "1D", "UTC")
    cols = dict(run(runtime_capture, b, live_last_bar=live).columns)
    d = {x.group: x for x in decide(_export_of(b, cols, tmp_path / "x.csv"))}
    assert d["barstate"].ok and d["barstate"].matching == ("live_last" if live else "closed_last",)
    assert d["barstate"].note == f"last bar was {'live' if live else 'closed'}"
    for g in ("timeframe", "in_seconds", "from_seconds", "tf_change"):
        assert d[g].ok, g
