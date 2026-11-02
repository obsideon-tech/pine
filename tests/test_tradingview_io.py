import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pine import Bars
from pine.io import (
    TradingViewExport,
    TradingViewFormatError,
    dump_tradingview_csv,
    load_tradingview_csv,
    parse_tradingview_csv,
)
from pine.manifest import load_manifest

SYN = "synthetic"


def _load(fixtures_dir, name):
    csv_path = fixtures_dir / SYN / f"{name}.csv"
    man = load_manifest(csv_path.with_suffix(".yaml"))
    exp = load_tradingview_csv(csv_path, timeframe=man.timeframe, session_tz=man.timezone)
    return csv_path, exp


@pytest.mark.parametrize("name", ["random_walk_60", "random_walk_1d_iso"])
def test_fixture_round_trips_byte_for_byte(fixtures_dir, name):
    csv_path, exp = _load(fixtures_dir, name)
    assert dump_tradingview_csv(exp) == csv_path.read_text(encoding="utf-8")


def test_unix_fixture_contents(fixtures_dir):
    _, exp = _load(fixtures_dir, "random_walk_60")
    assert len(exp.bars) == 300
    assert exp.time_format == "unix"
    assert list(exp.columns) == ["Plot"]
    plot = exp.columns["Plot"]
    assert np.isnan(plot[:13]).all() and not np.isnan(plot[13:]).any()
    assert np.all(np.diff(exp.bars.ts) == 3600)
    assert exp.observed_decimals["close"] == 2


def test_iso_fixture_converts_to_utc(fixtures_dir):
    _, exp = _load(fixtures_dir, "random_walk_1d_iso")
    assert exp.time_format == "iso"
    # 2026-01-01T00:00:00-05:00
    assert exp.bars.ts[0] == 1_767_243_600
    assert exp.bars.session_tz == "America/New_York"


HEADER = "time,open,high,low,close"


def test_no_volume_column_gives_nan_volume():
    exp = parse_tradingview_csv(f"{HEADER}\n0,1,2,0.5,1.5\n", timeframe="1")
    assert np.isnan(exp.bars.volume).all()
    assert exp.columns == {}


def test_duplicate_plot_titles_are_disambiguated_and_header_kept():
    text = f"{HEADER},Volume,Plot,Plot,Plot\n0,1,2,0.5,1.5,10,1,2,3\n"
    exp = parse_tradingview_csv(text, timeframe="1")
    assert list(exp.columns) == ["Plot", "Plot #2", "Plot #3"]
    assert dump_tradingview_csv(exp) == text


def test_header_case_insensitive_and_bom(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("﻿Time,Open,High,Low,Close\n0,1,2,0.5,1.5\n", encoding="utf-8")
    assert len(load_tradingview_csv(p, timeframe="1").bars) == 1


@pytest.mark.parametrize("tok", ["", "NaN", "nan", "na"])
def test_na_tokens(tok):
    exp = parse_tradingview_csv(f"{HEADER},P\n0,1,2,0.5,1.5,{tok}\n", timeframe="1")
    assert np.isnan(exp.columns["P"][0])


@pytest.mark.parametrize(
    "text, msg",
    [
        ("", "empty"),
        ("date,open,high,low,close\n0,1,2,0.5,1.5\n", "header"),
        (f"{HEADER}\n0,1,2,0.5\n", "fields"),
        (f"{HEADER}\n0,1,2,x,1.5\n", "not a number"),
        (f"{HEADER}\n2026-01-01T00:00:00,1,2,0.5,1.5\n", "no UTC offset"),
        (f"{HEADER}\n0,1,2,0.5,1.5\n1970-01-01T00:01:00+00:00,1,2,0.5,1.5\n", "mixes"),
    ],
)
def test_malformed_rejected(text, msg):
    with pytest.raises(TradingViewFormatError, match=msg):
        parse_tradingview_csv(text, timeframe="1")


finite = st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_subnormal=False)


@settings(max_examples=200)
@given(st.lists(st.tuples(finite, finite, finite), min_size=1, max_size=30))
def test_dump_parse_round_trips_any_float64(rows):
    # Exactness property: writing then reading recovers every float64 bit-for-bit.
    n = len(rows)
    o = np.array([r[0] for r in rows])
    c = np.array([r[1] for r in rows])
    h = np.maximum(o, c) + np.abs([r[2] for r in rows])
    lo = np.minimum(o, c) - np.abs([r[2] for r in rows])
    plot = np.array([r[2] for r in rows])
    plot[::3] = np.nan
    bars = Bars(np.arange(n, dtype=np.int64) * 60, o, h, lo, c, np.full(n, np.nan), timeframe="1")
    exp = TradingViewExport(bars, {"P": plot}, (*HEADER.split(","), "P"), "unix")
    back = parse_tradingview_csv(dump_tradingview_csv(exp), timeframe="1")
    assert back.bars.equals(bars)
    assert np.array_equal(back.columns["P"], plot, equal_nan=True)
