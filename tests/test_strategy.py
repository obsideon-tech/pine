"""pine.strategy broker emulator and the strategy capture (PINE_PLAN.md §13).

Oracle (a): hand-computed scenarios. Oracle (b): accounting identities. Oracle (c), a
TradingView export of tools/capture_strategy.pine, is checked by pine.capture.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from pine import Bars
from pine.io import TradingViewExport, load_tradingview_csv, write_tradingview_csv
from pine.lint import lint_file
from pine.lint._calls import find_calls
from pine.lint._lexer import tokenize
from pine.runtime import run
from pine.strategy import StrategyConfig
from pine.strategy_capture import COLUMNS, CONSTANTS, GROUPS, decide, strategy_capture

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "capture_strategy.pine"


def bars_from(rows):
    o, h, lo, c = (np.array(x, float) for x in zip(*rows, strict=True))
    n = len(rows)
    return Bars(np.arange(n) * 60, o, h, lo, c, np.full(n, np.nan), timeframe="1")


def run_orders(rows, orders, cfg=None):
    """``orders``: {bar: callable(strategy, ctx)}; returns (result, broker)."""

    def script(ctx):
        f = orders.get(ctx.i)
        if f:
            f(ctx.strategy, ctx)
        s = ctx.strategy
        ctx.plot("size", s.position_size)
        ctx.plot("net", s.netprofit)
        ctx.plot("eq", s.equity)

    r = run(script, bars_from(rows), strategy=cfg or StrategyConfig())
    return r, r.broker


FLAT = [(10, 10, 10, 10)] * 2


# ---------- oracle (a): hand-computed scenarios ----------


def test_market_entry_fills_next_open_and_close_realizes():
    rows = [(10, 11, 9, 10), (12, 13, 11, 12), (14, 15, 13, 15), (16, 17, 15, 16)]
    r, b = run_orders(rows, {0: lambda s, c: s.entry("L", "long"), 1: lambda s, c: s.close("L")})
    np.testing.assert_array_equal(
        r["size"], [0, 1, 0, 0]
    )  # filled at bar 1 open, closed at bar 2 open
    assert [(t.entry_price, t.exit_price) for t in b.trades] == [(12, 14)]
    np.testing.assert_array_equal(r["net"], [0, 0, 2, 2])
    assert r["eq"][1] == 100_000 + (12 - 12)  # open profit at bar 1 close: 12 - 12


def test_reversal_closes_and_opens_at_one_fill():
    rows = [(10, 10, 10, 10), (20, 20, 20, 20), (15, 15, 15, 15), (12, 12, 12, 12)]
    r, b = run_orders(
        rows, {0: lambda s, c: s.entry("S", "short"), 1: lambda s, c: s.entry("L", "long")}
    )
    np.testing.assert_array_equal(r["size"], [0, -1, 1, 1])
    assert len(b.trades) == 1 and b.trades[0].profit == 20 - 15  # short 20 -> 15
    assert b.avg_price == 15


def test_pyramiding_one_ignores_second_entry_and_close_all():
    rows = [(10, 10, 10, 10)] * 3 + [(13, 13, 13, 13)]
    r, b = run_orders(
        rows,
        {0: lambda s, c: s.entry("L", "long"), 1: lambda s, c: s.entry("L2", "long"),
         2: lambda s, c: s.close_all()},
    )  # fmt: skip
    np.testing.assert_array_equal(r["size"], [0, 1, 1, 0])
    assert b.trades[0].profit == 3


def test_close_with_other_id_does_nothing():
    rows = [(10, 10, 10, 10)] * 3
    r, b = run_orders(rows, {0: lambda s, c: s.entry("L", "long"), 1: lambda s, c: s.close("X")})
    np.testing.assert_array_equal(r["size"], [0, 1, 1])


def bracket(stop, limit):
    def f(s, c):
        s.entry("B", "long")
        s.exit("BX", "B", stop=stop, limit=limit)

    return f


@pytest.mark.parametrize(
    "bar1, path, exit_price",
    [
        # high nearer the open (2.5 < 3): O-H-L-C, the limit at 102 is reached first
        ((100, 102.5, 97, 101), "nearest_extreme_first", 102),
        ((100, 102.5, 97, 101), "high_first", 102),
        # low nearer the open (1.5 < 3): O-L-H-C, the stop at 99 is reached first
        ((100, 103, 98.5, 101), "nearest_extreme_first", 99),
        ((100, 103, 98.5, 101), "high_first", 102),
    ],
)
def test_bracket_follows_intrabar_path(bar1, path, exit_price):
    rows = [(100, 100, 100, 100), bar1, (100, 100, 100, 100)]
    _, b = run_orders(rows, {0: bracket(99, 102)}, StrategyConfig(path=path))
    assert [t.exit_price for t in b.trades] == [exit_price]
    assert b.trades[0].exit_bar == 1  # same bar as the entry fill


@pytest.mark.parametrize("gap_fill, price", [("open", 97), ("order_price", 99)])
def test_gap_through_stop(gap_fill, price):
    rows = [(100, 100, 100, 100), (100, 100.5, 99.5, 100), (97, 98, 96, 97.5)]
    _, b = run_orders(rows, {0: bracket(99, 110)}, StrategyConfig(gap_fill=gap_fill))
    assert [(t.exit_bar, t.exit_price) for t in b.trades] == [(2, price)]


def test_exit_same_bar_switch():
    rows = [(100, 100, 100, 100), (100, 103, 99.5, 101), (101, 101, 101, 101)]
    _, b = run_orders(rows, {0: bracket(99, 102)}, StrategyConfig(exit_same_bar=False))
    assert b.trades == [] and b.size == 1  # bar 1 would have hit the limit; not yet active


def test_same_close_fill_timing():
    rows = [(10, 11, 9, 10.5), (12, 13, 11, 12.5)]
    r, b = run_orders(
        rows, {0: lambda s, c: s.entry("L", "long")}, StrategyConfig(fill_timing="same_close")
    )
    assert b.avg_price == 10.5
    np.testing.assert_array_equal(r["size"], [0, 1])  # plotted before the close fill on bar 0


def test_bad_direction_rejected():
    with pytest.raises(ValueError):
        run_orders(FLAT, {0: lambda s, c: s.entry("L", "up")})


# ---------- oracle (b): accounting identities on the capture strategy ----------


def _long_bars(seed=1, n=1200, vol=1.5):
    rng = np.random.default_rng(seed)
    c = 200 + np.cumsum(rng.normal(0, vol, n))
    o = np.concatenate([[200.0], c[:-1]]) + rng.normal(0, vol, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, vol, n))
    lo = np.minimum(o, c) - np.abs(rng.normal(0, vol, n))
    return Bars(np.arange(n) * 86400, o, h, lo, c, np.full(n, 1e6), timeframe="1D")


@pytest.mark.parametrize("cfg", [GROUPS[g][h] for g in GROUPS for h in GROUPS[g]])
def test_accounting_identities(cfg):
    b = _long_bars()
    r = run(strategy_capture, b, strategy=cfg)
    br = r.broker
    np.testing.assert_allclose(
        r["strat_equity"], 100_000 + r["strat_netprofit"] + r["strat_openprofit"], atol=1e-9
    )
    assert math.isclose(br.netprofit, sum(t.profit for t in br.trades), abs_tol=1e-9)
    flat = r["strat_position_size"] == 0
    assert (r["strat_openprofit"][flat] == 0).all() and np.isnan(r["strat_avg_price"][flat]).all()
    assert np.all(np.diff(r["strat_closedtrades"]) >= 0)
    assert set(np.unique(r["strat_position_size"])) <= {-1.0, 0.0, 1.0}


# ---------- the capture script and the decision pipeline ----------


def test_pine_constants_and_titles_match():
    toks, _ = tokenize(SCRIPT.read_text(encoding="utf-8"))
    found = {}
    for i, t in enumerate(toks[:-4]):
        if t.text == "const" and toks[i + 1].text in ("int", "float") and toks[i + 3].is_op("="):
            found[toks[i + 2].text] = float(toks[i + 4].text)
    assert found == {k: float(v) for k, v in CONSTANTS.items()}
    titles = [c.args[1].text.strip("\"'") for c in find_calls(toks) if c.name == "plot"]
    assert tuple(titles) == COLUMNS
    assert lint_file(SCRIPT).findings == ()


def _export(bars, cfg, tmp_path, k=0):
    cols = dict(run(strategy_capture, bars, strategy=cfg).columns)
    sl = slice(k, None)
    b = Bars(bars.ts[sl], bars.open[sl], bars.high[sl], bars.low[sl], bars.close[sl],
             bars.volume[sl], timeframe="1D")  # fmt: skip
    cols = {c: np.asarray(v)[sl] for c, v in cols.items()}
    exp = TradingViewExport(
        b, cols, ("time", "open", "high", "low", "close", "Volume", *cols), "unix"
    )
    path = tmp_path / f"strat_{k}.csv"
    write_tradingview_csv(exp, path)
    return load_tradingview_csv(path, timeframe="1D")


@pytest.mark.parametrize("group", [g for g in GROUPS if len(GROUPS[g]) > 1])
def test_each_rule_is_decided_and_alternatives_detected(group, tmp_path):
    bars = _long_bars()
    for hyp, cfg in GROUPS[group].items():
        d = next(d for d in decide(_export(bars, cfg, tmp_path)) if d.group == group)
        assert d.matching == (hyp,), (group, hyp, d.matching)


def test_expected_export_passes_every_group(tmp_path):
    decisions = decide(_export(_long_bars(), StrategyConfig(), tmp_path))
    assert all(d.matching == (d.expected,) for d in decisions), [
        (d.group, d.matching) for d in decisions
    ]


def test_mid_history_export_syncs_at_first_flat_bar(tmp_path):
    decisions = decide(_export(_long_bars(), StrategyConfig(), tmp_path, k=517))
    assert all(d.ok for d in decisions)
    assert decisions[0].sync_row == (39 - 517) % 40


def test_live_last_bar_with_empty_strategy_columns_is_dropped(tmp_path):
    bars = _long_bars(n=400)
    exp = _export(bars, StrategyConfig(), tmp_path)
    cols = {c: np.array(v) for c, v in exp.columns.items()}
    for c in COLUMNS:
        cols[c][-1] = np.nan  # strategy not yet calculated on the live bar
    live = TradingViewExport(exp.bars, cols, exp.header, exp.time_format)
    decisions = decide(live)
    assert all(d.matching == (d.expected,) for d in decisions)
    assert decisions[0].dropped_live_bar
    from pine.strategy_capture import summary

    assert "Last row dropped" in summary(decisions)


def test_na_strategy_bar_index_elsewhere_is_still_an_error(tmp_path):
    exp = _export(_long_bars(n=400), StrategyConfig(), tmp_path)
    cols = {c: np.array(v) for c, v in exp.columns.items()}
    cols["strat_bar_index"][100] = np.nan
    with pytest.raises(ValueError, match="strat_bar_index: na at row 100"):
        decide(TradingViewExport(exp.bars, cols, exp.header, exp.time_format))
