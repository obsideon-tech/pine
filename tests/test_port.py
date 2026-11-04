"""pine.port.check_port: stage-by-stage acceptance and root-cause location (§11 H1)."""

import numpy as np
import pytest

from pine import Bars
from pine.io import TradingViewExport, load_tradingview_csv, write_tradingview_csv
from pine.port import Stage, check_port
from pine.runtime import run


def _bars(n=800, seed=3):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    o = np.concatenate([[100.0], c[:-1]])
    h = np.maximum(o, c) + 0.5
    lo = np.minimum(o, c) - 0.5
    return Bars(np.arange(n) * 86400, o, h, lo, c, np.full(n, 1e3), timeframe="1D")


def make_script(smooth_len=10, signal_len=5, flip_bug_at=None):
    """Pipeline: smooth = sma(close) -> trend = ema(smooth) -> up = crossover -> state."""

    def script(ctx):
        ctx.plot("bar_index", ctx.bar_index.value)
        smooth = ctx.ta.sma("smooth", ctx.close.value, smooth_len)
        trend = ctx.ta.ema("trend", smooth, signal_len)
        if flip_bug_at is not None and ctx.i >= flip_bug_at:
            trend += 0.5
        up = ctx.ta.crossover("up", smooth, trend)
        dn = ctx.ta.crossunder("dn", smooth, trend)
        state = ctx.var("state", 0)
        if up:
            state.set(1)
        elif dn:
            state.set(-1)
        ctx.plot("dbg_smooth", smooth)
        ctx.plot("dbg_trend", trend)
        ctx.plot("signal", 1 if up else 0)
        ctx.plot("state", state.value)

    return script


STAGES = [
    Stage("dbg_smooth", exclude=10),
    Stage("dbg_trend", exclude=15),
    Stage("signal", kind="discrete", exclude=15),
    Stage("state", sync=True, exclude=15),
]


# In a truncated export the EMA's unseen seed must decay below float tolerance first
# (pine.parity.convergence_bars: about 70 rows for length 5 at this price scale).
STAGES_TRUNCATED = [
    Stage("dbg_smooth", exclude=10),
    Stage("dbg_trend", exclude=80),
    Stage("signal", kind="discrete", exclude=80),
    Stage("state", sync=True, exclude=80),
]


def _export(tmp_path, k=0, name="x.csv"):
    bars = _bars()
    cols = dict(run(make_script(), bars).columns)
    sl = slice(k, None)
    b = Bars(bars.ts[sl], bars.open[sl], bars.high[sl], bars.low[sl], bars.close[sl],
             bars.volume[sl], timeframe="1D")  # fmt: skip
    c = {n: np.asarray(v)[sl] for n, v in cols.items()}
    path = tmp_path / name
    write_tradingview_csv(
        TradingViewExport(b, c, ("time", "open", "high", "low", "close", "Volume", *c), "unix"),
        path,
    )
    return load_tradingview_csv(path, timeframe="1D")


def test_correct_port_passes(tmp_path):
    rep = check_port(_export(tmp_path), make_script(), STAGES)
    assert rep.passed and rep.root_cause is None and rep.first_failing_stage is None
    assert rep.offset == 0 and set(rep.status.values()) == {"pass"}
    assert "## Port check: PASS" in rep.to_markdown()


def test_root_cause_is_the_earliest_divergence(tmp_path):
    exp = _export(tmp_path)
    # Wrong length in the first stage: every later stage also fails, but the first one is named.
    rep = check_port(exp, make_script(smooth_len=11), STAGES)
    assert not rep.passed
    assert rep.root_cause == "dbg_smooth" and rep.first_failing_stage == "dbg_smooth"
    # A bug introduced in the second stage at row 300: the first stage still passes.
    rep = check_port(exp, make_script(flip_bug_at=300), STAGES)
    assert rep.status["dbg_smooth"] == "pass" and rep.root_cause == "dbg_trend"
    assert rep.first_divergence["dbg_trend"] == 300
    md = rep.to_markdown()
    assert "Root cause candidate: **dbg_trend** (first divergence at row 300)" in md
    assert "| 2 | dbg_trend | 15 | 300 | FAIL |" in md


def test_truncated_export_uses_offset_and_sync(tmp_path):
    exp = _export(tmp_path, k=300)
    rep = check_port(exp, make_script(), STAGES_TRUNCATED)
    assert rep.passed and rep.offset == 300
    assert any(n.startswith("state: synchronised at row") for n in rep.notes)
    assert "bar_index 300" in rep.to_markdown()


def test_truncated_short_export_is_inconclusive(tmp_path):
    exp = _export(tmp_path, k=720)
    rep = check_port(exp, make_script(), STAGES_TRUNCATED)
    assert not rep.passed and rep.status["dbg_smooth"] == "inconclusive"
    assert rep.root_cause is None and rep.first_failing_stage == "dbg_smooth"


def test_never_synchronising_stage_is_reported(tmp_path):
    exp = _export(tmp_path, k=300)

    def wrong_state(ctx):
        make_script()(ctx)
        ctx._plots["state"][ctx.i] = 7.0

    rep = check_port(exp, wrong_state, STAGES_TRUNCATED)
    assert rep.status["state"] == "FAIL"
    assert "state: never synchronised in the first half of the export" in rep.notes


def test_errors(tmp_path):
    exp = _export(tmp_path)
    with pytest.raises(KeyError, match="dbg_missing"):
        check_port(exp, make_script(), [Stage("dbg_missing")])
    with pytest.raises(KeyError, match="never plotted 'other'"):
        check_port(exp, make_script(), [Stage("dbg_smooth", plot="other")])
    with pytest.raises(ValueError, match="unique"):
        check_port(exp, make_script(), [Stage("state"), Stage("state")])
    with pytest.raises(ValueError, match="no stages"):
        check_port(exp, make_script(), [])


def test_discrete_excuse_combined_with_warmup(tmp_path):
    exp = _export(tmp_path)
    flips = np.zeros(len(exp.bars), bool)
    flips[500] = True

    def wrong_signal(ctx):
        make_script()(ctx)
        if ctx.i == 500:
            ctx._plots["signal"][ctx.i] = 1.0 - ctx._plots["signal"][ctx.i]

    st = [Stage("signal", kind="discrete", exclude=15, excuse=lambda b: flips)]
    assert check_port(exp, wrong_signal, st).passed
    st = [Stage("signal", kind="discrete")]
    assert check_port(exp, wrong_signal, st).first_divergence["signal"] == 500
