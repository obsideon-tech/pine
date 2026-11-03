"""Exports that start after the chart's first bar (pine.truncation, PINE_PLAN.md §5.4).

The first real export started at bar_index 3116 of 6151. These tests build a long synthetic
history, compute every capture column on all of it ("what TradingView sees"), and export only
the tail ("what TradingView exports"). The checks must pass on the tail with the right
exclusions, still catch wrong ports, and refuse vacuous passes on short tails.
"""

from pathlib import Path

import numpy as np
import pytest

from pine import Bars, ta
from pine.io import TradingViewExport, load_tradingview_csv, write_tradingview_csv
from pine.reference import (
    NA_PORTS,
    PLOT_TITLES,
    _gapfree_specs,
    _src_na_expected,
    check_capture,
    decide_na_policy,
)
from pine.runtime import run
from pine.semantics import EXPECTED, GROUPS, decide, runtime_capture
from pine.truncation import bar_offset, decay_adjusted, sync_point

N, K = 1600, 700  # full history, first exported bar


@pytest.fixture(scope="module")
def full():
    rng = np.random.default_rng(2026)
    c = 300 + np.cumsum(rng.normal(0, 2, N))
    o = np.concatenate([[300.0], c[:-1]]) + rng.normal(0, 1.5, N)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.8, N))
    lo = np.minimum(o, c) - np.abs(rng.normal(0, 0.8, N))
    bars = Bars(np.arange(N) * 86400, o, h, lo, c, np.full(N, 1e6), timeframe="1D")
    pre = TradingViewExport(
        bars,
        {"ema_fast": ta.ema(c, 9), "ema_slow": ta.ema(c, 21)},
        ("time", "open", "high", "low", "close"),
        "unix",
    )
    cols = {n: np.asarray(s.fn(bars), dtype=float) for n, s in _gapfree_specs(pre, 0).items()}
    src = _src_na_expected(bars, 0)
    cols |= {n: f(src, L) for n, (f, L, _) in NA_PORTS.items()}
    cols |= dict(run(runtime_capture, bars).columns)  # includes bar_index
    return bars, cols


def tail(full, tmp_path: Path, k: int = K, edit=None) -> TradingViewExport:
    bars, cols = full
    sl = slice(k, None)
    b = Bars(bars.ts[sl], bars.open[sl], bars.high[sl], bars.low[sl], bars.close[sl],
             bars.volume[sl], timeframe="1D")  # fmt: skip
    tc = {n: np.asarray(v)[sl] for n, v in cols.items()}
    if edit:
        edit(tc, b)
    exp = TradingViewExport(b, tc, ("time", "open", "high", "low", "close", "Volume", *tc), "unix")
    path = tmp_path / f"tail_{k}.csv"
    write_tradingview_csv(exp, path)
    return load_tradingview_csv(path, timeframe="1D")


def test_reference_passes_on_truncated_export(full, tmp_path):
    exp = tail(full, tmp_path)
    assert bar_offset(exp.columns) == K
    rep = check_capture(exp)
    assert rep.passed, rep.to_markdown()
    for c in rep.columns:
        assert c.n_compared >= 100
        assert any("bar_index 700" in n for n in c.notes)
    assert all(d.passed for d in rep.discrete)


def test_na_policy_still_decided_on_truncated_export(full, tmp_path):
    decisions = decide_na_policy(tail(full, tmp_path))
    assert {d.column: d.matching for d in decisions} == {c: ("propagate",) for c in NA_PORTS}


def test_decay_check_catches_a_wrong_alpha(full, tmp_path):
    def wrong_ema(tc, b):  # TradingView's column computed with length 15 instead of 14
        tc["ema"] = ta.ema(full[0].close, 15)[K:]

    rep = check_capture(tail(full, tmp_path, edit=wrong_ema))
    ema = next(c for c in rep.columns if c.name == "ema")
    assert not ema.passed and ema.first_divergence is not None


def test_windowed_check_still_exact(full, tmp_path):
    def nudge(tc, b):
        v = np.array(tc["sma"])
        v[500] += 1e-6
        tc["sma"] = v

    rep = check_capture(tail(full, tmp_path, edit=nudge))
    sma = next(c for c in rep.columns if c.name == "sma")
    assert not sma.passed and sma.first_divergence == 500


def test_short_truncated_export_is_inconclusive_not_a_pass(full, tmp_path):
    exp = tail(full, tmp_path, k=N - 150)
    rep = check_capture(exp)
    rsi = next(c for c in rep.columns if c.name == "rsi")
    assert rsi.inconclusive and not rsi.passed
    assert "inconclusive" in rep.to_markdown()


def test_semantics_on_truncated_export(full, tmp_path):
    decisions = {d.group: d for d in decide(tail(full, tmp_path))}
    for g, d in decisions.items():
        assert d.ok, (g, d.matching, d.excluded)
    # Behaviour before the first event is not visible when the export starts later.
    assert decisions["barssince"].undecided
    for g in ("supertrend", "trail_stop", "regime"):
        assert "synchronised" in decisions[g].excluded, (g, decisions[g].excluded)
    # A one-bar whipsaw in this history separates per-call from every-bar evaluation.
    assert decisions["regime"].matching == ("per_call",)


@pytest.mark.parametrize("group, alt", [("cond_sma", "every_bar"), ("branch_hist", "last_assigned"),
                                        ("call_sites", "shared")])  # fmt: skip
def test_semantics_alternatives_still_detected_when_truncated(full, tmp_path, group, alt):
    from pine.semantics import _At

    bars, cols = full
    alt_cols = GROUPS[group][alt](_At(bars, 0, cols))

    def swap(tc, b):
        for c, v in alt_cols.items():
            tc[c] = np.asarray(v)[K:]

    d = next(d for d in decide(tail(full, tmp_path, edit=swap)) if d.group == group)
    assert d.matching == (alt,) and not d.ok and EXPECTED[group] != alt


def test_var_increments_checked_when_truncated(full, tmp_path):
    def break_var(tc, b):
        v = np.array(tc["var_acc"])
        v[300:] += 1.0  # one wrong increment
        tc["var_acc"] = v

    d = next(d for d in decide(tail(full, tmp_path, edit=break_var)) if d.group == "var")
    assert d.matching == ()


# ---------- unit tests ----------


def test_bar_offset():
    assert bar_offset({}) is None
    assert bar_offset({"bar_index": np.array([5.0, 6.0, 7.0])}) == 5
    with pytest.raises(ValueError):
        bar_offset({"bar_index": np.array([5.0, 7.0])})
    with pytest.raises(ValueError):
        bar_offset({"bar_index": np.array([np.nan, 1.0])})


def test_decay_adjusted_and_sync_point():
    pine = np.linspace(10, 20, 50)
    py = pine + 3.0 * 0.5 ** np.arange(50)
    np.testing.assert_allclose(decay_adjusted(py, pine, 0.5, 0), pine, atol=1e-12)
    assert np.array_equal(decay_adjusted(py, pine, 0.5, 60), py)
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([0.0, 2.0, 0.0, 4.0, 5.0])
    assert sync_point(a, b, 1e-9) == 3
    assert sync_point(a, a, 1e-9, start=2) == 2
    assert sync_point(a, b + 1, 1e-9) == 5


def test_reference_only_export_infers_gap_phase(full, tmp_path):
    def drop_bar_index(tc, b):
        tc.pop("bar_index")
        for c in [c for c in tc if c not in PLOT_TITLES]:
            tc.pop(c)

    exp = tail(full, tmp_path, edit=drop_bar_index)
    rep = check_capture(exp)
    assert rep.passed, rep.to_markdown()
    assert any(f"≡ {K % 50} (mod 50)" in n for c in rep.columns for n in c.notes)
