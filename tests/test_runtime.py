"""pine.runtime: execution-model mechanics and the two-implementation oracle (§11 E2)."""

import math

import numpy as np
import pytest
from hyp import examples
from hypothesis import given, settings
from hypothesis import strategies as st

from pine import Bars, ta
from pine.runtime import NA, DuplicateKeyError, Series, run
from pine.runtime import scalar as ps


def _bars(seed=0, n=120):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    o = np.concatenate([[100.0], c[:-1]]) + rng.normal(0, 0.8, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.3, n))
    lo = np.minimum(o, c) - np.abs(rng.normal(0, 0.3, n))
    return Bars(np.arange(n) * 60, o, h, lo, c, rng.integers(1, 9, n).astype(float), timeframe="1")


# ---------- Series, var, scopes ----------


def test_series_history_per_bar_and_assignment():
    def script(ctx):
        x = ctx.series("x", ctx.close.value)
        if ctx.bar_index.value % 2 == 0:
            x.set(x.value * 10)
        ctx.plot("x", x.value)
        ctx.plot("x1", x[1])

    b = _bars()
    r = run(script, b)
    expect = np.where(np.arange(len(b)) % 2 == 0, b.close * 10, b.close)
    np.testing.assert_array_equal(r["x"], expect)
    np.testing.assert_array_equal(r["x1"], np.concatenate([[np.nan], expect[:-1]]))


def test_series_in_a_branch_has_history_per_execution():
    def script(ctx):
        out = NA
        if ctx.bar_index.value % 3 == 0:
            s = ctx.series("s", ctx.close.value)
            out = s[1]  # previous *execution*, i.e. three bars ago
        ctx.plot("prev", out)

    b = _bars()
    r = run(script, b)
    idx = np.arange(len(b))
    calls = b.close[idx % 3 == 0]
    expect = np.full(len(b), np.nan)
    expect[idx % 3 == 0] = np.concatenate([[np.nan], calls[:-1]])
    np.testing.assert_array_equal(r["prev"], expect)


def test_var_initialises_once_and_carries():
    def script(ctx):
        v = ctx.var("v", 5.0)
        v.set(v.value + 1)
        ctx.plot("v", v.value)
        ctx.plot("v1", v[1])

    r = run(script, _bars(n=5))
    np.testing.assert_array_equal(r["v"], [6, 7, 8, 9, 10])
    np.testing.assert_array_equal(r["v1"], [np.nan, 6, 7, 8, 9])


def test_call_sites_keep_separate_history():
    def diff(ctx, x):
        s = ctx.series("x", x)
        return s.value - s[1]

    def script(ctx):
        ctx.plot("a", ctx.call("a", diff, ctx.close.value))
        ctx.plot("b", ctx.call("b", diff, ctx.open.value))

    b = _bars()
    r = run(script, b)
    np.testing.assert_array_equal(r["a"], ta.change(b.close))
    np.testing.assert_array_equal(r["b"], ta.change(b.open))


def test_nested_calls_are_namespaced():
    def inner(ctx, x):
        return ctx.ta.sma("s", x, 3)

    def outer(ctx, x):
        return ctx.call("i1", inner, x) - ctx.call("i2", inner, -x)

    b = _bars()
    r = run(lambda ctx: ctx.plot("o", ctx.call("o", outer, ctx.close.value)), b)
    np.testing.assert_allclose(r["o"], 2 * ta.sma(b.close, 3), equal_nan=True, atol=1e-12)


@pytest.mark.parametrize(
    "script",
    [
        lambda ctx: (ctx.series("k"), ctx.series("k")),
        lambda ctx: (ctx.ta.ema("k", 1.0, 3), ctx.ta.ema("k", 1.0, 3)),
        lambda ctx: (ctx.plot("p", 1), ctx.plot("p", 2)),
        lambda ctx: (ctx.call("f", lambda c: 0), ctx.call("f", lambda c: 0)),
    ],
)
def test_duplicate_keys_raise(script):
    with pytest.raises(DuplicateKeyError):
        run(script, _bars(n=3))


def test_same_key_for_different_builtins_raises():
    def script(ctx):
        if ctx.bar_index.value == 0:
            ctx.ta.ema("k", 1.0, 3)
        else:
            ctx.ta.sma("k", 1.0, 3)

    with pytest.raises(DuplicateKeyError, match="Ema and Sma"):
        run(script, _bars(n=3))


def test_history_bounds_and_errors():
    s = Series(max_bars_back=3)
    for v in range(20):
        s._push(v)
    assert s[0] == 19 and s[3] == 16
    with pytest.raises(IndexError):
        s[4]
    with pytest.raises(IndexError):
        s[-1]
    with pytest.raises(RuntimeError):
        Series(3).set(1.0)
    assert math.isnan(Series(3)[0])


def test_plot_not_called_is_na_and_columns_read_only():
    r = run(lambda ctx: ctx.plot("p", 1.0) if ctx.bar_index.value == 1 else None, _bars(n=3))
    np.testing.assert_array_equal(r["p"], [np.nan, 1.0, np.nan])
    with pytest.raises(ValueError):
        r["p"][0] = 2.0


# ---------- two-implementation oracle: incremental == vectorised, bit for bit ----------


def _with_gaps(x, seed):
    rng = np.random.default_rng(seed)
    x = np.array(x, dtype=float)
    x[rng.choice(x.size, size=4, replace=False)] = np.nan
    return x


SOURCE_FNS = {
    "sma": (lambda ctx, x, n: ctx.ta.sma("f", x, n), ta.sma),
    "ema": (lambda ctx, x, n: ctx.ta.ema("f", x, n), ta.ema),
    "rma": (lambda ctx, x, n: ctx.ta.rma("f", x, n), ta.rma),
    "rsi": (lambda ctx, x, n: ctx.ta.rsi("f", x, n), ta.rsi),
    "stdev": (lambda ctx, x, n: ctx.ta.stdev("f", x, n), ta.stdev),
    "highest": (lambda ctx, x, n: ctx.ta.highest("f", x, n), ta.highest),
    "lowest": (lambda ctx, x, n: ctx.ta.lowest("f", x, n), ta.lowest),
    "change": (lambda ctx, x, n: ctx.ta.change("f", x, n), ta.change),
}


@settings(max_examples=examples(25), deadline=None)
@given(
    st.integers(0, 10_000), st.integers(1, 20), st.sampled_from(sorted(SOURCE_FNS)), st.booleans()
)
def test_incremental_equals_vectorised_bit_for_bit(seed, n, name, gaps):
    b = _bars(seed)
    x = _with_gaps(b.close, seed) if gaps else np.asarray(b.close)
    inc, vec = SOURCE_FNS[name]
    r = run(lambda ctx: ctx.plot("y", inc(ctx, x[ctx.i], n)), b)
    np.testing.assert_array_equal(r["y"], vec(x, n))


POLICY_FNS = ("sma", "ema", "rma", "rsi", "stdev", "highest", "lowest")


@settings(max_examples=examples(30), deadline=None)
@given(
    st.integers(0, 10_000),
    st.integers(1, 15),
    st.sampled_from(POLICY_FNS),
    st.sampled_from(ta.NA_POLICIES),
)
def test_incremental_equals_vectorised_under_every_na_policy(seed, n, name, policy):
    b = _bars(seed)
    x = _with_gaps(b.close, seed)

    def script(ctx):
        ctx.plot("y", getattr(ctx.ta, name)("f", x[ctx.i], n, na_policy=policy))

    np.testing.assert_array_equal(run(script, b)["y"], getattr(ta, name)(x, n, na_policy=policy))


def test_runtime_defaults_follow_observed_policy():
    from pine.ta import DEFAULT_NA_POLICY, NA_POLICY_OBSERVED

    assert {k: DEFAULT_NA_POLICY[k] for k in NA_POLICY_OBSERVED} == {
        "sma": "skip", "stdev": "skip", "highest": "skip", "lowest": "skip",
        "ema": "skip_na_out", "rma": "skip_na_out", "rsi": "skip_na_out",
    }  # fmt: skip


@settings(max_examples=examples(20), deadline=None)
@given(st.integers(0, 10_000), st.integers(1, 20))
def test_bar_based_builtins_equal_vectorised(seed, n):
    b = _bars(seed)

    def script(ctx):
        ctx.plot("atr", ctx.ta.atr("atr", n))
        ctx.plot("tr", ctx.ta.tr())
        ctx.plot("tr1", ctx.ta.tr(handle_na=True))
        f, s = ctx.ta.ema("f", ctx.close.value, 3), ctx.ta.ema("s", ctx.close.value, 8)
        ctx.plot("co", ctx.ta.crossover("co", f, s))
        ctx.plot("cu", ctx.ta.crossunder("cu", f, s))
        ev = ctx.i % 5 == 2
        ctx.plot("bs", ctx.ta.barssince("bs", ev))
        ctx.plot("vw", ctx.ta.valuewhen("vw", ev, ctx.close.value, 1))

    r = run(script, b)
    np.testing.assert_array_equal(r["atr"], ta.atr(b.high, b.low, b.close, n))
    np.testing.assert_array_equal(r["tr"], ta.tr(b.high, b.low, b.close))
    np.testing.assert_array_equal(r["tr1"], ta.tr(b.high, b.low, b.close, handle_na=True))
    f, s = ta.ema(b.close, 3), ta.ema(b.close, 8)
    np.testing.assert_array_equal(r["co"], ta.crossover(f, s).astype(float))
    np.testing.assert_array_equal(r["cu"], ta.crossunder(f, s).astype(float))
    ev = np.arange(len(b)) % 5 == 2
    np.testing.assert_array_equal(r["bs"], ta.barssince(ev))
    np.testing.assert_array_equal(r["vw"], ta.valuewhen(ev, b.close, 1))


def test_stdev_sample_and_unbiased_through_runtime():
    b = _bars(3)
    r = run(lambda ctx: ctx.plot("s", ctx.ta.stdev("s", ctx.close.value, 7, False)), b)
    np.testing.assert_array_equal(r["s"], ta.stdev(b.close, 7, biased=False))


# ---------- conditional semantics ----------


def test_conditional_ema_equals_ema_of_the_called_subsequence():
    b = _bars(4)
    m = np.arange(len(b)) % 3 == 0

    def script(ctx):
        v = ctx.ta.ema("e", ctx.close.value, 5) if m[ctx.i] else NA
        ctx.plot("e", v)

    r = run(script, b)
    expect = np.full(len(b), np.nan)
    expect[m] = ta.ema(b.close[m], 5)
    np.testing.assert_array_equal(r["e"], expect)
    assert not np.allclose(r["e"][m], ta.ema(b.close, 5)[m], equal_nan=True)


def test_uncalled_builtin_keeps_its_state():
    b = _bars(5, n=30)

    def script(ctx):
        if ctx.i < 10 or ctx.i >= 20:  # skipped for ten bars
            ctx.plot("s", ctx.ta.sma("s", ctx.close.value, 3))

    r = run(script, b)
    # Bar 20's window is bars 8, 9, 20: the gap is invisible to the built-in.
    assert r["s"][20] == pytest.approx((b.close[8] + b.close[9] + b.close[20]) / 3)


# ---------- scalar helpers (expected hypotheses) ----------


def test_scalar_helpers():
    assert (ps.round_(2.5), ps.round_(-2.5), ps.round_(3.5)) == (3.0, -3.0, 4.0)
    assert ps.round_(-1.235, 2) == pytest.approx(-1.24) and ps.round_(0.4) == 0.0
    assert ps.round_(1.235, 2) == pytest.approx(1.24)
    assert ps.mod(-7, 3) == -1.0 and math.isnan(ps.mod(1, 0))
    assert ps.div(7, 2) == 3.5 and math.isnan(ps.div(1, 0))
    assert ps.to_int(-2.7) == -2.0 and math.isnan(ps.to_int(NA))
    assert ps.nz(NA) == 0.0 and ps.nz(NA, 5) == 5 and ps.nz(2) == 2
    assert ps.na(None) and not ps.na(1)
    assert math.isnan(ps.max_(1, NA)) and ps.max_(1, 3, 2) == 3 and ps.min_(1, 3) == 1
    assert not ps.eq(NA, NA) and ps.eq(2, 2.0)
    assert all(math.isnan(f(NA)) for f in (ps.round_,))


def test_max_bars_back_validated():
    with pytest.raises(ValueError):
        run(lambda ctx: None, _bars(n=2), max_bars_back=0)


# ---------- for loops (ctx.loop) ----------


def test_loop_call_site_shared_across_iterations():
    b = _bars(1, n=30)

    def script(ctx):
        got = []
        for j in ctx.loop("lp", range(1, 4)):
            got.append(ctx.ta.sum("s", ctx.close.value * j, 2))
            xs = ctx.series("x", float(j))
            got.append(xs[1])
        ctx.plot("first_sum", got[0])
        ctx.plot("last_sum", got[4])
        ctx.plot("first_prev", got[1])
        ctx.plot("last_prev", got[5])

    r = run(script, b)
    c = b.close
    # One state for the site: the window holds the last two *calls*.
    np.testing.assert_allclose(r["first_sum"][1:], 3 * c[:-1] + c[1:])
    np.testing.assert_allclose(r["last_sum"], 5 * c)
    assert np.isnan(r["first_sum"][0])
    # Series history inside the body is per iteration.
    assert np.isnan(r["first_prev"][0]) and (r["first_prev"][1:] == 3.0).all()
    assert (r["last_prev"] == 2.0).all()


def test_loop_keys_unique_per_iteration_and_outside():
    b = _bars(2, n=5)

    def twice_in_one_iteration(ctx):
        for _ in ctx.loop("lp", range(2)):
            ctx.series("x")
            ctx.series("x")

    with pytest.raises(DuplicateKeyError):
        run(twice_in_one_iteration, b)

    def loop_key_reused(ctx):
        for _ in ctx.loop("lp", range(1)):
            pass
        for _ in ctx.loop("lp", range(1)):
            pass

    with pytest.raises(DuplicateKeyError):
        run(loop_key_reused, b)


def test_loop_break_and_empty_restore_scope():
    b = _bars(3, n=10)

    def script(ctx):
        for j in ctx.loop("lp", range(5)):
            if j == 1:
                break
        for _ in ctx.loop("none", range(0)):
            pass  # pragma: no cover
        ctx.plot("after", ctx.series("x", 1.0).value)
        assert ctx._prefix == ""

    assert (run(script, b)["after"] == 1.0).all()


def test_nested_loops_and_calls_in_loops():
    b = _bars(4, n=12)

    def add_prev(ctx, v):
        s = ctx.series("v", v)
        return s.value + (0.0 if np.isnan(s[1]) else s[1])

    def script(ctx):
        tot = 0.0
        for i in ctx.loop("outer", range(2)):
            for j in ctx.loop("inner", range(2)):
                tot += ctx.call("f", add_prev, float(i * 2 + j))
        ctx.plot("tot", tot)

    r = run(script, b)
    # Calls in order 0,1,2,3 every bar; each adds the previous call's value (3 on bar t > 0).
    assert r["tot"][0] == 0 + 1 + 3 + 5
    assert (r["tot"][1:] == 3 + 1 + 3 + 5).all()
