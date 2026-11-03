"""Batch-F built-ins (PINE_PLAN.md §11 F): closed forms, naive references, na policies, and
bit-for-bit agreement between `pine.ta` and `pine.runtime`."""

import math

import numpy as np
import pytest
from hyp import examples
from hypothesis import given, settings
from hypothesis import strategies as st

from pine import Bars, ta
from pine.runtime import builtins as rb
from pine.runtime import run

NAN = np.nan


def _bars(seed=0, n=150, decimals=None):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    o = np.concatenate([[100.0], c[:-1]]) + rng.normal(0, 0.3, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.4, n))
    lo = np.minimum(o, c) - np.abs(rng.normal(0, 0.4, n))
    if decimals is not None:
        o, h, lo, c = (np.round(v, decimals) for v in (o, h, lo, c))
        h, lo = np.maximum.reduce([o, h, lo, c]), np.minimum.reduce([o, h, lo, c])
    v = rng.integers(1_000, 50_000, n).astype(float)
    return Bars(np.arange(n) * 3600, o, h, lo, c, v, timeframe="60")


def _gaps(x, every=17, width=3):
    y = np.array(x, dtype=float)
    for s in range(every - width, y.size, every):
        y[s : s + width] = NAN
    return y


def _naive_window(x, n, fn):
    out = np.full(x.size, NAN)
    for t in range(n - 1, x.size):
        w = x[t - n + 1 : t + 1]
        if not np.isnan(w).any():
            out[t] = fn(w)
    return out


# ---------- closed forms ----------


def test_closed_forms():
    x = np.arange(1.0, 11.0)
    assert ta.math_sum(x, 3)[-1] == 27.0
    assert ta.wma(x, 3)[-1] == pytest.approx((10 * 3 + 9 * 2 + 8) / 6)
    assert ta.variance(x, 4)[-1] == pytest.approx(1.25)
    assert ta.variance(x, 4, biased=False)[-1] == pytest.approx(5 / 3)
    assert ta.percentile_linear_interpolation(x, 4, 30)[-1] == pytest.approx(7.9)
    assert ta.percentile_linear_interpolation(x, 4, 100)[-1] == 10.0
    assert ta.percentile_linear_interpolation(x, 4, 0)[-1] == 7.0
    # On a straight line every WMA lags by the same amount, so HMA is exact.
    np.testing.assert_allclose(ta.hma(x, 4)[-3:], x[-3:])
    np.testing.assert_array_equal(ta.fixnan([NAN, 1, NAN, NAN, 2, NAN]), [NAN, 1, 1, 1, 2, 2])
    np.testing.assert_array_equal(ta.cum([1, 2, 3]), [1, 3, 6])
    assert np.isnan(ta.math_sum(x, 3)[:2]).all() and np.isnan(ta.wma(x, 3)[:2]).all()


def test_pivots_and_tie_rule():
    x = np.array([1, 2, 5, 3, 3, 1.0])
    # The pivot at index 2 is reported 2 bars later.
    np.testing.assert_array_equal(ta.pivothigh(x, 2, 2), [NAN] * 4 + [5, NAN])
    np.testing.assert_array_equal(ta.pivotlow(-x, 2, 2), [NAN] * 4 + [-5, NAN])
    # Flat top: the leftmost bar of the plateau is the pivot.
    flat = np.array([1, 2, 5, 5, 1, 0, 0.0])
    np.testing.assert_array_equal(ta.pivothigh(flat, 2, 2), [NAN] * 4 + [5, NAN, NAN])
    # na in the window: no pivot.
    assert np.isnan(ta.pivothigh([1, NAN, 5, 3, 1.0], 2, 2)).all()


def test_obv_and_pvt():
    close = np.array([10, 11, 11, 9, 12.0])
    vol = np.array([100, 200, 300, 400, 500.0])
    np.testing.assert_array_equal(ta.obv(close, vol), [NAN, 200, 200, -200, 300])
    expected = np.cumsum([0, 1 / 10 * 200, 0, -2 / 11 * 400, 3 / 9 * 500])
    np.testing.assert_allclose(ta.pvt(close, vol)[1:], expected[1:])
    assert np.isnan(ta.pvt(close, vol)[0])


@pytest.mark.parametrize(
    "policy, expected",
    [
        ("skip", [NAN, 1, 1, 3, 3]),
        ("skip_na_out", [NAN, 1, NAN, 3, NAN]),
        ("propagate", [NAN, NAN, NAN, NAN, NAN]),
    ],
)
def test_cum_na_policies(policy, expected):
    np.testing.assert_array_equal(ta.cum([NAN, 1, NAN, 2, NAN], na_policy=policy), expected)
    if policy == "propagate":
        np.testing.assert_array_equal(ta.cum([1, 2, NAN, 4], na_policy=policy), [1, 3, NAN, NAN])


# ---------- naive references ----------


@settings(max_examples=examples(25), deadline=None)
@given(st.integers(0, 10_000), st.integers(2, 15))
def test_window_builtins_match_naive_references(seed, n):
    x = _bars(seed).close
    w = np.arange(1, n + 1, dtype=float)
    for got, ref in [
        (ta.math_sum(x, n), _naive_window(x, n, np.sum)),
        (ta.wma(x, n), _naive_window(x, n, lambda v: np.dot(v, w) / w.sum())),
        (ta.variance(x, n), _naive_window(x, n, np.var)),
        (ta.variance(x, n, biased=False), _naive_window(x, n, lambda v: np.var(v, ddof=1))),
        (
            ta.percentile_linear_interpolation(x, n, 37.5),
            _naive_window(x, n, lambda v: np.percentile(v, 37.5)),
        ),
    ]:
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(ta.variance(x, n), ta.stdev(x, n) ** 2, rtol=1e-9, atol=1e-12)


@settings(max_examples=examples(25), deadline=None)
@given(st.integers(0, 10_000), st.integers(1, 4), st.integers(1, 4))
def test_pivots_match_naive_definition(seed, left, right):
    h = _bars(seed, decimals=1).high  # coarse prices: ties occur
    got = ta.pivothigh(h, left, right)
    for t in range(h.size):
        c = t - right
        ok = (
            c - left >= 0
            and all(h[c] > h[c - i] for i in range(1, left + 1))
            and all(h[c] >= h[c + i] for i in range(1, right + 1))
        )
        assert (got[t] == h[c]) if ok else np.isnan(got[t])


@pytest.mark.parametrize("policy", ta.NA_POLICIES)
def test_windowed_na_policies(policy):
    x = _gaps(_bars(5).close)
    n = 6
    for fn, ref in [(ta.math_sum, np.sum), (ta.variance, np.var)]:
        got = fn(x, n, na_policy=policy)
        if policy == "propagate":
            np.testing.assert_allclose(got, _naive_window(x, n, ref), rtol=1e-9, atol=1e-12)
        else:
            # Window over the last n valid values; na bars per policy.
            valid = ~np.isnan(x)
            assert np.isnan(got[~valid]).all() == (policy == "skip_na_out")
            t = np.flatnonzero(valid)[n + 5]
            last = x[: t + 1][~np.isnan(x[: t + 1])][-n:]
            assert got[t] == pytest.approx(ref(last))


def test_hma_uses_floored_lengths():
    x = _bars(7).close
    for n in (9, 14):
        inner = 2 * ta.wma(x, n // 2) - ta.wma(x, n)
        np.testing.assert_array_equal(ta.hma(x, n), ta.wma(inner, math.floor(math.sqrt(n))))


def test_argument_validation():
    x = np.arange(5.0)
    with pytest.raises(ValueError, match="length"):
        ta.wma(x, 0)
    with pytest.raises(ValueError, match="percentage"):
        ta.percentile_linear_interpolation(x, 3, 101)
    with pytest.raises(ValueError, match="hma length"):
        ta.hma(x, 1)
    with pytest.raises(ValueError, match="unbiased variance"):
        ta.variance(x, 1, biased=False)
    with pytest.raises(ValueError, match="same length"):
        ta.obv(x, x[:-1])
    with pytest.raises(ValueError, match="na_policy"):
        ta.math_sum(x, 2, na_policy="ignore")
    with pytest.raises(ValueError, match="leftbars"):
        rb.Pivothigh(0, 2)


# ---------- runtime == pine.ta ----------


@settings(max_examples=examples(15), deadline=None)
@given(st.integers(0, 10_000), st.integers(2, 12), st.booleans())
def test_runtime_equals_vectorised_bit_for_bit(seed, n, gaps):
    b = _bars(seed, decimals=2)
    x = _gaps(b.close) if gaps else np.asarray(b.close)

    def script(ctx):
        v = x[ctx.i]
        t = ctx.ta
        ctx.plot("sum", t.sum("sum", v, n))
        ctx.plot("wma", t.wma("wma", v, n))
        ctx.plot("hma", t.hma("hma", v, n))
        ctx.plot("var", t.variance("var", v, n))
        ctx.plot("pct", t.percentile_linear_interpolation("pct", v, n, 30))
        ctx.plot("ph", t.pivothigh("ph", ctx.high.value, 2, 3))
        ctx.plot("pl", t.pivotlow("pl", ctx.low.value, 3, 1))
        ctx.plot("fix", t.fixnan("fix", v))
        ctx.plot("cum", t.cum("cum", v))
        ctx.plot("obv", t.obv("obv"))
        ctx.plot("pvt", t.pvt("pvt"))

    r = run(script, b)
    eq = np.testing.assert_array_equal
    eq(r["sum"], ta.math_sum(x, n))
    eq(r["wma"], ta.wma(x, n))
    eq(r["hma"], ta.hma(x, n))
    eq(r["var"], ta.variance(x, n))
    eq(r["pct"], ta.percentile_linear_interpolation(x, n, 30))
    eq(r["ph"], ta.pivothigh(b.high, 2, 3))
    eq(r["pl"], ta.pivotlow(b.low, 3, 1))
    eq(r["fix"], ta.fixnan(x))
    eq(r["cum"], ta.cum(x))
    eq(r["obv"], ta.obv(b.close, b.volume))
    eq(r["pvt"], ta.pvt(b.close, b.volume))
