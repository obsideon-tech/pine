"""Oracles (a) closed-form and (b) property tests for pine.ta (PINE_PLAN.md §5.8).

Oracle (c), TradingView reference values, is pending a real export.
"""

import math

import numpy as np
import pytest
from hyp import examples
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pine import ta

MAS = {"sma": ta.sma, "ema": ta.ema, "rma": ta.rma}
LENGTHS = [1, 2, 5, 14]


# ---------- warm-up / na counts ----------


@pytest.mark.parametrize("name", MAS)
@pytest.mark.parametrize("n", LENGTHS)
def test_ma_warmup_is_length_minus_one(name, n):
    out = MAS[name](np.arange(50.0), n)
    assert np.isnan(out[: n - 1]).all() and not np.isnan(out[n - 1 :]).any()


@pytest.mark.parametrize("n", LENGTHS)
def test_rsi_warmup_is_length(n):
    x = np.sin(np.arange(60.0)) + 10
    out = ta.rsi(x, n)
    assert np.isnan(out[:n]).all() and not np.isnan(out[n:]).any()


@pytest.mark.parametrize("fn", [ta.sma, ta.ema, ta.rma, ta.rsi])
def test_shorter_than_length_is_all_na(fn):
    assert np.isnan(fn(np.arange(5.0), 10)).all()


@pytest.mark.parametrize("fn", [ta.sma, ta.ema, ta.rma, ta.rsi])
@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "14"])
def test_bad_length_rejected(fn, bad):
    with pytest.raises(ValueError):
        fn(np.arange(10.0), bad)


# ---------- closed form ----------


@pytest.mark.parametrize("name", MAS)
@pytest.mark.parametrize("n", LENGTHS)
def test_constant_series_gives_constant(name, n):
    out = MAS[name](np.full(40, 7.25), n)
    np.testing.assert_allclose(out[n - 1 :], 7.25, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", MAS)
def test_length_one_is_identity(name):
    x = np.random.default_rng(1).normal(size=30)
    np.testing.assert_allclose(MAS[name](x, 1), x, rtol=0, atol=1e-15)


@pytest.mark.parametrize("n", LENGTHS)
def test_sma_of_ramp_lags_by_half_window(n):
    t = np.arange(60.0)
    np.testing.assert_allclose(ta.sma(t, n)[n - 1 :], t[n - 1 :] - (n - 1) / 2, atol=1e-12)


@pytest.mark.parametrize("n", LENGTHS)
def test_ema_of_ramp_is_exact_from_seed(n):
    # With alpha = 2/(n+1) the steady-state lag (1-alpha)/alpha = (n-1)/2 equals the SMA seed's
    # lag, so an SMA-seeded EMA of a ramp is t - (n-1)/2 from its first bar. A first-value
    # seed would not satisfy this at the seed bar: the test checks seeding, not only recursion.
    t = np.arange(80.0)
    np.testing.assert_allclose(ta.ema(t, n)[n - 1 :], t[n - 1 :] - (n - 1) / 2, atol=1e-10)


@pytest.mark.parametrize("n", [2, 5, 14])
def test_rma_of_ramp_closed_form(n):
    # Steady-state lag n-1; the SMA seed starts (n-1)/2 above it and the gap decays by (1-1/n).
    t = np.arange(120.0)
    k = t[n - 1 :] - (n - 1)
    expected = t[n - 1 :] - (n - 1) + (n - 1) / 2 * (1 - 1 / n) ** k
    np.testing.assert_allclose(ta.rma(t, n)[n - 1 :], expected, atol=1e-10)


def test_rsi_monotone_series():
    up, down = np.arange(30.0), -np.arange(30.0)
    assert (ta.rsi(up, 14)[14:] == 100).all()
    assert (ta.rsi(down, 14)[14:] == 0).all()


def test_rsi_flat_series_follows_down_zero_branch():
    # down == 0 is tested first in the manual's code, so a flat series gives 100 (unverified).
    assert (ta.rsi(np.full(30, 5.0), 14)[14:] == 100).all()


def test_rsi_hand_computed_length_2():
    x = np.array([1.0, 2.0, 1.0, 3.0])
    # u = [na,1,0,2], d = [na,0,1,0]; rma len 2 (alpha .5), sma seed at bar 2.
    # bar 2: up=.5, down=.5 -> 50. bar 3: up=1.25, down=.25 -> 100-100/6
    out = ta.rsi(x, 2)
    assert np.isnan(out[:2]).all()
    assert out[2] == pytest.approx(50.0, abs=1e-12)
    assert out[3] == pytest.approx(100 - 100 / 6, abs=1e-12)


# ---------- na propagation and re-seeding ----------


def test_sma_na_in_window_propagates():
    x = np.arange(20.0)
    x[10] = np.nan
    out = ta.sma(x, 3)
    assert np.isnan(out[10:13]).all() and not np.isnan(out[13:]).any()


@pytest.mark.parametrize("fn", [ta.ema, ta.rma])
def test_recursive_ma_reseeds_from_sma_after_na(fn):
    x = np.arange(30.0)
    x[10] = np.nan
    out = fn(x, 3)
    assert np.isnan(out[10:13]).all()
    assert out[13] == pytest.approx(ta.sma(x, 3)[13])  # fresh SMA seed


def test_inputs_not_mutated():
    x = np.arange(20.0)
    before = x.copy()
    for f in (ta.sma, ta.ema, ta.rma, ta.rsi):
        f(x, 5)
    np.testing.assert_array_equal(x, before)


# ---------- properties ----------

series = st.lists(
    st.floats(min_value=1.0, max_value=1e4, allow_nan=False), min_size=20, max_size=60
).map(np.array)
lengths = st.integers(min_value=1, max_value=15)


@settings(max_examples=examples(60), deadline=None)
@given(series, lengths, st.floats(min_value=-1e3, max_value=1e3))
def test_ma_translation_equivariance(x, n, c):
    for f in MAS.values():
        np.testing.assert_allclose(f(x + c, n), f(x, n) + c, rtol=1e-9, atol=1e-8)


@settings(max_examples=examples(60), deadline=None)
@given(series, lengths, st.floats(min_value=1e-3, max_value=1e3))
def test_ma_scale_equivariance(x, n, k):
    for f in MAS.values():
        np.testing.assert_allclose(f(k * x, n), k * f(x, n), rtol=1e-9, atol=1e-9)


@settings(max_examples=examples(60), deadline=None)
@given(series, lengths)
def test_ma_bounded_by_window_range(x, n):
    # Every MA here is a convex combination of past values: within [min, max] of the input.
    for f in MAS.values():
        out = f(x, n)[n - 1 :]
        assert (out >= x.min() - 1e-9).all() and (out <= x.max() + 1e-9).all()


@settings(max_examples=examples(60), deadline=None)
@given(series, lengths, st.floats(min_value=1e-2, max_value=1e2), st.floats(-1e3, 1e3))
def test_rsi_positive_affine_invariance_and_bounds(x, n, a, b):
    r = ta.rsi(x, n)
    assert ((r[n:] >= 0) & (r[n:] <= 100)).all()
    y = a * x + b
    # Invariance holds in exact arithmetic; in float64 the transform can round a tiny move to
    # zero (found by a random run: near-constant x, b = 1), flipping RSI's 0/100 branches.
    # Require that every bar-to-bar move keeps its sign and is not destroyed by rounding.
    dx, dy = np.diff(x), np.diff(y)
    assume(np.array_equal(np.sign(dx), np.sign(dy)))
    assume(np.all((dx == 0) | (np.abs(dy) > 1e-9 * np.abs(y[1:]))))
    np.testing.assert_allclose(ta.rsi(y, n), r, atol=1e-7)


@settings(max_examples=examples(60), deadline=None)
@given(series, lengths)
def test_rsi_reflection_sums_to_100(x, n):
    r1, r2 = ta.rsi(x, n), ta.rsi(-x, n)
    # Holds where neither branch (up == 0 or down == 0) fires.
    d = np.diff(x)
    assume(np.count_nonzero(d > 0) and np.count_nonzero(d < 0))
    ok = (r1 > 0) & (r1 < 100)
    np.testing.assert_allclose(r1[ok] + r2[ok], 100.0, atol=1e-8)
    assert math.isnan(r1[0])


# ======================= stdev =======================


def test_stdev_hand_values():
    x = [1.0, 2.0, 3.0, 4.0]
    assert ta.stdev(x, 4)[-1] == pytest.approx(math.sqrt(1.25), abs=1e-15)
    assert ta.stdev(x, 4, biased=False)[-1] == pytest.approx(math.sqrt(5 / 3), abs=1e-15)
    assert np.isnan(ta.stdev(x, 4)[:3]).all()


def test_stdev_constant_is_exactly_zero():
    # The 1e-10 deviation zeroing makes this exact even when the SMA has rounding error.
    x = np.full(40, 0.1)
    assert (ta.stdev(x, 7)[6:] == 0.0).all()


def test_stdev_length_one_and_unbiased_guard():
    assert (ta.stdev(np.arange(5.0), 1) == 0).all()
    with pytest.raises(ValueError):
        ta.stdev(np.arange(5.0), 1, biased=False)


@settings(max_examples=examples(60), deadline=None)
@given(series, st.integers(2, 15), st.floats(-1e3, 1e3), st.floats(1e-2, 1e2))
def test_stdev_properties(x, n, c, k):
    s = ta.stdev(x, n)
    np.testing.assert_allclose(ta.stdev(x + c, n), s, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ta.stdev(k * x, n), k * s, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ta.stdev(-x, n), s, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        ta.stdev(x, n, biased=False), s * math.sqrt(n / (n - 1)), rtol=1e-9, atol=1e-9
    )
    assert (s[n - 1 :] >= 0).all()


# ======================= highest / lowest =======================


def test_highest_lowest_on_ramp():
    t = np.arange(30.0)
    np.testing.assert_array_equal(ta.highest(t, 5)[4:], t[4:])
    np.testing.assert_array_equal(ta.lowest(t, 5)[4:], t[:-4])
    assert np.isnan(ta.highest(t, 5)[:4]).all() and np.isnan(ta.lowest(t, 5)[:4]).all()


def test_highest_na_in_window_propagates():
    x = np.arange(20.0)
    x[10] = np.nan
    out = ta.highest(x, 3)
    assert np.isnan(out[10:13]).all() and not np.isnan(out[13:]).any()


@settings(max_examples=examples(60), deadline=None)
@given(series, lengths)
def test_highest_lowest_properties(x, n):
    hi, lo = ta.highest(x, n), ta.lowest(x, n)
    np.testing.assert_array_equal(ta.highest(-x, n), -lo)
    w = slice(n - 1, None)
    assert (lo[w] <= x[w]).all() and (x[w] <= hi[w]).all()
    assert (lo[w] <= ta.sma(x, n)[w] + 1e-9).all() and (ta.sma(x, n)[w] <= hi[w] + 1e-9).all()
    if n == 1:
        np.testing.assert_array_equal(hi, x)


# ======================= tr / atr =======================


def _bars(rng, n=60, gap=1.0):
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    o = np.concatenate([[100.0], c[:-1]]) + rng.normal(0, gap, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.3, n))
    lo = np.minimum(o, c) - np.abs(rng.normal(0, 0.3, n))
    return h, lo, c


def test_tr_first_bar_and_gap_terms():
    h, lo, c = np.array([10.0, 15.0]), np.array([9.0, 14.0]), np.array([9.5, 14.5])
    assert np.isnan(ta.tr(h, lo, c)[0])
    assert ta.tr(h, lo, c, handle_na=True)[0] == 1.0
    assert ta.tr(h, lo, c)[1] == 15.0 - 9.5  # gap up: |high - close[1]| dominates


def test_atr_constant_range_without_gaps():
    n = 40
    c = np.full(n, 100.0)
    h, lo = c + 1.5, c - 1.5
    out = ta.atr(h, lo, c, 14)
    assert np.isnan(out[:13]).all()
    np.testing.assert_allclose(out[13:], 3.0, atol=1e-12)


def test_atr_equals_rma_of_tr_with_first_bar_high_low():
    h, lo, c = _bars(np.random.default_rng(5))
    expected = ta.rma(ta.tr(h, lo, c, handle_na=True), 14)
    np.testing.assert_allclose(ta.atr(h, lo, c, 14), expected, rtol=0, atol=1e-12)


def test_atr_na_close_makes_next_true_range_na():
    h, lo, c = _bars(np.random.default_rng(6))
    c[30] = np.nan
    out = ta.atr(h, lo, c, 5)
    assert not np.isnan(out[30]) and np.isnan(out[31:36]).all() and not np.isnan(out[36])


@settings(max_examples=examples(40), deadline=None)
@given(st.integers(0, 10_000), st.integers(1, 15), st.floats(-50, 50), st.floats(0.1, 10))
def test_atr_translation_and_scale(seed, n, c0, k):
    h, lo, c = _bars(np.random.default_rng(seed))
    a = ta.atr(h, lo, c, n)
    np.testing.assert_allclose(ta.atr(h + c0, lo + c0, c + c0, n), a, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(ta.atr(k * h, k * lo, k * c, n), k * a, rtol=1e-9, atol=1e-9)
    assert (a[n - 1 :] > 0).all()


# ======================= crossover / crossunder =======================


def test_crossover_hand_example():
    a = [1.0, 2.0, 1.0, 1.5, 3.0]
    b = [1.5, 1.5, 1.5, 1.5, 1.5]
    # a[3] == b[3] is not "above"; bar 4 crosses from equal (a[1] <= b[1]) to above.
    np.testing.assert_array_equal(ta.crossover(a, b), [False, True, False, False, True])
    np.testing.assert_array_equal(ta.crossunder(a, b), [False, False, True, False, False])


def test_crossover_scalar_level_and_na_is_false():
    a = np.array([20.0, 35.0, np.nan, 25.0, 31.0])
    np.testing.assert_array_equal(ta.crossover(a, 30), [False, True, False, False, True])
    assert ta.crossover(a, 30).dtype == bool


@settings(max_examples=examples(60), deadline=None)
@given(series, series)
def test_crossover_properties(a, b):
    m = min(a.size, b.size)
    a, b = a[:m], b[:m]
    co, cu = ta.crossover(a, b), ta.crossunder(a, b)
    np.testing.assert_array_equal(co, ta.crossunder(b, a))
    assert not co[0] and not cu[0]
    assert not (co & cu).any()
    assert not (co[1:] & co[:-1]).any()  # cannot cross over on two consecutive bars


# ======================= na_policy (PINE_PLAN §10 A2) =======================

NAN = np.nan


@pytest.mark.parametrize(
    "policy, expected",
    [
        ("propagate", [NAN, 1.5, NAN, NAN, 4.5]),
        ("skip", [NAN, 1.5, 1.5, 3.0, 4.5]),
        ("skip_na_out", [NAN, 1.5, NAN, 3.0, 4.5]),
    ],
)
def test_sma_na_policies_hand_values(policy, expected):
    out = ta.sma([1.0, 2.0, NAN, 4.0, 5.0], 2, na_policy=policy)
    np.testing.assert_allclose(out, expected, equal_nan=True)


@pytest.mark.parametrize(
    "policy, expected",
    [
        ("propagate", [NAN, 1.5, NAN, NAN]),
        ("skip", [NAN, 1.5, 1.5, 2 / 3 * 4 + 1 / 3 * 1.5]),
        ("skip_na_out", [NAN, 1.5, NAN, 2 / 3 * 4 + 1 / 3 * 1.5]),
    ],
)
def test_ema_na_policies_carry_state(policy, expected):
    np.testing.assert_allclose(
        ta.ema([1.0, 2.0, NAN, 4.0], 2, na_policy=policy), expected, equal_nan=True
    )


def test_highest_and_stdev_skip():
    x = [1.0, 3.0, NAN, 2.0]
    np.testing.assert_array_equal(ta.highest(x, 2, na_policy="skip"), [NAN, 3, 3, 3])
    np.testing.assert_array_equal(ta.lowest(x, 2, na_policy="skip_na_out"), [NAN, 1, NAN, 2])
    np.testing.assert_allclose(
        ta.stdev(x, 2, na_policy="skip"), [NAN, 1.0, 1.0, 0.5], equal_nan=True
    )


@pytest.mark.parametrize("fn", [ta.sma, ta.ema, ta.rma, ta.rsi, ta.stdev, ta.highest, ta.lowest])
def test_unknown_policy_rejected(fn):
    with pytest.raises(ValueError, match="na_policy"):
        fn(np.arange(10.0), 3, na_policy="ignore")


@settings(max_examples=examples(40), deadline=None)
@given(series, lengths)
def test_policies_identical_on_gap_free_input(x, n):
    for fn in (ta.sma, ta.ema, ta.rma, ta.rsi, ta.stdev, ta.highest, ta.lowest):
        ref = fn(x, n)
        for p in ("skip", "skip_na_out"):
            np.testing.assert_array_equal(fn(x, n, na_policy=p), ref)


# ======================= change / barssince / valuewhen / supertrend =======================


def test_change_barssince_valuewhen_hand_values():
    np.testing.assert_array_equal(ta.change([1.0, 2.0, 4.0, NAN, 5.0]), [NAN, 1, 2, NAN, NAN])
    np.testing.assert_array_equal(ta.change([1.0, 2.0, 4.0], 2), [NAN, NAN, 3])
    np.testing.assert_array_equal(ta.barssince([0, 1, 0, 0, 1, 0]), [NAN, 0, 1, 2, 0, 1])
    np.testing.assert_array_equal(
        ta.valuewhen([0, 1, 0, 1, 0], [1.0, 2, 3, 4, 5], 1), [NAN, NAN, NAN, 2, 2]
    )
    with pytest.raises(ValueError):
        ta.valuewhen([1], [1.0], -1)
    with pytest.raises(ValueError):
        ta.valuewhen([1, 0], [1.0], 0)


def test_supertrend_properties():
    rng = np.random.default_rng(8)
    c = 100 + np.cumsum(rng.normal(0, 1, 300))
    h, lo = c + np.abs(rng.normal(0, 0.5, 300)), c - np.abs(rng.normal(0, 0.5, 300))
    line, d = ta.supertrend(h, lo, c, 3.0, 10)
    assert set(np.unique(d)) == {-1.0, 1.0}
    a = ta.atr(h, lo, c, 10)
    ok = np.arange(300) >= 11
    up = ok & (d == -1)
    dn = ok & (d == 1)
    # Uptrend: the line is a lower band and never sits above that bar's high.
    assert np.all(line[up] <= h[up])
    # The line never moves against the trend while the direction is unchanged.
    same = ok[1:] & (d[1:] == d[:-1]) & ok[:-1]
    rising_ok = np.where(d[1:] == -1, line[1:] >= line[:-1] - 1e-12, line[1:] <= line[:-1] + 1e-12)
    assert np.all(rising_ok[same])
    assert np.all(np.isfinite(line[ok])) and np.isfinite(a[ok]).all()
    assert dn.any() and up.any()
