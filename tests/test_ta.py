"""Oracles (a) closed-form and (b) property tests for pine.ta (PINE_PLAN.md §5.8).

Oracle (c), TradingView reference values, is pending a real export.
"""

import math

import numpy as np
import pytest
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


@settings(max_examples=60, deadline=None)
@given(series, lengths, st.floats(min_value=-1e3, max_value=1e3))
def test_ma_translation_equivariance(x, n, c):
    for f in MAS.values():
        np.testing.assert_allclose(f(x + c, n), f(x, n) + c, rtol=1e-9, atol=1e-8)


@settings(max_examples=60, deadline=None)
@given(series, lengths, st.floats(min_value=1e-3, max_value=1e3))
def test_ma_scale_equivariance(x, n, k):
    for f in MAS.values():
        np.testing.assert_allclose(f(k * x, n), k * f(x, n), rtol=1e-9, atol=1e-9)


@settings(max_examples=60, deadline=None)
@given(series, lengths)
def test_ma_bounded_by_window_range(x, n):
    # Every MA here is a convex combination of past values: within [min, max] of the input.
    for f in MAS.values():
        out = f(x, n)[n - 1 :]
        assert (out >= x.min() - 1e-9).all() and (out <= x.max() + 1e-9).all()


@settings(max_examples=60, deadline=None)
@given(series, lengths, st.floats(min_value=1e-2, max_value=1e2), st.floats(-1e3, 1e3))
def test_rsi_positive_affine_invariance_and_bounds(x, n, a, b):
    r = ta.rsi(x, n)
    assert ((r[n:] >= 0) & (r[n:] <= 100)).all()
    np.testing.assert_allclose(ta.rsi(a * x + b, n), r, atol=1e-7)


@settings(max_examples=60, deadline=None)
@given(series, lengths)
def test_rsi_reflection_sums_to_100(x, n):
    r1, r2 = ta.rsi(x, n), ta.rsi(-x, n)
    # Holds where neither branch (up == 0 or down == 0) fires.
    d = np.diff(x)
    assume(np.count_nonzero(d > 0) and np.count_nonzero(d < 0))
    ok = (r1 > 0) & (r1 < 100)
    np.testing.assert_allclose(r1[ok] + r2[ok], 100.0, atol=1e-8)
    assert math.isnan(r1[0])
