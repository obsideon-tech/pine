"""Input-rounding bounds (PINE_PLAN.md §5.1, §10 A3): sound, not vacuous, and wired in."""

import math

import numpy as np
import pytest
from hyp import examples
from hypothesis import given, settings
from hypothesis import strategies as st

from pine import ta
from pine.bars import Bars
from pine.io import TradingViewExport
from pine.parity import PortSpec, Tolerance, check_export, compare_series
from pine.sensitivity import LIPSCHITZ, input_rounding, rsi_input_error, stdev_lipschitz

EPS = 1e-9  # float slack, scaled by magnitude below


def _bars(rng, n=80, scale=100.0, gap=1.0):
    c = scale + np.cumsum(rng.normal(0, 1, n))
    o = np.concatenate([[scale], c[:-1]]) + rng.normal(0, gap, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.3, n))
    lo = np.minimum(o, c) - np.abs(rng.normal(0, 0.3, n))
    return o, h, lo, c


def _perturb(rng, shape, delta, mode):
    if mode == "uniform":
        return rng.uniform(-delta, delta, shape)
    if mode == "plus":
        return np.full(shape, delta)
    sign = np.where(np.arange(shape[-1]) % 2 == 0, 1.0, -1.0)  # alternating, worst for spread
    return delta * np.broadcast_to(sign, shape)


MODES = ("uniform", "plus", "alternating")
SINGLE = {
    "sma": (ta.sma, LIPSCHITZ["sma"]),
    "ema": (ta.ema, LIPSCHITZ["ema"]),
    "rma": (ta.rma, LIPSCHITZ["rma"]),
    "highest": (ta.highest, LIPSCHITZ["highest"]),
    "lowest": (ta.lowest, LIPSCHITZ["lowest"]),
    "stdev": (ta.stdev, LIPSCHITZ["stdev"]),
    "stdev_sample": (lambda x, n: ta.stdev(x, n, biased=False), None),
}


@settings(max_examples=examples(80), deadline=None)
@given(
    st.integers(0, 10_000),
    st.integers(2, 20),
    st.floats(1e-6, 1e-1),
    st.sampled_from(MODES),
    st.sampled_from(sorted(SINGLE)),
    st.sampled_from(ta.NA_POLICIES),
)
def test_single_input_bounds_are_sound(seed, n, delta, mode, name, policy):
    rng = np.random.default_rng(seed)
    x = 100 + np.cumsum(rng.normal(0, 1, 60))
    if policy != "propagate":
        x[[20, 21, 40]] = np.nan
    e = _perturb(rng, x.shape, delta, mode)
    fn, L = SINGLE[name]
    L = stdev_lipschitz(n, biased=False) if L is None else L
    kw = {} if name == "stdev_sample" else {"na_policy": policy}
    if name == "stdev_sample" and policy != "propagate":
        return
    diff = np.abs(fn(x + e, n, **kw) - fn(x, n, **kw))
    assert np.nanmax(diff, initial=0) <= L * delta + EPS * 100


@settings(max_examples=examples(60), deadline=None)
@given(st.integers(0, 10_000), st.integers(1, 20), st.floats(1e-6, 1e-1), st.sampled_from(MODES))
def test_tr_and_atr_bounds_are_sound(seed, n, delta, mode):
    rng = np.random.default_rng(seed)
    _, h, lo, c = _bars(rng)
    e = _perturb(rng, (3, h.size), delta, mode) * np.array([[1], [-1], [1]])  # worst-ish signs
    if mode == "uniform":
        e = rng.uniform(-delta, delta, (3, h.size))
    h2, lo2, c2 = h + e[0], lo + e[1], c + e[2]
    for f in (lambda *a: ta.tr(*a, handle_na=True), lambda *a: ta.atr(*a, n)):
        diff = np.abs(f(h2, lo2, c2) - f(h, lo, c))
        assert np.nanmax(diff) <= 2 * delta + EPS * 100


@settings(max_examples=examples(80), deadline=None)
@given(
    st.integers(0, 10_000),
    st.integers(2, 20),
    st.floats(1e-6, 1e-1),
    st.sampled_from(MODES),
    st.floats(0.001, 2.0),
)
def test_rsi_per_bar_bound_is_sound(seed, n, delta, mode, vol):
    rng = np.random.default_rng(seed)
    x = 100 + np.cumsum(rng.normal(0, vol, 60))  # small vol: quiet market, small U+D
    x_rounded_side = x + _perturb(rng, x.shape, delta, mode)
    bound = rsi_input_error(x_rounded_side, n, delta)  # computed from the port's side
    diff = np.abs(ta.rsi(x_rounded_side, n) - ta.rsi(x, n))
    ok = ~np.isnan(diff)
    assert np.all(diff[ok] <= bound[ok] + 1e-7)


# ---------- tightness: the bounds are attained, so they are not vacuous ----------


def test_sma_bound_attained_by_constant_shift():
    x = np.linspace(100, 110, 40)
    assert np.nanmax(np.abs(ta.sma(x + 0.005, 10) - ta.sma(x, 10))) == pytest.approx(0.005)


def test_tr_bound_attained_on_a_gap():
    h, lo, c = np.array([10.0, 15.0]), np.array([9.0, 14.0]), np.array([9.5, 14.5])
    d = 0.005
    tr0 = ta.tr(h, lo, c)[1]  # gap up: high - close[1]
    tr1 = ta.tr(h + d, lo, c - d)[1]
    assert tr1 - tr0 == pytest.approx(2 * d)


def test_rsi_has_no_useful_global_bound_in_quiet_markets():
    # Moves of a few ticks: one tick of rounding shifts RSI by several points.
    rng = np.random.default_rng(3)
    x = 100 + np.cumsum(rng.choice([-0.01, 0.0, 0.01], 80))
    r = ta.rsi(x, 14)
    r2 = ta.rsi(x + np.where(np.arange(80) % 2 == 0, 0.005, -0.005), 14)
    assert np.nanmax(np.abs(r2 - r)) > 1.0
    assert np.nanmax(rsi_input_error(x, 14, 0.005)) > 1.0


def test_helpers():
    assert input_rounding(2) == 0.005 and input_rounding(0) == 0.5
    assert stdev_lipschitz(4, biased=False) == pytest.approx(math.sqrt(4 / 3))
    assert stdev_lipschitz(4) == 1.0
    with pytest.raises(ValueError):
        input_rounding(-1)
    with pytest.raises(ValueError):
        rsi_input_error([1.0, 2.0], 2, -1.0)
    flat = rsi_input_error(np.full(30, 5.0), 14, 0.005)
    assert np.all(flat[14:] == 100.0)  # U + D = 0: only the range bound applies


# ---------- integration with check_export ----------


def _rounded_export(seed=11, decimals=2):
    """Reference computed on unrounded prices; export prints prices rounded to 2 decimals."""
    rng = np.random.default_rng(seed)
    o, h, lo, c = _bars(rng, n=200)
    ref = {
        "sma": ta.sma(c, 14),
        "ema": ta.ema(c, 14),
        "rsi": ta.rsi(c, 14),
        "atr": ta.atr(h, lo, c, 14),
        "stdev": ta.stdev(c, 14),
    }
    r = lambda a: np.round(a, decimals)  # noqa: E731
    o2, c2 = r(o), r(c)
    h2 = np.maximum(r(h), np.maximum(o2, c2))
    lo2 = np.minimum(r(lo), np.minimum(o2, c2))
    bars = Bars(np.arange(200) * 60, o2, h2, lo2, c2, np.full(200, np.nan), timeframe="1")
    observed = {k: decimals for k in ("open", "high", "low", "close")} | {k: 15 for k in ref}
    return TradingViewExport(bars, ref, ("time", "open", "high", "low", "close", *ref), "unix",
                             observed_decimals=observed)  # fmt: skip


SPECS = {
    "sma": PortSpec(lambda b: ta.sma(b.close, 14), input_error=LIPSCHITZ["sma"]),
    "ema": PortSpec(lambda b: ta.ema(b.close, 14), input_error=LIPSCHITZ["ema"]),
    "rsi": PortSpec(
        lambda b: ta.rsi(b.close, 14),
        kind="bounded",
        input_error=lambda b, d: rsi_input_error(b.close, 14, d),
    ),
    "atr": PortSpec(lambda b: ta.atr(b.high, b.low, b.close, 14), input_error=LIPSCHITZ["atr"]),
    "stdev": PortSpec(lambda b: ta.stdev(b.close, 14), input_error=LIPSCHITZ["stdev"]),
}


def test_rounded_inputs_fail_without_allowance_and_pass_with_it():
    exp = _rounded_export()
    assert not check_export(exp, SPECS).passed
    rep = check_export(exp, SPECS, input_rounding=input_rounding(2))
    assert rep.passed, rep.to_markdown()
    assert "inputs assumed rounded" in rep.to_markdown()
    assert "(per bar)" in rep.to_markdown()  # RSI's allowance is per bar


def test_allowance_does_not_mask_a_seed_bug():
    exp = _rounded_export()

    def bad_ema(b):  # first-value seed instead of the SMA seed
        x, n = np.asarray(b.close), 14
        a, out = 2 / (n + 1), np.full(x.size, np.nan)
        out[n - 1] = x[n - 1]
        for t in range(n, x.size):
            out[t] = a * x[t] + (1 - a) * out[t - 1]
        return out

    spec = {"ema": PortSpec(bad_ema, input_error=1.0)}
    rep = check_export(exp, spec, input_rounding=input_rounding(2))
    assert not rep.passed and rep.columns[0].first_divergence == 13


def test_input_rounding_requires_input_error_on_every_spec():
    exp = _rounded_export()
    with pytest.raises(ValueError, match="no input_error"):
        check_export(exp, {"sma": PortSpec(lambda b: ta.sma(b.close, 14))}, input_rounding=0.005)


def test_per_bar_tolerance_validation():
    with pytest.raises(ValueError):
        Tolerance(atol=np.array([0.1, -0.1]), rtol=0)
    with pytest.raises(ValueError):
        Tolerance(atol=np.array([0.1, np.nan]), rtol=0)
    t = Tolerance(atol=np.array([0.0, 1.0]), rtol=0)
    assert compare_series([1.0, 2.5], [1.0, 2.0], tolerance=t).passed
    assert not compare_series([1.5, 2.0], [1.0, 2.0], tolerance=t).passed


# ---------- batch F (PINE_PLAN.md §11 F) ----------

from pine.sensitivity import (  # noqa: E402
    cum_input_error,
    obv_input_error,
    pvt_input_error,
    sum_lipschitz,
    variance_input_error,
)

BATCH_F = {
    "math_sum": (ta.math_sum, lambda n: sum_lipschitz(n)),
    "wma": (ta.wma, lambda n: LIPSCHITZ["wma"]),
    "hma": (ta.hma, lambda n: LIPSCHITZ["hma"]),
    "percentile": (
        lambda x, n, na_policy=None: ta.percentile_linear_interpolation(x, n, 30, na_policy),
        lambda n: LIPSCHITZ["percentile_linear_interpolation"],
    ),
}


def _gapped(rng, n=70):
    x = 100 + np.cumsum(rng.normal(0, 1, n))
    x[rng.choice(n, 6, replace=False)] = np.nan
    return x


@settings(max_examples=examples(60), deadline=None)
@given(
    st.integers(0, 10_000),
    st.integers(2, 15),
    st.floats(1e-6, 1e-1),
    st.sampled_from(MODES),
    st.sampled_from(sorted(BATCH_F)),
    st.sampled_from(ta.NA_POLICIES),
)
def test_batch_f_lipschitz_bounds_are_sound(seed, n, delta, mode, name, policy):
    rng = np.random.default_rng(seed)
    x = _gapped(rng)
    fn, lip = BATCH_F[name]
    y = fn(x, n, na_policy=policy)
    y2 = fn(x + _perturb(rng, x.shape, delta, mode), n, na_policy=policy)
    np.testing.assert_array_equal(np.isnan(y), np.isnan(y2))
    err = np.nanmax(np.abs(y - y2), initial=0.0)
    assert err <= lip(n) * delta * (1 + EPS) + EPS * np.nanmax(np.abs(y), initial=1.0)


@settings(max_examples=examples(60), deadline=None)
@given(
    st.integers(0, 10_000),
    st.integers(2, 15),
    st.floats(1e-6, 1e-1),
    st.sampled_from(MODES),
    st.booleans(),
    st.sampled_from(ta.NA_POLICIES),
)
def test_variance_bound_is_sound(seed, n, delta, mode, biased, policy):
    rng = np.random.default_rng(seed)
    x = _gapped(rng)
    x2 = x + _perturb(rng, x.shape, delta, mode)  # x2: the port's (rounded) side
    exact = ta.variance(x, n, biased, policy)
    port = ta.variance(x2, n, biased, policy)
    bound = variance_input_error(x2, n, delta, biased, policy)
    ok = ~np.isnan(exact)
    assert (np.abs(exact - port)[ok] <= bound[ok] * (1 + 1e-9) + 1e-9).all()


@settings(max_examples=examples(60), deadline=None)
@given(st.integers(0, 10_000), st.floats(1e-6, 0.5), st.sampled_from(MODES))
def test_running_total_bounds_are_sound(seed, delta, mode):
    rng = np.random.default_rng(seed)
    n = 80
    c = np.round(20 + np.cumsum(rng.normal(0, 0.3, n)), 1)  # coarse: flat closes occur
    v = rng.integers(1, 1000, n).astype(float)
    port = c + _perturb(rng, c.shape, delta, mode)
    slack = 1e-9
    for exact, approx, bound in [
        (ta.cum(c), ta.cum(port), cum_input_error(port, delta)),
        (ta.obv(c, v), ta.obv(port, v), obv_input_error(port, v, delta)),
        (ta.pvt(c, v), ta.pvt(port, v), pvt_input_error(port, v, delta)),
    ]:
        ok = ~np.isnan(exact)
        assert (np.abs(exact - approx)[ok] <= bound[ok] * (1 + slack) + slack).all()


def test_obv_bound_is_attained_by_a_sign_flip():
    # A flat close rounded up by δ: obv's term flips from 0 to +volume.
    c, v = np.array([10.0, 10.0]), np.array([5.0, 7.0])
    assert ta.obv(c + [0.0, 0.004], v)[1] - ta.obv(c, v)[1] == 7.0
    assert obv_input_error(c, v, 0.005)[1] == 14.0


def test_batch_f_bound_validation():
    with pytest.raises(ValueError):
        sum_lipschitz(0)
    for f in (
        lambda: variance_input_error([1.0, 2.0], 2, -1),
        lambda: cum_input_error([1.0], -1),
        lambda: obv_input_error([1.0], [1.0], -1),
        lambda: pvt_input_error([1.0], [1.0], -1),
    ):
        with pytest.raises(ValueError, match="delta"):
            f()
    assert np.isinf(pvt_input_error([0.001, 1.0], [1.0, 1.0], 0.01)[1])
