"""Composite convergence bounds (PINE_PLAN.md §5.3, §10 B3).

Setup throughout: "Pine" computes on the full history; "Python" sees only a suffix (the first
``s`` bars dropped), as with a truncated export. The bounds must cover the realised error
(soundness), and excluding the computed number of bars must make convergence mode pass.
"""

import numpy as np
import pytest
from hyp import examples
from hypothesis import given, settings
from hypothesis import strategies as st

from pine import ta
from pine.convergence import (
    atr_convergence_bars,
    atr_truncation_error,
    rsi_convergence_bars,
    rsi_truncation_error,
)
from pine.parity import compare_series

L = 14


def _ohlc(seed, n=700, vol=1.0):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, vol, n))
    o = np.concatenate([[100.0], c[:-1]]) + rng.normal(0, vol, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.3 * vol, n))
    lo = np.minimum(o, c) - np.abs(rng.normal(0, 0.3 * vol, n))
    return h, lo, c


@settings(max_examples=examples(40), deadline=None)
@given(st.integers(0, 10_000), st.integers(1, 120), st.integers(2, 30))
def test_atr_bound_covers_realised_truncation_error(seed, s, n):
    h, lo, c = _ohlc(seed)
    pine = ta.atr(h, lo, c, n)[s:]
    py = ta.atr(h[s:], lo[s:], c[s:], n)
    e0_true = float(np.nanmax(ta.tr(h, lo, c, handle_na=True)))  # over the full history
    bound = atr_truncation_error(h[s:], lo[s:], c[s:], n, e0=e0_true)
    ok = ~np.isnan(py)
    assert np.all(np.abs(py - pine)[ok] <= bound[ok] + 1e-9)


def test_atr_residual_decays_exactly_geometrically():
    # After Python's seed bar the only difference is RMA state: residual_t = r_w (1-α)^(t-w).
    h, lo, c = _ohlc(1)
    s = 37
    r = ta.atr(h[s:], lo[s:], c[s:], L) - ta.atr(h, lo, c, L)[s:]
    w = L - 1
    k = np.arange(r.size - w)
    np.testing.assert_allclose(r[w:], r[w] * (1 - 1 / L) ** k, rtol=1e-7, atol=1e-12)


@settings(max_examples=examples(40), deadline=None)
@given(st.integers(0, 10_000), st.integers(1, 120), st.integers(2, 30), st.floats(0.01, 3.0))
def test_rsi_bound_covers_realised_truncation_error(seed, s, n, vol):
    _, _, x = _ohlc(seed, vol=vol)
    pine = ta.rsi(x, n)[s:]
    py = ta.rsi(x[s:], n)
    m_true = float(np.max(np.abs(np.diff(x))))  # over the full history
    bound = rsi_truncation_error(x[s:], n, max_change=m_true)
    ok = ~np.isnan(py)
    assert np.all(np.abs(py - pine)[ok] <= bound[ok] + 1e-7)


@pytest.mark.parametrize("which", ["atr", "rsi"])
def test_convergence_mode_passes_with_computed_exclusion_and_not_before(which):
    h, lo, c = _ohlc(7)
    s, tol = 50, 1e-8
    if which == "atr":
        pine = ta.atr(h, lo, c, L)[s:]
        py = ta.atr(h[s:], lo[s:], c[s:], L)
        e0 = float(np.nanmax(ta.tr(h, lo, c, handle_na=True)))
        n = atr_convergence_bars(h[s:], lo[s:], c[s:], L, tol, e0=e0)
        kind = "price"
    else:
        pine = ta.rsi(c, L)[s:]
        py = ta.rsi(c[s:], L)
        n = rsi_convergence_bars(c[s:], L, tol, max_change=float(np.max(np.abs(np.diff(c)))))
        kind = "bounded"
    from pine.parity import Tolerance

    t = Tolerance(atol=tol, rtol=0.0)
    assert L <= n < py.size
    assert not compare_series(py, pine, kind=kind, tolerance=t).passed  # aligned: na mask
    assert compare_series(py, pine, kind=kind, tolerance=t, mode="convergence", exclude=n).passed
    early = compare_series(py, pine, kind=kind, tolerance=t, mode="convergence", exclude=L + 1)
    assert not early.passed  # the bound is needed: too short an exclusion fails


def test_rsi_needs_longer_exclusion_in_quiet_markets():
    # Same moves scaled down by 100: U'+D' shrinks with them, but so does M — the ratio, and so
    # the bars needed, is unchanged (RSI is scale invariant). A *quiet stretch* inside a
    # volatile history is what lengthens it: M stays large while U'+D' falls.
    rng = np.random.default_rng(3)
    calm = 100 + np.cumsum(rng.normal(0, 0.01, 400))
    spike = np.concatenate([[100.0, 110.0, 100.0], calm])
    n_quiet = rsi_convergence_bars(spike, L, 1e-8)
    n_plain = rsi_convergence_bars(calm, L, 1e-8)
    assert n_quiet > n_plain


def test_arguments_validated():
    with pytest.raises(ValueError):
        rsi_convergence_bars(np.arange(50.0), L, 0.0)
    with pytest.raises(ValueError):
        atr_convergence_bars(np.ones(5), np.ones(5), np.ones(5), L, 1e-8)
    with pytest.raises(ValueError):
        rsi_truncation_error(np.arange(5.0), L)
