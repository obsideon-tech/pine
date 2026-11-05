"""Harness sensitivity for the batch-F built-ins (PINE_PLAN.md §5.9, §11 F).

Each mutant is a plausible porting bug. The harness must flag it, at the bar and with the
divergence type the bug implies; the reasons are given per case. Same fixture and conventions
as tests/test_mutation_suite.py.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from pine import ta
from pine.io import load_tradingview_csv
from pine.parity import compare_series

L = 14
NA_AT = 50


@pytest.fixture(scope="module")
def bars():
    p = Path(__file__).parent / "fixtures" / "synthetic" / "random_walk_60.csv"
    return load_tradingview_csv(p, timeframe="60").bars


# ---------- generic mutations ----------


def length_shift(fn, k):
    return lambda x, n: fn(x, n + k)


def lookahead(fn):
    return lambda x, n: np.concatenate([fn(x, n)[1:], [np.nan]])


def nz_source(fn):
    return lambda x, n: fn(np.nan_to_num(np.asarray(x, float), nan=0.0), n)


def drop_first_bar(fn):
    return lambda x, n: np.concatenate([[np.nan], fn(np.asarray(x, float)[1:], n)])


REF = {
    "math_sum": ta.math_sum,
    "wma": ta.wma,
    "variance": ta.variance,
    "percentile": lambda x, n: ta.percentile_linear_interpolation(x, n, 30),
}
W = dict.fromkeys(REF, L - 1)  # first non-na bar

CASES = []
for _name, _f in REF.items():
    CASES += [
        # Window one longer: first value one bar later -> na mask at the reference's first.
        (_name, "length + 1", length_shift(_f, +1), W[_name], "na", False),
        # One shorter: a value one bar early.
        (_name, "length - 1", length_shift(_f, -1), W[_name] - 1, "na", False),
        (_name, "one-bar look-ahead", lookahead(_f), W[_name] - 1, "na", False),
        # Under `skip` (expected default) the gap bar still has a value, so a 0 substituted
        # for na is a value error on that bar.
        (_name, "na -> 0", nz_source(_f), NA_AT, "value", True),
        (_name, "first bar dropped", drop_first_bar(_f), W[_name], "na", False),
    ]


@pytest.mark.parametrize(
    "name, mutation, mutant, bar, dtype, gap", CASES, ids=[f"{c[0]}-{c[1]}" for c in CASES]
)
def test_generic_mutation_detected(bars, name, mutation, mutant, bar, dtype, gap):
    x = np.array(bars.close)
    if gap:
        x[NA_AT] = np.nan
    r = compare_series(mutant(x, L), REF[name](x, L), kind="price")
    assert not r.passed, f"{name}/{mutation} not detected"
    assert (r.first_divergence, r.divergence_type) == (bar, dtype)


# ---------- built-in-specific mutations ----------


def wma_reversed(x, n):
    """Weights reversed (oldest bar weighted most): the commonest WMA porting bug."""
    w = np.arange(n, 0, -1, dtype=float)
    out = np.full(len(x), np.nan)
    for t in range(n - 1, len(x)):
        out[t] = np.dot(x[t - n + 1 : t + 1], w) / w.sum()
    return out


def hma_rounded(x, n):
    inner = 2 * ta.wma(x, math.floor(n / 2 + 0.5)) - ta.wma(x, n)
    return ta.wma(inner, math.floor(math.sqrt(n) + 0.5))


def pct_exclusive(x, n):
    out = np.full(len(x), np.nan)
    for t in range(n - 1, len(x)):
        out[t] = np.percentile(x[t - n + 1 : t + 1], 30, method="weibull")
    return out


def obv_zero_change_up(close, vol):
    """sign(0) taken as +1 (e.g. `close >= close[1] ? volume : -volume`)."""
    ch = np.concatenate([[np.nan], np.diff(close)])
    s = np.where(np.isnan(ch), np.nan, np.where(ch >= 0, 1.0, -1.0))
    return ta.cum(s * vol)


def pvt_divides_by_close(close, vol):
    ch = np.concatenate([[np.nan], np.diff(close)])
    return ta.cum(ch / close * vol)


def _first_valid(a):
    return int(np.flatnonzero(~np.isnan(a))[0])


def test_wma_reversed_weights(bars):
    x = np.array(bars.close)
    r = compare_series(wma_reversed(x, L), ta.wma(x, L), kind="price")
    assert (r.first_divergence, r.divergence_type) == (L - 1, "value")


@pytest.mark.parametrize("n", [9, 14])
def test_hma_rounding_mutant(bars, n):
    """Length 9: half rounds to 5 instead of 4; the outer wma (3) is the same, so values
    differ from the first output (bar n - 1 + 3 - 1). Length 14: sqrt rounds to 4 instead of
    3, so the mutant's first value is one bar later: an na-mask difference at the reference's
    first value (bar 14 - 1 + 3 - 1)."""
    x = np.array(bars.close)
    ref = ta.hma(x, n)
    r = compare_series(hma_rounded(x, n), ref, kind="price")
    assert not r.passed
    assert r.first_divergence == _first_valid(ref)
    assert r.divergence_type == ("value" if n == 9 else "na")
    assert _first_valid(ref) == n - 1 + math.floor(math.sqrt(n)) - 1


def test_variance_sample_instead_of_population(bars):
    x = np.array(bars.close)
    r = compare_series(ta.variance(x, L, biased=False), ta.variance(x, L), kind="unbounded")
    assert (r.first_divergence, r.divergence_type) == (L - 1, "value")


def test_percentile_rank_convention(bars):
    x = np.array(bars.close)
    ref = ta.percentile_linear_interpolation(x, L, 30)
    r = compare_series(pct_exclusive(x, L), ref, kind="price")
    assert (r.first_divergence, r.divergence_type) == (L - 1, "value")


def test_pivot_tie_rule_and_delay():
    # A flat top at bars 3-4. TradingView marks the right end (bar 4, observed 2026-11-05),
    # reported 2 bars later at bar 6. A port marking the left end reports at bar 5; a port
    # reporting at the pivot bar itself is 2 bars early (bar 4).
    h = np.array([1, 2, 3, 6, 6, 4, 3, 2, 1, 0.0])
    ref = ta.pivothigh(h, 3, 2)
    assert np.flatnonzero(~np.isnan(ref)).tolist() == [6]
    left_strict = np.full(h.size, np.nan)
    left_strict[5] = 6.0
    r = compare_series(left_strict, ref, kind="price")
    assert (r.first_divergence, r.divergence_type) == (5, "na")
    at_pivot = np.concatenate([ref[2:], [np.nan, np.nan]])
    r = compare_series(at_pivot, ref, kind="price")
    assert (r.first_divergence, r.divergence_type) == (4, "na")


def test_obv_sign_of_zero_change():
    close = np.array([10.0, 11.0, 11.0, 10.0])
    vol = np.array([1.0, 2.0, 3.0, 4.0])
    r = compare_series(obv_zero_change_up(close, vol), ta.obv(close, vol), kind="unbounded")
    assert (r.first_divergence, r.divergence_type) == (2, "value")


def test_pvt_wrong_denominator(bars):
    c, v = np.array(bars.close), np.array(bars.volume)
    r = compare_series(pvt_divides_by_close(c, v), ta.pvt(c, v), kind="unbounded")
    assert (r.first_divergence, r.divergence_type) == (1, "value")


def test_fixnan_and_cum_na_handling():
    x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    # fixnan ported as nz(): 0 instead of the last value.
    r = compare_series(np.nan_to_num(x), ta.fixnan(x), kind="price")
    assert (r.first_divergence, r.divergence_type) == (1, "value")
    # cum carrying its total onto na bars (TradingView gives na there, observed 2026-11-05).
    r = compare_series(ta.cum(x, na_policy="skip"), ta.cum(x), kind="price")
    assert (r.first_divergence, r.divergence_type) == (1, "na")
    # cum ported with propagate: na from the first gap on, so the next valid bar differs.
    r = compare_series(ta.cum(x, na_policy="propagate"), ta.cum(x), kind="price")
    assert (r.first_divergence, r.divergence_type) == (2, "na")


def test_references_pass_against_themselves(bars):
    x = np.array(bars.close)
    for f in REF.values():
        assert compare_series(f(x, L), f(x, L), kind="unbounded").passed
