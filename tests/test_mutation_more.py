"""Deliberate-bug cases for the built-ins the verification registry showed uncovered:
``tr``, ``crossunder``, ``change``, ``barssince``, ``valuewhen``, ``supertrend``,
``pivotlow`` (PINE_PLAN.md §5.9, §11 H2). Each mutant is a plausible porting bug; the harness
must flag it at the bar, and with the divergence type, the bug implies (reason per case).
"""

from pathlib import Path

import numpy as np
import pytest

from pine import ta
from pine.io import load_tradingview_csv
from pine.parity import compare_discrete, compare_series

FIX = Path(__file__).parent / "fixtures" / "synthetic"
NA_AT = 50


@pytest.fixture(scope="module")
def gapped():
    """Opening gaps, so the previous-close terms of true range matter."""
    b = load_tradingview_csv(FIX / "random_walk_gaps_60.csv", timeframe="60").bars
    return np.array(b.high), np.array(b.low), np.array(b.close)


def _first(mask) -> int:
    return int(np.flatnonzero(mask)[0])


def _lag(x):
    return np.concatenate([[np.nan], x[:-1]])


# ---------- ta.tr ----------


def test_tr_mutations(gapped):
    h, lo, c = gapped
    ref = ta.tr(h, lo, c)
    # Range only: wrong from the first bar where a gap term dominates high - low.
    pc = _lag(c)
    gap_bar = _first(np.maximum(np.abs(h - pc), np.abs(lo - pc)) > h - lo)
    r = compare_series(h - lo, ref, kind="price")
    assert (r.first_divergence, r.divergence_type) == (0, "na")  # also bar 0: h-lo vs na
    r = compare_series(np.where(np.arange(h.size) == 0, np.nan, h - lo), ref, kind="price")
    assert (r.first_divergence, r.divergence_type) == (gap_bar, "value")
    # handle_na ignored: plain ta.tr given a value on the first bar.
    r = compare_series(ta.tr(h, lo, c, handle_na=True), ref, kind="price")
    assert (r.first_divergence, r.divergence_type) == (0, "na")
    # Current close instead of the previous one.
    bad = np.maximum.reduce([h - lo, np.abs(h - c), np.abs(lo - c)])
    bad[0] = np.nan
    r = compare_series(bad, ref, kind="price")
    assert (r.first_divergence, r.divergence_type) == (gap_bar, "value")
    # na close -> 0: the next bar's true range uses a 0 previous close.
    c2 = c.copy()
    c2[NA_AT] = np.nan
    r = compare_series(ta.tr(h, lo, np.nan_to_num(c2)), ta.tr(h, lo, c2), kind="price")
    assert (r.first_divergence, r.divergence_type) == (NA_AT + 1, "na")


# ---------- ta.crossunder (discrete) ----------


def test_crossunder_mutations(gapped):
    _, _, c = gapped
    a, b = ta.ema(c, 5), ta.ema(c, 20)
    ref = ta.crossunder(a, b)
    assert ref.sum() >= 3
    below = np.nan_to_num(a, nan=np.inf) < np.nan_to_num(b, nan=-np.inf)
    r = compare_discrete(below, ref)  # previous-bar condition missing
    assert r.first_unexcused == _first(below & ~ref)
    r = compare_discrete(np.concatenate([ref[1:], [False]]), ref)  # one-bar look-ahead
    assert r.first_unexcused == _first(ref) - 1
    over = ta.crossover(a, b)  # direction swapped
    r = compare_discrete(over, ref)
    assert r.first_unexcused == min(_first(over), _first(ref))
    r = compare_discrete(ta.crossunder(a, b) | (a == b), ref)  # control: no extra bars
    assert r.passed


# ---------- ta.change ----------


def test_change_mutations(gapped):
    _, _, c = gapped
    ref = ta.change(c)
    r = compare_series(ta.change(c, 2), ref, kind="price")  # length + 1
    assert (r.first_divergence, r.divergence_type) == (1, "na")
    r = compare_series(-ref, ref, kind="price")  # x[1] - x: wrong from the first nonzero change
    assert (r.first_divergence, r.divergence_type) == (_first(~np.isnan(ref) & (ref != 0)), "value")
    # na handling: "last valid value" instead of propagating (the semantics alternative).
    x = c.copy()
    x[NA_AT] = np.nan
    last_valid = x - np.concatenate([[np.nan], x[:-1]])
    last_valid[NA_AT + 1] = x[NA_AT + 1] - x[NA_AT - 1]
    r = compare_series(last_valid, ta.change(x), kind="price")
    assert (r.first_divergence, r.divergence_type) == (NA_AT + 1, "na")


# ---------- ta.barssince / ta.valuewhen ----------


EVENTS = (np.arange(300) % 7 == 3) & (np.arange(300) >= 10)  # first event at bar 10


def test_barssince_mutations():
    ref = ta.barssince(EVENTS)
    count_from_start = np.where(np.isnan(ref), np.arange(300, dtype=float), ref)
    r = compare_series(count_from_start, ref, kind="unbounded")
    assert (r.first_divergence, r.divergence_type) == (0, "na")
    r = compare_series(ref + 1, ref, kind="unbounded")  # 1 on the event bar
    assert (r.first_divergence, r.divergence_type) == (10, "value")
    late = ta.barssince(np.concatenate([[False], EVENTS[:-1]]))  # event read a bar late
    r = compare_series(late, ref, kind="unbounded")
    assert (r.first_divergence, r.divergence_type) == (10, "na")


def test_valuewhen_mutations(gapped):
    _, _, c = gapped
    ref = ta.valuewhen(EVENTS, c, 0)
    r = compare_series(ta.valuewhen(EVENTS, c, 1), ref, kind="price")  # occurrence + 1
    assert (r.first_divergence, r.divergence_type) == (10, "na")
    r = compare_series(np.nan_to_num(ref), ref, kind="price")  # 0 before the first event
    assert (r.first_divergence, r.divergence_type) == (0, "na")
    r = compare_series(ta.valuewhen(EVENTS, _lag(c), 0), ref, kind="price")  # source lagged
    assert (r.first_divergence, r.divergence_type) == (10, "value")


# ---------- ta.supertrend ----------


def _supertrend_no_ratchet(h, lo, c, factor, n):
    """Bands not carried forward: the line is just hl2 ± factor·ATR on the trend side."""
    a = ta.atr(h, lo, c, n)
    src = (h + lo) / 2
    up, dn = src + factor * a, src - factor * a
    _, d = ta.supertrend(h, lo, c, factor, n)
    return np.where(d == -1, dn, up)


def test_supertrend_mutations(gapped):
    h, lo, c = gapped
    line, d = ta.supertrend(h, lo, c, 3.0, 10)
    first_atr = 9  # ATR(10)'s first value
    r = compare_series(ta.supertrend(h, lo, c, 2.5, 10)[0], line, kind="price")
    assert (r.first_divergence, r.divergence_type) == (first_atr, "value")
    # ATR length 11: its first value is a bar later, so at bar 9 the mutant has none yet.
    r = compare_series(ta.supertrend(h, lo, c, 3.0, 11)[0], line, kind="price")
    assert (r.first_divergence, r.divergence_type) == (first_atr, "na")
    # Without the ratchet the line differs from the first bar the band would have been held.
    bad = _supertrend_no_ratchet(h, lo, c, 3.0, 10)
    r = compare_series(bad, line, kind="price")
    held = (np.abs(bad - line) > 1e-9) & ~np.isnan(line) & ~np.isnan(bad)
    assert not r.passed and r.first_divergence == _first(held | (np.isnan(bad) != np.isnan(line)))
    # Direction (discrete): sign convention flipped is caught at the first bar.
    r = compare_discrete(d == 1, d == -1)
    assert r.first_unexcused == 0


# ---------- ta.pivotlow ----------


def test_pivotlow_mutations():
    lo = np.array([9, 8, 7, 3, 3, 5, 6, 7, 8, 9.0])  # flat bottom at bars 3-4
    ref = ta.pivotlow(lo, 3, 2)
    assert np.flatnonzero(~np.isnan(ref)).tolist() == [6]  # rightmost of the flat, 2 bars late
    left_strict = np.full(lo.size, np.nan)
    left_strict[5] = 3.0  # marks the left end of the flat instead
    r = compare_series(left_strict, ref, kind="price")
    assert (r.first_divergence, r.divergence_type) == (5, "na")
    r = compare_series(np.concatenate([ref[2:], [np.nan] * 2]), ref, kind="price")  # no delay
    assert (r.first_divergence, r.divergence_type) == (4, "na")
    r = compare_series(ta.pivotlow(lo, 2, 3), ref, kind="price")  # left/right swapped
    assert not r.passed and r.divergence_type == "na"
    r = compare_series(ta.pivothigh(-lo, 3, 2) * -1, ref, kind="price")  # control: mirror
    assert r.passed
