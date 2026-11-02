"""Harness sensitivity: deliberate defects must be caught at the expected bar (PINE_PLAN §5.9).

The reference is the correct port on the synthetic random walk; each mutant is a plausible
porting bug. Negative controls check that the harness does *not* flag implementation
differences that are only floating-point summation order.

ATR runs on a fixture with opening gaps; crossover is checked with `compare_discrete`.
"""

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from pine import ta
from pine.io import load_tradingview_csv
from pine.parity import (
    compare_discrete,
    compare_series,
    convergence_bars,
    near_tie,
    tolerance_for,
)

L = 14
NA_AT = 50  # where the na-substitution case injects a gap


@pytest.fixture(scope="module")
def close():
    p = Path(__file__).parent / "fixtures" / "synthetic" / "random_walk_60.csv"
    return np.array(load_tradingview_csv(p, timeframe="60").bars.close)


# ---------- mutant ports ----------


def _seed_first_value(x, n, alpha):
    """Seeded with the source value at the first full bar instead of the SMA."""
    out = np.full(x.size, np.nan)
    prev = math.nan
    for t in range(x.size):
        if math.isnan(prev):
            cur = x[t] if t >= n - 1 and not np.isnan(x[t - n + 1 : t + 1]).any() else math.nan
        else:
            cur = alpha * x[t] + (1 - alpha) * prev
        out[t] = cur
        prev = cur
    return out


def ema_bad_seed(x, n):
    return _seed_first_value(np.asarray(x, float), n, 2 / (n + 1))


def rma_bad_seed(x, n):
    return _seed_first_value(np.asarray(x, float), n, 1 / n)


def rsi_with(rma_fn):
    def f(x, n):
        x = np.asarray(x, float)
        ch = np.concatenate([[np.nan], np.diff(x)])
        u, d = np.maximum(ch, 0), np.maximum(-ch, 0)  # NaN propagates
        up, down = rma_fn(u, n), rma_fn(d, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = 100 - 100 / (1 + up / down)
        r = np.where(down == 0, 100.0, np.where(up == 0, 0.0, r))
        return np.where(np.isnan(up) | np.isnan(down), np.nan, r)

    return f


def length_shift(fn, k):
    return lambda x, n: fn(x, n + k)


def lookahead(fn):
    return lambda x, n: np.concatenate([fn(x, n)[1:], [np.nan]])


def nz_source(fn):
    return lambda x, n: fn(np.nan_to_num(np.asarray(x, float), nan=0.0), n)


def drop_first_bar(fn):
    return lambda x, n: np.concatenate([[np.nan], fn(np.asarray(x, float)[1:], n)])


REF = {
    "sma": ta.sma,
    "ema": ta.ema,
    "rma": ta.rma,
    "rsi": ta.rsi,
    "stdev": ta.stdev,
    "highest": ta.highest,
    "lowest": ta.lowest,
}
KIND = {k: "price" for k in REF} | {"rsi": "bounded"}
W = {k: L - 1 for k in REF} | {"rsi": L}  # first non-na bar

# (indicator, mutation, mutant(x, n), expected first divergence bar, divergence type, needs gap)
CASES: list[tuple[str, str, Callable, int, str, bool]] = [
    ("ema", "seed: first value", ema_bad_seed, W["ema"], "value", False),
    ("rma", "seed: first value", rma_bad_seed, W["rma"], "value", False),
    ("rsi", "seed: first value", rsi_with(rma_bad_seed), W["rsi"], "value", False),
    (
        "stdev",
        "sample instead of population",
        lambda x, n: ta.stdev(x, n, biased=False),
        W["stdev"],
        "value",
        False,
    ),
]
for _name, _f in REF.items():
    CASES += [
        (_name, "length + 1", length_shift(_f, +1), W[_name], "na", False),
        (_name, "length - 1", length_shift(_f, -1), W[_name] - 1, "na", False),
        (_name, "one-bar look-ahead", lookahead(_f), W[_name] - 1, "na", False),
        (_name, "na -> 0", nz_source(_f), NA_AT, "na", True),
        (_name, "first bar dropped", drop_first_bar(_f), W[_name], "na", False),
    ]


@pytest.mark.parametrize(
    "name, mutation, mutant, bar, dtype, gap", CASES, ids=[f"{c[0]}-{c[1]}" for c in CASES]
)
def test_mutation_detected_at_expected_bar(close, name, mutation, mutant, bar, dtype, gap):
    x = close.copy()
    if gap:
        x[NA_AT] = np.nan
    ref = REF[name](x, L)
    r = compare_series(mutant(x, L), ref, kind=KIND[name])
    assert not r.passed, f"{name}/{mutation} not detected"
    assert (r.first_divergence, r.divergence_type) == (bar, dtype)


def test_references_pass_against_themselves(close):
    # Sanity: each reference passes against itself, so failures above are due to the mutant.
    for name, f in REF.items():
        assert compare_series(f(close, L), f(close, L), kind=KIND[name]).passed


# ---------- negative controls ----------


def sma_cumsum(x, n):
    c = np.cumsum(np.concatenate([[0.0], x]))
    out = np.full(x.size, np.nan)
    out[n - 1 :] = (c[n:] - c[:-n]) / n
    return out


def ema_vectorised_order(x, n):
    # Same recursion, algebraically rearranged: prev + alpha * (x - prev).
    a = 2 / (n + 1)
    seed = sma_cumsum(x, n)
    out = np.full(x.size, np.nan)
    out[n - 1] = seed[n - 1]
    for t in range(n, x.size):
        out[t] = out[t - 1] + a * (x[t] - out[t - 1])
    return out


@pytest.mark.parametrize(
    "name, alt", [("sma", sma_cumsum), ("ema", ema_vectorised_order)], ids=["sma", "ema"]
)
def test_summation_order_differences_are_not_flagged(close, name, alt):
    r = compare_series(alt(close, L), REF[name](close, L), kind="price")
    assert r.passed
    assert 0 <= r.max_abs_err < 1e-10


# ---------- the masking hazard, §5.3 ----------


def test_seed_bug_hidden_by_convergence_exclusion(close):
    """Documents why aligned mode is the default: a seed bug passes once enough bars are
    excluded, exactly as the decay bound predicts."""
    ref, bad = ta.ema(close, L), ema_bad_seed(close, L)
    tol = tolerance_for("price")
    assert not compare_series(bad, ref, kind="price").passed

    e0 = float(close.max() - close.min())
    bound = tol.atol + tol.rtol * float(np.nanmin(np.abs(ref)))
    n = (L - 1) + convergence_bars(2 / (L + 1), bound, e0)
    assert n < close.size
    hidden = compare_series(bad, ref, kind="price", mode="convergence", exclude=n)
    assert hidden.passed
    assert any("does not certify seeding" in note for note in hidden.notes)


# ---------- ATR (needs high/low/close; gapped fixture) ----------


@pytest.fixture(scope="module")
def gapped():
    p = Path(__file__).parent / "fixtures" / "synthetic" / "random_walk_gaps_60.csv"
    b = load_tradingview_csv(p, timeframe="60").bars
    return np.array(b.high), np.array(b.low), np.array(b.close)


def lookahead_any(fn):
    return lambda *a: np.concatenate([fn(*a)[1:], [np.nan]])


def _atr_range_only(h, lo, c, n):
    return ta.rma(h - lo, n)


def _atr_tr_na_first(h, lo, c, n):
    return ta.rma(ta.tr(h, lo, c, handle_na=False), n)


def _atr_nz_close(h, lo, c, n):
    return ta.atr(h, lo, np.nan_to_num(c, nan=0.0), n)


def _atr_drop_first(h, lo, c, n):
    return np.concatenate([[np.nan], ta.atr(h[1:], lo[1:], c[1:], n)])


ATR_CASES = [
    ("true range = high - low (gaps ignored)", _atr_range_only, L - 1, "value", False),
    ("first-bar true range na", _atr_tr_na_first, L - 1, "na", False),
    ("length + 1", lambda h, lo, c, n: ta.atr(h, lo, c, n + 1), L - 1, "na", False),
    ("length - 1", lambda h, lo, c, n: ta.atr(h, lo, c, n - 1), L - 2, "na", False),
    ("one-bar look-ahead", lookahead_any(ta.atr), L - 2, "na", False),
    # na close at NA_AT: the *next* bar's true range is na, so divergence is at NA_AT + 1.
    ("na close -> 0", _atr_nz_close, NA_AT + 1, "na", True),
    ("first bar dropped", _atr_drop_first, L - 1, "na", False),
]


@pytest.mark.parametrize(
    "mutation, mutant, bar, dtype, gap", ATR_CASES, ids=[c[0] for c in ATR_CASES]
)
def test_atr_mutation_detected(gapped, mutation, mutant, bar, dtype, gap):
    h, lo, c = (a.copy() for a in gapped)
    if gap:
        c[NA_AT] = np.nan
    r = compare_series(mutant(h, lo, c, L), ta.atr(h, lo, c, L), kind="price")
    assert not r.passed, f"atr/{mutation} not detected"
    assert (r.first_divergence, r.divergence_type) == (bar, dtype)


def test_gapless_data_cannot_detect_range_only_atr(close):
    """Coverage note: on the gapless fixture every open equals the previous close, so the
    gap terms of true range never dominate and the 'high - low only' bug passes. Fixtures
    must be chosen to exercise each term of a formula."""
    p = Path(__file__).parent / "fixtures" / "synthetic" / "random_walk_60.csv"
    b = load_tradingview_csv(p, timeframe="60").bars
    h, lo, c = np.array(b.high), np.array(b.low), np.array(b.close)
    assert compare_series(_atr_range_only(h, lo, c, L), ta.atr(h, lo, c, L)).passed


# ---------- crossover (discrete) ----------


def _first_true(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0])


def test_crossover_mutations(close):
    a, b = ta.ema(close, 5), ta.ema(close, 20)
    ref = ta.crossover(a, b)
    assert ref.sum() >= 3, "fixture should contain several crossovers"

    # Missing the previous-bar condition: true on every bar a > b.
    above = np.nan_to_num(a, nan=-np.inf) > np.nan_to_num(b, nan=np.inf)
    r = compare_discrete(above, ref)
    assert not r.passed and r.first_unexcused == _first_true(above & ~ref)

    # One-bar look-ahead: fires the bar before the true crossover.
    early = np.concatenate([ref[1:], [False]])
    r = compare_discrete(early, ref)
    assert not r.passed and r.first_unexcused == _first_true(ref) - 1

    # Crossunder swapped in for crossover.
    r = compare_discrete(ta.crossunder(a, b), ref)
    assert not r.passed


def test_crossover_near_ties_are_excused_not_hidden():
    # Inputs that differ by 1e-12 can legitimately flip a crossover; such flips are excused
    # by near_tie on the continuous inputs, and still counted.
    b = np.zeros(6)
    a = np.array([-1.0, 1.0, -1e-13, 1e-13, -1.0, 1.0])
    a_alt = a.copy()
    a_alt[3] = -1e-13  # a different implementation lands on the other side of the tie
    ref, alt = ta.crossover(a, b), ta.crossover(a_alt, b)
    assert (ref != alt).any()
    tol = tolerance_for("unbounded")
    r = compare_discrete(alt, ref, excuse=near_tie(a, b, tol))
    assert r.passed and r.excused == 1 and r.excused_bars == (3,)
    assert not compare_discrete(alt, ref).passed
