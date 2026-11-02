"""Harness sensitivity: deliberate defects must be caught at the expected bar (PINE_PLAN §5.9).

The reference is the correct port on the synthetic random walk; each mutant is a plausible
porting bug. Negative controls check that the harness does *not* flag implementation
differences that are only floating-point summation order.

Pending: sample-vs-population `stdev` (needs `ta.stdev`).
"""

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from pine import ta
from pine.io import load_tradingview_csv
from pine.parity import compare_series, convergence_bars, tolerance_for

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


REF = {"sma": ta.sma, "ema": ta.ema, "rma": ta.rma, "rsi": ta.rsi}
KIND = {"sma": "price", "ema": "price", "rma": "price", "rsi": "bounded"}
W = {"sma": L - 1, "ema": L - 1, "rma": L - 1, "rsi": L}  # first non-na bar

# (indicator, mutation, mutant(x, n), expected first divergence bar, divergence type, needs gap)
CASES: list[tuple[str, str, Callable, int, str, bool]] = [
    ("ema", "seed: first value", ema_bad_seed, W["ema"], "value", False),
    ("rma", "seed: first value", rma_bad_seed, W["rma"], "value", False),
    ("rsi", "seed: first value", rsi_with(rma_bad_seed), W["rsi"], "value", False),
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
