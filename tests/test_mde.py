"""Minimum detectable error (PINE_PLAN.md §5.7, §10 A4): stated thresholds are real."""

import math
from pathlib import Path

import numpy as np
import pytest

from pine import ta
from pine.io import load_tradingview_csv
from pine.parity import PortSpec, Tolerance, check_export, compare_series, tolerance_for
from pine.sensitivity import input_rounding

GAPPED = Path(__file__).parent / "fixtures" / "synthetic" / "random_walk_gaps_60.csv"
L = 14


@pytest.fixture(scope="module")
def bars():
    return load_tradingview_csv(GAPPED, timeframe="60").bars


def test_mde_offset_is_min_bound_hand_example():
    ref = np.array([np.nan, 10.0, 20.0, 40.0])
    r = compare_series(ref, ref, tolerance=Tolerance(atol=0.1, rtol=0.01))
    assert r.mde_offset == pytest.approx(0.1 + 0.01 * 10)
    assert r.mde_seed is None


def test_mde_seed_formula_hand_example():
    ref = np.array([np.nan, 10.0, 10.0, 10.0])
    tol = Tolerance(atol=np.array([0.0, 1.0, 0.3, 0.2]), rtol=0.0)  # per-bar atol
    r = compare_series(ref, ref, tolerance=tol, alpha=0.5)
    # thresholds: bar1 1.0/0.5^0 = 1.0, bar2 0.3/0.5 = 0.6, bar3 0.2/0.25 = 0.8
    assert r.mde_seed == pytest.approx(0.6)
    assert r.mde_offset == pytest.approx(0.2)


def test_mde_with_nothing_compared_is_infinite():
    r = compare_series([np.nan, np.nan], [np.nan, np.nan], alpha=0.1)
    assert math.isinf(r.mde_offset) and math.isinf(r.mde_seed)


def test_alpha_validated():
    with pytest.raises(ValueError, match="alpha"):
        compare_series([1.0], [1.0], alpha=1.5)


SEEDED = {
    "ema": (lambda b: ta.ema(b.close, L), 2 / (L + 1)),
    "rma": (lambda b: ta.rma(b.close, L), 1 / L),
    "atr": (lambda b: ta.atr(b.high, b.low, b.close, L), 1 / L),
}


def _with_seed_error(ref, alpha, e0):
    w = int(np.flatnonzero(~np.isnan(ref))[0])
    t = np.arange(ref.size)
    return np.where(t >= w, ref + e0 * (1 - alpha) ** np.maximum(t - w, 0), np.nan)


@pytest.mark.parametrize("name", sorted(SEEDED))
@pytest.mark.parametrize("k, detected", [(0.5, False), (2.0, True)])
def test_seed_error_sweep_matches_stated_threshold(bars, name, k, detected):
    fn, alpha = SEEDED[name]
    ref = fn(bars)
    mde = compare_series(ref, ref, alpha=alpha).mde_seed
    assert 0 < mde < 1e-6  # full-precision defaults: tiny
    r = compare_series(_with_seed_error(ref, alpha, k * mde), ref, alpha=alpha)
    assert r.passed is not detected


@pytest.mark.parametrize("k, detected", [(0.5, False), (2.0, True)])
def test_offset_sweep_matches_stated_threshold(bars, k, detected):
    ref = ta.sma(bars.close, L)
    mde = compare_series(ref, ref).mde_offset
    assert compare_series(ref + k * mde, ref).passed is not detected


def test_convergence_mode_makes_seed_errors_practically_undetectable(bars):
    ref = ta.ema(bars.close, L)
    alpha = 2 / (L + 1)
    aligned = compare_series(ref, ref, alpha=alpha).mde_seed
    conv = compare_series(ref, ref, alpha=alpha, mode="convergence", exclude=150).mde_seed
    assert conv > 1e6 * aligned  # §5.3 hazard as a number


def test_input_allowance_raises_mde(bars):
    # A3's tradeoff, quantified: assuming rounded inputs makes small defects invisible.
    from pine.io import TradingViewExport

    ref = ta.sma(bars.close, L)
    exp = TradingViewExport(bars, {"sma": ref}, ("time", "open", "high", "low", "close", "sma"),
                            "unix", observed_decimals={"sma": 15})  # fmt: skip
    spec = {"sma": PortSpec(lambda b: ta.sma(b.close, L), input_error=1.0)}
    exact = check_export(exp, spec).columns[0].mde_offset
    rounded = check_export(exp, spec, input_rounding=input_rounding(2)).columns[0].mde_offset
    assert exact < 1e-6 and rounded >= 0.005
    assert rounded > 1e4 * exact


def test_markdown_shows_mde(bars):
    from pine.parity import ParityReport

    ref = ta.ema(bars.close, L)
    md = ParityReport(
        (compare_series(ref, ref, alpha=2 / 15, tolerance=tolerance_for("price")),)
    ).to_markdown()
    assert "MDE offset / seed" in md
