import math

import numpy as np
import pytest

from pine import ta
from pine.io import load_tradingview_csv
from pine.parity import (
    PortSpec,
    Tolerance,
    check_export,
    compare_discrete,
    compare_series,
    convergence_bars,
    near_tie,
    tolerance_for,
)

REF = np.array([np.nan, np.nan, 10.0, 20.0, 30.0, 40.0])


def test_identical_passes_with_zero_error():
    r = compare_series(REF, REF)
    assert r.passed and r.first_divergence is None
    assert r.n_compared == 4 and r.max_abs_err == 0 and r.within_tol_share == 1


def test_within_tolerance_passes():
    py = REF + np.array([0, 0, 1e-9, -1e-9, 0, 0])
    assert compare_series(py, REF, tolerance=Tolerance(atol=2e-9, rtol=0)).passed


def test_value_failure_located_with_metrics_and_ts():
    py = REF.copy()
    py[4] += 0.3
    py[5] += 0.1
    r = compare_series(py, REF, tolerance=Tolerance(atol=0.2, rtol=0), ts=np.arange(6) * 60)
    assert not r.passed
    assert (r.first_divergence, r.divergence_type, r.first_divergence_ts) == (4, "value", 240)
    assert r.value_failures == 1 and r.max_abs_err_bar == 4
    assert r.max_abs_err == pytest.approx(0.3)
    assert r.max_rel_err == pytest.approx(0.3 / 30)
    assert r.rmse == pytest.approx(math.sqrt((0.3**2 + 0.1**2) / 4))
    assert r.within_tol_share == pytest.approx(0.75)


def test_na_mismatch_is_structural_and_ignores_tolerance():
    py = REF.copy()
    py[1] = 5.0
    r = compare_series(py, REF, tolerance=Tolerance(atol=1e9, rtol=1))
    assert not r.passed and r.na_mismatches == 1
    assert (r.first_divergence, r.divergence_type) == (1, "na")


def test_na_mismatch_after_value_failure_reports_earliest():
    py = REF.copy()
    py[2] = 99.0
    py[5] = np.nan
    r = compare_series(py, REF)
    assert (r.first_divergence, r.divergence_type) == (2, "value")
    assert r.first_na_mismatch == 5


def test_relative_tolerance_scales_with_reference():
    pine = np.array([1.0, 1000.0])
    py = pine * (1 + 5e-10)
    assert compare_series(py, pine, tolerance=Tolerance(atol=0, rtol=1e-9)).passed
    assert not compare_series(py, pine, tolerance=Tolerance(atol=0, rtol=1e-10)).passed


def test_convergence_mode_excludes_and_annotates():
    py = REF.copy()
    py[2] = 0.0
    py[1] = 1.0
    r = compare_series(py, REF, mode="convergence", exclude=3)
    assert r.passed and r.n_excluded == 3 and r.n_compared == 3
    assert any("does not certify seeding" in n for n in r.notes)


def test_aligned_mode_refuses_exclusion():
    with pytest.raises(ValueError, match="aligned"):
        compare_series(REF, REF, exclude=2)


@pytest.mark.parametrize(
    "kw", [{"py": REF[:3]}, {"mode": "loose"}, {"mode": "convergence", "exclude": 99}]
)
def test_bad_arguments(kw):
    args = {"py": REF, "pine": REF, **kw}
    with pytest.raises(ValueError):
        compare_series(args.pop("py"), args.pop("pine"), **args)


def test_tolerance_floor_from_printed_precision():
    t = tolerance_for("bounded", 2)
    assert t.atol == pytest.approx(0.005) and t.rtol == 0
    assert tolerance_for("price", 15).atol == tolerance_for("price").atol
    with pytest.raises(ValueError):
        tolerance_for("weird")


def test_convergence_bars_matches_plan_example():
    # PINE_PLAN.md §5.3: RMA(14), e0 = 1, tol = 1e-8 -> 249 bars.
    assert convergence_bars(1 / 14, 1e-8, 1.0) == 249
    assert convergence_bars(0.5, 1.0, 0.5) == 0
    assert convergence_bars(1.0, 1e-8, 5.0) == 0
    n = convergence_bars(2 / 15, 1e-7, 30.0)
    assert 30 * (13 / 15) ** n <= 1e-7 < 30 * (13 / 15) ** (n - 1)


def test_discrete_exact_and_excused_flips():
    pine = np.array([False, True, False, True])
    py = np.array([False, False, False, True])
    assert not compare_discrete(py, pine).passed
    a, b = np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 2.0 + 1e-12, 0.0, 0.0])
    r = compare_discrete(py, pine, excuse=near_tie(a, b, Tolerance(1e-9, 0)))
    assert r.passed and r.excused == 1 and r.excused_bars == (1,)
    py2 = np.array([True, False, False, True])
    r2 = compare_discrete(py2, pine, excuse=near_tie(a, b, Tolerance(1e-9, 0)), ts=[0, 1, 2, 3])
    assert (r2.first_unexcused, r2.first_unexcused_ts) == (0, 0)


def _placeholder_export(fixtures_dir):
    return load_tradingview_csv(fixtures_dir / "synthetic" / "random_walk_60.csv", timeframe="60")


def test_check_export_end_to_end(fixtures_dir):
    # The synthetic fixture's "Plot" column is close with na for the first 13 bars.
    exp = _placeholder_export(fixtures_dir)

    def port(b):
        return np.where(np.arange(len(b)) >= 13, b.close, np.nan)

    rep = check_export(exp, {"Plot": PortSpec(fn=port)})
    assert rep.passed
    col = rep.columns[0]
    assert col.tolerance.atol == pytest.approx(0.005, rel=1e-6)  # 2 printed decimals
    md = rep.to_markdown()
    assert "PASS" in md and "| Plot |" in md


def test_check_export_failure_report(fixtures_dir):
    exp = _placeholder_export(fixtures_dir)
    rep = check_export(exp, {"Plot": PortSpec(fn=lambda b: ta.sma(b.close, 14))})
    assert not rep.passed
    c = rep.columns[0]
    assert c.first_divergence == 13 and c.first_divergence_ts == int(exp.bars.ts[13])
    assert "FAIL" in rep.to_markdown()


def test_check_export_unknown_column(fixtures_dir):
    with pytest.raises(KeyError, match="RSI"):
        check_export(_placeholder_export(fixtures_dir), {"RSI": PortSpec(fn=lambda b: b.close)})


def test_crossover_excuse_covers_a_tie_on_the_previous_bar():
    from pine import ta
    from pine.parity import crossover_excuse

    tol = Tolerance(1e-9, 0)
    b = np.zeros(4)
    a = np.array([-1.0, -1e-12, 1.0, 1.0])  # tie at bar 1, crossover at bar 2
    a_alt = a.copy()
    a_alt[1] = 1e-12  # other side of the tie: a_alt is already above at bar 1
    ref, alt = ta.crossover(a, b), ta.crossover(a_alt, b)
    assert ref[2] and not alt[2]  # the flip is at bar 2, caused by the tie at bar 1
    assert not compare_discrete(alt, ref, excuse=near_tie(a, b, tol)).passed
    r = compare_discrete(alt, ref, excuse=crossover_excuse(a, b, tol))
    assert r.passed and 2 in r.excused_bars


def test_crossover_excuse_does_not_excuse_real_flips():
    from pine.parity import crossover_excuse

    a = np.array([-1.0, -1.0, 1.0, 1.0, np.nan])
    b = np.zeros(5)
    ex = crossover_excuse(a, b, Tolerance(1e-9, 0))
    assert not ex.any()
