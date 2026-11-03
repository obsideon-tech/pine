"""Semantics capture (PINE_PLAN.md §11 E1): Pine script ↔ Python consistency, runtime ↔
expected hypotheses, discrimination, and the decision pipeline on synthetic exports."""

from pathlib import Path

import numpy as np
import pytest

from pine import ta
from pine.io import TradingViewExport, load_tradingview_csv, write_tradingview_csv
from pine.lint import lint_file
from pine.lint._calls import find_calls
from pine.lint._lexer import tokenize
from pine.runtime import run
from pine.semantics import (
    COLUMNS,
    CONSTANTS,
    EXPECTED,
    GROUPS,
    decide,
    main,
    runtime_capture,
)

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "capture_semantics.pine"
GAPPED = ROOT / "tests" / "fixtures" / "synthetic" / "random_walk_gaps_60.csv"


@pytest.fixture(scope="module")
def bars():
    return load_tradingview_csv(GAPPED, timeframe="60").bars


@pytest.fixture(scope="module")
def runtime_result(bars):
    return run(runtime_capture, bars)


# ---------- Pine script ↔ Python ----------


def test_constants_match():
    toks, _ = tokenize(SCRIPT.read_text(encoding="utf-8"))
    found = {}
    for i, t in enumerate(toks[:-4]):
        if t.text == "const" and toks[i + 1].text in ("int", "float") and toks[i + 3].is_op("="):
            found[toks[i + 2].text] = float(toks[i + 4].text)
    assert found == {k: float(v) for k, v in CONSTANTS.items()}


def test_plot_titles_match_in_order():
    toks, _ = tokenize(SCRIPT.read_text(encoding="utf-8"))
    titles = [c.args[1].text.strip("\"'") for c in find_calls(toks) if c.name == "plot"]
    assert tuple(titles) == COLUMNS
    assert sum(len(GROUPS[g][EXPECTED[g]](_dummy_bars())) for g in GROUPS) == len(COLUMNS)


def _dummy_bars():
    return load_tradingview_csv(GAPPED, timeframe="60").bars


def test_capture_script_lint():
    r = lint_file(SCRIPT)
    assert r.findings == ()
    assert [(f.rule, f.line) for f in r.suppressed] == [
        (
            "varip",
            next(
                i
                for i, ln in enumerate(SCRIPT.read_text().splitlines(), 1)
                if ln.startswith("varip")
            ),
        )
    ]


def test_expected_covers_every_group():
    assert set(EXPECTED) == set(GROUPS)
    assert all(EXPECTED[g] in GROUPS[g] for g in GROUPS)


# ---------- runtime ↔ expected hypotheses ----------


@pytest.mark.parametrize("group", sorted(GROUPS))
def test_runtime_reproduces_expected_hypothesis(group, bars, runtime_result):
    for col, v in GROUPS[group][EXPECTED[group]](bars).items():
        np.testing.assert_array_equal(runtime_result[col], v, err_msg=f"{group}/{col}")


@pytest.mark.parametrize("group", sorted(g for g in GROUPS if len(GROUPS[g]) > 1))
def test_hypotheses_are_distinguishable(group, bars):
    hyps = {h: f(bars) for h, f in GROUPS[group].items()}
    names = sorted(hyps)
    for i, a in enumerate(names):
        for bname in names[i + 1 :]:
            assert any(
                not np.array_equal(hyps[a][c], hyps[bname][c], equal_nan=True) for c in hyps[a]
            ), f"{group}: {a} and {bname} produce identical columns"


def test_ports_are_non_trivial(runtime_result):
    d = runtime_result["supertrend_dir"]
    assert set(np.unique(d)) == {-1.0, 1.0} and np.count_nonzero(np.diff(d)) >= 3
    assert set(np.unique(runtime_result["regime"])) >= {-1.0, 1.0}
    assert np.isfinite(runtime_result["trail_stop"]).sum() > 200


def test_supertrend_runtime_port_equals_loop_implementation(bars, runtime_result):
    line, d = ta.supertrend(bars.high, bars.low, bars.close, CONSTANTS["ST_FACTOR"], 10)
    np.testing.assert_array_equal(runtime_result["supertrend"], line)
    np.testing.assert_array_equal(runtime_result["supertrend_dir"], d)


# ---------- decision pipeline on synthetic exports ----------


def _export_from(bars, cols, path):
    exp = TradingViewExport(
        bars, cols, ("time", "open", "high", "low", "close", "Volume", *cols), "unix"
    )
    write_tradingview_csv(exp, path)
    return path


def test_runtime_export_decides_every_group_as_expected(bars, runtime_result, tmp_path):
    path = _export_from(bars, dict(runtime_result.columns), tmp_path / "sem.csv")
    exp = load_tradingview_csv(path, timeframe="60")
    decisions = decide(exp)
    assert {d.group: d.matching for d in decisions} == {g: (EXPECTED[g],) for g in GROUPS}
    assert main([str(path), "--timeframe", "60"]) == 0


@pytest.mark.parametrize(
    "group, alt",
    [("cond_ema", "every_bar"), ("rounding", "half_away"), ("branch_hist", "last_assigned")],
)
def test_alternative_behaviour_is_detected_and_named(
    bars, runtime_result, tmp_path, capsys, group, alt
):
    cols = dict(runtime_result.columns)
    cols.update(GROUPS[group][alt](bars))
    path = _export_from(bars, cols, tmp_path / "alt.csv")
    exp = load_tradingview_csv(path, timeframe="60")
    d = next(d for d in decide(exp) if d.group == group)
    assert d.matching == (alt,) and not d.ok
    assert main([str(path), "--timeframe", "60"]) == 1
    assert (
        f"| {group} | {EXPECTED[group]} | {alt} | **runtime must change**"
        in capsys.readouterr().out
    )


def test_truncated_export_fails_alignment(bars, runtime_result, tmp_path):
    path = _export_from(bars, dict(runtime_result.columns), tmp_path / "t.csv")
    lines = path.read_text().splitlines(keepends=True)
    path.write_text(lines[0] + "".join(lines[11:]))
    d = next(
        d for d in decide(load_tradingview_csv(path, timeframe="60")) if d.group == "alignment"
    )
    assert d.matching == ()


def test_cli_json_and_errors(bars, runtime_result, tmp_path, capsys):
    import json

    path = _export_from(bars, dict(runtime_result.columns), tmp_path / "j.csv")
    assert main([str(path), "--timeframe", "60", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {p["group"] for p in payload} == set(GROUPS) and all(p["ok"] for p in payload)
    assert main([str(GAPPED), "--timeframe", "60"]) == 2
    assert main([str(path)]) == 2
    assert main([str(path), "--manifest", str(tmp_path / "EXPORT.yaml")]) == 2
    err = capsys.readouterr().err
    assert "manifest not found" in err and "python -m pine.capture manifest" in err
