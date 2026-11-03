"""The reference-capture pipeline (PINE_PLAN.md §10 A1, A2).

Real TradingView exports are not available here, so these tests build *synthetic* exports
with pine.ta itself. They prove the plumbing and that the na-policy decision discriminates
between hypotheses; they are not evidence that pine.ta matches TradingView.
"""

from pathlib import Path

import numpy as np
import pytest

from pine import ta
from pine.io import TradingViewExport, load_tradingview_csv, write_tradingview_csv
from pine.lint import lint_file
from pine.lint._calls import find_calls
from pine.lint._lexer import tokenize
from pine.reference import (
    CONSTANTS,
    GAPFREE_COLUMNS,
    NA_COLUMNS,
    NA_PORTS,
    PLOT_TITLES,
    _gapfree_specs,
    _src_na_expected,
    check_capture,
    decide_na_policy,
    main,
)

ROOT = Path(__file__).resolve().parent.parent
CAPTURE = ROOT / "tools" / "reference_capture.pine"
GAPPED = ROOT / "tests" / "fixtures" / "synthetic" / "random_walk_gaps_60.csv"
MANIFEST = GAPPED.with_suffix(".yaml")


# ---------- the Pine script and the Python side agree ----------


def test_pine_constants_match_python():
    toks, _ = tokenize(CAPTURE.read_text(encoding="utf-8"))
    found = {}
    for i, t in enumerate(toks[:-4]):
        if t.text == "const" and toks[i + 1].text == "int" and toks[i + 3].is_op("="):
            found[toks[i + 2].text] = int(toks[i + 4].text)
    assert found == CONSTANTS


def test_pine_plot_titles_match_python_in_order():
    toks, _ = tokenize(CAPTURE.read_text(encoding="utf-8"))
    titles = [
        c.args[1].text.strip("\"'")
        for c in find_calls(toks)
        if c.name == "plot" and len(c.args) > 1
    ]
    assert tuple(titles) == PLOT_TITLES
    assert len(set(titles)) == len(titles)  # export columns must be unambiguous


def test_capture_script_is_lint_clean():
    assert lint_file(CAPTURE).findings == ()


# ---------- synthetic export of the capture script ----------


def synthetic_export(policy: str, tmp_path: Path, drop_first: int = 0) -> Path:
    """What TradingView *would* export if it behaved exactly like pine.ta under ``policy``."""
    b = load_tradingview_csv(GAPPED, timeframe="60").bars
    src_na = _src_na_expected(b)
    dummy = TradingViewExport(b, {}, ("time", "open", "high", "low", "close"), "unix")
    cols = {}
    ema_f, ema_s = ta.ema(b.close, CONSTANTS["FAST"]), ta.ema(b.close, CONSTANTS["SLOW"])
    pre = {"ema_fast": ema_f, "ema_slow": ema_s}
    tmp = TradingViewExport(b, pre, dummy.header, "unix")
    for name, spec in _gapfree_specs(tmp).items():
        cols[name] = np.asarray(spec.fn(b), dtype=float)
    for name, (fn, n, _) in NA_PORTS.items():
        cols[name] = fn(src_na, n, na_policy=policy)
    assert tuple(cols) == PLOT_TITLES
    exp = TradingViewExport(
        b, cols, ("time", "open", "high", "low", "close", "Volume", *cols), "unix"
    )
    path = tmp_path / f"capture_{policy}.csv"
    write_tradingview_csv(exp, path)
    if drop_first:
        lines = path.read_text().splitlines(keepends=True)
        path.write_text(lines[0] + "".join(lines[1 + drop_first :]))
    return path


def test_gapfree_columns_listed_consistently(tmp_path):
    exp = load_tradingview_csv(synthetic_export("propagate", tmp_path), timeframe="60")
    assert tuple(_gapfree_specs(exp)) == GAPFREE_COLUMNS


@pytest.mark.parametrize("policy", ta.NA_POLICIES)
def test_pipeline_passes_and_identifies_the_policy(policy, tmp_path):
    exp = load_tradingview_csv(synthetic_export(policy, tmp_path), timeframe="60")
    assert check_capture(exp).passed
    decisions = decide_na_policy(exp)
    assert [d.column for d in decisions] == list(NA_COLUMNS)
    # Discrimination: under each hypothesis, every column matches that hypothesis only.
    assert {d.column: d.matching for d in decisions} == {c: (policy,) for c in NA_COLUMNS}


def test_truncated_export_fails_alignment_check(tmp_path):
    exp = load_tradingview_csv(
        synthetic_export("propagate", tmp_path, drop_first=10), timeframe="60"
    )
    rep = check_capture(exp)
    src = next(c for c in rep.columns if c.name == "src_na")
    assert not src.passed  # gap pattern no longer lines up with bar_index


def test_unknown_behaviour_matches_no_policy(tmp_path):
    exp = load_tradingview_csv(synthetic_export("propagate", tmp_path), timeframe="60")
    cols = dict(exp.columns)
    fake = np.array(cols["sma_na"])
    fake[~np.isnan(fake)] += 1.0  # a fourth behaviour none of the hypotheses produce
    cols["sma_na"] = fake
    altered = TradingViewExport(exp.bars, cols, exp.header, exp.time_format)
    d = next(d for d in decide_na_policy(altered) if d.column == "sma_na")
    assert d.matching == () and d.decided is None


def test_cli_exit_codes(tmp_path, capsys):
    path = synthetic_export("skip", tmp_path)
    assert main([str(path), "--manifest", str(MANIFEST)]) == 0
    out = capsys.readouterr().out
    assert "all built-ins follow `skip`" in out
    assert main([str(GAPPED), "--timeframe", "60"]) == 2  # not a capture export


def test_cli_requires_timeframe_without_manifest(tmp_path, capsys):
    assert main([str(synthetic_export("skip", tmp_path))]) == 2
    assert "python -m pine.capture manifest" in capsys.readouterr().err


def test_manifest_template_is_valid_and_matches_constants():
    from pine.manifest import load_manifest

    m = load_manifest(ROOT / "tools" / "reference_capture.example.yaml")
    assert m.indicator["inputs"] == CONSTANTS


def test_cli_inputs_rounded_adds_allowance(tmp_path, capsys):
    path = synthetic_export("propagate", tmp_path)
    assert main([str(path), "--manifest", str(MANIFEST), "--inputs-rounded"]) == 0
    out = capsys.readouterr().out
    assert "inputs assumed rounded to ±0.005" in out


def test_capture_crossovers_are_discrete_and_json_cli(tmp_path, capsys):
    import json

    path = synthetic_export("propagate", tmp_path)
    exp = load_tradingview_csv(path, timeframe="60")
    rep = check_capture(exp)
    assert [d.name for d in rep.discrete] == ["crossover", "crossunder"]
    assert all(d.passed for d in rep.discrete)
    ema = next(c for c in rep.columns if c.name == "ema")
    assert ema.mde_seed is not None and ema.mde_seed < 1e-6
    assert main([str(path), "--timeframe", "60", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["parity"]["passed"] is True
    assert {d["decided"] for d in payload["na_policy"]} == {"propagate"}
