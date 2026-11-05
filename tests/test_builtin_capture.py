"""Built-in capture (PINE_PLAN.md §11 F): Pine script ↔ Python consistency, runtime ↔
expected hypotheses, discrimination, and the decision pipeline on synthetic exports."""

from pathlib import Path

import numpy as np
import pytest

from pine import Bars
from pine.builtin_capture import (
    COLUMNS,
    CONSTANTS,
    EXPECTED,
    GROUPS,
    decide,
    runtime_capture,
    summary,
)
from pine.io import TradingViewExport, load_tradingview_csv, write_tradingview_csv
from pine.lint import lint_file
from pine.lint._calls import find_calls
from pine.lint._lexer import tokenize
from pine.runtime import run

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "capture_builtins.pine"


def _bars(n=900, seed=11):
    """Two-decimal prices, so equal highs/lows (pivot ties) occur."""
    rng = np.random.default_rng(seed)
    c = np.round(50 + np.cumsum(rng.normal(0, 0.3, n)), 2)
    o = np.round(np.concatenate([[50.0], c[:-1]]) + rng.normal(0, 0.1, n), 2)
    h = np.round(np.maximum(o, c) + np.abs(rng.normal(0, 0.1, n)), 2)
    lo = np.round(np.minimum(o, c) - np.abs(rng.normal(0, 0.1, n)), 2)
    v = rng.integers(100, 10_000, n).astype(float)
    return Bars(np.arange(n) * 86400, o, h, lo, c, v, timeframe="1D")


@pytest.fixture(scope="module")
def bars():
    return _bars()


@pytest.fixture(scope="module")
def cols(bars):
    return dict(run(runtime_capture, bars).columns)


def _export(bars, cols, path, k=0):
    sl = slice(k, None)
    b = Bars(bars.ts[sl], bars.open[sl], bars.high[sl], bars.low[sl], bars.close[sl],
             bars.volume[sl], timeframe="1D")  # fmt: skip
    c = {n: np.asarray(v)[sl] for n, v in cols.items()}
    write_tradingview_csv(
        TradingViewExport(b, c, ("time", "open", "high", "low", "close", "Volume", *c), "unix"),
        path,
    )
    return load_tradingview_csv(path, timeframe="1D")


# ---------- Pine script ↔ Python ----------


def test_constants_and_titles_match_script():
    toks, _ = tokenize(SCRIPT.read_text(encoding="utf-8"))
    found = {}
    for i, t in enumerate(toks[:-4]):
        if t.text == "const" and toks[i + 1].text in ("int", "float") and toks[i + 3].is_op("="):
            found[toks[i + 2].text] = float(toks[i + 4].text)
    assert found == {k: float(v) for k, v in CONSTANTS.items()}
    titles = [c.args[1].text.strip("\"'") for c in find_calls(toks) if c.name == "plot"]
    assert tuple(titles) == COLUMNS


def test_capture_script_lint_clean():
    r = lint_file(SCRIPT)
    assert r.findings == ()
    # The barstate probes measure exactly what the rule warns about; nothing else is waived.
    assert {f.rule for f in r.suppressed} == {"barstate-realtime"} and len(r.suppressed) == 3


def test_expected_covers_every_group_and_column(bars):
    assert set(EXPECTED) == set(GROUPS) and all(EXPECTED[g] in GROUPS[g] for g in GROUPS)
    produced = {c for g in GROUPS for c in GROUPS[g][EXPECTED[g]](bars)}
    assert produced | {"bi_volume"} == set(COLUMNS)


# ---------- runtime ↔ expected hypotheses (independent implementations) ----------


@pytest.mark.parametrize("group", sorted(GROUPS))
def test_runtime_matches_expected_hypothesis(group, bars, cols):
    for col, v in GROUPS[group][EXPECTED[group]](bars).items():
        np.testing.assert_allclose(cols[col], v, rtol=1e-9, atol=1e-9, err_msg=f"{group}/{col}")


@pytest.mark.parametrize("group", sorted(g for g in GROUPS if len(GROUPS[g]) > 1))
def test_hypotheses_are_distinguishable(group, bars):
    hyps = {h: f(bars) for h, f in GROUPS[group].items()}
    names = sorted(hyps)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert any(
                not np.allclose(hyps[a][c], hyps[b][c], rtol=1e-6, atol=1e-6, equal_nan=True)
                for c in hyps[a]
            ), f"{group}: {a} and {b} agree on this data"


# ---------- decisions ----------


def test_runtime_export_decides_every_group_as_expected(bars, cols, tmp_path):
    exp = _export(bars, cols, tmp_path / "bi.csv")
    decisions = decide(exp)
    assert {d.group: d.matching for d in decisions} == {g: (EXPECTED[g],) for g in GROUPS}
    text = summary(decisions)
    assert "## Built-ins (batch F)" in text and "all groups match the runtime" in text


@pytest.mark.parametrize(
    "group, alt",
    [("pivots", "left_strict"), ("hma", "half_round_sqrt_round"), ("cum_na", "skip"),
     ("percentile", "inclusive"), ("variance", "sample"), ("wma_na", "window_drop_renorm"),
     ("from_seconds", "exact_unit")],
)  # fmt: skip
def test_alternative_is_detected_and_named(bars, cols, tmp_path, group, alt):
    c = dict(cols) | GROUPS[group][alt](bars)
    d = next(d for d in decide(_export(bars, c, tmp_path / "alt.csv")) if d.group == group)
    assert d.matching == (alt,) and not d.ok
    assert f"| {group} | {EXPECTED[group]} | {alt} | **runtime must change**" in summary([d])


def test_truncated_export_anchors_running_totals(bars, cols, tmp_path):
    exp = _export(bars, cols, tmp_path / "tail.csv", k=400)
    decisions = {d.group: d for d in decide(exp)}
    assert all(d.ok for d in decisions.values()), {
        g: d.matching for g, d in decisions.items() if not d.ok
    }
    assert decisions["obv"].excluded == "25" and "anchored" in decisions["obv"].note
    # A wrong increment is still caught after anchoring.
    bad = dict(cols)
    obv = np.array(bad["bi_obv"])
    obv[600:] += 1.0
    bad["bi_obv"] = obv
    d = next(d for d in decide(_export(bars, bad, tmp_path / "bad.csv", k=400)) if d.group == "obv")
    assert d.matching == ()


def test_no_volume_is_reported(bars, cols, tmp_path):
    c = dict(cols)
    c["bi_volume"] = np.full(len(bars), np.nan)
    c["bi_obv"] = np.full(len(bars), np.nan)
    c["bi_pvt"] = np.full(len(bars), np.nan)
    d = {d.group: d for d in decide(_export(bars, c, tmp_path / "nv.csv"))}
    assert d["obv"].ok and "no volume" in d["obv"].note


def test_input_rounding_allowance(bars, cols, tmp_path):
    # Exported OHLC rounded to 1 decimal while the columns were computed on 2-decimal prices.
    rounded = Bars(bars.ts, *(np.round(v, 1) for v in (bars.open, bars.high, bars.low,
                   bars.close)), bars.volume, timeframe="1D")  # fmt: skip
    exp = _export(rounded, cols, tmp_path / "r.csv")
    plain = {d.group: d for d in decide(exp)}
    assert not plain["wma"].ok  # rounding alone breaks exact parity
    allowed = {d.group: d for d in decide(exp, input_rounding=0.05)}
    for g in ("sum", "wma", "hma", "variance", "percentile", "cum", "obv", "pvt", "fixnan"):
        assert EXPECTED[g] in allowed[g].matching, g
    assert "no input-rounding allowance" in allowed["pivots"].note
