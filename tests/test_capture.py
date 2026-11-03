"""pine.capture: manifest writing and the combined check (docs/CAPTURE.md)."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_reference import synthetic_export

from pine.capture import main
from pine.io import TradingViewExport, load_tradingview_csv, write_tradingview_csv
from pine.manifest import load_manifest
from pine.runtime import run
from pine.semantics import runtime_capture
from pine.strategy import StrategyConfig
from pine.strategy_capture import strategy_capture

ARGS = ["--symbol", "EXAMPLE:SYM", "--timeframe", "60", "--timezone", "UTC"]


def combined_export(tmp_path: Path, semantics: bool = True) -> Path:
    """A synthetic export of a chart carrying both capture scripts."""
    ref = load_tradingview_csv(synthetic_export("propagate", tmp_path), timeframe="60")
    cols = dict(ref.columns)
    if semantics:
        cols |= dict(run(runtime_capture, ref.bars).columns)
        cols |= dict(run(strategy_capture, ref.bars, strategy=StrategyConfig()).columns)
    exp = TradingViewExport(
        ref.bars, cols, ("time", "open", "high", "low", "close", "Volume", *cols), "unix"
    )
    path = tmp_path / ("both.csv" if semantics else "ref_only.csv")
    write_tradingview_csv(exp, path)
    return path


def test_manifest_written_valid_and_not_overwritten(tmp_path, capsys):
    csv = combined_export(tmp_path)
    assert main(["manifest", str(csv), *ARGS]) == 0
    m = load_manifest(csv.with_suffix(".yaml"))
    assert m.symbol == "EXAMPLE:SYM" and m.timeframe == "60" and m.adjustment == "splits"
    assert m.indicator["name"] == (
        "pine reference capture + pine semantics capture + pine strategy capture"
    )
    assert m.indicator["inputs"]["semantics.ST_FACTOR"] == 3.0
    assert m.displayed_precision >= 2
    assert "next: uv run python -m pine.capture check" in capsys.readouterr().out
    assert main(["manifest", str(csv), *ARGS]) == 2
    assert "--force" in capsys.readouterr().err
    assert main(["manifest", str(csv), *ARGS, "--force"]) == 0


@pytest.mark.parametrize(
    "bad, msg", [(["--timeframe", "1h"], "timeframe"), (["--timezone", "Mars/Base"], "timezone")]
)
def test_manifest_rejects_invalid_values(tmp_path, capsys, bad, msg):
    csv = combined_export(tmp_path)
    args = dict(zip(ARGS[::2], ARGS[1::2], strict=True)) | dict([bad])
    flat = [x for kv in args.items() for x in kv]
    assert main(["manifest", str(csv), *flat]) == 2
    assert msg in capsys.readouterr().err
    assert not csv.with_suffix(".yaml").exists()


def test_manifest_needs_capture_columns(tmp_path, capsys):
    plain = Path(__file__).parent / "fixtures" / "synthetic" / "random_walk_60.csv"
    target = tmp_path / "plain.csv"
    target.write_bytes(plain.read_bytes())
    assert main(["manifest", str(target), *ARGS]) == 2
    assert "none of the capture columns" in capsys.readouterr().err


def test_check_runs_both_and_writes_report(tmp_path, capsys):
    csv = combined_export(tmp_path)
    main(["manifest", str(csv), *ARGS])
    capsys.readouterr()
    out = tmp_path / "report.md"
    assert main(["check", str(csv), "--out", str(out)]) == 0
    text = out.read_text()
    assert "**Overall: PASS**" in text
    assert "## Gap-free columns" in text and "## Pine semantics" in text
    assert "## Strategy (broker emulator)" in text
    assert "Symbol `EXAMPLE:SYM`" in text and "Skipped" not in text


def test_check_reference_only_reports_semantics_skipped(tmp_path, capsys):
    csv = combined_export(tmp_path, semantics=False)
    assert main(["check", str(csv), "--timeframe", "60"]) == 0
    out = capsys.readouterr().out
    assert "## Skipped" in out and "capture_semantics.pine not in the export" in out


def test_check_json(tmp_path, capsys):
    csv = combined_export(tmp_path)
    assert main(["check", str(csv), "--timeframe", "60", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["passed"] and data["skipped"] == []
    assert {d["group"] for d in data["semantics"]} >= {"cond_ema", "rounding"}


def test_check_errors_are_actionable(tmp_path, capsys):
    csv = combined_export(tmp_path)
    assert main(["check", str(csv)]) == 2  # no manifest next to it, no --timeframe
    err = capsys.readouterr().err
    assert "manifest not found" in err and "python -m pine.capture manifest" in err
    assert main(["check", str(tmp_path / "EXPORT.csv")]) == 2
    assert "export not found" in capsys.readouterr().err


def test_broken_strategy_bar_index_is_reported_and_others_still_run(tmp_path, capsys):
    csv = combined_export(tmp_path)
    exp = load_tradingview_csv(csv, timeframe="60")
    cols = dict(exp.columns)
    sbi = np.array(cols["strat_bar_index"])
    sbi[40] = np.nan
    cols["strat_bar_index"] = sbi
    broken = tmp_path / "broken.csv"
    write_tradingview_csv(TradingViewExport(exp.bars, cols, exp.header, exp.time_format), broken)
    assert main(["check", str(broken), "--timeframe", "60"]) == 1
    out = capsys.readouterr().out
    assert "## Could not run" in out and "strat_bar_index: na at row 40" in out
    assert "## Gap-free columns" in out and "## Pine semantics" in out


def test_bar_offset_messages():
    from pine.truncation import bar_offset

    with pytest.raises(ValueError, match=r"x: not consecutive at row 2 \(1 then 5"):
        bar_offset({"x": np.array([0.0, 1.0, 5.0])}, "x")
    with pytest.raises(ValueError, match="first value -1"):
        bar_offset({"bar_index": np.array([-1.0, 0.0])})
