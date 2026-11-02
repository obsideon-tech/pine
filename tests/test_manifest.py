import copy

import pytest

from pine.manifest import ManifestError, check_fixture_dir, validate_manifest

BASE = {
    "source": "tradingview_export",
    "symbol_class": "public",
    "symbol": "EXAMPLE:SPY",
    "timeframe": "1D",
    "session": "regular",
    "timezone": "America/New_York",
    "adjustment": "splits_dividends",
    "chart_type": "candles",
    "pine_version": 6,
    "export_date": "2026-11-03",
    "timestamp_convention": "bar_open",
    "displayed_precision": 2,
    "indicator": {"name": "RSI", "inputs": {"length": 14}},
}


def test_valid_manifest():
    m = validate_manifest(copy.deepcopy(BASE))
    assert m.export_date.isoformat() == "2026-11-03"
    assert m.indicator["inputs"]["length"] == 14


@pytest.mark.parametrize("field", list(BASE.keys() - {"symbol"}))
def test_every_required_field_is_enforced(field):
    data = copy.deepcopy(BASE)
    del data[field]
    with pytest.raises(ManifestError, match=f"missing required field '{field}'"):
        validate_manifest(data)


@pytest.mark.parametrize(
    "over, msg",
    [
        ({"chart_type": "heikin_ashi"}, "chart_type"),
        ({"pine_version": 4}, "pine_version"),
        ({"timeframe": "1h"}, "timeframe"),
        ({"timezone": "EST5EDT-ish"}, "timezone"),
        ({"export_date": "03/11/2026"}, "export_date"),
        ({"symbol": None}, "symbol is required"),
        ({"displayed_precision": None}, "required for a TradingView export"),
        ({"displayed_precision": -1}, "non-negative"),
        ({"displayed_precision": True}, "non-negative"),
        ({"indicator": {"inputs": {}}}, "indicator"),
        ({"source": "synthetic"}, "synthetic source"),
        ({"surprise": 1}, "unknown field"),
    ],
)
def test_invalid_values_rejected(over, msg):
    data = {**copy.deepcopy(BASE), **over}
    with pytest.raises(ManifestError, match=msg):
        validate_manifest(data)


def test_all_problems_reported_together():
    data = {**BASE, "chart_type": "renko", "pine_version": 3}
    with pytest.raises(ManifestError) as ei:
        validate_manifest(data)
    assert len(ei.value.problems) == 2


def test_committed_fixtures_all_have_valid_manifests(fixtures_dir):
    found = check_fixture_dir(fixtures_dir)
    assert len(found) >= 2


def test_missing_manifest_detected(tmp_path):
    (tmp_path / "orphan.csv").write_text("time,open,high,low,close\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="no manifest"):
        check_fixture_dir(tmp_path)


def test_yaml_numeric_timeframe_kept_as_string(tmp_path):
    lines = [f"{k}: {v}" for k, v in BASE.items() if k not in ("timeframe", "indicator")]
    lines += ["timeframe: 60", "indicator: null"]
    (tmp_path / "a.csv").write_text("", encoding="utf-8")
    (tmp_path / "a.yaml").write_text("\n".join(lines), encoding="utf-8")
    m = check_fixture_dir(tmp_path)[tmp_path / "a.csv"]
    assert m.timeframe == "60"
