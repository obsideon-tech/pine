import json
from pathlib import Path

from pine.lint import DISCLAIMER
from pine.lint.cli import main

FX = Path(__file__).parent / "fixtures" / "pine"


def test_clean_exits_zero_with_disclaimer(capsys):
    assert main([str(FX / "clean_indicator.pine")]) == 0
    assert DISCLAIMER in capsys.readouterr().out


def test_error_exits_one():
    assert main([str(FX / "security_lookahead_pos.pine")]) == 1


def test_warnings_only_need_strict():
    f = str(FX / "alerts_pos.pine")
    assert main([f]) == 0
    assert main([f, "--strict"]) == 1


def test_json_is_a_list_of_reports(capsys):
    main([str(FX / "alerts_pos.pine"), str(FX / "clean_indicator.pine"), "--format", "json"])
    data = json.loads(capsys.readouterr().out)
    assert [len(d["findings"]) for d in data] == [3, 0]


def test_unreadable_file_exits_two(tmp_path, capsys):
    assert main([str(tmp_path / "missing.pine")]) == 2
    bad = tmp_path / "bad.pine"
    bad.write_text('x = "unterminated\n', encoding="utf-8")
    assert main([str(bad)]) == 2
