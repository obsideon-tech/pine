"""Verification registry (PINE_PLAN.md §11 H2): consistent with the code, and the README
section is generated from it."""

import copy
from pathlib import Path

import pytest

from pine import registry

ROOT = Path(__file__).resolve().parent.parent


def test_registry_is_consistent_with_the_code():
    assert registry.problems() == []


def test_readme_section_is_up_to_date():
    assert registry.main(["--check", str(ROOT / "README.md")]) == 0


def _broken(edit):
    data = copy.deepcopy(registry.load())
    edit(data)
    return registry.problems(data, root=ROOT)


@pytest.mark.parametrize(
    "edit, msg",
    [
        (lambda d: d["builtins"].pop("wma"), "pine.ta.wma has no entry"),
        (lambda d: d["builtins"].__setitem__("nope", {}), "'nope' is not a public"),
        (lambda d: d["builtins"]["wma"].__setitem__("runtime", None), "ctx.ta.wma is not mapped"),
        (lambda d: d["builtins"]["wma"].__setitem__("runtime", "wmaa"), "ctx.ta.wmaa does not"),
        (lambda d: d["builtins"]["sma"]["tradingview"].append("x"), "unknown export 'x'"),
        (lambda d: d["builtins"]["sma"].__setitem__("capture", "cam"), "unknown capture"),
        (lambda d: d["builtins"]["sma"].__setitem__("bound", "LIPSCHITZ:zz"), "bound"),
        (lambda d: d["builtins"]["sma"].__setitem__("bound", None), "no bound and no note"),
        (lambda d: d["builtins"]["sma"].__setitem__("mutation", "tests/nope.py"), "missing"),
        (
            lambda d: d["builtins"]["tr"].__setitem__("mutation", "tests/test_mutation_batch_f.py"),
            "never calls ta.tr",
        ),
        (
            lambda d: d["builtins"]["wma"].__setitem__("na_policy", ["nflx_1d"]),
            "did not verify it",
        ),
        (lambda d: d["decisions"].pop("semantics.rounding"), "semantics.rounding has no entry"),
        (lambda d: d["decisions"].__setitem__("semantics.zz", {}), "not a probe group"),
        (
            lambda d: d["decisions"]["semantics.rounding"].__setitem__("expected", "half_up"),
            "code implements 'half_away'",
        ),
        (lambda d: d["exports"].__setitem__("orphan", {}), "orphan is not referenced"),
    ],
)  # fmt: skip
def test_drift_is_detected(edit, msg):
    assert any(msg in p for p in _broken(edit)), _broken(edit)


def test_status_is_derived():
    assert registry.status({"tradingview": ["a"]}) == "verified"
    assert registry.status({"tradingview": [], "capture": "builtins"}) == "capture ready"
    assert registry.status({"tradingview": []}) == "unverified"


def test_markdown_lists_every_pine_builtin_and_pending_rule():
    md = registry.markdown()
    data = registry.load()
    for b in data["builtins"].values():
        if b["pine"]:
            assert f"`{b['pine']}`" in md
    pending = [g for g, d in data["decisions"].items() if not d["tradingview"]]
    assert all(f"`{g}`" in md for g in pending) and md.startswith(registry.START)


def test_cli_modes(tmp_path, capsys):
    assert registry.main([]) == 0
    assert registry.START in capsys.readouterr().out
    f = tmp_path / "R.md"
    f.write_text(f"x\n{registry.START}\nold\n{registry.END}\ny\n")
    assert registry.main(["--check", str(f)]) == 1
    assert "out of date" in capsys.readouterr().err
    assert registry.main(["--write", str(f)]) == 0
    assert registry.main(["--check", str(f)]) == 0
    text = f.read_text()
    assert text.startswith("x\n") and text.endswith("\ny\n")
    f.write_text("no markers")
    assert registry.main(["--check", str(f)]) == 2
    assert registry.main(["--check", str(tmp_path / "missing.md")]) == 2


def test_cli_refuses_an_inconsistent_registry(monkeypatch, capsys):
    monkeypatch.setattr(registry, "problems", lambda: ["boom"])
    assert registry.main([]) == 2
    assert "boom" in capsys.readouterr().err


def test_load_rejects_wrong_schema(monkeypatch):
    registry.load.cache_clear()
    monkeypatch.setattr(registry.yaml, "safe_load", lambda text: {"schema": 99})
    with pytest.raises(ValueError, match="schema"):
        registry.load()
    registry.load.cache_clear()
