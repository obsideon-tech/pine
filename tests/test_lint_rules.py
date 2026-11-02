import re
from pathlib import Path

import pytest

from pine.lint import DISCLAIMER, RULES, lint_file, lint_source
from pine.lint._lexer import tokenize

FIXTURES = Path(__file__).parent / "fixtures" / "pine"
ALL = sorted(FIXTURES.glob("*.pine"))
EXPECT = re.compile(r"expect:\s*([\w\-, ]+)")


def expectations(path: Path) -> set[tuple[str, int]]:
    _, comments = tokenize(path.read_text(encoding="utf-8"))
    out = set()
    for c in comments:
        m = EXPECT.search(c.text)
        if m:
            out |= {(r.strip(), c.line) for r in m.group(1).split(",") if r.strip()}
    return out


@pytest.mark.parametrize("path", ALL, ids=[p.name for p in ALL])
def test_fixture_findings_match_annotations_exactly(path):
    found = {(f.rule, f.line) for f in lint_file(path).findings}
    assert found == expectations(path)


def test_every_rule_has_positive_and_negative_fixture():
    pos = {r for p in ALL for r, _ in expectations(p)}
    assert pos == set(RULES), f"rules without a positive fixture: {set(RULES) - pos}"
    # Negative coverage: each rule's trigger family appears in a fixture with no expectation for
    # that rule (the clean script plus *_neg files cover all families).
    negs = [p for p in ALL if p.name.endswith(("_neg.pine", "_neg_freq.pine"))]
    assert len(negs) >= 6
    for p in [*negs, FIXTURES / "clean_indicator.pine"]:
        assert not lint_file(p).findings, p.name


def test_clean_script_has_zero_findings():
    r = lint_file(FIXTURES / "clean_indicator.pine")
    assert r.findings == () and r.suppressed == () and r.version == 6


def test_confidence_definite_vs_possible():
    r = lint_file(FIXTURES / "security_lookahead_pos.pine")
    by_line = {f.line: f for f in r.findings}
    assert (by_line[3].severity, by_line[3].confidence) == ("error", "definite")
    assert (by_line[7].severity, by_line[7].confidence) == ("warning", "possible")  # src
    assert (by_line[8].severity, by_line[8].confidence) == ("error", "definite")  # ta.sma(close)
    assert "request `(ta.sma(close, 20))[1]`" in by_line[8].message
    assert all(f.category == "future-leak" for f in r.findings)


def test_suppression_is_per_rule_and_reported():
    r = lint_file(FIXTURES / "suppressed.pine")
    assert [(f.rule, f.line) for f in r.suppressed] == [("security-lookahead-leak", 3)]
    assert [(f.rule, f.line) for f in r.findings] == [("varip", 4)]
    assert "1 suppressed" in r.to_text()


# Injection test: the lint analogue of the mutation suite. Each pattern is inserted into the
# clean script before its final `if` block; exactly that rule must fire at the inserted line.
INJECTIONS = {
    "security-lookahead-leak": 'z = request.security(syminfo.tickerid, "D", close, lookahead = barmerge.lookahead_on)',
    "security-lookahead-unresolved": 'z = request.security(syminfo.tickerid, "D", close[1], lookahead = myLa)',
    "security-realtime-repaint": 'z = request.security(syminfo.tickerid, "D", close)',
    "security-lower-tf": 'z = request.security_lower_tf(syminfo.tickerid, "1", close)',
    "calc-on-every-tick": "indicator('x', calc_on_every_tick = true)",
    "fill-timing": "strategy('x', process_orders_on_close = true)",
    "varip": "varip float z = 0.0",
    "barstate-realtime": "z = barstate.isrealtime",
    "timenow": "z = timenow",
    "alert-unconfirmed": 'alert("x")',
    "alertcondition-unconfirmed": 'alertcondition(longSignal, "x", "y")',
    "negative-history-index": "z = high[-2]",
    "negative-plot-offset": "plot(r, offset = -3)",
}


def test_injection_covers_every_rule_except_meta():
    assert set(INJECTIONS) == set(RULES) - {"unsupported-version"}


@pytest.mark.parametrize("rule", sorted(INJECTIONS))
def test_injected_pattern_detected_at_its_line(rule):
    lines = (FIXTURES / "clean_indicator.pine").read_text(encoding="utf-8").splitlines()
    at = next(i for i, ln in enumerate(lines) if ln.startswith("if longSignal"))
    lines.insert(at, INJECTIONS[rule])
    r = lint_source("\n".join(lines) + "\n")
    assert [(f.rule, f.line) for f in r.findings] == [(rule, at + 1)]


def test_text_and_json_outputs_carry_disclaimer():
    r = lint_file(FIXTURES / "alerts_pos.pine")
    text = r.to_text()
    assert DISCLAIMER in text and "alerts_pos.pine:5:5: warning [alert-unconfirmed]" in text
    d = r.to_dict()
    assert d["disclaimer"] == DISCLAIMER and len(d["findings"]) == 3
    assert set(d["findings"][0]) == {
        "rule",
        "category",
        "severity",
        "confidence",
        "line",
        "col",
        "message",
        "snippet",
    }
