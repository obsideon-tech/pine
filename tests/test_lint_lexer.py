import pytest

from pine.lint._calls import find_calls
from pine.lint._lexer import LexError, tokenize
from pine.lint.rules import classify_offset


def kinds(src):
    return [(t.kind, t.text) for t in tokenize(src)[0]]


def test_dotted_names_strings_colors_and_ops():
    toks = kinds("x := request.security(\"A//B\", 'D', close[1]) == #FF00AA80")
    assert ("name", "request.security") in toks
    assert ("string", '"A//B"') in toks and ("string", "'D'") in toks
    assert ("op", ":=") in toks and ("op", "==") in toks
    assert ("color", "#FF00AA80") in toks


def test_comments_are_separated_and_code_in_comments_ignored():
    toks, comments = tokenize("a = 1 // varip close[-1]\n//@version=6\n")
    assert [t.text for t in toks] == ["a", "=", "1"]
    assert [(c.line, c.text.strip()) for c in comments] == [
        (1, "varip close[-1]"),
        (2, "@version=6"),
    ]


def test_string_escapes_and_unterminated():
    toks, _ = tokenize(r'"a \" // b" x')
    assert [t.kind for t in toks] == ["string", "name"]
    with pytest.raises(LexError):
        tokenize('"open')


def test_line_and_column():
    toks, _ = tokenize("a = 1\n  bb = close[2]\n")
    bb = next(t for t in toks if t.text == "bb")
    assert (bb.line, bb.col) == (2, 3)


def test_numbers():
    assert [t.text for t in tokenize("1 2.5 .5 1e-3 3.")[0]] == ["1", "2.5", ".5", "1e-3", "3."]


def test_calls_multiline_nested_and_named_args():
    src = "x = f(a, g(b, [1, 2]),\n     c = h(d)[1],\n     e = 'k, l')\n"
    calls = {c.name: c for c in find_calls(tokenize(src)[0])}
    f = calls["f"]
    assert [a.name for a in f.args] == [None, None, "c", "e"]
    assert f.args[1].text == "g(b, [1, 2])"
    assert f.arg("c").text == "h(d)[1]"
    assert f.arg("missing", 1) is f.args[1]
    assert f.arg("missing", 5) is None
    assert {"g", "h"} <= calls.keys()


def test_equality_is_not_a_named_argument():
    (call,) = [c for c in find_calls(tokenize("f(a == b)")[0]) if c.name == "f"]
    assert call.args[0].name is None


def _off(expr):
    return classify_offset(tuple(tokenize(expr)[0]))


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("close[1]", "proven"),
        ("close[3]", "proven"),
        ("(high + low)[1]", "proven"),
        ("ta.ema(close, 20)[1]", "proven"),
        ("ta.sma(close[1], len)", "proven"),
        ("close[2] - open[1]", "proven"),
        ("math.max(high[1], low[1])", "proven"),
        ("close", "definite-leak"),
        ("hl2", "definite-leak"),
        ("close[0]", "definite-leak"),
        ("ta.sma(close, 20)", "definite-leak"),
        ("nz(close)", "unproven"),
        ("myFunc(close, 3)", "unproven"),
        ("src", "unproven"),
        ("src[1]", "proven"),
        ("close[1] + src", "unproven"),
        ("ta.sma(src, 20)", "unproven"),
        ("close[n]", "unproven"),
        ("close[1] + close", "definite-leak"),
        ("", "unproven"),
    ],
)
def test_classify_offset(expr, expected):
    assert _off(expr) == expected
