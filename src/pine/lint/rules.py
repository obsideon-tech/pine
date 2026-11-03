"""Repainting rules (PINE_PLAN.md §6).

Each rule receives a `Context` and yields `Finding`s. Docstrings state what a rule catches and
what it misses. Rules are deliberately conservative about *proving safety*: when the linter
cannot establish that something is safe it says ``confidence="possible"`` rather than staying
silent, and when it can establish a defect it says ``"definite"``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

from pine.lint._calls import Arg, Call, matching_close, render
from pine.lint._lexer import Comment, Token
from pine.lint._scope import Scope, enclosing_conditions, expand, is_confirmed_gate

__all__ = ["Finding", "Context", "RuleInfo", "RULES", "Offset", "classify_offset"]

Category = Literal["future-leak", "realtime-repaint", "fill-timing", "display", "meta"]
Severity = Literal["error", "warning", "info"]
Confidence = Literal["definite", "possible"]


@dataclass(frozen=True)
class Finding:
    rule: str
    category: Category
    severity: Severity
    confidence: Confidence
    line: int
    col: int
    message: str
    snippet: str = ""


@dataclass(frozen=True)
class Context:
    tokens: list[Token]
    comments: list[Comment]
    calls: list[Call]
    lines: list[str]
    version: int | None
    scope: Scope

    def names(self) -> set[str]:
        return {t.text for t in self.tokens if t.kind == "name"}

    def snippet(self, line: int) -> str:
        return self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""


@dataclass(frozen=True)
class RuleInfo:
    id: str
    category: Category
    summary: str
    fn: Callable[[Context], Iterator[Finding]]


RULES: dict[str, RuleInfo] = {}


def _rule(rule_id: str, category: Category, summary: str):
    def deco(fn):
        RULES[rule_id] = RuleInfo(rule_id, category, summary, fn)
        return fn

    return deco


def _finding(ctx, rule_id, severity, confidence, tok: Token, message) -> Finding:
    info = RULES[rule_id]
    return Finding(
        rule_id,
        info.category,
        severity,
        confidence,
        tok.line,
        tok.col,
        message,
        ctx.snippet(tok.line),
    )


# ---------------------------------------------------------------- offset analysis

BUILTIN_SERIES = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "hl2",
        "hlc3",
        "ohlc4",
        "hlcc4",
        "time",
        "time_close",
    }
)
_LITERALS = frozenset({"true", "false", "na"})

Offset = Literal["proven", "definite-leak", "unproven"]


def _index_after(tokens: tuple[Token, ...], i: int) -> int | None:
    """If ``tokens[i+1:i+4]`` is ``[ k ]`` with integer literal k, return k."""
    if i + 3 < len(tokens):
        a, b, c = tokens[i + 1], tokens[i + 2], tokens[i + 3]
        if a.is_op("[") and c.is_op("]"):
            return b.int_value()
    return None


def _is_atom(head: tuple[Token, ...]) -> bool:
    if len(head) == 1:
        return head[0].kind == "name"
    if head and head[0].is_op("("):
        return matching_close(head, 0) == len(head) - 1
    if len(head) >= 3 and head[0].kind == "name" and head[1].is_op("("):
        return matching_close(head, 1) == len(head) - 1
    return False


def classify_offset(expr: tuple[Token, ...]) -> Offset:
    """Is the requested expression offset by at least one bar (PINE_PLAN.md §6.2)?

    - ``proven``: outermost form is ``atom[k]``/``(expr)[k]``/``f(...)[k]`` with literal
      ``k >= 1``; or every built-in price/time series inside carries ``[k >= 1]``, at least
      one is present, and no user identifier appears outside a second-or-later call
      argument (those are assumed to be parameters such as a length).
    - ``definite-leak``: a built-in series is read without an offset (``close``, ``close[0]``,
      ``ta.sma(close, 20)``, ``close[1] + close``) outside any bare-name call.
    - ``unproven``: anything else, including user variables, which are not traced.

    Known false negative: a user series passed as a second-or-later argument
    (``math.max(close[1], mySeries)``) is assumed to be a parameter.
    Known false positive (reported as ``unproven``, i.e. "possible"): bare-name calls such as
    ``nz(close)`` or a user function that offsets internally.
    """
    if not expr:
        return "unproven"
    if len(expr) >= 4 and expr[-1].is_op("]") and expr[-3].is_op("["):
        k = expr[-2].int_value()
        if k is not None and k >= 1 and _is_atom(expr[:-3]):
            return "proven"
    if len(expr) == 1 and expr[0].text in BUILTIN_SERIES:
        return "definite-leak"
    if len(expr) == 4 and expr[0].text in BUILTIN_SERIES and _index_after(expr, 0) == 0:
        return "definite-leak"

    # Walk with a stack recording, for each open paren: is it a call, which argument we are
    # in, and is it a bare-name (user-defined or helper) call that might offset internally.
    stack: list[list] = []  # [is_call, arg_index, is_bare_call]
    saw_series = definite = unproven = False
    for i, t in enumerate(expr):
        if t.kind == "op":
            if t.text == "(":
                prev = expr[i - 1] if i > 0 else None
                is_call = prev is not None and prev.kind == "name"
                stack.append([is_call, 0, is_call and "." not in prev.text])
            elif t.text == ")" and stack:
                stack.pop()
            elif t.text == "," and stack:
                stack[-1][1] += 1
            continue
        if t.kind != "name":
            continue
        followed_by_call = i + 1 < len(expr) and expr[i + 1].is_op("(")
        if followed_by_call or t.text in _LITERALS or "." in t.text:
            continue  # function name, literal, or namespaced constant (color.red, ...)
        k = _index_after(expr, i)
        if t.text in BUILTIN_SERIES:
            saw_series = True
            has_index = i + 1 < len(expr) and expr[i + 1].is_op("[")
            if k is not None and k >= 1:
                continue
            if has_index and k is None:
                unproven = True  # non-literal index such as close[n]
            elif any(frame[2] for frame in stack):
                unproven = True  # inside a user/helper call that may offset internally
            else:
                definite = True  # current-bar value read directly or via ta.*/math.*
            continue
        in_param_position = bool(stack) and stack[-1][0] and stack[-1][1] >= 1
        if not in_param_position:
            unproven = True
    if definite:
        return "definite-leak"
    if unproven or not saw_series:
        return "unproven"
    return "proven"


# ---------------------------------------------------------------- request.security

_SECURITY = ("request.security", "security")


def _lookahead(call: Call) -> Literal["on", "off", "unresolved"]:
    a = call.arg("lookahead", 4)
    if a is None or not a.tokens:
        return "off"
    if len(a.tokens) == 1 and a.tokens[0].text == "barmerge.lookahead_on":
        return "on"
    if len(a.tokens) == 1 and a.tokens[0].text == "barmerge.lookahead_off":
        return "off"
    return "unresolved"


def _expr(call: Call) -> Arg | None:
    return call.arg("expression", 2)


@_rule(
    "security-lookahead-leak",
    "future-leak",
    "request.security with lookahead_on and an expression not offset by >= 1 bar",
)
def _security_leak(ctx: Context):
    """Catches lookahead_on with an unoffset expression: on historical bars the value of the
    higher-timeframe bar is visible before that bar has closed. User variables assigned once
    at the top level are traced (§10 C1); others (``var``, reassigned, local, parameters) are
    not, and stay ``possible``. Misses: lookahead passed via a variable (see
    security-lookahead-unresolved)."""
    for c in ctx.calls:
        if c.name not in _SECURITY or _lookahead(c) != "on":
            continue
        e = _expr(c)
        raw = e.tokens if e else ()
        traced = expand(raw, ctx.scope)
        cls = classify_offset(traced)
        if cls == "proven":
            continue
        via = f" (traced: `{render(traced)}`)" if traced != raw else ""
        if cls == "definite-leak":
            bare = len(e.tokens) == 1 or (len(e.tokens) == 4 and e.tokens[2].text == "0")
            fix = f"{e.tokens[0].text}[1]" if bare else f"({e.text})[1]"
            yield _finding(
                ctx,
                "security-lookahead-leak",
                "error",
                "definite",
                c.token,
                f"lookahead_on with `{e.text}`{via} reads the current bar: future data leaks "
                f"into historical bars; request `{fix}` instead",
            )
        else:
            yield _finding(
                ctx,
                "security-lookahead-leak",
                "warning",
                "possible",
                c.token,
                f"lookahead_on with `{e.text if e else '?'}`{via}: cannot prove a >= 1 bar "
                "offset; if it is not offset, this leaks future data on historical bars",
            )


@_rule(
    "security-lookahead-unresolved",
    "future-leak",
    "request.security lookahead argument cannot be resolved statically",
)
def _security_unresolved(ctx: Context):
    """Catches a lookahead argument given by a variable or expression. Misses nothing it
    reports, but cannot say whether lookahead is on."""
    for c in ctx.calls:
        if c.name in _SECURITY and _lookahead(c) == "unresolved":
            a = c.arg("lookahead", 4)
            yield _finding(
                ctx,
                "security-lookahead-unresolved",
                "warning",
                "possible",
                c.token,
                f"lookahead = `{a.text}` cannot be resolved; if it can be lookahead_on, the "
                "expression must be offset by >= 1 bar",
            )


@_rule(
    "security-realtime-repaint",
    "realtime-repaint",
    "request.security without offset: realtime value changes until the requested bar closes",
)
def _security_realtime(ctx: Context):
    """Catches request.security with lookahead off and no provable offset, where the
    timeframe is not the chart's. History shows the requested bar's final value; realtime
    shows a value that changes until it closes. Misses: the timeframe argument is not
    evaluated, so a lower or equal timeframe given by a variable is still reported
    (``possible``)."""
    for c in ctx.calls:
        if c.name not in _SECURITY or _lookahead(c) != "off":
            continue
        tf = c.arg("timeframe", 1)
        if tf is not None and tf.text in ("timeframe.period", '""', "''"):
            continue
        e = _expr(c)
        if classify_offset(expand(e.tokens if e else (), ctx.scope)) == "proven":
            continue
        yield _finding(
            ctx,
            "security-realtime-repaint",
            "warning",
            "possible",
            c.token,
            f"`{e.text if e else '?'}` from timeframe `{tf.text if tf else '?'}` without an "
            "offset: on realtime bars the value changes until the requested bar closes; "
            "history shows only the final value. Use `expr[1]` with lookahead_on",
        )


@_rule(
    "security-lower-tf",
    "realtime-repaint",
    "request.security_lower_tf: intrabar data differs between history and realtime",
)
def _security_lower_tf(ctx: Context):
    """Catches every request.security_lower_tf call. Whether it repaints depends on use."""
    for c in ctx.calls:
        if c.name == "request.security_lower_tf":
            yield _finding(
                ctx,
                "security-lower-tf",
                "info",
                "possible",
                c.token,
                "intrabar data: available history is limited and the realtime bar's array "
                "grows tick by tick; check results on historical vs realtime bars",
            )


# ---------------------------------------------------------------- declarations


def _named_true(call: Call, name: str) -> Arg | None:
    a = call.arg(name)
    if a is not None and len(a.tokens) == 1 and a.tokens[0].text == "true":
        return a
    return None


@_rule("calc-on-every-tick", "realtime-repaint", "calc_on_every_tick = true")
def _calc_every_tick(ctx: Context):
    """Catches ``calc_on_every_tick = true`` in strategy()/indicator() declarations. History
    executes once per bar; realtime executes on every tick, so state and signals differ."""
    for c in ctx.calls:
        if c.name in ("strategy", "indicator") and (a := _named_true(c, "calc_on_every_tick")):
            yield _finding(
                ctx,
                "calc-on-every-tick",
                "warning",
                "definite",
                a.tokens[0],
                "calc_on_every_tick = true: realtime runs every tick, history once per bar; "
                "backtest and live behaviour differ",
            )


@_rule("fill-timing", "fill-timing", "optimistic strategy fill settings")
def _fill_timing(ctx: Context):
    """Catches ``calc_on_order_fills = true`` and ``process_orders_on_close = true``. These
    are fill-timing assumptions, not repainting: they can make a backtest optimistic."""
    for c in ctx.calls:
        if c.name != "strategy":
            continue
        for name in ("calc_on_order_fills", "process_orders_on_close"):
            if a := _named_true(c, name):
                yield _finding(
                    ctx,
                    "fill-timing",
                    "info",
                    "definite",
                    a.tokens[0],
                    f"{name} = true: fills are assumed at prices or times a live order may "
                    "not get; check backtest realism",
                )


# ---------------------------------------------------------------- keywords and variables


@_rule("varip", "realtime-repaint", "varip state persists across realtime ticks")
def _varip(ctx: Context):
    """Catches every ``varip`` declaration. On history it behaves like ``var``; on realtime
    it keeps changes made by intrabar ticks, so the two diverge."""
    for t in ctx.tokens:
        if t.kind == "name" and t.text == "varip":
            yield _finding(
                ctx,
                "varip",
                "warning",
                "definite",
                t,
                "varip keeps intrabar updates on realtime bars but not on history",
            )


_BARSTATE = ("barstate.isrealtime", "barstate.isnew", "barstate.islast")


@_rule("barstate-realtime", "realtime-repaint", "logic depending on realtime-only bar state")
def _barstate(ctx: Context):
    """Catches ``barstate.isrealtime``, ``barstate.isnew`` and ``barstate.islast``. Each
    evaluates differently on history than live. Misses other realtime-only state
    (e.g. ``syminfo`` changes, ``request.*`` timing)."""
    for t in ctx.tokens:
        if t.kind == "name" and t.text in _BARSTATE:
            yield _finding(
                ctx,
                "barstate-realtime",
                "warning",
                "possible",
                t,
                f"{t.text} evaluates differently on historical and realtime bars",
            )


@_rule("timenow", "realtime-repaint", "timenow used")
def _timenow(ctx: Context):
    """Catches ``timenow``: the current wall-clock time, which differs between a historical
    recalculation and live execution."""
    for t in ctx.tokens:
        if t.kind == "name" and t.text == "timenow":
            yield _finding(
                ctx,
                "timenow",
                "warning",
                "possible",
                t,
                "timenow is wall-clock time; results depend on when the script runs",
            )


# ---------------------------------------------------------------- alerts


@_rule("alert-unconfirmed", "realtime-repaint", "alert() may fire on an unconfirmed bar")
def _alert(ctx: Context):
    """Catches ``alert()`` without ``freq = alert.freq_once_per_bar_close`` that is not inside
    an ``if``/``else if`` whose condition provably requires ``barstate.isconfirmed`` (§10 C2):
    un-negated, no ``or``/``?`` that could bypass it; traced variables are expanded. The
    default frequency fires intrabar, on a condition that may be false at bar close.
    Misses/limits: gating inside a called function is not seen (reported ``possible``);
    an ``else`` branch never counts as gated."""
    for c in ctx.calls:
        if c.name != "alert":
            continue
        f = c.arg("freq", 1)
        if f is not None and f.text == "alert.freq_once_per_bar_close":
            continue
        gated = any(
            kind == "if" and is_confirmed_gate(cond, ctx.scope)
            for kind, cond in enclosing_conditions(ctx.scope, c.token.line)
        )
        if gated:
            continue
        yield _finding(
            ctx,
            "alert-unconfirmed",
            "warning",
            "possible",
            c.token,
            "alert() can fire intrabar on a condition that is false at bar close; use "
            "freq = alert.freq_once_per_bar_close or put it inside "
            "`if ... and barstate.isconfirmed`",
        )


@_rule(
    "alertcondition-unconfirmed",
    "realtime-repaint",
    "alertcondition() whose condition does not require barstate.isconfirmed",
)
def _alertcondition(ctx: Context):
    """Catches ``alertcondition()`` whose condition argument (traced) does not provably
    require ``barstate.isconfirmed``. Whether it repaints also depends on the frequency
    chosen in the alert dialog, which the script cannot see; hence ``info``."""
    for c in ctx.calls:
        if c.name != "alertcondition":
            continue
        cond = c.arg("condition", 0)
        if cond is not None and is_confirmed_gate(cond.tokens, ctx.scope):
            continue
        yield _finding(
            ctx,
            "alertcondition-unconfirmed",
            "info",
            "possible",
            c.token,
            "set the alert to 'Once Per Bar Close', or include `and barstate.isconfirmed` in "
            "the condition",
        )


# ---------------------------------------------------------------- history and display


@_rule("negative-history-index", "future-leak", "history reference with a negative index")
def _neg_index(ctx: Context):
    """Catches ``x[-n]`` with a literal negative index (a reference into the future). Pine is
    believed to reject this at compile time (unverified); the rule guards hand-edited or
    generated code."""
    toks = ctx.tokens
    for i in range(1, len(toks) - 3):
        prev, br = toks[i - 1], toks[i]
        if not br.is_op("[") or prev.line != br.line:
            continue
        if not (prev.kind == "name" or prev.is_op(")") or prev.is_op("]")):
            continue
        if toks[i + 1].is_op("-") and toks[i + 2].int_value() and toks[i + 3].is_op("]"):
            yield _finding(
                ctx,
                "negative-history-index",
                "error",
                "definite",
                br,
                f"`[-{toks[i + 2].text}]` references a future bar",
            )


_PLOTS = ("plot", "plotshape", "plotchar", "plotarrow", "plotcandle", "plotbar", "bgcolor")


@_rule("negative-plot-offset", "display", "plot drawn with a negative offset")
def _neg_offset(ctx: Context):
    """Catches a plot with literal ``offset = -n``. The values are not repainted, but they are
    drawn on bars before they could have been known (typical with pivots), which misleads
    visual review. Misses offsets given by a variable."""
    for c in ctx.calls:
        if c.name not in _PLOTS:
            continue
        a = c.arg("offset")
        if a and len(a.tokens) == 2 and a.tokens[0].is_op("-") and a.tokens[1].int_value():
            yield _finding(
                ctx,
                "negative-plot-offset",
                "info",
                "definite",
                a.tokens[0],
                f"offset = {a.text.replace(' ', '')}: values are drawn {a.tokens[1].text} bars "
                "before they were known",
            )


# ---------------------------------------------------------------- meta

_VERSION = re.compile(r"^\s*@version\s*=\s*(\d+)")


def parse_version(comments: list[Comment]) -> tuple[int | None, Comment | None]:
    for c in comments:
        m = _VERSION.match(c.text)
        if m:
            return int(m.group(1)), c
    return None, None


@_rule("unsupported-version", "meta", "//@version missing or below 5")
def _version(ctx: Context):
    """Rules are written for Pine v5/v6. Older scripts use different names (``security``
    without ``request.``, different defaults) and are linted on a best-effort basis."""
    v, c = parse_version(ctx.comments)
    if v is not None and v >= 5:
        return
    line, col = (c.line, c.col) if c else (1, 1)
    msg = "no //@version annotation" if v is None else f"//@version={v}"
    yield Finding(
        "unsupported-version",
        "meta",
        "info",
        "definite",
        line,
        col,
        f"{msg}: rules target Pine v5/v6; results are best-effort",
        ctx.snippet(line),
    )
