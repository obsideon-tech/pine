"""Light scope analysis for lint rules (PINE_PLAN.md §10 C1, C2): single-assignment variable
tracing and enclosing ``if`` conditions. Not a parser; every limitation errs towards *not*
tracing, so an untraceable case stays ``possible`` rather than becoming a false proof.
"""

from __future__ import annotations

from collections import defaultdict

from pine.lint._lexer import Token

__all__ = ["Scope", "build_scope", "expand", "enclosing_conditions", "is_confirmed_gate"]

_REASSIGN = frozenset({":=", "+=", "-=", "*=", "/=", "%="})
_KEYWORDS = frozenset(
    {"if", "else", "for", "while", "switch", "return", "import", "export", "method", "type",
     "enum", "break", "continue"}
)  # fmt: skip
_PERSISTENT = frozenset({"var", "varip"})


def _indent(line: str) -> int:
    expanded = line.expandtabs(4)
    return len(expanded) - len(expanded.lstrip(" "))


class Scope:
    """Per-script facts: traceable bindings, tokens per line, line indents."""

    def __init__(self, tokens: list[Token], lines: list[str]):
        self.tokens = tokens
        self.lines = lines
        self.indent = [_indent(ln) for ln in lines]
        self.by_line: dict[int, list[int]] = defaultdict(list)  # line -> token indices
        self.depth_before: list[int] = []  # bracket depth before each token
        depth = 0
        for i, t in enumerate(tokens):
            self.by_line[t.line].append(i)
            self.depth_before.append(depth)
            if t.kind == "op" and t.text in "([{":
                depth += 1
            elif t.kind == "op" and t.text in ")]}":
                depth -= 1
        self.bindings: dict[str, tuple[Token, ...]] = {}
        self._collect_bindings()

    # -- statement extent: the line plus continuation while brackets are open -------------
    def _statement_tokens(self, start: int) -> list[Token]:
        toks = self.tokens
        line = toks[start].line
        out, depth, i = [], 0, start
        while i < len(toks) and (toks[i].line == line or depth > 0):
            t = toks[i]
            if t.kind == "op" and t.text in "([{":
                depth += 1
            elif t.kind == "op" and t.text in ")]}":
                depth -= 1
            out.append(t)
            if depth == 0 and i + 1 < len(toks) and toks[i + 1].line != t.line:
                break
            i += 1
        return out

    def _collect_bindings(self) -> None:
        toks = self.tokens
        decls: dict[str, int] = defaultdict(int)
        top: dict[str, tuple[Token, ...]] = {}
        untraceable: set[str] = set()

        for i, t in enumerate(toks):
            if t.kind == "op" and t.text in _REASSIGN and i > 0 and toks[i - 1].kind == "name":
                untraceable.add(toks[i - 1].text)
            if t.is_op("=>"):  # function/method parameters shadow globals
                j = i - 1
                if j >= 0 and toks[j].is_op(")"):
                    depth = 0
                    while j >= 0:
                        if toks[j].is_op(")"):
                            depth += 1
                        elif toks[j].is_op("("):
                            depth -= 1
                            if depth == 0:
                                break
                        elif toks[j].kind == "name":
                            untraceable.add(toks[j].text)
                        j -= 1

        for line_no, idxs in self.by_line.items():
            first = toks[idxs[0]]
            if first.text in _KEYWORDS or self.depth_before[idxs[0]] > 0:
                continue  # keyword statement, or a continuation inside open brackets
            stmt = self._statement_tokens(idxs[0])
            eq = next((k for k, s in enumerate(stmt) if s.is_op("=")), None)
            if eq is None or eq == 0:
                continue
            head = stmt[:eq]
            if any(h.kind == "op" and h.text in "[]" for h in head):
                for h in head:
                    if h.kind == "name":
                        untraceable.add(h.text)  # tuple destructuring
                continue
            if not all(h.kind == "name" or h.text in "<>," for h in head):
                continue
            name = head[-1].text
            decls[name] += 1
            persistent = any(h.text in _PERSISTENT for h in head)
            local = self.indent[line_no - 1] > 0
            value = tuple(stmt[eq + 1 :])
            if persistent or local or not value or value[0].text in _KEYWORDS:
                untraceable.add(name)
            else:
                top[name] = value
        for name, value in top.items():
            if decls[name] == 1 and name not in untraceable:
                self.bindings[name] = value


def build_scope(tokens: list[Token], lines: list[str]) -> Scope:
    return Scope(tokens, lines)


def expand(tokens: tuple[Token, ...], scope: Scope, depth: int = 6) -> tuple[Token, ...]:
    """Replace traceable identifiers with ``( value )``, recursively (bounded)."""
    if depth == 0:
        return tokens
    out: list[Token] = []
    changed = False
    for i, t in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if t.kind == "name" and t.text in scope.bindings and not (nxt and nxt.is_op("(")):
            out.append(Token("op", "(", t.line, t.col))
            out.extend(scope.bindings[t.text])
            out.append(Token("op", ")", t.line, t.col))
            changed = True
        else:
            out.append(t)
    return expand(tuple(out), scope, depth - 1) if changed else tuple(out)


def enclosing_conditions(scope: Scope, line_no: int) -> list[tuple[str, tuple[Token, ...]]]:
    """``(kind, condition tokens)`` for each block header enclosing ``line_no``, innermost
    first. ``kind`` is ``"if"`` (an ``if`` or ``else if`` header) or ``"else"``. A header's
    condition includes continuation lines (indent not a multiple of 4, Pine's rule)."""
    out = []
    threshold = scope.indent[line_no - 1]
    for ln in range(line_no - 1, 0, -1):
        if threshold == 0:
            break
        idxs = scope.by_line.get(ln)
        if not idxs or scope.depth_before[idxs[0]] > 0:
            continue  # blank/comment line, or a continuation inside brackets
        ind = scope.indent[ln - 1]
        if ind >= threshold:
            continue
        threshold = ind
        first = scope.tokens[idxs[0]]
        toks = [scope.tokens[i] for i in idxs]
        if first.text == "if" or (first.text == "else" and len(toks) > 1 and toks[1].text == "if"):
            start = 1 if first.text == "if" else 2
            cond = toks[start:]
            nxt = ln + 1
            while nxt <= len(scope.lines) and scope.indent[nxt - 1] % 4 != 0:
                cond += [scope.tokens[i] for i in scope.by_line.get(nxt, [])]
                nxt += 1
            out.append(("if", tuple(cond)))
        elif first.text == "else":
            out.append(("else", ()))
        else:
            out.append(("other", ()))
    return out


def is_confirmed_gate(cond: tuple[Token, ...], scope: Scope) -> bool:
    """True only if ``cond`` (traced) provably requires ``barstate.isconfirmed``: it contains
    an un-negated ``barstate.isconfirmed`` and no ``or``/``?``, which could bypass it."""
    toks = expand(cond, scope)
    if any(t.text in ("or", "?") for t in toks):
        return False
    for i, t in enumerate(toks):
        if t.text != "barstate.isconfirmed":
            continue
        negated = i > 0 and toks[i - 1].text == "not"
        cmp = toks[i + 1 : i + 3]
        if len(cmp) == 2 and (
            (cmp[0].text == "==" and cmp[1].text == "false")
            or (cmp[0].text == "!=" and cmp[1].text == "true")
        ):
            negated = True
        if not negated:
            return True
    return False
