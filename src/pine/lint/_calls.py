"""Find function calls in a token stream and split their arguments."""

from __future__ import annotations

from dataclasses import dataclass

from pine.lint._lexer import Token

__all__ = ["Arg", "Call", "find_calls", "matching_close", "render"]


def render(tokens) -> str:
    """Tokens back to compact source text: ``ta.sma(close[1], 20)``."""
    out = ""
    prev = None
    for t in tokens:
        indexes_prev = prev is not None and (prev.kind == "name" or prev.text in (")", "]"))
        tight_before = (t.kind == "op" and t.text in (")", "]", ",")) or (
            t.kind == "op" and t.text in ("(", "[") and indexes_prev
        )
        tight_after_prev = prev is not None and prev.kind == "op" and prev.text in "(["
        if out and not tight_before and not tight_after_prev:
            out += " "
        out += t.text
        prev = t
    return out


_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {v: k for k, v in _OPEN.items()}


@dataclass(frozen=True)
class Arg:
    name: str | None
    tokens: tuple[Token, ...]  # value tokens only (after "name =" if named)

    @property
    def text(self) -> str:
        return render(self.tokens)

    @property
    def first(self) -> Token | None:
        return self.tokens[0] if self.tokens else None


@dataclass(frozen=True)
class Call:
    name: str
    token: Token
    args: tuple[Arg, ...]

    def arg(self, name: str, position: int | None = None) -> Arg | None:
        """Named argument ``name``, else the positional one at ``position`` (0-based)."""
        for a in self.args:
            if a.name == name:
                return a
        if position is not None:
            positional = [a for a in self.args if a.name is None]
            if position < len(positional):
                return positional[position]
        return None


def matching_close(tokens: list[Token] | tuple[Token, ...], i: int) -> int:
    """Index of the bracket closing ``tokens[i]``, or -1 if unbalanced."""
    stack = []
    for j in range(i, len(tokens)):
        t = tokens[j]
        if t.kind != "op":
            continue
        if t.text in _OPEN:
            stack.append(t.text)
        elif t.text in _CLOSE:
            if not stack or stack[-1] != _CLOSE[t.text]:
                return -1
            stack.pop()
            if not stack:
                return j
    return -1


def _split_args(inner: list[Token]) -> tuple[Arg, ...]:
    if not inner:
        return ()
    groups: list[list[Token]] = [[]]
    depth = 0
    for t in inner:
        if t.kind == "op" and t.text in _OPEN:
            depth += 1
        elif t.kind == "op" and t.text in _CLOSE:
            depth -= 1
        if depth == 0 and t.is_op(","):
            groups.append([])
        else:
            groups[-1].append(t)
    args = []
    for g in groups:
        if len(g) >= 2 and g[0].kind == "name" and g[1].is_op("="):
            args.append(Arg(g[0].text, tuple(g[2:])))
        else:
            args.append(Arg(None, tuple(g)))
    return tuple(args)


def find_calls(tokens: list[Token]) -> list[Call]:
    calls = []
    for i in range(len(tokens) - 1):
        t = tokens[i]
        if t.kind == "name" and tokens[i + 1].is_op("("):
            j = matching_close(tokens, i + 1)
            if j == -1:
                continue
            calls.append(Call(t.text, t, _split_args(tokens[i + 2 : j])))
    return calls
