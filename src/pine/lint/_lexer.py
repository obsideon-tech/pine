"""Tokenizer for Pine Script v5/v6 source, sufficient for rule matching (not a parser).

Comments are returned separately so suppressions, ``//@version`` and test expectations can be
read without affecting the token stream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

__all__ = ["Token", "Comment", "LexError", "tokenize"]

Kind = Literal["name", "number", "string", "color", "op"]

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_NUMBER = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_COLOR = re.compile(r"#[0-9A-Fa-f]{6}(?:[0-9A-Fa-f]{2})?\b")
_OPS = (":=", "=>", "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "%=")


class LexError(ValueError):
    pass


@dataclass(frozen=True)
class Token:
    kind: Kind
    text: str
    line: int  # 1-based
    col: int  # 1-based

    def is_op(self, text: str) -> bool:
        return self.kind == "op" and self.text == text

    def int_value(self) -> int | None:
        if self.kind == "number" and self.text.isdigit():
            return int(self.text)
        return None


@dataclass(frozen=True)
class Comment:
    text: str  # without the leading "//"
    line: int
    col: int


def tokenize(source: str) -> tuple[list[Token], list[Comment]]:
    tokens: list[Token] = []
    comments: list[Comment] = []
    i, line, line_start = 0, 1, 0
    n = len(source)
    while i < n:
        ch = source[i]
        col = i - line_start + 1
        if ch == "\n":
            i += 1
            line += 1
            line_start = i
            continue
        if ch in " \t\r\f﻿":
            i += 1
            continue
        if source.startswith("//", i):
            end = source.find("\n", i)
            end = n if end == -1 else end
            comments.append(Comment(source[i + 2 : end], line, col))
            i = end
            continue
        if ch in "\"'":
            j = i + 1
            while j < n and source[j] != ch:
                if source[j] == "\\":
                    j += 1
                elif source[j] == "\n":
                    raise LexError(f"line {line}: unterminated string")
                j += 1
            if j >= n:
                raise LexError(f"line {line}: unterminated string")
            tokens.append(Token("string", source[i : j + 1], line, col))
            i = j + 1
            continue
        if ch == "#":
            m = _COLOR.match(source, i)
            if m:
                tokens.append(Token("color", m.group(), line, col))
                i = m.end()
                continue
        m = _NAME.match(source, i)
        if m:
            tokens.append(Token("name", m.group(), line, col))
            i = m.end()
            continue
        m = _NUMBER.match(source, i)
        if m:
            tokens.append(Token("number", m.group(), line, col))
            i = m.end()
            continue
        op = next((o for o in _OPS if source.startswith(o, i)), ch)
        tokens.append(Token("op", op, line, col))
        i += len(op)
    return tokens, comments
