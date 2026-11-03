"""Static repainting check for Pine Script (PINE_PLAN.md §6).

Rule-based and incomplete: a clean result means no *known* repainting pattern was found,
not that the script does not repaint.

    from pine.lint import lint_file
    report = lint_file("script.pine")
    print(report.to_text())
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pine.lint._calls import find_calls
from pine.lint._lexer import LexError, tokenize
from pine.lint._scope import build_scope
from pine.lint.rules import RULES, Context, Finding, parse_version

__all__ = ["DISCLAIMER", "Finding", "LexError", "LintReport", "RULES", "lint_file", "lint_source"]

DISCLAIMER = (
    "pine-lint is rule-based and incomplete: no findings means no known repainting pattern "
    "was found, not that the script does not repaint."
)

_SUPPRESS = re.compile(r"^\s*pine-lint:\s*ignore\[([^\]]*)\]")
_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class LintReport:
    path: str | None
    version: int | None
    findings: tuple[Finding, ...]
    suppressed: tuple[Finding, ...]

    @property
    def errors(self) -> int:
        return sum(f.severity == "error" for f in self.findings)

    @property
    def warnings(self) -> int:
        return sum(f.severity == "warning" for f in self.findings)

    def to_text(self) -> str:
        name = self.path or "<source>"
        out = [
            f"{name}:{f.line}:{f.col}: {f.severity} [{f.rule}] ({f.confidence}) {f.message}"
            for f in self.findings
        ]
        out.append(
            f"{name}: {self.errors} error(s), {self.warnings} warning(s), "
            f"{len(self.findings) - self.errors - self.warnings} info, "
            f"{len(self.suppressed)} suppressed"
        )
        out.append(DISCLAIMER)
        return "\n".join(out) + "\n"

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "version": self.version,
            "findings": [asdict(f) for f in self.findings],
            "suppressed": [asdict(f) for f in self.suppressed],
            "disclaimer": DISCLAIMER,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def lint_source(source: str, path: str | None = None) -> LintReport:
    tokens, comments = tokenize(source)
    version, _ = parse_version(comments)
    lines = source.splitlines()
    ctx = Context(tokens, comments, find_calls(tokens), lines, version, build_scope(tokens, lines))

    suppress: dict[int, set[str]] = {}
    for c in comments:
        m = _SUPPRESS.match(c.text)
        if m:
            suppress.setdefault(c.line, set()).update(
                r.strip() for r in m.group(1).split(",") if r.strip()
            )

    kept, dropped = [], []
    seen = set()
    for info in RULES.values():
        for f in info.fn(ctx):
            key = (f.rule, f.line, f.col)
            if key in seen:
                continue
            seen.add(key)
            (dropped if f.rule in suppress.get(f.line, ()) else kept).append(f)
    order = lambda f: (f.line, f.col, _SEVERITY_ORDER[f.severity], f.rule)  # noqa: E731
    return LintReport(
        path, version, tuple(sorted(kept, key=order)), tuple(sorted(dropped, key=order))
    )


def lint_file(path: str | Path) -> LintReport:
    p = Path(path)
    return lint_source(p.read_text(encoding="utf-8-sig"), str(p))
