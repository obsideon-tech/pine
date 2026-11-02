"""``pine-lint FILE... [--format text|json] [--strict]``

Exit status: 0 clean (info only); 1 if any error, or any warning with ``--strict``;
2 on usage, read or tokenize errors.
"""

from __future__ import annotations

import argparse
import json
import sys

from pine.lint import LexError, lint_file


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pine-lint", description="Static repainting check for Pine.")
    p.add_argument("files", nargs="+")
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.add_argument("--strict", action="store_true", help="exit 1 on warnings as well as errors")
    args = p.parse_args(argv)

    reports, status = [], 0
    for f in args.files:
        try:
            r = lint_file(f)
        except (OSError, UnicodeDecodeError, LexError) as exc:
            print(f"{f}: {exc}", file=sys.stderr)
            status = 2
            continue
        reports.append(r)
        if r.errors or (args.strict and r.warnings):
            status = max(status, 1)

    if args.format == "json":
        print(json.dumps([r.to_dict() for r in reports], indent=2))
    else:
        for r in reports:
            sys.stdout.write(r.to_text())
    return status


if __name__ == "__main__":
    sys.exit(main())
