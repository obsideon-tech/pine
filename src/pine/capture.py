"""One entry point for TradingView capture exports (docs/CAPTURE.md).

Put the capture scripts (``tools/reference_capture.pine``, ``tools/capture_semantics.pine``,
optionally ``tools/capture_builtins.pine`` and ``tools/capture_strategy.pine``) on one chart,
export chart data once, then::

    uv run python -m pine.capture manifest ~/Downloads/NASDAQ_AAPL_1D.csv \\
        --symbol NASDAQ:AAPL --timeframe 1D --timezone America/New_York
    uv run python -m pine.capture check ~/Downloads/NASDAQ_AAPL_1D.csv --out report.md

``manifest`` writes ``<csv>.yaml`` next to the export (refuses to overwrite without
``--force``). ``check`` runs every check the export supports — built-in parity and the
``na``-handling decision (`pine.reference`), Pine semantics (`pine.semantics`), the batch-F
built-ins (`pine.builtin_capture`) and the strategy engine (`pine.strategy_capture`) — and
writes one combined report. Exit status: 0 all checks that ran passed; 1 something differs; 2 input
problem (missing file, no capture columns, invalid manifest).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import yaml

from pine import builtin_capture, reference, semantics, strategy_capture
from pine._cli import InputError, load_export
from pine.io import load_tradingview_csv
from pine.manifest import ManifestError, validate_manifest

__all__ = ["main", "write_manifest", "run_checks"]


def _present(exp, columns) -> bool:
    return all(c in exp.columns for c in columns)


def write_manifest(
    csv: str | Path,
    *,
    symbol: str,
    timeframe: str,
    timezone: str,
    adjustment: str = "splits",
    session: str = "regular",
    force: bool = False,
    mintick: float | None = None,
) -> Path:
    """Write ``<csv>.yaml`` describing a capture export; returns its path."""
    csv = Path(csv)
    if not csv.exists():
        raise InputError(
            f"export not found: {csv}\nExport it from TradingView first (docs/CAPTURE.md)."
        )
    out = csv.with_suffix(".yaml")
    if out.exists() and not force:
        raise InputError(f"{out} already exists; pass --force to overwrite")
    exp = load_tradingview_csv(csv, timeframe="1")  # timeframe only needed for Bars validation
    scripts, inputs, cols = [], {}, []
    if _present(exp, reference.PLOT_TITLES):
        scripts.append("pine reference capture")
        inputs |= {f"reference.{k}": v for k, v in reference.CONSTANTS.items()}
        cols += reference.PLOT_TITLES
    if _present(exp, semantics.COLUMNS):
        scripts.append("pine semantics capture")
        inputs |= {f"semantics.{k}": v for k, v in semantics.CONSTANTS.items()}
        cols += semantics.COLUMNS
    if _present(exp, builtin_capture.COLUMNS):
        scripts.append("pine builtin capture")
        inputs |= {f"builtins.{k}": v for k, v in builtin_capture.CONSTANTS.items()}
        cols += builtin_capture.COLUMNS
    if _present(exp, strategy_capture.COLUMNS):
        scripts.append("pine strategy capture")
        inputs |= {f"strategy.{k}": v for k, v in strategy_capture.CONSTANTS.items()}
        cols += strategy_capture.COLUMNS
    if not scripts:
        raise InputError(
            f"{csv} has none of the capture columns. Are both capture scripts on the chart, and "
            f"was the export made with them visible? Columns found: {sorted(exp.columns)}"
        )
    data = {
        "source": "tradingview_export",
        "symbol_class": "public",
        "symbol": symbol,
        "timeframe": timeframe,
        "session": session,
        "timezone": timezone,
        "adjustment": adjustment,
        "chart_type": "candles",
        "pine_version": 6,
        "export_date": dt.date.today(),
        "timestamp_convention": "bar_open",
        "displayed_precision": max(exp.observed_decimals.get(c, 0) for c in cols),
        "indicator": {"name": " + ".join(scripts), "inputs": inputs},
        "notes": "Written by `python -m pine.capture manifest`; edit if the chart settings differ.",
        **({"mintick": mintick} if mintick is not None else {}),
    }
    try:
        validate_manifest(dict(data), str(out))
    except ManifestError as exc:
        raise InputError(str(exc)) from exc
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return out


def run_checks(exp, inputs_rounded: bool = False, mintick: float | None = None) -> dict:
    """Run every check the export supports. Returns a dict with per-check results."""
    result: dict = {
        "reference": None, "semantics": None, "builtins": None, "strategy": None,
        "skipped": [], "errors": [],
    }  # fmt: skip
    if _present(exp, reference.PLOT_TITLES):
        delta = reference.ohlc_rounding(exp) if inputs_rounded else None
        try:
            rep = reference.check_capture(exp, input_rounding=delta)
            na = reference.decide_na_policy(exp, input_rounding=delta)
            result["reference"] = (rep, na)
        except ValueError as exc:
            result["errors"].append(f"reference: {exc}")
    else:
        result["skipped"].append("reference (tools/reference_capture.pine not in the export)")
    if _present(exp, semantics.COLUMNS):
        try:
            result["semantics"] = semantics.decide(exp)
        except ValueError as exc:
            result["errors"].append(f"semantics: {exc}")
    else:
        result["skipped"].append("semantics (tools/capture_semantics.pine not in the export)")
    if _present(exp, builtin_capture.COLUMNS):
        try:
            result["builtins"] = builtin_capture.decide(exp)
        except ValueError as exc:
            result["errors"].append(f"builtins: {exc}")
    else:
        result["skipped"].append("builtins (tools/capture_builtins.pine not in the export)")
    if _present(exp, strategy_capture.COLUMNS):
        try:
            result["strategy"] = strategy_capture.decide(exp, mintick=mintick)
        except ValueError as exc:
            result["errors"].append(f"strategy: {exc}")
    else:
        result["skipped"].append("strategy (tools/capture_strategy.pine not in the export)")
    return result


def _passed(result: dict) -> bool:
    ok = not result.get("errors")
    if result["reference"]:
        rep, na = result["reference"]
        ok &= rep.passed and all(d.decided for d in na)
    if result["semantics"]:
        ok &= all(d.ok for d in result["semantics"])
    if result["builtins"]:
        ok &= all(d.ok for d in result["builtins"])
    if result["strategy"]:
        ok &= all(d.ok for d in result["strategy"])
    return ok


def _markdown(csv: Path, manifest, result: dict) -> str:
    out = [f"# Capture check: `{csv.name}`", ""]
    if manifest is not None:
        out += [
            f"Symbol `{manifest.symbol}`, timeframe `{manifest.timeframe}`, time zone "
            f"`{manifest.timezone}`, adjustment `{manifest.adjustment}`, session "
            f"`{manifest.session}`.",
            "",
        ]
    out += [f"**Overall: {'PASS' if _passed(result) else 'DIFFERENCES FOUND'}**", ""]
    if result["reference"]:
        rep, na = result["reference"]
        out.append(reference._summary(rep, na))
    if result["semantics"]:
        out.append(semantics._summary(result["semantics"]))
    if result["builtins"]:
        out.append(builtin_capture.summary(result["builtins"]))
    if result["strategy"]:
        out.append(strategy_capture.summary(result["strategy"]))
    if result.get("errors"):
        out += [
            "## Could not run",
            "",
            *(f"- {e}" for e in result["errors"]),
            "",
            "A check above could not run on this export; the others did. See docs/CAPTURE.md "
            "troubleshooting.",
            "",
        ]
    if result["skipped"]:
        out += ["## Skipped", "", *(f"- {s}" for s in result["skipped"]), ""]
    return "\n".join(out)


def _json(csv: Path, result: dict) -> str:
    payload: dict = {
        "export": csv.name,
        "passed": _passed(result),
        "skipped": result["skipped"],
        "errors": result.get("errors", []),
    }
    if result["reference"]:
        rep, na = result["reference"]
        payload["reference"] = {
            "parity": rep.to_dict(),
            "na_policy": [
                {"column": d.column, "matching": list(d.matching), "decided": d.decided} for d in na
            ],
        }
    if result["semantics"]:
        payload["semantics"] = [
            {"group": d.group, "expected": d.expected, "matching": list(d.matching), "ok": d.ok}
            for d in result["semantics"]
        ]
    if result["builtins"]:
        payload["builtins"] = [
            {"group": d.group, "expected": d.expected, "matching": list(d.matching), "ok": d.ok}
            for d in result["builtins"]
        ]
    if result["strategy"]:
        payload["strategy"] = [
            {"rule": d.group, "expected": d.expected, "matching": list(d.matching), "ok": d.ok}
            for d in result["strategy"]
        ]
    return json.dumps(payload, indent=2, allow_nan=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pine.capture", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest", help="write <csv>.yaml describing the export")
    m.add_argument("csv")
    m.add_argument("--symbol", required=True, help="as shown on the chart, e.g. NASDAQ:AAPL")
    m.add_argument("--timeframe", required=True, help="Pine spelling: 1, 5, 60, 240, 1D, 1W, 1M")
    m.add_argument(
        "--timezone", required=True, help="exchange time zone, IANA name, e.g. America/New_York"
    )
    m.add_argument(
        "--adjustment",
        default="splits",
        choices=("none", "splits", "splits_dividends"),
        help="default 'splits': TradingView's usual stock default (unverified); use "
        "splits_dividends if the chart's 'Adjust data for dividends' (ADJ) is on",
    )
    m.add_argument("--session", default="regular", choices=("regular", "extended"))
    m.add_argument(
        "--mintick", type=float, help="symbol tick size (syminfo.mintick), for the strategy check"
    )
    m.add_argument("--force", action="store_true", help="overwrite an existing manifest")

    c = sub.add_parser("check", help="run every check the export supports")
    c.add_argument("csv")
    c.add_argument("--manifest", help="default: <csv>.yaml next to the export")
    c.add_argument("--timeframe", help="only if there is no manifest")
    c.add_argument("--tz", help="only if there is no manifest")
    c.add_argument("--inputs-rounded", action="store_true", help="see §5.1 of PINE_PLAN.md")
    c.add_argument(
        "--mintick",
        type=float,
        help="symbol tick size for the strategy check (default: manifest, then the export's "
        "strat_mintick column)",
    )
    c.add_argument("--format", choices=("md", "json"), default="md")
    c.add_argument("--out", help="write the report to this file as well as stdout")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "manifest":
            path = write_manifest(
                args.csv,
                symbol=args.symbol,
                timeframe=args.timeframe,
                timezone=args.timezone,
                adjustment=args.adjustment,
                session=args.session,
                force=args.force,
                mintick=args.mintick,
            )
            print(f"wrote {path}\nnext: uv run python -m pine.capture check {args.csv}")
            return 0

        csv = Path(args.csv)
        manifest = args.manifest
        if manifest is None and not args.timeframe:
            manifest = csv.with_suffix(".yaml")
        exp, man = load_export(csv, manifest, args.timeframe, args.tz)
        mintick = args.mintick
        if mintick is None and man is not None:
            mintick = man.mintick
        result = run_checks(exp, inputs_rounded=args.inputs_rounded, mintick=mintick)
        if not result["errors"] and all(
            result[k] is None for k in ("reference", "semantics", "builtins", "strategy")
        ):
            raise InputError(
                f"{csv} contains neither capture script's columns; see docs/CAPTURE.md step 2"
            )
    except (InputError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    text = _json(csv, result) if args.format == "json" else _markdown(csv, man, result)
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0 if _passed(result) else 1


if __name__ == "__main__":
    sys.exit(main())
