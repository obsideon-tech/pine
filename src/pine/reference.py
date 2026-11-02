"""Check a TradingView export of ``tools/reference_capture.pine`` (PINE_PLAN.md §10 A1, A2).

Two questions, answered separately:

1. **Parity** on gap-free columns: does each `pine.ta` port match TradingView in aligned mode?
2. **na policy** on the ``*_na`` columns: which of the `pine.ta.NA_POLICIES` hypotheses does
   each built-in follow? Every hypothesis is checked; the answer may be "none" (a fourth
   behaviour) or differ between built-ins, and both outcomes are reported as such.

Command line::

    python -m pine.reference EXPORT.csv --manifest EXPORT.yaml
    python -m pine.reference EXPORT.csv --timeframe 60 --tz America/New_York

Exit status: 0 if every gap-free column passes and every ``*_na`` column matches exactly one
policy; 1 otherwise; 2 on input errors.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pine import ta
from pine.bars import Bars
from pine.io import TradingViewExport, load_tradingview_csv
from pine.manifest import load_manifest
from pine.parity import ParityReport, PortSpec, Tolerance, check_export

__all__ = [
    "CONSTANTS",
    "GAPFREE_COLUMNS",
    "NA_COLUMNS",
    "PLOT_TITLES",
    "NaDecision",
    "check_capture",
    "decide_na_policy",
    "main",
]

# Must match tools/reference_capture.pine (enforced by tests/test_reference.py).
CONSTANTS = {"LEN": 14, "HL_LEN": 20, "FAST": 9, "SLOW": 21, "GAP_PERIOD": 50, "GAP_WIDTH": 3}
LEN, HL_LEN = CONSTANTS["LEN"], CONSTANTS["HL_LEN"]
_EXACT = Tolerance(atol=0.0, rtol=0.0)


def _col(export: TradingViewExport, name: str) -> np.ndarray:
    return np.asarray(export.columns[name])


def _src_na_expected(b: Bars) -> np.ndarray:
    i = np.arange(len(b))
    gap = i % CONSTANTS["GAP_PERIOD"] >= CONSTANTS["GAP_PERIOD"] - CONSTANTS["GAP_WIDTH"]
    return np.where(gap, np.nan, b.close)


def _gapfree_specs(export: TradingViewExport) -> dict[str, PortSpec]:
    """Port for each gap-free column. Crossover columns use the exported EMAs as inputs, so
    they test crossover logic alone; until §10 B2 lands they are compared as exact 0/1
    numbers (no near-tie excuse)."""

    def cross(fn):
        return lambda b: fn(_col(export, "ema_fast"), _col(export, "ema_slow")).astype(float)

    return {
        "sma": PortSpec(lambda b: ta.sma(b.close, LEN)),
        "ema": PortSpec(lambda b: ta.ema(b.close, LEN)),
        "rma": PortSpec(lambda b: ta.rma(b.close, LEN)),
        "rsi": PortSpec(lambda b: ta.rsi(b.close, LEN), kind="bounded"),
        "stdev": PortSpec(lambda b: ta.stdev(b.close, LEN)),
        "stdev_sample": PortSpec(lambda b: ta.stdev(b.close, LEN, biased=False)),
        "highest": PortSpec(lambda b: ta.highest(b.high, HL_LEN)),
        "lowest": PortSpec(lambda b: ta.lowest(b.low, HL_LEN)),
        "atr": PortSpec(lambda b: ta.atr(b.high, b.low, b.close, LEN)),
        "tr": PortSpec(lambda b: ta.tr(b.high, b.low, b.close)),
        "tr_handle_na": PortSpec(lambda b: ta.tr(b.high, b.low, b.close, handle_na=True)),
        "ema_fast": PortSpec(lambda b: ta.ema(b.close, CONSTANTS["FAST"])),
        "ema_slow": PortSpec(lambda b: ta.ema(b.close, CONSTANTS["SLOW"])),
        "crossover": PortSpec(cross(ta.crossover), kind="unbounded", tolerance=_EXACT),
        "crossunder": PortSpec(cross(ta.crossunder), kind="unbounded", tolerance=_EXACT),
        # Alignment check (§5.4): the gap pattern depends on bar_index, so this column only
        # matches if the export starts on the chart's first loaded bar.
        "src_na": PortSpec(_src_na_expected),
    }


GAPFREE_COLUMNS = (
    "sma", "ema", "rma", "rsi", "stdev", "stdev_sample", "highest", "lowest", "atr", "tr",
    "tr_handle_na", "ema_fast", "ema_slow", "crossover", "crossunder", "src_na",
)  # fmt: skip

NA_PORTS: dict[str, tuple[Callable, int, str]] = {
    "sma_na": (ta.sma, LEN, "price"),
    "ema_na": (ta.ema, LEN, "price"),
    "rma_na": (ta.rma, LEN, "price"),
    "rsi_na": (ta.rsi, LEN, "bounded"),
    "stdev_na": (ta.stdev, LEN, "price"),
    "highest_na": (ta.highest, HL_LEN, "price"),
    "lowest_na": (ta.lowest, HL_LEN, "price"),
}
NA_COLUMNS = tuple(NA_PORTS)
PLOT_TITLES = (*GAPFREE_COLUMNS, *NA_COLUMNS)


def check_capture(export: TradingViewExport) -> ParityReport:
    """Aligned-mode parity on the gap-free columns (and the ``src_na`` alignment check)."""
    return check_export(export, _gapfree_specs(export))


@dataclass(frozen=True)
class NaDecision:
    column: str
    matching: tuple[str, ...]
    """Policies under which the port matches this column; ideally exactly one."""
    reports: dict[str, ParityReport]

    @property
    def decided(self) -> str | None:
        return self.matching[0] if len(self.matching) == 1 else None


def decide_na_policy(export: TradingViewExport) -> list[NaDecision]:
    """Run each ``*_na`` port under every policy on the exported ``src_na`` column."""
    src = _col(export, "src_na")
    out = []
    for col, (fn, n, kind) in NA_PORTS.items():
        reports = {}
        for policy in ta.NA_POLICIES:
            spec = PortSpec(lambda b, fn=fn, n=n, p=policy: fn(src, n, na_policy=p), kind=kind)
            reports[policy] = check_export(export, {col: spec})
        matching = tuple(p for p, r in reports.items() if r.passed)
        out.append(NaDecision(col, matching, reports))
    return out


def _summary(report: ParityReport, decisions: list[NaDecision]) -> str:
    lines = ["## Gap-free columns (aligned mode)", "", report.to_markdown(), "## na policy", ""]
    lines += ["| column | matching policies | first divergence per policy |", "|---|---|---|"]
    for d in decisions:
        fd = ", ".join(
            f"{p}: {r.columns[0].first_divergence} ({r.columns[0].divergence_type})"
            for p, r in d.reports.items()
            if not r.passed
        )
        lines.append(f"| {d.column} | {', '.join(d.matching) or '**none**'} | {fd or '—'} |")
    decided = {d.decided for d in decisions}
    if None in decided:
        verdict = "**undecided**: some column matches no policy or several (see table)"
    elif len(decided) > 1:
        verdict = f"**built-ins differ**: {sorted(decided)}"
    else:
        verdict = f"**all built-ins follow `{decided.pop()}`**"
    lines += ["", f"Verdict: {verdict}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m pine.reference", description="Check a reference-capture export."
    )
    ap.add_argument("export")
    ap.add_argument("--manifest", help="fixture manifest (supplies timeframe and timezone)")
    ap.add_argument("--timeframe")
    ap.add_argument("--tz", default=None)
    args = ap.parse_args(argv)
    try:
        if args.manifest:
            m = load_manifest(args.manifest)
            timeframe, tz = m.timeframe, m.timezone
        else:
            if not args.timeframe:
                ap.error("--timeframe is required without --manifest")
            timeframe, tz = args.timeframe, args.tz or "UTC"
        exp = load_tradingview_csv(args.export, timeframe=timeframe, session_tz=tz)
        missing = [c for c in PLOT_TITLES if c not in exp.columns]
        if missing:
            raise KeyError(f"export lacks capture columns {missing}; is this the capture script?")
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report, decisions = check_capture(exp), decide_na_policy(exp)
    sys.stdout.write(_summary(report, decisions))
    ok = report.passed and all(d.decided for d in decisions)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
