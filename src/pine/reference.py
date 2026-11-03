"""Check a TradingView export of ``tools/reference_capture.pine`` (PINE_PLAN.md §10 A1, A2).

Two questions, answered separately:

1. **Parity** on gap-free columns: does each `pine.ta` port match TradingView in aligned mode?
2. **na policy** on the ``*_na`` columns: which of the `pine.ta.NA_POLICIES` hypotheses does
   each built-in follow? Every hypothesis is checked; the answer may be "none" (a fourth
   behaviour) or differ between built-ins, and both outcomes are reported as such.

Command line::

    python -m pine.reference aapl_1d.csv --manifest aapl_1d.yaml
    python -m pine.reference aapl_1d.csv --timeframe 1D --tz America/New_York

Usually run through ``python -m pine.capture check`` (docs/CAPTURE.md).

Exit status: 0 if every gap-free column passes and every ``*_na`` column matches exactly one
policy; 1 otherwise; 2 on input errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from pine import ta
from pine._cli import InputError, load_export
from pine.bars import Bars
from pine.convergence import rsi_convergence_bars
from pine.io import TradingViewExport
from pine.parity import (
    ParityReport,
    PortSpec,
    check_export,
    crossover_excuse,
    tolerance_for,
)
from pine.sensitivity import LIPSCHITZ, input_rounding, rsi_input_error, stdev_lipschitz
from pine.truncation import bar_offset, decay_adjusted, sync_point

__all__ = [
    "CONSTANTS",
    "GAPFREE_COLUMNS",
    "NA_COLUMNS",
    "PLOT_TITLES",
    "NaDecision",
    "check_capture",
    "ohlc_rounding",
    "decide_na_policy",
    "main",
]

# Must match tools/reference_capture.pine (enforced by tests/test_reference.py).
CONSTANTS = {"LEN": 14, "HL_LEN": 20, "FAST": 9, "SLOW": 21, "GAP_PERIOD": 50, "GAP_WIDTH": 3}
LEN, HL_LEN = CONSTANTS["LEN"], CONSTANTS["HL_LEN"]


def _col(export: TradingViewExport, name: str) -> np.ndarray:
    return np.asarray(export.columns[name])


def _src_na_expected(b: Bars, k: int = 0) -> np.ndarray:
    """``src_na`` given the true ``bar_index`` of the first row (``k``)."""
    i = k + np.arange(len(b))
    gap = i % CONSTANTS["GAP_PERIOD"] >= CONSTANTS["GAP_PERIOD"] - CONSTANTS["GAP_WIDTH"]
    return np.where(gap, np.nan, b.close)


def infer_gap_phase(export: TradingViewExport) -> int:
    """``bar_index mod GAP_PERIOD`` of the first row, inferred from the exported ``src_na``
    gap pattern (used when the export has no ``bar_index`` column)."""
    na = np.isnan(_col(export, "src_na"))
    period, width = CONSTANTS["GAP_PERIOD"], CONSTANTS["GAP_WIDTH"]
    for k in range(period):
        i = k + np.arange(na.size)
        if np.array_equal(na, i % period >= period - width):
            return k
    raise ValueError("src_na gap pattern does not match any phase; is this the capture script?")


def _gapfree_specs(export: TradingViewExport, k: int = 0) -> dict[str, PortSpec]:
    """Port for each gap-free column. Crossover columns use the exported EMAs as inputs, so
    they test crossover logic alone, and are compared as discrete columns with the two-bar
    near-tie excuse (§5.5). ``alpha`` is set on seeded recursive filters for ``mde_seed``."""

    def cross(fn):
        return lambda b: fn(_col(export, "ema_fast"), _col(export, "ema_slow"))

    def excuse(b):
        tie_tol = tolerance_for("price", export.observed_decimals.get("ema_fast"))
        return crossover_excuse(_col(export, "ema_fast"), _col(export, "ema_slow"), tie_tol)

    L = LIPSCHITZ
    return {
        "sma": PortSpec(lambda b: ta.sma(b.close, LEN), input_error=L["sma"]),
        "ema": PortSpec(lambda b: ta.ema(b.close, LEN), input_error=L["ema"], alpha=2 / (LEN + 1)),
        "rma": PortSpec(lambda b: ta.rma(b.close, LEN), input_error=L["rma"], alpha=1 / LEN),
        "rsi": PortSpec(
            lambda b: ta.rsi(b.close, LEN),
            kind="bounded",
            input_error=lambda b, d: rsi_input_error(b.close, LEN, d),
        ),
        "stdev": PortSpec(lambda b: ta.stdev(b.close, LEN), input_error=L["stdev"]),
        "stdev_sample": PortSpec(
            lambda b: ta.stdev(b.close, LEN, biased=False),
            input_error=stdev_lipschitz(LEN, biased=False),
        ),
        "highest": PortSpec(lambda b: ta.highest(b.high, HL_LEN), input_error=L["highest"]),
        "lowest": PortSpec(lambda b: ta.lowest(b.low, HL_LEN), input_error=L["lowest"]),
        "atr": PortSpec(
            lambda b: ta.atr(b.high, b.low, b.close, LEN), input_error=L["atr"], alpha=1 / LEN
        ),
        "tr": PortSpec(lambda b: ta.tr(b.high, b.low, b.close), input_error=L["tr"]),
        "tr_handle_na": PortSpec(
            lambda b: ta.tr(b.high, b.low, b.close, handle_na=True), input_error=L["tr"]
        ),
        "ema_fast": PortSpec(
            lambda b: ta.ema(b.close, CONSTANTS["FAST"]),
            input_error=1.0,
            alpha=2 / (CONSTANTS["FAST"] + 1),
        ),
        "ema_slow": PortSpec(
            lambda b: ta.ema(b.close, CONSTANTS["SLOW"]),
            input_error=1.0,
            alpha=2 / (CONSTANTS["SLOW"] + 1),
        ),
        # Crossovers use the exported EMAs, not the rounded prices: no input allowance.
        "crossover": PortSpec(cross(ta.crossover), kind="discrete", excuse=excuse),
        "crossunder": PortSpec(cross(ta.crossunder), kind="discrete", excuse=excuse),
        # Alignment check (§5.4): the gap pattern depends on bar_index, so this column only
        # matches if the export starts on the chart's first loaded bar. It is the exported
        # close itself on both sides, so rounding cancels.
        "src_na": PortSpec(lambda b: _src_na_expected(b, k), input_error=0.0),
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


def history_offset(export: TradingViewExport) -> tuple[int, bool]:
    """``(k, exact)``: the first row's ``bar_index`` and whether it is known exactly (from a
    ``bar_index`` column) or only modulo ``GAP_PERIOD`` (from the ``src_na`` gap pattern)."""
    k = bar_offset(export.columns)
    if k is not None:
        return k, True
    return infer_gap_phase(export), False


# In a truncated export, a column must still compare at least this many bars to pass.
MIN_COMPARED = 100

# Python warm-up rows for windowed columns (exact once the window is full).
_WARMUP = {
    "sma": LEN - 1,
    "stdev": LEN - 1,
    "stdev_sample": LEN - 1,
    "highest": HL_LEN - 1,
    "lowest": HL_LEN - 1,
    "tr": 1,
    "tr_handle_na": 1,
}
# Seeded recursive filters: (alpha, Python seed bar).
_DECAY = {
    "ema": (2 / (LEN + 1), LEN - 1),
    "rma": (1 / LEN, LEN - 1),
    "atr": (1 / LEN, LEN - 1),
    "ema_fast": (2 / (CONSTANTS["FAST"] + 1), CONSTANTS["FAST"] - 1),
    "ema_slow": (2 / (CONSTANTS["SLOW"] + 1), CONSTANTS["SLOW"] - 1),
}


def _truncated_specs(export: TradingViewExport, k: int) -> dict[str, PortSpec]:
    """Specs for an export that starts after the chart's first bar (`pine.truncation`)."""
    base = _gapfree_specs(export, k)
    out = {}
    for name, spec in base.items():
        if name in _WARMUP:
            out[name] = replace(
                spec, mode="convergence", exclude=_WARMUP[name], min_compared=MIN_COMPARED
            )
        elif name in _DECAY:
            alpha, w = _DECAY[name]
            pine = _col(export, name)
            out[name] = replace(
                spec,
                fn=lambda b, f=spec.fn, a=alpha, w=w, pine=pine: decay_adjusted(f(b), pine, a, w),
                mode="convergence",
                exclude=w,
                alpha=None,
                min_compared=MIN_COMPARED,
            )
        elif name == "rsi":
            tol = tolerance_for("bounded", export.observed_decimals.get("rsi")).atol
            close = np.asarray(export.bars.close)
            m = 2 * float(np.nanmax(np.abs(np.diff(close))))
            n = rsi_convergence_bars(close, LEN, float(tol), max_change=m)
            out[name] = replace(
                spec, mode="convergence", exclude=min(n, len(close)), min_compared=MIN_COMPARED
            )
        elif spec.kind == "discrete":
            ex = spec.excuse

            def first_row_excuse(b, ex=ex):
                e = np.zeros(len(b), bool) if ex is None else np.asarray(ex(b), bool).copy()
                e[0] = True  # crossover at row 0 needs the unseen previous bar
                return e

            out[name] = replace(spec, excuse=first_row_excuse)
        else:
            out[name] = spec
    return out


def check_capture(export: TradingViewExport, input_rounding: float | None = None) -> ParityReport:
    """Aligned-mode parity on the gap-free columns (and the ``src_na`` alignment check).

    If the export starts after the chart's first bar (``bar_index`` > 0, or unknown), the
    history-robust comparisons of `pine.truncation` are used instead and every column says so.
    ``input_rounding``: see `pine.parity.check_export`; `ohlc_rounding` estimates it."""
    k, exact = history_offset(export)
    if exact and k == 0:
        return check_export(export, _gapfree_specs(export, 0), input_rounding=input_rounding)
    rep = check_export(export, _truncated_specs(export, k), input_rounding=input_rounding)
    where = f"bar_index {k}" if exact else f"bar_index ≡ {k} (mod {CONSTANTS['GAP_PERIOD']})"
    note = (
        f"export starts at {where}, not the chart's first bar: warm-up rows excluded, "
        "EMA/RMA/ATR checked by exact residual decay, RSI by convergence bound; seeding not "
        "certified (pine.truncation)"
    )
    cols = tuple(replace(c, notes=(*c.notes, note)) for c in rep.columns)
    return replace(rep, columns=cols)


def ohlc_rounding(export: TradingViewExport) -> float:
    """``δ`` implied by the coarsest printed precision among open/high/low/close."""
    d = min(export.observed_decimals[k] for k in ("open", "high", "low", "close"))
    return input_rounding(d)


@dataclass(frozen=True)
class NaDecision:
    column: str
    matching: tuple[str, ...]
    """Policies under which the port matches this column; ideally exactly one."""
    reports: dict[str, ParityReport]

    @property
    def decided(self) -> str | None:
        return self.matching[0] if len(self.matching) == 1 else None


def decide_na_policy(
    export: TradingViewExport, input_rounding: float | None = None
) -> list[NaDecision]:
    """Run each ``*_na`` port under every policy on the exported ``src_na`` column.

    ``input_rounding`` as in `check_capture` (``src_na`` is the exported close)."""
    src = _col(export, "src_na")
    k, exact = history_offset(export)
    truncated = not (exact and k == 0)
    out = []
    for col, (fn, n, kind) in NA_PORTS.items():
        reports = {}
        for policy in ta.NA_POLICIES:
            exclude = 0
            if truncated:
                py = fn(src, n, na_policy=policy)
                valid = np.flatnonzero(~np.isnan(py))
                w = int(valid[0]) if valid.size else len(py)
                if col in ("ema_na", "rma_na", "rsi_na"):
                    # Recursive state: under `propagate` it re-seeds after every gap, under the
                    # skip policies it is carried. Compare from the observed sync row; a wrong
                    # hypothesis diverges again at each gap and cannot stay synchronised past
                    # the first half of the export.
                    pine = _col(export, col)
                    tol = tolerance_for(kind, export.observed_decimals.get(col))
                    r = sync_point(py, pine, tol.bound(np.nan_to_num(pine)), start=w)
                    w = r if r <= len(py) // 2 else w
                exclude = min(w, len(py))
            if col == "rsi_na":
                err = lambda b, d, n=n, p=policy: rsi_input_error(src, n, d, p)  # noqa: E731
            else:
                err = 1.0  # sma/ema/rma/highest/lowest/population stdev: L = 1
            spec = PortSpec(
                lambda b, fn=fn, n=n, p=policy: fn(src, n, na_policy=p),
                kind=kind,
                input_error=err,
                mode="convergence" if truncated else "aligned",
                exclude=exclude,
                min_compared=MIN_COMPARED if truncated else 0,
            )
            reports[policy] = check_export(export, {col: spec}, input_rounding=input_rounding)
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
    ap.add_argument(
        "--inputs-rounded",
        action="store_true",
        help="OHLC in the export are rounded beyond the instrument's tick (e.g. dividend-"
        "adjusted); add the input-rounding allowance from the printed OHLC precision",
    )
    ap.add_argument("--format", choices=("md", "json"), default="md")
    args = ap.parse_args(argv)
    try:
        exp, _ = load_export(args.export, args.manifest, args.timeframe, args.tz)
        missing = [c for c in PLOT_TITLES if c not in exp.columns]
        if missing:
            raise KeyError(f"export lacks capture columns {missing}; is this the capture script?")
    except (InputError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    delta = ohlc_rounding(exp) if args.inputs_rounded else None
    report = check_capture(exp, input_rounding=delta)
    decisions = decide_na_policy(exp, input_rounding=delta)
    if args.format == "json":
        payload = {
            "parity": report.to_dict(),
            "na_policy": [
                {"column": d.column, "matching": list(d.matching), "decided": d.decided}
                for d in decisions
            ],
        }
        print(json.dumps(payload, indent=2, allow_nan=False))
    else:
        sys.stdout.write(_summary(report, decisions))
    ok = report.passed and all(d.decided for d in decisions)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
