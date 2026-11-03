"""Check an export of ``tools/capture_strategy.pine`` against `pine.strategy` (§13).

Each probe group is one broker rule with competing hypotheses (`pine.strategy.StrategyConfig`
fields). For each group the emulator runs with the expected configuration except that rule,
and the six plotted strategy columns are compared bar by bar. ``baseline`` is the fully
expected configuration.

Exports that start mid-history: the true bar number comes from ``strat_bar_index``. Every
cycle ends flat (phase 39), so from the first phase-39 row Python and TradingView hold the
same position; cumulative columns (net profit, equity, closed trades) are anchored there.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from pine.bars import Bars
from pine.io import TradingViewExport
from pine.parity import ParityReport, PortSpec, check_export
from pine.runtime import Context, run
from pine.strategy import StrategyConfig
from pine.truncation import bar_offset

__all__ = ["COLUMNS", "CONSTANTS", "GROUPS", "StrategyDecision", "decide", "summary"]

# Must match tools/capture_strategy.pine (tests/test_strategy.py).
CONSTANTS: dict[str, float] = {"CYCLE": 40, "BRACKET": 0.01}
COLUMNS: tuple[str, ...] = (
    "strat_bar_index", "strat_position_size", "strat_avg_price", "strat_netprofit",
    "strat_openprofit", "strat_equity", "strat_closedtrades",
)  # fmt: skip
_CUMULATIVE = ("strat_netprofit", "strat_equity", "strat_closedtrades")

EXPECTED = StrategyConfig()
GROUPS: dict[str, dict[str, StrategyConfig]] = {
    "baseline": {"expected": EXPECTED},
    "fill_timing": {
        "next_open": EXPECTED,
        "same_close": replace(EXPECTED, fill_timing="same_close"),
    },
    "path": {
        "nearest_extreme_first": EXPECTED,
        "high_first": replace(EXPECTED, path="high_first"),
    },
    "gap_fill": {"open": EXPECTED, "order_price": replace(EXPECTED, gap_fill="order_price")},
    "exit_same_bar": {"yes": EXPECTED, "no": replace(EXPECTED, exit_same_bar=False)},
    # Tick rounding: only meaningful when the symbol's mintick is known (strat_mintick column
    # or --mintick); otherwise these groups are skipped.
    "price_rounding": {"tick": EXPECTED, "none": replace(EXPECTED, price_rounding="none")},
    "order_rounding": {
        "conservative": EXPECTED,
        "nearest": replace(EXPECTED, order_rounding="nearest"),
    },
    "path_basis": {"raw": EXPECTED, "rounded": replace(EXPECTED, path_basis="rounded")},
}
_NEEDS_MINTICK = ("price_rounding", "order_rounding", "path_basis")
OPTIONAL_COLUMNS = ("strat_mintick",)
EXPECTED_HYP = {
    "baseline": "expected",
    "fill_timing": "next_open",
    "path": "nearest_extreme_first",
    "gap_fill": "open",
    "exit_same_bar": "yes",
    "price_rounding": "tick",
    "order_rounding": "conservative",
    "path_basis": "raw",
}


def strategy_capture(ctx: Context) -> None:
    """``tools/capture_strategy.pine`` on `pine.runtime` with ``ctx.strategy``."""
    s = ctx.strategy
    ph = int(ctx.bar_index.value) % int(CONSTANTS["CYCLE"])
    close = ctx.close.value
    if ph == 0:
        s.entry("L", "long")
    if ph == 8:
        s.close("L")
    if ph == 12:
        s.entry("S", "short")
    if ph == 20:
        s.entry("L", "long")
    if ph == 26:
        s.close_all()
    if ph == 30:
        b = CONSTANTS["BRACKET"]
        s.entry("B", "long")
        s.exit("BX", "B", stop=close * (1 - b), limit=close * (1 + b))
    if ph == 38:
        s.close("B")
    ctx.plot("strat_bar_index", ctx.bar_index.value)
    ctx.plot("strat_position_size", s.position_size)
    ctx.plot("strat_avg_price", s.position_avg_price)
    ctx.plot("strat_netprofit", s.netprofit)
    ctx.plot("strat_openprofit", s.openprofit)
    ctx.plot("strat_equity", s.equity)
    ctx.plot("strat_closedtrades", s.closedtrades)


@dataclass(frozen=True)
class StrategyDecision:
    group: str
    expected: str
    matching: tuple[str, ...]
    reports: dict[str, ParityReport]
    sync_row: int
    dropped_live_bar: bool = False
    mintick: float | None = None

    @property
    def undecided(self) -> bool:
        return len(self.matching) > 1 and self.expected in self.matching

    @property
    def ok(self) -> bool:
        return self.matching == (self.expected,) or self.undecided


def _columns(export: TradingViewExport, cfg: StrategyConfig, k: int) -> dict[str, np.ndarray]:
    res = run(strategy_capture, export.bars, strategy=cfg, bar_offset=k)
    return {c: np.asarray(res[c]) for c in COLUMNS}


def _drop_live_bar(export: TradingViewExport) -> tuple[TradingViewExport, bool]:
    """Strategies (``calc_on_every_tick = false``) run on the live bar only when it closes, so
    an export taken mid-bar has every strategy column ``na`` on its last row while indicators
    have values. Drop that row; any other ``na`` is left for `bar_offset` to report."""
    if len(export.bars) < 2 or not all(np.isnan(export.columns[c][-1]) for c in COLUMNS):
        return export, False
    b = export.bars
    sl = slice(0, -1)
    bars = Bars(b.ts[sl], b.open[sl], b.high[sl], b.low[sl], b.close[sl], b.volume[sl],
                timeframe=b.timeframe, session_tz=b.session_tz, symbol=b.symbol)  # fmt: skip
    cols = {c: np.asarray(v)[sl] for c, v in export.columns.items()}
    return replace(export, bars=bars, columns=cols), True


def resolve_mintick(export: TradingViewExport, mintick: float | None = None) -> float | None:
    """The symbol's tick size: explicit argument, else the exported ``strat_mintick`` column."""
    if mintick is not None:
        return float(mintick)
    col = export.columns.get("strat_mintick")
    if col is None:
        return None
    vals = np.asarray(col, dtype=float)
    vals = vals[~np.isnan(vals)]
    return float(vals[0]) if vals.size else None


def decide(export: TradingViewExport, mintick: float | None = None) -> list[StrategyDecision]:
    """Decide each broker rule. ``mintick``: the symbol's tick size if the export lacks the
    ``strat_mintick`` column; without it the tick-rounding groups are skipped and the other
    groups run with rounding off (they will then fail on symbols whose prices have more
    decimals than their tick, as TradingView rounds)."""
    export, dropped = _drop_live_bar(export)
    mintick = resolve_mintick(export, mintick)
    k = bar_offset(export.columns, "strat_bar_index") or 0
    n = len(export.bars)
    cycle = int(CONSTANTS["CYCLE"])
    sync = 0 if k == 0 else (cycle - 1 - k) % cycle  # first row at phase 39 (flat)
    out = []
    for group, hyps in GROUPS.items():
        if mintick is None and group in _NEEDS_MINTICK:
            continue
        reports = {}
        for name, cfg in hyps.items():
            cols = _columns(export, replace(cfg, mintick=mintick), k)
            if sync:
                for c in _CUMULATIVE:
                    cols[c] = cols[c] - cols[c][sync] + export.columns[c][sync]
            specs = {
                c: PortSpec(
                    lambda b, v=v: v,
                    kind="price",
                    mode="convergence" if sync else "aligned",
                    exclude=min(sync, n),
                    min_compared=100 if sync else 0,
                )
                for c, v in cols.items()
            }
            reports[name] = check_export(export, specs)
        matching = tuple(h for h, r in reports.items() if r.passed)
        out.append(
            StrategyDecision(group, EXPECTED_HYP[group], matching, reports, sync, dropped, mintick)
        )
    return out


def summary(decisions: list[StrategyDecision]) -> str:
    lines = [
        "## Strategy (broker emulator)",
        "",
        "| rule | expected (emulator) | matching | status | first divergence of expected |",
        "|---|---|---|---|---|",
    ]
    for d in decisions:
        rep = d.reports[d.expected]
        fd = ", ".join(
            f"{c.name}: {c.first_divergence} ({c.divergence_type})"
            for c in rep.columns
            if not c.passed and c.first_divergence is not None
        )
        if d.matching == (d.expected,):
            status = "ok"
        elif d.undecided:
            status = "not decidable in this export"
        else:
            status = "**emulator must change**"
        lines.append(
            f"| {d.group} | {d.expected} | {', '.join(d.matching) or '**none**'} | {status} "
            f"| {fd or '—'} |"
        )
    sync = decisions[0].sync_row if decisions else 0
    if sync:
        lines += ["", f"Export starts mid-history: compared from row {sync} (first flat bar); "
                  "cumulative columns anchored there."]  # fmt: skip
    if decisions and decisions[0].mintick is None:
        lines += ["", "Tick size unknown: rounding rules not tested and rounding left off. "
                  "Pass `--mintick` (the symbol's tick, e.g. 0.00001) or re-export with the "
                  "current capture_strategy.pine, which plots it."]  # fmt: skip
    elif decisions:
        lines += ["", f"Tick size used: {decisions[0].mintick:g}."]
    if decisions and decisions[0].dropped_live_bar:
        lines += ["", "Last row dropped: the last bar was live and a strategy calculates only on "
                  "bar close, so its columns were empty there."]  # fmt: skip
    bad = [d.group for d in decisions if not d.ok]
    lines += [
        "",
        f"Verdict: {'all rules match the emulator' if not bad else f'mismatched: {bad}'}",
        "",
    ]
    return "\n".join(lines)
