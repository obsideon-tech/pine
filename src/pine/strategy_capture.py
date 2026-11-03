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
}
EXPECTED_HYP = {
    "baseline": "expected",
    "fill_timing": "next_open",
    "path": "nearest_extreme_first",
    "gap_fill": "open",
    "exit_same_bar": "yes",
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

    @property
    def undecided(self) -> bool:
        return len(self.matching) > 1 and self.expected in self.matching

    @property
    def ok(self) -> bool:
        return self.matching == (self.expected,) or self.undecided


def _columns(export: TradingViewExport, cfg: StrategyConfig, k: int) -> dict[str, np.ndarray]:
    res = run(strategy_capture, export.bars, strategy=cfg, bar_offset=k)
    return {c: np.asarray(res[c]) for c in COLUMNS}


def decide(export: TradingViewExport) -> list[StrategyDecision]:
    k = bar_offset(export.columns, "strat_bar_index") or 0
    n = len(export.bars)
    cycle = int(CONSTANTS["CYCLE"])
    sync = 0 if k == 0 else (cycle - 1 - k) % cycle  # first row at phase 39 (flat)
    out = []
    for group, hyps in GROUPS.items():
        reports = {}
        for name, cfg in hyps.items():
            cols = _columns(export, cfg, k)
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
        out.append(StrategyDecision(group, EXPECTED_HYP[group], matching, reports, sync))
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
    bad = [d.group for d in decisions if not d.ok]
    lines += [
        "",
        f"Verdict: {'all rules match the emulator' if not bad else f'mismatched: {bad}'}",
        "",
    ]
    return "\n".join(lines)
