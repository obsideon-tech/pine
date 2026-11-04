"""Acceptance check for a whole port, stage by stage (PINE_PLAN.md §11 H1).

A complex indicator is a pipeline: inputs → intermediate series → outputs. Comparing only the
final output says *that* a port is wrong, not *where*. `check_port` compares every stage the
Pine script exposes as a plot (debug plots of intermediates added to a copy of the script) with
the same-named ``ctx.plot`` of a `pine.runtime` port, and names the **root cause**: the stage
that diverges first in time (ties: the earliest stage in pipeline order). Fix that stage and
re-run; later stages usually follow.

Usage::

    from pine.port import Stage, check_port

    report = check_port(
        export,                     # pine.io.load_tradingview_csv(...) of the debug script
        script,                     # def script(ctx): ... ctx.plot("dbg_smooth", s) ...
        [Stage("dbg_smooth"), Stage("dbg_trend"), Stage("signal", kind="discrete")],
    )
    print(report.to_markdown())

If the export has a ``bar_index`` column (``plot(bar_index, "bar_index")`` in the debug
script, recommended), the port runs with the true bar number of the first row, and a
truncated export (§5.4) is handled: stages are compared after ``exclude`` warm-up rows, or,
for stateful stages (``sync=True``), from the observed synchronisation row
(`pine.truncation.sync_point`), which must fall in the first half of the export.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from pine.io import TradingViewExport
from pine.parity import Kind, ParityReport, PortSpec, Tolerance, check_export, tolerance_for
from pine.runtime import Context, RunResult, run
from pine.truncation import bar_offset, sync_point

__all__ = ["PortReport", "Stage", "check_port"]


@dataclass(frozen=True)
class Stage:
    """One compared series, listed in pipeline order (inputs before what depends on them)."""

    column: str
    """Exported column: the plot title in the (debug) Pine script."""
    plot: str | None = None
    """Name the port plots it under with ``ctx.plot``; default: ``column``."""
    kind: Kind | Literal["discrete"] = "unbounded"
    tolerance: Tolerance | None = None
    """Default: `pine.parity.tolerance_for` the kind, floored by the printed precision."""
    exclude: int = 0
    """Warm-up rows to skip (always applied; in a truncated export, also the earliest row
    from which a ``sync`` stage may synchronise)."""
    sync: bool = False
    """Stateful stage (state machine, trailing level): in a truncated export, compare from
    the row where port and Pine agree from then on."""
    excuse: Callable | None = None
    """``kind="discrete"`` only: see `pine.parity.PortSpec.excuse`."""


@dataclass(frozen=True)
class PortReport:
    parity: ParityReport
    stages: tuple[str, ...]
    first_divergence: dict[str, int | None]
    """Per stage: first row where it differs (`None` if it passes or is inconclusive)."""
    status: dict[str, str]
    """Per stage: ``pass``, ``FAIL`` or ``inconclusive`` (too few rows compared)."""
    offset: int
    """True ``bar_index`` of the first exported row."""
    compared_from: dict[str, int]
    notes: tuple[str, ...] = ()
    run: RunResult | None = None

    @property
    def passed(self) -> bool:
        return self.parity.passed

    @property
    def first_failing_stage(self) -> str | None:
        """First stage in pipeline order that fails."""
        return next((s for s in self.stages if self.status[s] != "pass"), None)

    @property
    def root_cause(self) -> str | None:
        """The failing stage that diverges earliest in time; ties go to the earlier stage."""
        failing = [(d, i, s) for i, s in enumerate(self.stages)
                   if (d := self.first_divergence[s]) is not None]  # fmt: skip
        return min(failing)[2] if failing else None

    def to_markdown(self) -> str:
        lines = [
            f"## Port check: {'PASS' if self.passed else 'DIFFERENCES FOUND'}",
            "",
        ]
        rc = self.root_cause
        if rc is not None:
            lines += [
                f"Root cause candidate: **{rc}** (first divergence at row "
                f"{self.first_divergence[rc]}); first failing stage in pipeline order: "
                f"**{self.first_failing_stage}**.",
                "",
            ]
        lines += [
            "| # | stage | compared from row | first divergence | result |",
            "|---|---|---|---|---|",
        ]
        for i, s in enumerate(self.stages, 1):
            fd = self.first_divergence[s]
            lines.append(
                f"| {i} | {s} | {self.compared_from[s]} | {'—' if fd is None else fd} "
                f"| {self.status[s]} |"
            )
        if self.offset:
            lines += ["", f"Export starts at bar_index {self.offset} (truncated history, §5.4)."]
        lines += [*([""] + [f"- {n}" for n in self.notes] if self.notes else []), ""]
        lines += ["### Detail", "", self.parity.to_markdown()]
        return "\n".join(lines)


def check_port(
    export: TradingViewExport,
    script: Callable[[Context], None],
    stages: Sequence[Stage],
    *,
    bar_index_column: str = "bar_index",
    strategy=None,
    max_bars_back: int = 5000,
    min_compared: int = 100,
) -> PortReport:
    """Run ``script`` on the export's bars and compare every stage (see module docstring).

    ``min_compared`` applies to truncated exports only: a stage with fewer compared rows is
    *inconclusive*, not a pass."""
    if not stages:
        raise ValueError("no stages given")
    names = [s.column for s in stages]
    if len(set(names)) != len(names):
        raise ValueError("stage columns must be unique")
    missing = [c for c in names if c not in export.columns]
    if missing:
        raise KeyError(
            f"columns not in export: {missing}. Add them to the debug script as plots "
            f'(plot(x, "{missing[0]}")) and re-export.'
        )
    k = bar_offset(export.columns, bar_index_column) or 0
    result = run(script, export.bars, max_bars_back, strategy=strategy, bar_offset=k)
    n = len(export.bars)
    specs: dict[str, PortSpec] = {}
    compared_from: dict[str, int] = {}
    notes: list[str] = []
    for st in stages:
        plot = st.plot or st.column
        if plot not in result.columns:
            raise KeyError(f"the port never plotted {plot!r} (stage {st.column!r})")
        values = np.asarray(result[plot], dtype=float)
        exclude = min(st.exclude, n)
        if k and st.sync and st.kind != "discrete":
            pine = np.asarray(export.columns[st.column], dtype=float)
            tol = st.tolerance or tolerance_for(st.kind, export.observed_decimals.get(st.column))
            r = sync_point(values, pine, tol.bound(pine), exclude)
            if r <= n // 2:
                exclude = r
                notes.append(f"{st.column}: synchronised at row {r}")
            else:
                notes.append(f"{st.column}: never synchronised in the first half of the export")
        compared_from[st.column] = exclude
        if st.kind == "discrete":
            specs[st.column] = PortSpec(
                lambda b, v=values: ~np.isnan(v) & (v != 0),
                kind="discrete",
                excuse=_mask_warmup(st.excuse, exclude),
            )
        else:
            specs[st.column] = PortSpec(
                lambda b, v=values: v,
                kind=st.kind,
                tolerance=st.tolerance,
                mode="convergence" if exclude else "aligned",
                exclude=exclude,
                min_compared=min_compared if k else 0,
            )
    rep = check_export(export, specs)
    first: dict[str, int | None] = {}
    status: dict[str, str] = {}
    for c in rep.columns:
        first[c.name] = None if c.passed else c.first_divergence
        status[c.name] = "inconclusive" if c.inconclusive else ("pass" if c.passed else "FAIL")
    for d in rep.discrete:
        first[d.name] = d.first_unexcused
        status[d.name] = "pass" if d.passed else "FAIL"
    return PortReport(
        parity=rep,
        stages=tuple(names),
        first_divergence={s: first[s] for s in names},
        status={s: status[s] for s in names},
        offset=k,
        compared_from=compared_from,
        notes=tuple(notes),
        run=result,
    )


def _mask_warmup(excuse: Callable | None, exclude: int) -> Callable | None:
    """Discrete stages: warm-up rows are excused rather than dropped."""
    if not exclude:
        return excuse

    def f(bars):
        m = np.zeros(len(bars), bool)
        m[:exclude] = True
        if excuse is not None:
            m |= np.asarray(excuse(bars), bool)
        return m

    return f
