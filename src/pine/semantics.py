"""Decide Pine's execution-model and scalar semantics from a capture export (§11 E1).

``tools/capture_semantics.pine`` plots one column per probe. For each probe *group* this module
holds competing hypotheses as **independent** numpy/loop implementations (not `pine.runtime`),
and `EXPECTED` names the hypothesis `pine.runtime` implements. `decide` reports which
hypotheses an export matches; `runtime_capture` re-implements the whole capture script on
`pine.runtime`, so tests can tie the runtime to `EXPECTED`.

Command line::

    python -m pine.semantics aapl_1d.csv --manifest aapl_1d.yaml [--format md|json]

Usually run through ``python -m pine.capture check`` (docs/CAPTURE.md).

Exit status: 0 if every group matches exactly its expected hypothesis; 1 if any group matches
another hypothesis, several, or none (the runtime or the expectation must change); 2 on input
errors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pine import ta
from pine._cli import InputError, load_export
from pine.bars import Bars
from pine.io import TradingViewExport
from pine.parity import ParityReport, PortSpec, check_export, convergence_bars, tolerance_for
from pine.runtime import Context
from pine.runtime import scalar as ps
from pine.truncation import bar_offset, e0_bound, sync_point

__all__ = [
    "CONSTANTS",
    "COLUMNS",
    "EXPECTED",
    "GROUPS",
    "GroupDecision",
    "decide",
    "main",
    "runtime_capture",
]

# Must match tools/capture_semantics.pine (enforced by tests/test_semantics.py).
CONSTANTS: dict[str, float] = {
    "CALL_EVERY": 3,
    "EVENT_EVERY": 7,
    "EVENT_PHASE": 3,
    "LEN": 5,
    "GAP_PERIOD": 50,
    "GAP_WIDTH": 3,
    "FAST": 9,
    "SLOW": 21,
    "ST_ATR": 10,
    "ST_FACTOR": 3.0,
    "TRAIL_ATR": 14,
    "TRAIL_MULT": 2.0,
}
K = CONSTANTS
Hyp = Callable[[Bars], dict[str, np.ndarray]]


# ------------------------------------------------------------------ shared inputs


class _At:
    """Bars seen from a known offset: ``k`` is the true ``bar_index`` of the first row and
    ``cols`` the exported columns (only used to anchor values that depend on unseen history)."""

    def __init__(self, bars: Bars, k: int = 0, cols: dict | None = None):
        self._b, self.k, self.cols = bars, k, cols or {}

    def __getattr__(self, name):
        return getattr(self._b, name)

    def __len__(self) -> int:
        return len(self._b)


def _idx(b) -> np.ndarray:
    return getattr(b, "k", 0) + np.arange(len(b))


def _calls(b: Bars) -> np.ndarray:
    return _idx(b) % K["CALL_EVERY"] == 0


def _events(b: Bars) -> np.ndarray:
    return _idx(b) % K["EVENT_EVERY"] == K["EVENT_PHASE"]


def _src_na(b: Bars) -> np.ndarray:
    gap = _idx(b) % K["GAP_PERIOD"] >= K["GAP_PERIOD"] - K["GAP_WIDTH"]
    return np.where(gap, np.nan, b.close)


def _on_calls(b: Bars, values_on_calls: np.ndarray) -> np.ndarray:
    out = np.full(len(b), np.nan)
    out[_calls(b)] = values_on_calls
    return out


def _const(b: Bars, v: float) -> np.ndarray:
    return np.full(len(b), float(v))


def _lag(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[np.nan], x[:-1]])


# ------------------------------------------------------------------ hypotheses


def _cond(fn):
    return {
        "per_call": lambda b: _on_calls(b, fn(np.asarray(b.close)[_calls(b)], K["LEN"])),
        "every_bar": lambda b: np.where(_calls(b), fn(b.close, K["LEN"]), np.nan),
    }


def _cond_cross_per_call(b: Bars) -> np.ndarray:
    f, s = ta.ema(b.close, K["FAST"]), ta.ema(b.close, K["SLOW"])
    m = _calls(b)
    return _on_calls(b, ta.crossover(f[m], s[m]).astype(float))


def _cond_cross_every_bar(b: Bars) -> np.ndarray:
    f, s = ta.ema(b.close, K["FAST"]), ta.ema(b.close, K["SLOW"])
    return np.where(_calls(b), ta.crossover(f, s).astype(float), np.nan)


def _branch_last_assigned(b: Bars) -> np.ndarray:
    out = np.full(len(b), np.nan)
    last = np.nan
    c, x = _calls(b), np.asarray(b.close)
    for t in range(len(b)):
        out[t] = last
        if c[t]:
            last = x[t]
    return out


def _change_last_valid(b: Bars) -> np.ndarray:
    x = _src_na(b)
    out = np.full(x.size, np.nan)
    last = np.nan
    for t in range(x.size):
        if not math.isnan(x[t]):
            out[t] = x[t] - last
            last = x[t]
    return out


def _trail_loop(b: Bars) -> np.ndarray:
    a = ta.atr(b.high, b.low, b.close, K["TRAIL_ATR"])
    cand = np.asarray(b.close) - K["TRAIL_MULT"] * a
    out = np.full(len(b), np.nan)
    trail = np.nan
    for t in range(len(b)):
        if math.isnan(trail) or b.close[t] < trail:
            trail = cand[t]
        else:
            trail = np.nan if math.isnan(cand[t]) else max(trail, cand[t])
        out[t] = trail
    return out


def _regime_loop(b: Bars, per_call: bool) -> dict[str, np.ndarray]:
    """The regime state machine. ``per_call``: ``ta.crossunder`` sits in the ``else if`` and
    is evaluated only when the crossover is false, so its previous values are from its last
    *evaluation*; after a one-bar whipsaw (crossover, then immediately back) it misses the
    crossunder. ``every_bar``: crossunder evaluated on every bar."""
    f, s = ta.ema(b.close, K["FAST"]), ta.ema(b.close, K["SLOW"])
    up = ta.crossover(f, s)
    dn_every = ta.crossunder(f, s)
    reg, bars_in = np.zeros(len(b)), np.zeros(len(b))
    r, k = 0.0, 0.0
    pa = pb = np.nan  # crossunder's own previous inputs (per_call)
    for t in range(len(b)):
        if up[t]:
            r, k = 1.0, 0.0
        else:
            if per_call:
                dn = f[t] < s[t] and pa >= pb
                pa, pb = f[t], s[t]
            else:
                dn = dn_every[t]
            if dn:
                r, k = -1.0, 0.0
            else:
                k += 1.0
        reg[t], bars_in[t] = r, k
    return {"regime": reg, "bars_in_regime": bars_in}


def _anchored_cumsum(b, col: str) -> np.ndarray:
    """Running sum of close. In a truncated export the sum of the unseen bars is unknown, so it
    is anchored on the exported first value; the check then verifies every increment."""
    c = np.cumsum(b.close)
    if getattr(b, "k", 0) > 0 and col in b.cols:
        c = c + (b.cols[col][0] - c[0])
    return c


def _barssince_closed(b, before_first: str) -> np.ndarray:
    """``ta.barssince`` of the periodic event, from the true bar index (exact for any
    offset): ``(g - PHASE) mod EVERY`` once the first event has happened."""
    g = _idx(b)
    e, ph = K["EVENT_EVERY"], K["EVENT_PHASE"]
    out = ((g - ph) % e).astype(float)
    if before_first == "na":
        return np.where(g >= ph, out, np.nan)
    return np.where(g >= ph, out, g.astype(float))


def _st(b: Bars) -> dict[str, np.ndarray]:
    line, d = ta.supertrend(b.high, b.low, b.close, K["ST_FACTOR"], K["ST_ATR"])
    return {"supertrend": line, "supertrend_dir": d}


def _cols(**fns) -> Hyp:
    return lambda b: {c: f(b) for c, f in fns.items()}


GROUPS: dict[str, dict[str, Hyp]] = {
    "alignment": {"aligned": _cols(bar_index=lambda b: _idx(b).astype(float))},
    "cond_ema": {h: _cols(cond_ema=f) for h, f in _cond(ta.ema).items()},
    "cond_sma": {h: _cols(cond_sma=f) for h, f in _cond(ta.sma).items()},
    "cond_userfn": {
        "per_call": _cols(
            cond_userfn_diff=lambda b: _on_calls(b, ta.change(np.asarray(b.close)[_calls(b)]))
        ),
        "every_bar": _cols(
            cond_userfn_diff=lambda b: np.where(_calls(b), ta.change(b.close), np.nan)
        ),
    },
    "cond_crossover": {
        "per_call": _cols(cond_crossover=_cond_cross_per_call),
        "every_bar": _cols(cond_crossover=_cond_cross_every_bar),
    },
    "call_sites": {
        "separate": _cols(
            site_a_diff=lambda b: ta.change(b.close), site_b_diff=lambda b: ta.change(b.open)
        ),
        "shared": _cols(
            site_a_diff=lambda b: np.asarray(b.close) - _lag(np.asarray(b.open)),
            site_b_diff=lambda b: np.asarray(b.open) - np.asarray(b.close),
        ),
    },
    "var": {
        "cumulative": _cols(
            var_acc=lambda b: _anchored_cumsum(b, "var_acc"),
            varip_acc=lambda b: _anchored_cumsum(b, "varip_acc"),
        )
    },
    "branch_hist": {
        "per_bar": _cols(branch_hist=lambda b: _lag(np.where(_calls(b), b.close, np.nan))),
        "last_assigned": _cols(branch_hist=_branch_last_assigned),
    },
    "rounding": {
        name: _cols(
            round_pos_half=lambda b, v=v: _const(b, v[0]),
            round_neg_half=lambda b, v=v: _const(b, v[1]),
            round_odd_half=lambda b, v=v: _const(b, v[2]),
        )
        for name, v in {
            "half_up": (3, -2, 4),
            "half_away": (3, -3, 4),
            "half_even": (2, -2, 4),
        }.items()
    },
    "mod": {
        "truncated": _cols(mod_neg=lambda b: _const(b, -1)),
        "floored": _cols(mod_neg=lambda b: _const(b, 2)),
    },
    "div_series": {
        "float": _cols(div_int_series=lambda b: _const(b, 3.5)),
        "int": _cols(div_int_series=lambda b: _const(b, 3)),
    },
    "div_const": {
        "float": _cols(div_int_const=lambda b: _const(b, 3.5)),
        "int": _cols(div_int_const=lambda b: _const(b, 3)),
    },
    "int_trunc": {
        "toward_zero": _cols(int_trunc=lambda b: _const(b, -2)),
        "floor": _cols(int_trunc=lambda b: _const(b, -3)),
    },
    "na_eq_na": {
        "false": _cols(na_eq_na=lambda b: _const(b, 0)),
        "true": _cols(na_eq_na=lambda b: _const(b, 1)),
    },
    "barssince": {
        "na_before_first": _cols(barssince=lambda b: _barssince_closed(b, "na")),
        "count_from_start": _cols(barssince=lambda b: _barssince_closed(b, "count")),
    },
    "valuewhen": {
        "na_before_first": _cols(valuewhen=lambda b: ta.valuewhen(_events(b), b.close)),
        "zero_before_first": _cols(
            valuewhen=lambda b: np.nan_to_num(ta.valuewhen(_events(b), b.close), nan=0.0)
        ),
    },
    "change_na": {
        "propagate": _cols(change_na=lambda b: ta.change(_src_na(b))),
        "last_valid": _cols(change_na=_change_last_valid),
    },
    "supertrend": {"manual_code": _st},
    "trail_stop": {"port": _cols(trail_stop=_trail_loop)},
    "regime": {
        "per_call": lambda b: _regime_loop(b, per_call=True),
        "every_bar": lambda b: _regime_loop(b, per_call=False),
    },
}

EXPECTED: dict[str, str] = {
    "alignment": "aligned",
    "cond_ema": "per_call",
    "cond_sma": "per_call",
    "cond_userfn": "per_call",
    "cond_crossover": "per_call",
    "call_sites": "separate",
    "var": "cumulative",
    "branch_hist": "per_bar",
    "rounding": "half_away",  # confirmed by export 2026-11-03
    "mod": "truncated",
    "div_series": "float",
    "div_const": "float",
    "int_trunc": "toward_zero",
    "na_eq_na": "false",
    "barssince": "na_before_first",
    "valuewhen": "na_before_first",
    "change_na": "propagate",
    "supertrend": "manual_code",
    "trail_stop": "port",
    "regime": "per_call",
}

COLUMNS: tuple[str, ...] = (
    "bar_index", "cond_ema", "cond_sma", "cond_userfn_diff", "cond_crossover", "site_a_diff",
    "site_b_diff", "var_acc", "varip_acc", "branch_hist", "round_pos_half", "round_neg_half",
    "round_odd_half", "mod_neg", "div_int_series", "div_int_const", "int_trunc", "na_eq_na",
    "barssince", "valuewhen", "change_na", "supertrend", "supertrend_dir", "trail_stop",
    "regime", "bars_in_regime",
)  # fmt: skip


# ------------------------------------------------------------------ the script on pine.runtime


def _diff(ctx: Context, x: float) -> float:
    xs = ctx.series("x", x)
    return xs.value - xs[1]


def _supertrend(ctx: Context, factor: float, atr_period: int) -> tuple[float, float]:
    """The manual's ``ta.supertrend`` code, line for line, on `pine.runtime`."""
    src = (ctx.high.value + ctx.low.value) / 2
    atr = ctx.series("atr", ctx.ta.atr("atr", atr_period))
    upper = ctx.series("upper", src + factor * atr.value)
    lower = ctx.series("lower", src - factor * atr.value)
    prev_lower, prev_upper = ps.nz(lower[1]), ps.nz(upper[1])
    close, close1 = ctx.close.value, ctx.close[1]
    if not (lower.value > prev_lower or close1 < prev_lower):
        lower.set(prev_lower)
    if not (upper.value < prev_upper or close1 > prev_upper):
        upper.set(prev_upper)
    direction = ctx.series("direction")
    st = ctx.series("supertrend")
    prev_st = st[1]
    if ps.na(atr[1]):
        direction.set(1)
    elif ps.eq(prev_st, prev_upper):
        direction.set(-1 if close > upper.value else 1)
    else:
        direction.set(1 if close < lower.value else -1)
    st.set(lower.value if direction.value == -1 else upper.value)
    return st.value, direction.value


def runtime_capture(ctx: Context) -> None:
    """``tools/capture_semantics.pine`` on `pine.runtime` (one call per bar)."""
    i = ctx.bar_index.value
    close, open_ = ctx.close.value, ctx.open.value
    ctx.plot("bar_index", i)
    c = i % K["CALL_EVERY"] == 0

    cond_ema = ctx.series("condEma")
    if c:
        cond_ema.set(ctx.ta.ema("condEma", close, K["LEN"]))
    ctx.plot("cond_ema", cond_ema.value)

    cond_sma = ctx.series("condSma")
    if c:
        cond_sma.set(ctx.ta.sma("condSma", close, K["LEN"]))
    ctx.plot("cond_sma", cond_sma.value)

    cond_diff = ctx.series("condDiff")
    if c:
        cond_diff.set(ctx.call("condDiff", _diff, close))
    ctx.plot("cond_userfn_diff", cond_diff.value)

    fast = ctx.ta.ema("fastE", close, K["FAST"])
    slow = ctx.ta.ema("slowE", close, K["SLOW"])
    cond_cross = ctx.series("condCross")
    if c:
        cond_cross.set(1 if ctx.ta.crossover("condCross", fast, slow) else 0)
    ctx.plot("cond_crossover", cond_cross.value)

    ctx.plot("site_a_diff", ctx.call("siteA", _diff, close))
    ctx.plot("site_b_diff", ctx.call("siteB", _diff, open_))

    acc = ctx.var("acc", 0.0)
    acc.set(acc.value + close)
    ctx.plot("var_acc", acc.value)
    accip = ctx.var("accip", 0.0)  # varip == var on historical bars
    accip.set(accip.value + close)
    ctx.plot("varip_acc", accip.value)

    y = ctx.series("y")
    if c:
        y.set(close)
    ctx.plot("branch_hist", y[1])

    ctx.plot("round_pos_half", ps.round_(2.5))
    ctx.plot("round_neg_half", ps.round_(-2.5))
    ctx.plot("round_odd_half", ps.round_(3.5))
    seven = 7 + i * 0
    ctx.plot("mod_neg", ps.mod(-seven, 3))
    ctx.plot("div_int_series", ps.div(seven, 2))
    ctx.plot("div_int_const", ps.div(7, 2))
    ctx.plot("int_trunc", ps.to_int(-2.7))
    ctx.plot("na_eq_na", 1 if ps.eq(ps.NA, ps.NA) else 0)

    ev = i % K["EVENT_EVERY"] == K["EVENT_PHASE"]
    ctx.plot("barssince", ctx.ta.barssince("barssince", ev))
    ctx.plot("valuewhen", ctx.ta.valuewhen("valuewhen", ev, close))
    gap = i % K["GAP_PERIOD"] >= K["GAP_PERIOD"] - K["GAP_WIDTH"]
    ctx.plot("change_na", ctx.ta.change("srcNa", ps.NA if gap else close))

    line, d = ctx.call("st", _supertrend, K["ST_FACTOR"], K["ST_ATR"])
    ctx.plot("supertrend", line)
    ctx.plot("supertrend_dir", d)

    atr_t = ctx.ta.atr("atrT", K["TRAIL_ATR"])
    candidate = close - K["TRAIL_MULT"] * atr_t
    trail = ctx.var("trail")
    if ps.na(trail.value) or close < trail.value:
        trail.set(candidate)
    else:
        trail.set(ps.max_(trail.value, candidate))
    ctx.plot("trail_stop", trail.value)

    regime, bars_in = ctx.var("regime", 0), ctx.var("barsIn", 0)
    if ctx.ta.crossover("regimeUp", fast, slow):
        regime.set(1)
        bars_in.set(0)
    elif ctx.ta.crossunder("regimeDn", fast, slow):  # evaluated only when the first is false
        regime.set(-1)
        bars_in.set(0)
    else:
        bars_in.set(bars_in.value + 1)
    ctx.plot("regime", regime.value)
    ctx.plot("bars_in_regime", bars_in.value)


# ------------------------------------------------------------------ deciding


@dataclass(frozen=True)
class GroupDecision:
    group: str
    expected: str
    matching: tuple[str, ...]
    reports: dict[str, ParityReport]
    excluded: str = "0"
    """Rows excluded because they depend on history before the export ("0" if none)."""

    @property
    def undecided(self) -> bool:
        """The expected hypothesis matches, but so does another: this export cannot tell
        them apart (e.g. behaviour before the first event, when the export starts later)."""
        return len(self.matching) > 1 and self.expected in self.matching

    @property
    def ok(self) -> bool:
        return self.matching == (self.expected,) or self.undecided


# In a truncated export, a history-dependent group must still compare this many rows.
MIN_COMPARED = 100


def _truncation_rule(group: str, b, tol: float):
    """Rows to exclude for ``group`` when the export starts after the chart's first bar:
    an int, or ``("sync", start)`` for stateful ports compared from their observed
    synchronisation row (`pine.truncation`)."""
    close = np.asarray(b.close)
    e0 = e0_bound(close)
    tr = ta.tr(b.high, b.low, b.close, handle_na=True)
    e0_tr = e0_bound(tr)
    every = K["CALL_EVERY"]

    def cb(alpha, bound=e0):
        return convergence_bars(alpha, tol, bound)

    k = getattr(b, "k", 0)
    rules = {
        "cond_ema": every * (K["LEN"] - 1 + cb(2 / (K["LEN"] + 1))) + 2 * every,
        "cond_sma": every * K["LEN"] + every,
        "cond_userfn": 2 * every,
        "cond_crossover": K["SLOW"] - 1 + cb(2 / (K["SLOW"] + 1)) + 2 * every,
        "call_sites": 1,
        "branch_hist": 2 * every,  # fair to `last_assigned`, which needs a call in the export
        "valuewhen": (K["EVENT_PHASE"] - k) % K["EVENT_EVERY"],
        "change_na": int(np.flatnonzero(~np.isnan(_src_na(b)))[0]) + 1,
        "supertrend": ("sync", K["ST_ATR"] - 1 + cb(1 / K["ST_ATR"], e0_tr)),
        "trail_stop": ("sync", K["TRAIL_ATR"] - 1 + cb(1 / K["TRAIL_ATR"], e0_tr)),
        "regime": ("sync", K["SLOW"] - 1 + cb(2 / (K["SLOW"] + 1))),
    }
    return rules.get(group, 0)


def decide(export: TradingViewExport) -> list[GroupDecision]:
    """For each probe group, the hypotheses whose columns all pass.

    The exported ``bar_index`` gives the true bar number of the first row. If it is not 0, the
    export starts after the chart's first bar and history-dependent groups are compared after
    the exclusions of `_truncation_rule` (`pine.truncation`)."""
    k = bar_offset(export.columns) or 0
    b = _At(export.bars, k, export.columns)
    n = len(export.bars)
    out = []
    for group, hyps in GROUPS.items():
        cols_of = {name: hyp(b) for name, hyp in hyps.items()}
        tols = {
            c: tolerance_for("unbounded", export.observed_decimals.get(c))
            for c in next(iter(cols_of.values()))
        }
        # Internal filters (EMAs, ATR) must converge to float precision, whatever the plotted
        # column's own tolerance (0.5 for integer columns such as `regime`).
        conv_tol = min(1e-10, min(float(np.min(t.atol)) for t in tols.values()))
        rule = _truncation_rule(group, b, conv_tol) if k else 0
        excluded_by: dict[str, str] = {}
        reports = {}
        for name, cols in cols_of.items():
            if isinstance(rule, tuple):  # stateful port: observed synchronisation row
                start = min(rule[1], n)
                r = max(
                    sync_point(v, export.columns[c], tols[c].bound(export.columns[c]), start)
                    for c, v in cols.items()
                )
                exclude = r if r <= n // 2 else start
                excluded_by[name] = (
                    f"{exclude} (synchronised)"
                    if r <= n // 2
                    else f"never in first half (from {start})"
                )
            else:
                exclude = min(rule, n)
                excluded_by[name] = str(exclude)
            specs = {
                c: PortSpec(
                    lambda bb, v=v: v,
                    kind="unbounded",
                    tolerance=tols[c],
                    mode="convergence" if exclude else "aligned",
                    exclude=exclude,
                    min_compared=MIN_COMPARED if exclude else 0,
                )
                for c, v in cols.items()
            }
            reports[name] = check_export(export, specs)
        matching = tuple(h for h, r in reports.items() if r.passed)
        out.append(
            GroupDecision(group, EXPECTED[group], matching, reports, excluded_by[EXPECTED[group]])
        )
    return out


def _status(d: GroupDecision) -> str:
    if d.matching == (d.expected,):
        return "ok"
    if d.undecided:
        return "not decidable in this export"
    if any(c.inconclusive for c in d.reports[d.expected].columns):
        return "inconclusive: export too short after exclusions"
    return "**runtime must change**"


def _summary(decisions: list[GroupDecision]) -> str:
    lines = [
        "## Pine semantics",
        "",
        "| group | expected (runtime) | matching | status | rows excluded "
        "| first divergence of expected |",
        "|---|---|---|---|---|---|",
    ]
    for d in decisions:
        exp_rep = d.reports[d.expected]
        fd = ", ".join(
            f"{c.name}: {c.first_divergence} ({c.divergence_type})"
            for c in exp_rep.columns
            if not c.passed and c.first_divergence is not None
        )
        lines.append(
            f"| {d.group} | {d.expected} | {', '.join(d.matching) or '**none**'} "
            f"| {_status(d)} | {d.excluded} | {fd or '—'} |"
        )
    bad = [d.group for d in decisions if not d.ok]
    undecided = [d.group for d in decisions if d.undecided]
    verdict = "all groups match the runtime" if not bad else f"mismatched groups: {bad}"
    if undecided:
        verdict += f"; not decidable in this export: {undecided}"
    lines += [
        "",
        f"Verdict: {verdict}",
        "",
        "Rows excluded: history-dependent probes are compared only after the rows that depend "
        "on bars before the export (PINE_PLAN.md §5.4, pine.truncation).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m pine.semantics", description="Check a semantics-capture export."
    )
    ap.add_argument("export")
    ap.add_argument("--manifest")
    ap.add_argument("--timeframe")
    ap.add_argument("--tz", default=None)
    ap.add_argument("--format", choices=("md", "json"), default="md")
    args = ap.parse_args(argv)
    try:
        exp, _ = load_export(args.export, args.manifest, args.timeframe, args.tz)
        missing = [c for c in COLUMNS if c not in exp.columns]
        if missing:
            raise KeyError(f"export lacks semantics columns {missing}")
    except (InputError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    decisions = decide(exp)
    if args.format == "json":
        payload = [
            {
                "group": d.group,
                "expected": d.expected,
                "matching": list(d.matching),
                "ok": d.ok,
                "expected_report": d.reports[d.expected].to_dict(),
            }
            for d in decisions
        ]
        print(json.dumps(payload, indent=2, allow_nan=False))
    else:
        sys.stdout.write(_summary(decisions))
    return 0 if all(d.ok for d in decisions) else 1


if __name__ == "__main__":
    sys.exit(main())
