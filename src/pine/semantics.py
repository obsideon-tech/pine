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
from pine.parity import ParityReport, PortSpec, check_export
from pine.runtime import Context
from pine.runtime import scalar as ps

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


def _idx(b: Bars) -> np.ndarray:
    return np.arange(len(b))


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


def _regime_loop(b: Bars) -> dict[str, np.ndarray]:
    f, s = ta.ema(b.close, K["FAST"]), ta.ema(b.close, K["SLOW"])
    up, dn = ta.crossover(f, s), ta.crossunder(f, s)
    reg, bars_in = np.zeros(len(b)), np.zeros(len(b))
    r, k = 0.0, 0.0
    for t in range(len(b)):
        if up[t]:
            r, k = 1.0, 0.0
        elif dn[t]:
            r, k = -1.0, 0.0
        else:
            k += 1.0
        reg[t], bars_in[t] = r, k
    return {"regime": reg, "bars_in_regime": bars_in}


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
            var_acc=lambda b: np.cumsum(b.close), varip_acc=lambda b: np.cumsum(b.close)
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
        "na_before_first": _cols(barssince=lambda b: ta.barssince(_events(b))),
        "count_from_start": _cols(
            barssince=lambda b: np.where(
                np.isnan(ta.barssince(_events(b))), _idx(b), ta.barssince(_events(b))
            )
        ),
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
    "regime": {"port": _regime_loop},
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
    "rounding": "half_up",
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
    "regime": "port",
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

    @property
    def ok(self) -> bool:
        return self.matching == (self.expected,)


def decide(export: TradingViewExport) -> list[GroupDecision]:
    """For each probe group, the hypotheses whose columns all pass in aligned mode."""
    out = []
    for group, hyps in GROUPS.items():
        reports = {}
        for name, hyp in hyps.items():
            cols = hyp(export.bars)
            specs = {c: PortSpec(lambda b, v=v: v, kind="unbounded") for c, v in cols.items()}
            reports[name] = check_export(export, specs)
        matching = tuple(h for h, r in reports.items() if r.passed)
        out.append(GroupDecision(group, EXPECTED[group], matching, reports))
    return out


def _summary(decisions: list[GroupDecision]) -> str:
    lines = [
        "## Pine semantics",
        "",
        "| group | expected (runtime) | matching | status | first divergence of expected |",
        "|---|---|---|---|---|",
    ]
    for d in decisions:
        exp_rep = d.reports[d.expected]
        fd = ", ".join(
            f"{c.name}: {c.first_divergence} ({c.divergence_type})"
            for c in exp_rep.columns
            if not c.passed
        )
        status = "ok" if d.ok else "**runtime must change**"
        lines.append(
            f"| {d.group} | {d.expected} | {', '.join(d.matching) or '**none**'} | {status} "
            f"| {fd or '—'} |"
        )
    bad = [d.group for d in decisions if not d.ok]
    verdict = "all groups match the runtime" if not bad else f"mismatched groups: {bad}"
    lines += ["", f"Verdict: {verdict}", ""]
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
