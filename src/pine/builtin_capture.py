"""Check the batch-F built-ins against a TradingView export (PINE_PLAN.md §11 F).

``tools/capture_builtins.pine`` plots ``math.sum``, ``ta.wma``, ``ta.hma``, ``ta.variance``,
``ta.percentile_linear_interpolation``, ``ta.pivothigh``/``pivotlow``, ``fixnan``, ``ta.cum``,
``ta.obv`` and ``ta.pvt``, on a gap-free source and on one with periodic `na`. For each group
this module holds competing hypotheses as **independent** numpy implementations (not
`pine.ta`), and `EXPECTED` names the one `pine.ta` / `pine.runtime` implement; tests tie the two
together. `decide` reports which hypotheses an export matches.

Usually run through ``python -m pine.capture check`` (docs/CAPTURE.md).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from pine import sensitivity as sens
from pine import ta
from pine.bars import Bars
from pine.io import TradingViewExport
from pine.parity import PortSpec, check_export, tolerance_for
from pine.runtime import Context
from pine.semantics import GroupDecision, _At, _summary
from pine.truncation import bar_offset

__all__ = ["COLUMNS", "CONSTANTS", "EXPECTED", "GROUPS", "decide", "runtime_capture", "summary"]

# Must match tools/capture_builtins.pine (enforced by tests/test_builtin_capture.py).
CONSTANTS: dict[str, float] = {
    "LEN": 10,
    "HMA_ODD": 9,
    "HMA_NONSQUARE": 14,
    "PIV_LEFT": 3,
    "PIV_RIGHT": 2,
    "PCT": 30.0,
    "GAP_PERIOD": 50,
    "GAP_WIDTH": 3,
}
K = CONSTANTS
L = int(K["LEN"])
Hyp = Callable[[Bars], dict[str, np.ndarray]]

COLUMNS: tuple[str, ...] = (
    "bi_bar_index", "bi_volume", "bi_sum", "bi_sum_na", "bi_wma", "bi_wma_na", "bi_hma_odd",
    "bi_hma_nonsquare", "bi_variance", "bi_variance_na", "bi_pctl", "bi_pctl_na",
    "bi_pivothigh", "bi_pivotlow", "bi_fixnan", "bi_cum", "bi_cum_na", "bi_obv", "bi_pvt",
)  # fmt: skip


# ------------------------------------------------------------------ independent references


def _idx(b) -> np.ndarray:
    return getattr(b, "k", 0) + np.arange(len(b))


def _src_na(b) -> np.ndarray:
    gap = _idx(b) % K["GAP_PERIOD"] >= K["GAP_PERIOD"] - K["GAP_WIDTH"]
    return np.where(gap, np.nan, np.asarray(b.close, dtype=float))


def _volume(b) -> np.ndarray:
    cols = getattr(b, "cols", {})
    return np.asarray(cols["bi_volume"] if "bi_volume" in cols else b.volume, dtype=float)


def _rolling(x: np.ndarray, n: int, policy: str, fn) -> np.ndarray:
    """``fn(window)`` per bar under an ``na`` policy (see `pine.ta` docstring)."""
    out = np.full(x.size, np.nan)
    valid: list[float] = []
    for t in range(x.size):
        own_na = math.isnan(x[t])
        if policy == "propagate":
            if t >= n - 1 and not np.isnan(x[t - n + 1 : t + 1]).any():
                out[t] = fn(x[t - n + 1 : t + 1])
            continue
        if not own_na:
            valid.append(x[t])
        if len(valid) >= n and not (own_na and policy == "skip_na_out"):
            out[t] = fn(np.array(valid[-n:]))
    return out


def _wma_ref(x: np.ndarray, n: int, policy: str = "skip") -> np.ndarray:
    w = np.arange(1, n + 1, dtype=float)  # oldest .. newest
    return _rolling(x, n, policy, lambda win: float(np.dot(win, w) / w.sum()))


def _hma_ref(x: np.ndarray, n: int, half: str, root: str) -> np.ndarray:
    def rnd(v: float, how: str) -> int:
        return math.floor(v) if how == "floor" else math.floor(v + 0.5)

    inner = 2 * _wma_ref(x, rnd(n / 2, half)) - _wma_ref(x, n)
    return _wma_ref(inner, max(1, rnd(math.sqrt(n), root)))


def _pivot_ref(x: np.ndarray, left: int, right: int, rule: str, sign: float) -> np.ndarray:
    y = sign * np.asarray(x, dtype=float)
    out = np.full(y.size, np.nan)
    lt = (lambda c, v: c > v) if rule in ("left_strict", "strict") else (lambda c, v: c >= v)
    rt = (lambda c, v: c > v) if rule in ("right_strict", "strict") else (lambda c, v: c >= v)
    for t in range(left + right, y.size):
        c = y[t - right]
        lo, hi = y[t - right - left : t - right], y[t - right + 1 : t + 1]
        if np.isnan(y[t - right - left : t + 1]).any():
            continue
        if all(lt(c, v) for v in lo) and all(rt(c, v) for v in hi):
            out[t] = x[t - right]
    return out


def _cum_ref(x: np.ndarray, policy: str, dead_at_start: bool = False) -> np.ndarray:
    out = np.full(x.size, np.nan)
    total, dead = math.nan, dead_at_start
    for t, v in enumerate(x):
        if math.isnan(v):
            dead |= policy == "propagate"
            out[t] = total if (policy == "skip" and not dead) else math.nan
            continue
        if dead:
            continue
        total = v if math.isnan(total) else total + v
        out[t] = total
    return out


def _change(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[np.nan], x[1:] - x[:-1]])


def _prev(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[np.nan], x[:-1]])


def _policies(fn) -> dict[str, Hyp]:
    return {p: (lambda b, p=p: fn(b, p)) for p in ta.NA_POLICIES}


def _close(b) -> np.ndarray:
    return np.asarray(b.close, dtype=float)


GROUPS: dict[str, dict[str, Hyp]] = {
    "alignment": {"aligned": lambda b: {"bi_bar_index": _idx(b).astype(float)}},
    "sum": {"window": lambda b: {"bi_sum": _rolling(_close(b), L, "skip", np.sum)}},
    "sum_na": _policies(lambda b, p: {"bi_sum_na": _rolling(_src_na(b), L, p, np.sum)}),
    "wma": {"linear_weights": lambda b: {"bi_wma": _wma_ref(_close(b), L)}},
    "wma_na": _policies(lambda b, p: {"bi_wma_na": _wma_ref(_src_na(b), L, p)}),
    "hma": {
        f"half_{h}_sqrt_{r}": (
            lambda b, h=h, r=r: {
                "bi_hma_odd": _hma_ref(_close(b), int(K["HMA_ODD"]), h, r),
                "bi_hma_nonsquare": _hma_ref(_close(b), int(K["HMA_NONSQUARE"]), h, r),
            }
        )
        for h in ("floor", "round")
        for r in ("floor", "round")
    },
    "variance": {
        name: (
            lambda b, d=d: {
                "bi_variance": _rolling(_close(b), L, "skip", lambda w: np.var(w, ddof=d))
            }
        )
        for name, d in (("population", 0), ("sample", 1))
    },
    "variance_na": _policies(lambda b, p: {"bi_variance_na": _rolling(_src_na(b), L, p, np.var)}),
    "percentile": {
        name: (
            lambda b, m=m: {
                "bi_pctl": _rolling(
                    _close(b), L, "skip", lambda w: np.percentile(w, K["PCT"], method=m)
                )
            }
        )
        for name, m in (("inclusive", "linear"), ("exclusive", "weibull"), ("hazen", "hazen"))
    },
    "percentile_na": _policies(
        lambda b, p: {
            "bi_pctl_na": _rolling(_src_na(b), L, p, lambda w: np.percentile(w, K["PCT"]))
        }
    ),
    "pivots": {
        rule: (
            lambda b, rule=rule: {
                "bi_pivothigh": _pivot_ref(
                    b.high, int(K["PIV_LEFT"]), int(K["PIV_RIGHT"]), rule, 1.0
                ),
                "bi_pivotlow": _pivot_ref(
                    b.low, int(K["PIV_LEFT"]), int(K["PIV_RIGHT"]), rule, -1.0
                ),
            }
        )
        for rule in ("left_strict", "right_strict", "strict", "weak")
    },
    "fixnan": {"last_valid": lambda b: {"bi_fixnan": _cum_last(_src_na(b))}},
    "cum": {"running_total": lambda b: {"bi_cum": _cum_ref(_close(b), "skip")}},
    "cum_na": _policies(
        lambda b, p: {
            "bi_cum_na": _cum_ref(
                _src_na(b), p, dead_at_start=p == "propagate" and getattr(b, "k", 0) > 0
            )
        }
    ),
    "obv": {
        "manual": lambda b: {"bi_obv": _cum_ref(np.sign(_change(_close(b))) * _volume(b), "skip")}
    },
    "pvt": {
        "manual": lambda b: {
            "bi_pvt": _cum_ref(_change(_close(b)) / _prev(_close(b)) * _volume(b), "skip")
        }
    },
}


def _cum_last(x: np.ndarray) -> np.ndarray:
    out = np.full(x.size, np.nan)
    last = math.nan
    for t, v in enumerate(x):
        if not math.isnan(v):
            last = v
        out[t] = last
    return out


EXPECTED: dict[str, str] = {
    "alignment": "aligned",
    "sum": "window",
    "sum_na": ta.DEFAULT_NA_POLICY["sum"],
    "wma": "linear_weights",
    "wma_na": ta.DEFAULT_NA_POLICY["wma"],
    "hma": "half_floor_sqrt_floor",
    "variance": "population",
    "variance_na": ta.DEFAULT_NA_POLICY["variance"],
    "percentile": "inclusive",
    "percentile_na": ta.DEFAULT_NA_POLICY["percentile_linear_interpolation"],
    "pivots": "left_strict",
    "fixnan": "last_valid",
    "cum": "running_total",
    "cum_na": ta.DEFAULT_NA_POLICY["cum"],
    "obv": "manual",
    "pvt": "manual",
}

# Groups whose value at a row is a running total over all earlier bars.
_CUMULATIVE = ("cum", "cum_na", "obv", "pvt")
# Rows that depend on history before a truncated export (longest window: hma 14 + its outer
# wma 3, plus a gap of GAP_WIDTH inside a skipped window), rounded up.
TRUNCATED_EXCLUDE = 25
MIN_COMPARED = 100
# Mostly-`na` columns (a value only on pivot bars): their `na` mask is compared on every row,
# so the minimum count of compared values does not apply.
_SPARSE = ("pivots",)


# ------------------------------------------------------------------ the script on pine.runtime


def runtime_capture(ctx: Context) -> None:
    """``tools/capture_builtins.pine`` on `pine.runtime` (uses `pine.runtime.builtins`)."""
    i = ctx.bar_index.value
    close = ctx.close.value
    gap = i % K["GAP_PERIOD"] >= K["GAP_PERIOD"] - K["GAP_WIDTH"]
    src_na = math.nan if gap else close
    t = ctx.ta
    ctx.plot("bi_bar_index", i)
    ctx.plot("bi_volume", ctx.volume.value)
    ctx.plot("bi_sum", t.sum("sum", close, L))
    ctx.plot("bi_sum_na", t.sum("sumNa", src_na, L))
    ctx.plot("bi_wma", t.wma("wma", close, L))
    ctx.plot("bi_wma_na", t.wma("wmaNa", src_na, L))
    ctx.plot("bi_hma_odd", t.hma("hmaOdd", close, int(K["HMA_ODD"])))
    ctx.plot("bi_hma_nonsquare", t.hma("hmaNs", close, int(K["HMA_NONSQUARE"])))
    ctx.plot("bi_variance", t.variance("var", close, L))
    ctx.plot("bi_variance_na", t.variance("varNa", src_na, L))
    ctx.plot("bi_pctl", t.percentile_linear_interpolation("pctl", close, L, K["PCT"]))
    ctx.plot("bi_pctl_na", t.percentile_linear_interpolation("pctlNa", src_na, L, K["PCT"]))
    left, right = int(K["PIV_LEFT"]), int(K["PIV_RIGHT"])
    ctx.plot("bi_pivothigh", t.pivothigh("ph", ctx.high.value, left, right))
    ctx.plot("bi_pivotlow", t.pivotlow("pl", ctx.low.value, left, right))
    ctx.plot("bi_fixnan", t.fixnan("fix", src_na))
    ctx.plot("bi_cum", t.cum("cum", close))
    ctx.plot("bi_cum_na", t.cum("cumNa", src_na))
    ctx.plot("bi_obv", t.obv("obv"))
    ctx.plot("bi_pvt", t.pvt("pvt"))


# ------------------------------------------------------------------ deciding


def _anchor(v: np.ndarray, pine: np.ndarray, start: int) -> np.ndarray:
    """Shift a running total so it equals the export at the first row ``>= start`` where both
    have a value: the total over unseen bars is unknown, every later increment is checked."""
    both = np.flatnonzero(~np.isnan(v[start:]) & ~np.isnan(pine[start:]))
    if not both.size:
        return v
    r = start + int(both[0])
    return v + (pine[r] - v[r])


def _input_errors(b) -> dict[str, object]:
    """Per column: the input-rounding allowance (`pine.sensitivity`) as a float Lipschitz
    constant or a callable ``(bars, δ)``. Running totals use the full-history bound, which
    also bounds an anchored comparison (the anchor removes the error accumulated before it).
    Pivots have none: they are discontinuous in their inputs."""
    close, src = _close(b), _src_na(b)
    vol = _volume(b)
    return {
        "bi_bar_index": 0.0,
        "bi_sum": sens.sum_lipschitz(L),
        "bi_sum_na": sens.sum_lipschitz(L),
        "bi_wma": sens.LIPSCHITZ["wma"],
        "bi_wma_na": sens.LIPSCHITZ["wma"],
        "bi_hma_odd": sens.LIPSCHITZ["hma"],
        "bi_hma_nonsquare": sens.LIPSCHITZ["hma"],
        # The sample bound is the larger one, so it covers both hypotheses.
        "bi_variance": lambda bb, d: sens.variance_input_error(close, L, d, biased=False),
        "bi_variance_na": lambda bb, d: sens.variance_input_error(src, L, d, biased=False),
        "bi_pctl": sens.LIPSCHITZ["percentile_linear_interpolation"],
        "bi_pctl_na": sens.LIPSCHITZ["percentile_linear_interpolation"],
        "bi_fixnan": sens.LIPSCHITZ["fixnan"],
        "bi_cum": lambda bb, d: sens.cum_input_error(close, d),
        "bi_cum_na": lambda bb, d: sens.cum_input_error(src, d),
        "bi_obv": lambda bb, d: sens.obv_input_error(close, vol, d),
        "bi_pvt": lambda bb, d: sens.pvt_input_error(close, vol, d),
    }


def decide(export: TradingViewExport, input_rounding: float | None = None) -> list[GroupDecision]:
    """For each group, the hypotheses whose columns all match the export.

    ``input_rounding``: half-width of the rounding of the exported OHLC (§5.1); each column's
    tolerance then grows by its bound from `pine.sensitivity`. Pivots get no allowance."""
    k = bar_offset(export.columns, "bi_bar_index") or 0
    b = _At(export.bars, k, export.columns)
    n = len(export.bars)
    exclude = min(TRUNCATED_EXCLUDE, n) if k else 0
    errors = _input_errors(b)
    out = []
    for group, hyps in GROUPS.items():
        ex = 0 if group == "alignment" else exclude
        reports = {}
        for name, hyp in hyps.items():
            cols = hyp(b)
            if k and group in _CUMULATIVE:
                cols = {c: _anchor(v, export.columns[c], ex) for c, v in cols.items()}
            specs = {
                c: PortSpec(
                    lambda bb, v=v: v,
                    kind="unbounded",
                    tolerance=tolerance_for("unbounded", export.observed_decimals.get(c)),
                    mode="convergence" if ex else "aligned",
                    exclude=ex,
                    min_compared=MIN_COMPARED if ex and group not in _SPARSE else 0,
                    input_error=errors.get(c),
                )
                for c, v in cols.items()
            }
            rounding = None if group == "pivots" else input_rounding
            reports[name] = check_export(export, specs, input_rounding=rounding)
        matching = tuple(h for h, r in reports.items() if r.passed)
        note = ""
        if group in ("obv", "pvt") and not np.isfinite(_volume(b)).any():
            note = "no volume on this symbol; obv/pvt are na"
        elif group == "pivots" and input_rounding is not None:
            note = "no input-rounding allowance: rounding can create or remove a pivot"
        elif k and group in _CUMULATIVE:
            note = "running total anchored on the export (unseen history); increments checked"
        out.append(GroupDecision(group, EXPECTED[group], matching, reports, str(ex), note))
    return out


def summary(decisions: list[GroupDecision]) -> str:
    return _summary(decisions, title="Built-ins (batch F)")
