"""Parity harness: does a Python port agree with Pine? (PINE_PLAN.md §5)

Order of checks for each numeric column:

1. **na mask** must match exactly (§5.2). A mismatch is a structural failure; tolerance
   never applies to it.
2. **Values**, where both are non-`na`, must satisfy ``|py - pine| <= atol + rtol*|pine|``
   (§5.1).

Two comparison modes (§5.3):

- ``"aligned"`` (default): every bar is compared. Requires Python and Pine to start on the
  same first bar (§5.4). Only this mode certifies seeding.
- ``"convergence"``: the first ``exclude`` bars are skipped, for when Python's history starts
  later than Pine's. Size ``exclude`` with `convergence_bars`. A pass says nothing about
  seeding, and the report says so.

Boolean outputs use `compare_discrete` (§5.5).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from pine.bars import Bars

__all__ = [
    "Tolerance",
    "Kind",
    "Mode",
    "DEFAULT_TOLERANCES",
    "tolerance_for",
    "convergence_bars",
    "ColumnReport",
    "DiscreteReport",
    "ParityReport",
    "PortSpec",
    "compare_series",
    "compare_discrete",
    "near_tie",
    "crossover_excuse",
    "format_atol",
    "check_export",
]

Kind = Literal["price", "bounded", "unbounded"]
Mode = Literal["aligned", "convergence"]


@dataclass(frozen=True)
class Tolerance:
    """``|py - pine| <= atol + rtol * |pine|``. ``atol`` may be a per-bar array (e.g. the
    RSI input-rounding allowance, §5.1); ``rtol`` is a scalar."""

    atol: float | np.ndarray
    rtol: float

    def __post_init__(self) -> None:
        a = np.asarray(self.atol, dtype=np.float64)
        if np.isnan(a).any() or not (np.all(a >= 0) and self.rtol >= 0):
            raise ValueError("atol and rtol must be non-negative (and atol not NaN)")

    def bound(self, reference: np.ndarray) -> np.ndarray:
        return self.atol + self.rtol * np.abs(reference)


# Defaults for full-precision data. Provisional until export precision is measured (§5.1).
DEFAULT_TOLERANCES: dict[str, Tolerance] = {
    "price": Tolerance(atol=1e-10, rtol=1e-9),
    "bounded": Tolerance(atol=1e-8, rtol=0.0),
    "unbounded": Tolerance(atol=1e-10, rtol=1e-9),
}


def tolerance_for(kind: Kind, displayed_decimals: int | None = None) -> Tolerance:
    """Default tolerance for an output class, floored by the reference's printed precision.

    A reference printed with ``d`` decimals carries rounding error up to ``0.5 * 10**-d``,
    so ``atol`` is at least that (with a little slack for float formatting).
    """
    try:
        base = DEFAULT_TOLERANCES[kind]
    except KeyError:
        raise ValueError(
            f"unknown kind {kind!r}; expected one of {list(DEFAULT_TOLERANCES)}"
        ) from None
    if displayed_decimals is None:
        return base
    if displayed_decimals < 0:
        raise ValueError("displayed_decimals must be non-negative")
    floor = 0.5 * 10.0 ** (-displayed_decimals) * (1 + 1e-9)
    return Tolerance(atol=max(base.atol, floor), rtol=base.rtol)


def convergence_bars(alpha: float, tol: float, e0: float) -> int:
    """Bars for an initial-state error ``e0`` to decay below ``tol`` in a filter with
    smoothing ``alpha``: smallest ``n`` with ``e0 * (1 - alpha)**n <= tol`` (§5.3).

    For EMA ``alpha = 2/(length+1)``; for RMA ``alpha = 1/length``. Bound ``e0`` by the
    source's range over the warm-up.
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if tol <= 0:
        raise ValueError("tol must be positive")
    e0 = abs(e0)
    if e0 <= tol or alpha == 1:
        return 0
    return math.ceil(math.log(tol / e0) / math.log(1 - alpha))


def _first(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def _as_1d(name: str, values: ArrayLike, dtype: type = np.float64) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return arr


@dataclass(frozen=True)
class ColumnReport:
    name: str
    kind: str
    mode: Mode
    tolerance: Tolerance
    n_bars: int
    n_excluded: int
    n_compared: int
    """Bars where both values are non-na and inside the compared region."""
    na_mismatches: int
    first_na_mismatch: int | None
    value_failures: int
    first_value_failure: int | None
    max_abs_err: float
    max_abs_err_bar: int | None
    max_rel_err: float
    rmse: float
    within_tol_share: float
    first_divergence_ts: int | None = None
    notes: tuple[str, ...] = ()
    mde_offset: float = math.inf
    """Smallest constant error that would fail: ``min B_t`` over compared bars (§5.7)."""
    mde_seed: float | None = None
    """Smallest seed error a filter with the given ``alpha`` would show (§5.7); ``None`` when
    no ``alpha`` was given."""
    min_compared: int = 0
    """A column comparing fewer bars than this is *inconclusive* and does not pass (guards
    against vacuous passes when convergence exclusions consume a short export)."""

    @property
    def inconclusive(self) -> bool:
        return self.n_compared < self.min_compared

    @property
    def first_divergence(self) -> int | None:
        cands = [b for b in (self.first_na_mismatch, self.first_value_failure) if b is not None]
        return min(cands) if cands else None

    @property
    def divergence_type(self) -> Literal["na", "value"] | None:
        fd = self.first_divergence
        if fd is None:
            return None
        return "na" if fd == self.first_na_mismatch else "value"

    @property
    def passed(self) -> bool:
        return self.na_mismatches == 0 and self.value_failures == 0 and not self.inconclusive


def compare_series(
    py: ArrayLike,
    pine: ArrayLike,
    *,
    name: str = "value",
    kind: Kind = "price",
    tolerance: Tolerance | None = None,
    mode: Mode = "aligned",
    exclude: int = 0,
    ts: ArrayLike | None = None,
    alpha: float | None = None,
    min_compared: int = 0,
) -> ColumnReport:
    """Compare one numeric output column. ``pine`` is the reference.

    ``alpha``: smoothing factor if the column is a seeded recursive filter (EMA ``2/(n+1)``,
    RMA/ATR ``1/n``); enables ``mde_seed``. Not meaningful for RSI, whose seed errors enter
    through the up/down averages rather than in RSI units.
    """
    p = _as_1d("py", py)
    r = _as_1d("pine", pine)
    if p.size != r.size:
        raise ValueError(f"length mismatch: py has {p.size} bars, pine has {r.size}")
    if mode == "aligned" and exclude != 0:
        raise ValueError("aligned mode compares every bar; use mode='convergence' to exclude")
    if mode not in ("aligned", "convergence"):
        raise ValueError(f"unknown mode {mode!r}")
    if not 0 <= exclude <= p.size:
        raise ValueError(f"exclude must be in [0, {p.size}]")
    tol = tolerance or tolerance_for(kind)
    ts_arr = None if ts is None else _as_1d("ts", ts, np.int64)

    region = np.arange(p.size) >= exclude
    na_p, na_r = np.isnan(p), np.isnan(r)
    na_mis = (na_p != na_r) & region
    both = ~na_p & ~na_r & region

    err = np.zeros(p.size)
    err[both] = np.abs(p[both] - r[both])
    bound = np.broadcast_to(tol.bound(np.where(both, r, 0.0)), p.shape)
    fail = both & (err > bound)
    mde_offset, mde_seed = _mde(bound, both, ~na_r, alpha)

    nonzero = both & (r != 0)
    rel = np.zeros(p.size)
    rel[nonzero] = err[nonzero] / np.abs(r[nonzero])

    n_cmp = int(both.sum())
    max_bar = int(np.argmax(np.where(both, err, -1.0))) if n_cmp else None
    notes: list[str] = []
    if n_cmp < min_compared:
        notes.append(
            f"{name}: inconclusive, only {n_cmp} bars compared (need {min_compared}); the "
            "export is too short after exclusions"
        )
    if mode == "convergence":
        notes.append(
            f"convergence mode: first {exclude} bars excluded; a pass does not certify "
            "seeding (PINE_PLAN.md §5.3)"
        )
    report = ColumnReport(
        name=name,
        kind=kind,
        mode=mode,
        tolerance=tol,
        n_bars=int(p.size),
        n_excluded=int(exclude),
        n_compared=n_cmp,
        na_mismatches=int(na_mis.sum()),
        first_na_mismatch=_first(na_mis),
        value_failures=int(fail.sum()),
        first_value_failure=_first(fail),
        max_abs_err=float(err.max()) if n_cmp else 0.0,
        max_abs_err_bar=max_bar,
        max_rel_err=float(rel.max()) if n_cmp else 0.0,
        rmse=float(np.sqrt(np.mean(err[both] ** 2))) if n_cmp else 0.0,
        within_tol_share=float(1 - fail.sum() / n_cmp) if n_cmp else 1.0,
        notes=tuple(notes),
        mde_offset=mde_offset,
        mde_seed=mde_seed,
        min_compared=min_compared,
    )
    fd = report.first_divergence
    if ts_arr is not None and fd is not None:
        object.__setattr__(report, "first_divergence_ts", int(ts_arr[fd]))
    return report


def _mde(
    bound: np.ndarray, compared: np.ndarray, ref_valid: np.ndarray, alpha: float | None
) -> tuple[float, float | None]:
    """Minimum detectable errors (§5.7).

    - offset: a constant error ``c`` fails iff ``c > B_t`` at some compared bar.
    - seed: a seed error ``e0`` at the reference's first valid bar ``w`` appears as
      ``e0 (1-alpha)^(t-w)``; it fails iff ``e0 > min_t B_t / (1-alpha)^(t-w)``.
    """
    if not compared.any():
        return math.inf, (None if alpha is None else math.inf)
    offset = float(np.min(bound[compared]))
    if alpha is None:
        return offset, None
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    w = int(np.flatnonzero(ref_valid)[0])
    t = np.flatnonzero(compared)
    if alpha == 1:
        # The seed error vanishes after the seed bar: detectable only there.
        return offset, float(bound[w]) if compared[w] else math.inf
    with np.errstate(over="ignore"):
        thresholds = bound[t] / (1 - alpha) ** (t - w).astype(float)
    return offset, float(np.min(thresholds))


def _fmt_mde(x: float | None) -> str:
    if x is None:
        return "—"
    return "n/a" if math.isinf(x) else f"{x:.3g}"


def format_atol(atol: float | np.ndarray) -> str:
    if np.ndim(atol) == 0:
        return f"{float(atol):.3g}"
    return f"≤{float(np.max(atol)):.3g} (per bar)"


def near_tie(a: ArrayLike, b: ArrayLike, tolerance: Tolerance) -> np.ndarray:
    """Bars where ``a`` and ``b`` are within tolerance of each other: a crossover between them
    may legitimately flip between implementations (§5.5)."""
    a_, b_ = _as_1d("a", a), _as_1d("b", b)
    with np.errstate(invalid="ignore"):
        return np.abs(a_ - b_) <= tolerance.bound(b_)


def crossover_excuse(a: ArrayLike, b: ArrayLike, tolerance: Tolerance) -> np.ndarray:
    """Excuse mask for ``ta.crossover(a, b)`` / ``ta.crossunder(a, b)`` mismatches (§5.5).

    A crossover at bar ``t`` is ``a[t] > b[t] and a[t-1] <= b[t-1]``: it depends on the
    comparison at *both* bars, so a legitimate flip can be caused by a near-tie at ``t`` or at
    ``t - 1``. `near_tie` alone covers only ``t``. NaN inputs are never excused.
    """
    tie = near_tie(a, b, tolerance)
    ex = tie.copy()
    ex[1:] |= tie[:-1]
    return ex


@dataclass(frozen=True)
class DiscreteReport:
    name: str
    n_bars: int
    mismatches: int
    excused: int
    excused_bars: tuple[int, ...]
    first_unexcused: int | None
    first_unexcused_ts: int | None = None
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.first_unexcused is None


def compare_discrete(
    py: ArrayLike,
    pine: ArrayLike,
    *,
    name: str = "signal",
    excuse: ArrayLike | None = None,
    ts: ArrayLike | None = None,
) -> DiscreteReport:
    """Exact comparison of boolean outputs. A mismatch is excused only where ``excuse`` is
    true (normally from `near_tie` on the continuous inputs); excused bars are reported."""
    p = _as_1d("py", py, bool)
    r = _as_1d("pine", pine, bool)
    if p.size != r.size:
        raise ValueError(f"length mismatch: py has {p.size} bars, pine has {r.size}")
    ex = np.zeros(p.size, bool) if excuse is None else _as_1d("excuse", excuse, bool)
    mis = p != r
    excused = mis & ex
    first = _first(mis & ~ex)
    ts_arr = None if ts is None else _as_1d("ts", ts, np.int64)
    return DiscreteReport(
        name=name,
        n_bars=int(p.size),
        mismatches=int(mis.sum()),
        excused=int(excused.sum()),
        excused_bars=tuple(int(i) for i in np.flatnonzero(excused)),
        first_unexcused=first,
        first_unexcused_ts=None if (ts_arr is None or first is None) else int(ts_arr[first]),
    )


@dataclass(frozen=True)
class PortSpec:
    """How to check one exported column: ``fn(bars) -> values`` and the comparison setup."""

    fn: Callable[[Bars], ArrayLike]
    kind: Kind | Literal["discrete"] = "price"
    mode: Mode = "aligned"
    exclude: int = 0
    tolerance: Tolerance | None = None
    input_error: float | Callable[[Bars, float], ArrayLike] | None = None
    """Output error caused by inputs rounded to ``±δ`` (§5.1, `pine.sensitivity`): a float is
    the sensitivity ``L`` (error ``L·δ``); a callable ``(bars, δ)`` returns per-bar error.
    ``None``: unknown, and `check_export` refuses ``input_rounding`` for this column."""
    alpha: float | None = None
    """Smoothing factor of a seeded recursive filter; enables ``mde_seed`` (§5.7)."""
    excuse: Callable[[Bars], ArrayLike] | None = None
    """``kind="discrete"`` only: bars where a mismatch is excused (e.g. `crossover_excuse`)."""
    min_compared: int = 0
    """See `ColumnReport.min_compared`."""


@dataclass(frozen=True)
class ParityReport:
    columns: tuple[ColumnReport, ...]
    discrete: tuple[DiscreteReport, ...] = field(default=())

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.columns) and all(d.passed for d in self.discrete)

    def to_markdown(self) -> str:
        lines = [
            f"**Parity: {'PASS' if self.passed else 'FAIL'}**",
            "",
            "| column | kind | mode | compared | excluded | na mismatches | first divergence "
            "| max abs err (bar) | max rel err | RMSE | within tol | atol / rtol "
            "| MDE offset / seed | result |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for c in self.columns:
            fd = (
                "—" if c.first_divergence is None else f"{c.first_divergence} ({c.divergence_type})"
            )
            if c.first_divergence_ts is not None:
                fd += f" @ {c.first_divergence_ts}"
            lines.append(
                f"| {c.name} | {c.kind} | {c.mode} | {c.n_compared} | {c.n_excluded} "
                f"| {c.na_mismatches} | {fd} | {c.max_abs_err:.3g} ({c.max_abs_err_bar}) "
                f"| {c.max_rel_err:.3g} | {c.rmse:.3g} | {c.within_tol_share:.2%} "
                f"| {format_atol(c.tolerance.atol)} / {c.tolerance.rtol:.3g} "
                f"| {_fmt_mde(c.mde_offset)} / {_fmt_mde(c.mde_seed)} "
                f"| {'inconclusive' if c.inconclusive else ('pass' if c.passed else 'FAIL')} |"
            )
        for d in self.discrete:
            lines.append(
                f"| {d.name} | discrete | aligned | {d.n_bars} | 0 | — "
                f"| {'—' if d.first_unexcused is None else d.first_unexcused} | — | — | — "
                f"| excused flips: {d.excused} | — | — | {'pass' if d.passed else 'FAIL'} |"
            )
        notes = [n for c in (*self.columns, *self.discrete) for n in c.notes]
        if notes:
            lines += ["", *(f"- {n}" for n in dict.fromkeys(notes))]
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict:
        """JSON-safe dict: per-bar ``atol`` summarised, ``inf`` as ``None``."""

        def num(x):
            return None if x is None or (isinstance(x, float) and math.isinf(x)) else x

        cols = []
        for c in self.columns:
            atol = c.tolerance.atol
            atol_d = (
                float(atol) if np.ndim(atol) == 0 else {"max": float(np.max(atol)), "per_bar": True}
            )
            cols.append(
                {
                    "name": c.name,
                    "kind": c.kind,
                    "mode": c.mode,
                    "passed": c.passed,
                    "n_bars": c.n_bars,
                    "n_compared": c.n_compared,
                    "n_excluded": c.n_excluded,
                    "na_mismatches": c.na_mismatches,
                    "first_divergence": c.first_divergence,
                    "divergence_type": c.divergence_type,
                    "first_divergence_ts": c.first_divergence_ts,
                    "max_abs_err": c.max_abs_err,
                    "max_abs_err_bar": c.max_abs_err_bar,
                    "max_rel_err": c.max_rel_err,
                    "rmse": c.rmse,
                    "within_tol_share": c.within_tol_share,
                    "atol": atol_d,
                    "rtol": c.tolerance.rtol,
                    "mde_offset": num(c.mde_offset),
                    "mde_seed": num(c.mde_seed),
                    "inconclusive": c.inconclusive,
                    "notes": list(c.notes),
                }
            )
        disc = [
            {
                "name": d.name,
                "passed": d.passed,
                "n_bars": d.n_bars,
                "mismatches": d.mismatches,
                "excused": d.excused,
                "excused_bars": list(d.excused_bars),
                "first_unexcused": d.first_unexcused,
                "first_unexcused_ts": d.first_unexcused_ts,
                "notes": list(d.notes),
            }
            for d in self.discrete
        ]
        return {"passed": self.passed, "columns": cols, "discrete": disc}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)


def check_export(
    export, specs: Mapping[str, PortSpec], input_rounding: float | None = None
) -> ParityReport:
    """Run each port on ``export.bars`` and compare with the exported column of that name.

    ``export`` is a `pine.io.TradingViewExport`. Unless a spec gives a tolerance, the default
    for its kind is floored by the column's observed printed precision.

    ``input_rounding``: half-width ``δ`` of the rounding applied to the exported OHLC inputs
    (`pine.sensitivity.input_rounding`). ``None`` (default) means inputs are exact. When set,
    each column's ``atol`` grows by its spec's ``input_error``. This widens tolerances and so
    raises the smallest detectable defect: use it only when prices are rounded beyond the
    instrument's tick (e.g. dividend-adjusted data), §5.1.
    """
    missing: Sequence[str] = [c for c in specs if c not in export.columns]
    if missing:
        raise KeyError(f"columns not in export: {missing}; available: {list(export.columns)}")
    if input_rounding is not None and input_rounding < 0:
        raise ValueError("input_rounding must be non-negative")
    reports, discrete = [], []
    for col, spec in specs.items():
        if spec.kind == "discrete":
            discrete.append(_check_discrete(export, col, spec))
            continue
        tol = spec.tolerance or tolerance_for(spec.kind, export.observed_decimals.get(col))
        note = None
        if input_rounding is not None:
            if spec.input_error is None:
                raise ValueError(
                    f"column {col!r}: input_rounding given but the spec has no input_error"
                )
            if callable(spec.input_error):
                extra = np.asarray(spec.input_error(export.bars, input_rounding), np.float64)
                extra = np.nan_to_num(extra, nan=0.0)  # na bars are compared on the na mask
            else:
                extra = float(spec.input_error) * input_rounding
            tol = Tolerance(atol=tol.atol + extra, rtol=tol.rtol)
            note = f"{col}: inputs assumed rounded to ±{input_rounding:.3g}; allowance added"
        rep = compare_series(
            spec.fn(export.bars),
            export.columns[col],
            name=col,
            kind=spec.kind,
            tolerance=tol,
            mode=spec.mode,
            exclude=spec.exclude,
            ts=export.bars.ts,
            alpha=spec.alpha,
            min_compared=spec.min_compared,
        )
        if note:
            rep = replace(rep, notes=(*rep.notes, note))
        reports.append(rep)
    return ParityReport(columns=tuple(reports), discrete=tuple(discrete))


def _check_discrete(export, col: str, spec: PortSpec) -> DiscreteReport:
    """Exported 0/1 column vs a boolean port (§5.5). NaN in the export reads as False."""
    raw = np.asarray(export.columns[col], dtype=np.float64)
    nan = np.isnan(raw)
    pine = ~nan & (raw != 0)
    py = np.asarray(spec.fn(export.bars))
    py = py.astype(bool) if py.dtype == bool else (~np.isnan(py.astype(float)) & (py != 0))
    excuse = None if spec.excuse is None else spec.excuse(export.bars)
    rep = compare_discrete(py, pine, name=col, excuse=excuse, ts=export.bars.ts)
    if nan.any():
        rep = replace(rep, notes=(f"{col}: {int(nan.sum())} na value(s) in export read as false",))
    return rep
