"""How much input rounding can move each built-in's output (PINE_PLAN.md §5.1, §10 A3).

If an export prints OHLC with ``d`` decimals while TradingView computed on unrounded prices,
every input carries an error ``|e| <= δ = 0.5 * 10**-d``. A correct port fed the rounded
inputs then differs from the reference by at most the bound below, in the sup norm:

==================  =========================  ==============================================
built-in            bound on ``|Δ output|``    argument
==================  =========================  ==============================================
sma, ema, rma       ``δ``                      non-negative weights summing to 1 (incl. seed)
highest, lowest     ``δ``                      max/min are 1-Lipschitz
stdev (population)  ``δ``                      ``σ = ‖P x‖/√n``, ``P`` the centring projection
stdev (sample)      ``√(n/(n-1))·δ``           population bound times the denominator ratio
tr, atr             ``2δ``                     each term is a difference of two inputs; max
                                               and RMA are 1-Lipschitz
rsi                 ``min(100, 200δ/(U'+D'))``  per bar; see `rsi_input_error`
wma, percentile,    ``δ``                      weights non-negative summing to 1; order
fixnan                                         statistics are 1-Lipschitz; fixnan copies
math.sum            ``length·δ``               sum of ``length`` terms
hma                 ``3δ``                     ``2·wma - wma`` moves by at most ``3δ``; the
                                               outer wma is 1-Lipschitz
variance            ``(2σ' + cδ)·cδ``          per bar, ``c = √(n/(n-1))`` if sample else 1;
                                               see `variance_input_error`
cum                 ``m·δ``                    per bar, ``m`` = values summed so far
obv                 ``Σ 2·volume``             over bars with ``|Δclose'| <= 2δ`` (the sign
                                               can flip there); volume assumed exact
pvt                 ``Σ v·δ(p'+|c'|)/((p'-δ)p')``  per term, ``c'``/``p'`` the port's close
                                               and previous close; see `pvt_input_error`
pivothigh/low       none                       discontinuous: rounding can create or remove a
                                               pivot; checked without an allowance
==================  =========================  ==============================================

``stdev`` zeroes deviations below 1e-10, which can add ~1e-10; the default ``atol`` absorbs it.
These bounds hold for the ``propagate`` na policy and, since the skip policies apply the same
weights to a different selection of bars, for those too.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from pine import ta

__all__ = [
    "LIPSCHITZ",
    "input_rounding",
    "stdev_lipschitz",
    "sum_lipschitz",
    "rsi_input_error",
    "variance_input_error",
    "cum_input_error",
    "obv_input_error",
    "pvt_input_error",
]

LIPSCHITZ: dict[str, float] = {
    "sma": 1.0,
    "ema": 1.0,
    "rma": 1.0,
    "highest": 1.0,
    "lowest": 1.0,
    "stdev": 1.0,
    "tr": 2.0,
    "atr": 2.0,
    "wma": 1.0,
    "hma": 3.0,
    "percentile_linear_interpolation": 1.0,
    "fixnan": 1.0,
}


def input_rounding(decimals: int) -> float:
    """Half-width of rounding to ``decimals`` places."""
    if decimals < 0:
        raise ValueError("decimals must be non-negative")
    return 0.5 * 10.0 ** (-decimals)


def stdev_lipschitz(length: int, biased: bool = True) -> float:
    if biased:
        return 1.0
    if length < 2:
        raise ValueError("unbiased stdev needs length >= 2")
    return math.sqrt(length / (length - 1))


def rsi_input_error(
    source: ArrayLike, length: int, delta: float, na_policy: ta.NaPolicy | None = None
) -> np.ndarray:
    """Per-bar bound on ``|RSI(x) - RSI(x')|`` for inputs within ``±delta`` of each other.

    Derivation: each change ``x[t] - x[t-1]`` moves by at most ``2δ``, so ``u`` and ``d``
    (1-Lipschitz clips of it) and their RMAs ``U``, ``D`` each move by at most ``2δ``. With
    ``RSI = 100·U/(U+D)``::

        |RSI' - RSI| = 100·|ΔU·D - U·ΔD| / ((U'+D')(U+D)) <= 200δ / (U'+D')

    where ``U', D'`` are computed from ``source`` (the port's side), so the bound needs no
    knowledge of the unrounded series. RSI lies in [0, 100], hence the cap at 100, which also
    covers ``U'+D' = 0`` and the ``down == 0``/``up == 0`` branches (they agree with the
    continuous formula except at 0/0). Unlike the other built-ins there is no useful global
    bound: in a quiet market ``U'+D'`` is small and one tick of rounding moves RSI by points.
    """
    if delta < 0:
        raise ValueError("delta must be non-negative")
    up, down = ta.rsi_components(source, length, na_policy)
    total = up + down
    with np.errstate(divide="ignore", invalid="ignore"):
        bound = np.where(total > 0, 200.0 * delta / total, 100.0)
    bound = np.minimum(bound, 100.0)
    return np.where(np.isnan(total), np.nan, bound)


def sum_lipschitz(length: int) -> float:
    """``math.sum``: each of the ``length`` terms moves by at most ``δ``."""
    if length < 1:
        raise ValueError("length must be >= 1")
    return float(length)


def _delta(delta: float) -> None:
    if delta < 0:
        raise ValueError("delta must be non-negative")


def variance_input_error(
    source: ArrayLike,
    length: int,
    delta: float,
    biased: bool = True,
    na_policy: ta.NaPolicy | None = None,
) -> np.ndarray:
    """Per-bar bound on ``|var(x) - var(x')|`` for inputs within ``±delta``.

    With ``P`` the centring projection and ``k = n`` (population) or ``n - 1`` (sample),
    ``var = ‖Px‖²/k``, so ``|var - var'| <= (‖Px‖ + ‖Px'‖)·‖PΔ‖/k`` and ``‖PΔ‖ <= √n·δ``.
    In terms of standard deviations, ``|Δvar| <= (σ + σ')·cδ`` with ``c = √(n/k)``, and
    ``σ <= σ' + cδ`` (the stdev bound), giving ``(2σ' + cδ)·cδ`` from the port's side alone.
    Variance is not globally Lipschitz: the bound grows with the window's spread.
    """
    _delta(delta)
    c = 1.0 if biased else stdev_lipschitz(length, biased=False)
    sigma = np.sqrt(ta.variance(source, length, biased, na_policy))
    return (2.0 * sigma + c * delta) * c * delta


def _count_valid(x: np.ndarray) -> np.ndarray:
    return np.cumsum(~np.isnan(x)).astype(float)


def cum_input_error(source: ArrayLike, delta: float) -> np.ndarray:
    """Per-bar bound for ``ta.cum``: ``δ`` times the number of values summed so far. It
    grows without limit, so on long histories rounded inputs leave little detection power;
    anchored comparisons (`pine.builtin_capture`) only need the count since the anchor."""
    _delta(delta)
    return delta * _count_valid(np.asarray(source, dtype=float))


def obv_input_error(close: ArrayLike, volume: ArrayLike, delta: float) -> np.ndarray:
    """Per-bar bound for ``ta.obv``: a term ``sign(Δclose)·volume`` can change only where the
    port's ``|Δclose'| <= 2δ`` (rounding moves a change by at most ``2δ``), and then by at
    most ``2·volume`` (sign −1 ↔ +1). Volume is assumed exact."""
    _delta(delta)
    c = np.asarray(close, dtype=float)
    v = np.abs(np.asarray(volume, dtype=float))
    ch = np.concatenate([[np.nan], np.diff(c)])
    # Float slack: a change moved by exactly 2δ can compute an ulp above it.
    at_risk = np.abs(ch) <= 2 * delta * (1 + 1e-9) + 1e-12 * np.abs(c)
    return np.cumsum(np.where(at_risk, 2.0 * np.nan_to_num(v), 0.0))


def pvt_input_error(close: ArrayLike, volume: ArrayLike, delta: float) -> np.ndarray:
    """Per-bar bound for ``ta.pvt`` (volume assumed exact). For one term
    ``v·(c/p - 1)`` with ``|c - c'|, |p - p'| <= δ`` and ``p >= p' - δ > 0``::

        |c/p - c'/p'| = |(c - c')p' + c'(p' - p)| / (p·p') <= δ(p' + |c'|) / ((p' - δ)·p')

    Terms where ``p' <= δ`` (price within rounding of zero) have no bound (``inf``)."""
    _delta(delta)
    c = np.asarray(close, dtype=float)
    v = np.abs(np.asarray(volume, dtype=float))
    p = np.concatenate([[np.nan], c[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(p > delta, v * delta * (p + np.abs(c)) / ((p - delta) * p), np.inf)
    term = np.where(np.isnan(p), 0.0, term)
    return np.cumsum(term)
