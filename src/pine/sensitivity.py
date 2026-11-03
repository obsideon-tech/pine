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

__all__ = ["LIPSCHITZ", "input_rounding", "stdev_lipschitz", "rsi_input_error"]

LIPSCHITZ: dict[str, float] = {
    "sma": 1.0,
    "ema": 1.0,
    "rma": 1.0,
    "highest": 1.0,
    "lowest": 1.0,
    "stdev": 1.0,
    "tr": 2.0,
    "atr": 2.0,
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
    source: ArrayLike, length: int, delta: float, na_policy: ta.NaPolicy = "propagate"
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
