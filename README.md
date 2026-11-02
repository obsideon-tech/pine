# pine
Port TradingView indicators, with a check that the Python output matches Pine bar for bar before you trust it.

**Status:** pre-alpha — plan only, no code yet. See [`PINE_PLAN.md`](PINE_PLAN.md).

## What "matches" will mean

A port agrees with Pine when, on the same bars from the same first loaded bar: the pattern of
`na` values is identical; every numeric value is within a stated per-column tolerance
(`atol + rtol·|pine|`); and boolean outputs are identical except at near-ties, which are counted
and reported. Comparison starts at the first bar where Pine produces a value, so seeding errors
are not hidden by a warm-up exclusion. The full specification is in
[`PINE_PLAN.md` §5](PINE_PLAN.md#5-analytical-framework-parity-spec).

`pine.lint` finds known repainting patterns. A clean lint result does not prove that a script
does not repaint.

## Data

Whether TradingView chart exports may be committed to this repository is not yet confirmed.
Until it is, fixtures are synthetic bars or expected values committed as numbers only. The
answer will be recorded here.

## Disclaimer

Not affiliated with or endorsed by TradingView. "TradingView" and "Pine Script" are used only
to describe compatibility.
