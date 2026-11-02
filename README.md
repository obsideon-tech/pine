# pine
Port TradingView indicators, with a check that the Python output matches Pine bar for bar before you trust it.

**Status:** pre-alpha. Done: bar schema, TradingView CSV loader, fixture manifests, parity
harness with mutation suite, and `ta.sma`, `ema`, `rma`, `rsi`, `stdev`, `highest`, `lowest`,
`tr`, `atr`, `crossover`, `crossunder`, and the `pine-lint` repainting check. The built-ins
pass closed-form and property tests but have **not yet been checked against TradingView
output**; do not rely on them until they have. See [`PINE_PLAN.md`](PINE_PLAN.md).

```bash
uv sync
```

```bash
uv run pytest
```

```python
from pine.io import load_tradingview_csv

exp = load_tradingview_csv("chart.csv", timeframe="60", session_tz="America/New_York")
exp.bars.close  # float64, read-only
exp.columns["RSI"]  # plotted indicator columns by title
exp.observed_decimals  # printed precision per column (sets the tolerance floor)
```

```python
from pine import ta
from pine.parity import PortSpec, check_export

report = check_export(
    exp,
    {
        "RSI": PortSpec(fn=lambda b: ta.rsi(b.close, 14), kind="bounded"),
    },
)
print(report.to_markdown())  # na-mask mismatches, first divergent bar, error metrics
```

Static repainting check:

```bash
uv run pine-lint script.pine
```

```bash
uv run pine-lint script.pine --strict --format json
```

Each finding says whether the problem is established (`definite`) or whether safety could not
be proven (`possible`). Rules and their known blind spots are in
[`PINE_PLAN.md` §6](PINE_PLAN.md#6-lint-soundness).

Distribution name `pine-parity`, import name `pine`. Every fixture CSV needs a sibling YAML
manifest describing how it was produced; see `src/pine/manifest.py`.

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
