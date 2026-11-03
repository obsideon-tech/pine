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

Checking `pine.ta` against TradingView (the step that turns "unverified" into "verified"): add
`tools/reference_capture.pine` to a chart, export chart data, write a manifest from
`tools/reference_capture.example.yaml`, then:

```bash
uv run python -m pine.reference EXPORT.csv --manifest EXPORT.yaml
```

It reports aligned-mode parity for every core built-in and which `na`-handling hypothesis
TradingView follows. See [`PINE_PLAN.md` §10](PINE_PLAN.md#10-review-2-2026-11-03-next-steps).

Distribution name `pine-parity`, import name `pine`. Every fixture CSV needs a sibling YAML
manifest describing how it was produced; see `src/pine/manifest.py`.

## Known gaps

- **No external verification yet.** Every built-in is checked against closed-form cases and
  property tests derived from the same reading of the Pine manual; none against TradingView
  output. Treat results as unverified.
- **`na` inside windows** may be handled differently by TradingView. Three hypotheses are
  implemented (`na_policy`); the default `propagate` is unconfirmed until a capture export
  decides. Gap-free data is unaffected.
- **Rounded inputs** are accounted for only when asked (`input_rounding=` / `--inputs-rounded`):
  the allowance is provably sufficient but widens tolerances, so it is off by default.
- **Lint** traces only simple variables (assigned once at the top level) and sees alert gating
  only in enclosing `if` blocks, not inside called functions; its precision on real scripts is
  unmeasured. Parity reports are Markdown or JSON (`to_json()`,
  `--format json`); HTML is deferred.

Ranked next steps: [`PINE_PLAN.md` §10](PINE_PLAN.md#10-review-2-2026-11-03-next-steps).

## What "matches" will mean

A port agrees with Pine when, on the same bars from the same first loaded bar: the pattern of
`na` values is identical; every numeric value is within a stated per-column tolerance
(`atol + rtol·|pine|`); and boolean outputs are identical except at near-ties, which are counted
and reported. Comparison starts at the first bar where Pine produces a value, so seeding errors
are not hidden by a warm-up exclusion. Every report also states its minimum detectable error:
the smallest constant bias (and, for EMA/RMA-type filters, the smallest seed error) that the
tolerances in force would have caught. The full specification is in
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
