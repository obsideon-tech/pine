# pine
Port TradingView indicators, with a check that the Python output matches Pine bar for bar before you trust it.

**Status:** pre-alpha. Done: bar schema, TradingView CSV loader, fixture manifests, parity
harness with mutation suite, `ta.sma`, `ema`, `rma`, `rsi`, `stdev`, `highest`, `lowest`, `tr`,
`atr`, `crossover`, `crossunder`, `change`, `barssince`, `valuewhen`, `supertrend`, the
`pine.runtime` bar-by-bar execution model, and the `pine-lint` repainting check. The eleven
core built-ins (`sma` … `crossunder`) and the runtime's semantics **match TradingView on two
real exports**, including seeding; `change`, `barssince`, `valuewhen` and `supertrend` match
through the semantics capture. Anything else is checked only against closed-form and property
tests so far. See [`PINE_PLAN.md` §12](PINE_PLAN.md#12-evidence-log-external-results).

```bash
uv sync
```

```bash
uv run pytest
```

CI runs property tests derandomized (`HYPOTHESIS_PROFILE=ci`) with a branch-coverage floor; a
weekly job runs them with random seeds and 10× the examples (`HYPOTHESIS_PROFILE=thorough`).

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

### Strategies

`pine.strategy` emulates TradingView's backtest engine for ports written on `pine.runtime`
(`run(script, bars, strategy=StrategyConfig())`, then `ctx.strategy.entry/close/exit` and
`position_size`, `netprofit`, `equity`, …). First increment: market orders, reversals and
stop/limit brackets with TradingView's intrabar path assumption; checked against TradingView by
`tools/capture_strategy.pine` (pending an export). Plan: [`PINE_PLAN.md` §13](PINE_PLAN.md#13-strategy-parity-strategy).

### Run the TradingView capture

This is the step that turns "unverified" into "verified". Put `tools/reference_capture.pine`
and `tools/capture_semantics.pine` on one chart, export the chart data, then run (with your
own file name and chart settings):

```bash
uv run python -m pine.capture manifest ~/Downloads/aapl_1d.csv --symbol NASDAQ:AAPL --timeframe 1D --timezone America/New_York
```

```bash
uv run python -m pine.capture check ~/Downloads/aapl_1d.csv --out ~/Downloads/aapl_1d_report.md
```

The report covers parity for every core built-in, the `na`-handling decision, and the Pine
execution semantics `pine.runtime` assumes. Step-by-step guide, including where the menus are
and how to read the report: [`docs/CAPTURE.md`](docs/CAPTURE.md).

Porting stateful or conditional logic: write the port in Pine's shape on `pine.runtime`, which
runs the script bar by bar with Pine's execution model (history per execution, `var`, separate
state per call site, built-ins that advance only when called):

```python
from pine.runtime import run


def script(ctx):
    fast = ctx.ta.ema("fast", ctx.close.value, 9)
    slow = ctx.ta.ema("slow", ctx.close.value, 21)
    bars_in = ctx.var("bars_in", 0)
    if ctx.ta.crossover("up", fast, slow):
        bars_in.set(0)
    else:
        bars_in.set(bars_in.value + 1)
    ctx.plot("bars_in", bars_in.value)


result = run(script, exp.bars)
```

The semantics `pine.runtime` assumes are checked by the second capture script (see
[`docs/CAPTURE.md`](docs/CAPTURE.md)).

Distribution name `pine-parity`, import name `pine`. Every fixture CSV needs a sibling YAML
manifest describing how it was produced; see `src/pine/manifest.py`.

## Known gaps

- **Verified against TradingView so far** (two NFLX exports, daily and weekly; see
  [`PINE_PLAN.md` §12](PINE_PLAN.md#12-evidence-log-external-results)): the eleven core
  built-ins including seeding, `na` handling, the runtime's execution model and scalar rules,
  and the three reference ports. Not yet: prices below 1 or above 10 000, intraday charts,
  and every built-in outside the core eleven.
- **`na` inside windows** follows the observed behaviour by default: `sma`, `stdev`,
  `highest`, `lowest` skip `na`; `ema`, `rma`, `rsi` carry state across `na` and output `na`
  on the `na` bar (`pine.ta.DEFAULT_NA_POLICY`; override with `na_policy=`). Observed on one
  export so far.
- **Rounded inputs** are accounted for only when asked (`input_rounding=` / `--inputs-rounded`):
  the allowance is provably sufficient but widens tolerances, so it is off by default.
- **Lint** traces only simple variables (assigned once at the top level) and sees alert gating
  only in enclosing `if` blocks, not inside called functions; its precision on real scripts is
  unmeasured. Parity reports are Markdown or JSON (`to_json()`,
  `--format json`); HTML is deferred.

- **Complex indicators are not yet supported with a parity guarantee.** `pine.runtime` now
  provides Pine's bar-by-bar execution model and scalar rules, confirmed by one
  `capture_semantics.pine` export (§12). Still missing: sessions,
  `request.security`, and most built-ins beyond the ones above. Parity claims cover confirmed historical bars only; live/realtime behaviour cannot
  be checked from exports.

Ranked next steps: [`PINE_PLAN.md` §10](PINE_PLAN.md#10-review-2-2026-11-03-next-steps)
(built-ins and harness) and [§11](PINE_PLAN.md#11-review-3-2026-11-03-from-built-ins-to-complex-indicators)
(complex indicators: execution model, sessions, multi-timeframe, port acceptance).

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
