# pine
Port TradingView indicators, with a check that the Python output matches Pine bar for bar before you trust it.

**Status:** pre-alpha. Built: bar schema, TradingView CSV loader, fixture manifests, parity
harness with mutation suites, 26 `ta.*` built-ins in `pine.ta` and in `pine.runtime` (the
bar-by-bar execution model, including `for` loops), the strategy emulator's first increment,
`pine.port.check_port` for whole ports, and the `pine-lint` repainting check. What has been
**checked against TradingView** is listed per item under
[Verification status](#verification-status), generated from the verification registry; the
evidence behind each entry is in [`PINE_PLAN.md` §12](PINE_PLAN.md#12-evidence-log-external-results).

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
stop/limit brackets with TradingView's intrabar path assumption and tick rounding, **verified
bar for bar against TradingView** on a 2 678-bar DOGE backtest (`tools/capture_strategy.pine`);
gap fills still to observe. Plan: [`PINE_PLAN.md` §13](PINE_PLAN.md#13-strategy-parity-strategy).

### Run the TradingView capture

This is the step that turns "unverified" into "verified". Put `tools/reference_capture.pine`,
`tools/capture_semantics.pine` and `tools/capture_builtins.pine` (plus
`tools/capture_strategy.pine` for the backtest engine) on one chart, export the chart data,
then run (with your own file name and chart settings):

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
[`docs/CAPTURE.md`](docs/CAPTURE.md)). Loops: `for i in ctx.loop("k", range(1, 4)):`; each
call site in the body keeps one state across iterations (the expected Pine behaviour, probed
by the semantics capture).

### Checking a port

Add debug plots of the intermediate series to a copy of the Pine script, export, and compare
each stage with the same-named `ctx.plot` of the port. The report names the stage that
diverges first, so you fix the cause rather than its downstream effects:

```python
from pine.port import Stage, check_port

report = check_port(
    exp, script, [Stage("dbg_smooth"), Stage("dbg_trend"), Stage("signal", kind="discrete")]
)
print(report.to_markdown())  # root-cause stage, first divergence per stage, full detail
```

Procedure: [`docs/CAPTURE.md`](docs/CAPTURE.md#checking-a-port). `port_manifest` /
`write_port_manifest` record what a port passed against (Pine source hash, inputs, export,
per-stage result), and `verify_port_manifest` says when that record stops applying. Ports of private indicators
belong in a private repository that depends on `pine`; nothing in this repository needs their
source.

Distribution name `pine-parity`, import name `pine`. Every fixture CSV needs a sibling YAML
manifest describing how it was produced; see `src/pine/manifest.py`.

## Verification status

<!-- verification-table:start (generated by python -m pine.registry) -->

Generated from [`src/pine/verification.yaml`](src/pine/verification.yaml); **verified** means at least one TradingView export confirmed it.

Exports: `nflx_1d` BATS:NFLX 1D (from bar 3116); `nflx_1w` BATS:NFLX 1W (from bar 0); `doge_1d` BINANCE:DOGEUSDT 1D (from bar 0); `btc_1d` BITSTAMP:BTCUSD 1D (from bar 2662); `aapl_60` BATS:AAPL 60 (from bar 17845); `doge_1d_strategy` BINANCE:DOGEUSDT 1D (from bar 0).

| built-in | runtime | status | TradingView exports | rounding bound | bug tests |
|---|---|---|---|---|---|
| `ta.sma` | yes | verified | nflx_1d, nflx_1w, doge_1d, btc_1d, aapl_60 | yes | yes |
| `ta.ema` | yes | verified | nflx_1d, nflx_1w, doge_1d, btc_1d, aapl_60 | yes | yes |
| `ta.rma` | yes | verified | nflx_1d, nflx_1w, doge_1d, btc_1d, aapl_60 | yes | yes |
| `ta.rsi` | yes | verified | nflx_1d, nflx_1w, doge_1d, btc_1d, aapl_60 | yes | yes |
| `ta.stdev` | yes | verified | nflx_1d, nflx_1w, doge_1d, btc_1d, aapl_60 | yes | yes |
| `ta.highest` | yes | verified | nflx_1d, nflx_1w, doge_1d, btc_1d, aapl_60 | yes | yes |
| `ta.lowest` | yes | verified | nflx_1d, nflx_1w, doge_1d, btc_1d, aapl_60 | yes | yes |
| `ta.tr` | yes | verified | nflx_1d, nflx_1w, doge_1d, btc_1d, aapl_60 | yes | yes |
| `ta.atr` | yes | verified | nflx_1d, nflx_1w, doge_1d, btc_1d, aapl_60 | yes | yes |
| `ta.crossover` | yes | verified | nflx_1d, nflx_1w, doge_1d, btc_1d, aapl_60 | none | yes |
| `ta.crossunder` | yes | verified | nflx_1d, nflx_1w, doge_1d, btc_1d, aapl_60 | none | yes |
| `ta.change` | yes | verified | nflx_1d, nflx_1w, doge_1d, btc_1d, aapl_60 | none | yes |
| `ta.barssince` | yes | verified | nflx_1d, nflx_1w, doge_1d, btc_1d, aapl_60 | none | yes |
| `ta.valuewhen` | yes | verified | nflx_1d, nflx_1w, doge_1d, btc_1d, aapl_60 | none | yes |
| `ta.supertrend` | — | verified | nflx_1d, nflx_1w, doge_1d, btc_1d, aapl_60 | none | yes |
| `math.sum` | yes | capture ready | — | yes | yes |
| `ta.wma` | yes | capture ready | — | yes | yes |
| `ta.hma` | yes | capture ready | — | yes | yes |
| `ta.variance` | yes | capture ready | — | yes | yes |
| `ta.percentile_linear_interpolation` | yes | capture ready | — | yes | yes |
| `ta.pivothigh` | yes | capture ready | — | none | yes |
| `ta.pivotlow` | yes | capture ready | — | none | yes |
| `fixnan` | yes | capture ready | — | yes | yes |
| `ta.cum` | yes | capture ready | — | yes | yes |
| `ta.obv` | yes | capture ready | — | yes | yes |
| `ta.pvt` | yes | capture ready | — | yes | yes |

| execution rules | verified | capture ready |
|---|---|---|
| semantics | 20 of 22 | 2 |
| builtins | 0 of 16 | 16 |
| strategy | 7 of 8 | 1 |

Awaiting an export: `semantics.loop_builtin`, `semantics.loop_userfn`, `builtins.alignment`, `builtins.sum`, `builtins.sum_na`, `builtins.wma`, `builtins.wma_na`, `builtins.hma`, `builtins.variance`, `builtins.variance_na`, `builtins.percentile`, `builtins.percentile_na`, `builtins.pivots`, `builtins.fixnan`, `builtins.cum`, `builtins.cum_na`, `builtins.obv`, `builtins.pvt`, `strategy.gap_fill`.

<!-- verification-table:end -->

## Known gaps

- **Not covered by any capture yet:** sessions as a feature (an hourly export only shows that
  session bars don't break anything), `request.security`, and built-ins beyond the 26 above.
  Item-by-item status: [Verification status](#verification-status).
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
