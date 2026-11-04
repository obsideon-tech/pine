# Running the TradingView capture

This is the step that turns `pine`'s checks from "consistent with itself" into "matches
TradingView". You put the capture scripts on one chart, export the chart data once, and run
two commands. It takes about ten minutes. Background: `PINE_PLAN.md` §10 A1/A2, §11 E1/F, §13.

You need:

- a TradingView account that can **export chart data**. I believe this needs a paid plan, but
  have not verified it; if the menu item in step 4 is missing or locked, that is why.
- this repository checked out, with `uv sync` run once.

TradingView's menus move from time to time. If a label below doesn't match exactly, look for
the nearest equivalent.

## 1. Pick a chart

Open a chart for the first row of the coverage matrix (`PINE_PLAN.md` §10 A1). A liquid US
stock on the daily timeframe is a good start, e.g. `NASDAQ:AAPL`, `1D`. Use normal candles,
not Heikin Ashi.

Note three things for step 5:

- **Symbol**, exactly as the chart shows it (`NASDAQ:AAPL`).
- **Timeframe**, in Pine's spelling: `1D`, `1W`, `60` (for 1h), `240` (for 4h), `15`, and so on.
- **Exchange time zone**, as an IANA name: `America/New_York` for US stocks, `Europe/London`,
  `Asia/Tokyo`, `UTC` for most crypto. The symbol's info panel shows the exchange.

Also check whether **ADJ** ("Adjust data for dividends") is switched on at the bottom of the
chart, and whether **extended hours** are shown. You'll pass both in step 5.

## 2. Add the scripts

For each of `tools/reference_capture.pine`, `tools/capture_semantics.pine`,
`tools/capture_builtins.pine` (the batch-F built-ins: `wma`, `hma`, pivots, `math.sum`, …) and
`tools/capture_strategy.pine` (the backtest engine, `PINE_PLAN.md` §13; strategies are added
with **Open → New strategy**, or by pasting into a new script, since the file starts with
`strategy(...)`):

1. Open the **Pine Editor** (bottom panel of the chart).
2. Create a new indicator (**Open → New indicator**, or the "+" / new-script button).
3. Replace the editor's contents with the whole file from this repository.
4. **Save** it under any name, then click **Add to chart**.

The scripts appear in panes below the price chart. The semantics script shows a compiler
**warning** that some `ta.*` calls are not executed on every bar. That is intended: it is one
of the behaviours being measured. A compiler **error** is not intended; copy its message and
send it to me, because it means my Pine code needs fixing.

Do not change the scripts' inputs or rename their plots. The checkers find columns by plot title.

## 3. Load history

Scroll the chart left until no more bars load, or as far as is practical. The checks compare
from the first bar the chart has loaded, so more history is better, and the export must start
at that first loaded bar (step 6 checks this).

## 4. Export

Open the layout menu (top right, next to the layout name) and choose **Export chart data…**.
Leave the chart selected, pick either time format (ISO or UNIX; both work), and export. You
get a CSV, for example `~/Downloads/NASDAQ_AAPL, 1D.csv`.

Rename it to something without spaces or commas, and keep it **outside this repository**, or in
`tests/fixtures/tradingview/`, which git ignores. Don't commit it: whether TradingView's terms
allow redistributing exports is still an open question (`PINE_PLAN.md` §9).

```bash
mv "$HOME/Downloads/NASDAQ_AAPL, 1D.csv" "$HOME/Downloads/aapl_1d.csv"
```

## 5. Describe the export (manifest)

This writes `aapl_1d.yaml` next to the CSV, recording what the chart was set to. Use the
values you noted in step 1:

```bash
uv run python -m pine.capture manifest ~/Downloads/aapl_1d.csv --symbol NASDAQ:AAPL --timeframe 1D --timezone America/New_York
```

Add `--adjustment splits_dividends` if ADJ was on, `--adjustment none` for instruments with no
corporate actions (most futures and crypto), and `--session extended` if extended hours were
shown. The default is `--adjustment splits --session regular`.

## 6. Run the checks

```bash
uv run python -m pine.capture check ~/Downloads/aapl_1d.csv --out ~/Downloads/aapl_1d_report.md
```

It prints a report and saves it to the `--out` file:

- **Overall: PASS** means everything matched what `pine` expects.
- **Gap-free columns**: bar-by-bar parity for each built-in (`sma`, `ema`, `rsi`, `atr`, …).
  The `src_na` row is the alignment check. An export that starts after the chart's first bar
  is handled (see below); `src_na` fails only if rows are missing or the script was changed.
- **na policy**: which of the three hypotheses about `na` inside windows TradingView follows.
- **Pine semantics**: one row per behaviour (conditional `ta.*` calls, rounding, `%`, and so
  on). A row marked **runtime must change** is a real finding, not a failure of the process.

The exit code is 0 for a full pass and 1 when differences were found. Both are useful results.

## 7. Send back the report

Send me the report file (`aapl_1d_report.md`), not the CSV. It contains only comparison
results: error sizes, first divergent bars and which hypothesis matched. If something went
wrong earlier, send the error message instead.

## If the export starts part-way through history

TradingView may export only the later part of the chart's history (a daily NFLX export
started at bar 3116 of 6151). That's fine: the checkers read the true bar number from the
`bar_index` column, skip the rows that depend on unseen earlier bars, and say so in the
report. Things to know:

- **EMA / RMA / ATR**: checked by an exact decay test. Their formula is verified on every bar;
  only the starting value is not.
- **inconclusive**: a column with fewer than 100 bars left after those exclusions. Export a
  longer history if you see it.
- **not decidable in this export**: the behaviour only shows before an event that happened
  before your export started. That's not a failure.
- **synchronised**: stateful ports (Supertrend, the trailing stop, the regime machine) are
  compared from the first row where their state matches. The report gives that row.

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `export not found: …` | The path is wrong; use the real file name from step 4. |
| `manifest not found: …` | Run step 5 first, or pass `--timeframe 1D` to `check` to skip the manifest. |
| `none of the capture columns` / `contains neither capture script's columns` | The scripts weren't on the chart when you exported, or TradingView renamed the columns. Check the CSV header and send it to me if the names differ. |
| `export lacks … columns […]` | Only some plots were exported. Check that both scripts are visible and unchanged. |
| `src_na` fails | The gap pattern doesn't line up with the exported `bar_index`. Send the report: either rows are missing in the middle of the export or the script was edited. (A late start alone is handled.) |
| `varip_acc` differs only on the last row | The last bar was live (24/7 markets, or a session still open); `varip` keeps intrabar ticks there. Handled: the checker excludes that row and says so. |
| No "Strategy" section and no "Skipped" line about it | Your checkout predates the strategy checker: `git pull` (after the PR adding it is merged). The strategy script also has to be on the chart when you export. |
| Strategy section says "Tick size unknown" | The export lacks `strat_mintick` (older capture script). Pass the symbol's tick, e.g. `--mintick 0.00001` for BINANCE:DOGEUSDT, to `check` (or `manifest`). TradingView rounds strategy fill prices to that tick. |
| Report says "Last row dropped" in the strategy section | The last bar was live; strategies calculate only on bar close, so their columns are empty there. Handled automatically. |
| `strat_bar_index: …` under "Could not run" | The strategy script's bar-number column has `na` or gaps (the message names the first row). Usually the strategy wasn't calculated on every bar (e.g. it errored or was added after a partial load). Remove and re-add the strategy, let the chart finish loading, re-export. The other checks still ran. |
| Semantics rows `loop_builtin` / `loop_userfn` say **runtime must change** | A real finding about calls inside `for` loops: send the report; `ctx.loop` gets changed to the matching hypothesis. |
| Built-ins row `pivots` says "not decidable" | No equal highs or lows in this export, so the tie rule can't be seen. Normal on prices with many decimals; a stock on 1D usually has ties. |
| Built-ins rows `obv` / `pvt` note "no volume" | The symbol has no volume (some indices, FX). Those rows can't be checked on it. |
| `bar_index column is not consecutive` | Rows are missing or reordered in the CSV. Re-export without editing the file. |
| `timeframe '1h' is not a Pine timeframe string` | Use Pine spelling: `60` rather than `1h`, `1D` rather than `D1`. |

## Checking a port

To check a port of your own indicator (`PINE_PLAN.md` §11 H1), the TradingView side needs
the intermediate values too, not only the final lines:

1. Copy the indicator in the Pine Editor (**Save as**). Work on the copy.
2. Add `plot(bar_index, "bar_index")` and one plot per intermediate value you want checked,
   with short unique titles, e.g. `plot(smoothed, "dbg_smoothed")`. Plots in `if` blocks are
   not allowed in Pine; assign to a variable and plot that. Labels, lines and fills are not
   exported; plot the numbers they are drawn from instead. Pine allows 64 plots per script;
   split across two copies if needed.
3. Export as in steps 3–4, and keep the CSV outside this repository.
4. In the port, `ctx.plot` the same names, then:

```python
from pine.io import load_tradingview_csv
from pine.port import Stage, check_port

exp = load_tradingview_csv("my_export.csv", timeframe="1D", session_tz="America/New_York")
stages = [Stage("dbg_smoothed", exclude=20), Stage("dbg_signal", kind="discrete")]
print(check_port(exp, script, stages).to_markdown())
```

List stages in pipeline order (inputs first). `exclude` skips warm-up rows; in an export that
starts part-way through history, recursive filters (EMA-like) need enough rows for their
unseen start value to decay (`pine.parity.convergence_bars`), and `sync=True` compares a
state machine from the row where both sides agree. The report names the **root cause**: the
stage that diverges earliest. Fix that one first.

When it passes, record what it passed against, in the port's own repository:

```python
from pine.port import port_manifest, verify_port_manifest, write_port_manifest

report = check_port(exp, script, stages)
m = port_manifest(
    report,
    pine_source=Path("debug_copy.pine"),
    port="my_port",
    port_version="1.0",
    export="my_export.csv",
    inputs={"length": 14},
)
write_port_manifest("my_port.port.yaml", m)
```

The manifest holds a hash of the Pine source, not the source. Later,
`verify_port_manifest(load_port_manifest(...), Path("debug_copy.pine"), "1.0")` returns the
reasons the acceptance no longer applies (source edited, port version changed), or nothing.
Stage names and inputs can still describe your indicator, so keep the manifest private with
the port.

If the indicator imports a TradingView library, the port needs that library's source too:
`check_port` can only compare what the port computes.

## More rows

The full verification matrix (`PINE_PLAN.md` §10 A1) repeats steps 1 and 3–7 on different
charts. Done so far: NFLX daily (opening gaps, DST, long history) and NFLX weekly (from bar 0).
Remaining rows, with exact commands. In TradingView, switch the chart's symbol and timeframe
(the scripts stay on the chart), export, then run the three commands. `ls -t … | head -1`
picks the newest export matching the symbol, whatever suffix TradingView adds.

**Strategy on a stock: `BATS:NFLX`, daily** (decides how stop/limit orders fill across
opening gaps, which the crypto export could not show). All four scripts on the chart:

```bash
F=$(ls -t ~/Sync/obsideon/vault/pine_test/BATS_NFLX*1D*.csv | head -1); echo "$F"
```

```bash
uv run python -m pine.capture manifest "$F" --symbol BATS:NFLX --timeframe 1D --timezone America/New_York --mintick 0.01 --force
```

```bash
uv run python -m pine.capture check "$F" --out ~/Sync/obsideon/vault/pine_test/nflx_1d_report.md
```

**Price below 1: `BINANCE:DOGEUSDT`, daily** (crypto: UTC, no adjustment)

```bash
F=$(ls -t ~/Sync/obsideon/vault/pine_test/BINANCE_DOGEUSDT*1D*.csv | head -1); echo "$F"
```

```bash
uv run python -m pine.capture manifest "$F" --symbol BINANCE:DOGEUSDT --timeframe 1D --timezone UTC --adjustment none
```

```bash
uv run python -m pine.capture check "$F" --out ~/Sync/obsideon/vault/pine_test/doge_1d_report.md
```

**Price above 10 000: `BITSTAMP:BTCUSD`, daily**

```bash
F=$(ls -t ~/Sync/obsideon/vault/pine_test/BITSTAMP_BTCUSD*1D*.csv | head -1); echo "$F"
```

```bash
uv run python -m pine.capture manifest "$F" --symbol BITSTAMP:BTCUSD --timeframe 1D --timezone UTC --adjustment none
```

```bash
uv run python -m pine.capture check "$F" --out ~/Sync/obsideon/vault/pine_test/btc_1d_report.md
```

**Intraday stock: `BATS:AAPL`, 1 hour** (regular session: extended hours off on the chart)

```bash
F=$(ls -t ~/Sync/obsideon/vault/pine_test/BATS_AAPL*60*.csv | head -1); echo "$F"
```

```bash
uv run python -m pine.capture manifest "$F" --symbol BATS:AAPL --timeframe 60 --timezone America/New_York
```

```bash
uv run python -m pine.capture check "$F" --out ~/Sync/obsideon/vault/pine_test/aapl_60_report.md
```

If `echo "$F"` prints nothing, the file name differs from the pattern; run
`ls -t ~/Sync/obsideon/vault/pine_test | head` and use the real name. If the symbol isn't
available under that exchange prefix in your account, any exchange works; pass the symbol
exactly as the chart header shows it.
