# Running the TradingView capture

This is the step that turns `pine`'s checks from "consistent with itself" into "matches
TradingView". You put two scripts on one chart, export the chart data once, and run two
commands. It takes about ten minutes. Background: `PINE_PLAN.md` §10 A1/A2 and §11 E1.

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

For each of `tools/reference_capture.pine`, `tools/capture_semantics.pine` and
`tools/capture_strategy.pine` (the strategy one is optional; it checks the backtest engine
emulation, `PINE_PLAN.md` §13):

1. Open the **Pine Editor** (bottom panel of the chart).
2. Create a new indicator (**Open → New indicator**, or the "+" / new-script button).
3. Replace the editor's contents with the whole file from this repository.
4. **Save** it under any name, then click **Add to chart**.

Both scripts appear in panes below the price chart. The semantics script shows a compiler
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
| `bar_index column is not consecutive` | Rows are missing or reordered in the CSV. Re-export without editing the file. |
| `timeframe '1h' is not a Pine timeframe string` | Use Pine spelling: `60` rather than `1h`, `1D` rather than `D1`. |

## More rows

The full verification matrix (`PINE_PLAN.md` §10 A1) repeats steps 1 and 3–7 on different
charts. Done so far: NFLX daily (opening gaps, DST, long history) and NFLX weekly (from bar 0).
Remaining rows, with exact commands. In TradingView, switch the chart's symbol and timeframe
(the scripts stay on the chart), export, then run the three commands. `ls -t … | head -1`
picks the newest export matching the symbol, whatever suffix TradingView adds.

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
