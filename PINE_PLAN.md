# pine: public Pine Script → Python toolkit (preliminary plan)

Status: preliminary plan, 2026-11-03. Nothing is built. `pine` is a **public** repository
(GitHub), separate from obsideon. Its private counterpart is `field` (`FIELD_PLAN.md`).

## 1. What it is

A public, self-contained toolkit for moving TradingView Pine Script indicators into Python
and **proving the two agree**. Three parts, in order of value:

1. **`pine.ta`** — exact Python reimplementations of Pine's built-ins (`ta.sma`, `ta.ema`,
   `ta.rma`, `ta.atr`, `ta.rsi`, `ta.stdev`, `ta.highest`, `ta.crossover`, `ta.supertrend`, …),
   matching Pine's `na` propagation, warm-up and seeding rules rather than TA-Lib's or pandas'.
2. **`pine.parity`** — the harness: load a TradingView chart export (bars plus plotted
   indicator columns), run the Python version on the same bars, and report agreement within a
   stated tolerance, the first divergent bar, and the warm-up length excluded.
3. **`pine.lint`** — a static repainting check of a `.pine` file: `request.security` with
   `lookahead_on`, `calc_on_every_tick`, `varip`, `barstate.isrealtime`, intrabar fill
   settings (`fill_orders_on_standard_ohlc`), and history references into the future.

A transpiler (Pine source → Python) is **phase 3 and optional**. The harness is what makes a
conversion trustworthy, however it was produced (by hand, by an LLM, or by a transpiler).

## 2. The public/private boundary

- **Allowed in `pine`:** TradingView built-ins, well-known public indicators (RSI, MACD,
  Bollinger, Supertrend, Ichimoku), synthetic bars, and public data whose terms permit it.
- **Never in `pine`:** anything from pineapple, `island/docs/pine/island_triggers.pine`,
  `the_oscillator.pine`, pyramid's `scan_pine_signals.py`, tickers or settings used in
  live strategies, broker or account details. Issues and commit messages follow the same rule.
- **Direction is one-way.** obsideon uses `pine`; `pine` never imports or mentions obsideon.

## 3. How obsideon uses it

It is the tool for two open items in `PIPELINE_ADDENDUM.md`:

| Item | Use |
|---|---|
| 74 — repainting review of every strategy that alerts strait | run `pine.lint` on each strategy and the indicators it calls; the verdict goes in the strategy's registration |
| 75 — Pine parity references | for each indicator feeding a strategy that seeks graduation: Python reference in academy, parity test via `pine.parity` on exported TradingView bars |

Installed from the Gitea mirror at a pinned tag (`uv` git source), as for any dependency.

## 4. Phases

| Phase | Delivers | Done when |
|---|---|---|
| P0 | Repo, licence, CI, `Bars` schema (ts, OHLCV, timeframe, session tz), TradingView CSV loader | loader round-trips a TradingView export fixture; CI green |
| P1 | `pine.ta`: ~25 most-used built-ins with Pine semantics | every built-in has a test against a documented Pine reference value, incl. warm-up `na` count |
| P2 | `pine.parity`: tolerance report, first divergent bar, warm-up handling, HTML/MD report | a deliberately wrong port (seed changed) is caught at the right bar |
| P3 | `pine.lint`: repainting rules with line numbers | each rule has a positive and negative fixture; zero findings on a known-clean script |
| P4 (optional) | transpiler for a declared Pine v5/v6 subset (series, `var`, `[]`, `if`, `for`, `ta.*`, `math.*`) | transpiled output passes `pine.parity` on 10 public indicators |

P0–P3 fit one cloud session before the credit expires on **2026-11-05**. P4 does not.

## 5. Risks and open questions

- **Exact Pine semantics come from recall unless checked.** Seed rules (`ta.rma` vs `ta.ema`
  initialisation), `na` handling and `ta.stdev`'s population vs sample choice must be checked
  against TradingView's documentation and a real export before they are frozen in tests.
- **Fixture licensing.** Whether TradingView chart exports may be committed to a public repo
  is unchecked. Fallback: synthetic bars, with expected values generated in TradingView and
  committed as numbers only, or kept out of the repo and fetched by the user locally.
- **Name.** `pine` on PyPI is probably taken; the import name can stay `pine` with a
  distribution name such as `pine-parity`. Check before P0.
- **Pine version.** v5 or v6 as the target; pineapple is v5.

## 6. Checklist

- [ ] **P2 — create the `pine` GitHub repo and run P0–P3 in a cloud session** — uses the
      expiring cloud credit on public-safe work.
      - _Raised:_ 2026-11-03
      - _Why:_ the credit expires 2026-11-05; `pine` is the only planned item that may sit on GitHub.
      - _Done when:_ repo exists with CI green and `pytest` collecting tests for `ta`, `parity` and `lint`.
- [ ] **P3 — confirm TradingView export terms before committing fixtures**
      - _Raised:_ 2026-11-03
      - _Why:_ a public repo with redistributed data that the terms forbid is a takedown risk.
      - _Done when:_ a recorded answer in the `pine` README's data section.
