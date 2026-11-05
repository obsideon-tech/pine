# pine: public Pine Script → Python toolkit (plan)

Status, 2026-11-03: **P0, P2 and P3 built**; P1 built for the core subset (§7.5); P4 not
started. **No built-in is verified against TradingView output yet** (§5.8): every check so
far is self-referential (§8, §10). **Complex indicators cannot yet be ported with a parity
guarantee**: there is no emulation of Pine's execution model, sessions or
`request.security` (§11). `pine` is a **public** repository (GitHub). Sections: parity spec
(§5), lint (§6), data and operations (§7), risks (§8), checklist (§9), second review (§10),
third review — complex indicators (§11).

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
- **Never in `pine`:** proprietary indicators or strategy source, tickers or settings used in
  live strategies, broker or account details, and the **names, paths or internal
  identifiers of private consuming projects**. Issues, commit messages, fixtures, test names
  and CI logs follow the same rule.
- **Direction is one-way.** Private projects may depend on `pine`; `pine` never imports,
  references or describes them.

## 3. Downstream consumers (generic)

Consumers install `pine` at a pinned version (see §7.2) and use it for two things:

| Use | Tool |
|---|---|
| Repainting review of a strategy and the indicators it calls, before it is allowed to alert | `pine.lint`; the verdict and the lint version are recorded by the consumer |
| Reference ports of indicators a strategy depends on | Python port + parity test via `pine.parity` on exported TradingView bars |

How a consumer records or gates on these results is the consumer's business and is not
documented here.

## 4. Phases

| Phase | Delivers | Done when | Status (2026-11-03) |
|---|---|---|---|
| P0 | Repo, licence, CI, `Bars` schema (ts, OHLCV, timeframe, session tz), TradingView CSV loader, fixture manifest schema + validator (§7.1) | loader round-trips a TradingView export fixture; every fixture has a valid manifest; CI green | done on synthetic fixtures; layout unchecked against a real export |
| P1 | `pine.ta`: core built-ins with Pine semantics (see §7.5 for scope) | every built-in meets the two-oracle rule (§5.8), incl. warm-up `na` count | **core subset done** (2026-11-03): sma, ema, rma, rsi, stdev, highest, lowest, tr, atr, crossover, crossunder meet (a)+(b)+(c) on two exports (§12); batch 1 (§11 F) not started |
| P2 | `pine.parity`: tolerance report (§5.1–5.7), first divergent bar, warm-up/convergence modes, HTML/MD report | every mutation in the suite (§5.9) is detected at its expected bar | done; Markdown report only (HTML deferred) |
| P3 | `pine.lint`: repainting rules with line numbers (§6) | each rule has a positive and negative fixture, incl. the safe `lookahead_on` idiom as a negative; zero findings on a known-clean script | done; precision/recall on real scripts unmeasured (§10 C3) |
| P4 (optional) | transpiler for a declared Pine v5/v6 subset (series, `var`, `[]`, `if`, `for`, `ta.*`, `math.*`) | transpiled output passes `pine.parity` on 10 public indicators | not started; after P5–P6 (§11 J) |
| P5 (new) | `pine.runtime`: bar-by-bar execution model — `Series` with history, `var`, per-call-site state, conditional evaluation (§11 E) | execution-model probes pass against a TradingView export; incremental built-ins equal `pine.ta` | **in progress**: runtime built, equals `pine.ta` bit for bit; export pending |
| P6 (new) | time, sessions and `request.security` emulation (§11 G) | capture of session/HTF probes passes, or documented feed differences | not started |

P0–P3 were completed inside the cloud credit window (deadline 2026-11-05) by following the
§7.5 cut order. P4 is not scheduled; §10 ranks the work that should come first.

## 5. Analytical framework (parity spec)

"Agrees with Pine" must be a precise, falsifiable statement. This section defines it. Items
marked **(unverified)** are design decisions resting on assumptions that must be checked
against TradingView documentation or a real export before tests are frozen.

### 5.1 Tolerance

A value passes if `|py − pine| ≤ atol + rtol · |pine|`, evaluated per output column.
Defaults are set per output class, not globally:

| Class | Examples | Rationale |
|---|---|---|
| price-scaled | SMA, EMA, ATR, Bollinger bands, Supertrend line | error scales with price level → `rtol` dominates |
| bounded oscillator | RSI, stoch, MFI (0–100) | fixed scale → `atol` dominates |
| unbounded, sign-changing | MACD, momentum | values cross zero → `rtol` alone fails near 0; needs `atol` |

`atol` must be at least half a unit in the last place of the export's printed precision.
Whether TradingView CSV exports round plotted values to the chart's display precision is
**(unverified)**: measure it on the first real export and set defaults from the result.
Tolerances are part of the public API; loosening one is a breaking change (§7.2).

**Rounded inputs** (implemented 2026-11-03, `pine.sensitivity`). The floor above covers
rounding of the *reference output*. If the export also rounds the OHLC *inputs* to `±δ`
(`δ = 0.5·10^−d` for `d` printed decimals), a correct port fed those inputs differs from the
reference by up to the propagated error below, which `check_export(..., input_rounding=δ)`
adds to `atol` per column (per bar for RSI):

| Built-in | Propagated error | Argument |
|---|---|---|
| sma, ema, rma | `δ` | non-negative weights summing to 1, seed included |
| highest, lowest | `δ` | max/min are 1-Lipschitz |
| stdev | `δ` (population), `√(n/(n−1))·δ` (sample) | `σ = ‖Px‖/√n`, `P` the centring projection |
| tr, atr | `2δ` | each term is a difference of two inputs; max and RMA are 1-Lipschitz |
| rsi | `min(100, 200δ/(U′+D′))` per bar | `|ΔU|, |ΔD| ≤ 2δ` ⇒ `|ΔRSI| = 100·|ΔU·D − U·ΔD| / ((U′+D′)(U+D)) ≤ 200δ/(U′+D′)`; `U′, D′` from the port |

The RSI bound is exact (not first-order) and needs only the port's own `U′, D′`. There is no
useful global RSI tolerance: in a quiet market one tick of rounding moves RSI by points (a test
demonstrates > 1 point at `δ = 0.005`). Soundness is tested by Hypothesis with uniform, all-`+δ`
and alternating `±δ` perturbations; tightness by cases that attain `δ` (SMA) and `2δ` (TR).

The allowance is **opt-in** (default: inputs exact). It widens tolerances and so raises the
smallest detectable defect; use it only when prices are rounded beyond the instrument's tick,
typically dividend-adjusted data. `python -m pine.reference --inputs-rounded` takes `δ` from
the coarsest printed OHLC precision. A test confirms the allowance does not mask the seed-swap
mutation on rounded data.

### 5.2 na-mask parity before value parity

The boolean pattern of `na` must match **exactly**. It is reported first and separately: an
`na` mismatch is a structural bug (wrong warm-up length, wrong `na` propagation), not
numerical noise, and must not be absorbed by a tolerance.

### 5.3 Warm-up versus convergence

For recursive filters (EMA, RMA, and anything built on them: ATR, RSI, Supertrend) an
initial-state error `e₀` decays as `e_n = e₀ · (1 − α)ⁿ`. Consequence: **excluding a long
warm-up hides seeding bugs** — a port with the wrong seed converges and passes. Rules:

- **Aligned mode (default).** Compare from the first bar on which Pine emits a non-`na` value.
  No additional exclusion. This is the only mode in which a parity pass certifies seeding.
- **Convergence mode.** Used only when Python's history starts later than Pine's (§5.4).
  Exclude `n ≥ ln(tol / |e₀|) / ln(1 − α)` bars, with `|e₀|` bounded by the series range.
  Worked example: RMA(14), `α = 1/14`, `e₀ = 1`, `tol = 1e-8` → `n ≥ 18.42 / 0.0741 ≈ 249`
  bars. A pass in this mode says nothing about seeding and the report must say so.

The report always states the mode used and the number of bars excluded.

**Composite indicators** (implemented 2026-11-03, `pine.convergence`). For a Python history
that is a suffix of Pine's, with Python's seed bar `w`:

| Indicator | Per-bar bound on `|py − pine|` | Why |
|---|---|---|
| ATR | `e₀ (1 − 1/L)^(t−w)`, `e₀ ≤ max TR` | from Python's second bar the true ranges agree, so only the RMA state at `w` differs, and that decays exactly geometrically; both ATRs are convex combinations of non-negative true ranges |
| RSI | `min(100, 100·M·(1 − 1/L)^(t−w) / (U′_t + D′_t))` | `u, d ∈ [0, M]` (`M` = largest bar-to-bar move) so `|ΔU|, |ΔD| ≤ M` at `w`, decaying geometrically; then `|ΔRSI| ≤ 100·max(|ΔU|, |ΔD|)/(U′+D′)` |

`atr_convergence_bars` / `rsi_convergence_bars` return the exclusion for
`compare_series(mode="convergence", exclude=n)`. RSI's answer depends on the data, not only on
`L`: a quiet stretch after a volatile one (large `M`, small `U′+D′`) needs more bars. The
unseen history's magnitude (`e₀`, `M`) defaults to what Python's data shows — an explicit
assumption; pass larger values if earlier history may have been more volatile. Tests:
Hypothesis checks that the bounds cover the realised error across random histories,
truncations, lengths and volatilities; the computed exclusion passes while a short one fails;
the ATR residual is shown to decay exactly geometrically.

### 5.4 History alignment

Pine's state depends on the first bar loaded on the chart, not the first bar the user looks
at. Python must start on the same first bar as the export. If it cannot (truncated export,
data gap), only convergence mode (§5.3) applies.

**Truncated exports are the normal case** (observed 2026-11-03: the NFLX daily export began
at `bar_index` 3116 of 6151). Since then (`pine.truncation`), the checkers read the true bar
offset from the exported `bar_index` column (or, for a reference-only export, infer it modulo
50 from the `src_na` gap pattern) and compare only what the export can establish:

| Kind | Method | What is verified | What is not |
|---|---|---|---|
| windowed (sma, stdev, highest, lowest, tr) | exclude Python's own warm-up | every later bar, exactly | — |
| seeded recursive (ema, rma, atr) | remove the predicted residual `r_w(1−α)^(t−w)`, compare from `w` | recursion and `α` on every bar | the seed |
| RSI | convergence bound (B3) | bars after the bound | the seed |
| probes keyed to `bar_index` | true bar numbers | exact | — |
| `var` running sum | anchor on the first exported value | every increment | the pre-export sum |
| stateful ports, recursive `*_na` | observed synchronisation row, which must fall in the first half of the export | every later bar | state before sync |

A column that compares fewer than 100 bars after exclusions is **inconclusive**, not a pass
(`ColumnReport.min_compared`). A probe whose hypotheses agree on every compared row (e.g.
`barssince` before the first event, when the export starts later) is reported as *not
decidable in this export* rather than confirmed.

### 5.5 Discrete outputs

Booleans and categorical outputs (`ta.crossover`, Supertrend direction, signal flags) are
compared exactly. A mismatch is **excused** only if the continuous inputs to the decision
are within tolerance of the tie (e.g. `|a − b| ≤ atol + rtol·|b|` for a crossover of `a` over
`b`). For crossovers the decision at bar `t` uses bars `t` **and** `t−1`, so the excuse mask is
`crossover_excuse(a, b, tol)` = near-tie at `t` or `t−1` (`near_tie` alone covers only `t`).
`check_export` checks exported 0/1 columns with `PortSpec(kind="discrete", excuse=...)`; `na` in
an exported discrete column reads as false and is noted. Excused flips are counted and listed in the report, never silently passed.
State that depends on a flipped decision (e.g. Supertrend after a direction change) is
re-synchronised to Pine's value for the next comparison, and the re-sync is reported.

### 5.6 Floating-point model

float64 throughout. Summation order matters: numpy's pairwise `sum` differs from a
sequential running sum at the ~1e-16 relative level, and a rolling sum that adds and
subtracts accumulates drift over long series. Pine's internal method for `ta.sma` and
`ta.stdev` is **(unverified)**. The reference implementation is therefore sequential and
documented, and any rolling-sum optimisation must state a drift bound or periodically
recompute the window from scratch.

### 5.7 Report metrics

Per column: mode (§5.3), bars compared, bars excluded, na-mask mismatches, first divergent
bar (index and timestamp), max absolute error and its bar, max relative error, RMSE, share of
bars within tolerance, excused discrete flips. A mean error is never reported on its own — it
hides the localised errors that matter most (seeding, regime changes).

**Minimum detectable error** (implemented 2026-11-03). Every column also states what a pass
could *not* have seen. With `B_t = atol_t + rtol·|pine_t|` and `C` the compared bars:

- `mde_offset = min_{t∈C} B_t`: a constant bias below this passes on every bar.
- `mde_seed = min_{t∈C} B_t / (1−α)^(t−w)` for seeded recursive filters given `alpha`
  (`w` = the reference's first valid bar): a seed error below this decays inside the tolerance
  everywhere. In aligned mode it is about `B_w`; in convergence mode it grows like
  `B/(1−α)^(exclude−w)` — the §5.3 hazard as a number (> 10⁶× the aligned value for EMA(14)
  with 150 bars excluded, per test). Not given for RSI, whose seed errors live in the up/down
  averages rather than RSI units.

Magnitude sweeps confirm both thresholds (an injected error at 0.5× passes, at 2× fails) for
EMA, RMA, ATR and SMA. The MDE makes tradeoffs visible: the A3 input allowance raises
`mde_offset` for SMA from ~1e-7 to ≥ 0.005 at two-decimal inputs, and a column printed with two
decimals cannot detect anything below 0.005 regardless of the port.

Reports are available as Markdown (`to_markdown`) and standard JSON (`to_dict`/`to_json`;
per-bar `atol` summarised as `{"max", "per_bar"}`, infinities as `null`).

### 5.8 Independent oracles: the two-oracle rule

A built-in is **verified** only when it passes (c) and at least one of (a) or (b):

- **(a) Closed-form cases.** Constant series → SMA = EMA = RMA = the constant; linear ramp →
  SMA lags the input by `(n − 1)/2` bars; strictly rising series → RSI = 100 once warm;
  `ta.highest` of a rising series = current value.
- **(b) Property tests** (`hypothesis`): translation equivariance `f(x + c) = f(x) + c` and
  scale equivariance `f(kx) = k·f(x)` for moving averages; invariance of RSI and stochastic
  to positive affine maps; output bounds; `na` count independent of the values.
- **(c) TradingView reference values** from an export, subject to §7.1.

The reason is that a single oracle can share a misconception with the port (e.g. both
remembered from the same blog post).

### 5.9 Harness sensitivity: mutation suite

The harness is itself a measurement instrument and needs a measured detection rate.
P2's single "seed changed" check becomes a suite of deliberate defects applied to correct
ports:

Let `w` be the first non-`na` bar of the correct output (`length − 1` for SMA/EMA/RMA,
`length` for RSI). Implemented in `tests/test_mutation_suite.py`; all cases pass (2026-11-03).

| Mutation | Expected first divergence | Status |
|---|---|---|
| seed swapped (SMA seed → first-value seed), EMA/RMA/RSI | bar `w`, value | ✅ |
| length + 1 | bar `w`, na-mask | ✅ sma, ema, rma, rsi |
| length − 1 | bar `w − 1`, na-mask | ✅ |
| one-bar look-ahead (`py[t] = ref[t+1]`) | bar `w − 1`, na-mask (shows up as early output, not "first bar") | ✅ |
| `na` → 0 substitution on an input gap | the gap bar, na-mask | ✅ |
| first bar dropped | bar `w`, na-mask | ✅ |
| `stdev` sample instead of population | bar `w`, value | ✅ |
| stdev, highest, lowest: length ± 1, look-ahead, `na` → 0, first bar dropped | as for SMA | ✅ |
| ATR: true range = `high − low` (gap terms ignored) | bar `w`, value — **only on gapped data** | ✅ |
| ATR: first-bar true range `na` (`ta.tr` default instead of `high − low`) | bar `w`, na-mask | ✅ |
| ATR: `na` close → 0 | gap bar **+ 1**, na-mask (true range uses `close[1]`) | ✅ |
| ATR: length ± 1, look-ahead, first bar dropped | as for SMA | ✅ |
| crossover: previous-bar condition dropped; one-bar look-ahead; crossunder swapped in | first differing bar, via `compare_discrete` | ✅ |

46 numeric cases plus the discrete ones; every predicted bar matched on the first run.

Fixture coverage matters as much as the harness: on the gapless random walk every open
equals the previous close, so the true-range gap terms never dominate and the "`high − low`
only" ATR bug **passes**. A test records this, and ATR uses a gapped fixture
(`random_walk_gaps_60`, gap term dominant on 114 of 300 bars). Rule: a fixture must
exercise every term of the formula under test, and §7.1 manifests should say which terms a
fixture covers.

Negative controls (must **not** be flagged): SMA via cumulative sums and EMA rearranged as
`prev + α(x − prev)` both pass, with max error < 1e-10 on the fixture. A further test
demonstrates the §5.3 hazard: the seed mutant passes in convergence mode once
`convergence_bars` bars are excluded, and fails in aligned mode.

Observation: every case except the seed swap is caught by the **na mask**, not by values. The
na-mask check carries most of the suite's detection power, so it must never be relaxed.

P2 is done when every mutation is detected at its expected bar. New mutations are added
whenever a real porting bug is found that the suite would have missed.

### 5.10 Semantics to verify before tests are frozen

Each item gets a source citation (TradingView reference manual URL and date) or an
export-derived value in the test docstring:

- seeding of `ta.ema` versus `ta.rma` (SMA seed versus first-value seed). Implemented as SMA
  seed for both, re-seeding after `na`, per the recalled manual code **(unverified)**;
- `ta.sma` summation: implemented newest-first with each term divided by `length`, per the
  recalled manual code; the built-in may sum differently (§5.6). Effect is < 1e-10 on the
  fixture, so not material at the default tolerances;
- `ta.stdev`: population by default; deviations with `|d| ≤ 1e-10` zeroed as in the manual's
  `SUM` helper. The recalled helper also returns the literal `15` for `1e-10 < |d| ≤ 1e-4`,
  which reads as a documentation bug and is **not** ported — if the built-in really does
  that, an export will show it;
- `ta.atr`: first-bar true range `high − low`; the manual's `na(high[1])` test means a `na`
  close makes the *next* true range `na`, not `high − low`. `ta.tr` without `handle_na`
  returns `na` on the first bar **(unverified)**;
- `ta.crossover`/`crossunder`: comparisons with `na` are false, so output is never `na`
  (v6 semantics; v5 may differ);
- **`na` inside windows — decided by export (2026-11-03, §12):** `sma`, `stdev`, `highest`,
  `lowest` *skip* `na` (window over the last `length` valid values; a value is output on the
  `na` bar too); `ema`, `rma`, `rsi` carry their state across `na` and output `na` on the `na`
  bar (`skip_na_out`); `ta.change` propagates `na`. Now the defaults
  (`pine.ta.DEFAULT_NA_POLICY`). `atr` inherits `rma`'s default by construction — not yet
  observed directly;
- facts the lint rules rest on **(unverified)**: `lookahead` is the 5th positional parameter
  of `request.security` (after symbol, timeframe, expression, gaps); `alert()`'s default
  `freq` is `alert.freq_once_per_bar` (fires intrabar); a negative literal history index
  does not compile; `barstate.isnew`/`islast` differ between history and realtime as
  described;
- `varip` on the live (last, unconfirmed) bar keeps intrabar ticks, so it differs from `var`
  there; observed on DOGE/BTC exports (§12). Checks exclude that row.
- a strategy (`calc_on_every_tick = false`) is not calculated on the live bar until it
  closes: all strategy columns are `na` on an export's last row when it was taken mid-bar
  (observed, DOGE 1D, 2026-11-03). The strategy check drops that row.
- execution model and scalar rules (`pine.runtime`, §11 E1). **Confirmed by export
  (2026-11-03, §12):** `math.round` ties round **away from zero** (`round(-2.5) = -3`; the
  earlier `floor(x + 0.5)` assumption was wrong and is fixed); `%` truncates (`-7 % 3 = -1`);
  `7 / 2 = 3.5` for series and const ints; `int()` truncates toward zero; `na == na` is false.
  **Still unverified** (that export started mid-history; re-check with the offset-aware
  checker): conditional built-ins and functions advance per call; separate call-site
  histories; `barssince`/`valuewhen` before the first event; `ta.change` around `na`;
- `ta.rsi` zero branches: `down == 0 → 100` checked before `up == 0 → 0`, so a flat series
  gives 100 **(unverified)**;
- `ta.stdev`'s `biased` default (believed `true`, i.e. population) **(unverified)**;
- `na` inside a window for `ta.sma`, `ta.highest`, `ta.lowest` (propagate versus skip);
- `ta.atr` true-range definition on the first bar;
- v5 versus v6 differences affecting ports: in v6, `bool` cannot be `na`, and `and`/`or`
  evaluate lazily **(unverified in detail)**.

## 6. Lint soundness

### 6.1 Stance

`pine.lint` is rule-based and **incomplete**. "No findings" means "no known repainting
pattern", not "does not repaint". The README and the CLI output must both say this.

### 6.2 Correction to the `lookahead_on` rule

Flagging every `lookahead_on` produces false positives on the standard non-repainting idiom:

```pine
request.security(syminfo.tickerid, "D", close[1], lookahead = barmerge.lookahead_on)
```

The rule is: flag `lookahead_on` only when the requested expression is not offset by at least
one bar. Also flag a higher-timeframe `request.security` **without** `lookahead_on` and without
an offset, because its value changes on realtime bars until the higher-timeframe bar closes.
The two cases get different messages: future leak in history versus realtime repaint.

### 6.3 Implemented rules (P3, 2026-11-03)

`pine.lint` tokenizes the source (comments and strings handled, calls split into
positional/named arguments across lines) and applies rules to tokens and calls; it is not a
parser. Every finding has a **category**, a **severity** and a **confidence**: `definite`
means the defect is established, `possible` means safety could not be proven. "Could not
prove safe" is never reported as "proven unsafe".

| id | category | severity / confidence | Fires on |
|---|---|---|---|
| `security-lookahead-leak` | future-leak | error/definite when a built-in series is read unoffset (`close`, `close[0]`, `ta.sma(close, 20)`); warning/possible when the offset cannot be proven | `request.security`/`security` with `lookahead_on` (named or 5th positional) |
| `security-lookahead-unresolved` | future-leak | warning/possible | lookahead given by a variable or expression |
| `security-realtime-repaint` | realtime-repaint | warning/possible | lookahead off, no provable offset, timeframe not `timeframe.period`/`""` |
| `security-lower-tf` | realtime-repaint | info/possible | `request.security_lower_tf` |
| `calc-on-every-tick` | realtime-repaint | warning/definite | `calc_on_every_tick = true` |
| `varip` | realtime-repaint | warning/definite | any `varip` |
| `barstate-realtime` | realtime-repaint | warning/possible | `barstate.isrealtime`, `.isnew`, `.islast` |
| `timenow` | realtime-repaint | warning/possible | `timenow` |
| `alert-unconfirmed` | realtime-repaint | warning/possible | `alert()` without `freq = alert.freq_once_per_bar_close` and not inside an `if`/`else if` whose condition provably requires `barstate.isconfirmed` |
| `alertcondition-unconfirmed` | realtime-repaint | info/possible | `alertcondition()` whose condition does not provably require `barstate.isconfirmed` |
| `fill-timing` | fill-timing | info/definite | `calc_on_order_fills = true`, `process_orders_on_close = true` |
| `negative-history-index` | future-leak | error/definite | `x[-n]`, literal `n` |
| `negative-plot-offset` | display | info/definite | `plot*`/`bgcolor` with literal `offset = -n` |
| `unsupported-version` | meta | info/definite | `//@version` missing or < 5 |

**Offset proof (§6.2).** An expression is *proven* offset when its outermost form is
`atom[k]`, `(expr)[k]` or `f(...)[k]` with literal `k ≥ 1`, or every built-in price/time
series in it carries `[k ≥ 1]` and no user identifier appears outside a second-or-later call
argument. It is a *definite leak* when a built-in series is read unoffset outside any
bare-name call. Everything else is *unproven*. Known error sources:

- **Variable tracing** (since 2026-11-03, §10 C1). Before classification, identifiers that
  are assigned exactly once with `=` at the top level (indent 0), never reassigned (`:=`,
  `+=`, …), not `var`/`varip`, not tuple-destructured, not an `if`/`switch` expression and
  not shadowed by any function parameter are replaced by `(value)`, recursively. So
  `src = close` → `request.security(..., src, lookahead_on)` is a *definite* leak, and
  `prev = close[1]` → `..., prev, ...` is *proven* safe. Untraceable variables stay
  `possible`.
- **False negative (narrowed):** an *untraceable* user series passed as a second-or-later
  call argument (`math.max(close[1], heldVar)`) is still assumed to be a parameter. Traceable
  ones are now expanded and checked (`math.max(close[1], src)` with `src = close` is definite).
- **Conservative ("possible"):** bare-name calls (`nz(close)`, user functions) may offset
  internally; untraceable variables (above).
- **Alert gating** (since 2026-11-03, §10 C2). An `alert()` counts as gated only if an
  enclosing `if`/`else if` (found by indentation; continuation lines with non-multiple-of-4
  indent included) has a condition that, with traced variables expanded, contains an
  un-negated `barstate.isconfirmed` and no `or`/`?` that could bypass it. `else` branches
  never count. `alertcondition` applies the same test to its condition argument. This closes
  the earlier false negative where any `isconfirmed` anywhere counted. Limit: gating inside a
  called function is not seen (reported `possible`).
- **Conservative:** the timeframe argument is not evaluated, so a variable timeframe that is
  in fact the chart's is still reported by `security-realtime-repaint`.

**Testing.** Fixtures in `tests/fixtures/pine/` carry their expectations inline
(`// expect: rule-id` on the offending line); the test requires the set of
`(rule, line)` findings to equal the annotated set exactly, so misses, false positives and
wrong line numbers all fail. Every rule has a positive fixture; `*_neg.pine` files and
`clean_indicator.pine` (zero findings, including risky words inside strings and comments)
are the negatives. An **injection test**, the lint analogue of the mutation suite, inserts
each rule's pattern into the clean script and requires exactly that rule at that line.
Suppression: `// pine-lint: ignore[rule-id]` on the line; suppressed findings are reported.

CLI: `pine-lint FILE... [--format text|json] [--strict]`; exit 1 on errors (or warnings
with `--strict`), 2 on read/tokenize errors.

**Deferred:** `fill_orders_on_standard_ohlc` (default unverified; the rule would fire on
nearly every strategy); tracing user variables into `request.security`; evaluating
timeframe arguments against the chart's.

### 6.4 Dynamic check (later, optional)

The only empirical repainting test is to compare values recorded live (e.g. from an alert
log) against the same bars recomputed on history. Out of scope for P3; noted so the static
check is not mistaken for proof.

## 7. Data provenance and operations

### 7.1 Fixture manifest

Every fixture ships with a YAML manifest. Parity results are not interpretable without it.

| Field | Why |
|---|---|
| symbol class (`synthetic` / public symbol) | licensing (§8) |
| timeframe, session, timezone | bar boundaries |
| extended hours on/off | different bars |
| dividend / split adjustment on/off | different prices |
| chart type (must be standard candles; Heikin Ashi rejected) | HA changes OHLC |
| Pine version, indicator inputs | reproducibility |
| export date | data revisions |
| timestamp convention (bar open vs close) | off-by-one-bar errors |
| displayed precision | sets `atol` floor (§5.1) |

The P0 validator rejects a fixture with a missing or inconsistent manifest.

### 7.2 Releases and pinning

Semver plus a CHANGELOG. Breaking changes include any tolerance loosened, any default changed,
and any change to a built-in's output beyond tolerance. Consumers pin a **commit SHA**, not
only a tag, because tags are mutable.

### 7.3 CI

Python version matrix. numpy and other numeric deps pinned in the lockfile, because reduction
results can shift between versions. The mutation suite (§5.9) runs in CI.

Since 2026-11-03 (§10 D2):

| Job | When | Dependencies | Hypothesis | Purpose |
|---|---|---|---|---|
| `ci` (3.11–3.13) | every push / PR | locked | `ci`: derandomized | gate; failures reproduce exactly; branch coverage with `fail_under = 95` (measured 97%) |
| `weekly / random-examples` | Mondays + manual | locked | `thorough`: random seeds, 10× examples | keep searching for counterexamples CI cannot find |
| `weekly / latest-deps` | Mondays + manual | upgraded (lockfile ignored) | `ci` | catch upstream numeric changes before they reach the lockfile |

The two weekly jobs are separate so a failure's cause is unambiguous (new example vs new
dependency). Example budgets go through `tests/hyp.py` (`examples(n)`), because an explicit
`max_examples` would otherwise override the profile. A Hypothesis failure prints a
`@reproduce_failure` blob; keep the counterexample as an explicit `@example` once fixed.
Coverage is measured inside CI only (no external coverage service). Dependabot proposes weekly
grouped PRs for the SHA-pinned actions and the uv lockfile; support for the `uv` ecosystem in
Dependabot is believed current but **unverified** here — if it is rejected, only the
`github-actions` entry is needed.

### 7.4 Public-repo hygiene

- A denylist of private names checked in pre-commit or CI would itself leak those names if
  stored here. The denylist lives in the private consumer's repo and runs as a **local** hook
  against this checkout.
- Enable GitHub secret scanning and push protection on the repo.
- "TradingView" and "Pine Script" are third-party marks. Use them descriptively, with a
  non-affiliation disclaimer in the README. Whether the repo or distribution name needs
  changing is **(unverified)**.

### 7.5 Scope under the 2026-11-05 deadline

The harness is the value; 25 built-ins that have not met the two-oracle rule are a liability,
not an asset. Cut order if time runs short, keeping from the top:

1. P0 (loader, `Bars`, manifest validator, CI).
2. P2 harness with the mutation suite.
3. Core built-ins to exercise it: `sma`, `ema`, `rma`, `stdev`, `highest`, `lowest`, `atr`,
   `rsi`, `crossover` — closed-form and property oracles first; TradingView values when an
   export is available.
4. P3 lint with the corrected `lookahead_on` rule and the existing rules.
5. Remaining built-ins and §6.3 candidate rules — defer.

Outcome: items 1–4 were completed on 2026-11-03; item 5 stays deferred behind §10.

## 8. Risks and open questions

- **Self-referential evidence (largest risk).** The ports, the closed-form oracles, the
  property tests and the mutants all derive from one recalled reading of the Pine manual. A
  coherent misreading (e.g. the `na`-window rule, §5.10) would pass every test in the suite.
  Only oracle (c), TradingView output, breaks the circle (§10 A1).
- **Execution-model mismatch (largest risk for complex indicators).** Pine runs scripts bar by
  bar: a stateful built-in called inside an `if` only updates on bars where it executes;
  each call site of a function keeps its own history; `var` persists. Vectorised ports (all
  of `pine.ta` today) silently diverge exactly where these rules matter (§11 E).
- **Higher-timeframe data may not be reconstructible.** TradingView builds `request.security`
  values from its own higher-timeframe feed (e.g. settlement-based daily bars, session
  rules), which need not equal an aggregation of the chart's bars (§11 G2).
- **Realtime behaviour is untestable by export.** Exports contain historical, confirmed bars
  only; parity claims are therefore claims about confirmed bars, and repainting scripts can
  match history while differing live (§6, §11 H1).
- **Tolerances ignore rounded inputs.** If an export rounds OHLC, the port computes on
  rounded prices; the resulting output error is not covered by the output-rounding floor of
  §5.1 and is unbounded for RSI near `up + down → 0` (§10 A3).
- **Exact Pine semantics come from recall unless checked.** See §5.10. Nothing in that list is
  frozen in a test without a citation or an export-derived value.
- **Export precision unknown.** If exports are rounded to display precision, `atol` must be
  set from it (§5.1), and low-precision symbols weaken the parity claim. Measure first.
- **Fixture licensing.** Whether TradingView chart exports may be committed to a public repo
  is unchecked. Fallback: synthetic bars, with expected values generated in TradingView and
  committed as numbers only, or kept out of the repo and fetched by the user locally.
- **Seed bugs masked by warm-up exclusion.** Mitigated by aligned mode as the default (§5.3)
  and the mutation suite (§5.9).
- **Lint false confidence.** Mitigated by the stated stance (§6.1); residual risk remains.
- **Name.** Checked 2026-11-03: `pine` is taken on PyPI, `pine-parity` was free. The
  distribution is `pine-parity`; the import name stays `pine`. Tradeoff: installing both
  `pine` (PyPI) and `pine-parity` in one environment would clash on the import name.
- **Pine version.** v5 or v6 as the target. Existing private scripts are v5; new public
  examples should be v6. Tradeoff: supporting both doubles the semantics to verify (§5.10).
- **Private references in history.** The first revision of this file on `main` named private
  projects. Scrubbed from the current text; still present in git history (see checklist).

## 9. Checklist

Priority tags are `[high]`, `[med]`, `[low]`; they are unrelated to phases P0–P4.

- [ ] **[high] Decide on purging private names from `main` history** — rewrite or squash the
      first plan commit on `main`; move the original §2/§3 detail to the private repo.
      - _Raised:_ 2026-11-03
      - _Why:_ the repo is public, and the first revision named private projects and files.
        Scrubbing the current text does not remove them from history, forks or caches.
      - _Done when:_ the owner has decided and, if rewriting, force-pushed `main` themselves.
        An agent does not force-push `main` without an explicit go-ahead.
- [ ] **[high] Create the repo structure and run P0–P3 per the §7.5 cut order in a cloud session**
      — uses the expiring cloud credit on public-safe work.
      - _Progress 2026-11-03:_ **P0 built** — `pine.Bars`, `pine.io` TradingView CSV
        loader/writer (byte-exact round trip on both synthetic fixtures, float64-exact by
        property test), `pine.manifest` validator, synthetic fixtures + generator, CI workflow
        with actions pinned by SHA. Open P0 gap: the loader's column layout is assumed, not
        checked against a real export.
      - _Progress 2026-11-03:_ **P2 built on `sma`, `ema`, `rma`, `rsi`** — `pine.ta` (literal
        ports of the manual's reference code), `pine.parity` (§5.1–5.7: tolerance classes with
        printed-precision floor, na-mask first, aligned/convergence modes, discrete compare with
        near-tie excusal, Markdown report, `check_export`), mutation suite (§5.9) in CI.
        Built-ins meet oracles (a) and (b) only; (c) awaits a real export, so none is
        "verified" under §5.8 yet. HTML report deferred; Markdown only.
      - _Progress 2026-11-03:_ **`stdev`, `highest`, `lowest`, `tr`, `atr`, `crossover`,
        `crossunder` added**, completing the §7.5 core list; same oracle status. Mutation suite
        extended (46 numeric cases + discrete), gapped fixture added for ATR.
      - _Progress 2026-11-03:_ **P3 built** — `pine.lint` with the 14 rules of §6.3, `pine-lint`
        CLI, annotated fixtures, injection test. P3's exit criterion is met. Remaining for this
        item: CI green on the pushed commit (checked per push).
      - _Raised:_ 2026-11-03
      - _Why:_ the credit expires 2026-11-05.
      - _Done when:_ CI green and `pytest` collecting tests for `ta`, `parity` (incl. mutation
        suite) and `lint`.
- [ ] **[high] Measure TradingView export precision**
      - _Raised:_ 2026-11-03
      - _Why:_ sets the `atol` floor (§5.1); every tolerance default depends on it.
      - _Done when:_ one real export inspected, finding recorded in §5.1, defaults set.
- [ ] **[high] Verify the §5.10 semantics with citations**
      - _Raised:_ 2026-11-03
      - _Why:_ tests frozen on recalled semantics would certify wrong ports.
      - _Done when:_ each §5.10 item has a citation or export-derived value.
- [ ] **[high] Confirm TradingView export terms before committing fixtures**
      - _Raised:_ 2026-11-03
      - _Why:_ a public repo with redistributed data that the terms forbid is a takedown risk.
      - _Done when:_ a recorded answer in the `pine` README's data section.
- [x] **[med] Mutation suite as P2's exit criterion** (§5.9) — done 2026-11-03 except the
      `stdev` case, which waits for `ta.stdev`.
- [x] **[med] Fixture manifest schema and validator in P0** (§7.1) — done 2026-11-03.
- [x] **[high] A1 — Reference values for the core built-ins via a capture script and a
      coverage matrix** (§10 A1; supersedes the earlier "commit reference values" item) —
      _2026-11-03: two exports passed (NFLX 1D mid-history, NFLX 1W from bar 0; §12). Core
      eleven verified incl. seeding. Remaining rows: price < 1, price > 10 000, intraday._
      - _Raised:_ 2026-11-03
      - _Why:_ the two-oracle rule (§5.8) needs oracle (c); without it the ports are checked
        only against their own reading of the manual, which shares any misremembering.
      - _Done when:_ `tools/reference_capture.pine` exists, and every core built-in passes
        `check_export` in aligned mode on every row of the §10 A1 matrix, each export with a
        manifest (numbers only or kept local, per the terms item).
      - _Progress 2026-11-03:_ **tooling done** — `tools/reference_capture.pine` (31 plots),
        `python -m pine.reference` (now via `python -m pine.capture`, docs/CAPTURE.md), manifest template
        `tools/reference_capture.example.yaml`. Consistency tests tie the Pine constants and
        plot titles to `pine.reference`; the script is lint-clean. Synthetic end-to-end tests
        prove the plumbing only. **Remaining: the exports themselves (owner action).**
      _2026-11-03: **matrix complete** — DOGE 1D (price < 1, from bar 0), BTC 1D (price
      > 10 000, mid-history), AAPL 60 (intraday session, mid-history) added; see §12._
- [x] **[high] A2 — `na_policy` switch and the opposite-policy mutant** (§10 A2) — **decided
      2026-11-03** by the NFLX export (§12): windowed built-ins `skip`, recursive ones
      `skip_na_out`; these are now the defaults (`pine.ta.DEFAULT_NA_POLICY`) in `pine.ta` and
      `pine.runtime`. The switch is kept as an override rather than removed (see §12).
      - _Raised:_ 2026-11-03
      - _Why:_ the `na`-window rule is the single most consequential unverified semantic.
      - _Done when:_ both policies implemented, the mutant is detected, and an export with `na`
        selects one; the switch is then removed.
      - _Progress 2026-11-03:_ **switch and mutants done** — three hypotheses (`propagate`,
        `skip`, `skip_na_out`) on `sma/ema/rma/rsi/stdev/highest/lowest`; 14 policy mutants
        detected at predicted bars; each `*_na` capture column discriminates all three (a
        synthetic export under any one policy matches that policy only). Default stays
        `propagate`. **Remaining: one real capture export to decide.**
- [x] **[high] A3 — Input-rounding-aware tolerances** (§10 A3) — done 2026-11-03:
      `pine.sensitivity` bounds (§5.1), per-bar `Tolerance.atol`, `PortSpec.input_error`,
      `check_export(..., input_rounding=δ)` (opt-in), `--inputs-rounded` in `pine.reference`
      (also applied to the na-policy decision). Sound and tight per tests.
- [x] **[high] A4 — Minimum detectable error in reports** (§10 A4) — done 2026-11-03:
      `mde_offset` on every column, `mde_seed` where `alpha` is given (§5.7); sweeps at 0.5×/2×
      confirm both for EMA, RMA, ATR, SMA (`tests/test_mde.py`).
- [x] **[high] B1 — Two-bar crossover excuse** (§10 B1) — done 2026-11-03:
      `pine.parity.crossover_excuse`; a flip caused by a tie at t−1 fails with `near_tie` and
      passes with it; real flips and NaN inputs are not excused.
- [x] **[med] B2 — Discrete columns in `check_export`; JSON parity report** (§10 B2) — done
      2026-11-03: `PortSpec(kind="discrete", excuse=...)`, `ParityReport.to_dict/to_json`,
      `python -m pine.reference --format json`; capture crossovers now use the two-bar excuse
      on the exported EMAs.
- [x] **[med] B3 — Composite convergence bounds for RSI and ATR** (§10 B3) — done 2026-11-03:
      `pine.convergence` (§5.3), soundness by Hypothesis, exclusion tests for both.
- [x] **[med] C1 — Lint: single-assignment tracing** (§10 C1) — done 2026-11-03
      (`pine.lint._scope`; `tracing.pine` covers traced, `var`, reassigned, shadowed-parameter
      and destructured cases with per-case confidence asserted).
- [x] **[med] C2 — Lint: block-aware alert gating** (§10 C2) — done 2026-11-03
      (`alerts_block_pos/neg.pine`: isconfirmed elsewhere, `or`, `not`, `else`, nested,
      continued and traced conditions).
- [ ] **[high] E1 — Execution-model probes in a second capture script** (§11 E1)
      - _Raised:_ 2026-11-03
      - _Why:_ the runtime must copy TradingView's behaviour, which is observed, not recalled.
      - _Done when:_ `tools/capture_semantics.pine` + `python -m pine.reference` support, each
        probe with competing hypotheses (like `na_policy`) and a decided outcome from an export.
      - _Progress 2026-11-03:_ **tooling done** — `tools/capture_semantics.pine` (26 plots, 20
        probe groups), `python -m pine.semantics` (md/json; exit 1 names any group where
        TradingView differs from the runtime). Every multi-hypothesis group is shown to
        discriminate on fixture data. **Remaining: one export (owner).**
      - _Progress 2026-11-03 (later):_ the original 20 groups match on exports (§12). Added two
        groups for calls inside a `for` loop (`loop_builtin`: `math.sum`; `loop_userfn`: a user
        function's `x[1]` and `x[p]`), three hypotheses each (`per_call`, `per_bar_last`,
        `per_iteration`), all shown to discriminate. **Remaining: an export with them.**
- [ ] **[high] E2 — `pine.runtime` bar-by-bar execution model** (§11 E2)
      - _Raised:_ 2026-11-03
      - _Why:_ complex indicators are stateful and conditional; without the execution model a
        port can pass on simple data and diverge where it matters.
      - _Done when:_ incremental built-ins equal `pine.ta` (≤ 1e-12) on unconditional use; E1
        probes pass; three stateful reference ports (Supertrend, a trailing stop, a pivot/zigzag
        state machine) pass parity against exports.
      - _Progress 2026-11-03:_ **runtime built** (`pine.runtime`): incremental built-ins equal
        `pine.ta` **bit for bit** (property test, incl. `na` gaps); the runtime reproduces the
        expected hypothesis of every probe exactly; three stateful ports (Supertrend, ratcheting
        stop, regime state machine with a conditionally evaluated crossunder) run on it, and the
        Supertrend port equals an independent loop implementation (`pine.ta.supertrend`).
        The zigzag port was replaced by the regime machine, which exercises conditional
        evaluation directly. **Remaining: parity against an export.**
- [ ] **[high] H1 — Port acceptance protocol with intermediate-series parity** (§11 H1)
      - _Raised:_ 2026-11-03
      - _Why:_ a single output column says *that* a port diverges, not *where*.
      - _Done when:_ documented procedure + `check_port` that reports the upstream-most
        diverging intermediate and its first bar, and writes a port manifest.
      - _Progress 2026-11-03:_ `pine.port.check_port` built: compares every debug-plotted
        stage with the port's `ctx.plot`, names the root-cause stage (earliest divergence in
        time, ties to the upstream stage) and the first failing stage in pipeline order;
        handles truncated exports (offset from `bar_index`, warm-up exclusion, `sync` for
        stateful stages, inconclusive below 100 rows). Procedure: `docs/CAPTURE.md`
        "Checking a port". **Remaining: the port manifest; a first real port through it.**
      - _Progress 2026-11-04:_ port manifest done (`port_manifest`, `write_port_manifest`,
        `load_port_manifest`, `verify_port_manifest`): Pine source digest (line endings and
        trailing spaces normalised) and `//@version`, port name/version, inputs, export file
        hash and chart settings, per-stage status, tolerance, max error and MDE. Verification
        says when a stored acceptance stops applying (source changed, port version changed,
        or it never passed). The manifest records a hash, never the source; stage names can
        still describe a private indicator, so private ports keep it in their own repository.
        **Remaining: a first real port through it.**
- [ ] **[med] F1 — Built-in batch 1 with capture coverage** (§11 F)
      - _Raised:_ 2026-11-03
      - _Why:_ complex indicators are built from more than the 11 current built-ins.
      - _Done when:_ each listed built-in meets §5.8 (incl. oracle (c)), has an A3 sensitivity,
        `alpha` where seeded, mutation cases, and capture columns.
      - _Progress 2026-11-03:_ first subset implemented (`math.sum`, `wma`, `hma`, `variance`,
        `percentile_linear_interpolation`, `pivothigh/low`, `fixnan`, `cum`, `obv`, `pvt`) in
        `pine.ta` and `pine.runtime` (bit-for-bit equal by construction), with naive-reference
        property tests and `tools/capture_builtins.pine` + `pine.builtin_capture` (16 groups,
        competing hypotheses for `na` handling, hma length rounding, pivot ties, percentile
        rank). **Not yet:** an export; A3 sensitivities and mutation cases for these; the rest
        of batch 1.
      - _Progress 2026-11-04:_ A3 bounds and mutation cases done for the subset (below and
        `tests/test_mutation_batch_f.py`, 30 cases, every one caught at the predicted bar);
        `--inputs-rounded` now applies to the built-in capture too. **Remaining: an export;
        the rest of batch 1.**
      - _Progress 2026-11-05:_ `timeframe.*` and `barstate.*` (`pine.chart`; `ctx.timeframe`,
        `ctx.barstate`, `run(..., live_last_bar=...)`), probed by 15 new columns in
        `tools/capture_builtins.pine` (5 groups). Open points, each a hypothesis set:
        seconds per month (2 628 003 expected), `from_seconds` choosing the largest exact unit,
        `timeframe.period` spelled `"1D"` (v6) not `"D"`, `timeframe.change` true on the
        chart's first bar with weeks from Monday in the exchange time zone, and the last bar's
        `barstate` (live or closed; both implemented, the check reports which).
        `timeframe.change` supports 1D/1W/1M targets only; intraday targets need session
        anchoring (G1). Lint observation while adding them: `barstate-realtime` flags
        `islast`, `isrealtime`, `isnew` but not `ishistory` or `islastconfirmedhistory`,
        which also differ on the realtime bar (a C-section blind spot to fix).
- [x] **[med] H2 — Verification registry** (§11 H2)
      - _Raised:_ 2026-11-03
      - _Why:_ "verified" must be a recorded, checkable state per built-in, not prose.
      - _Done when:_ `verification.yaml` with per-built-in oracle status and matrix rows; a test
        fails if a public `ta` function lacks an entry; README table generated from it.
      - _Progress 2026-11-04:_ **done.** `src/pine/verification.yaml` (shipped in the wheel) +
        `pine.registry`. Records the six exports, each public `pine.ta` function (Pine name,
        runtime method, capture script, exports that verified it and decided its `na` policy,
        input-rounding bound, deliberate-bug test file) and every probe group of the three
        checkers (expected hypothesis, exports that decided it). Status is derived, never
        typed: *verified* = at least one export confirmed it. `tests/test_registry.py` fails
        on drift in either direction (a function, runtime method or probe group without an
        entry or vice versa; an expected hypothesis differing from the code; unknown or
        unused exports; a bound or test file that doesn't exist or doesn't call the
        function; a built-in without a bound and without a note saying why). The README
        "Verification status" section is generated (`python -m pine.registry --write
        README.md`) and CI fails if it is stale. Gaps it makes visible: no deliberate-bug
        tests yet for `tr` (covered only via `atr`), `crossunder`, `change`, `barssince`,
        `valuewhen`, `supertrend`, `pivotlow`. Closed 2026-11-04 by
        `tests/test_mutation_more.py`: every public built-in now has deliberate-bug cases.
- [ ] **[med] G1 — Sessions and calendar** (§11 G1)
      - _Raised:_ 2026-11-03
      - _Why:_ `ta.vwap` and session-based logic reset on session boundaries.
      - _Done when:_ `Bars` carries a session spec; `time_close`, session-first-bar, DST cases
        pass a capture.
- [ ] **[med] G2 — `request.security` emulation** (§11 G2)
      - _Raised:_ 2026-11-03
      - _Why:_ multi-timeframe logic is common in complex indicators.
      - _Done when:_ HTF alignment with lookahead on/off and gaps on/off passes a capture, or the
        feed difference is documented and an exported HTF series can be supplied as input.
- [x] **[low] I — Decide whether strategy (`strategy.*`) parity is in scope** (§11 I) — owner
      decided 2026-11-03: in scope. See §13.
- [x] **[high] S1 — Broker emulator, first increment, verified by export** (§13) — **passed
      2026-11-03** on BINANCE:DOGEUSDT 1D (all 2 678 bars, §12); `gap_fill` and stop-price
      rounding still to observe on an equity export with gaps.
      - _Raised:_ 2026-11-03
      - _Why:_ strategy results (positions, P&L, equity) must match TradingView bar for bar.
      - _Done when:_ an export of `tools/capture_strategy.pine` decides every S1 rule and the
        baseline passes. _Progress 2026-11-03: emulator, capture script and checker built;
        synthetic tests pass; export pending (owner)._
- [ ] **[med] C3 — Lint precision/recall on a local corpus** (§10 C3)
      - _Raised:_ 2026-11-03
      - _Why:_ rule quality on real scripts is unknown.
      - _Done when:_ results recorded in §6.3 (counts only, no script content); `definite`
        precision is 100% or the offending rule is fixed.
- [ ] **[low] D1 — v0.1.0 release and CHANGELOG** (§10 D1)
      - _Raised:_ 2026-11-03
      - _Why:_ consumers need a pinned, documented version (§7.2).
      - _Done when:_ tag on `main` after the history decision; CHANGELOG lists the API.
- [x] **[low] D2 — Deterministic CI profile, weekly random run, coverage floor** (§10 D2) —
      done 2026-11-03 (§7.3). Owner check: the first scheduled run and the first Dependabot PRs
      appear only after merge to the default branch.
- [x] **[med] Offset-aware `lookahead_on` rule** (§6.2) — done 2026-11-03; the `close[1]`
      idiom and four variants are negative fixtures.
- [x] **[low] PyPI name check and trademark disclaimer** (§7.4, §8) — `pine-parity`;
      disclaimer in README. Name not yet *reserved* on PyPI (needs the owner's account).
      - _Raised:_ 2026-11-03
      - _Why:_ a name collision or trademark complaint after release is costlier than before.
      - _Done when:_ distribution name chosen; README disclaimer present.

## 10. Review 2 (2026-11-03): next steps

Objective assessment after P0, P2 and P3. What is strong: na-mask-first comparison, aligned
mode as the default, the mutation suite with exact expected bars and negative controls,
fixture-annotated lint tests with an injection test, and explicit `definite`/`possible`
confidence. What is weak: **all evidence is internal**. The recommendations below are ranked
by how much each reduces the chance of certifying a wrong port. Each has a checklist item in
§9 with a done-when.

### A. Evidence

**A1. Reference capture script and coverage matrix.** Write `tools/reference_capture.pine`, a
self-written, public-safe indicator that plots every core built-in with fixed inputs
(`sma/ema/rma/rsi/stdev(14)`, `highest/lowest(20)`, `atr(14)`, `crossover(ema 9, ema 21)` as
0/1), so oracle (c) needs one export per row rather than hand assembly. Required matrix before
any built-in is called "verified":

| Row | Purpose |
|---|---|
| price < 1 (e.g. an FX cross or low-priced asset) | `rtol` vs `atol` regime, many decimals |
| price ≈ 100 | baseline |
| price > 10 000 | `rtol` dominates; float64 headroom |
| symbol with opening gaps | true-range gap terms (§5.9 coverage note) |
| series with `na` (e.g. an indicator of a sparse series) | settles the `na`-window rule (A2) |
| ≥ 5 000 bars | convergence and summation drift (§5.6) |
| daily bars across a DST change | session/timestamp convention (§7.1) |

Tradeoff: one capture script is fast but couples every built-in to one chart setup; the matrix
rows compensate. Export terms are still open, so commit numbers only or keep files local.

**How to run A1/A2 and E1 (owner):** follow [`docs/CAPTURE.md`](docs/CAPTURE.md). Both
capture scripts go on one chart, there is one export, `python -m pine.capture manifest`
writes the manifest, and `python -m pine.capture check` runs every applicable check into one
report. (`pine.reference` and `pine.semantics` remain available individually.) The reference
part of the report has two sections:

- *Gap-free columns* in aligned mode. `src_na` doubles as the history-alignment check
  (§5.4): its gap pattern is keyed to `bar_index`, so a truncated export fails it, and then
  every other result is suspect. `stdev` vs `stdev_sample` settles the `biased` default:
  whichever column our population port matches is the default. Crossover is checked on the
  exported EMAs, isolating crossover logic, as exact 0/1 until §10 B2.
- *na policy*: for each `*_na` column, which of the three hypotheses matches. "none" means a
  fourth behaviour (record it and its first divergent bars); different answers per built-in
  are possible and reported as such.

Expected design limit: the capture script uses one set of lengths (14/20/9/21). A length-
dependent bug (e.g. an off-by-one only for even lengths) would not show; add a second capture
with different constants if any built-in's code branches on length.

**A2. Settle the `na`-window rule empirically.** Until an export decides, give the window
functions an `na_policy` (`"propagate"` as now, `"skip"` = extend the window back over `length`
non-`na` values), and add the other policy as a mutant so the harness is shown to separate
them. Remove the switch once settled. Tradeoff: temporary API surface; the alternative
(guessing) risks a systematic error that every current test would miss. *Status
2026-11-03: decided by export (§12) — windowed `skip`, recursive `skip_na_out`; adopted as
defaults. The switch stays as an explicit override: one export and one gap pattern decided
it, and keeping the alternatives costs little while a second capture is outstanding.*

**A3. Tolerances that account for rounded inputs.** If the export rounds inputs to `δ_in` and
outputs to `δ_out`, the port's legitimate error is bounded by `L·δ_in + δ_out`, where `L` is
the indicator's sup-norm sensitivity to its inputs:

| Indicator | Bound on `L` | Reason |
|---|---|---|
| SMA, EMA, RMA | 1 | convex combination of inputs (weights sum to 1) |
| highest, lowest | 1 | order statistics are 1-Lipschitz |
| stdev (population) | 1 | `σ = ‖P x‖/√n` with `P` the centring projection, `‖P‖ ≤ 1`; sample: `√(n/(n−1))` |
| ATR | 2 | each true-range term is a difference of two inputs; `max` and RMA are 1-Lipschitz |
| RSI | per bar, `min(100, 200 / (U′ + D′))` per unit `δ_in` | `RSI = 100·U/(U+D)`, `U, D` each perturbed by ≤ `2δ_in`; exact bound, see §5.1 |

RSI has no global bound: when `up + down` is small (flat markets) a one-tick rounding can move
RSI by many points, so its tolerance must be computed per bar. Tradeoff: per-indicator code,
but without it a rounded-input export would either fail correct ports or force a blanket
tolerance loose enough to pass wrong ones.

*Status 2026-11-03: done* — implemented as `pine.sensitivity` + `PortSpec.input_error` +
`check_export(input_rounding=...)`, opt-in (§5.1). The planned `propagated_tolerance` function
became a per-spec `input_error` so ports declare their own sensitivity.

**A4. Minimum detectable error (MDE).** A pass should say what it could not see. For a
recursive filter, a seed error `e₀` is detectable in aligned mode only if `e₀ > tol` at the
seed bar, so the report states "seed errors below `tol(w)` are undetectable". In convergence
mode, `e₀·(1−α)^exclude` must exceed `tol` — usually impossible by construction, which is
the §5.3 hazard stated as a number. Extend the mutation suite with a magnitude sweep (seed
error at 0.1×, 1×, 10× the tolerance) confirming the stated threshold. Tradeoff: reports get
longer; the alternative is a pass that implies more than it shows. *Status 2026-11-03: done
(§5.7).*

### B. Harness correctness

**B1. Two-bar crossover excuse.** `near_tie` is evaluated per bar, but `crossover` at bar t
depends on the comparisons at t and t−1. A legitimate flip caused by a tie at t−1 is
currently not excused. Add `crossover_excuse(a, b, tol) = near_tie_t | near_tie_{t−1}` and
use it in examples and tests. *Status 2026-11-03: done.*

**B2. Discrete columns and machine-readable reports.** `PortSpec` accepts numeric kinds only,
so an exported 0/1 signal column cannot be checked by `check_export`. Add `kind="discrete"`
routed to `compare_discrete`, and `ParityReport.to_dict()`/JSON so downstream CI can gate on
results without parsing Markdown. HTML stays deferred. *Status 2026-11-03: done.*

**B3. Composite convergence.** `convergence_bars` covers a single filter. For RSI (two RMAs of
a differenced series) and ATR (RMA of true range), the bound is the slowest internal filter's
plus the window warm-up (+1 bar for differencing). Document and test it the way the §5.3
hazard test does for EMA. *Status 2026-11-03: done (§5.3). Outcome differed from the sketch:
ATR reduces exactly to a single-filter bound, while RSI needs a per-bar, data-dependent bound
because its sensitivity scales with `1/(U′+D′)`.*

### C. Lint quality

**C1. Single-assignment tracing.** Resolve identifiers assigned once with `=` and never
reassigned with `:=`, so `src = close` followed by `request.security(..., src, lookahead_on)`
becomes `definite`. This removes the largest class of `possible` findings. Reassigned or
conditionally assigned variables stay `possible`. *Status 2026-11-03: done (§6.3).*

**C2. Block-aware alert gating.** Use indentation to find the `if` enclosing each `alert()`
and require `barstate.isconfirmed` in that condition (or `freq_once_per_bar_close`). This
closes the documented false negative where any `isconfirmed` anywhere in the script counts.
*Status 2026-11-03: done (§6.3).*

**C3. Measured precision and recall.** Run `pine-lint` over 30–50 open-source scripts kept
locally (never committed; licences vary), hand-label a sample, and record counts in §6.3:
target 100% precision for `definite` findings and a published rate of `possible` findings.
Tradeoff: a few hours of labelling; without it the rule set's quality is asserted, not
measured.

### D. Operations

**D1. Release.** After PR #1 merges and the history decision is made: CHANGELOG, tag
`v0.1.0`, consumers pin the commit SHA (§7.2).

**D2. CI.** A Hypothesis CI profile with `derandomize=True` so failures reproduce, plus a
scheduled weekly job with random seeds to keep searching; `pytest --cov` with a floor;
dependabot for the SHA-pinned actions. Tradeoff: deterministic CI finds fewer new
counterexamples, hence the weekly random run. *Status 2026-11-03: done (§7.3); a local
thorough run (10× examples) found no counterexample.*

**D3. Performance, only when needed.** The reference loops are O(n·L) pure Python. Add
vectorised fast paths only when a consumer needs more than ~10⁵ bars, each validated against
the loop reference with the existing negative-control pattern (§5.9); the loops remain the
oracle.

**D4. Split the documents.** This file now mixes plan, normative specification and progress
log. With D1, move §5–§6 to `docs/SPEC.md` (normative), progress notes to `CHANGELOG.md`, and
keep this file to plan, risks and checklist.

**D5. Owner-only actions.** Decide the history purge **before** merging PR #1; confirm
GitHub secret scanning and push protection are enabled (not verified from here); confirm the
TradingView export terms; reserve `pine-parity` on PyPI.

### Suggested order

A1 + A2 together (one export session settles both), then B1 and A3 (small, correctness
critical), then A4 and B2, then C1–C2, then D1–D2 at merge time. C3, B3, D3 and D4 follow.

## 11. Review 3 (2026-11-03): from built-ins to complex indicators

Goal restated: a Python port of **any** Pine indicator, including complex, stateful,
multi-timeframe ones, should produce the same values as TradingView, and the toolkit should
*prove* it per indicator, with the limits of the proof stated.

### Objective assessment

| Capability a complex indicator needs | Status | Consequence if missing |
|---|---|---|
| Built-ins correct in isolation | 11 built-ins; internal oracles only | unknown until A1 exports exist |
| External evidence (TradingView values) | tooling ready, **no exports** | every "pass" is self-referential |
| Pine execution model (bar loop, conditional evaluation, per-call-site state, `var`) | **absent** | ports diverge where state or branches matter, possibly only on some bars |
| Pine scalar semantics (`math.round`, `%`, int/float division, `na` comparisons, `nz`, `fixnan`) | **unverified, not implemented** | small value differences that propagate through state |
| Time, sessions, calendar | timestamps only | session resets (`ta.vwap`), time filters wrong |
| `request.security` | lint only | multi-timeframe logic cannot be ported |
| Built-in breadth | 11 of the commonly used ~40 | most real indicators need missing pieces |
| Localising a divergence inside a port | output columns only | debugging a complex port is guesswork |
| Strategy results (`strategy.*`) | out of scope so far | trade lists/P&L not comparable |

The harness, tolerance theory (A3/A4), lint and capture tooling are in good shape. The gap is
the **semantics layer between built-ins and whole scripts**, and evidence.

### E. Execution model (highest priority after the exports)

**E1. Observe before building: a semantics capture script.** Write
`tools/capture_semantics.pine`, analogous to the reference capture, with one plot per probe
and Python-side competing hypotheses decided by an export (the `na_policy` pattern of §10 A2):

- stateful built-in called inside `if cond` vs outside — does `ta.ema` update only on executed
  bars (expected), and what does the next unconditional read return;
- a user function with internal `[1]` history called from two call sites — separate histories
  (expected);
- `var` accumulation, `varip` on history (should equal `var`), history of a variable assigned
  only in some branches (`x[1]` after a non-executed assignment);
- scalar semantics: `math.round(2.5)`, `math.round(-2.5)`, `-7 % 3`, `7 / 2` for `int` operands,
  `int(-2.7)`, `na == na`, `nz`, `fixnan`;
- `ta.change`, `ta.valuewhen`, `ta.barssince` around `na` and before the first event.

These probes can go on the same chart and export session as A1/A2.

*Status 2026-11-03: tooling done.* `tools/capture_semantics.pine` + `pine.semantics`; run
through `python -m pine.capture check` per [`docs/CAPTURE.md`](docs/CAPTURE.md). 20 probe groups; the expected
hypothesis of each (what `pine.runtime` implements) and its alternatives:

| Group | Expected (runtime) | Alternatives |
|---|---|---|
| `cond_ema`, `cond_sma`, `cond_userfn`, `cond_crossover` | `per_call` | `every_bar` |
| `call_sites` | `separate` | `shared` |
| `branch_hist` | `per_bar` | `last_assigned` |
| `rounding` (2.5, −2.5, 3.5) | `half_away` (**confirmed** 2026-11-03) | `half_up` (the original assumption, refuted), `half_even` |
| `mod` (−7 % 3) | `truncated` (−1) | `floored` (2) |
| `div_series`, `div_const` (7 / 2) | `float` (3.5) | `int` (3) |
| `int_trunc` (int(−2.7)) | `toward_zero` | `floor` |
| `na_eq_na` | `false` | `true` |
| `barssince`, `valuewhen` | `na_before_first` | count from start / zero |
| `change_na` | `propagate` | `last_valid` |
| `regime` (state machine with `else if ta.crossunder`) | `per_call` | `every_bar` (they differ after a one-bar whipsaw, found on long synthetic data) |
| `alignment`, `var`, `supertrend`, `trail_stop` | single (parity checks) | — |
| `loop_builtin` (`math.sum` in a 3-iteration loop), `loop_userfn` (`x[1]`, `x[p]` in a function called in the loop) | `per_call` (one state per call site; history advances per call) | `per_bar_last` (one value per bar, the last call's), `per_iteration` (as if unrolled) |

Every expectation is **unverified**. Any mismatch makes the checker exit 1 and name the
group; the fix is to change the runtime (and its scalar helper) to the observed hypothesis.

**E2. `pine.runtime`.** A bar-by-bar interpreter-style framework, not a transpiler:

- `Series` values with bounded history (`s[n]`, `na` beyond history, `max_bars_back`);
- a script is a Python function run once per bar; stateful built-ins are **incremental
  objects keyed by call site** (e.g. `ctx.ema("fast", src, 9)`), updated only when called —
  reproducing conditional-evaluation semantics by construction;
- `var`/`varip` as explicit per-script state; `na`-aware scalar helpers implementing the E1
  outcomes.

Validation: (i) every incremental built-in equals the vectorised `pine.ta` on unconditional
use (a second, independent implementation: a new internal oracle), (ii) E1 probes,
(iii) three stateful reference ports — Supertrend (direction state), a ratcheting trailing
stop, and a pivot/zigzag state machine — passing parity against exports. Tradeoff: a Python
bar loop costs microseconds per operation (≈ seconds for 10⁵ bars × 50 calls); `pine.ta`
remains the fast path where no conditional semantics are involved.

*Status 2026-11-03: built.* `pine.runtime`: `run(script, bars)`, `ctx.series` / `ctx.var` /
`ctx.call` / `ctx.plot`, `ctx.ta.*` (sma, ema, rma, rsi, stdev, highest, lowest, tr, atr,
crossover, crossunder, change, barssince, valuewhen), `pine.runtime.scalar`. Keys are explicit
strings; reusing a key in one bar raises (it caught a copy-paste in the capture port during
development). History is per execution of a declaration, so the same mechanism gives per-bar
history at top level and per-call history inside branches and functions. Only the `propagate`
na policy is implemented until A2 is decided. Validation without exports: bit-for-bit equality
with `pine.ta`, exact reproduction of every expected probe hypothesis, and the Supertrend port
equal to an independent loop implementation.

*Update 2026-11-03: `for` loops.* `ctx.loop(key, iterable)` runs a loop body in Pine's shape:
keys may repeat across iterations, and each call site in the body keeps one state shared by
all iterations (hypothesis `per_call` of the `loop_*` probes, **unverified**). Should an
export show `per_bar_last` or `per_iteration`, only `ctx.loop` changes. Why it matters: ports
that call stateful built-ins or history-reading functions inside loops (common in
optimisation-style indicators) are otherwise silently wrong in one of three ways.

### F. Built-in breadth

Batch 1, chosen by expected use in public indicators (to be re-ranked with C3's corpus
counts): `ta.change`, `wma`, `vwma`, `hma`, `alma`, `swma`, `macd`, `bb`, `kc`, `stoch`,
`cci`, `mfi`, `dmi`/ADX, `supertrend`, `sar`, `linreg`, `correlation`, `percentrank`, `cum`,
`valuewhen`, `barssince`, `pivothigh`/`pivotlow`, `vwap`. Each must meet §5.8 including oracle
(c), declare its A3 input sensitivity, `alpha` where seeded (A4), get mutation cases, and
columns in a capture script (Pine's 64-plot limit means several captures, each with the
consistency tests used for `reference_capture.pine`). Stateful/discrete outputs
(Supertrend/SAR direction, pivots) are compared with `compare_discrete` and excuses.
`pivothigh/low` and `valuewhen` also need delay semantics checked (a pivot is confirmed `right`
bars later).

*Status 2026-11-03: first subset built, export pending.* `tools/capture_builtins.pine`
(19 plots) and `pine.builtin_capture`, run by `python -m pine.capture check`. Each group's
hypotheses are independent numpy implementations; the expected one is what `pine.ta` /
`pine.runtime` implement:

| Group | Expected | Alternatives |
|---|---|---|
| `sum`, `wma`, `variance`, `percentile`, `cum` (gap-free) | `window` / `linear_weights` / `population` / `inclusive` (rank p(n−1)) / `running_total` | `variance`: `sample`; `percentile`: `exclusive` (rank p(n+1)−1), `hazen` (np − ½) |
| `sum_na`, `wma_na`, `variance_na`, `percentile_na`, `cum_na` | `skip` (by analogy with the observed windowed built-ins) | `skip_na_out`, `propagate` |
| `hma` (lengths 9 and 14) | `half_floor_sqrt_floor` | the other three floor/round combinations |
| `pivots` (`high`/`low`, 3 left, 2 right) | `left_strict` (> left bars, ≥ right bars: a flat top marks its leftmost bar) | `right_strict`, `strict`, `weak` |
| `fixnan`, `obv`, `pvt`, `alignment` | single | — |

*Input-rounding bounds (A3), 2026-11-04* (`pine.sensitivity`, each tested for soundness by
perturbing inputs within ±δ): `wma`, `percentile_linear_interpolation`, `fixnan` 1;
`math.sum` `length`; `hma` 3 (`2·wma − wma` moves by ≤ 3δ, the outer wma is 1-Lipschitz);
`variance` per bar `(2σ' + cδ)·cδ` with `c = √(n/k)`, not globally Lipschitz; `cum` `m·δ` after
`m` values, unbounded in time; `obv` `Σ 2·volume` over bars whose port-side change is within
`2δ` (the sign can flip there; volume assumed exact); `pvt` per term
`v·δ(p' + |c'|)/((p' − δ)p')`. Pivots have no bound (rounding can create or remove a pivot)
and are compared without an allowance. Practical consequence: with rounded inputs the
running-total bounds grow with history, so long exports keep little detection power on
`cum`/`obv`/`pvt`. The checker uses the full-history bound even for anchored (truncated)
comparisons: valid, but wider than the true error since the anchor.

Pivot ties need equal highs or lows in the export; on prices with few decimals they occur,
otherwise the group reports "not decidable". `obv`/`pvt` use the exported volume (the script
plots it) and are na on symbols without volume. Truncated exports: 25 warm-up rows excluded,
running totals anchored on the export and their increments checked.

### G. Time, sessions, multi-timeframe

**G1. Sessions and calendar.** Extend `Bars` (or the manifest) with a session spec; implement
`time_close`, day/week boundaries in the exchange time zone, session first/last bar, DST.
Needed for `ta.vwap` (session-anchored by default) and any session-filtered logic. Verified
with a capture across a DST change (already a matrix row, §10 A1).

**G2. `request.security`.** Build higher-timeframe bars from chart bars with TradingView's
alignment rules; implement `lookahead` on/off and `gaps` on/off; the non-repainting idiom
returns the previous *completed* HTF bar. **Known limit:** TradingView's HTF values come from
its own HTF feed, which can differ from aggregated chart bars (futures settlement, extended
hours, adjustments). Therefore: validate with a capture that plots `request.security` results;
where the feed differs, support supplying the exported HTF series as an input instead of
reconstructing it, and say so in the port manifest. `request.security_lower_tf` deferred.

### H. Port acceptance and evidence

**H1. Port acceptance protocol** for any non-trivial indicator:

1. `pine-lint` the source; parity claims apply to **confirmed historical bars** only, and a
   repainting script's live behaviour is out of scope by construction.
2. Instrument the Pine source with debug plots of key intermediate series.
3. Export over the relevant §10 A1 matrix rows.
4. `check_port` (new): aligned parity on outputs **and** intermediates; reports the
   upstream-most intermediate that diverges and its first bar (divergence localisation), the
   MDE of every column, and excused discrete flips. *Built 2026-11-03* (`pine.port`): root
   cause = the failing stage with the earliest first divergence (ties: upstream); shown on
   synthetic pipelines to name a broken first stage even though every later stage also fails.
5. Write a port manifest: Pine source hash and version, inputs, port version, matrix rows
   passed, tolerances, MDE, date.

**H2. Verification registry.** A checked-in `verification.yaml` recording, per built-in and
per accepted port, which oracles passed and on which matrix rows; tests fail if a public
`ta` function lacks an entry; the README status table is generated from it. "Verified" becomes
data, not prose.

**H3. Re-capture cadence.** TradingView can change built-ins, data feeds or Pine versions.
Re-run the capture exports quarterly and on any Pine version change; compare with stored
numbers; a difference is either a TradingView change (record it) or a regression.

**H4. Error budgets for long chains.** In a deep port, rounding allowances compose:
sensitivities multiply along the chain (Lipschitz composition), and RSI-like stages amplify.
Later option: have `pine.runtime` propagate per-bar error bounds alongside values (interval
arithmetic) so A3 allowances follow automatically for any port. Tradeoff: complexity and
over-conservative bounds through `max`/branches; defer until a real port needs it.

### I. Strategies (decision for the owner)

If "same results" includes backtests, `strategy.*` needs a broker emulator (order types, the
OHLC path assumption for intrabar fills, `process_orders_on_close`, pyramiding, commission,
slippage, margin) and a different reference: TradingView's *List of Trades* export, compared
trade by trade. This is a project of similar size to E–G. Recommendation: finish indicator
parity first; decide explicitly whether strategy parity is a goal. *Decided 2026-11-03 by the
owner: **in scope**. Plan and first increment: §13.*

### J. Transpiler, last

A Pine → Python transpiler (P4) should target `pine.runtime`, not emit vectorised code: it
then only translates syntax, while semantics live in one validated place. Building it before
E–H would multiply unverified semantics.

### Suggested order

1. **Owner:** A1/A2 exports, adding the E1 semantics probes to the same session.
2. E1 checker and hypotheses → E2 runtime with the three stateful reference ports.
3. H1 protocol + `check_port`, H2 registry.
4. F batch 1, re-ranked by C3 corpus counts (C3 doubles as a usage survey).
5. G1 sessions → G2 `request.security`.
6. Owner decision on I; J only after the above.

What would most change the plan: an export showing that a current assumption is wrong (seeding,
`na` handling, rounding). That is cheap to obtain and invalidates work built on top of it, so
it stays first.

## 12. Evidence log (external results)

Results from real TradingView exports. Each row records what a specific export established,
so "verified" claims can be traced. Exports themselves are kept outside the repository (§8).

### 2026-11-03 — BATS:NFLX, 1D, both capture scripts

Export rows 3035; first row `bar_index` 3116 of 6151 (the export started mid-history, which the
checkers did not yet handle: most rows failed on warm-up `na` and history-dependent values).
What it nevertheless established:

| Item | Result |
|---|---|
| `ta.sma`, `ta.stdev` (population and sample), `ta.highest`, `ta.lowest`, `ta.tr` | **match** once Python's window is full: max error 1.3e-12 (stdev), 4.3e-14 (sma), exactly 0 (highest, lowest, tr) |
| `ta.crossover`, `ta.crossunder` (on exported EMAs) | **match** exactly |
| `ta.ema`, `ta.rma`, `ta.atr`, `ta.rsi` | error largest at Python's seed bar and shrinking: consistent with a pure seed difference; recursion not yet verified (needs the decay check, now implemented) |
| `math.round` | **ties away from zero** — runtime changed (was `floor(x + 0.5)`) |
| `%`, int division, `int()`, `na == na` | **as expected**: −1, 3.5, −2, false |
| execution-model probes, `na` policy, ports | not decidable from this run (history offset); re-run `python -m pine.capture check` on the same file with the offset-aware checker |

Consequences: `pine.truncation` (offset-aware, history-robust checks), `min_compared`
(no vacuous passes), the regime probe split into `per_call`/`every_bar`, and a test-assumption
bug fixed in the RSI affine-invariance property test (found by a random-seed run while
validating these changes).

### 2026-11-03 — same export, re-checked with the offset-aware checker

**Overall: PASS** (exit 0). With the export's `bar_index` offset (3116) handled:

| Item | Result |
|---|---|
| `sma`, `ema`, `rma`, `rsi`, `atr`, `stdev` (both), `highest`, `lowest`, `tr` (both), `ema(9)`, `ema(21)` | **match** over 2647–3034 bars each; max abs error 7.7e-12 (rsi), all others ≤ 1.3e-12, highest/lowest/tr exactly 0 |
| EMA / RMA / ATR recursion and `α` | **verified on every bar** by the exact residual-decay check; seeds remain unverified (export starts mid-history) |
| `crossover`, `crossunder` | **match** exactly (0 excused flips) |
| `na` inside windows | **decided**: `sma`/`stdev`/`highest`/`lowest` → `skip`; `ema`/`rma`/`rsi` → `skip_na_out` (each column matched exactly one policy) — adopted as defaults |
| Execution model | **confirmed**: conditional `ta.ema`/`ta.sma`/user function/`ta.crossover` advance per call; call sites separate; `var`/`varip` increments; branch history per bar; `ta.change` propagates `na` |
| Scalar rules | **confirmed**: rounding ties away from zero, truncated `%`, fractional int division, `int()` toward zero, `na == na` false |
| Ports | Supertrend (manual code), ratcheting stop, regime machine (`per_call`): **match** after synchronising at rows 259 / 369 / 321 |
| `barssince` / `valuewhen` before the first event | not decidable from this export (needs one starting at bar 0) |

Changes made on this evidence: `pine.ta.DEFAULT_NA_POLICY` and per-built-in defaults in
`pine.ta` and `pine.runtime` (incremental built-ins now implement `skip` / `skip_na_out`,
equal to `pine.ta` bit for bit under every policy); mutation-suite expectations re-derived
for the new defaults (`highest` with `na → 0` is detected three bars after the gap on the
fixture, data-dependently, since a 0 cannot raise a maximum).

Closed by the weekly export below: seeding and pre-event `barssince`/`valuewhen`.

### 2026-11-03 — BATS:NFLX, 1W, both capture scripts (starts at bar 0)

1277 rows, `bar_index` 0 → 1276: the whole history in one export, so every check ran in
**aligned mode with zero rows excluded**. **Overall: PASS.**

| Item | Result |
|---|---|
| All 13 numeric built-in columns | **match** over 1257–1277 bars; max abs error 4.3e-13 (stdev), others ≤ 1.1e-13, highest/lowest/tr exactly 0 |
| **Seeding** of `ema`, `rma`, `atr`, `rsi`, `ema(9)`, `ema(21)` | **verified**: first values match; seed-error detection threshold (MDE, §5.7) ≈ 2e-10 |
| `crossover`, `crossunder` | **match** exactly |
| `na` inside windows | **same decision as the daily export** (second, independent confirmation): windowed `skip`, recursive `skip_na_out` |
| All 20 semantics groups | **match** from bar 0, no synchronisation: incl. `barssince`/`valuewhen` before the first event (`na`), Supertrend through its ATR warm-up (lower band held at `nz(na) = 0`), trailing stop, regime machine (`per_call`) |

With both exports, the eleven core built-ins meet the two-oracle rule (§5.8: closed-form or
property oracles plus TradingView values) on: daily and weekly bars, a 6 151-bar history,
opening gaps and daylight-saving changes (both inherent in NFLX daily), an export starting
at bar 0 and one starting mid-history. **Remaining matrix rows (§10 A1):** price < 1,
price > 10 000, an intraday timeframe (sessions/time zones). Built-ins outside the core
eleven, `request.security` and sessions remain unverified.

### 2026-11-03 — BINANCE:DOGEUSDT 1D, BITSTAMP:BTCUSD 1D, BATS:AAPL 60 (matrix rows)

Indicator and semantics captures only (the strategy capture was not yet in the checker the
owner ran, so §13 remains unverified).

| Export | Start | Indicators | `na` policy | Semantics |
|---|---|---|---|---|
| DOGEUSDT 1D (price < 1) | bar 0, 2 679 rows | all 16 columns **match**, aligned, incl. seeding (max error ≤ 4.3e-14) | same decision | all match except `varip_acc` on the **last row only** (below) |
| BTCUSD 1D (price > 10 000) | bar 2662 | all **match** (max abs 1.6e-9 on stdev = 3.5e-12 relative; `rtol` regime) | same decision | all match except the same last-row `varip_acc`; `barssince`/`valuewhen` not decidable |
| AAPL 60 (intraday, sessions) | bar 17845 | all **match** | same decision | all match; `regime` not decidable (no whipsaw after sync) |

**`varip` on the last bar.** On the two 24/7 crypto charts the last exported bar was live;
there `varip_acc` exceeded `var_acc` while every earlier row matched exactly. That is
documented `varip` behaviour (it keeps intrabar ticks on the realtime bar), not a runtime
error: the AAPL export, taken with the market closed, shows no difference. The checker now
excludes the last row for `varip_acc` and reports when the last bar was live
(`pine.semantics`, test `test_live_last_bar_varip_is_excluded_and_reported`).

**Coverage.** With NFLX 1D/1W these five exports cover: price scales ~0.1, ~100–1 000 and
> 10 000; daily, weekly and hourly bars; equities and 24/7 crypto; opening gaps; DST; exports
from bar 0 and mid-history. The `na`-handling decision was identical in all five. The core
eleven built-ins, `na` handling, execution model and scalar rules are verified across the
§10 A1 matrix. Still unverified: the strategy emulator (§13), built-ins outside the core,
sessions as a *feature* (AAPL hourly only shows that session bars don't break anything), and
`request.security`.

### 2026-11-03 — first strategy export (BINANCE:DOGEUSDT 1D, from bar 0)

The strategy section failed under every original fill-rule hypothesis, but the per-bar values
pinned down why: **TradingView's broker emulator works on prices rounded to the symbol's tick**
(`syminfo.mintick` = 0.00001 here) although the historical OHLC carry 7 decimals.

| Row | Raw | TradingView | Explained by |
|---|---|---|---|
| 1 (entry) | open 0.0038741 | avg price 0.00387 | market fill at the open rounded to the nearest tick |
| 9 (exit) | open 0.0032474 | net −0.00062 = 0.00325 − 0.00387 | same |
| 31 (bracket) | limit 0.0029222 × 1.01 = 0.0029514 | +0.00004 from entry 0.00292 → exit 0.00296 | sell-limit price rounded *up* |
| 471 (bracket) | high 0.0026055 < limit 0.0026092 | exits on that bar, +0.00003 | high rounded to 0.00261 ≥ limit rounded to 0.00261 |

Position timing matched for 470 bars before the first rounding-driven divergence, consistent
with `fill_timing = next_open`. Also observed: a strategy is not calculated on the live bar
until it closes (all strategy columns `na` on the last row). Changes: `StrategyConfig.mintick`
with `price_rounding` / `order_rounding` rules (new probe groups), `strat_mintick` plotted by
the capture script, `--mintick` for exports without it; the four rows above are regression
tests.

**Re-check with `--mintick 0.00001`:** position size and closed-trade count then matched over
the whole 2 678-bar history. Remaining price differences traced to two causes, both fixed:
decimal ties round *down* in TradingView because it rounds in binary floating point
(0.002695 → 0.00269, 0.002665 → 0.00266; rows 44, 133), and the intrabar path is judged on the
raw bar (row 231: rounded bar symmetric around the open, raw low nearer, TradingView hit the
stop). Both are regression tests; `path_basis` is a new probe group.

**Final re-check (same export, `--mintick 0.00001`): Overall PASS.** The emulator reproduces
TradingView's strategy state on every one of 2 678 bars — position size, average price, open
and net profit, equity, closed-trade count — with every fill exact to the tick. Rules decided:
`fill_timing = next_open`, `path = nearest_extreme_first`, `exit_same_bar = yes`,
`price_rounding = tick`, `order_rounding = conservative` (sell limit observed; stops still by
symmetry), `path_basis = raw`. **Not decidable here:** `gap_fill` — on a 24/7 crypto series the
open equals the previous close, so no gap ever jumped a bracket level; needs an equity export
(e.g. NFLX 1D) with the strategy script.

### 2026-11-05 — BATS:NFLX 1D, reference + built-in + strategy captures (from bar 2849)

3 304 rows. The semantics capture was not on the chart, so the loop probes remain open.

| Item | Result |
|---|---|
| Core built-ins, `na` policies | **match** again (third NFLX daily confirmation of the `na` decision) |
| `math.sum`, `wma`, `hma`, `variance`, `fixnan`, `cum`, `obv`, `pvt` (gap-free) | **match** as implemented; `hma` lengths floor(n/2), floor(√n) |
| `na` handling | `sum`, `variance` → `skip`; **`cum` → `skip_na_out`** (implementation was `skip`; changed) |
| `percentile_linear_interpolation` | **Hazen rank** `p·n − ½` (implementation used `p(n − 1)`; changed) |
| `pivothigh` / `pivotlow` ties | **flat top marks its rightmost bar** (≥ left, > right; implementation had it reversed; changed) |
| `wma` with `na` | **none of skip / skip_na_out / propagate**: na on the `na` bar but different values after. Two more hypotheses added (fixed window with `na` terms dropped, renormalised or not); re-run the checker on the same file |
| `percentile` with `na` | not decidable yet: its hypotheses used the wrong rank; re-run on the same file |
| `timeframe.from_seconds(90000)` | **none of** "1500" / "1D": more hypotheses added and the checker now prints the observed value |
| `barstate`, `timeframe.*` constants, `in_seconds("1M")` = 2 628 003 | **match** (last bar was closed) |
| `timeframe.change` on the first bar | not decidable (export starts mid-history) |
| Strategy | all rules **match**; **`gap_fill = open` decided** (stops/limits gapped past fill at the open); `path_basis` not decidable here (already decided on DOGE) |

**Re-check of the same file with the extra hypotheses:** `percentile` → **Hazen** confirmed;
`timeframe.from_seconds(90000)` → **"2D"** (`ceil_day`: a day or more rounds *up* to whole
days; implemented). `wma` and `percentile` with `na` still match **none** of five / four
hypotheses. Rather than guess again, `python -m pine.capture rows <csv> --group wma_na
--from 40` now prints TradingView's values next to every hypothesis around a row, so the
rule can be read off the data.

**Read off the data (`rows`, NFLX rows 44–63, gap at 48–50), both reproduced on every row of
the excerpt with a full window (11 rows each, ≤ 1e-9):**

- `ta.wma` → **`fill_na_out`**: each `na` in the window takes the last valid value before it;
  the output is `na` only on bars whose own source is `na`.
- `ta.percentile_linear_interpolation` → **`fill_key`**: `na` entries sort as if they were the
  last valid value (after real values with the same key); the result is `na` when the
  interpolation uses an `na` entry (rows 54–57), while the `na` bars themselves keep a value.

Neither matches the windowed built-ins' `skip`: `na` handling differs per built-in, so every
new built-in needs its own `na` column. The excerpt's 20 closes and TradingView's values are
committed as a regression test (numbers only, as for the DOGE strategy rows). `hma` is built
from `wma` and follows it by analogy, unverified.

### 2026-11-05 — BATS:NFLX 1W, all four captures (from bar 0)

1 277 rows, aligned mode throughout.

| Item | Result |
|---|---|
| Core built-ins, `na` policies, the 20 original semantics groups | **match** again |
| **Calls inside `for` loops** (`loop_builtin`, `loop_userfn`) | **`per_bar_last`**: a call site in a loop keeps one history entry per bar, the last iteration's (`math.sum(x, 2)` in the loop = current input + previous bar's last input; `x[1]` = previous bar's last value). `per_call` (the runtime's guess) and `per_iteration` did not match. **Runtime changed** (`ctx.loop` restores each site to its start-of-bar state on every call after the first) |
| `timeframe.change` on the chart's first bar | **false** (`first_bar_false`; implementation was true; changed) |
| Batch-F built-ins incl. `wma` `na` rule, `from_seconds`, `in_seconds`, `timeframe` | **match** (second export) |
| `percentile` with `na` | `fill_key` **fails** at row 52 (it matched every comparable row of the daily export): the rule is not yet right in some case; needs the `rows` output |
| `barstate` on the last bar | **neither** closed nor live: the last weekly bar was still open but the market was closed at export (`varip` saw no ticks). `isconfirmed` = 0 there and `islastconfirmedhistory` on the bar before. Three more hypotheses added (realtime/not, new/not); needs the `rows` output |
| Strategy | **not checked**: run without `--mintick` (my command omitted it), so tick rounding was off; the failures match the pre-rounding pattern of the first DOGE run. Re-check with `--mintick 0.01` |

## 13. Strategy parity (`strategy.*`)

Decided in scope on 2026-11-03 (§11 I). Goal: a Python port of a Pine strategy reproduces
TradingView's per-bar strategy state — position size, average price, net profit, open profit,
equity, closed-trade count — and, later, its trade list.

**Reference.** The chart-data export already used for indicators: a strategy can `plot()`
`strategy.position_size`, `strategy.netprofit` etc., so per-bar parity needs no new export
format and reuses the whole harness (tolerances, MDE, truncation handling). A trade-by-trade
comparison against the Strategy Tester's *List of Trades* export comes later; that export's
availability and format are **unverified**.

**Emulator.** `pine.strategy.Broker`, attached by `run(script, bars, strategy=StrategyConfig())`
as `ctx.strategy` (`entry`, `close`, `close_all`, `exit(stop=, limit=)`, and the state
properties). Each fill rule is a probed hypothesis, as for the execution model (§11 E1):

| Rule | Expected (default) | Alternative | Why it matters |
|---|---|---|---|
| fill timing | `next_open`: orders from bar t fill at bar t+1's open | `same_close` | every fill price |
| intrabar path | `nearest_extreme_first`: open → nearer extreme → other extreme → close (ties: high first) | `high_first` | which of stop/limit fills when both are inside a bar |
| gap fill | `open`: a level already passed at the open fills at the open | `order_price` | stops through gaps |
| exit on the entry bar | `yes`: a bracket can fill on the bar its entry filled | `no` | same-bar exits |
| price rounding | `tick`: OHLC rounded to the nearest `syminfo.mintick`; market orders fill at the rounded open | `none` | **observed** (DOGE 1D): every fill price |
| order-price rounding | `conservative`: sell limit / buy stop rounded up, buy limit / sell stop down | `nearest` | observed once for a sell limit; stops by symmetry |
| path basis | `raw`: which extreme is nearer the open is judged on the unrounded bar; levels are checked on the rounded one | `rounded` (ties high first) | observed once (DOGE row 231) |

Tick rounding is plain float `floor(price / mintick + 0.5)`: a decimal tie such as 0.002695 is
269.49999999999994 ticks in binary and rounds *down* (observed, DOGE rows 44 and 133).

The first four expectations are **unverified**; the rounding rules come from the first
strategy export (§12). `tools/capture_strategy.pine` is a deterministic
40-bar cycle (signals from `bar_index` only): long entry, close, short entry, a reversal to
long, close-all, and a long with a ±1 % stop/limit bracket. `python -m pine.capture check`
compares its six plotted columns with the emulator under each hypothesis (`pine.strategy_capture`).
Exports that start mid-history are compared from the first flat bar (phase 39), with
cumulative columns anchored there.

Verification so far: hand-computed scenarios (fills, reversal, pyramiding 1, brackets along
both paths, gaps, same-bar exits, same-close timing) and accounting identities
(equity = capital + net + open profit; net profit = sum of trades); on long synthetic data
every rule's hypotheses are distinguishable and each is recovered from an export made with it.

**Increments.**

| Step | Adds | Status |
|---|---|---|
| S1 | market entries/closes/reversals, stop/limit brackets, fixed qty, no costs, pyramiding 1, tick rounding | **verified** (DOGE 1D, all bars); `gap_fill` pending an equity export |
| S2 | commission (percent / per contract), slippage (ticks), qty types (percent of equity, cash), pyramiding > 1, `strategy.order`, trailing stops (`trail_points`/`trail_offset`) | — |
| S3 | `process_orders_on_close`, `calc_on_order_fills`, margin calls, `strategy.risk.*` | — |
| S4 | trade-list comparison (List of Trades export), strategy reports (max drawdown, profit factor) | — |

**Known limits.** The intrabar path is TradingView's *assumption*, not real tick order; the
emulator reproduces the assumption, which is the goal (parity), not market reality. The Bar
Magnifier option (intrabar data) changes fills and is out of scope until S3+. Results apply to
historical bars only (§8).
