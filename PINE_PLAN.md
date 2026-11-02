# pine: public Pine Script → Python toolkit (plan)

Status, 2026-11-03: **P0, P2 and P3 built**; P1 built for the core subset (§7.5); P4 not
started. **No built-in is verified against TradingView output yet** (§5.8): every check so
far is self-referential (§8, §10). `pine` is a **public** repository (GitHub). Sections:
parity spec (§5), lint (§6), data and operations (§7), risks (§8), checklist (§9), second
review and next steps (§10).

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
| P1 | `pine.ta`: core built-ins with Pine semantics (see §7.5 for scope) | every built-in meets the two-oracle rule (§5.8), incl. warm-up `na` count | core subset built; oracles (a)+(b) only, (c) missing → **not done** |
| P2 | `pine.parity`: tolerance report (§5.1–5.7), first divergent bar, warm-up/convergence modes, HTML/MD report | every mutation in the suite (§5.9) is detected at its expected bar | done; Markdown report only (HTML deferred) |
| P3 | `pine.lint`: repainting rules with line numbers (§6) | each rule has a positive and negative fixture, incl. the safe `lookahead_on` idiom as a negative; zero findings on a known-clean script | done; precision/recall on real scripts unmeasured (§10 C3) |
| P4 (optional) | transpiler for a declared Pine v5/v6 subset (series, `var`, `[]`, `if`, `for`, `ta.*`, `math.*`) | transpiled output passes `pine.parity` on 10 public indicators | not started |

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

### 5.4 History alignment

Pine's state depends on the first bar loaded on the chart, not the first bar the user looks
at. Python must start on the same first bar as the export. If it cannot (truncated export,
data gap), only convergence mode (§5.3) applies.

### 5.5 Discrete outputs

Booleans and categorical outputs (`ta.crossover`, Supertrend direction, signal flags) are
compared exactly. A mismatch is **excused** only if the continuous inputs to the decision at
that bar are within tolerance of the tie (e.g. `|a − b| ≤ atol + rtol·|b|` for a crossover of
`a` over `b`). Excused flips are counted and listed in the report, never silently passed.
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
- **possible systematic difference:** recent manual pages may say that `na` values in
  `source` are *ignored*, with the window extending back over `length` non-`na` values. If
  so, the propagate-`na` behaviour used here for `sma`, `ema`, `rma`, `stdev`, `highest` and
  `lowest` is wrong at gaps (it is unaffected on gap-free data). Settle with an export of a
  series containing `na` before any consumer relies on behaviour at gaps;
- facts the lint rules rest on **(unverified)**: `lookahead` is the 5th positional parameter
  of `request.security` (after symbol, timeframe, expression, gaps); `alert()`'s default
  `freq` is `alert.freq_once_per_bar` (fires intrabar); a negative literal history index
  does not compile; `barstate.isnew`/`islast` differ between history and realtime as
  described;
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
| `alert-unconfirmed` | realtime-repaint | warning/possible | `alert()` without `freq = alert.freq_once_per_bar_close`, in a script with no `barstate.isconfirmed` |
| `alertcondition-unconfirmed` | realtime-repaint | info/possible | `alertcondition()` in a script with no `barstate.isconfirmed` |
| `fill-timing` | fill-timing | info/definite | `calc_on_order_fills = true`, `process_orders_on_close = true` |
| `negative-history-index` | future-leak | error/definite | `x[-n]`, literal `n` |
| `negative-plot-offset` | display | info/definite | `plot*`/`bgcolor` with literal `offset = -n` |
| `unsupported-version` | meta | info/definite | `//@version` missing or < 5 |

**Offset proof (§6.2).** An expression is *proven* offset when its outermost form is
`atom[k]`, `(expr)[k]` or `f(...)[k]` with literal `k ≥ 1`, or every built-in price/time
series in it carries `[k ≥ 1]` and no user identifier appears outside a second-or-later call
argument. It is a *definite leak* when a built-in series is read unoffset outside any
bare-name call. Everything else is *unproven*. Known error sources:

- **False negative:** a user series passed as a second-or-later call argument
  (`math.max(close[1], mySeries)`) is assumed to be a parameter such as a length.
- **False negative:** `alert-unconfirmed` accepts any reference to `barstate.isconfirmed` in
  the script as a gate, whether or not it gates that alert.
- **Conservative ("possible"):** user variables are not traced (`src = close` then
  `request.security(..., src, lookahead_on)`); bare-name calls (`nz(close)`, user functions)
  may offset internally.
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
- [ ] **[high] A1 — Reference values for the core built-ins via a capture script and a
      coverage matrix** (§10 A1; supersedes the earlier "commit reference values" item)
      - _Raised:_ 2026-11-03
      - _Why:_ the two-oracle rule (§5.8) needs oracle (c); without it the ports are checked
        only against their own reading of the manual, which shares any misremembering.
      - _Done when:_ `tools/reference_capture.pine` exists, and every core built-in passes
        `check_export` in aligned mode on every row of the §10 A1 matrix, each export with a
        manifest (numbers only or kept local, per the terms item).
- [ ] **[high] A2 — `na_policy` switch and the opposite-policy mutant** (§10 A2)
      - _Raised:_ 2026-11-03
      - _Why:_ the `na`-window rule is the single most consequential unverified semantic.
      - _Done when:_ both policies implemented, the mutant is detected, and an export with `na`
        selects one; the switch is then removed.
- [ ] **[high] A3 — Input-rounding-aware tolerances** (§10 A3)
      - _Raised:_ 2026-11-03
      - _Why:_ rounded OHLC in an export produces output error the current floor ignores.
      - _Done when:_ `propagated_tolerance` with the §10 A3 bounds, each bound tested by ±δ
        input perturbation, and `check_export` uses it when the input decimals are known.
- [ ] **[high] A4 — Minimum detectable error in reports** (§10 A4)
      - _Raised:_ 2026-11-03
      - _Why:_ a pass without a sensitivity statement overstates what was shown.
      - _Done when:_ each `ColumnReport` states its seed-error detection threshold, and a
        magnitude sweep in the mutation suite confirms it.
- [ ] **[high] B1 — Two-bar crossover excuse** (§10 B1)
      - _Raised:_ 2026-11-03
      - _Why:_ `near_tie` is per bar; a crossover depends on bars t and t−1, so legitimate flips
        caused by a tie at t−1 currently fail.
      - _Done when:_ `crossover_excuse` exists and a test with a tie at t−1 passes only with it.
- [ ] **[med] B2 — Discrete columns in `check_export`; JSON parity report** (§10 B2)
      - _Raised:_ 2026-11-03
      - _Why:_ exported signal columns cannot be checked today; consumers' CI needs JSON.
      - _Done when:_ `PortSpec(kind="discrete")` routes to `compare_discrete`;
        `ParityReport.to_dict()` exists.
- [ ] **[med] B3 — Composite convergence bounds for RSI and ATR** (§10 B3)
      - _Raised:_ 2026-11-03
      - _Why:_ `convergence_bars` covers one filter; RSI/ATR compose filters with windows.
      - _Done when:_ documented bound plus a test like the §5.3 hazard test for RSI.
- [ ] **[med] C1 — Lint: single-assignment tracing** (§10 C1)
      - _Raised:_ 2026-11-03
      - _Why:_ untraced variables are the largest source of `possible` findings.
      - _Done when:_ `src = close` then `request.security(..., src, lookahead_on)` is
        `definite`; reassigned variables stay `possible`.
- [ ] **[med] C2 — Lint: block-aware alert gating** (§10 C2)
      - _Raised:_ 2026-11-03
      - _Why:_ closes the documented false negative in `alert-unconfirmed`.
      - _Done when:_ an `alert()` outside an `isconfirmed`-gated block is reported even when
        `barstate.isconfirmed` appears elsewhere.
- [ ] **[med] C3 — Lint precision/recall on a local corpus** (§10 C3)
      - _Raised:_ 2026-11-03
      - _Why:_ rule quality on real scripts is unknown.
      - _Done when:_ results recorded in §6.3 (counts only, no script content); `definite`
        precision is 100% or the offending rule is fixed.
- [ ] **[low] D1 — v0.1.0 release and CHANGELOG** (§10 D1)
      - _Raised:_ 2026-11-03
      - _Why:_ consumers need a pinned, documented version (§7.2).
      - _Done when:_ tag on `main` after the history decision; CHANGELOG lists the API.
- [ ] **[low] D2 — Deterministic CI profile, weekly random run, coverage floor** (§10 D2)
      - _Raised:_ 2026-11-03
      - _Why:_ property tests are currently random per run; coverage is unmeasured.
      - _Done when:_ CI derandomized; scheduled job with random seeds; `--cov` with a floor.
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

**A2. Settle the `na`-window rule empirically.** Until an export decides, give the window
functions an `na_policy` (`"propagate"` as now, `"skip"` = extend the window back over `length`
non-`na` values), and add the other policy as a mutant so the harness is shown to separate
them. Remove the switch once settled. Tradeoff: temporary API surface; the alternative
(guessing) risks a systematic error that every current test would miss.

**A3. Tolerances that account for rounded inputs.** If the export rounds inputs to `δ_in` and
outputs to `δ_out`, the port's legitimate error is bounded by `L·δ_in + δ_out`, where `L` is
the indicator's sup-norm sensitivity to its inputs:

| Indicator | Bound on `L` | Reason |
|---|---|---|
| SMA, EMA, RMA | 1 | convex combination of inputs (weights sum to 1) |
| highest, lowest | 1 | order statistics are 1-Lipschitz |
| stdev (population) | 1 | `σ = ‖P x‖/√n` with `P` the centring projection, `‖P‖ ≤ 1`; sample: `√(n/(n−1))` |
| ATR | 2 | each true-range term is a difference of two inputs; `max` and RMA are 1-Lipschitz |
| RSI | per bar, ≈ `200 / (up + down)` | `RSI = 100·U/(U+D)`; first-order sensitivity to `U, D` each perturbed by ≤ `2δ_in` |

RSI has no global bound: when `up + down` is small (flat markets) a one-tick rounding can move
RSI by many points, so its tolerance must be computed per bar. Implement
`propagated_tolerance(indicator, δ_in, δ_out)`; test each bound by perturbing inputs by ±δ and
asserting the output change stays within it. Tradeoff: per-indicator code, but without it a
rounded-input export would either fail correct ports or force a blanket tolerance loose enough
to pass wrong ones.

**A4. Minimum detectable error (MDE).** A pass should say what it could not see. For a
recursive filter, a seed error `e₀` is detectable in aligned mode only if `e₀ > tol` at the
seed bar, so the report states "seed errors below `tol(w)` are undetectable". In convergence
mode, `e₀·(1−α)^exclude` must exceed `tol` — usually impossible by construction, which is
the §5.3 hazard stated as a number. Extend the mutation suite with a magnitude sweep (seed
error at 0.1×, 1×, 10× the tolerance) confirming the stated threshold. Tradeoff: reports get
longer; the alternative is a pass that implies more than it shows.

### B. Harness correctness

**B1. Two-bar crossover excuse.** `near_tie` is evaluated per bar, but `crossover` at bar t
depends on the comparisons at t and t−1. A legitimate flip caused by a tie at t−1 is
currently not excused. Add `crossover_excuse(a, b, tol) = near_tie_t | near_tie_{t−1}` and
use it in examples and tests.

**B2. Discrete columns and machine-readable reports.** `PortSpec` accepts numeric kinds only,
so an exported 0/1 signal column cannot be checked by `check_export`. Add `kind="discrete"`
routed to `compare_discrete`, and `ParityReport.to_dict()`/JSON so downstream CI can gate on
results without parsing Markdown. HTML stays deferred.

**B3. Composite convergence.** `convergence_bars` covers a single filter. For RSI (two RMAs of
a differenced series) and ATR (RMA of true range), the bound is the slowest internal filter's
plus the window warm-up (+1 bar for differencing). Document and test it the way the §5.3
hazard test does for EMA.

### C. Lint quality

**C1. Single-assignment tracing.** Resolve identifiers assigned once with `=` and never
reassigned with `:=`, so `src = close` followed by `request.security(..., src, lookahead_on)`
becomes `definite`. This removes the largest class of `possible` findings. Reassigned or
conditionally assigned variables stay `possible`.

**C2. Block-aware alert gating.** Use indentation to find the `if` enclosing each `alert()`
and require `barstate.isconfirmed` in that condition (or `freq_once_per_bar_close`). This
closes the documented false negative where any `isconfirmed` anywhere in the script counts.

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
counterexamples, hence the weekly random run.

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
