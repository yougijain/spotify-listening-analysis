# Spotify Listening Analysis — Build Spec

**Type:** Data-analyst portfolio project
**Author:** Yougi
**Date:** 2026-06-26
**Status:** Spec (not yet built)

---

## 1. One-liner

A rigorous analysis of my own Spotify listening behavior, built entirely on SQL, that goes past "Spotify Wrapped" to answer real behavioral questions: how my taste concentrates, how I discover and abandon artists, and how my listening changes by time and context.

The hook is **"Wrapped, but honest."** The substance is **product-analyst technique** — sessionization, cohort retention, and a hypothesis test — applied to a personal dataset.

---

## 2. Why this is a data-analyst project (positioning)

This project is deliberately scoped to send a **data-analyst** signal, not a data-scientist or ML-engineer one. The defensible core is the **SQL and the metric definitions**, not a model.

What an interviewer should take away:

- Can pull and clean a messy real-world dataset.
- Can define metrics precisely and defend every choice (what counts as a "play," a "skip," a "session").
- Knows cohort/retention analysis — the bread-and-butter of product analytics — and can apply it.
- Can run and correctly interpret an inferential test.
- Can turn analysis into a clear, honest insight readout for a stakeholder.

Every line is mine to defend. See §12 for the interview-prep angle.

---

## 3. Goals and non-goals

### Goals

- Produce a documented analysis with 4–6 defensible findings and supporting visuals.
- Do all analysis in **SQL (DuckDB)**; keep Python to data loading and charting only.
- Ship a clean GitHub repo with a README that reads like a real case study.
- Define every metric explicitly, with the rationale and the tunable knobs written down.

### Non-goals (what keeps this *strictly* analyst)

- **No machine learning** — no clustering, no predictive models, no recommender.
- **No audio-features analysis** — the Spotify audio-features API was deprecated in Nov 2024 and is gone; we do not use valence/energy/tempo. The project runs on behavioral data instead, which is richer for this purpose.
- **No real-time / connector streaming** — the live Spotify connector only exposes search, currently-playing, and playlist creation; it cannot supply history and is not used.
- **No app/dashboard framework sprawl** — an optional static dashboard is a stretch goal, not the project.

---

## 4. Data

### 4.1 Source — the Extended Streaming History export

The Spotify Web API does **not** provide full listening history (the "recently played" endpoint caps at the last 50 tracks). The real source is the **Extended Streaming History** export requested from the Spotify account privacy page.

How to request it:

1. Spotify → Account → Privacy Settings → "Download your data."
2. **Tick only "Extended streaming history." Untick "Account data."** (Account data gives only ~1 year and coarser fields; Extended is lifetime and granular.)
3. Submit. Spotify says up to 30 days; in practice the email usually arrives within 1–3 days.
4. The download is a `.zip` of JSON files named like `Streaming_History_Audio_2018-2020_0.json`.

**Action required now:** request this immediately so the data is waiting. Development proceeds against a public sample in the meantime (§4.4).

### 4.2 Schema — key fields per stream

Each row in `Streaming_History_Audio_*.json` is one stream (one play event), keyed on when it ended.

| Field | Meaning | Use |
|---|---|---|
| `ts` | Timestamp the stream ended (UTC, ISO 8601) | All time-based analysis; convert to local time |
| `ms_played` | Milliseconds played | Play/skip thresholds, listening volume |
| `master_metadata_track_name` | Track name | Track-level grouping |
| `master_metadata_album_artist_name` | Artist name | Artist-level grouping, cohorts |
| `master_metadata_album_album_name` | Album name | Album-level cuts |
| `spotify_track_uri` | Track URI | Stable track key (better than name) |
| `reason_start` | Why playback started (`trackdone`, `clickrow`, `fwdbtn`, `playbtn`, `appload`, …) | Behavioral context |
| `reason_end` | Why playback ended (`trackdone`, `fwdbtn`, `backbtn`, `endplay`, `logout`, …) | **Skip detection** |
| `shuffle` | Shuffle on/off | Behavioral segmentation |
| `skipped` | Skip flag (may be null) | Skip detection (cross-check with `reason_end`) |
| `offline` | Played offline | Context |
| `platform` | Device/OS | Device segmentation |
| `conn_country` | Country code | Travel/context; usually filtered to home country |
| `episode_name`, `episode_show_name` | Podcast fields (null for music) | **Filter these out** — music only |

> **Format note:** older "Account data" exports use different field names (`endTime`, `artistName`, `trackName`, `msPlayed`) in `StreamingHistory*.json`. The loader should detect and normalize both, but we standardize on the Extended schema above.

### 4.3 Known data gotchas (must handle)

- **UTC timestamps.** `ts` is UTC. Hour-of-day and day-of-week analysis must convert to a fixed local timezone; document the assumed tz.
- **Podcasts mixed in.** Rows with non-null episode fields are podcasts — exclude from music analysis.
- **No track IDs in some fields / name collisions.** Prefer `spotify_track_uri` as the track key; fall back to `(track_name, artist_name)`.
- **Partial edge months.** The first and last months of history are partial — flag and exclude them from any per-month trend that assumes full months.
- **Very short plays / duplicates.** Sub-second plays and rapid repeats need a defined rule (see §6).
- **Null `skipped`.** The `skipped` field is inconsistently populated across the export's date range; the skip definition must not depend on it alone.

### 4.4 Development dataset

Build the whole pipeline against the [Maven Analytics Spotify streaming history sample](https://mavenanalytics.io/data-playground/spotify-streaming-history) (or a synthetic generator if field coverage differs), then swap in the real export in Phase 7. Code must not hardcode anything personal.

---

## 5. Architecture and stack

### 5.1 Stack

- **DuckDB** — file-based SQL engine; reads JSON directly via `read_json_auto`, no server. This is the analytical core and where all metric logic lives.
- **Python (thin)** — only to orchestrate the load (JSON → DuckDB) and render charts. No analysis logic in Python.
- **Plotly or matplotlib** — static figures for the report.
- **Jupyter notebook or a markdown report** — the narrative analysis layer.
- **Git/GitHub** — the repo *is* the portfolio artifact.
- *(Stretch)* a static HTML dashboard for interactivity.

Rationale for DuckDB over Postgres/Supabase: zero setup, runs on the raw files, and "I did the analysis in DuckDB/SQL" is a clean, modern analyst story. Postgres is the fallback if a hosted DB is wanted later.

### 5.2 Data flow

```
Spotify export (.zip of JSON)
        │  src/load.py (read_json_auto)
        ▼
  raw_streams          ← one row per raw stream event
        │  sql/01_clean.sql
        ▼
  plays (view)         ← music only, typed, local time, derived flags
        │
        ├── sql/02_sessions.sql   → sessions
        ├── sql/03_metrics.sql    → daily/artist/track aggregates
        ├── sql/04_cohorts.sql    → artist_first_listen, retention
        └── sql/05_hypothesis.sql → test inputs
        │
        ▼
  figures/ + analysis report (the deliverable)
```

### 5.3 Repo structure

```
spotify-listening-analysis/
├── README.md                 # case-study writeup (the portfolio piece)
├── SPEC.md                   # this file
├── requirements.txt
├── .gitignore                # ignores data/raw/ (personal data)
├── data/
│   ├── raw/                  # personal export (gitignored)
│   └── sample/               # dev data, committed
├── src/
│   ├── load.py               # JSON → DuckDB, schema normalization
│   └── charts.py             # figure rendering
├── sql/
│   ├── 01_clean.sql          # raw_streams → plays
│   ├── 02_sessions.sql       # sessionization
│   ├── 03_metrics.sql        # volume, skip rate, concentration, discovery
│   ├── 04_cohorts.sql        # artist cohort retention
│   └── 05_hypothesis.sql     # hypothesis-test inputs
├── notebooks/
│   └── analysis.ipynb        # narrative + charts
└── figures/                  # exported PNGs
```

---

## 6. Data model and metric definitions

This section is the analytical heart — the decisions to defend. Every threshold below is a knob with a stated default and rationale.

### 6.1 Cleaned base view: `plays`

From `raw_streams`, produce `plays` by:

- Excluding podcast rows (non-null episode fields).
- Casting `ts` to timestamp; deriving `ts_local`, `date_local`, `hour_local`, `dow_local` in the assumed home timezone.
- Keeping `ms_played`, `shuffle`, `reason_start`, `reason_end`, `platform`, `conn_country`.
- Deriving `track_key = coalesce(spotify_track_uri, track_name || '␟' || artist_name)`.
- Deriving the `is_play` and `is_skip` flags below.

### 6.2 Core definitions

- **Counted play (`is_play`):** `ms_played >= 30000` (30 seconds). Rationale: 30s is Spotify's own royalty-counting threshold, so it's a defensible, externally-grounded line between "listened" and "sampled."
- **Skip (`is_skip`):** `ms_played < 30000` **and** `reason_end = 'fwdbtn'` (user pressed next early). Rationale: combines a duration signal with intent; does not rely on the inconsistently-populated `skipped` field, though we cross-check against it.
- **Session:** a maximal run of consecutive plays (ordered by `ts`) where the gap from one play's end to the next play's start is **< 30 minutes**. Rationale: 30 min is the conventional web-analytics inactivity timeout; documented as a tunable knob with a sensitivity check at 15/45 min.
- **Discovery event:** the first appearance of an artist in the full history. **Discovery rate** = distinct first-seen artists per calendar month.
- **Concentration:** share of total `ms_played` attributable to the top 1% and top 10% of artists. Optionally a Herfindahl index (HHI) for a single-number summary. Rationale: measures how "monogamous" the taste is.
- **Retention cohort:** an artist's cohort = the calendar month of its discovery event. **Retention at month k** = fraction of a cohort's artists played at least once (≥1 counted play) in the k-th month after discovery. Denominator = cohort size; only cohorts with a full k-month observation window are included.
- **Binge:** the maximum count of consecutive plays of the same `track_key` within a single session.

### 6.3 Approach sketches (illustrative, not final)

Sessionization via window functions:

```sql
-- sketch only
with ordered as (
  select *,
         lag(ts) over (order by ts)                         as prev_ts,
         lag(ms_played) over (order by ts)                  as prev_ms
  from plays
),
flagged as (
  select *,
         case when prev_ts is null
                or ts > prev_ts + interval (prev_ms/1000) second
                       + interval 30 minute
              then 1 else 0 end as is_new_session
  from ordered
)
select *, sum(is_new_session) over (order by ts) as session_id
from flagged;
```

Cohort retention:

```sql
-- sketch only
with first_listen as (
  select artist_name, date_trunc('month', min(ts_local)) as cohort_month
  from plays where is_play group by 1
),
activity as (
  select artist_name, date_trunc('month', ts_local) as active_month
  from plays where is_play group by 1, 2
)
select cohort_month,
       date_diff('month', cohort_month, active_month) as k,
       count(distinct a.artist_name)::float
         / count(distinct f.artist_name) over (partition by cohort_month) as retention
from first_listen f join activity a using (artist_name)
group by 1, 2;
```

---

## 7. The analyses (deliverables)

### 7.1 Spine — listening behavior and retention (primary)

1. **Listening volume over time** — hours/day by year, month, weekday, and hour-of-day heatmap. Establishes the baseline narrative.
2. **Skip-rate analysis** — overall skip rate, plus skip rate by shuffle, by hour, by discovery-vs-familiar. The behavioral centerpiece.
3. **Artist cohort retention** — the retention curve described in §6.2. The "this person knows product analytics" exhibit.

### 7.2 Hook — "Wrapped, but honest"

The metrics Spotify Wrapped omits, framed as an honest counter-readout:

- Real skip rate (Wrapped never shows this).
- Taste concentration (top-1%/10% share, HHI).
- Discovery-rate trend across years — is taste still expanding or calcifying?
- Most-binged track (consecutive plays).
- Listening by hour/day — when am I actually a listener?

### 7.3 Rigor — one hypothesis test

State a real hypothesis and test it properly. Default: **"Skip rate is higher on shuffle than on intentional plays."**

- Two-proportion z-test (or chi-square) on skip rate, shuffle vs. non-shuffle.
- Report effect size, not just significance.
- Explicit caveat: plays are not independent (autocorrelated within sessions), so treat the test as descriptive evidence, not a clean experiment. Naming this limitation is itself a strong analyst signal.

---

## 8. Visualizations

| Figure | Type | Shows |
|---|---|---|
| Volume trend | Line | Listening hours by month/year |
| Hour×weekday heatmap | Heatmap | When listening happens |
| Skip-rate breakdown | Bar | Skip rate by shuffle / context |
| Cohort retention | Line (curves) or triangle heatmap | Artist retention by months-since-discovery |
| Discovery trend | Line | New artists per month over time |
| Concentration | Lorenz curve or top-N bar | How taste concentrates |

Keep figures clean and labeled; each must support a specific stated finding.

---

## 9. Deliverables

1. **GitHub repo** — structured per §5.3, runnable end-to-end on the sample.
2. **README case study** — question → data → method → findings → caveats, with embedded figures. This is what a recruiter reads.
3. **Analysis notebook/report** — the narrative with charts.
4. **The SQL** — clean, commented, the defensible spine.
5. *(Stretch)* a deployed static dashboard.

---

## 10. Work split

The division that keeps the spine yours to defend:

**You write:**

- All metric definitions and the rationale behind each threshold.
- The cleaning SQL (`01_clean.sql`).
- The metrics, sessions, cohort, and hypothesis SQL (`02`–`05`).
- The insight readout and README narrative.

**I scaffold / coach:**

- Repo skeleton, `.gitignore`, `requirements.txt`.
- `load.py` (JSON → DuckDB, schema normalization across export formats).
- `charts.py` plumbing and figure styling.
- Explaining each SQL pattern (window functions, cohort logic) before you write it, then reviewing and debugging with you.

---

## 11. Milestones

| Phase | Outcome | Owner |
|---|---|---|
| 0 — Setup | Repo, DuckDB, sample data loads | Me |
| 1 — Clean/model | `plays` view; podcasts filtered; local time derived | You (coached) |
| 2 — Core metrics | Volume trends, skip rate, concentration, discovery | You |
| 3 — Cohort retention | Retention curve built and validated | You (coached on window logic) |
| 4 — Hypothesis test | Test run, effect size, caveats written | You |
| 5 — Viz + writeup | Figures + "honest Wrapped" narrative | You + me |
| 6 — README + polish | Case-study README, repo cleanup, optional deploy | You + me |
| 7 — Swap in real data | Re-run pipeline on actual export, refresh findings | You |

---

## 12. Validation and QA

Analyst credibility lives here — bake these checks in:

- **Row reconciliation:** total streams in vs. rows modeled out; account for every dropped row (podcasts, nulls).
- **Spot checks:** pick a day you remember and confirm the data matches reality.
- **Session sanity:** eyeball a few sessions; run the 15/45-min sensitivity check.
- **Cohort denominators:** confirm each retention point divides by the right cohort size and only uses fully-observed windows.
- **Timezone check:** confirm hour-of-day peaks make sense for your actual habits (catches a UTC/local bug).
- **Edge months:** confirm partial first/last months are excluded from trend lines.

---

## 13. Interview-prep angle (defend every line)

Be ready to answer, cold:

- Why 30 seconds for a "play"? (Spotify royalty threshold.)
- How do you define a skip, and why not just trust the `skipped` field?
- Why a 30-minute session gap? How sensitive are results to that?
- What's the cohort denominator, and how do you avoid survivorship bias in the retention curve?
- How did you keep podcasts out, and how big was that slice?
- How did you handle UTC vs. local time?
- Your hypothesis test ignores autocorrelation — what would you do differently with more time?
- What would change if this were 10 billion rows instead of yours? (Partitioning, pushing to a warehouse, incremental loads.)

---

## 14. Risks

- **Export delay** — mitigated by building on the sample first.
- **Schema drift between export formats** — loader normalizes both; validate field coverage on arrival.
- **Personal-data leakage** — `data/raw/` is gitignored; never commit personal history.
- **Scope creep toward data science** — any urge to cluster or model is out of scope by design (§3).

---

## 15. Stretch goals (only after the core ships)

- Static HTML dashboard (filters by year/artist; the heatmap and trends).
- Compare two eras of your listening (e.g., pre/post a life change).
- Per-platform behavioral differences (mobile vs. desktop skip rates).

---

# Part II — Crate intelligence

*Added after the listening-analysis core shipped. §1–15 describe a personal
analytics project; this half describes what makes it a DJ tool, and it is where
the design decisions that actually needed arguing live.*

## 16. From listening metrics to booth decisions

### 16.1 Positioning

The spine in §7 answers "how do I listen". A DJ needs three narrower answers,
and each one reuses a spine metric with a changed definition:

| Spine metric (§7) | Booth question | What has to change |
|---|---|---|
| Skip rate | Will this clear the floor? | Raw rates are unusable for *ranking* at small n — every rate becomes a Wilson lower bound |
| Cohort retention | Does a new artist survive rotation? | Unchanged; it was already the right shape |
| Play recency | Have I over-rotated this? | New: exponentially-weighted play load, percent-ranked |
| Sessionization | Which mixes have I played? | New: a directed transition graph with a survival outcome |

The harmonic domain model (Camelot wheel, tempo reachability, energy arcs) is
**pure and separately testable** (`src/harmonic.py`), with no DuckDB or I/O, so
every mixing rule can be argued with in isolation from the pipeline.

### 16.2 Non-goals (unchanged in spirit from §3)

- No audio DSP. No beatgrid detection, no key detection from waveforms. The
  features come from software that already does this properly.
- No recommender. Nothing here predicts what you *will* like; it sequences what
  you already own.
- No learned model. Objective weights are stated judgement calls, in one place,
  not fitted parameters dressed up as findings.

### 16.3 Musical features: the provider chain

**The constraint that shaped this section.** Spotify restricted
`/v1/audio-features` and `/v1/audio-analysis` to existing applications in
November 2024. A project started after that cannot obtain tempo or key from the
API. Any design that assumes otherwise is describing a system that cannot be
built today.

Features therefore come from a chain of providers, first hit wins
(`src/enrich.py`):

1. `CsvFeatureProvider` — Mixed In Key / Rekordbox / Traktor export. The real
   path: a working DJ's library is already analysed, on their own disk. Column
   names are resolved against per-tool aliases rather than a fixed schema.
2. `SyntheticFeatureProvider` — deterministic stand-ins (BLAKE2b over the match
   key, so identical across runs, machines and Python versions, unlike `hash()`
   which is per-process salted). Tagged as its own source and counted separately
   in the coverage report so it can never be read as a measurement.

A third provider is one method. Nothing downstream changes.

**The join.** A feature export carries `Artist` and `Title` strings; the history
carries those plus a URI. There is no shared identifier, so matching is fuzzy by
necessity. `match_key()` folds accents, punctuation, and *version* suffixes
(`Remastered 2011`, `Radio Edit`, `Original Mix`) but deliberately **not**
`Live` or `Acoustic` — those are different recordings with different tempos, and
collapsing them attaches the wrong BPM to real plays. Apostrophes are deleted
rather than spaced so `Don't` matches `Dont`.

Coverage is reported on every run. An unmatched track is a visible number, not a
silent NULL.

### 16.4 The crate model (`sql/06_crate.sql`)

Three questions per track.

**Does it hold?** `hold_lcb` = Wilson lower bound on the non-skip rate. The
naive rate is actively misleading for ranking: a track played twice and never
skipped scores 1.0 and tops the crate on no evidence. The bound shrinks thin
samples, so "probably good, watched 200 times" beats "flawless across two".

**Is it burned?** `rotation_burn` = exponentially-weighted recent play count
(60-day half-life — roughly a residency's memory), then **percent-ranked across
the crate**, because staleness only means anything relative to everything else
you own.

**Do I know?** `n_plays`, surfaced as an `unproven` status below a threshold
rather than hidden inside a score.

`set_readiness` = `0.60 · hold_lcb + 0.40 · (1 − rotation_burn)`. Weighted toward
holding, because a stale banger still works and a floor-killer never does.

> **A finding worth stating, since the chart contradicts the intuition:** the
> burned tracks have *high* hold rates. Reliability is what earns a track its
> overplay. "Rest these" is a freshness call, not a quality judgement, and any
> UI that implies otherwise is lying about its own data.

### 16.5 Does harmony hold the listener? (`sql/07_transitions.sql`)

**H2: a harmonically incompatible transition loses the incoming track more
often than a compatible one.**

- **Unit:** an ordered in-session pair A→B. Same-track repeats are excluded — a
  track following itself is a replay, not a mix, and counting it would stuff the
  "same key" bucket with something no DJ would call a transition.
- **Outcome:** `to_is_skip`. The closest thing streaming data has to *the
  transition did not work*.
- **Exposure:** the Camelot move, via a 24×24 lookup **generated from
  `src/harmonic.py`** rather than reimplemented in SQL. 576 rows is nothing, and
  it makes drift between the two implementations structurally impossible instead
  of something a parity test catches after the fact.

**Why this is not a two-proportion z-test.** Shuffle raises the skip rate *and*
produces more clashing transitions — it is a common cause of both variables.
Pooling hands shuffle's skips to bad harmony. On the sample this is not a
hypothetical: the crude odds ratio is 2.35 against an adjusted 1.70, so pooling
**overstates the effect by 38%**.

The test is Cochran–Mantel–Haenszel stratified on shuffle, with a
Mantel–Haenszel common odds ratio and a Robins–Breslow–Greenland interval
(`src/stats.py`), verified against `statsmodels.StratifiedTable` to six decimal
places and pinned in the tests. A Simpson's-paradox case is a regression test,
because that is the exact failure mode being defended against.

**Honest limitation, stated up front:** the committed sample's generator queues
harmonically-adjacent tracks on purpose and skips more on a clash. The test is
therefore *guaranteed* to find an effect on sample data. What it demonstrates is
the measurement machinery. The experiment is running it against a real export.

---

## 17. Set construction

### 17.1 The problem

Given a crate, a target duration and an energy shape, produce a playable
ordering. This is constrained sequencing, not a sort.

### 17.2 Objective and constraints (`src/setbuilder.py`)

Each A→B step scores on five weighted components summing to 1:

| component | weight | what it asks |
|---|---:|---|
| harmonic | 0.30 | does the Camelot move work |
| tempo | 0.20 | reachable through a ±6% fader (half/double-time allowed) |
| energy | 0.20 | does B sit where the arc wants this slot |
| continuity | 0.12 | is the step from A survivable |
| quality | 0.18 | is B worth playing (`set_readiness`) |

plus `0.10 ×` an observed-hold bonus for pairs already played back-to-back.
Small on purpose: it breaks ties, it does not override musical scoring, because
listening history reflects habits rather than dancefloor results.

**Continuity earns its place.** An objective scoring only distance-from-target
will happily drop 0.9 → 0.2 → 0.9 if each track individually sits near the arc.
On a floor that reads as the set falling over. Arc fit is a *shape* constraint;
continuity is a *local* one, and both are needed.

**Hard constraints reject rather than penalise:** no repeats, no artist inside a
3-track gap, no tempo jump past the fader. Folding these into the score would
let a high enough harmonic score buy its way past a rule.

**Beam search, not greedy,** because greedy fails predictably — it takes the best
transition available now and strands itself in a corner of the wheel with
nothing that mixes out. Beams are deduplicated by track sequence; without that
the width collapses and it quietly degrades back into greedy. The width-1 chain
is computed alongside and the better of the two returned, so "the smarter search
is at least as good" is a guarantee rather than usually true.

### 17.3 Validating the optimiser

"It works" needs a number. `evaluate()` benchmarks every arc against greedy and
200 constraint-respecting random orderings (not a strawman — the random baseline
honours the same hard constraints, so it is "what you get from shuffling a crate
you already curated").

Result on the sample: **100% harmonic transitions against 17% for random**, but
the lift over greedy is ~0.02. **Most of the gain is the objective and the hard
constraints, not the lookahead.** That is worth reporting plainly.

### 17.4 Arc feasibility — can the crate do this at all?

Before blaming the sequencer for missing an arc, ask whether the material
exists. `arc_feasibility()` measures, slot by slot, how many candidates sit
within tolerance of that slot's energy target.

**The version that only counted energy was wrong**, and usefully so. It reported
the crate as well-supplied for a warmup while the built set missed its arc by
0.30, because the quiet tracks *do* exist — they are downtempo, and you cannot
reach 92 BPM from a 137 BPM floor through a ±8% fader. **Energy and tempo are
correlated in any real crate**, so a slot is only supplied if its candidates are
also tempo-reachable.

With reachability included the diagnostic discriminates (warmup 70% vs peak
100%) and predicts the built set's arc miss (0.30 vs 0.07); there is a test
asserting that relationship. "This crate cannot open a night" is therefore a
finding the tool reports, not a defect it hides.

---

## 18. Hosting and delivery

**Decision: a static site, no backend.** The analysis is a batch job over a
fixed export, so its output is computed once at build time and shipped as files.
That buys a free host, no cold starts, no secrets in an environment, and nothing
that can fall over unattended. Vercel is the primary host (`vercel.json`); a
GitHub Pages workflow sits alongside so the site is not tied to one vendor.

**The one thing that must be interactive is the set builder** — a page of
pre-rendered setlists is a screenshot, not a tool. So the browser runs the beam
search itself.

**Keeping the domain model single-sourced.** Every component of a transition's
score except the energy-arc term is position-independent, so it is precomputed
per edge (`base_transition_score`) and shipped as a weighted graph capped at 24
candidates per node. The client adds one term and walks the graph. Arc curves
ship as 101 sampled points rather than formulas transcribed into JS, because a
sample cannot drift from its source and a transcription can.

`tests/test_web_export.py` holds that contract: base plus arc term must
reconstruct the full score, interpolated samples must land within 1e-3 of
`arc_target`, and every exported edge weight must equal what Python computes.

**Reproducibility is a CI gate.** `data/sample` and `web/data` are both
committed, and CI diffs each against what its generator produces. Enforcing that
surfaced a real defect: unordered `GROUP BY` and a missing `ORDER BY` in
`load_crate` made the export non-reproducible across builds — and since row
order in `crate.json` *is* the node numbering of the transition graph, a rebuild
could renumber every node.

Payload: 132 KB for ~7,700 plays.
