"""Orchestrate the analysis: load JSON -> DuckDB, run the SQL in order, and run
the one inferential test in Python.

Thin glue only (SPEC §5.1) — every metric lives in the SQL files; this module
just sequences them and exposes small helpers the runner and notebook share.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
from scipy import stats

from . import stats as st
from .enrich import (
    CoverageReport,
    build_provider_chain,
    load_camelot_moves,
    load_track_features,
)
from .load import load_streams

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"
# The SQL runs in two blocks. Everything before enrichment works on the
# listening history alone; everything after can also see musical features, which
# have to come in through Python because the history<->library join is fuzzy.
SQL_BEFORE_ENRICH = [
    "01_clean.sql",
    "02_sessions.sql",
]
SQL_AFTER_ENRICH = [
    "03_metrics.sql",
    "04_cohorts.sql",
    "05_hypothesis.sql",
    "06_crate.sql",
    "07_transitions.sql",
]
SQL_FILES = SQL_BEFORE_ENRICH + SQL_AFTER_ENRICH


def connect(db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path) if db_path else duckdb.connect()


def df(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    """Run a query and return a DataFrame."""
    return con.execute(sql).df()


def _strip_line_comments(text: str) -> str:
    # Drop `-- ...` line comments before splitting on `;`, so a semicolon inside
    # a comment can't be mistaken for a statement boundary. None of the SQL uses
    # `--` inside a string literal, so this is safe.
    out = []
    for line in text.splitlines():
        i = line.find("--")
        out.append(line if i < 0 else line[:i])
    return "\n".join(out)


def _run_script(con: duckdb.DuckDBPyConnection, text: str) -> None:
    for stmt in _strip_line_comments(text).split(";"):
        if stmt.strip():
            con.execute(stmt)


def run_sql_file(con: duckdb.DuckDBPyConnection, name: str) -> None:
    _run_script(con, (SQL_DIR / name).read_text(encoding="utf-8"))


@dataclass
class BuildResult:
    """What a build produced, for the runner's QA section."""

    rows_loaded: int
    coverage: CoverageReport
    providers: tuple


def build(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    tz_offset_min: int = 330,
    session_gap_min: int = 30,
    burn_half_life_days: float = 60.0,
    features_file: str | None = None,
    allow_synthetic_features: bool = True,
) -> BuildResult:
    """Load data, enrich it, and materialize every model.

    The knobs (home-tz offset, session gap, burn half-life) are injected by
    overriding macros before the SQL runs; the SQL files keep IF NOT EXISTS
    defaults so they still work standalone in the DuckDB CLI.

    Enrichment happens between the two SQL blocks because the history-to-library
    join is fuzzy string matching, which belongs in Python (src/enrich.py), and
    everything from 06 onward needs its output.
    """
    n = load_streams(con, data_dir, home_offset_minutes=tz_offset_min)
    con.execute(f"CREATE OR REPLACE MACRO to_local(t) AS t + INTERVAL {int(tz_offset_min)} MINUTE")
    con.execute(f"CREATE OR REPLACE MACRO session_gap() AS INTERVAL {int(session_gap_min)} MINUTE")
    con.execute(f"CREATE OR REPLACE MACRO burn_half_life() AS {float(burn_half_life_days)}")

    for name in SQL_BEFORE_ENRICH:
        run_sql_file(con, name)

    chain = build_provider_chain(data_dir, features_file, allow_synthetic_features)
    coverage = load_track_features(con, chain)
    load_camelot_moves(con)

    for name in SQL_AFTER_ENRICH:
        run_sql_file(con, name)

    return BuildResult(rows_loaded=n, coverage=coverage,
                       providers=tuple(p.name for p in chain))


def hypothesis_test(con: duckdb.DuckDBPyConnection) -> dict:
    """Two-proportion z-test: is skip rate higher on shuffle than off? (SPEC §7.3)

    Returns the counts, both rates, the z-statistic and one-sided p-value, the
    absolute difference, and Cohen's h (effect size). Plays are autocorrelated
    within sessions, so this is descriptive evidence, not a clean experiment.
    """
    g = con.execute("""
        SELECT shuffle, n_trials, n_skips, skip_rate
        FROM hypothesis_shuffle_skip
    """).df().set_index("shuffle")

    n1, x1 = int(g.loc[True, "n_trials"]), int(g.loc[True, "n_skips"])     # shuffle
    n0, x0 = int(g.loc[False, "n_trials"]), int(g.loc[False, "n_skips"])   # intentional
    p1, p0 = x1 / n1, x0 / n0

    p_pool = (x1 + x0) / (n1 + n0)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n0))
    z = (p1 - p0) / se
    p_one_sided = float(stats.norm.sf(z))               # H1: shuffle > intentional
    cohens_h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p0))

    return {
        "shuffle_rate": p1, "shuffle_n": n1, "shuffle_skips": x1,
        "intentional_rate": p0, "intentional_n": n0, "intentional_skips": x0,
        "difference": p1 - p0,
        "z": z, "p_one_sided": p_one_sided,
        "cohens_h": cohens_h,
    }


def session_gap_sensitivity(data_dir: str, gaps=(15, 30, 45)) -> pd.DataFrame:
    """Rebuild sessions at several gap thresholds and compare (SPEC §12)."""
    rows = []
    for gap in gaps:
        con = connect()
        build(con, data_dir, session_gap_min=gap)
        n_sessions, mean_plays = con.execute(
            "SELECT count(*), avg(n_plays) FROM sessions"
        ).fetchone()
        rows.append({"gap_min": gap, "n_sessions": n_sessions,
                     "mean_plays_per_session": mean_plays})
        con.close()
    return pd.DataFrame(rows)


def harmonic_hypothesis_test(con: duckdb.DuckDBPyConnection) -> st.CMHResult:
    """H2: a clashing transition loses the incoming track more often (SPEC §8.5).

    Stratified on shuffle rather than pooled. Shuffle is a common cause of both
    sides of this comparison — it raises the skip rate and it produces more
    clashes — so pooling would hand shuffle's skips to bad harmony and overstate
    the effect. The returned object carries the pooled comparison too, so the
    report can show how much the confound was actually worth.
    """
    rows = con.execute("""
        SELECT shuffle, is_harmonic, n_trials, n_skips
        FROM hypothesis_harmonic_skip
    """).fetchall()

    strata = []
    for shuffle in (False, True):
        arm = {bool(h): (int(n), int(x)) for s, h, n, x in rows if bool(s) is shuffle}
        if True not in arm or False not in arm:
            continue
        n_clash, x_clash = arm[False]      # exposed = harmonically incompatible
        n_clean, x_clean = arm[True]       # control = compatible
        strata.append(st.Stratum(
            label="shuffle" if shuffle else "intentional",
            x_exposed=x_clash, n_exposed=n_clash,
            x_control=x_clean, n_control=n_clean,
        ))
    return st.cochran_mantel_haenszel(strata)


def tempo_hypothesis_test(con: duckdb.DuckDBPyConnection) -> st.CMHResult:
    """The same question asked of tempo: does a jump past the fader cost you?"""
    rows = con.execute("""
        SELECT shuffle, tempo_jump, n_trials, n_skips FROM hypothesis_tempo_skip
    """).fetchall()

    strata = []
    for shuffle in (False, True):
        arm = {bool(j): (int(n), int(x)) for s, j, n, x in rows if bool(s) is shuffle}
        if True not in arm or False not in arm:
            continue
        n_jump, x_jump = arm[True]
        n_close, x_close = arm[False]
        strata.append(st.Stratum(
            label="shuffle" if shuffle else "intentional",
            x_exposed=x_jump, n_exposed=n_jump,
            x_control=x_close, n_control=n_close,
        ))
    return st.cochran_mantel_haenszel(strata)
