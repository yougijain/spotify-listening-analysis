"""Orchestrate the analysis: load JSON -> DuckDB, run the SQL in order, and run
the one inferential test in Python.

Thin glue only (SPEC §5.1) — every metric lives in the SQL files; this module
just sequences them and exposes small helpers the runner and notebook share.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from scipy import stats

from .load import load_streams

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"
SQL_FILES = [
    "01_clean.sql",
    "02_sessions.sql",
    "03_metrics.sql",
    "04_cohorts.sql",
    "05_hypothesis.sql",
]


def connect(db_path: Optional[str] = None) -> duckdb.DuckDBPyConnection:
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


def build(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    tz_offset_min: int = 330,
    session_gap_min: int = 30,
) -> int:
    """Load data and materialize every model. Returns rows loaded.

    The two knobs (home-tz offset, session gap) are injected by overriding the
    macros before the SQL runs; the SQL files keep IF NOT EXISTS defaults so they
    still work standalone in the DuckDB CLI.
    """
    n = load_streams(con, data_dir, home_offset_minutes=tz_offset_min)
    con.execute(f"CREATE OR REPLACE MACRO to_local(t) AS t + INTERVAL {int(tz_offset_min)} MINUTE")
    con.execute(f"CREATE OR REPLACE MACRO session_gap() AS INTERVAL {int(session_gap_min)} MINUTE")
    for name in SQL_FILES:
        run_sql_file(con, name)
    return n


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
