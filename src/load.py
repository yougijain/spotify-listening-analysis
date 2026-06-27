"""Load Spotify streaming-history JSON into a DuckDB table `raw_streams`.

This is deliberately thin (SPEC §5.1): Python's only job is to get the JSON into
DuckDB as one clean, typed table. All analysis happens in SQL downstream.

The one piece of real logic here is **schema normalization** (SPEC §4.2 note):
Spotify ships two export shapes and we standardize both onto one schema —

  * Extended streaming history  (`Streaming_History_Audio_*.json`)
      ts, ms_played, master_metadata_track_name, master_metadata_album_artist_name,
      master_metadata_album_album_name, spotify_track_uri, reason_start/end,
      shuffle, skipped, offline, platform, conn_country, episode_*
  * Legacy "account data"       (`StreamingHistory*.json`)
      endTime (LOCAL time), artistName, trackName, msPlayed

Legacy `endTime` is local wall-clock, not UTC, so we convert it to a UTC ISO
string on the way in (subtracting the home offset) so everything downstream can
treat `ts` uniformly as UTC.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List

import duckdb

# Standard target schema for raw_streams: (column, duckdb_type).
RAW_SCHEMA: List[tuple] = [
    ("ts", "VARCHAR"),                 # UTC, ISO 8601
    ("ms_played", "BIGINT"),
    ("track_name", "VARCHAR"),
    ("artist_name", "VARCHAR"),
    ("album_name", "VARCHAR"),
    ("spotify_track_uri", "VARCHAR"),
    ("reason_start", "VARCHAR"),
    ("reason_end", "VARCHAR"),
    ("shuffle", "BOOLEAN"),
    ("skipped", "BOOLEAN"),
    ("offline", "BOOLEAN"),
    ("platform", "VARCHAR"),
    ("conn_country", "VARCHAR"),
    ("episode_name", "VARCHAR"),
    ("episode_show_name", "VARCHAR"),
    ("source_file", "VARCHAR"),        # provenance, for QA/reconciliation
]

# Map each standard column to its source column in the Extended export.
_EXTENDED_MAP = {
    "ts": "ts",
    "ms_played": "ms_played",
    "track_name": "master_metadata_track_name",
    "artist_name": "master_metadata_album_artist_name",
    "album_name": "master_metadata_album_album_name",
    "spotify_track_uri": "spotify_track_uri",
    "reason_start": "reason_start",
    "reason_end": "reason_end",
    "shuffle": "shuffle",
    "skipped": "skipped",
    "offline": "offline",
    "platform": "platform",
    "conn_country": "conn_country",
    "episode_name": "episode_name",
    "episode_show_name": "episode_show_name",
}

# Legacy export only carries these four fields; everything else is unknown (NULL).
_LEGACY_MAP = {
    "ms_played": "msPlayed",
    "track_name": "trackName",
    "artist_name": "artistName",
}


def _posix(path: str) -> str:
    return Path(path).as_posix()


def _columns(con: duckdb.DuckDBPyConnection, path: str) -> set:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_json_auto('{_posix(path)}')"
    ).fetchall()
    return {r[0].lower() for r in rows}


def _is_legacy(cols: set) -> bool:
    return "endtime" in cols or ("msplayed" in cols and "ms_played" not in cols)


def _select_expr(target: str, dtype: str, source: str | None, cols: set) -> str:
    """One column expression for the normalizing SELECT, or a typed NULL if the
    source column is absent in this file."""
    if source and source.lower() in cols:
        return f'CAST("{source}" AS {dtype}) AS {target}'
    return f"CAST(NULL AS {dtype}) AS {target}"


def _normalizing_select(path: str, cols: set, home_offset_minutes: int) -> str:
    legacy = _is_legacy(cols)
    pieces = []
    for target, dtype in RAW_SCHEMA:
        if target == "source_file":
            pieces.append(f"'{os.path.basename(path)}' AS source_file")
            continue
        if legacy and target == "ts":
            # endTime is LOCAL -> convert to a UTC ISO string.
            pieces.append(
                "strftime(CAST(\"endTime\" AS TIMESTAMP) "
                f"- INTERVAL {home_offset_minutes} MINUTE, "
                "'%Y-%m-%dT%H:%M:%SZ') AS ts"
            )
            continue
        source = (_LEGACY_MAP if legacy else _EXTENDED_MAP).get(target)
        pieces.append(_select_expr(target, dtype, source, cols))
    select = ",\n       ".join(pieces)
    return f"SELECT {select}\nFROM read_json_auto('{_posix(path)}')"


def load_streams(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    home_offset_minutes: int = 330,
) -> int:
    """Load every *.json under `data_dir` into a fresh `raw_streams` table.

    Returns the number of rows loaded. `home_offset_minutes` is only used to
    convert legacy-export local timestamps to UTC (default 330 = IST).
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No .json files found in {data_dir!r}")

    cols_ddl = ",\n  ".join(f"{c} {t}" for c, t in RAW_SCHEMA)
    con.execute(f"DROP TABLE IF EXISTS raw_streams; CREATE TABLE raw_streams (\n  {cols_ddl}\n);")

    for path in files:
        cols = _columns(con, path)
        select = _normalizing_select(path, cols, home_offset_minutes)
        con.execute(f"INSERT INTO raw_streams {select}")

    return con.execute("SELECT count(*) FROM raw_streams").fetchone()[0]


def summarize(con: duckdb.DuckDBPyConnection) -> None:
    """Print a quick load summary for row reconciliation (SPEC §12)."""
    total, music, podcast, files = con.execute("""
        SELECT count(*),
               count(*) FILTER (WHERE episode_name IS NULL AND episode_show_name IS NULL),
               count(*) FILTER (WHERE episode_name IS NOT NULL OR episode_show_name IS NOT NULL),
               count(DISTINCT source_file)
        FROM raw_streams
    """).fetchone()
    lo, hi = con.execute("SELECT min(ts), max(ts) FROM raw_streams").fetchone()
    print(f"raw_streams: {total:,} rows from {files} file(s) | "
          f"{music:,} music, {podcast:,} podcast | {lo} .. {hi}")


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/sample"
    con = duckdb.connect()
    n = load_streams(con, data_dir)
    summarize(con)
    print(f"Loaded {n:,} rows.")
