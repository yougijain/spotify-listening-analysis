-- 02_sessions.sql  ·  sessionize plays and summarize each session
-- See SPEC §6.2 (session definition), §6.3 (window-function sketch), §7.

-- Inactivity gap that ends a session. 30 min = the conventional web-analytics
-- timeout. Documented as a tunable knob; the pipeline overrides it for the
-- 15/45-min sensitivity check (SPEC §12). IF NOT EXISTS keeps this standalone.
CREATE MACRO IF NOT EXISTS session_gap() AS INTERVAL 30 MINUTE;

-- Assign a session_id to every play. A new session starts when the gap from the
-- previous play's END (ts_utc) to this play's START (ts_utc - ms_played) is at
-- least `session_gap()`. (Recall ts is the moment the stream ENDED.)
CREATE OR REPLACE TABLE session_plays AS
WITH ordered AS (
    SELECT *,
           lag(ts_utc) OVER (ORDER BY ts_utc) AS prev_end
    FROM plays
),
flagged AS (
    SELECT *,
           CASE
               WHEN prev_end IS NULL
                 OR (ts_utc - to_milliseconds(ms_played)) - prev_end >= session_gap()
               THEN 1 ELSE 0
           END AS is_new_session
    FROM ordered
)
SELECT
    * EXCLUDE (prev_end, is_new_session),
    sum(is_new_session) OVER (ORDER BY ts_utc ROWS UNBOUNDED PRECEDING) AS session_id
FROM flagged;

-- One row per session.
CREATE OR REPLACE TABLE sessions AS
SELECT
    session_id,
    min(ts_utc)                      AS started_utc,
    max(ts_utc)                      AS ended_utc,
    CAST(min(ts_local) AS DATE)      AS date_local,
    EXTRACT(hour FROM min(ts_local)) AS start_hour_local,
    EXTRACT(dow  FROM min(ts_local)) AS start_dow_local,
    count(*)                         AS n_plays,
    count(*) FILTER (WHERE is_play)  AS n_counted,
    count(*) FILTER (WHERE is_skip)  AS n_skips,
    sum(ms_played) / 60000.0         AS minutes,
    count(DISTINCT artist_name)      AS distinct_artists,
    count(DISTINCT track_key)        AS distinct_tracks,
    bool_or(shuffle)                 AS any_shuffle
FROM session_plays
GROUP BY session_id;

-- Binge runs: maximal stretches of the SAME track_key back-to-back within one
-- session (gaps-and-islands). The longest such run is the "most-binged" track.
CREATE OR REPLACE TABLE binges AS
WITH seq AS (
    SELECT session_id, track_key, track_name, artist_name, ts_utc,
           row_number() OVER (PARTITION BY session_id ORDER BY ts_utc)            AS rn,
           row_number() OVER (PARTITION BY session_id, track_key ORDER BY ts_utc) AS rn_track
    FROM session_plays
)
SELECT
    session_id,
    track_key,
    any_value(track_name)  AS track_name,
    any_value(artist_name) AS artist_name,
    count(*)               AS run_len
FROM seq
GROUP BY session_id, track_key, (rn - rn_track);
