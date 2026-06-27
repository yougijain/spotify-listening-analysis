-- 01_clean.sql  ·  raw_streams -> plays  (the cleaned base view)
-- Music only, typed, local time derived, with is_play / is_skip flags.
-- See SPEC §6.1 (base view) and §6.2 (core definitions).

-- Home-timezone knob. `ts` is UTC; convert to local for hour/day-of-week cuts.
-- Default 330 min = IST (Asia/Kolkata, UTC+5:30). IST has no DST, so a fixed
-- offset is exact here; for a DST timezone you'd swap in the ICU extension.
-- The pipeline can override this macro (CREATE OR REPLACE) before querying;
-- IF NOT EXISTS keeps this file runnable standalone in the DuckDB CLI.
CREATE MACRO IF NOT EXISTS to_local(t) AS t + INTERVAL 330 MINUTE;

CREATE OR REPLACE VIEW plays AS
WITH music AS (
    SELECT
        CAST(ts AS TIMESTAMP) AS ts_utc,
        ms_played,
        track_name,
        artist_name,
        album_name,
        spotify_track_uri,
        reason_start,
        reason_end,
        shuffle,
        skipped,
        offline,
        platform,
        conn_country
    FROM raw_streams
    WHERE episode_name IS NULL            -- drop podcasts (SPEC §4.3)
      AND episode_show_name IS NULL
      AND track_name IS NOT NULL
      AND artist_name IS NOT NULL
)
SELECT
    ts_utc,
    to_local(ts_utc)                    AS ts_local,
    CAST(to_local(ts_utc) AS DATE)      AS date_local,
    EXTRACT(hour FROM to_local(ts_utc)) AS hour_local,
    EXTRACT(dow  FROM to_local(ts_utc)) AS dow_local,   -- 0=Sun .. 6=Sat
    dayname(to_local(ts_utc))           AS day_name,
    ms_played,
    track_name,
    artist_name,
    album_name,
    -- Stable track key: prefer the URI, fall back to track+artist (SPEC §6.1).
    coalesce(spotify_track_uri, track_name || '␟' || artist_name) AS track_key,
    shuffle,
    reason_start,
    reason_end,
    platform,
    conn_country,
    skipped,
    offline,
    -- A "counted play" = >= 30s (Spotify's royalty threshold, SPEC §6.2).
    (ms_played >= 30000)                          AS is_play,
    -- A "skip" = a short play the user actively skipped past. Uses intent
    -- (fwdbtn) + duration, NOT the inconsistently-populated `skipped` field.
    (ms_played < 30000 AND reason_end = 'fwdbtn') AS is_skip
FROM music;
