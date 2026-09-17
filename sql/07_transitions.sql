-- 07_transitions.sql  ·  the transition graph mined from real session history
-- See SPEC §16.5. Depends on `session_plays` (02), `track_features` and
-- `camelot_moves` (both loaded by src/enrich.py).
--
-- A crate tells you what to play. A transition graph tells you what to play
-- NEXT, and it is the one thing three years of listening history can supply
-- that a Mixed In Key export cannot: which pairs actually got played
-- back-to-back, and whether the second one survived.
--
-- The outcome variable is `to_is_skip` -- did the incoming track get skipped
-- past? That is the closest thing in streaming data to "the transition did not
-- work", and it is what the §16.5 hypothesis test is built on.

-- Every observed A -> B pair inside a session, with both sides' features.
CREATE OR REPLACE TABLE transitions AS
WITH seq AS (
    SELECT
        session_id,
        ts_utc,
        shuffle,
        track_key                                          AS to_key,
        track_name                                         AS to_name,
        artist_name                                        AS to_artist,
        is_skip                                            AS to_is_skip,
        is_play                                            AS to_is_play,
        lag(track_key)   OVER w                            AS from_key,
        lag(track_name)  OVER w                            AS from_name,
        lag(artist_name) OVER w                            AS from_artist,
        -- Position in the session: the first transition of a set behaves
        -- differently from the twentieth, and we want to be able to see that.
        row_number()     OVER w                            AS position_in_session
    FROM session_plays
    WINDOW w AS (PARTITION BY session_id ORDER BY ts_utc)
)
SELECT
    s.session_id,
    s.position_in_session,
    s.ts_utc,
    s.shuffle,
    s.from_key, s.from_name, s.from_artist,
    s.to_key,   s.to_name,   s.to_artist,
    s.to_is_skip,
    s.to_is_play,
    (s.from_artist = s.to_artist)     AS same_artist,
    ff.camelot                        AS from_camelot,
    tf.camelot                        AS to_camelot,
    ff.bpm                            AS from_bpm,
    tf.bpm                            AS to_bpm,
    ff.energy                         AS from_energy,
    tf.energy                         AS to_energy,
    cm.move,
    cm.move_score,
    cm.is_harmonic,
    -- Fractional tempo change, allowing half- and double-time so 87 -> 174
    -- reads as a match rather than a 100% jump. Mirrors harmonic.bpm_delta.
    CASE WHEN ff.bpm > 0 AND tf.bpm > 0 THEN
        least(
            abs(tf.bpm       - ff.bpm),
            abs(tf.bpm * 2.0 - ff.bpm),
            abs(tf.bpm / 2.0 - ff.bpm)
        ) / ff.bpm
    END                               AS bpm_delta_pct,
    (tf.energy - ff.energy)           AS energy_delta
FROM seq s
LEFT JOIN track_features ff ON ff.track_key = s.from_key
LEFT JOIN track_features tf ON tf.track_key = s.to_key
LEFT JOIN camelot_moves  cm ON cm.from_code = ff.camelot AND cm.to_code = tf.camelot
WHERE s.from_key IS NOT NULL
  -- Drop same-track repeats. A track following itself is a replay, not a
  -- transition, and counting it would inflate the "same key" bucket with
  -- something no DJ would call a mix. Binges are measured in 02_sessions.
  AND s.from_key <> s.to_key;

-- The weighted directed graph: one row per ordered pair, with how often it was
-- played and how often the incoming track survived.
CREATE OR REPLACE TABLE transition_edges AS
SELECT
    from_key,
    to_key,
    any_value(from_name)                      AS from_name,
    any_value(from_artist)                    AS from_artist,
    any_value(to_name)                        AS to_name,
    any_value(to_artist)                      AS to_artist,
    any_value(move)                           AS move,
    any_value(is_harmonic)                    AS is_harmonic,
    any_value(bpm_delta_pct)                  AS bpm_delta_pct,
    count(*)                                  AS n,
    count(*) FILTER (WHERE NOT to_is_skip)    AS n_held,
    wilson_low(count(*) FILTER (WHERE NOT to_is_skip), count(*), 1.96) AS hold_lcb
FROM transitions
GROUP BY from_key, to_key;

-- How each harmonic move actually performed. This is the descriptive table
-- behind the claim that harmony matters; the inferential version is below.
CREATE OR REPLACE VIEW transition_move_performance AS
SELECT
    move,
    count(*)                                            AS n,
    count(*) FILTER (WHERE to_is_skip)                  AS n_skips,
    round(avg(CAST(to_is_skip AS DOUBLE)), 4)           AS skip_rate,
    round(wilson_low(count(*) FILTER (WHERE NOT to_is_skip), count(*), 1.96), 4) AS hold_lcb,
    round(avg(bpm_delta_pct), 4)                        AS avg_bpm_delta
FROM transitions
WHERE move IS NOT NULL
GROUP BY move;

-- Inputs for the stratified hypothesis test (SPEC §16.5).
--
-- Stratifying on `shuffle` is the whole point. Shuffle raises the skip rate AND
-- produces more clashing transitions, so it is a common cause of both sides of
-- the comparison. Pooling the two would credit shuffle's skips to bad harmony
-- and overstate the effect; the Cochran-Mantel-Haenszel test in src/stats.py
-- consumes these strata and keeps them separate.
CREATE OR REPLACE VIEW hypothesis_harmonic_skip AS
SELECT
    shuffle,
    is_harmonic,
    count(*)                            AS n_trials,
    count(*) FILTER (WHERE to_is_skip)  AS n_skips,
    avg(CAST(to_is_skip AS DOUBLE))     AS skip_rate
FROM transitions
WHERE is_harmonic IS NOT NULL          -- both sides tagged, or there is no test
GROUP BY shuffle, is_harmonic;

-- The same question asked of tempo instead of key: does a big BPM jump cost
-- you the next track? Reported alongside, since the two travel together.
CREATE OR REPLACE VIEW hypothesis_tempo_skip AS
SELECT
    shuffle,
    (bpm_delta_pct > 0.06)              AS tempo_jump,   -- beyond the pitch fader
    count(*)                            AS n_trials,
    count(*) FILTER (WHERE to_is_skip)  AS n_skips,
    avg(CAST(to_is_skip AS DOUBLE))     AS skip_rate
FROM transitions
WHERE bpm_delta_pct IS NOT NULL
GROUP BY shuffle, tempo_jump;

-- Transitions I reach for most often and that reliably land: the pairs already
-- proven on my own ears, ranked by evidence rather than raw hold rate.
CREATE OR REPLACE VIEW proven_transitions AS
SELECT from_name, from_artist, to_name, to_artist, move,
       round(bpm_delta_pct, 3) AS bpm_delta_pct,
       n, n_held, round(hold_lcb, 3) AS hold_lcb
FROM transition_edges
WHERE n >= 3
ORDER BY hold_lcb DESC, n DESC;
