-- 06_crate.sql  ·  the crate: every track scored for whether it earns a slot
-- See SPEC §8.4. Depends on `session_plays` (02) and `track_features` (enrich).
--
-- This is the file that turns listening analytics into DJ decisions. The three
-- questions a crate has to answer before a gig are:
--
--   1. Does it hold?   -- do people (or I) stay with it, or reach for the next?
--   2. Is it burned?   -- have I hammered it so recently that it is stale?
--   3. Do I know?      -- is there enough evidence to have an opinion at all?

-- Half-life in days for the recency weighting behind rotation burn. 60 days is
-- roughly a residency's memory: something played hard two months ago has half
-- the staleness of something played hard last week. Overridable by the pipeline.
CREATE MACRO IF NOT EXISTS burn_half_life() AS 60.0;

-- Wilson score lower bound for a proportion. The naive rate is unusable for
-- ranking here: a track played twice and never skipped scores a perfect 1.0 and
-- would top the crate on no evidence. The lower bound penalises thin samples,
-- so "probably good, and I have watched it happen 200 times" beats "flawless
-- across two plays". Mirrors src/stats.py:wilson_interval, and tests/test_sql_models.py
-- asserts the two agree across a grid of (successes, trials).
CREATE OR REPLACE MACRO wilson_low(x, n, z) AS
    CASE WHEN n IS NULL OR n <= 0 THEN 0.0
    ELSE greatest(0.0,
        (
            (CAST(x AS DOUBLE) / n + (z * z) / (2.0 * n))
            - z * sqrt(
                (CAST(x AS DOUBLE) / n) * (1.0 - CAST(x AS DOUBLE) / n) / n
                + (z * z) / (4.0 * n * n)
            )
        ) / (1.0 + (z * z) / n)
    ) END;

-- Per-track listening behaviour, with a recency-weighted play load.
CREATE OR REPLACE TABLE track_plays AS
WITH bounds AS (SELECT max(date_local) AS as_of FROM plays)
SELECT
    sp.track_key,
    any_value(sp.track_name)             AS track_name,
    any_value(sp.artist_name)            AS artist_name,
    count(*)                             AS n_plays,
    count(*) FILTER (WHERE sp.is_play)   AS n_counted,
    count(*) FILTER (WHERE sp.is_skip)   AS n_skips,
    count(DISTINCT sp.session_id)        AS n_sessions,
    count(DISTINCT sp.date_local)        AS n_days,
    min(sp.date_local)                   AS first_played,
    max(sp.date_local)                   AS last_played,
    sum(sp.ms_played) / 60000.0          AS minutes,
    -- Exponentially-weighted recent play count: a play today counts 1.0, a play
    -- one half-life ago counts 0.5, and so on. This is the raw staleness signal.
    sum(exp(-ln(2.0)
            * date_diff('day', sp.date_local, (SELECT as_of FROM bounds))
            / burn_half_life()))         AS burn_raw
FROM session_plays sp
GROUP BY sp.track_key;

-- The crate proper: behaviour + musical features + the derived scores.
CREATE OR REPLACE TABLE crate AS
WITH bounds AS (SELECT max(date_local) AS as_of FROM plays),
scored AS (
    SELECT
        tp.*,
        tf.bpm,
        tf.camelot,
        tf.camelot_number,
        tf.camelot_letter,
        tf.energy,
        tf.feature_source,
        date_diff('day', tp.last_played, (SELECT as_of FROM bounds)) AS days_rested,
        -- Held = played and not skipped past. The complement of the skip rate,
        -- expressed as a Wilson lower bound so thin evidence cannot win.
        wilson_low(tp.n_plays - tp.n_skips, tp.n_plays, 1.96)        AS hold_lcb,
        CAST(tp.n_skips AS DOUBLE) / nullif(tp.n_plays, 0)           AS skip_rate,
        -- Burn is only meaningful relative to the rest of the crate, so it is
        -- ranked rather than used raw: 1.0 = the most-flogged track I own.
        percent_rank() OVER (ORDER BY tp.burn_raw)                   AS rotation_burn
    FROM track_plays tp
    LEFT JOIN track_features tf USING (track_key)
)
SELECT
    *,
    -- Set-readiness: does it hold, and have I rested it? Weighted toward
    -- holding, because a stale banger still works and a floor-killer never does.
    round(0.60 * hold_lcb + 0.40 * (1.0 - rotation_burn), 4) AS set_readiness,
    -- A label for the ordering above, checked most-specific first.
    CASE
        WHEN n_plays < 5                        THEN 'unproven'
        WHEN rotation_burn >= 0.85              THEN 'burned'
        WHEN hold_lcb < 0.35                    THEN 'risky'
        WHEN days_rested >= 90 AND n_plays >= 15 THEN 'rested'
        WHEN hold_lcb >= 0.60                   THEN 'proven'
        ELSE 'working'
    END AS crate_status
FROM scored;

-- Crate composition by tempo family, for the "what can I actually play" read.
-- Buckets follow the tempo ranges a DJ thinks in, not equal-width bins.
CREATE OR REPLACE VIEW crate_tempo_bands AS
SELECT
    CASE
        WHEN bpm IS NULL     THEN 'untagged'
        WHEN bpm < 100       THEN '<100 downtempo'
        WHEN bpm < 118       THEN '100-117 slow'
        WHEN bpm < 130       THEN '118-129 house'
        WHEN bpm < 145       THEN '130-144 techno'
        ELSE                      '145+ fast'
    END                                              AS band,
    count(*)                                         AS n_tracks,
    round(avg(energy), 3)                            AS avg_energy,
    round(avg(hold_lcb), 3)                          AS avg_hold,
    round(avg(set_readiness), 3)                     AS avg_readiness
FROM crate
GROUP BY band;

-- How the crate spreads across the wheel. A crate bunched into three keys is
-- harmonically easy but monotonous; one spread evenly is hard to mix but varied.
CREATE OR REPLACE VIEW crate_key_coverage AS
SELECT
    camelot,
    camelot_number,
    camelot_letter,
    count(*)                      AS n_tracks,
    round(avg(energy), 3)         AS avg_energy,
    round(avg(set_readiness), 3)  AS avg_readiness
FROM crate
WHERE camelot IS NOT NULL
GROUP BY camelot, camelot_number, camelot_letter;

-- Artist-level view of the same question: who is carrying the crate, and who
-- has been over-rotated.
CREATE OR REPLACE VIEW crate_by_artist AS
SELECT
    artist_name,
    count(*)                                   AS n_tracks,
    sum(n_plays)                               AS n_plays,
    round(avg(hold_lcb), 3)                    AS avg_hold,
    round(avg(rotation_burn), 3)               AS avg_burn,
    round(avg(set_readiness), 3)               AS avg_readiness,
    min(days_rested)                           AS days_since_last
FROM crate
GROUP BY artist_name;

-- The two shortlists a DJ actually wants: rest these, and bring these back.
CREATE OR REPLACE VIEW crate_burned AS
SELECT track_name, artist_name, camelot, bpm, n_plays, days_rested,
       round(rotation_burn, 3) AS rotation_burn, round(hold_lcb, 3) AS hold_lcb
FROM crate
WHERE crate_status = 'burned'
ORDER BY rotation_burn DESC, n_plays DESC;

CREATE OR REPLACE VIEW crate_rested AS
SELECT track_name, artist_name, camelot, bpm, n_plays, days_rested,
       round(hold_lcb, 3) AS hold_lcb, round(set_readiness, 3) AS set_readiness
FROM crate
WHERE crate_status = 'rested'
ORDER BY hold_lcb DESC, days_rested DESC;
