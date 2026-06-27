-- 03_metrics.sql  ·  volume, skip rate, concentration, discovery
-- See SPEC §6.2 (definitions), §7.1 (spine), §7.2 (honest-Wrapped metrics).

-- ── Listening volume ──────────────────────────────────────────────────────
-- Monthly volume, flagging the partial first/last months so trend lines can
-- exclude them (SPEC §4.3, §12).
CREATE OR REPLACE TABLE monthly_volume AS
WITH bounds AS (
    SELECT date_trunc('month', min(date_local)) AS first_month,
           date_trunc('month', max(date_local)) AS last_month
    FROM plays
)
SELECT
    date_trunc('month', date_local)                    AS month,
    count(*)                                           AS n_plays,
    count(*) FILTER (WHERE is_play)                    AS n_counted,
    sum(ms_played) / 3600000.0                         AS hours,
    (date_trunc('month', date_local) = b.first_month
     OR date_trunc('month', date_local) = b.last_month) AS is_partial_month
FROM plays, bounds b
GROUP BY 1, b.first_month, b.last_month
ORDER BY 1;

CREATE OR REPLACE TABLE yearly_volume AS
SELECT EXTRACT(year FROM date_local)        AS year,
       count(*) FILTER (WHERE is_play)      AS n_counted,
       sum(ms_played) / 3600000.0           AS hours
FROM plays
GROUP BY 1
ORDER BY 1;

-- Hour x day-of-week grid for the "when do I listen" heatmap.
CREATE OR REPLACE TABLE volume_hour_dow AS
SELECT dow_local,
       any_value(day_name)             AS day_name,
       hour_local,
       count(*) FILTER (WHERE is_play) AS n_counted,
       sum(ms_played) / 60000.0        AS minutes
FROM plays
GROUP BY dow_local, hour_local
ORDER BY dow_local, hour_local;

-- ── Skip rate ─────────────────────────────────────────────────────────────
-- Each started track is one trial; skip_rate = skips / started.
CREATE OR REPLACE TABLE skip_overall AS
SELECT count(*)                                 AS n_started,
       sum(CASE WHEN is_skip THEN 1 ELSE 0 END) AS n_skips,
       avg(CASE WHEN is_skip THEN 1.0 ELSE 0.0 END) AS skip_rate
FROM plays;

CREATE OR REPLACE TABLE skip_by_shuffle AS
SELECT shuffle,
       count(*)                                 AS n_started,
       sum(CASE WHEN is_skip THEN 1 ELSE 0 END) AS n_skips,
       avg(CASE WHEN is_skip THEN 1.0 ELSE 0.0 END) AS skip_rate
FROM plays
GROUP BY shuffle
ORDER BY shuffle;

CREATE OR REPLACE TABLE skip_by_hour AS
SELECT hour_local,
       count(*)                                     AS n_started,
       avg(CASE WHEN is_skip THEN 1.0 ELSE 0.0 END) AS skip_rate
FROM plays
GROUP BY hour_local
ORDER BY hour_local;

-- Discovery vs familiar: a play is "discovery" the first time that track_key
-- ever appears in history, "familiar" on every later play.
CREATE OR REPLACE TABLE skip_by_familiarity AS
WITH ranked AS (
    SELECT is_skip,
           row_number() OVER (PARTITION BY track_key ORDER BY ts_utc) AS appearance
    FROM plays
)
SELECT CASE WHEN appearance = 1 THEN 'discovery' ELSE 'familiar' END AS familiarity,
       count(*)                                     AS n_started,
       avg(CASE WHEN is_skip THEN 1.0 ELSE 0.0 END) AS skip_rate
FROM ranked
GROUP BY 1
ORDER BY 1;

-- ── Taste concentration ───────────────────────────────────────────────────
CREATE OR REPLACE TABLE artist_volume AS
SELECT artist_name,
       sum(ms_played)              AS ms,
       sum(ms_played) / 3600000.0  AS hours,
       count(*) FILTER (WHERE is_play) AS counted_plays
FROM plays
WHERE is_play
GROUP BY artist_name;

-- Top-1% / top-10% artist share of listening time + Herfindahl index (HHI).
-- Percentile cutoffs round up to at least one artist so the metric is defined
-- even on small samples.
CREATE OR REPLACE TABLE concentration AS
WITH ranked AS (
    SELECT ms, row_number() OVER (ORDER BY ms DESC) AS rnk FROM artist_volume
),
tot AS (
    SELECT sum(ms) AS total_ms, count(*) AS n_artists FROM artist_volume
)
SELECT
    t.n_artists,
    greatest(1, CAST(ceil(0.01 * t.n_artists) AS INTEGER)) AS top1_n,
    greatest(1, CAST(ceil(0.10 * t.n_artists) AS INTEGER)) AS top10_n,
    sum(CASE WHEN r.rnk <= greatest(1, CAST(ceil(0.01 * t.n_artists) AS INTEGER))
             THEN r.ms ELSE 0 END)::DOUBLE / t.total_ms      AS top1pct_share,
    sum(CASE WHEN r.rnk <= greatest(1, CAST(ceil(0.10 * t.n_artists) AS INTEGER))
             THEN r.ms ELSE 0 END)::DOUBLE / t.total_ms      AS top10pct_share,
    sum((r.ms::DOUBLE / t.total_ms) * (r.ms::DOUBLE / t.total_ms)) AS hhi
FROM ranked r, tot t
GROUP BY t.n_artists, t.total_ms;

-- Lorenz curve: cumulative share of listening vs cumulative share of artists
-- (artists ordered least- to most-played).
CREATE OR REPLACE TABLE lorenz AS
WITH ranked AS (
    SELECT ms,
           row_number() OVER (ORDER BY ms ASC) AS rnk,
           count(*) OVER ()                    AS n,
           sum(ms)  OVER ()                    AS total
    FROM artist_volume
)
SELECT rnk::DOUBLE / n AS cum_artist_frac,
       sum(ms) OVER (ORDER BY ms ASC ROWS UNBOUNDED PRECEDING)::DOUBLE / total AS cum_listen_frac
FROM ranked
ORDER BY rnk;

-- ── Discovery & binge ─────────────────────────────────────────────────────
-- New artists discovered per month (first counted play). NB: the first month is
-- left-censored — everything heard then looks "new" (caveat in the README).
CREATE OR REPLACE TABLE discovery_monthly AS
WITH firsts AS (
    SELECT artist_name, date_trunc('month', min(ts_local)) AS discover_month
    FROM plays WHERE is_play
    GROUP BY artist_name
)
SELECT discover_month AS month, count(*) AS new_artists
FROM firsts
GROUP BY 1
ORDER BY 1;

CREATE OR REPLACE TABLE most_binged AS
SELECT track_name, artist_name, max(run_len) AS max_consecutive
FROM binges
GROUP BY track_name, artist_name
ORDER BY max_consecutive DESC, track_name
LIMIT 10;
