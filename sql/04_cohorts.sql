-- 04_cohorts.sql  ·  artist cohort retention
-- See SPEC §6.2 (retention definition), §6.3 (cohort sketch), §7.1.3.
--
-- Cohort = the calendar month of an artist's first COUNTED play. Retention at
-- month k = fraction of that cohort's artists with >=1 counted play in the k-th
-- month after discovery. Only fully-observed (cohort, k) cells are kept, and the
-- scaffold below includes zero-activity cells so the denominator is the true
-- cohort size — this is what avoids survivorship bias (SPEC §12).

CREATE OR REPLACE TABLE artist_first_listen AS
SELECT artist_name,
       date_trunc('month', min(ts_local)) AS cohort_month
FROM plays
WHERE is_play
GROUP BY artist_name;

CREATE OR REPLACE TABLE artist_activity AS
SELECT DISTINCT artist_name,
       date_trunc('month', ts_local) AS active_month
FROM plays
WHERE is_play;

-- Observation window: the final calendar month is partial, so the last FULL
-- month is the latest one we can score retention against.
CREATE OR REPLACE TABLE obs_window AS
SELECT date_trunc('month', max(date_local)) - INTERVAL 1 MONTH AS last_full_month
FROM plays;

CREATE OR REPLACE TABLE retention_by_cohort AS
WITH sizes AS (
    SELECT cohort_month, count(*) AS cohort_size
    FROM artist_first_listen
    GROUP BY cohort_month
),
-- Largest k each cohort can be fully observed at.
kmax AS (
    SELECT s.cohort_month,
           s.cohort_size,
           date_diff('month', s.cohort_month, w.last_full_month) AS max_k
    FROM sizes s, obs_window w
),
-- Every (cohort, k) cell from 0..max_k, including months with zero activity.
scaffold AS (
    SELECT cohort_month, cohort_size, k
    FROM kmax, generate_series(0, max_k) AS g(k)
    WHERE max_k >= 0
),
active AS (
    SELECT f.cohort_month,
           date_diff('month', f.cohort_month, a.active_month) AS k,
           count(DISTINCT a.artist_name) AS active_artists
    FROM artist_first_listen f
    JOIN artist_activity a USING (artist_name)
    WHERE a.active_month >= f.cohort_month
    GROUP BY 1, 2
)
SELECT
    sc.cohort_month,
    sc.k,
    sc.cohort_size,
    coalesce(ac.active_artists, 0)                          AS active_artists,
    coalesce(ac.active_artists, 0)::DOUBLE / sc.cohort_size AS retention
FROM scaffold sc
LEFT JOIN active ac
       ON sc.cohort_month = ac.cohort_month AND sc.k = ac.k
ORDER BY sc.cohort_month, sc.k;

-- Pooled retention curve across all eligible cohorts (size-weighted, so it is
-- the true fraction of at-risk artists still active at month k).
CREATE OR REPLACE TABLE retention_curve AS
SELECT
    k,
    sum(active_artists)::DOUBLE / sum(cohort_size) AS retention,
    count(DISTINCT cohort_month)                   AS n_cohorts,
    sum(cohort_size)                               AS artists_at_risk
FROM retention_by_cohort
GROUP BY k
ORDER BY k;
