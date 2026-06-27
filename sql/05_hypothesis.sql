-- 05_hypothesis.sql  ·  inputs for the hypothesis test
-- H1: skip rate is higher on shuffle than on intentional plays (SPEC §7.3).
--
-- This file just builds the 2x2 contingency (shuffle x skipped) as counts. The
-- test itself — a two-proportion z-test with effect size — is run in Python
-- (scipy) from these counts, because the analysis layer reports z, p, and the
-- difference/Cohen's h together. The plays-are-autocorrelated caveat (§7.3) is
-- documented alongside the result, not corrected here.

CREATE OR REPLACE TABLE hypothesis_shuffle_skip AS
SELECT
    shuffle,
    count(*)                                     AS n_trials,   -- started tracks
    sum(CASE WHEN is_skip THEN 1 ELSE 0 END)     AS n_skips,
    avg(CASE WHEN is_skip THEN 1.0 ELSE 0.0 END) AS skip_rate
FROM plays
WHERE shuffle IS NOT NULL
GROUP BY shuffle
ORDER BY shuffle;
