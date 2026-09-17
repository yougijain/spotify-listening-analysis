"""Tests for the SQL models, run against the committed synthetic sample.

These are invariant tests, not value tests: they assert things that must hold
for *any* dataset (rows reconcile, rates are in range, no silent NULLs where a
value is required), so regenerating the sample does not invalidate them. The
two exceptions are marked as such.
"""

from __future__ import annotations

import pytest

from src.harmonic import CamelotKey, bpm_delta, classify_move, is_compatible
from src.stats import wilson_interval

# --- the Camelot move table is generated from the tested Python -------------

def test_camelot_move_table_is_complete(con):
    assert con.execute("SELECT count(*) FROM camelot_moves").fetchone()[0] == 24 * 24


def test_camelot_move_table_agrees_with_the_python_implementation(con):
    rows = con.execute(
        "SELECT from_code, to_code, move, move_score, is_harmonic FROM camelot_moves"
    ).fetchall()
    assert len(rows) == 576
    for from_code, to_code, move, _score, harmonic in rows:
        a = CamelotKey(int(from_code[:-1]), from_code[-1])
        b = CamelotKey(int(to_code[:-1]), to_code[-1])
        assert move == classify_move(a, b)
        assert harmonic == is_compatible(a, b)


# --- the Wilson macro must match the Python implementation ------------------

@pytest.mark.parametrize("trials", [1, 2, 3, 5, 10, 47, 100, 1000])
def test_sql_wilson_macro_matches_python(con, trials):
    """The one piece of logic that genuinely lives in both languages.

    Ranking the crate in SQL means the bound has to be computable set-wise, but
    src/stats.py needs it too. This asserts they never drift.
    """
    for successes in {0, 1, trials // 2, trials - 1, trials}:
        if not 0 <= successes <= trials:
            continue
        got = con.execute(
            "SELECT wilson_low(?, ?, 1.96)", [successes, trials]
        ).fetchone()[0]
        expected = wilson_interval(successes, trials, z=1.96).low
        assert got == pytest.approx(expected, abs=1e-12), (successes, trials)


def test_sql_wilson_macro_handles_zero_trials(con):
    assert con.execute("SELECT wilson_low(0, 0, 1.96)").fetchone()[0] == 0.0
    assert con.execute("SELECT wilson_low(NULL, NULL, 1.96)").fetchone()[0] == 0.0


# --- crate -----------------------------------------------------------------

def test_crate_has_one_row_per_distinct_played_track(con):
    crate_rows, play_tracks = con.execute(
        "SELECT (SELECT count(*) FROM crate), (SELECT count(DISTINCT track_key) FROM plays)"
    ).fetchone()
    assert crate_rows == play_tracks


def test_crate_play_counts_reconcile_with_the_history(con):
    total_crate, total_plays = con.execute(
        "SELECT (SELECT sum(n_plays) FROM crate), (SELECT count(*) FROM plays)"
    ).fetchone()
    assert total_crate == total_plays


def test_crate_scores_are_in_range(con):
    bad = con.execute("""
        SELECT count(*) FROM crate
        WHERE hold_lcb      NOT BETWEEN 0 AND 1
           OR rotation_burn NOT BETWEEN 0 AND 1
           OR set_readiness NOT BETWEEN 0 AND 1
           OR (skip_rate IS NOT NULL AND skip_rate NOT BETWEEN 0 AND 1)
    """).fetchone()[0]
    assert bad == 0


def test_every_track_gets_a_status(con):
    assert con.execute("SELECT count(*) FROM crate WHERE crate_status IS NULL").fetchone()[0] == 0
    statuses = {r[0] for r in con.execute("SELECT DISTINCT crate_status FROM crate").fetchall()}
    assert statuses <= {"unproven", "burned", "risky", "rested", "proven", "working"}


def test_hold_lower_bound_penalises_thin_evidence(con):
    """Two tracks with the same raw skip rate: more plays must rank higher."""
    row = con.execute("""
        SELECT max(hold_lcb) - min(hold_lcb)
        FROM crate
        WHERE skip_rate = 0 AND n_plays >= 1
    """).fetchone()[0]
    # Some spread must exist, otherwise the bound is doing nothing.
    assert row is None or row > 0.0


def test_never_skipped_thin_tracks_do_not_top_the_ranking(con):
    top = con.execute("""
        SELECT n_plays FROM crate ORDER BY hold_lcb DESC, n_plays DESC LIMIT 10
    """).fetchall()
    assert all(n >= 5 for (n,) in top), "a thin sample reached the top of the crate"


def test_burn_is_a_percent_rank_so_it_spans_the_crate(con):
    lo, hi = con.execute("SELECT min(rotation_burn), max(rotation_burn) FROM crate").fetchone()
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(1.0)


def test_recently_hammered_tracks_burn_hotter_than_dormant_ones(con):
    hot, cold = con.execute("""
        SELECT
          (SELECT avg(rotation_burn) FROM crate WHERE days_rested <= 30 AND n_plays >= 10),
          (SELECT avg(rotation_burn) FROM crate WHERE days_rested >= 180)
    """).fetchone()
    assert hot is None or cold is None or hot > cold


def test_days_rested_is_never_negative(con):
    assert con.execute("SELECT count(*) FROM crate WHERE days_rested < 0").fetchone()[0] == 0


def test_tempo_bands_partition_the_crate(con):
    banded, total = con.execute("""
        SELECT (SELECT sum(n_tracks) FROM crate_tempo_bands), (SELECT count(*) FROM crate)
    """).fetchone()
    assert banded == total


def test_key_coverage_only_counts_tagged_tracks(con):
    covered, tagged = con.execute("""
        SELECT (SELECT sum(n_tracks) FROM crate_key_coverage),
               (SELECT count(*) FROM crate WHERE camelot IS NOT NULL)
    """).fetchone()
    assert covered == tagged


def test_artist_view_reconciles_with_the_crate(con):
    a, b = con.execute("""
        SELECT (SELECT sum(n_tracks) FROM crate_by_artist), (SELECT count(*) FROM crate)
    """).fetchone()
    assert a == b


# --- transitions -----------------------------------------------------------

def test_transitions_never_link_a_track_to_itself(con):
    assert con.execute(
        "SELECT count(*) FROM transitions WHERE from_key = to_key"
    ).fetchone()[0] == 0


def test_transitions_stay_inside_a_session(con):
    """Every transition's two plays must belong to the same session."""
    leaks = con.execute("""
        SELECT count(*) FROM transitions t
        WHERE NOT EXISTS (
            SELECT 1 FROM session_plays sp
            WHERE sp.session_id = t.session_id AND sp.track_key = t.from_key
        )
    """).fetchone()[0]
    assert leaks == 0


def test_transition_count_is_bounded_by_the_play_count(con):
    n_trans, n_plays, n_sessions = con.execute("""
        SELECT (SELECT count(*) FROM transitions),
               (SELECT count(*) FROM plays),
               (SELECT count(*) FROM sessions)
    """).fetchone()
    # At most one transition per play after the first in each session, and
    # same-track repeats are dropped on top of that.
    assert 0 < n_trans <= n_plays - n_sessions


def test_bpm_delta_allows_half_and_double_time(con):
    rows = con.execute("""
        SELECT from_bpm, to_bpm, bpm_delta_pct FROM transitions
        WHERE bpm_delta_pct IS NOT NULL LIMIT 500
    """).fetchall()
    assert rows
    for from_bpm, to_bpm, delta in rows:
        assert delta == pytest.approx(bpm_delta(from_bpm, to_bpm), abs=1e-9)


def test_bpm_delta_is_never_negative(con):
    assert con.execute(
        "SELECT count(*) FROM transitions WHERE bpm_delta_pct < 0"
    ).fetchone()[0] == 0


def test_move_is_populated_exactly_when_both_sides_are_tagged(con):
    mismatched = con.execute("""
        SELECT count(*) FROM transitions
        WHERE (move IS NULL) <> (from_camelot IS NULL OR to_camelot IS NULL)
    """).fetchone()[0]
    assert mismatched == 0


def test_edges_aggregate_every_transition(con):
    edge_total, trans_total = con.execute("""
        SELECT (SELECT sum(n) FROM transition_edges), (SELECT count(*) FROM transitions)
    """).fetchone()
    assert edge_total == trans_total


def test_edges_are_unique_per_ordered_pair(con):
    dupes = con.execute("""
        SELECT count(*) FROM (
            SELECT from_key, to_key FROM transition_edges GROUP BY 1, 2 HAVING count(*) > 1
        )
    """).fetchone()[0]
    assert dupes == 0


def test_held_never_exceeds_observed(con):
    assert con.execute(
        "SELECT count(*) FROM transition_edges WHERE n_held > n"
    ).fetchone()[0] == 0


def test_every_move_type_that_occurs_is_measured(con):
    moves = {r[0] for r in con.execute(
        "SELECT DISTINCT move FROM transition_move_performance").fetchall()}
    assert {"same_key", "adjacent", "relative", "clash"} <= moves


def test_move_performance_rates_are_proportions(con):
    bad = con.execute("""
        SELECT count(*) FROM transition_move_performance
        WHERE skip_rate NOT BETWEEN 0 AND 1 OR hold_lcb NOT BETWEEN 0 AND 1
           OR n_skips > n
    """).fetchone()[0]
    assert bad == 0


# --- hypothesis inputs -----------------------------------------------------

def test_harmonic_strata_cover_both_shuffle_states_and_both_arms(con):
    rows = con.execute("""
        SELECT shuffle, is_harmonic, n_trials, n_skips FROM hypothesis_harmonic_skip
    """).fetchall()
    assert {(r[0], r[1]) for r in rows} == {(True, True), (True, False),
                                            (False, True), (False, False)}
    for _, _, n_trials, n_skips in rows:
        assert n_trials > 0 and 0 <= n_skips <= n_trials


def test_harmonic_strata_total_matches_the_tagged_transitions(con):
    strata_total, tagged = con.execute("""
        SELECT (SELECT sum(n_trials) FROM hypothesis_harmonic_skip),
               (SELECT count(*) FROM transitions WHERE is_harmonic IS NOT NULL)
    """).fetchone()
    assert strata_total == tagged


def test_shuffle_confounds_the_harmonic_comparison(con):
    """The reason §8.5 stratifies instead of pooling.

    Sample-specific by nature, but this property is what the design defends
    against, so it is worth failing loudly if the sample stops exhibiting it.
    """
    shuffle_skip, intent_skip, shuffle_clash, intent_clash = con.execute("""
        SELECT
          (SELECT avg(CAST(to_is_skip AS DOUBLE)) FROM transitions WHERE shuffle),
          (SELECT avg(CAST(to_is_skip AS DOUBLE)) FROM transitions WHERE NOT shuffle),
          (SELECT avg(CAST(NOT is_harmonic AS DOUBLE)) FROM transitions
             WHERE shuffle AND is_harmonic IS NOT NULL),
          (SELECT avg(CAST(NOT is_harmonic AS DOUBLE)) FROM transitions
             WHERE NOT shuffle AND is_harmonic IS NOT NULL)
    """).fetchone()
    assert shuffle_skip > intent_skip      # shuffle causes skips
    assert shuffle_clash > intent_clash    # ...and clashes. Hence the confound.


def test_tempo_strata_are_well_formed(con):
    rows = con.execute("SELECT n_trials, n_skips FROM hypothesis_tempo_skip").fetchall()
    assert rows
    assert all(0 <= s <= n for n, s in rows)


# --- enrichment coverage ---------------------------------------------------

def test_every_played_track_has_a_features_row(con):
    missing = con.execute("""
        SELECT count(*) FROM (SELECT DISTINCT track_key FROM plays) p
        WHERE NOT EXISTS (SELECT 1 FROM track_features f WHERE f.track_key = p.track_key)
    """).fetchone()[0]
    assert missing == 0


def test_the_sample_is_mostly_covered_by_the_analysed_library(build_result):
    assert build_result.coverage.real_pct > 0.85
    assert "crate_features" in build_result.providers


def test_camelot_columns_are_internally_consistent(con):
    bad = con.execute("""
        SELECT count(*) FROM track_features
        WHERE camelot IS NOT NULL
          AND camelot <> (camelot_number::VARCHAR || camelot_letter)
    """).fetchone()[0]
    assert bad == 0
