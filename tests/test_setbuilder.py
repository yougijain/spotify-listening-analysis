"""Tests for set construction.

Split in two: pure-objective tests on hand-built crates where the right answer
is known by construction, and end-to-end tests against the real sample crate.
"""

from __future__ import annotations

import random

import pytest

from src.harmonic import parse_camelot
from src.setbuilder import (
    Constraints,
    SetPlan,
    Track,
    build_set,
    build_set_greedy,
    build_set_random,
    evaluate,
    load_crate,
    load_proven_edges,
    score_transition,
)


def track(key, name="T", artist=None, bpm=128.0, camelot="8A", energy=0.7,
          readiness=0.7, status="working", minutes=4.0):
    # Distinct artist per key by default, so a test about tempo is not silently
    # answered by the artist-gap rule.
    return Track(
        track_key=key, track_name=name, artist_name=artist or f"Artist {key}", bpm=bpm,
        key=parse_camelot(camelot) if camelot else None, energy=energy,
        set_readiness=readiness, hold_lcb=readiness, rotation_burn=0.3,
        crate_status=status, minutes=minutes,
    )


def a_crate(n=40, seed=3):
    """A varied crate: every key, a spread of tempos, distinct artists."""
    rng = random.Random(seed)
    codes = [f"{i}{l}" for i in range(1, 13) for l in ("A", "B")]
    return [
        track(f"k{i}", name=f"Track {i}", artist=f"Artist {i}",
              bpm=rng.choice([124.0, 126.0, 128.0, 130.0]),
              camelot=codes[i % len(codes)],
              energy=rng.uniform(0.3, 0.95),
              readiness=rng.uniform(0.3, 0.9))
        for i in range(n)
    ]


# --- objective -------------------------------------------------------------

def test_transition_score_is_a_weighted_blend_in_range():
    t = score_transition(track("a"), track("b"), 1, 10, "peak")
    assert 0.0 <= t.score <= 1.0
    assert t.move == "same_key"


def test_a_perfect_transition_outscores_a_clashing_one():
    src = track("a", camelot="8A", bpm=128)
    clean = score_transition(src, track("b", camelot="9A", bpm=128), 1, 10, "peak")
    clash = score_transition(src, track("c", camelot="2A", bpm=128), 1, 10, "peak")
    assert clean.score > clash.score


def test_tempo_jump_costs_score():
    src = track("a", bpm=128)
    close = score_transition(src, track("b", bpm=129), 1, 10, "peak")
    far = score_transition(src, track("c", bpm=140), 1, 10, "peak")
    assert close.score > far.score
    assert close.tempo > far.tempo


def test_arc_changes_which_energy_wins():
    src = track("a")
    loud, quiet = track("b", energy=0.95), track("c", energy=0.3)
    # Opening slot of a warmup wants the quiet one; peak wants the loud one.
    assert (score_transition(src, quiet, 1, 10, "warmup").energy
            > score_transition(src, loud, 1, 10, "warmup").energy)
    assert (score_transition(src, loud, 1, 10, "peak").energy
            > score_transition(src, quiet, 1, 10, "peak").energy)


def test_proven_edges_break_ties():
    src, dst = track("a"), track("b")
    cold = score_transition(src, dst, 1, 10, "peak", proven=0.0)
    warm = score_transition(src, dst, 1, 10, "peak", proven=0.9)
    assert warm.score > cold.score


def test_untagged_tracks_are_scored_not_rejected():
    src = track("a")
    untagged = track("b", camelot=None, bpm=None)
    t = score_transition(src, untagged, 1, 10, "peak")
    assert t.move is None and t.bpm_delta is None
    assert t.score > 0


# --- constraints -----------------------------------------------------------

def test_a_track_cannot_repeat():
    c = Constraints()
    assert not c.allows([track("a")], track("a"))


def test_artist_gap_is_enforced_then_released():
    c = Constraints(artist_gap=2)
    chosen = [track("a", artist="X"), track("b", artist="Y")]
    assert not c.allows(chosen, track("c", artist="X"))
    chosen.append(track("d", artist="Z"))
    assert c.allows(chosen, track("c", artist="X"))


def test_artist_gap_of_zero_disables_the_rule():
    c = Constraints(artist_gap=0)
    assert c.allows([track("a", artist="X")], track("b", artist="X"))


def test_tempo_jumps_past_the_cap_are_rejected_outright():
    c = Constraints(max_drift=0.06)
    assert not c.allows([track("a", bpm=128)], track("b", bpm=150))
    assert c.allows([track("a", bpm=128)], track("b", bpm=131))


def test_double_time_is_not_treated_as_a_jump():
    c = Constraints(max_drift=0.06)
    assert c.allows([track("a", bpm=87)], track("b", bpm=174))


def test_burned_tracks_are_excluded_by_default():
    assert not Constraints().allows([], track("a", status="burned"))
    assert Constraints(exclude_status=frozenset()).allows([], track("a", status="burned"))


def test_untagged_tempo_cannot_be_rejected_for_drift():
    c = Constraints(max_drift=0.01)
    assert c.allows([track("a", bpm=128)], track("b", bpm=None))


# --- building --------------------------------------------------------------

def test_build_set_respects_every_constraint_it_was_given():
    crate = a_crate(60)
    c = Constraints(artist_gap=3, max_drift=0.06)
    plan = build_set(crate, target_minutes=60, arc="peak", constraints=c)
    keys = [t.track_key for t in plan.tracks]
    assert len(keys) == len(set(keys)), "a track was repeated"
    for i in range(1, len(plan.tracks)):
        window = plan.tracks[max(0, i - 3):i]
        assert plan.tracks[i].artist_name not in {t.artist_name for t in window}
    for t in plan.transitions:
        assert t.bpm_delta is None or t.bpm_delta <= 0.06 + 1e-9


def test_set_length_tracks_the_target_duration():
    crate = a_crate(80)
    short = build_set(crate, target_minutes=30, arc="peak")
    long = build_set(crate, target_minutes=90, arc="peak")
    assert len(short.tracks) < len(long.tracks)
    assert 20 <= short.minutes <= 40


def test_transitions_are_one_fewer_than_tracks():
    plan = build_set(a_crate(50), target_minutes=45, arc="peak")
    assert len(plan.transitions) == len(plan.tracks) - 1


def test_transitions_chain_head_to_tail():
    plan = build_set(a_crate(50), target_minutes=45, arc="peak")
    for i, t in enumerate(plan.transitions):
        assert t.from_key == plan.tracks[i].track_key
        assert t.to_key == plan.tracks[i + 1].track_key


def test_seed_track_pins_the_opener():
    crate = a_crate(50)
    plan = build_set(crate, target_minutes=40, arc="peak", seed_key="k7")
    assert plan.tracks[0].track_key == "k7"


def test_an_unknown_seed_is_an_error_not_a_silent_fallback():
    with pytest.raises(KeyError):
        build_set(a_crate(20), seed_key="not-in-crate")


def test_an_empty_crate_returns_an_empty_plan():
    plan = build_set([])
    assert plan.tracks == [] and plan.mean_score == 0.0


def test_a_single_track_crate_does_not_crash():
    plan = build_set([track("a")], target_minutes=60)
    assert len(plan.tracks) == 1 and plan.transitions == []


def test_a_crate_too_small_for_the_target_returns_a_short_set():
    plan = build_set(a_crate(5), target_minutes=180, arc="peak")
    assert 0 < len(plan.tracks) <= 5


def test_heavily_constrained_crate_still_returns_something_playable():
    # Every track by the same artist, so the artist gap blocks nearly everything.
    crate = [track(f"k{i}", artist="Solo", camelot="8A") for i in range(10)]
    plan = build_set(crate, target_minutes=40, constraints=Constraints(artist_gap=5))
    assert len(plan.tracks) >= 1


def test_the_built_set_is_mostly_harmonic():
    plan = build_set(a_crate(80), target_minutes=60, arc="peak")
    assert plan.harmonic_rate > 0.8


def test_energy_follows_a_closing_arc_downward():
    crate = a_crate(80)
    plan = build_set(crate, target_minutes=60, arc="closing")
    energies = [t.energy for t in plan.tracks if t.energy is not None]
    first_half = sum(energies[:len(energies) // 2]) / max(1, len(energies) // 2)
    second_half = sum(energies[len(energies) // 2:]) / max(1, len(energies) - len(energies) // 2)
    assert first_half > second_half


def test_building_is_deterministic():
    crate = a_crate(60)
    a = build_set(crate, target_minutes=60, arc="peak")
    b = build_set(crate, target_minutes=60, arc="peak")
    assert [t.track_key for t in a.tracks] == [t.track_key for t in b.tracks]


# --- baselines -------------------------------------------------------------

def test_random_baseline_still_honours_constraints():
    crate = a_crate(60)
    plan = build_set_random(crate, 60, "peak", rng=random.Random(1))
    keys = [t.track_key for t in plan.tracks]
    assert len(keys) == len(set(keys))
    for t in plan.transitions:
        assert t.bpm_delta is None or t.bpm_delta <= Constraints().max_drift + 1e-9


def test_beam_beats_random_by_a_wide_margin():
    crate = a_crate(80)
    ev = evaluate(crate, 60, "peak", random_trials=40)
    assert ev.beam > ev.random_best
    assert ev.beam_harmonic_rate > ev.random_harmonic_rate


def test_beam_is_at_least_as_good_as_greedy():
    """The honest claim. Lookahead helps, but the objective and the hard
    constraints do most of the work - see the README."""
    crate = a_crate(80)
    ev = evaluate(crate, 60, "peak", random_trials=10)
    assert ev.beam >= ev.greedy - 1e-9


def test_greedy_is_a_width_one_beam():
    crate = a_crate(40)
    assert (build_set_greedy(crate, target_minutes=40).mean_score
            == pytest.approx(build_set(crate, target_minutes=40,
                                       beam_width=1, branching=1).mean_score))


def test_evaluation_summary_is_reportable():
    ev = evaluate(a_crate(40), 40, "peak", random_trials=5)
    text = ev.summary()
    assert "beam" in text and "greedy" in text and "random" in text


# --- plan reporting --------------------------------------------------------

def test_empty_plan_metrics_do_not_divide_by_zero():
    p = SetPlan()
    assert p.mean_score == 0.0 and p.harmonic_rate == 0.0 and p.minutes == 0.0


def test_describe_lists_every_track():
    plan = build_set(a_crate(40), target_minutes=40, arc="peak")
    text = plan.describe()
    for t in plan.tracks:
        assert t.track_name in text


# --- loading from the warehouse --------------------------------------------

def test_load_crate_reads_the_real_crate(con):
    crate = load_crate(con)
    assert len(crate) > 50
    assert all(isinstance(t, Track) for t in crate)
    assert any(t.key is not None for t in crate)
    assert all(2.0 <= t.minutes <= 12.0 for t in crate)


def test_load_crate_filters_by_plays_and_status(con):
    everything = load_crate(con, min_plays=1)
    filtered = load_crate(con, min_plays=10, exclude_status=("burned",))
    assert len(filtered) < len(everything)
    assert all(t.crate_status != "burned" for t in filtered)


def test_load_proven_edges_returns_scored_pairs(con):
    edges = load_proven_edges(con, min_plays=2)
    assert edges
    for (src, dst), score in list(edges.items())[:50]:
        assert src != dst
        assert 0.0 <= score <= 1.0


def test_a_real_set_from_the_real_crate_holds_together(con):
    crate = load_crate(con)
    plan = build_set(crate, target_minutes=60, arc="peak",
                     proven_edges=load_proven_edges(con))
    assert len(plan.tracks) >= 10
    assert plan.harmonic_rate > 0.75
    assert plan.mean_score > 0.6
    assert 45 <= plan.minutes <= 80


# --- energy continuity -----------------------------------------------------

def test_energy_continuity_is_free_for_small_steps():
    from src.setbuilder import energy_continuity

    assert energy_continuity(0.70, 0.75) == 1.0
    assert energy_continuity(0.70, 0.70) == 1.0


def test_energy_continuity_punishes_a_cliff():
    from src.setbuilder import energy_continuity

    assert energy_continuity(0.90, 0.20) == 0.0
    assert 0.0 < energy_continuity(0.90, 0.55) < 1.0


def test_energy_continuity_is_symmetric_and_neutral_when_unknown():
    from src.setbuilder import energy_continuity

    assert energy_continuity(0.9, 0.5) == energy_continuity(0.5, 0.9)
    assert energy_continuity(None, 0.5) == 0.5


def test_continuity_makes_the_sequencer_avoid_energy_cliffs():
    """The regression this term was added for: without it the optimiser was
    happy to drop 0.9 -> 0.2 -> 0.9 as long as each track sat near the arc."""
    crate = a_crate(80)
    plan = build_set(crate, target_minutes=60, arc="peak")
    assert plan.max_energy_step() < 0.5


def test_a_smooth_step_outscores_a_cliff_all_else_equal():
    src = track("a", energy=0.80, camelot="8A", bpm=128)
    smooth = score_transition(src, track("b", energy=0.78, camelot="8A", bpm=128),
                              1, 10, "peak")
    cliff = score_transition(src, track("c", energy=0.25, camelot="8A", bpm=128),
                             1, 10, "peak")
    assert smooth.continuity > cliff.continuity
    assert smooth.score > cliff.score


# --- arc feasibility -------------------------------------------------------

def test_a_crate_of_bangers_cannot_support_a_warmup():
    from src.setbuilder import arc_feasibility

    peak_only = [track(f"k{i}", artist=f"A{i}", energy=0.9, bpm=138,
                       camelot=f"{(i % 12) + 1}A") for i in range(60)]
    assert arc_feasibility(peak_only, "peak").worst_supply >= 1.0
    assert arc_feasibility(peak_only, "warmup").verdict == "not supported by this crate"


def test_feasibility_counts_tempo_stranded_material_as_unavailable():
    """The distinction the naive version missed.

    The quiet tracks exist, but at 92 BPM they are out of fader reach from a
    138 BPM set, so they cannot actually serve a warmup slot.
    """
    from src.setbuilder import arc_feasibility

    crate = (
        [track(f"loud{i}", artist=f"L{i}", energy=0.9, bpm=138) for i in range(40)]
        + [track(f"quiet{i}", artist=f"Q{i}", energy=0.32, bpm=92) for i in range(40)]
    )
    by_energy_only = arc_feasibility(crate, "warmup", reference_bpm=None, max_drift=10.0)
    reachable = arc_feasibility(crate, "warmup")
    assert by_energy_only.mean_supply > reachable.mean_supply
    assert reachable.stranded > 0


def test_feasibility_predicts_where_the_set_will_miss_its_arc(con):
    """Validates the diagnostic against the thing it claims to predict."""
    from src.setbuilder import arc_feasibility

    crate = load_crate(con)
    scored = [
        (arc_feasibility(crate, arc, 60).mean_supply,
         build_set(crate, 60, arc).arc_deviation(arc))
        for arc in ("warmup", "peak", "journey", "closing")
    ]
    best_supply = max(scored, key=lambda p: p[0])
    worst_supply = min(scored, key=lambda p: p[0])
    assert best_supply[1] < worst_supply[1], "better-supplied arcs should miss less"


def test_feasibility_handles_an_empty_or_untagged_crate():
    from src.setbuilder import arc_feasibility

    assert arc_feasibility([], "peak").mean_supply == 0.0
    untagged = [track(f"k{i}", artist=f"A{i}", energy=None) for i in range(10)]
    assert arc_feasibility(untagged, "peak").mean_supply == 0.0


def test_feasibility_summary_names_the_thin_slot():
    from src.setbuilder import arc_feasibility

    peak_only = [track(f"k{i}", artist=f"A{i}", energy=0.9, bpm=138) for i in range(60)]
    text = arc_feasibility(peak_only, "warmup").summary()
    assert "warmup" in text and "slot" in text


# --- plan diagnostics ------------------------------------------------------

def test_arc_deviation_is_zero_for_a_perfectly_matched_set():
    from src.harmonic import arc_target

    n = 6
    tracks = [track(f"k{i}", artist=f"A{i}", energy=arc_target("peak", i, n))
              for i in range(n)]
    assert SetPlan(tracks=tracks).arc_deviation("peak") == pytest.approx(0.0)


def test_arc_deviation_and_max_step_handle_missing_energy():
    p = SetPlan(tracks=[track("a", energy=None), track("b", energy=None)])
    assert p.arc_deviation("peak") == 0.0
    assert p.max_energy_step() == 0.0


def test_beam_never_loses_to_greedy_on_the_real_crate(con):
    """Regression: on the sample crate's closing arc the raw beam scored 0.001
    below greedy, because pruning can drop a prefix that pays off late."""
    crate = load_crate(con)
    for arc in ("warmup", "peak", "journey", "closing"):
        ev = evaluate(crate, 60, arc, proven_edges=load_proven_edges(con),
                      random_trials=3)
        assert ev.beam >= ev.greedy - 1e-12, f"{arc}: beam {ev.beam} < greedy {ev.greedy}"
