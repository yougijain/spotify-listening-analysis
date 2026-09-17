"""Tests for the Camelot / BPM / energy-arc primitives.

These pin down the domain rules the rest of the project leans on. If the set
builder ever starts producing odd orderings, the question "is the harmony model
wrong or is the search wrong?" should be answerable by running this file.
"""

from __future__ import annotations

import math

import pytest

from src.harmonic import (
    ARCS,
    BPM_TOLERANCE,
    CamelotKey,
    arc_target,
    bpm_delta,
    bpm_score,
    camelot_from_pitch,
    classify_move,
    energy_score,
    harmonic_score,
    is_compatible,
    key_name,
    parse_camelot,
)

# The published Camelot wheel, used here as an external oracle rather than
# re-deriving the formula the implementation uses.
# (pitch_class, mode) -> code.  mode 1 = major (B), 0 = minor (A).
KNOWN_MAJOR = {
    0: "8B",   # C
    7: "9B",   # G
    2: "10B",  # D
    9: "11B",  # A
    4: "12B",  # E
    11: "1B",  # B
    6: "2B",   # F#
    1: "3B",   # Db
    8: "4B",   # Ab
    3: "5B",   # Eb
    10: "6B",  # Bb
    5: "7B",   # F
}
KNOWN_MINOR = {
    9: "8A",   # Am
    4: "9A",   # Em
    11: "10A",  # Bm
    6: "11A",  # F#m
    1: "12A",  # C#m
    8: "1A",   # G#m
    3: "2A",   # Ebm
    10: "3A",  # Bbm
    5: "4A",   # Fm
    0: "5A",   # Cm
    7: "6A",   # Gm
    2: "7A",   # Dm
}


@pytest.mark.parametrize("pitch,code", sorted(KNOWN_MAJOR.items()))
def test_major_keys_match_the_published_wheel(pitch, code):
    assert camelot_from_pitch(pitch, 1).code == code


@pytest.mark.parametrize("pitch,code", sorted(KNOWN_MINOR.items()))
def test_minor_keys_match_the_published_wheel(pitch, code):
    assert camelot_from_pitch(pitch, 0).code == code


def test_every_pitch_and_mode_maps_to_a_distinct_code():
    codes = {camelot_from_pitch(p, m).code for p in range(12) for m in (0, 1)}
    assert len(codes) == 24


def test_relative_major_and_minor_share_a_number():
    # A minor (pitch 9) is the relative minor of C major (pitch 0).
    assert camelot_from_pitch(9, 0).number == camelot_from_pitch(0, 1).number


@pytest.mark.parametrize("pitch", range(12))
@pytest.mark.parametrize("mode", (0, 1))
def test_key_name_round_trips_through_camelot(pitch, mode):
    key = camelot_from_pitch(pitch, mode)
    name = key_name(key)
    expected_mode = "major" if mode == 1 else "minor"
    assert name.endswith(expected_mode)
    # Re-deriving the pitch from the name must land back where we started.
    from src.harmonic import _PITCH_NAMES

    assert _PITCH_NAMES.index(name.rsplit(" ", 1)[0]) == pitch


def test_parse_camelot_is_case_and_whitespace_insensitive():
    assert parse_camelot(" 9a ") == CamelotKey(9, "A")
    assert parse_camelot("12B") == CamelotKey(12, "B")


@pytest.mark.parametrize("bad", ["", "A", "0A", "13A", "9C", "nine-A", None])
def test_parse_camelot_rejects_junk(bad):
    with pytest.raises(ValueError):
        parse_camelot(bad)


@pytest.mark.parametrize("bad_number", [0, 13, -1])
def test_camelot_number_must_be_on_the_wheel(bad_number):
    with pytest.raises(ValueError):
        CamelotKey(bad_number, "A")


# --- moves -----------------------------------------------------------------

@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("8A", "8A", "same_key"),
        ("8A", "9A", "adjacent"),
        ("8A", "7A", "adjacent"),
        ("1A", "12A", "adjacent"),   # wraps round the bottom of the wheel
        ("12B", "1B", "adjacent"),   # wraps round the top
        ("8A", "8B", "relative"),
        ("8B", "8A", "relative"),
        ("8A", "3A", "energy_boost"),  # +7 on the wheel
        ("8A", "9B", "diagonal"),
        ("8A", "7B", "diagonal"),
        ("8A", "2A", "clash"),
        ("8A", "11B", "clash"),
    ],
)
def test_move_classification(a, b, expected):
    assert classify_move(parse_camelot(a), parse_camelot(b)) == expected


def test_adjacent_and_relative_are_symmetric_but_energy_boost_is_not():
    a, b = parse_camelot("8A"), parse_camelot("9A")
    assert classify_move(a, b) == classify_move(b, a) == "adjacent"

    # +7 one way is -7 (i.e. +5) the other, which is not a named move.
    up, down = parse_camelot("8A"), parse_camelot("3A")
    assert classify_move(up, down) == "energy_boost"
    assert classify_move(down, up) == "clash"


def test_every_key_pair_classifies_without_raising():
    keys = [CamelotKey(n, letter) for n in range(1, 13) for letter in ("A", "B")]
    for a in keys:
        for b in keys:
            assert classify_move(a, b) in {
                "same_key", "adjacent", "relative",
                "energy_boost", "diagonal", "clash",
            }


def test_harmonic_score_orders_moves_sensibly():
    same = harmonic_score(parse_camelot("8A"), parse_camelot("8A"))
    adj = harmonic_score(parse_camelot("8A"), parse_camelot("9A"))
    rel = harmonic_score(parse_camelot("8A"), parse_camelot("8B"))
    clash = harmonic_score(parse_camelot("8A"), parse_camelot("2A"))
    assert same > adj > rel > clash


def test_missing_key_is_neutral_not_penalised():
    # An untagged track scores better than a clash, so it can still be used.
    assert harmonic_score(None, parse_camelot("8A")) == 0.5
    assert harmonic_score(parse_camelot("8A"), None) == 0.5
    assert harmonic_score(None, None) == 0.5
    assert harmonic_score(None, None) > harmonic_score(
        parse_camelot("8A"), parse_camelot("2A")
    )


def test_compatibility_is_the_three_safe_moves_only():
    assert is_compatible(parse_camelot("8A"), parse_camelot("8A"))
    assert is_compatible(parse_camelot("8A"), parse_camelot("9A"))
    assert is_compatible(parse_camelot("8A"), parse_camelot("8B"))
    assert not is_compatible(parse_camelot("8A"), parse_camelot("3A"))
    assert not is_compatible(parse_camelot("8A"), parse_camelot("2A"))
    # Unknown keys are not claimed to be compatible.
    assert not is_compatible(None, parse_camelot("8A"))


# --- tempo -----------------------------------------------------------------

def test_half_and_double_time_count_as_a_match():
    assert bpm_delta(87, 174) == pytest.approx(0.0)
    assert bpm_delta(174, 87) == pytest.approx(0.0)
    assert bpm_score(87, 174) == 1.0


def test_bpm_delta_is_relative_to_the_outgoing_track():
    # 4 BPM matters more at 90 than at 180.
    assert bpm_delta(90, 94) > bpm_delta(180, 184)


def test_bpm_score_curve():
    assert bpm_score(128, 128) == 1.0
    assert bpm_score(128, 130) == 1.0                     # inside the free zone
    assert 0.0 < bpm_score(128, 133) < 1.0                # in the taper
    assert bpm_score(128, 128 * (1 + BPM_TOLERANCE)) == 0.0
    assert bpm_score(128, 200) == 0.0                     # beyond any fader


def test_bpm_score_is_monotonic_in_the_taper():
    scores = [bpm_score(128, 128 + d) for d in range(0, 10)]
    assert scores == sorted(scores, reverse=True)


def test_missing_bpm_is_neutral():
    assert bpm_score(None, 128) == 0.5
    assert bpm_score(128, None) == 0.5


@pytest.mark.parametrize("bad", [0, -1])
def test_bpm_must_be_positive(bad):
    with pytest.raises(ValueError):
        bpm_delta(bad, 128)


# --- arcs ------------------------------------------------------------------

@pytest.mark.parametrize("arc", sorted(ARCS))
def test_arc_targets_stay_in_range(arc):
    assert all(0.0 <= arc_target(arc, i, 20) <= 1.0 for i in range(20))


def test_warmup_rises_and_closing_falls():
    assert arc_target("warmup", 0, 10) < arc_target("warmup", 9, 10)
    assert arc_target("closing", 0, 10) > arc_target("closing", 9, 10)


def test_peak_arc_never_drops_into_warmup_territory():
    assert min(arc_target("peak", i, 12) for i in range(12)) > 0.7


def test_journey_arc_has_a_mid_set_breather():
    targets = [arc_target("journey", i, 20) for i in range(20)]
    mid = targets[7:12]
    assert min(mid) < targets[6]                            # it dips
    assert targets[-1] == pytest.approx(max(targets))       # and still ends highest


def test_single_track_set_uses_the_arc_start():
    assert arc_target("warmup", 0, 1) == pytest.approx(ARCS["warmup"](0.0))


def test_unknown_arc_is_rejected():
    with pytest.raises(ValueError):
        arc_target("afterparty", 0, 10)


def test_energy_score_peaks_on_an_exact_match():
    assert energy_score(0.8, 0.8) == 1.0
    assert energy_score(0.2, 0.8) == pytest.approx(0.4)
    assert energy_score(0.0, 1.0) == 0.0
    assert energy_score(None, 0.8) == 0.5
    assert not math.isnan(energy_score(0.5, 0.5))
