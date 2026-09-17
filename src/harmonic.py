"""Harmonic-mixing primitives: the Camelot wheel, BPM matching, and energy arcs.

This is the DJ half of the project's domain model, and it is deliberately pure —
no DuckDB, no I/O, no global state — so every rule here is unit-testable and can
be reasoned about independently of the data pipeline (see SPEC §8.1).

Why the Camelot wheel at all
----------------------------
Two tracks played back-to-back sound "wrong" if their keys clash, and the
standard working tool for avoiding that is the Camelot wheel: a relabelling of
the circle of fifths onto 1..12 with a letter for mode (``A`` = minor,
``B`` = major). ``8B`` is C major, ``8A`` is A minor. Neighbours on the wheel
share all but one note, so the moves DJs actually use are all short hops:

    same code          identical key                     — safest
    +/-1 same letter   one step round the circle of 5ths — the workhorse mix
    same number, flip  relative major/minor              — mood flip, same notes
    +7 same letter     "energy boost" (a semitone up)    — lifts the room
    +/-1 with a flip   diagonal mix                      — usable, less safe
    anything else      clash                             — avoid

Keys arrive as Spotify's ``(pitch_class, mode)`` pair (pitch class 0 = C, 11 = B;
mode 1 = major, 0 = minor), which is also what Rekordbox / Mixed In Key exports
reduce to, so one converter serves both feature providers (SPEC §8.3).

Tempo
-----
BPM compatibility is not "are the numbers close" — it is "can I pitch one into
the other without it sounding sped up". CDJs give roughly +/-6% before artefacts
get obvious, so that is the tolerance used here. It is also *scale-free*: 174 BPM
drum & bass mixes cleanly over 87 BPM hip-hop, so every comparison is taken
against the other track's tempo and its half- and double-time, best match wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

# Circle of fifths in pitch-class order starting at C. Position in this list is
# what the Camelot number encodes: C major sits at 8B, each fifth adds one.
_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Camelot anchors: C major = 8B, A minor = 8A.
_MAJOR_ANCHOR = 8

MoveName = Literal[
    "same_key",
    "adjacent",
    "relative",
    "energy_boost",
    "diagonal",
    "clash",
]

# How good each move sounds, as a 0..1 multiplier. These are judgement calls, not
# measurements — they encode conventional booth practice, and they are the single
# place to tune if the transition model's behaviour needs to change.
MOVE_SCORES: dict[str, float] = {
    "same_key": 1.00,
    "adjacent": 0.90,
    "relative": 0.85,
    "energy_boost": 0.60,
    "diagonal": 0.50,
    "clash": 0.15,
}

# Moves a DJ would happily play in public. Used by the "is this transition
# harmonically compatible" label that the hypothesis test in SPEC §8.5 splits on.
COMPATIBLE_MOVES = frozenset({"same_key", "adjacent", "relative"})

# CDJ pitch fader range before the artefacts get audible, as a fraction.
BPM_TOLERANCE = 0.06
# Below this the match is effectively free — beatgrids line up with no pitching.
BPM_FREE_ZONE = 0.02


@dataclass(frozen=True)
class CamelotKey:
    """A key on the Camelot wheel: ``number`` in 1..12, ``letter`` in {A, B}."""

    number: int
    letter: str

    def __post_init__(self) -> None:
        if not 1 <= self.number <= 12:
            raise ValueError(f"Camelot number out of range: {self.number}")
        if self.letter not in ("A", "B"):
            raise ValueError(f"Camelot letter must be A or B, got {self.letter!r}")

    @property
    def code(self) -> str:
        return f"{self.number}{self.letter}"

    @property
    def is_minor(self) -> bool:
        return self.letter == "A"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.code


def _wrap12(n: int) -> int:
    """Map any integer onto the 1..12 wheel."""
    return ((n - 1) % 12) + 1


def camelot_from_pitch(pitch_class: int, mode: int) -> CamelotKey:
    """Convert Spotify-style ``(pitch_class, mode)`` to a Camelot key.

    ``pitch_class`` is 0=C .. 11=B; ``mode`` is 1 for major, 0 for minor.

    The major number is the position on the circle of fifths offset so that
    C major lands on 8. Since a fifth is 7 semitones and 7 is its own inverse
    mod 12, the position of pitch class ``p`` is ``(7 * p) % 12``. A minor key
    takes the number of its relative major, three semitones up — which is what
    makes ``8A`` (A minor) and ``8B`` (C major) share a number.
    """
    if not 0 <= pitch_class <= 11:
        raise ValueError(f"pitch_class must be 0..11, got {pitch_class}")
    if mode not in (0, 1):
        raise ValueError(f"mode must be 0 (minor) or 1 (major), got {mode}")

    effective = pitch_class if mode == 1 else (pitch_class + 3) % 12
    number = _wrap12(((7 * effective) % 12) + _MAJOR_ANCHOR)
    return CamelotKey(number, "B" if mode == 1 else "A")


def parse_camelot(code: str) -> CamelotKey:
    """Parse a ``"9A"`` / ``"12b"`` style code. Raises on anything malformed."""
    text = (code or "").strip().upper()
    if len(text) < 2:
        raise ValueError(f"not a Camelot code: {code!r}")
    number_part, letter = text[:-1], text[-1]
    if not number_part.isdigit():
        raise ValueError(f"not a Camelot code: {code!r}")
    return CamelotKey(int(number_part), letter)


# Accepted spellings for a musical key, for feature files that use note names
# instead of Camelot codes. Rekordbox writes "Am", Traktor writes "A min",
# Serato writes "A minor", and plenty of tag editors just write "A".
_MODE_SUFFIXES = [
    ("MINOR", 0), ("MIN", 0), ("M", 0),   # longest first: "MIN" must beat "M"
    ("MAJOR", 1), ("MAJ", 1),
]
_NOTE_ALIASES = {
    "DB": 1, "EB": 3, "GB": 6, "AB": 8, "BB": 10,   # flats -> sharps
    "CB": 11, "FB": 4, "E#": 5, "B#": 0,            # rare enharmonics
}


def parse_key(text: Optional[str]) -> Optional[CamelotKey]:
    """Best-effort parse of whatever a feature file calls a key.

    Accepts Camelot codes (``"9A"``), note names with an explicit mode
    (``"Am"``, ``"A min"``, ``"A minor"``), and bare note names (``"A"``, read
    as major, matching the convention of tag editors that omit the mode).
    Returns ``None`` for blanks and anything unrecognised rather than raising —
    a feature file with one bad row should lose that row, not the whole import.
    """
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None

    try:
        return parse_camelot(raw)
    except ValueError:
        pass

    # Note name + optional mode. Normalise unicode-ish sharps/flats first.
    s = raw.upper().replace("\u266f", "#").replace("\u266d", "B").replace(" ", "")
    mode = 1
    for suffix, m in _MODE_SUFFIXES:
        if s.endswith(suffix) and len(s) > len(suffix):
            s, mode = s[: -len(suffix)], m
            break

    if s in _NOTE_ALIASES:
        pitch = _NOTE_ALIASES[s]
    elif s in _PITCH_NAMES:
        pitch = _PITCH_NAMES.index(s)
    else:
        return None
    return camelot_from_pitch(pitch, mode)


def key_name(key: CamelotKey) -> str:
    """Human-readable musical name, e.g. ``9A`` -> ``"E minor"``.

    Inverse of :func:`camelot_from_pitch`: undo the +8 anchor, undo the fifths
    multiplication (7 is self-inverse mod 12), then step back down the minor
    third for minor keys.
    """
    position = (key.number - _MAJOR_ANCHOR) % 12
    effective = (7 * position) % 12
    pitch = effective if key.letter == "B" else (effective - 3) % 12
    return f"{_PITCH_NAMES[pitch]} {'major' if key.letter == 'B' else 'minor'}"


def classify_move(a: CamelotKey, b: CamelotKey) -> MoveName:
    """Name the harmonic move from key ``a`` to key ``b``.

    Order matters: the checks run from safest to loosest so that a move which
    qualifies as several things is reported as the best one. (``8A -> 8A`` is
    both "same key" and trivially "adjacent by 0"; it is reported as same key.)
    """
    step = (b.number - a.number) % 12
    same_letter = a.letter == b.letter

    if same_letter and step == 0:
        return "same_key"
    if same_letter and step in (1, 11):
        return "adjacent"
    if not same_letter and step == 0:
        return "relative"
    if same_letter and step == 7:
        return "energy_boost"
    if not same_letter and step in (1, 11):
        return "diagonal"
    return "clash"


def harmonic_score(a: Optional[CamelotKey], b: Optional[CamelotKey]) -> float:
    """0..1 score for mixing ``a`` into ``b``.

    A missing key on either side returns the neutral 0.5 rather than a penalty:
    an untagged track should not be pushed out of a set just for being untagged
    (real crates are always partially tagged — SPEC §8.3).
    """
    if a is None or b is None:
        return 0.5
    return MOVE_SCORES[classify_move(a, b)]


def is_compatible(a: Optional[CamelotKey], b: Optional[CamelotKey]) -> bool:
    """Would a DJ call this transition harmonically clean?"""
    if a is None or b is None:
        return False
    return classify_move(a, b) in COMPATIBLE_MOVES


def bpm_delta(from_bpm: float, to_bpm: float) -> float:
    """Smallest fractional tempo change needed, allowing half- and double-time.

    Returns ``abs(delta) / from_bpm`` for the best of the three alignments, so
    87 -> 174 reads as a perfect match (0.0), not a 100% jump.
    """
    if from_bpm <= 0 or to_bpm <= 0:
        raise ValueError("BPM must be positive")
    candidates = (to_bpm, to_bpm * 2.0, to_bpm / 2.0)
    return min(abs(c - from_bpm) / from_bpm for c in candidates)


def bpm_score(from_bpm: Optional[float], to_bpm: Optional[float]) -> float:
    """0..1 tempo compatibility, flat inside the free zone then linear to 0.

    Anything inside +/-2% is free (1.0), the score falls linearly to 0 at the
    +/-6% pitch-fader limit, and stays at 0 beyond it. Missing tempo is neutral
    (0.5), same as a missing key.
    """
    if from_bpm is None or to_bpm is None:
        return 0.5
    delta = bpm_delta(from_bpm, to_bpm)
    if delta <= BPM_FREE_ZONE:
        return 1.0
    if delta >= BPM_TOLERANCE:
        return 0.0
    return 1.0 - (delta - BPM_FREE_ZONE) / (BPM_TOLERANCE - BPM_FREE_ZONE)


# --- Energy arcs -----------------------------------------------------------
#
# A set is not a playlist: it has a shape. Each arc maps "how far through the
# set are we" (0..1) to a target energy (0..1). The set builder scores every
# candidate on how close its energy sits to that target, which is what stops a
# purely harmonic optimiser from opening on the biggest track in the crate.

ARCS = {
    # Opening slot: start low, hand over warm but not peaked.
    "warmup": lambda t: 0.30 + 0.35 * t,
    # Prime time: straight in high, small lift, no dips.
    "peak": lambda t: 0.75 + 0.20 * t,
    # Full night in one set: rise, breathe in the middle, rise higher.
    "journey": lambda t: 0.40 + 0.45 * t + 0.18 * (-1.0 if 0.35 < t < 0.6 else 0.0),
    # Last hour: come down deliberately instead of falling off a cliff.
    "closing": lambda t: 0.85 - 0.45 * t,
}


def arc_target(arc: str, position: int, total: int) -> float:
    """Target energy for slot ``position`` (0-based) of a ``total``-track set."""
    if arc not in ARCS:
        raise ValueError(f"unknown arc {arc!r}; expected one of {sorted(ARCS)}")
    if total <= 1:
        return float(ARCS[arc](0.0))
    t = position / (total - 1)
    return float(min(1.0, max(0.0, ARCS[arc](t))))


def energy_score(energy: Optional[float], target: float) -> float:
    """0..1 for how well a track's energy fits the arc target at that slot."""
    if energy is None:
        return 0.5
    return max(0.0, 1.0 - abs(energy - target))
