"""Build a playable set out of the crate.

The analysis up to here answers "what should be in the bag". This module answers
the harder question: **in what order**, which is a sequencing problem with real
constraints, not a sort.

The objective
-------------
Every transition A -> B is scored on four things, weighted (SPEC §9.2):

    harmonic    do the keys mix                      (Camelot move score)
    tempo       can I pitch one into the other       (+/- 6% fader, half/double ok)
    energy      does B sit where the arc wants it    (warmup / peak / journey / closing)
    continuity  is the step up or down survivable    (no cliff between neighbours)
    quality     is B worth playing at all            (crate set-readiness)

Continuity earns its place: an objective that only scores distance-from-target
will happily drop from 0.9 to 0.2 and back if each track individually sits near
the arc, which on a floor reads as the set falling over. Arc fit is a shape
constraint; continuity is a local one, and both are needed.

plus a bonus when the pair appears in the observed transition graph and held —
a mix already proven on my own ears beats one that merely scores well.

Hard constraints reject a candidate outright rather than scoring it down: no
repeats, no artist twice inside a gap, no tempo jump past the pitch fader. Those
are rules, not preferences, and mixing them into the score would let a high
enough harmonic score buy its way past them.

Why beam search
---------------
Greedy sequencing fails a specific, predictable way: it takes the best
transition available now and strands itself in a corner of the wheel with
nothing left that mixes out. Exhaustive search is a travelling-salesman variant
and is not happening over a 250-track crate. Beam search keeps the best `width`
partial sets alive at each step, which recovers most of the lookahead for a
constant factor of work.

:func:`evaluate` exists because "my optimiser works" is a claim that needs a
number. It runs the beam against greedy and random baselines on the same crate
and reports the score distributions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import duckdb

from .harmonic import (
    CamelotKey,
    arc_target,
    bpm_score,
    classify_move,
    harmonic_score,
    parse_camelot,
)

# Objective weights. They sum to 1 so a transition score is readable as 0..1.
W_HARMONIC = 0.30
W_TEMPO = 0.20
W_ENERGY = 0.20
W_CONTINUITY = 0.12
W_QUALITY = 0.18

# Energy step a mix absorbs without anyone noticing. Beyond this the penalty
# ramps, hitting zero at ENERGY_CLIFF.
ENERGY_STEP_FREE = 0.15
ENERGY_CLIFF = 0.55

# Extra credit for a pair I have actually played back-to-back and stayed with.
# Deliberately small: it should break ties, not override the musical scoring,
# because the history reflects listening habits and not dancefloor results.
W_PROVEN = 0.10

# Defaults for the hard constraints.
DEFAULT_ARTIST_GAP = 3        # no artist twice within this many slots
DEFAULT_MAX_DRIFT = 0.08      # fractional tempo jump a transition may not exceed
DEFAULT_BEAM_WIDTH = 12
DEFAULT_BRANCHING = 20        # candidates considered per partial set per step
DEFAULT_TRACK_MINUTES = 4.0   # assumed length when a track has no duration


@dataclass(frozen=True)
class Track:
    """A crate entry, with everything the objective needs and nothing else."""

    track_key: str
    track_name: str
    artist_name: str
    bpm: Optional[float] = None
    key: Optional[CamelotKey] = None
    energy: Optional[float] = None
    set_readiness: float = 0.5
    hold_lcb: float = 0.5
    rotation_burn: float = 0.5
    crate_status: str = "working"
    minutes: float = DEFAULT_TRACK_MINUTES

    @property
    def camelot(self) -> Optional[str]:
        return self.key.code if self.key else None

    def label(self) -> str:
        bits = [f"{self.track_name} — {self.artist_name}"]
        if self.camelot:
            bits.append(self.camelot)
        if self.bpm:
            bits.append(f"{self.bpm:.0f} BPM")
        return "  ".join(bits)


@dataclass(frozen=True)
class Transition:
    """One scored A -> B step, kept so a set can explain itself."""

    from_key: str
    to_key: str
    score: float
    harmonic: float
    tempo: float
    energy: float
    continuity: float
    quality: float
    proven: float
    move: Optional[str]
    bpm_delta: Optional[float]


@dataclass
class SetPlan:
    """An ordered set plus the transitions between its tracks."""

    tracks: list = field(default_factory=list)
    transitions: list = field(default_factory=list)

    @property
    def total_score(self) -> float:
        return sum(t.score for t in self.transitions)

    @property
    def mean_score(self) -> float:
        """Average transition quality, so sets of different lengths compare."""
        return self.total_score / len(self.transitions) if self.transitions else 0.0

    @property
    def minutes(self) -> float:
        return sum(t.minutes for t in self.tracks)

    @property
    def harmonic_rate(self) -> float:
        """Share of transitions a DJ would call harmonically clean."""
        scored = [t for t in self.transitions if t.move is not None]
        if not scored:
            return 0.0
        clean = sum(1 for t in scored if t.move in ("same_key", "adjacent", "relative"))
        return clean / len(scored)

    def arc_deviation(self, arc: str) -> float:
        """Mean absolute gap between the set's energy and the arc it targeted.

        The number that keeps the arc claim honest: a set can be 100% harmonic
        and still ignore the shape it was asked for.
        """
        energies = [(i, t.energy) for i, t in enumerate(self.tracks)
                    if t.energy is not None]
        if not energies:
            return 0.0
        n = len(self.tracks)
        return sum(abs(e - arc_target(arc, i, n)) for i, e in energies) / len(energies)

    def max_energy_step(self) -> float:
        """Biggest energy jump between neighbouring tracks."""
        energies = [t.energy for t in self.tracks if t.energy is not None]
        if len(energies) < 2:
            return 0.0
        return max(abs(b - a) for a, b in zip(energies, energies[1:]))

    def describe(self) -> str:
        lines = [f"{len(self.tracks)} tracks · {self.minutes:.0f} min · "
                 f"mean transition {self.mean_score:.3f} · "
                 f"{self.harmonic_rate:.0%} harmonic"]
        for i, track in enumerate(self.tracks):
            prefix = f"{i + 1:2d}. "
            if i == 0:
                lines.append(f"{prefix}{track.label()}")
            else:
                t = self.transitions[i - 1]
                drift = f"{t.bpm_delta * 100:+.1f}%" if t.bpm_delta is not None else "  ? "
                lines.append(f"{prefix}{track.label()}"
                             f"   [{t.move or 'untagged'} {drift} → {t.score:.2f}]")
        return "\n".join(lines)


def _slots_for(target_minutes: float, tracks: Sequence[Track]) -> int:
    """How many slots a target duration implies, from the crate's average length."""
    if not tracks:
        return 0
    avg = sum(t.minutes for t in tracks) / len(tracks) or DEFAULT_TRACK_MINUTES
    return max(2, min(len(tracks), round(target_minutes / avg)))


def score_transition(
    a: Track,
    b: Track,
    position: int,
    total: int,
    arc: str = "peak",
    proven: float = 0.0,
) -> Transition:
    """Score the step from ``a`` to ``b`` landing in slot ``position``.

    ``proven`` is the observed hold rate for this pair (0 if never played), and
    contributes on top of the weighted components rather than inside them.
    """
    harmonic = harmonic_score(a.key, b.key)
    tempo = bpm_score(a.bpm, b.bpm)
    energy_fit = _energy_fit(b, arc, position, total)
    continuity = energy_continuity(a.energy, b.energy)
    quality = b.set_readiness

    total_score = (
        W_HARMONIC * harmonic
        + W_TEMPO * tempo
        + W_ENERGY * energy_fit
        + W_CONTINUITY * continuity
        + W_QUALITY * quality
        + W_PROVEN * proven
    )
    move = classify_move(a.key, b.key) if (a.key and b.key) else None
    delta = _bpm_delta_or_none(a.bpm, b.bpm)
    return Transition(a.track_key, b.track_key, total_score, harmonic, tempo,
                      energy_fit, continuity, quality, proven, move, delta)


def energy_continuity(from_energy: Optional[float], to_energy: Optional[float]) -> float:
    """1..0 for how survivable the energy step between two tracks is.

    Free inside ``ENERGY_STEP_FREE``, then linear to 0 at ``ENERGY_CLIFF``.
    Unknown energy is neutral, like every other missing feature.
    """
    if from_energy is None or to_energy is None:
        return 0.5
    step = abs(to_energy - from_energy)
    if step <= ENERGY_STEP_FREE:
        return 1.0
    if step >= ENERGY_CLIFF:
        return 0.0
    return 1.0 - (step - ENERGY_STEP_FREE) / (ENERGY_CLIFF - ENERGY_STEP_FREE)


def _energy_fit(track: Track, arc: str, position: int, total: int) -> float:
    from .harmonic import energy_score

    return energy_score(track.energy, arc_target(arc, position, total))


def _bpm_delta_or_none(a: Optional[float], b: Optional[float]) -> Optional[float]:
    from .harmonic import bpm_delta

    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return bpm_delta(a, b)


@dataclass(frozen=True)
class Constraints:
    """The rules a candidate must satisfy to be considered at all."""

    artist_gap: int = DEFAULT_ARTIST_GAP
    max_drift: float = DEFAULT_MAX_DRIFT
    exclude_status: frozenset = frozenset({"burned"})

    def allows(self, chosen: Sequence[Track], candidate: Track) -> bool:
        if candidate.crate_status in self.exclude_status:
            return False
        if any(t.track_key == candidate.track_key for t in chosen):
            return False
        if self.artist_gap > 0:
            recent = chosen[-self.artist_gap:]
            if any(t.artist_name == candidate.artist_name for t in recent):
                return False
        if chosen:
            delta = _bpm_delta_or_none(chosen[-1].bpm, candidate.bpm)
            if delta is not None and delta > self.max_drift:
                return False
        return True


@dataclass
class _Beam:
    """A partial set inside the search."""

    tracks: list
    transitions: list
    score: float


def build_set(
    tracks: Sequence[Track],
    target_minutes: float = 60.0,
    arc: str = "peak",
    seed_key: Optional[str] = None,
    constraints: Optional[Constraints] = None,
    proven_edges: Optional[dict] = None,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    branching: int = DEFAULT_BRANCHING,
) -> SetPlan:
    """Sequence a set from the crate by beam search.

    ``seed_key`` pins the opening track; otherwise the openers whose energy best
    fits the start of the arc are tried, one per beam slot, so the search is not
    hostage to a single first guess.

    Returns the best plan found. A crate too small or too constrained to fill
    the target simply returns a shorter set rather than failing — a 40-minute
    set is a usable answer, a crash is not.
    """
    if not tracks:
        return SetPlan()
    constraints = constraints or Constraints()
    proven_edges = proven_edges or {}
    slots = _slots_for(target_minutes, tracks)
    if slots < 2:
        return SetPlan(tracks=list(tracks[:1]))

    by_key = {t.track_key: t for t in tracks}
    openers = _pick_openers(tracks, arc, slots, seed_key, by_key, beam_width,
                            constraints)
    if not openers:
        return SetPlan()

    beams = [_Beam([t], [], 0.0) for t in openers]

    for position in range(1, slots):
        nxt: list = []
        for beam in beams:
            last = beam.tracks[-1]
            candidates = [
                score_transition(last, cand, position, slots, arc,
                                 proven_edges.get((last.track_key, cand.track_key), 0.0))
                for cand in tracks
                if constraints.allows(beam.tracks, cand)
            ]
            if not candidates:
                nxt.append(beam)                   # dead end: carry it unchanged
                continue
            candidates.sort(key=lambda t: t.score, reverse=True)
            for trans in candidates[:branching]:
                nxt.append(_Beam(
                    beam.tracks + [by_key[trans.to_key]],
                    beam.transitions + [trans],
                    beam.score + trans.score,
                ))
        if not nxt:
            break
        # Rank on mean score, not total, so a beam that dead-ended early is not
        # rewarded for being short.
        nxt.sort(key=lambda b: b.score / max(1, len(b.transitions)), reverse=True)
        beams = _dedupe(nxt)[:beam_width]

    best = max(beams, key=lambda b: b.score / max(1, len(b.transitions)))
    return SetPlan(tracks=best.tracks, transitions=best.transitions)


def _dedupe(beams: Sequence[_Beam]) -> list:
    """Drop beams holding the same tracks in the same order.

    Without this the beam fills with near-identical sequences and the effective
    width collapses, which is the classic way a beam search quietly degrades
    into greedy.
    """
    seen, out = set(), []
    for beam in beams:
        signature = tuple(t.track_key for t in beam.tracks)
        if signature not in seen:
            seen.add(signature)
            out.append(beam)
    return out


def _pick_openers(
    tracks: Sequence[Track],
    arc: str,
    slots: int,
    seed_key: Optional[str],
    by_key: dict,
    beam_width: int,
    constraints: Constraints,
) -> list:
    if seed_key is not None:
        found = by_key.get(seed_key)
        if found is None:
            raise KeyError(f"seed track not in crate: {seed_key!r}")
        return [found]

    eligible = [t for t in tracks if constraints.allows([], t)]
    if not eligible:
        # Every track is excluded by status; the set has to open on something.
        eligible = list(tracks)
    eligible.sort(
        key=lambda t: (_energy_fit(t, arc, 0, slots), t.set_readiness), reverse=True
    )
    return eligible[:beam_width]


# --- can the crate even do this? -------------------------------------------

# How far from an arc's target a track can sit and still serve that slot.
ARC_TOLERANCE = 0.15
# Tracks needed within tolerance before a slot counts as properly supplied.
# Fewer than this and the optimiser has no real choice, so the other
# constraints (artist gap, tempo, key) will force a compromise somewhere.
ARC_DEPTH = 8


@dataclass(frozen=True)
class ArcFeasibility:
    """Whether the crate contains material an arc asks for *and can reach*.

    Worth computing before blaming the sequencer. A crate that is 90% peak-time
    material cannot open a night, and no amount of search fixes that — the
    honest output is "you are missing opener material", not a warmup set
    assembled out of bangers.

    Reachability is the part that is easy to get wrong. Counting tracks by
    energy alone says the crate is fine, because the quiet tracks do exist —
    they are just downtempo, and you cannot get from a 137 BPM floor to a 92 BPM
    track through a +/-8% pitch fader. Energy and tempo are correlated in any
    real crate, so a slot is only genuinely supplied if its candidates are also
    tempo-reachable from where the set is running.
    """

    arc: str
    mean_supply: float
    worst_slot: int
    worst_supply: float
    worst_target: float
    n_slots: int
    reference_bpm: Optional[float]
    stranded: int

    @property
    def verdict(self) -> str:
        if self.worst_supply >= 1.0:
            return "well supplied"
        if self.worst_supply >= 0.5:
            return "thin in places"
        return "not supported by this crate"

    def summary(self) -> str:
        stranded = (f", {self.stranded} on-energy tracks stranded out of tempo range"
                    if self.stranded else "")
        return (f"{self.arc}: {self.verdict} "
                f"(mean supply {self.mean_supply:.0%}, thinnest at slot "
                f"{self.worst_slot + 1}/{self.n_slots} wanting energy "
                f"{self.worst_target:.2f}{stranded})")


def _median(values: Sequence[float]) -> Optional[float]:
    ordered = sorted(values)
    if not ordered:
        return None
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def arc_feasibility(
    tracks: Sequence[Track],
    arc: str = "peak",
    target_minutes: float = 60.0,
    tolerance: float = ARC_TOLERANCE,
    depth: int = ARC_DEPTH,
    reference_bpm: Optional[float] = None,
    max_drift: float = DEFAULT_MAX_DRIFT,
) -> ArcFeasibility:
    """Measure, slot by slot, how much *reachable* material the crate has.

    ``reference_bpm`` is where the set is assumed to be running; it defaults to
    the crate's median tempo. A candidate counts toward a slot only if it is
    both within ``tolerance`` of that slot's energy target and within
    ``max_drift`` of the reference tempo (half- and double-time allowed).

    Supply is capped at 1.0 per slot: having 40 candidates where 8 would do is
    not four times better, it just means that slot is solved.
    """
    usable = [t for t in tracks if t.energy is not None]
    slots = _slots_for(target_minutes, tracks) if tracks else 0
    if not usable or slots < 1:
        return ArcFeasibility(arc, 0.0, 0, 0.0, 0.0, slots, reference_bpm, 0)

    if reference_bpm is None:
        reference_bpm = _median([t.bpm for t in usable if t.bpm])

    def reachable(track: Track) -> bool:
        if reference_bpm is None or track.bpm is None:
            return True                  # untagged tempo is not held against it
        delta = _bpm_delta_or_none(reference_bpm, track.bpm)
        return delta is None or delta <= max_drift

    supplies, stranded = [], set()
    for i in range(slots):
        target = arc_target(arc, i, slots)
        # Epsilon because a track exactly on the tolerance boundary should
        # count: abs(0.90 - 0.75) is 0.15000000000000002 in binary floating
        # point, and a slot flipping to "unsupplied" on that is nonsense.
        on_energy = [t for t in usable if abs(t.energy - target) <= tolerance + 1e-9]
        within = [t for t in on_energy if reachable(t)]
        stranded.update(t.track_key for t in on_energy if not reachable(t))
        supplies.append((min(1.0, len(within) / depth), target))

    worst_slot = min(range(slots), key=lambda i: supplies[i][0])
    return ArcFeasibility(
        arc=arc,
        mean_supply=sum(s for s, _ in supplies) / slots,
        worst_slot=worst_slot,
        worst_supply=supplies[worst_slot][0],
        worst_target=supplies[worst_slot][1],
        n_slots=slots,
        reference_bpm=reference_bpm,
        stranded=len(stranded),
    )


# --- baselines and evaluation ----------------------------------------------

def build_set_greedy(tracks: Sequence[Track], **kwargs) -> SetPlan:
    """Beam width 1. The thing beam search has to beat to justify itself."""
    kwargs["beam_width"] = 1
    kwargs["branching"] = 1
    return build_set(tracks, **kwargs)


def build_set_random(
    tracks: Sequence[Track],
    target_minutes: float = 60.0,
    arc: str = "peak",
    constraints: Optional[Constraints] = None,
    proven_edges: Optional[dict] = None,
    rng: Optional[random.Random] = None,
) -> SetPlan:
    """A constraint-respecting random ordering: the floor for the objective.

    Constraints are still honoured, so this is not a strawman — it is "what you
    get from shuffling a crate you already curated".
    """
    rng = rng or random.Random()
    constraints = constraints or Constraints()
    proven_edges = proven_edges or {}
    slots = _slots_for(target_minutes, tracks)
    if slots < 2:
        return SetPlan(tracks=list(tracks[:1]))

    pool = list(tracks)
    rng.shuffle(pool)
    chosen: list = [pool[0]]
    transitions: list = []
    for position in range(1, slots):
        options = [t for t in pool if constraints.allows(chosen, t)]
        if not options:
            break
        pick = rng.choice(options)
        transitions.append(score_transition(
            chosen[-1], pick, position, slots, arc,
            proven_edges.get((chosen[-1].track_key, pick.track_key), 0.0)))
        chosen.append(pick)
    return SetPlan(tracks=chosen, transitions=transitions)


@dataclass
class Evaluation:
    """Beam vs greedy vs random on the same crate and the same objective."""

    beam: float
    greedy: float
    random_mean: float
    random_best: float
    random_trials: int
    beam_harmonic_rate: float
    random_harmonic_rate: float
    beam_len: int

    @property
    def lift_over_greedy(self) -> float:
        return self.beam - self.greedy

    @property
    def lift_over_random(self) -> float:
        return self.beam - self.random_mean

    def summary(self) -> str:
        return (f"beam {self.beam:.3f} | greedy {self.greedy:.3f} "
                f"(+{self.lift_over_greedy:.3f}) | random {self.random_mean:.3f} "
                f"best-of-{self.random_trials} {self.random_best:.3f} "
                f"(+{self.lift_over_random:.3f}) | harmonic "
                f"{self.beam_harmonic_rate:.0%} vs {self.random_harmonic_rate:.0%}")


def evaluate(
    tracks: Sequence[Track],
    target_minutes: float = 60.0,
    arc: str = "peak",
    proven_edges: Optional[dict] = None,
    random_trials: int = 200,
    seed: int = 7,
    **kwargs,
) -> Evaluation:
    """Measure the beam against its baselines.

    Random is run many times and reported as both a mean and a best-of, because
    beating the average shuffle is easy and beating the luckiest of 200 is the
    claim actually worth making.
    """
    rng = random.Random(seed)
    beam = build_set(tracks, target_minutes, arc, proven_edges=proven_edges, **kwargs)
    greedy = build_set_greedy(tracks, target_minutes=target_minutes, arc=arc,
                              proven_edges=proven_edges)
    randoms = [
        build_set_random(tracks, target_minutes, arc, proven_edges=proven_edges, rng=rng)
        for _ in range(random_trials)
    ]
    scores = [r.mean_score for r in randoms]
    return Evaluation(
        beam=beam.mean_score,
        greedy=greedy.mean_score,
        random_mean=sum(scores) / len(scores),
        random_best=max(scores),
        random_trials=random_trials,
        beam_harmonic_rate=beam.harmonic_rate,
        random_harmonic_rate=sum(r.harmonic_rate for r in randoms) / len(randoms),
        beam_len=len(beam.tracks),
    )


# --- loading from the warehouse --------------------------------------------

def load_crate(
    con: duckdb.DuckDBPyConnection,
    min_plays: int = 1,
    exclude_status: Iterable[str] = (),
) -> list:
    """Read the `crate` table into :class:`Track` objects."""
    exclusions = tuple(exclude_status)
    sql = """
        SELECT track_key, track_name, artist_name, bpm, camelot, energy,
               set_readiness, hold_lcb, rotation_burn, crate_status,
               minutes / nullif(n_plays, 0) AS avg_minutes
        FROM crate
        WHERE n_plays >= ?
    """
    params: list = [min_plays]
    if exclusions:
        sql += f" AND crate_status NOT IN ({','.join('?' * len(exclusions))})"
        params.extend(exclusions)

    out = []
    for row in con.execute(sql, params).fetchall():
        (key, name, artist, bpm, camelot, energy, readiness,
         hold, burn, status, avg_minutes) = row
        out.append(Track(
            track_key=key, track_name=name, artist_name=artist,
            bpm=float(bpm) if bpm else None,
            key=parse_camelot(camelot) if camelot else None,
            energy=float(energy) if energy is not None else None,
            set_readiness=float(readiness or 0.0),
            hold_lcb=float(hold or 0.0),
            rotation_burn=float(burn or 0.0),
            crate_status=status or "working",
            # A track's own average play length is a better duration estimate
            # than a constant, but skips drag it down, so it is clamped.
            minutes=min(12.0, max(2.0, float(avg_minutes or DEFAULT_TRACK_MINUTES))),
        ))
    return out


def load_proven_edges(con: duckdb.DuckDBPyConnection, min_plays: int = 2) -> dict:
    """Observed transitions that held, as ``{(from_key, to_key): hold_lcb}``.

    ``min_plays`` keeps one-off pairs out: a transition played once and not
    skipped is not evidence of anything.
    """
    rows = con.execute(
        "SELECT from_key, to_key, hold_lcb FROM transition_edges WHERE n >= ?",
        [min_plays],
    ).fetchall()
    return {(f, t): float(h or 0.0) for f, t, h in rows}
