"""Generate a synthetic Spotify *Extended Streaming History* sample.

The real export (`Streaming_History_Audio_*.json`) is personal data and is never
committed (SPEC §4.1, §14). This script fabricates a realistic, fully synthetic
stand-in so the whole pipeline runs end-to-end with no private data — and so the
analyses have something interesting to find:

* power-law artist popularity        -> meaningful taste concentration (HHI)
* artists with debut + lifespan       -> real cohort-retention structure
* sessions clustered by hour/weekday  -> a non-trivial hour x weekday heatmap
* higher skip rate on shuffle         -> a real effect for the hypothesis test
* podcasts, partial edge months,      -> exercises every cleaning rule in §4.3
  null `skipped`, sub-second plays
* harmonically-aware track sequencing -> a real transition graph to mine

Two artefacts come out of this, mirroring what a DJ actually has on disk:

1. ``Streaming_History_Audio_*.json`` — faithful to the real export, which
   carries **no** musical features (no BPM, no key). Nothing is smuggled in.
2. ``crate_features.csv`` — a Mixed In Key / Rekordbox-shaped side-car. This is
   where BPM/key/energy come from in the real world too, because Spotify's
   audio-features endpoint was closed to new apps in Nov 2024 (SPEC §8.3).

**On the harmonic effect being "found" later:** the sequencer below deliberately
queues harmonically-adjacent tracks more often than chance and skips more on a
clash. So the hypothesis test in §8.5 is *guaranteed* to find an effect here —
it demonstrates the measurement machinery, not a discovery. The real experiment
is running it against a real export.

Everything is seeded, so the committed sample is reproducible. Output mimics the
real export: a couple of `Streaming_History_Audio_*.json` files in data/sample/.

Run:  python src/generate_sample.py
"""

from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

SEED = 42
START = date(2021, 1, 20)          # partial first month (starts on the 20th)
END = date(2023, 12, 10)           # partial last month (ends on the 10th)
HOME_OFFSET = timedelta(hours=5, minutes=30)  # IST; `ts` is stored as UTC
N_ARTISTS = 45
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"

PLATFORMS = ["Android OS 13 (Pixel 7)", "iOS 16.5 (iPhone13,2)",
             "Windows 10 (10.0.19045)", "OS X 13.4 (Macbook)", "Web Player"]
PLATFORM_P = [0.42, 0.20, 0.22, 0.10, 0.06]
COUNTRIES = ["IN", "US", "GB", "AE"]
COUNTRY_P = [0.90, 0.06, 0.02, 0.02]

# Local-hour weights (0..23): quiet overnight, commute + lunch + evening peaks.
HOUR_W_WEEKDAY = np.array([
    1, 1, 1, 1, 1, 2, 4, 8, 12, 9, 7, 6, 8, 9, 6, 5, 6, 9, 14, 16, 15, 12, 8, 4
], dtype=float)
HOUR_W_WEEKEND = np.array([
    1, 1, 1, 1, 1, 1, 2, 3, 5, 8, 11, 12, 12, 11, 11, 10, 10, 11, 12, 13, 13, 12, 9, 5
], dtype=float)

ADJECTIVES = [
    "Velvet", "Neon", "Paper", "Glass", "Midnight", "Golden", "Silent", "Electric",
    "Wild", "Hollow", "Crimson", "Lunar", "Coastal", "Marble", "Echo", "Amber",
    "Iron", "Saffron", "Cobalt", "Phantom", "Quiet", "Northern", "Drifting", "Cosmic",
]
NOUNS = [
    "Foxes", "Avenue", "Tigers", "Atlas", "Harbor", "Circuit", "Garden", "Static",
    "Mirage", "Comet", "Pines", "Anchor", "Signal", "Meadow", "Lantern", "Current",
    "Cathedral", "Voyage", "Cinder", "Halo", "Tide", "Ember", "Parade", "Monsoon",
]
TITLE_WORDS = [
    "Runaway", "Slow Burn", "Paper Moons", "Gravity", "Aftertaste", "Open Road",
    "Ceilings", "Saltwater", "Static Bloom", "Backseat", "Overgrown", "Daydream",
    "Cold Glass", "Half Light", "Undertow", "Featherweight", "Lowlands", "Afterglow",
    "Stillwater", "Rooftops", "Telegraph", "Ferris Wheel", "Marrow", "Wildfire",
    "Coastline", "Nightshift", "Soft Focus", "Bittersweet", "Long Division", "Mercury",
]
ALBUM_WORDS = ["Vol. I", "Vol. II", "Sketches", "Live at Home", "B-Sides",
               "The Quiet Year", "Reissue", "Demos", "Singles"]
PODCASTS = [
    ("The Long Game", ["Compounding Time", "On Boredom", "The 10x Myth",
                       "Notes on Focus", "Why Defaults Win"]),
    ("Tape Deck", ["The Loudness Wars", "Sampling 101", "Liner Notes",
                   "Lost Records", "The B-Side Theory"]),
    ("Edge Cases", ["Off-by-One", "The Null Hypothesis", "Race Conditions",
                    "Postmortem", "Cache Invalidation"]),
]


# --- musical features ------------------------------------------------------
#
# Each artist gets a booth identity: a home key, a tempo centre and an energy
# band. Tracks scatter around it, which is what makes an artist's catalogue
# mixable with itself and gives the crate real harmonic structure rather than
# uniform noise.

CAMELOT_CODES = [f"{n}{l}" for n in range(1, 13) for l in ("A", "B")]

# Tempo families a DJ would recognise, with the energy band that tends to go
# with them. (label, bpm_centre, bpm_spread, energy_low, energy_high)
TEMPO_FAMILIES = [
    ("downtempo",  92,  6, 0.20, 0.45),
    ("house",     124,  4, 0.55, 0.80),
    ("techno",    138,  5, 0.70, 0.95),
    ("garage",    134,  4, 0.60, 0.85),
    ("dnb",       174,  4, 0.75, 0.98),
]
TEMPO_FAMILY_P = [0.22, 0.34, 0.20, 0.14, 0.10]


def _camelot_neighbours(code: str) -> list:
    """Codes a DJ would mix into `code`: itself, +/-1 on the wheel, relative."""
    number, letter = int(code[:-1]), code[-1]
    other = "B" if letter == "A" else "A"
    wrap = lambda n: ((n - 1) % 12) + 1
    return [
        f"{number}{letter}",
        f"{wrap(number + 1)}{letter}",
        f"{wrap(number - 1)}{letter}",
        f"{number}{other}",
    ]


def assign_features(rng: np.random.Generator, artists) -> None:
    """Give every artist a tempo family + home key, and every track features.

    Mutates `artists` in place, adding `family`/`home_key` to each artist and
    `bpm`/`camelot`/`energy` to each track.
    """
    for artist in artists:
        fi = int(rng.choice(len(TEMPO_FAMILIES), p=TEMPO_FAMILY_P))
        label, centre, spread, e_lo, e_hi = TEMPO_FAMILIES[fi]
        home = str(rng.choice(CAMELOT_CODES))
        artist["family"] = label
        artist["home_key"] = home
        pool = _camelot_neighbours(home)
        for track in artist["tracks"]:
            # Most of a catalogue sits in or beside the artist's home key; the
            # occasional outlier keeps the crate from being trivially mixable.
            track["camelot"] = (
                str(rng.choice(pool)) if rng.random() < 0.82
                else str(rng.choice(CAMELOT_CODES))
            )
            track["bpm"] = round(float(np.clip(rng.normal(centre, spread),
                                               centre - 3 * spread,
                                               centre + 3 * spread)), 1)
            track["energy"] = round(float(np.clip(rng.uniform(e_lo, e_hi)
                                                  + rng.normal(0, 0.05), 0.02, 0.99)), 3)


def is_harmonic(a: str, b: str) -> bool:
    """Same key, one step round the wheel, or the relative major/minor."""
    na, la = int(a[:-1]), a[-1]
    nb, lb = int(b[:-1]), b[-1]
    step = (nb - na) % 12
    if la == lb:
        return step in (0, 1, 11)
    return step == 0


def tempo_gap(a: float, b: float) -> float:
    """Fractional tempo change, allowing half- and double-time (87 <-> 174)."""
    return min(abs(c - a) / a for c in (b, b * 2.0, b / 2.0))


def _b62(rng: np.random.Generator) -> str:
    alphabet = np.array(list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    return "".join(rng.choice(alphabet, size=22))


def _iso_utc(local_end: datetime) -> str:
    return (local_end - HOME_OFFSET).strftime("%Y-%m-%dT%H:%M:%SZ")


def _maybe_bool(rng, value: bool, null_p: float):
    """Return value, or None with probability null_p (the export's `skipped`
    field is inconsistently populated — SPEC §4.3)."""
    return None if rng.random() < null_p else value


def build_artists(rng: np.random.Generator):
    combos = [(a, n) for a in ADJECTIVES for n in NOUNS]
    idx = rng.choice(len(combos), size=N_ARTISTS, replace=False)
    artists = []
    span_days = (END - START).days
    for rank, ci in enumerate(idx):
        a, n = combos[ci]
        name = f"{a} {n}"
        # Zipf-ish popularity so a few artists dominate listening volume.
        pop = 1.0 / (rank + 1.5) ** 1.1
        # Debut skewed earlier so we get multi-year cohorts to retain.
        debut = START + timedelta(days=int((rng.random() ** 1.8) * (span_days - 25)))
        kind = rng.choice(["core", "fading", "oneoff"], p=[0.25, 0.50, 0.25])
        if kind == "core":
            end = END
        elif kind == "fading":
            end = min(END, debut + timedelta(days=int(rng.integers(60, 420))))
        else:
            end = min(END, debut + timedelta(days=int(rng.integers(10, 30))))
        n_tracks = int(rng.integers(4, 9))
        title_idx = rng.choice(len(TITLE_WORDS), size=n_tracks, replace=False)
        tracks = []
        for t in title_idx:
            tracks.append({
                "name": TITLE_WORDS[t],
                "album": f"{NOUNS[ci % len(NOUNS)]} ({rng.choice(ALBUM_WORDS)})",
                "uri": f"spotify:track:{_b62(rng)}",
                "dur_ms": int(np.clip(rng.normal(210_000, 55_000), 95_000, 360_000)),
            })
        artists.append({"name": name, "pop": pop, "debut": debut,
                        "end": end, "kind": kind, "tracks": tracks})
    return artists


def make_music_row(rng, local_end, artist, track, ms, r_start, r_end, shuffle):
    is_skip = ms < 30_000 and r_end == "fwdbtn"
    return {
        "ts": _iso_utc(local_end),
        "platform": str(rng.choice(PLATFORMS, p=PLATFORM_P)),
        "ms_played": int(ms),
        "conn_country": str(rng.choice(COUNTRIES, p=COUNTRY_P)),
        "master_metadata_track_name": track["name"],
        "master_metadata_album_artist_name": artist["name"],
        "master_metadata_album_album_name": track["album"],
        "spotify_track_uri": track["uri"],
        "episode_name": None,
        "episode_show_name": None,
        "spotify_episode_uri": None,
        "reason_start": r_start,
        "reason_end": r_end,
        "shuffle": bool(shuffle),
        # `skipped` is right more often than not, but frequently null.
        "skipped": _maybe_bool(rng, is_skip, null_p=0.30 if is_skip else 0.45),
        "offline": bool(rng.random() < 0.10),
    }


# How often an intentional (non-shuffle) listen picks something that actually
# mixes out of the previous track, rather than anything at all. This is what
# puts harmonic structure into the observed transition graph.
HARMONIC_INTENT_P = 0.55
# Candidates considered when making such a pick.
CANDIDATE_POOL = 12
# Extra skip probability when a transition clashes / jumps tempo. These two
# numbers ARE the effect the §8.5 hypothesis test recovers.
CLASH_SKIP_PENALTY = 0.10
TEMPO_SKIP_PENALTY = 0.08


def _draw(rng, active, weights):
    artist = active[int(rng.choice(len(active), p=weights))]
    return artist, artist["tracks"][int(rng.integers(len(artist["tracks"])))]


def _pick_next(rng, active, weights, prev, shuffle):
    """Choose the next (artist, track).

    On shuffle the picker is blind — that is the point of shuffle. Otherwise it
    usually reaches for something that mixes: sample a handful of candidates and
    prefer one that is both harmonically compatible and tempo-reachable.
    """
    if shuffle or prev is None or rng.random() >= HARMONIC_INTENT_P:
        return _draw(rng, active, weights)

    _, prev_track = prev
    fallback = None
    for _ in range(CANDIDATE_POOL):
        artist, track = _draw(rng, active, weights)
        if track["camelot"] == prev_track["camelot"] and track is prev_track:
            continue
        if is_harmonic(prev_track["camelot"], track["camelot"]):
            if tempo_gap(prev_track["bpm"], track["bpm"]) <= 0.06:
                return artist, track
            fallback = fallback or (artist, track)
    return fallback or _draw(rng, active, weights)


def generate_session(rng, day: date, active, weights, events):
    weekend = day.weekday() >= 5
    hw = HOUR_W_WEEKEND if weekend else HOUR_W_WEEKDAY
    hour = int(rng.choice(24, p=hw / hw.sum()))
    cur = datetime(day.year, day.month, day.day, hour,
                   int(rng.integers(0, 60)), int(rng.integers(0, 60)))
    shuffle = bool(rng.random() < 0.45)
    base_skip_p = 0.35 if shuffle else 0.12     # the effect the §7.3 test finds
    length = int(min(20, 2 + rng.geometric(0.25)))
    prev_skipped = False
    prev = None
    for t in range(length):
        # Occasionally binge: replay the previous track instead of a new pick.
        if prev is not None and rng.random() < 0.08:
            artist, track = prev
        else:
            artist, track = _pick_next(rng, active, weights, prev, shuffle)

        # A rough transition is likelier to get skipped past, whether or not the
        # listener could name why. This is what the harmonic test measures.
        skip_p = base_skip_p
        if prev is not None and track is not prev[1]:
            _, prev_track = prev
            if not is_harmonic(prev_track["camelot"], track["camelot"]):
                skip_p += CLASH_SKIP_PENALTY
            if tempo_gap(prev_track["bpm"], track["bpm"]) > 0.06:
                skip_p += TEMPO_SKIP_PENALTY

        skip = rng.random() < skip_p
        if skip:
            ms = int(rng.integers(3_000, 29_000))
            r_end = "fwdbtn"
        elif rng.random() < 0.06:
            ms = int(rng.integers(30_000, track["dur_ms"]))      # paused / moved on
            r_end = "endplay"
        else:
            ms = int(track["dur_ms"] * rng.uniform(0.92, 1.0))   # played through
            r_end = "trackdone"
        r_start = "clickrow" if t == 0 else ("fwdbtn" if prev_skipped else "trackdone")

        end = cur + timedelta(milliseconds=ms)
        events.append(make_music_row(rng, end, artist, track, ms, r_start, r_end, shuffle))
        cur = end + timedelta(seconds=int(rng.integers(0, 20)))
        prev_skipped = skip
        prev = (artist, track)


def generate_podcasts(rng, day: date, events):
    show, eps = PODCASTS[int(rng.integers(len(PODCASTS)))]
    cur = datetime(day.year, day.month, day.day,
                   int(rng.integers(7, 22)), int(rng.integers(0, 60)))
    for _ in range(int(rng.integers(1, 4))):
        ms = int(rng.integers(300_000, 3_600_000))
        end = cur + timedelta(milliseconds=ms)
        events.append({
            "ts": _iso_utc(end),
            "platform": str(rng.choice(PLATFORMS, p=PLATFORM_P)),
            "ms_played": ms,
            "conn_country": str(rng.choice(COUNTRIES, p=COUNTRY_P)),
            "master_metadata_track_name": None,
            "master_metadata_album_artist_name": None,
            "master_metadata_album_album_name": None,
            "spotify_track_uri": None,
            "episode_name": str(rng.choice(eps)),
            "episode_show_name": show,
            "spotify_episode_uri": f"spotify:episode:{_b62(rng)}",
            "reason_start": "clickrow",
            "reason_end": "trackdone" if rng.random() < 0.7 else "endplay",
            "shuffle": False,
            "skipped": None,
            "offline": bool(rng.random() < 0.15),
        })
        cur = end + timedelta(seconds=int(rng.integers(0, 30)))


def inject_edge_rows(rng, artists, events):
    """A handful of sub-second / junk plays that the `is_play` rule must drop."""
    for _ in range(8):
        day = START + timedelta(days=int(rng.integers(0, (END - START).days)))
        artist = artists[int(rng.integers(len(artists)))]
        track = artist["tracks"][int(rng.integers(len(artist["tracks"])))]
        end = datetime(day.year, day.month, day.day,
                       int(rng.integers(0, 24)), int(rng.integers(0, 60)))
        events.append(make_music_row(
            rng, end, artist, track, int(rng.integers(100, 900)),
            "fwdbtn", "fwdbtn", bool(rng.random() < 0.5)))


def write_crate_features(artists, out_dir: Path) -> Path:
    """Write the Mixed In Key / Rekordbox-shaped side-car (SPEC §8.3).

    Deliberately *not* a perfect mirror of the listening history: a real DJ's
    analysed library never covers everything they have streamed, so a slice of
    tracks is withheld. The enrichment layer has to cope with partial coverage,
    and the crate metrics have to report it honestly.
    """
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for artist in artists:
        for track in artist["tracks"]:
            if rng.random() < 0.08:        # never got analysed - stays untagged
                continue
            rows.append({
                "Artist": artist["name"],
                "Title": track["name"],
                "Key": track["camelot"],
                "BPM": f"{track['bpm']:.1f}",
                # Mixed In Key reports energy as an integer 1-10, not a fraction.
                "Energy": str(int(round(track["energy"] * 9)) + 1),
                "Duration": f"{track['dur_ms'] // 60000}:{(track['dur_ms'] // 1000) % 60:02d}",
                "Genre": artist["family"],
            })
    rows.sort(key=lambda r: (r["Artist"], r["Title"]))

    path = out_dir / "crate_features.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Artist", "Title", "Key", "BPM", "Energy", "Duration", "Genre"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    rng = np.random.default_rng(SEED)
    artists = build_artists(rng)
    assign_features(rng, artists)
    events: list[dict] = []

    day = START
    while day <= END:
        weekend = day.weekday() >= 5
        n_sessions = int(rng.poisson(1.0 + (0.7 if weekend else 0.0)))
        active = [a for a in artists if a["debut"] <= day <= a["end"]]
        if active and n_sessions:
            w = np.array([a["pop"] for a in active])
            w = w / w.sum()
            for _ in range(n_sessions):
                generate_session(rng, day, active, w, events)
        if rng.random() < 0.08:
            generate_podcasts(rng, day, events)
        day += timedelta(days=1)

    inject_edge_rows(rng, artists, events)
    events.sort(key=lambda r: r["ts"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mid = len(events) // 2
    chunks = {
        "Streaming_History_Audio_2021-2022_0.json": events[:mid],
        "Streaming_History_Audio_2022-2024_1.json": events[mid:],
    }
    for fname, rows in chunks.items():
        # newline="\n" forces LF on every platform so regenerating on Windows
        # doesn't rewrite the committed sample with CRLF.
        with open(OUT_DIR / fname, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rows, indent=1))

    features_path = write_crate_features(artists, OUT_DIR)

    music = sum(1 for e in events if e["episode_name"] is None)
    n_tracks = sum(len(a["tracks"]) for a in artists)
    with open(features_path, encoding="utf-8") as f:
        n_features = sum(1 for _ in f) - 1
    print(f"Wrote {len(events):,} events ({music:,} music, "
          f"{len(events) - music:,} podcast) across {len(chunks)} files -> {OUT_DIR}")
    print(f"Wrote {n_features:,}/{n_tracks:,} analysed tracks "
          f"({n_features / n_tracks:.0%} crate coverage) -> {features_path.name}")


if __name__ == "__main__":
    main()
