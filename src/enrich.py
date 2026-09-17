"""Attach musical features (BPM, key, energy) to the tracks in the history.

Why this module exists
----------------------
The obvious design is "call Spotify's ``/v1/audio-features`` for every track
URI". That door closed: in November 2024 Spotify restricted audio-features and
audio-analysis to applications that already had access, so a project started
after that cannot get tempo or key from the API at all (SPEC §8.3).

Rather than pretend otherwise, features come from a **provider chain**. Each
provider is asked for a track in turn and the first hit wins:

  1. :class:`CsvFeatureProvider` — a Mixed In Key / Rekordbox / Traktor export.
     This is the real path. A working DJ has already analysed their library, so
     the authoritative BPM and key are sitting on their own disk in a CSV.
  2. :class:`SyntheticFeatureProvider` — deterministic stand-in features derived
     from the track's own name. Never confused for real data (it is labelled as
     its own source and reported separately), but it means the repo runs
     end-to-end for someone who has no analysed library at all.

Adding a third provider — a local audio analyser, MusicBrainz, a scraped tag
dump — means implementing one method. Nothing downstream changes.

The join is the hard part
-------------------------
A feature export has ``Artist`` and ``Title`` strings; the history has those
plus a Spotify URI. There is no shared identifier, so matching is fuzzy by
necessity: ``"Ceilings - Radio Edit"`` in one file is ``"Ceilings (Radio
Edit)"`` in the other and ``"ceilings"`` in a third. :func:`match_key` does the
normalisation, and coverage is always reported rather than assumed — an
unmatched track is a visible number, not a silent NULL.
"""

from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

import duckdb

from .harmonic import CamelotKey, parse_key

# Suffixes that describe a *version* rather than a different song. Stripping
# them is what lets an analysed "Ceilings" match a streamed "Ceilings -
# Remastered 2019". Deliberately conservative: "Live" and "Acoustic" are NOT in
# here, because those really are different recordings with different tempos.
_VERSION_NOISE = re.compile(
    r"""\s*
    (?:[-–—]\s*|\(|\[)
    (?:
        remaster(?:ed)?(?:\s*\d{4})?
      | \d{4}\s*remaster(?:ed)?
      | radio\s*edit
      | single\s*version
      | album\s*version
      | original\s*mix
      | bonus\s*track
      | explicit | clean
      | deluxe(?:\s*edition)?
      | mono | stereo
    )
    \s*(?:\)|\])?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Apostrophes are DELETED rather than spaced out, so "Don't" and "Dont" match.
# Every other punctuation mark becomes a space, so "A/B" and "A B" match too.
_APOSTROPHE = re.compile(r"['\u2018\u2019\u02bc]")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize(text: Optional[str]) -> str:
    """Fold a title or artist name down to something joinable.

    Lowercases, strips accents, drops punctuation, collapses whitespace, and
    removes version suffixes. ``"Café — Remastered 2011"`` and ``"cafe"`` land
    on the same string, as do ``"Don't"`` and ``"Dont"``.
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Strip version noise before punctuation, while the brackets still exist.
    prev = None
    while prev != s:
        prev = s
        s = _VERSION_NOISE.sub("", s)
    s = _PUNCT.sub(" ", _APOSTROPHE.sub("", s.lower()))
    return _SPACE.sub(" ", s).strip()


def match_key(artist: Optional[str], title: Optional[str]) -> str:
    """The join key between a feature file and the listening history."""
    return f"{normalize(artist)}␟{normalize(title)}"


@dataclass(frozen=True)
class TrackFeatures:
    """Musical features for one track, plus where they came from."""

    bpm: Optional[float] = None
    key: Optional[CamelotKey] = None
    energy: Optional[float] = None
    source: str = "unknown"

    @property
    def is_empty(self) -> bool:
        return self.bpm is None and self.key is None and self.energy is None


class FeatureProvider(Protocol):
    """Anything that can answer "what are this track's features?"."""

    name: str

    def lookup(self, artist: str, title: str) -> Optional[TrackFeatures]:
        """Return features, or ``None`` to defer to the next provider."""


def _to_float(value) -> Optional[float]:
    try:
        f = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return f if f == f else None          # reject NaN


def _normalize_energy(value) -> Optional[float]:
    """Accept energy as 0..1 (Spotify-style) or 1..10 (Mixed In Key-style).

    The two scales overlap only at 1.0, which is read as the top of the 0..1
    scale — the reading that is wrong less often, since a whole library of
    Mixed In Key "1" energy tracks is not a thing anyone has.
    """
    f = _to_float(value)
    if f is None or f < 0:
        return None
    if f <= 1.0:
        return f
    if f <= 10.0:
        return (f - 1.0) / 9.0
    return None


# Column names seen in the wild, lowercased. First match wins.
_CSV_FIELDS = {
    "artist": ("artist", "artist name", "album artist", "albumartist"),
    "title": ("title", "track title", "track", "name", "song", "track name"),
    "bpm": ("bpm", "tempo"),
    "key": ("key", "key result", "initial key", "camelot", "tone"),
    "energy": ("energy", "energy level", "energy result"),
}


class CsvFeatureProvider:
    """Features from a DJ-software export (Mixed In Key, Rekordbox, Traktor).

    Column names vary by tool, so headers are matched against a list of known
    aliases rather than a fixed schema. A row missing artist or title is
    skipped; a row with an unparseable key or BPM keeps whatever else it has.
    """

    def __init__(self, path: str | Path, name: str = "csv"):
        self.name = name
        self.path = Path(path)
        self._by_key: dict[str, TrackFeatures] = {}
        self.rows_read = 0
        self.rows_skipped = 0
        self.duplicates = 0
        self._load()

    def _resolve_headers(self, fieldnames: Sequence[str]) -> dict[str, str]:
        lowered = {(f or "").strip().lower(): f for f in fieldnames}
        resolved = {}
        for target, aliases in _CSV_FIELDS.items():
            for alias in aliases:
                if alias in lowered:
                    resolved[target] = lowered[alias]
                    break
        return resolved

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"feature file not found: {self.path}")

        with open(self.path, encoding="utf-8-sig", newline="") as f:
            sample = f.read(8192)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel                      # fall back to plain CSV
            reader = csv.DictReader(f, dialect=dialect)
            cols = self._resolve_headers(reader.fieldnames or [])
            missing = {"artist", "title"} - cols.keys()
            if missing:
                raise ValueError(
                    f"{self.path.name}: no column found for {sorted(missing)}; "
                    f"saw headers {reader.fieldnames}"
                )

            for row in reader:
                self.rows_read += 1
                artist, title = row.get(cols["artist"]), row.get(cols["title"])
                if not (artist or "").strip() or not (title or "").strip():
                    self.rows_skipped += 1
                    continue
                key = match_key(artist, title)
                if key in self._by_key:
                    self.duplicates += 1                 # first analysis wins
                    continue
                self._by_key[key] = TrackFeatures(
                    bpm=_to_float(row.get(cols.get("bpm", ""), None)),
                    key=parse_key(row.get(cols.get("key", ""), None)),
                    energy=_normalize_energy(row.get(cols.get("energy", ""), None)),
                    source=self.name,
                )

    def __len__(self) -> int:
        return len(self._by_key)

    def lookup(self, artist: str, title: str) -> Optional[TrackFeatures]:
        found = self._by_key.get(match_key(artist, title))
        # An all-NULL row is no better than a miss; let the next provider try.
        return None if (found is None or found.is_empty) else found


# Tempo families the synthetic provider draws from, matching the shape of a
# real crate rather than a flat distribution across every possible BPM.
_SYNTH_FAMILIES = [
    (92, 6, 0.20, 0.45),
    (124, 4, 0.55, 0.80),
    (138, 5, 0.70, 0.95),
    (174, 4, 0.75, 0.98),
]


class SyntheticFeatureProvider:
    """Deterministic stand-in features, derived from the track's own name.

    This is a *fallback*, not a model: it makes nothing up about the actual
    recording and must never be read as a measurement. It exists so the pipeline
    has a defined behaviour for tracks no feature file covers, and so the repo
    demos end-to-end for someone with no analysed library. Its output is tagged
    ``synthetic`` throughout and counted separately in the coverage report.

    Determinism comes from BLAKE2b over the match key, so the same track always
    gets the same features across runs, machines and Python versions (unlike
    ``hash()``, which is salted per process).
    """

    name = "synthetic"

    def __init__(self, name: str = "synthetic"):
        self.name = name

    @staticmethod
    def _digest(artist: str, title: str) -> bytes:
        return hashlib.blake2b(match_key(artist, title).encode("utf-8"),
                               digest_size=8).digest()

    def lookup(self, artist: str, title: str) -> Optional[TrackFeatures]:
        d = self._digest(artist, title)
        family = _SYNTH_FAMILIES[d[0] % len(_SYNTH_FAMILIES)]
        centre, spread, e_lo, e_hi = family
        # Map bytes onto the ranges; d[1]/255 and d[2]/255 are uniform in 0..1.
        bpm = round(centre + (d[1] / 255.0 - 0.5) * 2 * spread, 1)
        number = (d[2] % 12) + 1
        letter = "A" if d[3] % 2 == 0 else "B"
        energy = round(e_lo + (d[4] / 255.0) * (e_hi - e_lo), 3)
        return TrackFeatures(bpm=bpm, key=CamelotKey(number, letter),
                             energy=energy, source=self.name)


@dataclass
class CoverageReport:
    """Where each track's features came from, for the QA section (SPEC §12)."""

    total: int
    by_source: dict[str, int]

    @property
    def real(self) -> int:
        """Tracks covered by a genuine feature file, not the fallback."""
        return sum(n for s, n in self.by_source.items() if s != "synthetic")

    @property
    def real_pct(self) -> float:
        return self.real / self.total if self.total else 0.0

    def summary(self) -> str:
        parts = ", ".join(f"{s} {n:,}" for s, n in sorted(self.by_source.items()))
        return (f"{self.total:,} tracks | {self.real_pct:.0%} from analysed "
                f"library ({parts})")


def build_provider_chain(
    data_dir: str | Path,
    features_file: Optional[str] = None,
    allow_synthetic: bool = True,
) -> list[FeatureProvider]:
    """Assemble the providers for a run: the feature CSV first, fallback last.

    Looks for ``features_file`` if given, otherwise any ``*features*.csv`` in
    ``data_dir``. Missing files are not an error when the fallback is allowed —
    the coverage report will make the absence obvious.
    """
    chain: list[FeatureProvider] = []
    if features_file:
        candidates = [Path(features_file)]
    else:
        candidates = sorted(Path(data_dir).glob("*features*.csv"))
    for path in candidates:
        if path.exists():
            chain.append(CsvFeatureProvider(path, name=path.stem))
    if allow_synthetic:
        chain.append(SyntheticFeatureProvider())
    if not chain:
        raise FileNotFoundError(
            f"no feature file found in {data_dir!r} and synthetic fallback disabled"
        )
    return chain


def resolve(chain: Sequence[FeatureProvider], artist: str, title: str) -> TrackFeatures:
    """First provider with an answer wins; an all-miss returns empty features."""
    for provider in chain:
        found = provider.lookup(artist, title)
        if found is not None and not found.is_empty:
            return found
    return TrackFeatures(source="none")


def load_track_features(
    con: duckdb.DuckDBPyConnection,
    chain: Sequence[FeatureProvider],
) -> CoverageReport:
    """Materialise `track_features`, one row per distinct track in `plays`.

    Reads the track list out of DuckDB, resolves each through the chain in
    Python (the matching logic is too fuzzy for SQL), and writes the result
    back as a table the SQL models join against.
    """
    tracks = con.execute("""
        SELECT track_key,
               any_value(track_name)  AS track_name,
               any_value(artist_name) AS artist_name
        FROM plays
        GROUP BY track_key
    """).fetchall()

    rows, by_source = [], {}
    for track_key, track_name, artist_name in tracks:
        feat = resolve(chain, artist_name, track_name)
        by_source[feat.source] = by_source.get(feat.source, 0) + 1
        rows.append((
            track_key, track_name, artist_name,
            feat.bpm,
            feat.key.code if feat.key else None,
            feat.key.number if feat.key else None,
            feat.key.letter if feat.key else None,
            feat.energy,
            feat.source,
        ))

    con.execute("""
        CREATE OR REPLACE TABLE track_features (
            track_key      VARCHAR,
            track_name     VARCHAR,
            artist_name    VARCHAR,
            bpm            DOUBLE,
            camelot        VARCHAR,
            camelot_number INTEGER,
            camelot_letter VARCHAR,
            energy         DOUBLE,
            feature_source VARCHAR
        )
    """)
    if rows:
        con.executemany(
            "INSERT INTO track_features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
    return CoverageReport(total=len(rows), by_source=by_source)
