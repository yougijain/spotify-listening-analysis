"""Tests for the feature provider chain and the fuzzy history<->library join.

The join is the part most likely to silently rot, so most of these tests are
about matching behaviour: what should collapse together, and what must not.
"""

from __future__ import annotations

import duckdb
import pytest

from src.enrich import (
    CoverageReport,
    CsvFeatureProvider,
    SyntheticFeatureProvider,
    TrackFeatures,
    _normalize_energy,
    build_provider_chain,
    load_track_features,
    match_key,
    normalize,
    resolve,
)


def write_csv(tmp_path, text, name="crate_features.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- normalisation ---------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Ceilings", "ceilings"),
        ("  Ceilings  ", "ceilings"),
        ("CEILINGS", "ceilings"),
        ("Café", "cafe"),
        ("Ceilings - Radio Edit", "ceilings"),
        ("Ceilings (Radio Edit)", "ceilings"),
        ("Ceilings [2019 Remaster]", "ceilings"),
        ("Ceilings - Remastered", "ceilings"),
        ("Ceilings — Remastered 2011", "ceilings"),   # em dash
        ("Ceilings – Single Version", "ceilings"),    # en dash
        ("Ceilings (Original Mix)", "ceilings"),
        ("Ceilings (Deluxe Edition)", "ceilings"),
        ("Don't Look Back", "dont look back"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize("variant", ["Gravity (Live)", "Gravity (Acoustic)"])
def test_alternate_recordings_are_not_collapsed(variant):
    # A live cut has a different tempo and key than the studio version, so
    # matching them would attach the wrong BPM to real plays.
    assert normalize(variant) != normalize("Gravity")


def test_stacked_version_suffixes_all_come_off():
    assert normalize("Ceilings - Radio Edit (2019 Remaster)") == "ceilings"


def test_match_key_combines_artist_and_title():
    assert match_key("Amber Current", "Afterglow") == match_key(
        "amber  current", "Afterglow - Remastered"
    )
    assert match_key("A", "B") != match_key("B", "A")


# --- energy scales ---------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        (0.0, 0.0), (0.5, 0.5), (1.0, 1.0),      # Spotify-style 0..1
        (10, 1.0), (5.5, 0.5), ("7", 2 / 3),     # Mixed In Key-style 1..10
        (None, None), ("", None), ("loud", None), (-1, None), (11, None),
    ],
)
def test_energy_scale_detection(raw, expected):
    got = _normalize_energy(raw)
    assert got is None if expected is None else got == pytest.approx(expected)


# --- CSV provider ----------------------------------------------------------

BASIC_CSV = """Artist,Title,Key,BPM,Energy
Amber Current,Afterglow,10A,141.6,7
Velvet Foxes,Gravity,Am,124.0,5
Neon Atlas,Runaway,,128.5,
"""


def test_csv_provider_reads_features(tmp_path):
    p = CsvFeatureProvider(write_csv(tmp_path, BASIC_CSV))
    assert len(p) == 3
    f = p.lookup("Amber Current", "Afterglow")
    assert f.bpm == pytest.approx(141.6)
    assert f.key.code == "10A"
    assert f.energy == pytest.approx(2 / 3)


def test_csv_provider_accepts_musical_key_names(tmp_path):
    p = CsvFeatureProvider(write_csv(tmp_path, BASIC_CSV))
    assert p.lookup("Velvet Foxes", "Gravity").key.code == "8A"   # Am


def test_csv_provider_keeps_partial_rows(tmp_path):
    # Missing key and energy must not discard a perfectly good BPM.
    f = CsvFeatureProvider(write_csv(tmp_path, BASIC_CSV)).lookup("Neon Atlas", "Runaway")
    assert f.bpm == pytest.approx(128.5)
    assert f.key is None and f.energy is None


def test_csv_provider_matches_fuzzily(tmp_path):
    p = CsvFeatureProvider(write_csv(tmp_path, BASIC_CSV))
    assert p.lookup("amber current", "Afterglow - Remastered 2019") is not None


def test_csv_provider_misses_return_none(tmp_path):
    assert CsvFeatureProvider(write_csv(tmp_path, BASIC_CSV)).lookup("Nobody", "Nothing") is None


@pytest.mark.parametrize(
    "header",
    [
        "Artist,Title,Key,BPM,Energy",
        "artist name,track title,initial key,tempo,energy level",   # Rekordbox-ish
        "Artist,Track Title,Key result,BPM,Energy Result",          # Mixed In Key-ish
        "ALBUM ARTIST,SONG,CAMELOT,BPM,ENERGY",                     # shouty tag dump
    ],
)
def test_csv_provider_tolerates_tool_specific_headers(tmp_path, header):
    p = CsvFeatureProvider(write_csv(tmp_path, f"{header}\nAmber Current,Afterglow,10A,141.6,7\n"))
    assert p.lookup("Amber Current", "Afterglow").bpm == pytest.approx(141.6)


def test_csv_provider_handles_semicolons_and_tabs(tmp_path):
    p = CsvFeatureProvider(write_csv(tmp_path, "Artist;Title;Key;BPM\nA;B;9A;120\n"))
    assert p.lookup("A", "B").key.code == "9A"


def test_csv_provider_survives_a_bom(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_bytes("Artist,Title,BPM\nA,B,120\n".encode("utf-8-sig"))
    assert CsvFeatureProvider(path).lookup("A", "B").bpm == pytest.approx(120)


def test_csv_provider_skips_rows_without_artist_or_title(tmp_path):
    p = CsvFeatureProvider(write_csv(tmp_path, "Artist,Title,BPM\n,Orphan,120\nA,,130\nA,B,140\n"))
    assert len(p) == 1 and p.rows_skipped == 2


def test_csv_provider_keeps_the_first_of_duplicate_rows(tmp_path):
    p = CsvFeatureProvider(write_csv(tmp_path, "Artist,Title,BPM\nA,B,120\nA,B,999\n"))
    assert p.duplicates == 1
    assert p.lookup("A", "B").bpm == pytest.approx(120)


def test_csv_provider_tolerates_unparseable_values(tmp_path):
    p = CsvFeatureProvider(write_csv(tmp_path, "Artist,Title,Key,BPM\nA,B,ZZ9,fast\n"))
    # Nothing usable on the row, so it defers rather than reporting empties.
    assert p.lookup("A", "B") is None


def test_csv_provider_rejects_a_file_with_no_identifying_columns(tmp_path):
    with pytest.raises(ValueError, match="no column found"):
        CsvFeatureProvider(write_csv(tmp_path, "Foo,Bar\n1,2\n"))


def test_csv_provider_reports_a_missing_file():
    with pytest.raises(FileNotFoundError):
        CsvFeatureProvider("does/not/exist.csv")


# --- synthetic provider ----------------------------------------------------

def test_synthetic_is_deterministic_across_instances():
    a = SyntheticFeatureProvider().lookup("Amber Current", "Afterglow")
    b = SyntheticFeatureProvider().lookup("Amber Current", "Afterglow")
    assert a == b


def test_synthetic_matches_the_same_fuzzy_key():
    p = SyntheticFeatureProvider()
    assert p.lookup("Amber Current", "Afterglow") == p.lookup(
        "amber current", "Afterglow (Radio Edit)"
    )


def test_synthetic_always_answers_and_stays_in_range():
    p = SyntheticFeatureProvider()
    for i in range(300):
        f = p.lookup(f"Artist {i}", f"Track {i}")
        assert 60 <= f.bpm <= 200
        assert 0.0 <= f.energy <= 1.0
        assert 1 <= f.key.number <= 12 and f.key.letter in ("A", "B")
        assert f.source == "synthetic"


def test_synthetic_spreads_across_the_wheel():
    p = SyntheticFeatureProvider()
    codes = {p.lookup(f"A{i}", f"T{i}").key.code for i in range(400)}
    assert len(codes) >= 20      # not collapsing onto a handful of keys


# --- the chain -------------------------------------------------------------

def test_chain_prefers_the_csv_and_falls_back_to_synthetic(tmp_path):
    write_csv(tmp_path, BASIC_CSV)
    chain = build_provider_chain(tmp_path)
    assert resolve(chain, "Amber Current", "Afterglow").source == "crate_features"
    assert resolve(chain, "Unknown", "Track").source == "synthetic"


def test_chain_without_a_csv_still_works(tmp_path):
    assert len(build_provider_chain(tmp_path)) == 1


def test_chain_refuses_to_be_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_provider_chain(tmp_path, allow_synthetic=False)


def test_resolve_with_no_provider_answers_returns_empty():
    assert resolve([], "A", "B") == TrackFeatures(source="none")


def test_a_csv_row_with_only_nulls_defers_to_the_fallback(tmp_path):
    write_csv(tmp_path, "Artist,Title,Key,BPM,Energy\nA,B,,,\n")
    assert resolve(build_provider_chain(tmp_path), "A", "B").source == "synthetic"


# --- loading into DuckDB ---------------------------------------------------

def test_load_track_features_writes_one_row_per_track(tmp_path):
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE plays AS SELECT * FROM (VALUES
            ('uri:1', 'Afterglow', 'Amber Current'),
            ('uri:1', 'Afterglow', 'Amber Current'),
            ('uri:2', 'Gravity',   'Velvet Foxes')
        ) AS t(track_key, track_name, artist_name)
    """)
    write_csv(tmp_path, BASIC_CSV)
    report = load_track_features(con, build_provider_chain(tmp_path))

    assert report.total == 2
    rows = con.execute(
        "SELECT track_key, camelot, camelot_number, bpm FROM track_features ORDER BY 1"
    ).fetchall()
    assert rows[0] == ("uri:1", "10A", 10, pytest.approx(141.6))
    assert con.execute("SELECT count(*) FROM track_features").fetchone()[0] == 2


def test_coverage_report_separates_real_from_fallback():
    r = CoverageReport(total=10, by_source={"crate_features": 8, "synthetic": 2})
    assert r.real == 8
    assert r.real_pct == pytest.approx(0.8)
    assert "80%" in r.summary()


def test_coverage_report_handles_an_empty_dataset():
    assert CoverageReport(total=0, by_source={}).real_pct == 0.0
