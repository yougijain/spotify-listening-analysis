"""Tests for the static-dashboard export.

The point of these is the contract between Python and the browser. The client
re-scores transitions while it searches, so if the exported graph and the
exported arc curves do not reproduce what src/setbuilder.py computes, the
dashboard silently disagrees with the report next to it. That is the failure
mode worth a test file.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from src.export_web import (
    ARCS_EXPORTED,
    build_graph,
    charts_payload,
    config_payload,
    crate_payload,
    export_all,
    summary_payload,
)
from src.harmonic import arc_target
from src.setbuilder import (
    W_ENERGY,
    W_PROVEN,
    Constraints,
    base_transition_score,
    load_crate,
    load_proven_edges,
    score_transition,
)
from tests.conftest import SAMPLE


@pytest.fixture(scope="module")
def crate(con):
    return load_crate(con)


@pytest.fixture(scope="module")
def proven(con):
    return load_proven_edges(con)


@pytest.fixture(scope="module")
def summary(con, crate, proven):
    """Computed once: it benchmarks four arcs against 100 random sets each."""
    return summary_payload(con, crate, proven)


# --- the scoring contract --------------------------------------------------

def test_base_plus_arc_term_reconstructs_the_full_score(crate, proven):
    """The split the client depends on: everything except the energy-arc term
    is position-independent, so base + W_ENERGY * fit must equal the real score.
    """
    from src.setbuilder import _energy_fit

    checked = 0
    for a in crate[:40]:
        for b in crate[:40]:
            if a.track_key == b.track_key:
                continue
            p = proven.get((a.track_key, b.track_key), 0.0)
            for arc in ARCS_EXPORTED:
                for position, total in ((0, 20), (7, 20), (19, 20)):
                    full = score_transition(a, b, position, total, arc, p).score
                    rebuilt = (base_transition_score(a, b, p)
                               + W_ENERGY * _energy_fit(b, arc, position, total))
                    assert full == pytest.approx(rebuilt, abs=1e-12)
                    checked += 1
    assert checked > 1000


def test_sampled_arc_curves_reproduce_arc_target(crate):
    """Arc curves are exported as 101 sampled points rather than a formula
    transcribed into JS. Linear interpolation between them must land within
    a thousandth of the real curve, or the client drifts from the report."""
    curves = config_payload()["arcs"]
    for arc in ARCS_EXPORTED:
        samples = curves[arc]
        assert len(samples) == 101
        for total in (8, 17, 23, 40):
            for position in range(total):
                t = position / (total - 1)
                lo = int(t * 100)
                hi = min(100, lo + 1)
                frac = t * 100 - lo
                interpolated = samples[lo] * (1 - frac) + samples[hi] * frac
                assert interpolated == pytest.approx(
                    arc_target(arc, position, total), abs=1e-3
                ), (arc, position, total)


def test_exported_weights_match_the_module(crate):
    from src import setbuilder

    w = config_payload()["weights"]
    assert w["harmonic"] == setbuilder.W_HARMONIC
    assert w["tempo"] == setbuilder.W_TEMPO
    assert w["energy"] == setbuilder.W_ENERGY
    assert w["continuity"] == setbuilder.W_CONTINUITY
    assert w["quality"] == setbuilder.W_QUALITY
    assert w["proven"] == setbuilder.W_PROVEN


# --- the graph -------------------------------------------------------------

def test_graph_has_one_adjacency_list_per_track(crate, proven):
    g = build_graph(crate, proven, top_k=8)
    assert len(g["edges"]) == len(crate)
    assert g["top_k"] == 8


def test_graph_respects_the_cap(crate, proven):
    g = build_graph(crate, proven, top_k=6)
    assert all(len(row) <= 6 for row in g["edges"])


def test_graph_edges_are_sorted_best_first(crate, proven):
    g = build_graph(crate, proven, top_k=12)
    for row in g["edges"]:
        scores = [s for _, s, _ in row]
        assert scores == sorted(scores, reverse=True)


def test_graph_never_points_a_track_at_itself(crate, proven):
    g = build_graph(crate, proven, top_k=12)
    for i, row in enumerate(g["edges"]):
        assert all(j != i for j, _, _ in row)


def test_graph_edge_scores_match_python(crate, proven):
    """Every exported number must be the one Python would compute."""
    g = build_graph(crate, proven, top_k=10)
    for i, row in enumerate(g["edges"][:60]):
        src = crate[i]
        for j, score, proven_term in row:
            dst = crate[j]
            p = proven.get((src.track_key, dst.track_key), 0.0)
            assert score == pytest.approx(base_transition_score(src, dst, p), abs=1e-4)
            # The proven term ships separately so the client can subtract it.
            assert proven_term == pytest.approx(W_PROVEN * p, abs=1e-4)


def test_graph_pre_applies_the_pair_level_constraints(crate, proven):
    """The client should never see an edge it would have to reject on tempo."""
    g = build_graph(crate, proven, top_k=24)
    c = Constraints()
    for i, row in enumerate(g["edges"][:80]):
        for j, _, _ in row:
            assert c.allows([crate[i]], crate[j])


def test_graph_indices_are_in_range(crate, proven):
    g = build_graph(crate, proven, top_k=12)
    for row in g["edges"]:
        assert all(0 <= j < len(crate) for j, _, _ in row)


# --- payload shapes --------------------------------------------------------

def test_crate_payload_is_indexed_consistently(crate):
    payload = crate_payload(crate)
    assert len(payload) == len(crate)
    for i, row in enumerate(payload):
        assert row["i"] == i
        assert row["key"] == crate[i].track_key
        assert row["title"] and row["artist"]


def test_summary_carries_both_odds_ratios(summary):
    h = summary["harmonic_test"]
    assert h["odds_ratio"] > 0
    assert h["crude_odds_ratio"] > 0
    # The whole reason the test is stratified: the two must differ.
    assert h["crude_odds_ratio"] != h["odds_ratio"]
    assert len(h["strata"]) == 2


def test_summary_covers_every_exported_arc(summary):
    arcs = summary["arcs"]
    assert set(arcs) == set(ARCS_EXPORTED)
    for row in arcs.values():
        assert 0 <= row["supply"] <= 1
        assert row["beam"] >= row["greedy"] - 1e-9
        assert row["beam"] > row["random"]


def test_charts_payload_is_complete(con):
    c = charts_payload(con)
    assert len(c["wheel"]) > 12
    assert {m["move"] for m in c["moves"]} >= {"same_key", "adjacent", "clash"}
    assert c["bands"] and c["status"]


# --- the written files -----------------------------------------------------

def test_export_writes_valid_json(con, tmp_path):
    paths = export_all(con, tmp_path, top_k=8)
    names = {p.name for p in paths}
    assert names == {"summary.json", "crate.json", "graph.json",
                     "charts.json", "config.json"}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload


def test_exported_payload_stays_small_enough_to_be_a_static_asset(con, tmp_path):
    total = sum(p.stat().st_size for p in export_all(con, tmp_path))
    assert total < 2_000_000, "payload is too large to ship as a static file"


def test_exported_files_use_lf_endings(con, tmp_path):
    for path in export_all(con, tmp_path, top_k=4):
        assert b"\r\n" not in path.read_bytes()


def test_every_edge_carries_its_proven_term(crate, proven):
    """The dashboard's 'favour mixes I have played' toggle subtracts this, so it
    has to be present on every edge — including the many where it is zero."""
    g = build_graph(crate, proven, top_k=10)
    for row in g["edges"]:
        for edge in row:
            assert len(edge) == 3
            assert 0.0 <= edge[2] <= W_PROVEN + 1e-9


def test_export_is_byte_stable_across_independent_builds(tmp_path):
    """These files are committed and CI diffs them, so an unstable row order
    shows up as a phantom diff and a flaky check.

    Deliberately rebuilds the warehouse from scratch each time rather than
    reusing one connection. The first version of this test shared a connection
    and passed while the export was in fact unstable: an unordered GROUP BY
    returns rows consistently within a build and differently between builds.
    """
    from src.pipeline import build, connect

    digests = []
    for run in range(3):
        con = connect()
        build(con, SAMPLE)
        paths = export_all(con, tmp_path / str(run), top_k=8)
        digests.append({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in paths})
        con.close()

    for name in digests[0]:
        assert digests[0][name] == digests[1][name] == digests[2][name], (
            f"{name} is not reproducible across rebuilds"
        )
