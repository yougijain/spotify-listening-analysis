"""Precompute every number the static dashboard needs, as JSON.

Hosting decision (SPEC §18)
---------------------------
The dashboard is a static page. There is no API, no database behind it and no
server-side Python, because nothing about this project needs one: the analysis
is a batch job over a fixed export, so its output can be computed once at build
time and shipped as files. That buys a free host, no cold starts, no secrets in
an environment, and a site that cannot break at 2am because a container died.

The one thing that genuinely has to be interactive is the set builder — a
dashboard of pre-rendered setlists is a screenshot, not a tool. So the browser
runs the beam search itself, over a graph exported here.

What the split looks like
-------------------------
Transition scoring stays in Python, where it is tested. Every component except
the energy-arc term is position-independent, so it is precomputed per edge as a
single number (:func:`~src.setbuilder.base_transition_score`). The browser adds
only the arc term — one line of arithmetic against a target curve — and walks
the graph. The domain model does not exist twice.

Edges are capped at ``TOP_K`` per track, which is what keeps the payload small
enough to be a static asset. A beam search never reaches past the top handful of
candidates from any node anyway, so the cap costs nothing in practice.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from .harmonic import ARCS
from .pipeline import (
    build,
    connect,
    harmonic_hypothesis_test,
    hypothesis_test,
    tempo_hypothesis_test,
)
from .setbuilder import (
    W_CONTINUITY,
    W_ENERGY,
    W_HARMONIC,
    W_PROVEN,
    W_QUALITY,
    W_TEMPO,
    Constraints,
    arc_feasibility,
    base_transition_score,
    build_set,
    evaluate,
    load_crate,
    load_proven_edges,
)

ROOT = Path(__file__).resolve().parents[1]
WEB_DATA = ROOT / "web" / "data"

# Candidates kept per track in the exported graph.
TOP_K = 24
# Arcs the dashboard offers.
ARCS_EXPORTED = ("warmup", "peak", "journey", "closing")


def _round(value, places: int = 4):
    return None if value is None else round(float(value), places)


def build_graph(tracks, proven_edges: dict, top_k: int = TOP_K) -> dict:
    """Top-``top_k`` outgoing candidates per track, scored and pre-sorted.

    Each edge is ``[to_index, base_score, proven_contribution]``. The proven
    term is carried separately rather than only folded into the base, so the
    dashboard's "favour mixes I've already played" toggle can actually subtract
    it. A control that silently does nothing is worse than no control.

    Hard constraints that depend only on the pair — the tempo cap and the
    same-track rule — are applied here, so the browser never sees an edge it
    would have to reject. Artist spacing depends on the whole path, so it stays
    a runtime check in the client.
    """
    constraints = Constraints()
    index = {t.track_key: i for i, t in enumerate(tracks)}

    edges = []
    for src in tracks:
        scored = []
        for dst in tracks:
            if src.track_key == dst.track_key:
                continue
            if not constraints.allows([src], dst):
                continue
            proven = proven_edges.get((src.track_key, dst.track_key), 0.0)
            scored.append((base_transition_score(src, dst, proven),
                           index[dst.track_key], W_PROVEN * proven))
        scored.sort(key=lambda row: row[0], reverse=True)
        edges.append([[i, round(s, 4), round(p, 4)]
                      for s, i, p in scored[:top_k]])
    return {"top_k": top_k, "edges": edges}


def crate_payload(tracks) -> list:
    return [
        {
            "i": i,
            "key": t.track_key,
            "title": t.track_name,
            "artist": t.artist_name,
            "bpm": _round(t.bpm, 1),
            "camelot": t.camelot,
            "energy": _round(t.energy, 3),
            "readiness": _round(t.set_readiness, 3),
            "hold": _round(t.hold_lcb, 3),
            "burn": _round(t.rotation_burn, 3),
            "status": t.crate_status,
            "minutes": _round(t.minutes, 2),
        }
        for i, t in enumerate(tracks)
    ]


def summary_payload(con: duckdb.DuckDBPyConnection, tracks, proven_edges) -> dict:
    """Headline numbers, the two hypothesis tests, and the set benchmarks."""
    h = hypothesis_test(con)
    hh = harmonic_hypothesis_test(con)
    ht = tempo_hypothesis_test(con)

    n_plays, n_sessions, n_artists, lo, hi, hours = con.execute("""
        SELECT (SELECT count(*) FROM plays),
               (SELECT count(*) FROM sessions),
               (SELECT count(DISTINCT artist_name) FROM plays),
               (SELECT min(date_local) FROM plays),
               (SELECT max(date_local) FROM plays),
               (SELECT sum(ms_played)/3600000.0 FROM plays WHERE is_play)
    """).fetchone()
    n_transitions, n_edges = con.execute("""
        SELECT (SELECT count(*) FROM transitions), (SELECT count(*) FROM transition_edges)
    """).fetchone()

    arcs = {}
    for arc in ARCS_EXPORTED:
        plan = build_set(tracks, 60.0, arc, proven_edges=proven_edges)
        feas = arc_feasibility(tracks, arc, 60.0)
        ev = evaluate(tracks, 60.0, arc, proven_edges=proven_edges, random_trials=100)
        arcs[arc] = {
            "supply": _round(feas.mean_supply, 3),
            "verdict": feas.verdict,
            "stranded": feas.stranded,
            "harmonic_rate": _round(plan.harmonic_rate, 3),
            "arc_miss": _round(plan.arc_deviation(arc), 3),
            "beam": _round(ev.beam, 3),
            "greedy": _round(ev.greedy, 3),
            "random": _round(ev.random_mean, 3),
            "random_best": _round(ev.random_best, 3),
            "random_harmonic_rate": _round(ev.random_harmonic_rate, 3),
        }

    return {
        "dataset": {
            "plays": n_plays, "sessions": n_sessions, "artists": n_artists,
            "from": str(lo), "to": str(hi), "hours": _round(hours, 0),
            "tracks": len(tracks), "transitions": n_transitions, "edges": n_edges,
        },
        "shuffle_test": {
            "shuffle_rate": _round(h["shuffle_rate"], 4),
            "intentional_rate": _round(h["intentional_rate"], 4),
            "z": _round(h["z"], 2), "p": h["p_one_sided"],
            "cohens_h": _round(h["cohens_h"], 3),
        },
        "harmonic_test": _test_payload(hh),
        "tempo_test": _test_payload(ht),
        "arcs": arcs,
    }


def _test_payload(result) -> dict:
    return {
        "odds_ratio": _round(result.odds_ratio, 3),
        "ci": [_round(result.or_low, 3), _round(result.or_high, 3)],
        "crude_odds_ratio": _round(result.crude_odds_ratio, 3),
        "confounding_pct": _round(result.confounding_pct, 1),
        "statistic": _round(result.statistic, 2),
        "p": result.p_value,
        "strata": [
            {
                "label": s.label,
                "exposed_rate": _round(s.rate_exposed, 4),
                "control_rate": _round(s.rate_control, 4),
                "n_exposed": s.n_exposed, "n_control": s.n_control,
                "odds_ratio": _round(s.odds_ratio, 3),
            }
            for s in result.strata
        ],
    }


def charts_payload(con: duckdb.DuckDBPyConnection) -> dict:
    """Tabular inputs for the dashboard's own SVG charts."""
    wheel = con.execute("""
        SELECT camelot_number, camelot_letter, n_tracks, avg_energy, avg_readiness
        FROM crate_key_coverage ORDER BY camelot_number, camelot_letter
    """).fetchall()
    moves = con.execute("""
        SELECT move, n, skip_rate, hold_lcb FROM transition_move_performance
        ORDER BY skip_rate, move
    """).fetchall()
    # Explicit, total ordering on every query below. A bare GROUP BY leaves
    # DuckDB free to return rows in whatever order the aggregation produced,
    # which differs between runs — and these files are committed, so an
    # unstable order shows up as phantom diffs and a flaky CI sync check.
    bands = con.execute("""
        SELECT band, n_tracks, avg_energy, avg_readiness
        FROM crate_tempo_bands ORDER BY band
    """).fetchall()
    status = con.execute("""
        SELECT crate_status, count(*) AS n FROM crate
        GROUP BY 1 ORDER BY n DESC, crate_status
    """).fetchall()
    return {
        "wheel": [{"number": int(n), "letter": letter, "tracks": int(c),
                   "energy": _round(e, 3), "readiness": _round(r, 3)}
                  for n, letter, c, e, r in wheel],
        "moves": [{"move": m, "n": int(n), "skip_rate": _round(s, 4),
                   "hold_lcb": _round(h, 4)} for m, n, s, h in moves],
        "bands": [{"band": b, "tracks": int(n), "energy": _round(e, 3),
                   "readiness": _round(r, 3)} for b, n, e, r in bands],
        "status": [{"status": s, "tracks": int(n)} for s, n in status],
    }


def config_payload() -> dict:
    """The objective's constants, so the client scores exactly as Python does."""
    return {
        "weights": {
            "harmonic": W_HARMONIC, "tempo": W_TEMPO, "energy": W_ENERGY,
            "continuity": W_CONTINUITY, "quality": W_QUALITY, "proven": W_PROVEN,
        },
        # Arc curves are sampled rather than reimplemented in JS: 101 points is
        # exact to three decimals under linear interpolation and, unlike a
        # transcribed formula, cannot drift from src/harmonic.py.
        "arcs": {
            arc: [round(float(ARCS[arc](i / 100.0)), 4) for i in range(101)]
            for arc in ARCS_EXPORTED
        },
        "arc_names": list(ARCS_EXPORTED),
    }


def export_all(
    con: duckdb.DuckDBPyConnection,
    out_dir: Path | None = None,
    top_k: int = TOP_K,
) -> list:
    """Write every payload. Returns the paths written."""
    out = Path(out_dir or WEB_DATA)
    out.mkdir(parents=True, exist_ok=True)

    tracks = load_crate(con)
    proven = load_proven_edges(con)

    payloads = {
        "summary.json": summary_payload(con, tracks, proven),
        "crate.json": crate_payload(tracks),
        "graph.json": build_graph(tracks, proven, top_k),
        "charts.json": charts_payload(con),
        "config.json": config_payload(),
    }

    written = []
    for name, payload in payloads.items():
        path = out / name
        # Compact separators and LF: these are assets, not documents.
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
            f.write("\n")
        written.append(path)
    return written


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Export the dashboard's data files.")
    ap.add_argument("--data", default="data/sample")
    ap.add_argument("--features", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    args = ap.parse_args()

    con = connect()
    build(con, args.data, features_file=args.features)
    for path in export_all(con, args.out, args.top_k):
        size = path.stat().st_size
        print(f"wrote {path.relative_to(ROOT)}  ({size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
