"""Build a set from the command line.

    python -m src.setlist --arc peak --minutes 90
    python -m src.setlist --arc warmup --minutes 60 --explain
    python -m src.setlist --seed "spotify:track:..." --arc journey

This is the tool the rest of the project exists to feed. It rebuilds the
warehouse, loads the crate, and prints a sequenced set with the reasoning behind
each transition.
"""

from __future__ import annotations

import argparse
import contextlib
import sys

from .harmonic import ARCS
from .pipeline import build, connect
from .setbuilder import (
    Constraints,
    arc_feasibility,
    build_set,
    evaluate,
    load_crate,
    load_proven_edges,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sequence a set from the crate.")
    ap.add_argument("--data", default="data/sample", help="dir of *.json export files")
    ap.add_argument("--features", default=None, help="Mixed In Key / Rekordbox CSV")
    ap.add_argument("--arc", default="peak", choices=sorted(ARCS),
                    help="energy shape the set should follow")
    ap.add_argument("--minutes", type=float, default=60.0, help="target set length")
    ap.add_argument("--seed", default=None, help="track_key to open on")
    ap.add_argument("--artist-gap", type=int, default=3,
                    help="minimum slots between two tracks by the same artist")
    ap.add_argument("--max-drift", type=float, default=0.08,
                    help="largest fractional tempo jump allowed per transition")
    ap.add_argument("--include-burned", action="store_true",
                    help="allow over-rotated tracks back into the set")
    ap.add_argument("--min-plays", type=int, default=1,
                    help="drop tracks with fewer plays than this from the crate")
    ap.add_argument("--explain", action="store_true",
                    help="also print feasibility and the baseline comparison")
    args = ap.parse_args()

    # Console output carries a few non-ASCII glyphs; make stdout UTF-8 so it
    # prints on a Windows cp1252 terminal too.
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")

    con = connect()
    build(con, args.data, features_file=args.features)
    crate = load_crate(con, min_plays=args.min_plays)
    edges = load_proven_edges(con)

    if not crate:
        print("Crate is empty — nothing to sequence.", file=sys.stderr)
        return 1

    constraints = Constraints(
        artist_gap=args.artist_gap,
        max_drift=args.max_drift,
        exclude_status=frozenset() if args.include_burned else frozenset({"burned"}),
    )

    try:
        plan = build_set(crate, args.minutes, args.arc, seed_key=args.seed,
                         constraints=constraints, proven_edges=edges)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(plan.describe())

    if args.explain:
        feas = arc_feasibility(crate, args.arc, args.minutes,
                               max_drift=args.max_drift)
        ev = evaluate(crate, args.minutes, args.arc, proven_edges=edges,
                      random_trials=100, constraints=constraints)
        print()
        print(f"crate      {len(crate):,} tracks, {feas.summary()}")
        print(f"arc miss   {plan.arc_deviation(args.arc):.3f} mean energy gap "
              f"from target, biggest step {plan.max_energy_step():.2f}")
        print(f"baselines  {ev.summary()}")
        if feas.worst_supply < 0.5:
            print(f"warning    the crate is thin for a '{args.arc}' arc; the set "
                  f"above is the best available, not a good one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
