"""End-to-end runner: build the models, validate them (SPEC §12), render the
figures, and write REPORT.md.

    python -m src.run_analysis                       # defaults: data/sample, IST, 30-min
    python -m src.run_analysis --data data/raw       # swap in the real export (Phase 7)
    python -m src.run_analysis --no-figures          # numbers only

REPORT.md is regenerated from whatever data was loaded, so it always matches the
current dataset and knob settings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import charts
from .pipeline import (
    build,
    connect,
    harmonic_hypothesis_test,
    hypothesis_test,
    session_gap_sensitivity,
    tempo_hypothesis_test,
)
from .setbuilder import (
    arc_feasibility,
    build_set,
    evaluate,
    load_crate,
    load_proven_edges,
)

ROOT = Path(__file__).resolve().parents[1]

# Arcs the report builds and benchmarks a set for.
ARCS_REPORTED = ("warmup", "peak", "journey", "closing")


def _scalar(con, sql):
    return con.execute(sql).fetchone()


def collect(con) -> dict:
    m = {}
    m["raw_total"] = _scalar(con, "SELECT count(*) FROM raw_streams")[0]
    m["raw_podcast"] = _scalar(con, """
        SELECT count(*) FROM raw_streams
        WHERE episode_name IS NOT NULL OR episode_show_name IS NOT NULL""")[0]
    m["raw_nullname"] = _scalar(con, """
        SELECT count(*) FROM raw_streams
        WHERE (episode_name IS NULL AND episode_show_name IS NULL)
          AND (track_name IS NULL OR artist_name IS NULL)""")[0]
    m["n_plays"] = _scalar(con, "SELECT count(*) FROM plays")[0]
    m["n_counted"] = _scalar(con, "SELECT count(*) FROM plays WHERE is_play")[0]
    m["n_sessions"] = _scalar(con, "SELECT count(*) FROM sessions")[0]
    m["n_artists"] = _scalar(con, "SELECT count(DISTINCT artist_name) FROM plays")[0]
    m["date_lo"], m["date_hi"] = _scalar(con, "SELECT min(date_local), max(date_local) FROM plays")
    m["total_hours"] = _scalar(con, "SELECT sum(ms_played)/3600000.0 FROM plays WHERE is_play")[0]

    m["skip_overall"] = _scalar(con, "SELECT skip_rate FROM skip_overall")[0]
    m["skip_shuffle"] = _scalar(con, "SELECT skip_rate FROM skip_by_shuffle WHERE shuffle")[0]
    m["skip_intentional"] = _scalar(con, "SELECT skip_rate FROM skip_by_shuffle WHERE NOT shuffle")[0]
    m["skip_discovery"] = _scalar(con, "SELECT skip_rate FROM skip_by_familiarity WHERE familiarity='discovery'")[0]
    m["skip_familiar"] = _scalar(con, "SELECT skip_rate FROM skip_by_familiarity WHERE familiarity='familiar'")[0]

    c = _scalar(con, "SELECT top1pct_share, top10pct_share, hhi, n_artists FROM concentration")
    m["top1"], m["top10"], m["hhi"], m["conc_artists"] = c

    m["peak"] = _scalar(con, """
        SELECT day_name, hour_local FROM volume_hour_dow
        ORDER BY minutes DESC LIMIT 1""")
    m["busiest_year"] = _scalar(con, "SELECT year, hours FROM yearly_volume ORDER BY hours DESC LIMIT 1")
    m["discovery_avg"] = _scalar(con, """
        SELECT avg(new_artists) FROM discovery_monthly
        WHERE month > (SELECT min(month) FROM discovery_monthly)""")[0]
    m["binge"] = _scalar(con, "SELECT track_name, artist_name, max_consecutive FROM most_binged LIMIT 1")
    for k in (1, 3, 6):
        row = _scalar(con, f"SELECT retention FROM retention_curve WHERE k={k}")
        m[f"ret_{k}"] = row[0] if row else None
    m["partial_months"] = [r[0] for r in con.execute(
        "SELECT strftime(month,'%Y-%m') FROM monthly_volume WHERE is_partial_month ORDER BY month"
    ).fetchall()]

    # --- crate (SPEC §8.4) ---
    m["crate_size"] = _scalar(con, "SELECT count(*) FROM crate")[0]
    m["crate_tagged"] = _scalar(con, "SELECT count(*) FROM crate WHERE camelot IS NOT NULL")[0]
    m["crate_status"] = con.execute("""
        SELECT crate_status, count(*) AS n FROM crate GROUP BY 1 ORDER BY n DESC
    """).fetchall()
    m["burned_top"] = con.execute(
        "SELECT track_name, artist_name, n_plays, days_rested FROM crate_burned LIMIT 3"
    ).fetchall()
    m["rested_top"] = con.execute(
        "SELECT track_name, artist_name, n_plays, days_rested FROM crate_rested LIMIT 3"
    ).fetchall()
    m["deepest_band"] = _scalar(con, """
        SELECT band, n_tracks FROM crate_tempo_bands
        WHERE band <> 'untagged' ORDER BY n_tracks DESC LIMIT 1""")
    m["key_spread"] = _scalar(con, "SELECT count(*) FROM crate_key_coverage")[0]

    # --- transitions (SPEC §8.5) ---
    m["n_transitions"] = _scalar(con, "SELECT count(*) FROM transitions")[0]
    m["n_edges"] = _scalar(con, "SELECT count(*) FROM transition_edges")[0]
    m["move_perf"] = con.execute("""
        SELECT move, n, skip_rate FROM transition_move_performance ORDER BY skip_rate
    """).fetchall()
    return m


def print_validation(con, m, data_dir, coverage=None) -> list:
    """SPEC §12 checks. Returns the report lines for the QA section too."""
    lines = []
    kept = m["raw_total"] - m["raw_podcast"] - m["raw_nullname"]
    recon_ok = kept == m["n_plays"]
    lines.append(f"- **Row reconciliation:** {m['raw_total']:,} raw "
                 f"= {m['n_plays']:,} music plays + {m['raw_podcast']:,} podcasts "
                 f"+ {m['raw_nullname']:,} null-name dropped "
                 f"-> {'balanced ✓' if recon_ok else 'MISMATCH ✗'}")
    lines.append(f"- **Edge months excluded from trends:** {', '.join(m['partial_months']) or 'none'}")
    peak_day, peak_hr = m["peak"]
    tz_ok = 6 <= int(peak_hr) <= 23
    lines.append(f"- **Timezone sanity:** peak listening at {peak_day} {int(peak_hr):02d}:00 local "
                 f"-> {'plausible ✓' if tz_ok else 'check UTC/local ✗'}")

    if coverage is not None:
        lines.append(f"- **Feature coverage:** {coverage.summary()} -> "
                     f"{'usable ✓' if coverage.real_pct > 0.5 else 'mostly synthetic ✗'}")
        lines.append(f"- **Crate tagged for mixing:** {m['crate_tagged']:,}/"
                     f"{m['crate_size']:,} tracks carry a key, spread over "
                     f"{m['key_spread']}/24 Camelot slots")

    sens = session_gap_sensitivity(data_dir)
    sens_str = "; ".join(f"{int(r.gap_min)}min -> {int(r.n_sessions):,} sessions"
                         for _, r in sens.iterrows())
    lines.append(f"- **Session-gap sensitivity:** {sens_str}")

    print("\nValidation (SPEC §12):")
    for ln in lines:
        print("  " + ln.replace("**", ""))
    return lines


def dj_sections(m, hh, ht, sets, figs) -> list:
    """The crate / transition / set-building half of the report (SPEC §8-9)."""
    pct = lambda x: f"{x*100:.1f}%"
    L = []

    L.append("## The crate\n")
    status = ", ".join(f"**{n}** {s}" for s, n in m["crate_status"])
    L.append(f"{m['crate_size']:,} distinct tracks, {m['crate_tagged']:,} carrying a "
             f"key across {m['key_spread']}/24 Camelot slots. By status: {status}.")
    L.append(f"Deepest tempo band is **{m['deepest_band'][0]}** "
             f"({m['deepest_band'][1]} tracks).")
    L.append(f"   ![crate health]({figs['crate_health']})")
    L.append(f"   ![camelot wheel]({figs['camelot_wheel']})")
    L.append(f"   ![tempo bands]({figs['tempo_bands']})\n")

    if m["burned_top"]:
        items = "; ".join(f"*{t}* — {a} ({n} plays, {d}d rested)"
                          for t, a, n, d in m["burned_top"])
        L.append(f"**Rest these** (top of the burn ranking): {items}.")
    if m["rested_top"]:
        items = "; ".join(f"*{t}* — {a} ({n} plays, {d}d rested)"
                          for t, a, n, d in m["rested_top"])
        L.append(f"**Bring these back** (reliable, long dormant): {items}.\n")

    L.append("## Do harmonic transitions actually hold?\n")
    L.append(f"{m['n_transitions']:,} in-session transitions over "
             f"{m['n_edges']:,} distinct ordered pairs. Skip rate of the *incoming* "
             "track, by harmonic move:\n")
    L.append("| move | n | skip rate |")
    L.append("|---|---:|---:|")
    for move, n, rate in m["move_perf"]:
        L.append(f"| {move.replace('_', ' ')} | {n:,} | {pct(rate)} |")
    L.append("")
    L.append(f"   ![transition performance]({figs['transition_performance']})\n")

    L.append("**H2: a clashing transition loses the incoming track more often.**\n")
    L.append("Tested with Cochran-Mantel-Haenszel, stratified on shuffle. Shuffle "
             "raises the skip rate *and* produces more clashes, so it is a common "
             "cause of both variables and pooling would credit its skips to bad "
             "harmony:\n")
    for s in hh.strata:
        L.append(f"- *{s.label}*: clash {pct(s.rate_exposed)} "
                 f"({s.x_exposed:,}/{s.n_exposed:,}) vs compatible "
                 f"{pct(s.rate_control)} ({s.x_control:,}/{s.n_control:,}), "
                 f"OR **{s.odds_ratio:.2f}**")
    L.append(f"- adjusted (MH) odds ratio **{hh.odds_ratio:.2f}** "
             f"[{hh.or_low:.2f}, {hh.or_high:.2f}], CMH chi-square "
             f"**{hh.statistic:.1f}**, p = **{hh.p_value:.2e}**")
    L.append(f"- **the confound was worth measuring:** pooling gives an odds ratio "
             f"of {hh.crude_odds_ratio:.2f}, so ignoring shuffle would have "
             f"overstated the harmonic effect by **{hh.confounding_pct:.0f}%**")
    L.append(f"- the same test on tempo: jumping past the ±6% pitch fader carries "
             f"an adjusted odds ratio of **{ht.odds_ratio:.2f}** "
             f"[{ht.or_low:.2f}, {ht.or_high:.2f}], p = {ht.p_value:.2e}\n")
    L.append("> *Caveat, and the honest one:* the committed sample is synthetic and "
             "its generator queues harmonically-adjacent tracks on purpose, so this "
             "test is **guaranteed** to find an effect here. What it demonstrates is "
             "the measurement machinery — the stratification, the odds ratios, the "
             "confounding estimate. The actual experiment is running it against a "
             "real export.\n")

    L.append("## Building a set\n")
    L.append("Beam search over the crate, scoring every transition on harmonic move, "
             "tempo reachability, fit to the arc's energy target, energy continuity, "
             "and the track's own crate readiness — with hard constraints for "
             "repeats, artist spacing and the pitch fader.\n")
    L.append("| arc | crate supply | harmonic | arc miss | beam | greedy | random |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for arc, plan, feas, ev in sets:
        L.append(f"| {arc} | {feas.mean_supply:.0%} | {plan.harmonic_rate:.0%} | "
                 f"{plan.arc_deviation(arc):.2f} | **{ev.beam:.3f}** | "
                 f"{ev.greedy:.3f} | {ev.random_mean:.3f} |")
    L.append("")
    L.append(f"   ![set energy arc]({figs['set_energy_arc']})\n")

    weak = [(arc, feas) for arc, _, feas, _ in sets if feas.worst_supply < 0.5]
    if weak:
        arc, feas = weak[0]
        L.append(f"**Finding: this crate cannot open a night.** The *{arc}* arc is "
                 f"{feas.verdict} — mean supply {feas.mean_supply:.0%}, thinnest at "
                 f"slot {feas.worst_slot + 1}/{feas.n_slots} where it wants energy "
                 f"{feas.worst_target:.2f}. The quiet tracks exist, but "
                 f"{feas.stranded} of them sit outside pitch-fader reach of where "
                 f"the set is running, because energy and tempo travel together. "
                 f"That is a gap in the record bag, not a bug in the sequencer.\n")

    best = max(sets, key=lambda s: s[3].beam - s[3].random_mean)
    L.append(f"Against the baselines on the *{best[0]}* arc: beam "
             f"**{best[3].beam:.3f}** vs greedy {best[3].greedy:.3f} vs random "
             f"{best[3].random_mean:.3f} (best of {best[3].random_trials}: "
             f"{best[3].random_best:.3f}); "
             f"**{best[3].beam_harmonic_rate:.0%}** of the beam's transitions are "
             f"harmonically clean against {best[3].random_harmonic_rate:.0%} for "
             f"random. The lift over greedy is real but small — most of the gain is "
             f"in the objective and the hard constraints, not the lookahead.\n")

    L.append("### Example: the peak-time set\n")
    peak = next((s for s in sets if s[0] == "peak"), sets[0])
    L.append("```")
    L.append(peak[1].describe())
    L.append("```\n")
    return L


def write_report(m, h, qa_lines, fig_paths, hh=None, ht=None, sets=None) -> Path:
    pct = lambda x: f"{x*100:.1f}%"
    sig = "statistically significant" if h["p_one_sided"] < 0.05 else "not significant"
    direction = "expanding" if m["discovery_avg"] and m["discovery_avg"] > 0 else "flat"
    figs = {p.stem: f"figures/{p.name}" for p in fig_paths}

    L = []
    L.append("# Spotify Listening Analysis — Findings\n")
    L.append("> Auto-generated by `python -m src.run_analysis`. Numbers below are "
             "computed from the loaded dataset; re-run to refresh.\n")
    L.append("## Dataset\n")
    L.append(f"- **{m['n_plays']:,}** music plays "
             f"(**{m['n_counted']:,}** counted, ≥30s) across "
             f"**{m['n_sessions']:,}** sessions and **{m['n_artists']}** artists")
    L.append(f"- Range **{m['date_lo']} → {m['date_hi']}**, "
             f"**{m['total_hours']:.0f} hours** of counted listening\n")

    L.append("## Findings\n")
    L.append(f"1. **Listening volume.** Busiest year was **{int(m['busiest_year'][0])}** "
             f"(~{m['busiest_year'][1]:.0f} h). Listening peaks **{m['peak'][0]} "
             f"around {int(m['peak'][1]):02d}:00** local.")
    L.append(f"   ![volume]({figs['volume_trend']})")
    L.append(f"   ![heatmap]({figs['hour_dow_heatmap']})\n")
    L.append(f"2. **Skip rate — the honest number Wrapped never shows: "
             f"{pct(m['skip_overall'])}.** On shuffle it is **{pct(m['skip_shuffle'])}** "
             f"vs **{pct(m['skip_intentional'])}** on intentional plays; "
             f"discovery plays skip at {pct(m['skip_discovery'])} vs "
             f"{pct(m['skip_familiar'])} for familiar tracks.")
    L.append(f"   ![skip]({figs['skip_breakdown']})\n")
    L.append(f"3. **Taste concentration.** The top 10% of artists account for "
             f"**{pct(m['top10'])}** of listening (top 1%: {pct(m['top1'])}); "
             f"HHI = **{m['hhi']:.3f}**.")
    L.append(f"   ![concentration]({figs['concentration']})\n")
    L.append(f"4. **Discovery.** ~**{m['discovery_avg']:.1f} new artists/month** "
             f"(taste looks {direction}).")
    L.append(f"   ![discovery]({figs['discovery_trend']})\n")
    L.append(f"5. **Artist retention.** Of artists discovered in a month, "
             f"**{pct(m['ret_1'])}** are still played 1 month later, "
             f"**{pct(m['ret_3'])}** at 3 months, **{pct(m['ret_6'])}** at 6 months.")
    L.append(f"   ![retention]({figs['cohort_retention']})\n")
    L.append(f"6. **Most binged.** *{m['binge'][0]}* by {m['binge'][1]} — "
             f"**{int(m['binge'][2])}** consecutive plays in one session.\n")

    L.append("## Hypothesis test — shuffle\n")
    L.append("**H1: skip rate is higher on shuffle than on intentional plays.**\n")
    L.append(f"- shuffle {pct(h['shuffle_rate'])} ({h['shuffle_skips']:,}/{h['shuffle_n']:,}) "
             f"vs intentional {pct(h['intentional_rate'])} "
             f"({h['intentional_skips']:,}/{h['intentional_n']:,})")
    L.append(f"- difference **{pct(h['difference'])}**, two-proportion z = "
             f"**{h['z']:.1f}**, one-sided p = **{h['p_one_sided']:.2e}** ({sig})")
    L.append(f"- effect size Cohen's h = **{h['cohens_h']:.2f}**")
    L.append("- *Caveat:* plays are autocorrelated within sessions, so treat this "
             "as descriptive evidence, not a clean randomized experiment (SPEC §7.3).\n")

    if hh is not None and sets:
        L.extend(dj_sections(m, hh, ht, sets, figs))

    L.append("## Validation (SPEC §12)\n")
    L.extend(qa_lines)
    L.append("")

    path = ROOT / "REPORT.md"
    # newline="\n" keeps REPORT.md as LF on every platform (clean git status).
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Spotify listening analysis.")
    ap.add_argument("--data", default="data/sample", help="dir of *.json export files")
    ap.add_argument("--tz-offset", type=int, default=330, help="home tz offset minutes from UTC")
    ap.add_argument("--session-gap", type=int, default=30, help="session inactivity gap (min)")
    ap.add_argument("--burn-half-life", type=float, default=60.0,
                    help="days; half-life for the rotation-burn weighting")
    ap.add_argument("--features", default=None,
                    help="path to a Mixed In Key / Rekordbox feature CSV")
    ap.add_argument("--set-minutes", type=float, default=60.0,
                    help="target length of the sets built for the report")
    ap.add_argument("--random-trials", type=int, default=200,
                    help="random orderings to benchmark the set builder against")
    ap.add_argument("--no-figures", action="store_true", help="skip figure rendering")
    args = ap.parse_args()

    # Console output includes a few non-ASCII glyphs (§, ✓, →); make stdout UTF-8
    # so it prints on a Windows cp1252 terminal too.
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    con = connect()
    result = build(con, args.data, tz_offset_min=args.tz_offset,
                   session_gap_min=args.session_gap,
                   burn_half_life_days=args.burn_half_life,
                   features_file=args.features)
    print(f"Loaded {result.rows_loaded:,} rows from {args.data}.")
    print(f"Features: {result.coverage.summary()}")

    m = collect(con)
    h = hypothesis_test(con)
    print(f"Skip rate overall {m['skip_overall']*100:.1f}% | "
          f"shuffle {m['skip_shuffle']*100:.1f}% vs intentional {m['skip_intentional']*100:.1f}% "
          f"(z={h['z']:.1f}, p={h['p_one_sided']:.1e})")

    hh = harmonic_hypothesis_test(con)
    ht = tempo_hypothesis_test(con)
    print(f"Harmonic: clash vs compatible, adjusted OR {hh.odds_ratio:.2f} "
          f"[{hh.or_low:.2f}, {hh.or_high:.2f}], p={hh.p_value:.1e} "
          f"(pooling would say {hh.crude_odds_ratio:.2f}, "
          f"{hh.confounding_pct:+.0f}% confounded)")

    crate = load_crate(con)
    edges = load_proven_edges(con)
    sets = []
    for arc in ARCS_REPORTED:
        plan = build_set(crate, args.set_minutes, arc, proven_edges=edges)
        feas = arc_feasibility(crate, arc, args.set_minutes)
        ev = evaluate(crate, args.set_minutes, arc, proven_edges=edges,
                      random_trials=args.random_trials)
        sets.append((arc, plan, feas, ev))
        print(f"  {arc:8} {feas.mean_supply:4.0%} supply | "
              f"{plan.harmonic_rate:4.0%} harmonic | arc miss "
              f"{plan.arc_deviation(arc):.2f} | {ev.summary()}")

    qa_lines = print_validation(con, m, args.data, result.coverage)

    fig_paths = []
    if not args.no_figures:
        fig_paths = charts.render_all(con)
        print(f"\nRendered {len(fig_paths)} figures to figures/.")
    else:
        fig_paths = [Path(f"figures/{n}.png") for n in
                     ("volume_trend", "hour_dow_heatmap", "skip_breakdown",
                      "cohort_retention", "discovery_trend", "concentration",
                      "camelot_wheel", "transition_performance", "crate_health",
                      "tempo_bands", "set_energy_arc")]

    report = write_report(m, h, qa_lines, fig_paths, hh, ht, sets)
    print(f"Wrote {report.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
