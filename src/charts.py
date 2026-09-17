"""Render the report figures from the DuckDB models (SPEC §8).

Thin plotting glue only — every number plotted comes straight from a SQL table.
Each function writes one PNG to figures/ and returns its path. Run this module
directly to (re)build the database and render all figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
ACCENT = "#1DB954"   # Spotify green
INK = "#191414"
MUTED = "#9aa0a6"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": "#eeeeee",
    "figure.autolayout": True,
})

DOW_ORDER = [1, 2, 3, 4, 5, 6, 0]            # Mon..Sun (DuckDB dow: 0=Sun)
DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def volume_trend(con: duckdb.DuckDBPyConnection) -> Path:
    d = con.execute("SELECT month, hours, is_partial_month FROM monthly_volume ORDER BY month").df()
    full = d[~d.is_partial_month]
    partial = d[d.is_partial_month]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(full.month, full.hours, color=ACCENT, lw=2, marker="o", ms=4, label="full month")
    ax.scatter(partial.month, partial.hours, facecolors="white", edgecolors=MUTED,
               zorder=5, label="partial month (excluded from trend)")
    ax.set_title("Listening volume over time")
    ax.set_ylabel("hours / month")
    ax.set_xlabel("")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    return _save(fig, "volume_trend.png")


def hour_dow_heatmap(con: duckdb.DuckDBPyConnection) -> Path:
    d = con.execute("SELECT dow_local, hour_local, minutes FROM volume_hour_dow").df()
    grid = np.zeros((7, 24))
    lookup = {(int(r.dow_local), int(r.hour_local)): r.minutes for _, r in d.iterrows()}
    for ri, dow in enumerate(DOW_ORDER):
        for h in range(24):
            grid[ri, h] = lookup.get((dow, h), 0.0) / 60.0   # hours
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(grid, aspect="auto", cmap="Greens", origin="upper")
    ax.set_yticks(range(7))
    ax.set_yticklabels(DOW_LABELS)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_xlabel("hour of day (local)")
    ax.set_title("When listening happens (hours, local time)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="hours")
    return _save(fig, "hour_dow_heatmap.png")


def skip_breakdown(con: duckdb.DuckDBPyConnection) -> Path:
    overall = con.execute("SELECT skip_rate FROM skip_overall").fetchone()[0]
    shuf = con.execute("SELECT shuffle, skip_rate FROM skip_by_shuffle ORDER BY shuffle").df()
    fam = con.execute("SELECT familiarity, skip_rate FROM skip_by_familiarity ORDER BY familiarity").df()
    labels, vals, colors = ["overall"], [overall], [MUTED]
    for _, r in shuf.iterrows():
        labels.append("shuffle on" if r.shuffle else "shuffle off")
        vals.append(r.skip_rate)
        colors.append(ACCENT if r.shuffle else "#7bd49b")
    for _, r in fam.iterrows():
        labels.append(r.familiarity)
        vals.append(r.skip_rate)
        colors.append("#b3b3b3")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(labels, [v * 100 for v in vals], color=colors)
    ax.set_ylabel("skip rate (%)")
    ax.set_title("Skip rate by context")
    ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9)
    ax.margins(y=0.15)
    return _save(fig, "skip_breakdown.png")


def cohort_retention(con: duckdb.DuckDBPyConnection) -> Path:
    d = con.execute("SELECT k, retention, artists_at_risk FROM retention_curve ORDER BY k").df()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(d.k, d.retention * 100, color=ACCENT, lw=2, marker="o", ms=4)
    ax.set_title("Artist cohort retention")
    ax.set_xlabel("months since discovery (k)")
    ax.set_ylabel("% of cohort still active")
    ax.set_ylim(0, 105)
    return _save(fig, "cohort_retention.png")


def discovery_trend(con: duckdb.DuckDBPyConnection) -> Path:
    # Drop the left-censored first month (everything looks "new" then).
    d = con.execute("""
        SELECT month, new_artists FROM discovery_monthly
        WHERE month > (SELECT min(month) FROM discovery_monthly)
        ORDER BY month
    """).df()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(d.month, d.new_artists, width=20, color=ACCENT)
    ax.set_title("New artists discovered per month")
    ax.set_ylabel("new artists")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    return _save(fig, "discovery_trend.png")


def concentration(con: duckdb.DuckDBPyConnection) -> Path:
    d = con.execute("SELECT cum_artist_frac, cum_listen_frac FROM lorenz ORDER BY cum_artist_frac").df()
    top10 = con.execute("SELECT top10pct_share FROM concentration").fetchone()[0]
    fig, ax = plt.subplots(figsize=(6.2, 6))
    x = np.concatenate([[0], d.cum_artist_frac.values])
    y = np.concatenate([[0], d.cum_listen_frac.values])
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1, label="perfectly even")
    ax.plot(x, y, color=ACCENT, lw=2, label="actual")
    ax.fill_between(x, y, x, color=ACCENT, alpha=0.08)
    ax.set_title("Taste concentration (Lorenz curve)")
    ax.set_xlabel("cumulative share of artists")
    ax.set_ylabel("cumulative share of listening")
    ax.set_aspect("equal")
    ax.annotate(f"top 10% of artists = {top10*100:.0f}% of listening",
                xy=(0.9, 0.32), xytext=(0.18, 0.62), fontsize=9, color=INK,
                arrowprops={"arrowstyle": "->", "color": MUTED})
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    return _save(fig, "concentration.png")




# --- DJ figures (SPEC §8-9) -------------------------------------------------

# A second accent for the DJ-side charts, so crate/transition figures read as a
# distinct family from the listening-behaviour ones.
DECK = "#7c5cff"
WARN = "#e8574a"


def camelot_wheel(con: duckdb.DuckDBPyConnection) -> Path:
    """Crate coverage around the Camelot wheel.

    Polar on purpose: the wheel's whole point is that neighbours mix, so the
    gaps in a crate are a spatial fact and a bar chart hides them. Inner ring is
    minor (A), outer is major (B).
    """
    d = con.execute("""
        SELECT camelot_number, camelot_letter, n_tracks, avg_readiness
        FROM crate_key_coverage
    """).df()

    counts = {(int(r.camelot_number), r.camelot_letter): int(r.n_tracks)
              for _, r in d.iterrows()}
    biggest = max(counts.values()) if counts else 1

    fig, ax = plt.subplots(figsize=(6.4, 6.4), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    width = 2 * np.pi / 12

    for letter, radius, colour in (("A", 0.55, DECK), ("B", 1.15, ACCENT)):
        for number in range(1, 13):
            theta = (number - 1) * width
            n = counts.get((number, letter), 0)
            ax.bar(theta, 0.5, width=width * 0.92, bottom=radius,
                   color=colour, alpha=0.12 + 0.78 * (n / biggest),
                   edgecolor="white", linewidth=1.2)
            if n:
                ax.text(theta, radius + 0.25, str(n), ha="center", va="center",
                        fontsize=7.5, color="white" if n > biggest * 0.45 else INK)

    ax.set_xticks([(i - 1) * width for i in range(1, 13)])
    ax.set_xticklabels([str(i) for i in range(1, 13)], fontsize=9)
    ax.set_yticks([])
    ax.set_ylim(0, 1.75)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_title("Crate coverage around the Camelot wheel\n"
                 "inner = minor (A), outer = major (B)", pad=18)
    return _save(fig, "camelot_wheel.png")


def transition_performance(con: duckdb.DuckDBPyConnection) -> Path:
    """Skip rate by harmonic move, with Wilson bounds on the hold rate."""
    d = con.execute("""
        SELECT move, n, skip_rate, hold_lcb
        FROM transition_move_performance
        ORDER BY skip_rate
    """).df()

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    compatible = {"same_key", "adjacent", "relative"}
    colours = [ACCENT if m in compatible else WARN for m in d.move]
    bars = ax.barh(d.move.str.replace("_", " "), d.skip_rate * 100, color=colours)
    for bar, n in zip(bars, d.n):
        ax.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}%  (n={n:,})", va="center", fontsize=8.5,
                color=INK)
    ax.set_title("Does the next track survive? Skip rate by harmonic move")
    ax.set_xlabel("skip rate of the incoming track (%)")
    ax.set_xlim(0, max(d.skip_rate * 100) * 1.35)
    ax.grid(axis="y", visible=False)
    green = plt.Rectangle((0, 0), 1, 1, color=ACCENT)
    red = plt.Rectangle((0, 0), 1, 1, color=WARN)
    ax.legend([green, red], ["harmonically compatible", "clash"],
              frameon=False, fontsize=8.5, loc="lower right")
    return _save(fig, "transition_performance.png")


def crate_health(con: duckdb.DuckDBPyConnection) -> Path:
    """Hold rate against rotation burn: the two axes a crate decision turns on.

    Worth reading carefully, because the obvious expectation is wrong: the
    burned tracks sit at the TOP right, not the bottom. They are flogged
    *because* they hold — reliability is what earns a track its overplay. So
    "rest these" is not a quality judgement, it is a freshness one, and the
    left-hand column is where a set that does not sound like last month's set
    has to come from.
    """
    d = con.execute("""
        SELECT hold_lcb, rotation_burn, n_plays, crate_status FROM crate
    """).df()

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    palette = {"proven": ACCENT, "working": "#4a90d9", "rested": DECK,
               "unproven": MUTED, "risky": "#f0a202", "burned": WARN}
    for status, colour in palette.items():
        sub = d[d.crate_status == status]
        if sub.empty:
            continue
        ax.scatter(sub.rotation_burn, sub.hold_lcb, s=8 + sub.n_plays * 0.45,
                   alpha=0.62, color=colour, edgecolors="none", label=status)

    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.axvline(0.85, color=MUTED, lw=0.8, ls="--")
    ax.text(0.03, 0.94, "rested + reliable\n→ play these", fontsize=8.5,
            color=INK, va="top")
    ax.text(0.97, 0.06, "flogged + unreliable\n→ rest these", fontsize=8.5,
            color=WARN, ha="right", va="bottom")
    ax.set_title("Crate health: does it hold, and have I flogged it?")
    ax.set_xlabel("rotation burn (percentile of recency-weighted plays)")
    ax.set_ylabel("hold rate, Wilson lower bound")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    return _save(fig, "crate_health.png")


def set_energy_arc(con: duckdb.DuckDBPyConnection) -> Path:
    """Built sets against the arcs they targeted — including where it fails.

    The peak panel tracks its arc closely; the warmup panel does not, and that
    is the point of showing both. The crate is 90% house/techno/dnb, so the
    quiet tracks a warmup needs are downtempo and sit outside the pitch fader's
    reach from where the set is running. That is a gap in the record bag, not a
    bug in the sequencer, and the feasibility number in each title says which.
    """
    from .harmonic import arc_target
    from .setbuilder import arc_feasibility, build_set, load_crate, load_proven_edges

    crate = load_crate(con)
    edges = load_proven_edges(con)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, arc in zip(axes, ("peak", "warmup")):
        plan = build_set(crate, target_minutes=60, arc=arc, proven_edges=edges)
        feas = arc_feasibility(crate, arc, target_minutes=60)
        n = len(plan.tracks)
        xs = list(range(n))
        target = [arc_target(arc, i, n) for i in xs]
        actual = [t.energy if t.energy is not None else np.nan for t in plan.tracks]
        ok = feas.worst_supply >= 0.5

        ax.plot(xs, target, color=MUTED, lw=1.6, ls="--", label=f"{arc} target")
        ax.plot(xs, actual, color=DECK if ok else WARN, lw=2, marker="o", ms=4,
                label="built set")
        ax.fill_between(xs, target, actual, color=DECK if ok else WARN, alpha=0.10)
        ax.set_title(f"{arc}  ·  crate supply {feas.mean_supply:.0%}\n"
                     f"arc miss {plan.arc_deviation(arc):.2f}  ·  "
                     f"{plan.harmonic_rate:.0%} harmonic", fontsize=10.5)
        ax.set_xlabel("slot in set")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_ylabel("energy")
    axes[0].set_ylim(0, 1.05)
    fig.suptitle("The sequencer hits the arc it has material for",
                 y=1.03, fontsize=12, fontweight="bold")
    return _save(fig, "set_energy_arc.png")


def tempo_bands(con: duckdb.DuckDBPyConnection) -> Path:
    """What the crate can actually play, by tempo band."""
    d = con.execute("""
        SELECT band, n_tracks, avg_readiness FROM crate_tempo_bands
    """).df()
    order = ["<100 downtempo", "100-117 slow", "118-129 house",
             "130-144 techno", "145+ fast", "untagged"]
    d["rank"] = d.band.apply(lambda b: order.index(b) if b in order else 99)
    d = d.sort_values("rank")

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    colours = [MUTED if b == "untagged" else ACCENT for b in d.band]
    bars = ax.bar(d.band, d.n_tracks, color=colours)
    for bar, r in zip(bars, d.avg_readiness):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{r:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_title("Crate depth by tempo band  (number = mean set-readiness)")
    ax.set_ylabel("tracks")
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    return _save(fig, "tempo_bands.png")


def render_all(con: duckdb.DuckDBPyConnection) -> list:
    return [
        # Listening behaviour (SPEC §7)
        volume_trend(con), hour_dow_heatmap(con), skip_breakdown(con),
        cohort_retention(con), discovery_trend(con), concentration(con),
        # Crate and set construction (SPEC §8-9)
        camelot_wheel(con), transition_performance(con), crate_health(con),
        tempo_bands(con), set_energy_arc(con),
    ]


if __name__ == "__main__":
    from .pipeline import build, connect

    con = connect()
    build(con, "data/sample")
    for p in render_all(con):
        print("wrote", p)
