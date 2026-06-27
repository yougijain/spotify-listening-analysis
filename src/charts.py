"""Render the report figures from the DuckDB models (SPEC §8).

Thin plotting glue only — every number plotted comes straight from a SQL table.
Each function writes one PNG to figures/ and returns its path. Run this module
directly to (re)build the database and render all figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import duckdb

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
    ax.set_yticks(range(7)); ax.set_yticklabels(DOW_LABELS)
    ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels(range(0, 24, 2))
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
        vals.append(r.skip_rate); colors.append(ACCENT if r.shuffle else "#7bd49b")
    for _, r in fam.iterrows():
        labels.append(r.familiarity); vals.append(r.skip_rate); colors.append("#b3b3b3")
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
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    return _save(fig, "concentration.png")


def render_all(con: duckdb.DuckDBPyConnection) -> list:
    return [
        volume_trend(con), hour_dow_heatmap(con), skip_breakdown(con),
        cohort_retention(con), discovery_trend(con), concentration(con),
    ]


if __name__ == "__main__":
    from .pipeline import build, connect

    con = connect()
    build(con, "data/sample")
    for p in render_all(con):
        print("wrote", p)
