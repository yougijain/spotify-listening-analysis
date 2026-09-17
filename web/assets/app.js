/* Crate Intelligence — dashboard logic.
 *
 * Two jobs: render the precomputed findings, and run the set builder live.
 *
 * The set builder is the only place this file does anything clever, and it is
 * deliberately not much: all the domain scoring happened in Python and shipped
 * as a weighted graph (web/data/graph.json). Every component of a transition's
 * score except the energy-arc term is position-independent, so it is already
 * baked into the edge weight. The search below adds one term and walks the
 * graph. That keeps the mixing rules in one tested place instead of two.
 *
 * No framework and no build step on purpose: the whole page is four static
 * files, which is what makes it hostable anywhere for nothing.
 */

const DATA = {};
const $ = (sel) => document.querySelector(sel);

const fmtPct = (x, digits = 0) => `${(x * 100).toFixed(digits)}%`;
const fmtNum = (x) => x.toLocaleString("en-US");

// ---------------------------------------------------------------- data load

async function loadAll() {
  const names = ["summary", "crate", "graph", "charts", "config"];
  const loaded = await Promise.all(
    names.map((n) => fetch(`data/${n}.json`).then((r) => {
      if (!r.ok) throw new Error(`could not load ${n}.json (${r.status})`);
      return r.json();
    }))
  );
  names.forEach((n, i) => { DATA[n] = loaded[i]; });
}

// ------------------------------------------------------------ scoring (thin)

/* Target energy for a slot. The curve is 101 sampled points from
 * src/harmonic.py rather than a formula transcribed into JS — a transcription
 * can drift from the source, a sample cannot. Interpolation between points is
 * verified against arc_target() in tests/test_web_export.py. */
function arcTarget(arc, position, total) {
  const curve = DATA.config.arcs[arc];
  if (total <= 1) return curve[0];
  const x = (position / (total - 1)) * 100;
  const lo = Math.floor(x);
  const hi = Math.min(100, lo + 1);
  const frac = x - lo;
  return curve[lo] * (1 - frac) + curve[hi] * frac;
}

function energyFit(track, arc, position, total) {
  if (track.energy == null) return 0.5;
  return Math.max(0, 1 - Math.abs(track.energy - arcTarget(arc, position, total)));
}

// ------------------------------------------------------------- beam search

const BEAM_WIDTH = 12;

/* Mirrors src/setbuilder.build_set. Edges already encode the tempo constraint
 * and the position-independent score; artist spacing depends on the whole path
 * so it has to be checked here. */
function buildSet(opts) {
  const { arc, slots, artistGap, includeBurned, preferProven } = opts;
  const tracks = DATA.crate;
  const edges = DATA.graph.edges;
  const wEnergy = DATA.config.weights.energy;

  const eligible = (i) => includeBurned || tracks[i].status !== "burned";

  const allowed = (beam, candidate) => {
    if (!eligible(candidate)) return false;
    if (beam.used.has(candidate)) return false;
    if (artistGap > 0) {
      const artist = tracks[candidate].artist;
      const from = Math.max(0, beam.path.length - artistGap);
      for (let k = from; k < beam.path.length; k++) {
        if (tracks[beam.path[k]].artist === artist) return false;
      }
    }
    return true;
  };

  // Openers: the tracks whose energy best fits slot 0 of the arc.
  const openers = tracks
    .map((t, i) => i)
    .filter(eligible)
    .sort((a, b) => {
      const d = energyFit(tracks[b], arc, 0, slots) - energyFit(tracks[a], arc, 0, slots);
      return d !== 0 ? d : tracks[b].readiness - tracks[a].readiness;
    })
    .slice(0, BEAM_WIDTH);

  if (!openers.length) return { path: [], steps: [] };

  let beams = openers.map((i) => ({
    path: [i], steps: [], score: 0, used: new Set([i]),
  }));

  for (let position = 1; position < slots; position++) {
    const next = [];
    for (const beam of beams) {
      const last = beam.path[beam.path.length - 1];
      const candidates = [];
      for (const [to, base, proven] of edges[last]) {
        if (!allowed(beam, to)) continue;
        // Edges ship the proven term separately so it can be subtracted back
        // out when the user turns that preference off.
        const score = (preferProven ? base : base - proven)
          + wEnergy * energyFit(tracks[to], arc, position, slots);
        candidates.push({ to, score, proven });
      }
      if (!candidates.length) { next.push(beam); continue; }
      candidates.sort((a, b) => b.score - a.score);
      for (const c of candidates.slice(0, 8)) {
        const used = new Set(beam.used);
        used.add(c.to);
        next.push({
          path: [...beam.path, c.to],
          steps: [...beam.steps, { from: last, to: c.to, score: c.score, proven: c.proven }],
          score: beam.score + c.score,
          used,
        });
      }
    }
    if (!next.length) break;
    next.sort((a, b) => b.score / Math.max(1, b.steps.length)
                      - a.score / Math.max(1, a.steps.length));
    beams = dedupe(next).slice(0, BEAM_WIDTH);
  }

  return beams.reduce((best, b) =>
    b.score / Math.max(1, b.steps.length) > best.score / Math.max(1, best.steps.length)
      ? b : best);
}

/* Without this the beam fills with near-identical sequences and the effective
 * width collapses — the classic way a beam search degrades into greedy. */
function dedupe(beams) {
  const seen = new Set();
  const out = [];
  for (const b of beams) {
    const sig = b.path.join(",");
    if (!seen.has(sig)) { seen.add(sig); out.push(b); }
  }
  return out;
}

// ----------------------------------------------------------- set rendering

const MOVE_LABEL = {
  same_key: "same key", adjacent: "adjacent", relative: "relative",
  energy_boost: "+7 boost", diagonal: "diagonal", clash: "clash",
};

/* Recompute the harmonic move and tempo delta for display. These mirror
 * src/harmonic.py; they are presentational only — the search already used the
 * authoritative values baked into the edge weights. */
function classifyMove(a, b) {
  if (!a || !b) return null;
  const na = parseInt(a, 10), la = a.slice(-1);
  const nb = parseInt(b, 10), lb = b.slice(-1);
  const step = ((nb - na) % 12 + 12) % 12;
  if (la === lb && step === 0) return "same_key";
  if (la === lb && (step === 1 || step === 11)) return "adjacent";
  if (la !== lb && step === 0) return "relative";
  if (la === lb && step === 7) return "energy_boost";
  if (la !== lb && (step === 1 || step === 11)) return "diagonal";
  return "clash";
}

function bpmDelta(a, b) {
  if (a == null || b == null || a <= 0 || b <= 0) return null;
  return Math.min(...[b, b * 2, b / 2].map((c) => Math.abs(c - a) / a));
}

function renderSet(plan, arc) {
  const tracks = DATA.crate;
  const list = $("#setlist");
  list.innerHTML = "";

  let harmonic = 0, moves = 0, biggestStep = 0, arcMiss = 0, energyCount = 0;

  plan.path.forEach((idx, i) => {
    const t = tracks[idx];
    const li = document.createElement("li");

    const prev = i > 0 ? tracks[plan.path[i - 1]] : null;
    const move = prev ? classifyMove(prev.camelot, t.camelot) : null;
    const delta = prev ? bpmDelta(prev.bpm, t.bpm) : null;

    if (move) {
      moves++;
      if (["same_key", "adjacent", "relative"].includes(move)) harmonic++;
    }
    if (prev && prev.energy != null && t.energy != null) {
      biggestStep = Math.max(biggestStep, Math.abs(t.energy - prev.energy));
    }
    if (t.energy != null) {
      arcMiss += Math.abs(t.energy - arcTarget(arc, i, plan.path.length));
      energyCount++;
    }

    const tags = [];
    if (move) tags.push(`<span class="tag move-${move}">${MOVE_LABEL[move]}</span>`);
    if (delta != null) tags.push(`<span class="tag">${(delta * 100).toFixed(1)}%</span>`);
    if (t.camelot) tags.push(`<span class="tag key">${t.camelot}</span>`);
    if (t.bpm != null) tags.push(`<span class="tag bpm">${t.bpm.toFixed(0)}</span>`);
    if (t.energy != null) {
      tags.push(`<span class="energy-bar" title="energy ${t.energy.toFixed(2)}">
        <i style="width:${Math.round(t.energy * 100)}%"></i></span>`);
    }

    li.innerHTML = `
      <span class="slot">${String(i + 1).padStart(2, "0")}</span>
      <span><span class="title">${escapeHtml(t.title)}</span>
            <span class="artist"> — ${escapeHtml(t.artist)}</span></span>
      <span class="tags">${tags.join("")}</span>`;
    list.appendChild(li);
  });

  const harmonicRate = moves ? harmonic / moves : 0;
  const totalMinutes = plan.path.reduce((s, i) => s + (tracks[i].minutes || 4), 0);
  const meanScore = plan.steps.length ? plan.score / plan.steps.length : 0;

  $("#set-metrics").innerHTML = `
    <span>${plan.path.length} tracks</span>
    <span><b>${totalMinutes.toFixed(0)}</b> min</span>
    <span>harmonic <b class="${harmonicRate >= 0.8 ? "good" : "warn"}">${fmtPct(harmonicRate)}</b></span>
    <span>arc miss <b class="${energyCount && arcMiss / energyCount < 0.15 ? "good" : "warn"}">${
      energyCount ? (arcMiss / energyCount).toFixed(2) : "—"}</b></span>
    <span>biggest energy step <b>${biggestStep.toFixed(2)}</b></span>
    <span>mean transition <b>${meanScore.toFixed(3)}</b></span>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// --------------------------------------------------------------- SVG charts

const SVG_NS = "http://www.w3.org/2000/svg";
const el = (name, attrs = {}, text = null) => {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
};

function chartMoves(mount) {
  const rows = DATA.charts.moves;
  const W = 520, rowH = 38, padL = 96, padR = 96, H = rows.length * rowH + 20;
  const max = Math.max(...rows.map((r) => r.skip_rate));
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  const compatible = new Set(["same_key", "adjacent", "relative"]);

  rows.forEach((r, i) => {
    const y = i * rowH + 10;
    const w = (r.skip_rate / max) * (W - padL - padR);
    const colour = compatible.has(r.move) ? "var(--accent)" : "var(--warn)";
    svg.appendChild(el("text", {
      x: padL - 12, y: y + 15, "text-anchor": "end", class: "bar-label",
    }, MOVE_LABEL[r.move] || r.move));
    svg.appendChild(el("rect", {
      x: padL, y, width: Math.max(2, w), height: 21, rx: 4, fill: colour, opacity: 0.85,
    }));
    svg.appendChild(el("text", {
      x: padL + w + 10, y: y + 15, class: "bar-value",
    }, `${fmtPct(r.skip_rate, 1)}  n=${fmtNum(r.n)}`));
  });
  mount.appendChild(svg);
}

function chartWheel(mount) {
  const size = 420, cx = size / 2, cy = size / 2;
  const svg = el("svg", { viewBox: `0 0 ${size} ${size}`, role: "img" });
  const counts = new Map(DATA.charts.wheel.map((d) => [`${d.number}${d.letter}`, d.tracks]));
  const biggest = Math.max(1, ...counts.values());

  const arc = (r0, r1, a0, a1) => {
    const p = (r, a) => [cx + r * Math.sin(a), cy - r * Math.cos(a)];
    const [x0, y0] = p(r1, a0), [x1, y1] = p(r1, a1);
    const [x2, y2] = p(r0, a1), [x3, y3] = p(r0, a0);
    return `M${x0} ${y0}A${r1} ${r1} 0 0 1 ${x1} ${y1}L${x2} ${y2}A${r0} ${r0} 0 0 0 ${x3} ${y3}Z`;
  };

  const step = (2 * Math.PI) / 12;
  [["A", 78, 124, "var(--deck)"], ["B", 132, 182, "var(--accent)"]].forEach(
    ([letter, r0, r1, colour]) => {
      for (let n = 1; n <= 12; n++) {
        const a0 = (n - 1) * step + step * 0.04;
        const a1 = n * step - step * 0.04;
        const count = counts.get(`${n}${letter}`) || 0;
        const opacity = 0.12 + 0.8 * (count / biggest);
        const path = el("path", {
          d: arc(r0, r1, a0, a1),
          fill: colour, opacity,
          stroke: "var(--bg-raised)", "stroke-width": 2,
        });
        path.appendChild(el("title", {}, `${n}${letter} — ${count} tracks`));
        svg.appendChild(path);
        if (count) {
          const mid = (a0 + a1) / 2, rm = (r0 + r1) / 2;
          // Contrast off the wedge's own opacity, not the global maximum. The
          // inner ring never reaches the crate-wide peak, so keying off that
          // left every minor-key label grey-on-saturated-purple.
          // Set via `style`, not a `fill` attribute: presentation attributes
          // lose to the stylesheet's `svg text { fill }` rule, which silently
          // repainted every one of these grey.
          svg.appendChild(el("text", {
            x: cx + rm * Math.sin(mid), y: cy - rm * Math.cos(mid) + 4,
            "text-anchor": "middle", "font-size": 11, "font-weight": 600,
            style: `fill:${opacity > 0.52 ? "#07100b" : "#e8edf2"}`,
          }, String(count)));
        }
      }
    });

  for (let n = 1; n <= 12; n++) {
    const a = (n - 0.5) * step;
    svg.appendChild(el("text", {
      x: cx + 200 * Math.sin(a), y: cy - 200 * Math.cos(a) + 4,
      "text-anchor": "middle", "font-size": 12,
    }, String(n)));
  }
  mount.appendChild(svg);
}

function chartHealth(mount) {
  const W = 480, H = 340, pad = 46;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  const colour = {
    proven: "var(--accent)", working: "var(--blue)", rested: "var(--deck)",
    unproven: "var(--ink-faint)", risky: "var(--amber)", burned: "var(--warn)",
  };
  const x = (v) => pad + v * (W - pad - 16);
  const y = (v) => H - pad - v * (H - pad - 18);

  [0.25, 0.5, 0.75].forEach((g) => {
    svg.appendChild(el("line", { x1: pad, x2: W - 16, y1: y(g), y2: y(g), class: "gridline" }));
    svg.appendChild(el("line", { y1: 18, y2: H - pad, x1: x(g), x2: x(g), class: "gridline" }));
  });
  svg.appendChild(el("line", { x1: pad, x2: W - 16, y1: y(0), y2: y(0), stroke: "var(--line)" }));
  svg.appendChild(el("line", { x1: pad, x2: pad, y1: 18, y2: y(0), stroke: "var(--line)" }));

  for (const t of DATA.crate) {
    const dot = el("circle", {
      cx: x(t.burn), cy: y(t.hold), r: 3.4,
      fill: colour[t.status] || "var(--ink-faint)", opacity: 0.66,
    });
    dot.appendChild(el("title", {},
      `${t.title} — ${t.artist}\nhold ${t.hold.toFixed(2)} · burn ${t.burn.toFixed(2)} · ${t.status}`));
    svg.appendChild(dot);
  }

  svg.appendChild(el("text", {
    x: W / 2, y: H - 10, "text-anchor": "middle", "font-size": 11,
  }, "rotation burn  →"));
  svg.appendChild(el("text", {
    x: 14, y: H / 2, "font-size": 11, transform: `rotate(-90 14 ${H / 2})`, "text-anchor": "middle",
  }, "hold rate (Wilson LB)  →"));
  mount.appendChild(svg);

  const legend = document.createElement("div");
  legend.className = "metrics";
  legend.style.marginTop = "12px";
  legend.innerHTML = DATA.charts.status
    .map((s) => `<span style="color:${colour[s.status]}">● ${s.status} <b>${s.tracks}</b></span>`)
    .join("");
  mount.appendChild(legend);
}

// ------------------------------------------------------------ static panels

function renderHeroStats() {
  const d = DATA.summary.dataset;
  const h = DATA.summary.harmonic_test;
  const stats = [
    { value: fmtNum(d.plays), label: "plays analysed", cls: "" },
    { value: fmtNum(d.transitions), label: "in-session transitions", cls: "" },
    { value: fmtNum(d.tracks), label: "tracks in the crate", cls: "" },
    { value: `${h.odds_ratio.toFixed(2)}×`, label: "skip odds on a clashing transition", cls: "warn" },
    { value: fmtPct(DATA.summary.arcs.peak.harmonic_rate), label: "harmonic transitions in a built set", cls: "deck" },
  ];
  $("#hero-stats").innerHTML = stats.map((s) =>
    `<div class="stat"><div class="value ${s.cls}">${s.value}</div>
     <div class="label">${s.label}</div></div>`).join("");
}

function renderHarmonicTest() {
  const h = DATA.summary.harmonic_test;
  const t = DATA.summary.tempo_test;
  const rows = h.strata.map((s) => `
    <div class="row"><span class="k">${s.label}: clash vs compatible</span>
    <span class="v">${fmtPct(s.exposed_rate, 1)} vs ${fmtPct(s.control_rate, 1)}
    &nbsp;OR ${s.odds_ratio.toFixed(2)}</span></div>`).join("");

  $("#harmonic-test").innerHTML = `
    <h3>H2 · Cochran–Mantel–Haenszel, stratified on shuffle</h3>
    <div class="readout">
      ${rows}
      <div class="row"><span class="k">adjusted odds ratio</span>
        <span class="v good">${h.odds_ratio.toFixed(2)} [${h.ci[0].toFixed(2)}, ${h.ci[1].toFixed(2)}]</span></div>
      <div class="row"><span class="k">CMH χ², p</span>
        <span class="v">${h.statistic.toFixed(1)}, ${h.p.toExponential(1)}</span></div>
      <div class="row"><span class="k">if you pooled instead</span>
        <span class="v warn">${h.crude_odds_ratio.toFixed(2)} — ${h.confounding_pct.toFixed(0)}% overstated</span></div>
      <div class="row"><span class="k">same test on tempo jumps</span>
        <span class="v">${t.odds_ratio.toFixed(2)} [${t.ci[0].toFixed(2)}, ${t.ci[1].toFixed(2)}]</span></div>
      <p class="note">
        Shuffle raises the skip rate <em>and</em> produces more clashes, so it is
        a common cause of both variables. Pooling hands shuffle's skips to bad
        harmony and overstates the effect by
        <strong>${h.confounding_pct.toFixed(0)}%</strong> — which is the entire
        reason this is stratified rather than a single two-proportion test.
      </p>
    </div>`;
}

function renderFeasibility(arc) {
  const a = DATA.summary.arcs[arc];
  const box = $("#feasibility");
  if (a.supply >= 0.9) { box.hidden = true; return; }
  box.hidden = false;
  box.innerHTML = `<strong>This crate is thin for a ${arc} set.</strong>
    Mean material supply ${fmtPct(a.supply)} — ${a.verdict}. ${a.stranded} tracks
    sit at the right energy but outside pitch-fader reach of where the set is
    running, because energy and tempo travel together in any real crate. The set
    below is the best available, not a good one. That's a gap in the record bag,
    not a bug in the sequencer.`;
}

function renderMethod() {
  const d = DATA.summary.dataset;
  const cards = [
    ["The export, not the API", `Spotify closed <code>/v1/audio-features</code> to new apps in 2024, so BPM and key cannot come from the API. They come from a Mixed In Key / Rekordbox CSV through a provider chain — the file a working DJ already has on disk.`],
    ["Wilson, not raw rates", `A track played twice and never skipped shows a 100% hold rate. Ranking a crate on that puts the least-evidenced tracks on top, so every rate is a Wilson lower bound.`],
    ["One Camelot wheel", `The SQL models join a 24×24 move table generated <em>from</em> the tested Python rather than reimplementing modular arithmetic in SQL. Drift between the two is structurally impossible.`],
    ["Beam, not greedy", `Greedy sequencing strands itself in a corner of the wheel with nothing that mixes out. Beam search keeps ${BEAM_WIDTH} partial sets alive. The lift is real but small — most of the gain is the objective and the hard constraints.`],
    ["Measured against baselines", `Every arc is benchmarked against greedy and 100 constraint-respecting random orderings, because "the optimiser works" needs a number.`],
    ["Static by design", `${fmtNum(d.plays)} plays reduce to a 132 KB payload. No API, no database, no cold starts — the search runs in your browser over a graph Python scored.`],
  ];
  $("#method-grid").innerHTML = cards.map(([h, p]) =>
    `<div class="card"><h3>${h}</h3><p>${p}</p></div>`).join("");

  $("#caveats").innerHTML = [
    `<strong>The sample is synthetic and its generator queues harmonically-adjacent tracks on purpose.</strong> The harmonic test is therefore <em>guaranteed</em> to find an effect here. What it demonstrates is the measurement machinery — the stratification, the odds ratios, the confounding estimate. The real experiment is running it against a real export.`,
    `<strong>A skip is a proxy for a floor clearing, and not a great one.</strong> This is listening history, not a club. It measures what I reach past on my own headphones, which correlates with what fails in a room but is not the same thing.`,
    `<strong>Plays are autocorrelated within sessions.</strong> Stratifying on shuffle handles the obvious confound, not the dependence structure. This is descriptive evidence, not a randomised experiment.`,
    `<strong>The objective's weights are judgement calls.</strong> They encode conventional booth practice, not anything measured. They are in one place and easy to argue with, which is the most that can honestly be claimed for them.`,
  ].map((c) => `<li>${c}</li>`).join("");
}

// -------------------------------------------------------------------- wiring

function currentOptions() {
  const arc = $("#arc").querySelector('[aria-checked="true"]').dataset.arc;
  const minutes = Number($("#minutes").value);
  const avg = DATA.crate.reduce((s, t) => s + (t.minutes || 4), 0) / DATA.crate.length;
  return {
    arc,
    slots: Math.max(2, Math.min(DATA.crate.length, Math.round(minutes / avg))),
    artistGap: Number($("#gap").value),
    includeBurned: $("#include-burned").checked,
    preferProven: $("#prefer-proven").checked,
  };
}

function rebuild() {
  const opts = currentOptions();
  renderFeasibility(opts.arc);
  renderSet(buildSet(opts), opts.arc);
}

function wireControls() {
  const arcBox = $("#arc");
  arcBox.innerHTML = DATA.config.arc_names.map((name, i) =>
    `<button role="radio" data-arc="${name}" aria-checked="${name === "peak"}">${name}</button>`
  ).join("");
  arcBox.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    arcBox.querySelectorAll("button").forEach((b) =>
      b.setAttribute("aria-checked", String(b === btn)));
    rebuild();
  });

  const minutes = $("#minutes");
  minutes.addEventListener("input", () => { $("#minutes-out").textContent = `${minutes.value} min`; });
  minutes.addEventListener("change", rebuild);

  const gap = $("#gap");
  gap.addEventListener("input", () => {
    $("#gap-out").textContent = gap.value === "0" ? "off" : `${gap.value} tracks`;
  });
  gap.addEventListener("change", rebuild);

  $("#include-burned").addEventListener("change", rebuild);
  $("#prefer-proven").addEventListener("change", rebuild);
  $("#rebuild").addEventListener("click", rebuild);
}

async function main() {
  try {
    await loadAll();
  } catch (err) {
    document.body.insertAdjacentHTML("afterbegin",
      `<div class="wrap" style="padding:32px 24px;color:var(--warn)">
         Could not load the data files: ${escapeHtml(err.message)}.
         Run <code>python -m src.export_web</code> and serve this directory.
       </div>`);
    return;
  }
  renderHeroStats();
  renderHarmonicTest();
  renderMethod();
  chartMoves($("#chart-moves"));
  chartWheel($("#chart-wheel"));
  chartHealth($("#chart-health"));
  wireControls();
  rebuild();
}

main();
