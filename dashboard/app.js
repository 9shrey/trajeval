"use strict";
// trajeval dashboard: static, reads data/*.json written by `trajeval export`.

const $ = (s, el = document) => el.querySelector(s);
const view = $("#view");
const tip = $("#tip");
const cache = {};
const MODEL_SLOT = {};  // model name -> categorical slot (1..3), fixed by first appearance in index

async function get(path) {
  if (!cache[path]) {
    cache[path] = fetch(`data/${path}`).then(r => {
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
      return r.json();
    });
  }
  return cache[path];
}

const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 0) => x == null || Number.isNaN(x) ? "–" : `${(x * 100).toFixed(d)}%`;
const num = (x, d = 1) => x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(d);
const signed = (x, d = 1) => x == null ? "–" : `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(d)} pts`;
const AGENT_LABEL = { text: "text tags", native: "native", "gold-path": "gold path (diagnostic)" };
const CAT_LABEL = { lookup: "Lookup", multi_step: "Multi-step", action: "Action", trap: "Trap", open_ended: "Open-ended" };
const CATS = Object.keys(CAT_LABEL);

function swatch(model) {
  const slot = MODEL_SLOT[model] || 1;
  return `<span class="sw" style="background:var(--model-${slot})" aria-hidden="true"></span>`;
}

function strip(results) {
  // results: array of true/false/null, one per trial
  return `<span class="strip" role="img" aria-label="${results.filter(r => r === true).length} of ${results.length} trials passed">${
    results.map(r => `<i class="${r === true ? "p" : r === false ? "f" : "u"}"></i>`).join("")}</span>`;
}

// ---------------------------------------------------------------- tooltip
document.addEventListener("pointermove", e => {
  const t = e.target.closest("[data-tip]");
  if (!t) { tip.hidden = true; return; }
  tip.innerHTML = t.dataset.tip;
  tip.hidden = false;
  const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  const y = e.clientY + 16 + tip.offsetHeight > window.innerHeight ? e.clientY - tip.offsetHeight - 10 : e.clientY + 16;
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
});
document.addEventListener("scroll", () => { tip.hidden = true; }, { passive: true });

// ---------------------------------------------------------------- charts
// Horizontal dot + CI whisker chart. rows: [{label, value, lo, hi, color, tip}], domain [min,max].
function dotWhisker(rows, { domain = [0, 1], fmt = pct, zero = null, rowH = 30, labelW = 190 } = {}) {
  // Size the viewBox to the real container width so SVG text renders at its nominal pixel size.
  const width = Math.max(300, Math.min(1060, view.clientWidth - 34));
  if (width < 560) labelW = Math.min(labelW, 132);
  const h = rows.length * rowH + 34, plotW = width - labelW - 20;
  const nTicks = width < 560 ? 2 : 4;
  const x = v => labelW + ((v - domain[0]) / (domain[1] - domain[0])) * plotW;
  const ticks = [];
  const step = (domain[1] - domain[0]) / nTicks;
  for (let i = 0; i <= nTicks; i++) ticks.push(domain[0] + i * step);
  let s = `<svg viewBox="0 0 ${width} ${h}" role="img" aria-label="Point estimates with 95% confidence intervals">`;
  for (const t of ticks) s += `<line class="grid" x1="${x(t)}" x2="${x(t)}" y1="4" y2="${h - 26}"/><text x="${x(t)}" y="${h - 8}" text-anchor="middle">${fmt(t)}</text>`;
  if (zero != null) s += `<line class="zero" x1="${x(zero)}" x2="${x(zero)}" y1="4" y2="${h - 26}"/>`;
  rows.forEach((r, i) => {
    const y = 14 + i * rowH;
    const maxChars = Math.floor(labelW / 7.2);
    const lab = r.label.length > maxChars ? r.label.slice(0, maxChars - 1) + "…" : r.label;
    s += `<text class="lab" x="0" y="${y + 4}">${esc(lab)}<title>${esc(r.label)}</title></text>`;
    if (r.lo != null && r.hi != null) s += `<line class="ci" x1="${x(r.lo)}" x2="${x(r.hi)}" y1="${y}" y2="${y}" stroke="${r.color}"/>`;
    s += `<circle cx="${x(r.value)}" cy="${y}" r="5.5" fill="${r.color}" stroke="var(--sheet)" stroke-width="2"/>`;
    s += `<rect class="hit" x="${labelW}" y="${y - rowH / 2}" width="${plotW + 10}" height="${rowH}" data-tip="${esc(r.tip)}"/>`;
  });
  return s + "</svg>";
}

// ---------------------------------------------------------------- routing
const routes = { runs: renderRuns, run: renderRun, diffs: renderDiffs, judge: renderJudge, about: renderAbout };
async function route() {
  const [tab = "runs", ...rest] = location.hash.replace(/^#\/?/, "").split("/");
  const fn = routes[tab] || renderRuns;
  const top = tab === "run" ? "runs" : tab;
  document.querySelectorAll(".tabs a").forEach(a =>
    a.dataset.tab === top ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current"));
  try {
    await fn(...rest.map(decodeURIComponent));
  } catch (err) {
    view.innerHTML = `<p>Could not load this view: ${esc(err.message)}. Run <code>trajeval export</code> to rebuild the data files.</p>`;
  }
}
window.addEventListener("hashchange", () => { route(); view.focus({ preventScroll: true }); });
let resizeTimer;
let lastW = window.innerWidth;
window.addEventListener("resize", () => {
  if (Math.abs(window.innerWidth - lastW) < 40) return;  // ignore mobile URL-bar jitter
  lastW = window.innerWidth;
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(route, 200);
});

// ---------------------------------------------------------------- runs
async function renderRuns() {
  const idx = await get("index.json");
  const runs = idx.runs;
  const chart = dotWhisker(runs.map(r => ({
    label: r.run_id, value: r.checks.pass_rate, lo: r.checks.ci?.[0], hi: r.checks.ci?.[1],
    color: `var(--model-${MODEL_SLOT[r.model]})`,
    tip: `<b>${esc(r.run_id)}</b><br>checks-only pass rate ${pct(r.checks.pass_rate, 1)} (95% CI ${pct(r.checks.ci?.[0])}–${pct(r.checks.ci?.[1])})` +
      (r.agent === "gold-path" ? "<br>diagnostic: the harness made the tool calls" : `<br>with judge-graded tasks: ${pct(r.pass_rate, 1)}`),
  })));
  const models = [...new Set(runs.map(r => r.model))];
  view.innerHTML = `
  <section>
    <h2>Runs</h2>
    <p class="intro">Each run puts one model and configuration through all ${idx.suite.n_tasks} tasks, several times each, with a fresh
      simulated company (files, SQL database, web pages, mail outbox) for every attempt. Open-ended answers are graded by
      the LLM judge; everything else by deterministic checks.</p>
    <div class="figure">
      <div class="legend">${models.map(m => `<span>${swatch(m)}${esc(m)}</span>`).join("")}</div>
      ${chart}
      <p class="caption">Pass rate on the 48 deterministically graded tasks. Whiskers: 95% bootstrap interval that
        resamples tasks, since trials of the same task are not independent. Gold-path runs are a diagnostic: the harness
        made the right tool calls and the model only wrote the answer.</p>
    </div>
  </section>
  <section>
    <h3>Every run, side by side</h3>
    <div class="tablewrap"><table>
      <thead><tr><th>Run</th><th>Agent</th><th>Prompt</th><th class="num">Checks only</th><th class="num">With judge</th>
        ${CATS.map(c => `<th class="num">${CAT_LABEL[c]}</th>`).join("")}
        <th class="num">Invalid calls</th><th class="num">Invented recipients</th><th class="num">Redundant</th><th class="num">Efficiency</th>
        <th class="num">Tokens / task</th><th class="num">Time p50</th></tr></thead>
      <tbody>${runs.map(r => `
        <tr class="clickable" tabindex="0" data-href="#/run/${encodeURIComponent(r.run_id)}">
          <td>${swatch(r.model)}<a href="#/run/${encodeURIComponent(r.run_id)}">${esc(r.run_id)}</a>
            ${r.faults?.length ? ' <span class="flag suspect">faults</span>' : ""}</td>
          <td>${AGENT_LABEL[r.agent] || esc(r.agent)}</td><td>${esc(r.prompt)}</td>
          <td class="num"><b>${pct(r.checks.pass_rate, 1)}</b></td>
          <td class="num">${r.agent === "gold-path" ? "–" : pct(r.pass_rate, 1)}</td>
          ${CATS.map(c => `<td class="num">${r.agent === "gold-path" && c === "open_ended" ? "–" : pct(r.pass_rate_by_category[c])}</td>`).join("")}
          <td class="num" data-tip="hallucinated tools ${pct(r.unknown_tool_rate, 1)}, schema errors ${pct(r.schema_error_rate, 1)}, malformed ${pct(r.malformed_rate, 1)}">${pct(r.invalid_call_rate, 1)}</td>
          <td class="num" data-tip="share of email and meeting recipients whose address never appeared in the prompt or a tool result">${pct(r.ungrounded_recipient_rate)}</td>
          <td class="num">${pct(r.redundant_call_rate, 1)}</td><td class="num">${num(r.efficiency, 2)}</td>
          <td class="num">${Math.round(r.tokens_per_task).toLocaleString()}</td><td class="num">${num(r.wall_s_p50)} s</td>
        </tr>`).join("")}</tbody>
    </table></div>
    <p class="caption muted">Invalid calls: share of tool calls that named a tool that doesn't exist, broke the argument
      schema, or couldn't be parsed. Invented recipients: email and meeting addresses the agent made up rather than looked
      up; those calls are valid and succeed, which is what makes them dangerous. Efficiency: (gold calls + 1) / (calls + 1) on passed trajectories, capped at 1.</p>
  </section>`;
  bindRowLinks();
}

function bindRowLinks() {
  view.querySelectorAll("tr[data-href]").forEach(tr => {
    tr.addEventListener("click", e => { if (!e.target.closest("a")) location.hash = tr.dataset.href; });
    tr.addEventListener("keydown", e => { if (e.key === "Enter") location.hash = tr.dataset.href; });
  });
}

async function renderRun(runId, taskId) {
  const [idx, run] = await Promise.all([get("index.json"), get(`runs/${encodeURIComponent(runId)}.json`)]);
  const s = run.summary;
  const byTask = {};
  for (const t of run.trajectories) (byTask[t.task_id] ||= []).push(t);
  const tasks = idx.suite.tasks;
  let rows = "";
  for (const cat of CATS) {
    const ts = tasks.filter(t => t.category === cat && run.tasks[t.id]);
    if (!ts.length) continue;
    rows += `<tr class="group"><td colspan="6">${CAT_LABEL[cat]} · ${pct(s.pass_rate_by_category[cat])} pass</td></tr>`;
    for (const t of ts) {
      const pt = run.tasks[t.id];
      const trials = (byTask[t.id] || []).sort((a, b) => a.trial - b.trial);
      rows += `<tr class="clickable${taskId === t.id ? " sel" : ""}" tabindex="0" data-href="#/run/${encodeURIComponent(runId)}/${encodeURIComponent(t.id)}">
        <td><a href="#/run/${encodeURIComponent(runId)}/${encodeURIComponent(t.id)}">${esc(t.id)}</a></td>
        <td>${strip(trials.map(x => x.metrics.correct))}</td>
        <td class="num">${pct(pt.pass_rate)}</td><td class="num">${num(pt.tool_calls)}</td>
        <td class="num">${num(pt.invalid_calls)}</td><td class="num">${num(pt.wall_ms / 1000)} s</td></tr>`;
    }
  }
  view.innerHTML = `
  <p><a href="#/runs">All runs</a></p>
  <h2>${swatch(s.model)}${esc(runId)}</h2>
  <p class="intro">${esc(s.model)} · ${{ text: "tool calls parsed from text tags", native: "native function calling",
    "gold-path": "gold path: the harness made the gold tool calls, the model wrote the answer" }[s.agent] || esc(s.agent)} ·
    prompt ${esc(s.prompt)} · temperature ${s.temperature} · ${s.trials} trials per task${s.faults?.length ? ` · injected faults: ${s.faults.map(esc).join(", ")}` : ""}.
    ${esc(run.meta.notes || "")}</p>
  <div class="facts">
    <div class="fact"><b>${pct(s.pass_rate, 1)}</b><span>of ${s.graded} graded trajectories passed</span></div>
    <div class="fact"><b>${num(s.tool_calls_per_task)}</b><span>tool calls per trajectory</span></div>
    <div class="fact"><b>${pct(s.invalid_call_rate, 1)}</b><span>of tool calls were invalid</span></div>
    <div class="fact"><b>${num(s.wall_s_p50)} s</b><span>median time per trajectory (p95 ${num(s.wall_s_p95)} s)</span></div>
  </div>
  <div id="traj"></div>
  <h3>Every task, every trial</h3>
  <p class="muted">A filled square is a passed trial, a hollow one a failed trial. Select a task to read its trajectories.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Task</th><th>Trials</th><th class="num">Pass</th><th class="num">Calls</th><th class="num">Invalid</th><th class="num">Time</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
  bindRowLinks();
  if (taskId) {
    renderTrajectories($("#traj"), tasks.find(t => t.id === taskId), byTask[taskId] || []);
    $("#traj").scrollIntoView({ block: "start" });
  }
}

function renderTrajectories(el, task, trials) {
  const bad = new Set(["unknown_tool", "malformed_args", "schema_error"]);
  el.innerHTML = `<div class="traj">
    <h3>${esc(task.id)}</h3>
    <p><b>Task:</b> ${esc(task.prompt)}</p>
    ${trials.sort((a, b) => a.trial - b.trial).map(t => `
      <div class="trial">
        <div class="trial-head">${strip([t.metrics.correct])}<b>Trial ${t.trial + 1}</b>
          <span>${t.metrics.correct === true ? "passed" : t.metrics.correct === false ? "failed" : "not graded"}${t.metrics.graded_by ? ` (${esc(t.metrics.graded_by)})` : ""}</span>
          <span>${t.metrics.steps} LLM turns, ${t.metrics.tool_calls} tool calls</span>
          <span>${(t.metrics.wall_ms / 1000).toFixed(1)} s</span>
          ${t.metrics.stop_reason !== "answered" ? `<span class="flag suspect">${esc(t.metrics.stop_reason)}</span>` : ""}</div>
        <ul class="calls">${t.tool_calls.map(c => `<li><span class="st ${bad.has(c.status) ? "bad" : ""}">${esc(c.status.replace("_", " "))}</span>${esc(c.name)}(${esc(c.args)})
          <span class="res">${esc(c.result)}</span></li>`).join("") || "<li>No tool calls.</li>"}</ul>
        <div class="answer">${esc(t.final_answer ?? t.error ?? "(no final answer)")}</div>
        <div class="checks">${t.checks.map(c => `${esc(c.type)}: ${c.passed === null ? "judge" : c.passed ? "pass" : "fail"} — ${esc(c.detail)}`).join("<br>")}</div>
      </div>`).join("")}
  </div>`;
}

// ---------------------------------------------------------------- regressions
async function renderDiffs(diffId, mode) {
  const idx = await get("index.json");
  if (!idx.diffs.length) { view.innerHTML = "<h2>Regressions</h2><p>No comparisons exported yet.</p>"; return; }
  diffId ||= idx.diffs[0].id;
  const d = await get(`diffs/${encodeURIComponent(diffId)}.json`);
  const showAll = mode === "all";
  const c = d.counts;
  const order = { regressed: 0, improved: 1, suspect: 2, flaky: 3, unchanged: 4 };
  const rows = d.tasks.filter(r => showAll || ["regressed", "improved", "suspect"].includes(r.status))
    .sort((a, b) => order[a.status] - order[b.status] || a.p_regress - b.p_regress);
  const [runA, runB] = await Promise.all([get(`runs/${encodeURIComponent(d.base.run_id)}.json`), get(`runs/${encodeURIComponent(d.cand.run_id)}.json`)]);
  const trialsOf = (run, tid) => run.trajectories.filter(t => t.task_id === tid).sort((a, b) => a.trial - b.trial).map(t => t.metrics.correct);
  const lo = d.suite.ci[0], hi = d.suite.ci[1];
  const dom = [Math.min(-0.3, lo - 0.05), Math.max(0.3, hi + 0.05)];
  const det = d.detector;
  view.innerHTML = `
  <h2>Regressions</h2>
  <p class="intro">A task counts as regressed only if a one-sided Fisher exact test on its trial counts survives
    Benjamini-Hochberg correction across all tasks (false discovery rate ${d.alpha}). "Suspect" means p &lt; 0.05 before
    correction: worth reading, not enough to fail a build.</p>
  <nav class="picker" aria-label="Comparisons">${idx.diffs.map(x => `<a href="#/diffs/${encodeURIComponent(x.id)}" aria-current="${x.id === diffId}">
    ${esc(x.title)}<small>${esc(x.base)} → ${esc(x.cand)}</small></a>`).join("")}</nav>
  <div class="facts">
    <div class="fact"><b>${signed(d.suite.mean_delta)}</b><span>mean change in per-task pass rate (95% CI ${signed(lo)} to ${signed(hi)})${d.suite.significant ? "" : ", not significant"}</span></div>
    <div class="fact"><b>${c.regressed}</b><span>tasks regressed (BH-corrected)</span></div>
    <div class="fact"><b>${c.raw_p05_regressed}</b><span>with uncorrected p &lt; 0.05</span></div>
    <div class="fact"><b>${c.naive_regressed}</b><span>would be flagged by "pass rate went down"</span></div>
  </div>
  <div class="figure" style="margin-top:16px">
    ${dotWhisker([{ label: "suite pass-rate change", value: d.suite.mean_delta, lo, hi, color: "var(--ink)",
      tip: `mean per-task delta ${signed(d.suite.mean_delta)}<br>95% CI ${signed(lo)} to ${signed(hi)}` }],
      { domain: dom, fmt: v => signed(v, 0), zero: 0, rowH: 34 })}
    <p class="caption">Bootstrap over tasks of the change in per-task pass rate. An interval that crosses zero means the
      suite as a whole did not measurably move.</p>
  </div>
  ${det ? detectorTable(det) : ""}
  <div class="controls">
    <span>${showAll ? `Showing all ${d.tasks.length} tasks.` : `Showing ${rows.length} flagged tasks.`}</span>
    <a href="#/diffs/${encodeURIComponent(diffId)}${showAll ? "" : "/all"}">${showAll ? "Show flagged only" : "Show every task"}</a>
  </div>
  <div class="tablewrap"><table>
    <thead><tr><th>Status</th><th>Task</th><th>${esc(d.base.run_id)} → ${esc(d.cand.run_id)}</th><th class="num">Change</th>
      <th class="num">p</th><th class="num">q (BH)</th></tr></thead>
    <tbody>${rows.map(r => `<tr class="is-${r.status}">
      <td><span class="flag ${r.status}">${r.status}</span></td>
      <td>${esc(r.task_id)}${det && det.affected.includes(r.task_id) ? ' <span class="muted">(fault touches this task)</span>' : ""}</td>
      <td><span class="strips">${strip(trialsOf(runA, r.task_id))}<span class="arrow" aria-hidden="true">→</span>${strip(trialsOf(runB, r.task_id))}</span></td>
      <td class="num">${signed(r.delta, 0)}</td><td class="num">${r.p_regress.toFixed(3)}</td><td class="num">${r.q_regress.toFixed(3)}</td></tr>`).join("")
      || `<tr><td colspan="6">No task was flagged. ${c.naive_regressed} tasks had a lower pass rate, all within trial-to-trial noise.</td></tr>`}</tbody>
  </table></div>`;
}

function detectorTable(det) {
  const rules = [["bh", "Fisher + BH (q ≤ 0.10)"], ["raw_p05", "Fisher, uncorrected p < 0.05"], ["naive", "Any drop in pass rate"]];
  return `<h3>Did the detector find the injected faults?</h3>
  <p class="muted">The faults touch ${det.affected.length} tasks (found by replaying each task's gold trajectory with and
    without the fault). ${det.detectable.length} of them passed at least ${pct(det.min_base_rate)} of baseline trials, so a drop there is
    detectable in principle.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Rule</th><th class="num">Flagged</th><th class="num">Caught (of ${det.detectable.length} detectable)</th>
      <th class="num">False alarms (untouched tasks)</th><th class="num">Precision</th></tr></thead>
    <tbody>${rules.map(([k, name]) => { const x = det[k]; return `<tr><td>${name}</td><td class="num">${x.flagged}</td>
      <td class="num">${x.recall_detectable == null ? "–" : `${Math.round(x.recall_detectable * det.detectable.length)} (${pct(x.recall_detectable)})`}</td>
      <td class="num">${x.false_alarms}</td><td class="num">${pct(x.precision)}</td></tr>`; }).join("")}</tbody>
  </table></div>`;
}

// ---------------------------------------------------------------- judge
async function renderJudge(judgeName) {
  let jd;
  try { jd = await get("judge.json"); } catch { view.innerHTML = "<h2>Judge agreement</h2><p>No labelled samples exported yet.</p>"; return; }
  const ag = jd.agreement;
  const src = jd.label_sources;
  const claudeN = src["claude-opus-5.5"] || Object.entries(src).filter(([k]) => k.startsWith("claude")).reduce((a, [, v]) => a + v, 0);
  const humanN = Object.entries(src).filter(([k]) => !k.startsWith("claude") && k !== "unlabelled").reduce((a, [, v]) => a + v, 0);
  judgeName ||= [...ag].sort((a, b) => (b.dev?.kappa ?? -1) - (a.dev?.kappa ?? -1))[0]?.judge;
  const sel = ag.find(a => a.judge === judgeName);
  const chartRows = [];
  for (const a of ag) for (const split of ["dev", "test"]) if (a[split]) chartRows.push({
    label: `${a.judge} · ${split}`, value: a[split].kappa, lo: a[split].kappa_ci[0], hi: a[split].kappa_ci[1],
    color: split === "dev" ? "var(--faint)" : "var(--model-1)",
    tip: `<b>${esc(a.judge)}</b> on ${split} (${a[split].n} samples)<br>kappa ${a[split].kappa.toFixed(2)} (95% CI ${a[split].kappa_ci[0].toFixed(2)} to ${a[split].kappa_ci[1].toFixed(2)})<br>accuracy ${pct(a[split].accuracy)}`,
  });
  const dis = jd.samples.filter(s => s.label && s.verdicts[judgeName] && s.verdicts[judgeName] !== s.label);
  const cm = sp => { const m = sel?.[sp]?.confusion.rows_truth_cols_judge; return m ? `<table class="cm"><caption>${sp} (${sel[sp].n})</caption>
    <tr><th></th><th>judge: pass</th><th>judge: fail</th></tr>
    <tr><th>label: pass</th><td>${m[0][0]}</td><td>${m[0][1]}</td></tr>
    <tr><th>label: fail</th><td>${m[1][0]}</td><td>${m[1][1]}</td></tr></table>` : ""; };
  view.innerHTML = `
  <h2>Judge agreement</h2>
  <p class="intro">${jd.samples.length} open-ended answers drawn from the runs, each labelled pass or fail against its task
    rubric, then graded by every judge configuration. Prompts were tuned on the dev tasks only; the test tasks were held out.</p>
  <div class="note"><b>Who labelled these:</b> ${claudeN} labels were written by Claude (an LLM) as a first pass and are
    awaiting human review; ${humanN} have been reviewed by a person with <code>trajeval label</code>. Until review is
    complete, read the kappa below as judge agreement with Claude's labels, not with human judgement.</div>
  <div class="figure">
    <div class="legend"><span><span class="sw" style="background:var(--faint)"></span>dev tasks (used for prompt tuning)</span>
      <span><span class="sw" style="background:var(--model-1)"></span>held-out test tasks</span></div>
    ${dotWhisker(chartRows, { domain: [-0.2, 1], fmt: v => v.toFixed(1), zero: 0, labelW: 230 })}
    <p class="caption">Cohen's kappa against the labels, with 95% bootstrap intervals. 0 is chance agreement, 1 is perfect.</p>
  </div>
  <h3>Every judge configuration</h3>
  <div class="tablewrap"><table>
    <thead><tr><th>Judge</th><th>Split</th><th class="num">n</th><th class="num">Kappa</th><th class="num">Accuracy</th>
      <th class="num">Judge pass rate</th><th class="num">Label pass rate</th><th class="num">Unparsed</th></tr></thead>
    <tbody>${ag.flatMap(a => ["dev", "test"].filter(sp => a[sp]).map(sp => `<tr class="clickable" tabindex="0" data-href="#/judge/${encodeURIComponent(a.judge)}">
      <td><a href="#/judge/${encodeURIComponent(a.judge)}">${esc(a.judge)}</a></td><td>${sp}</td><td class="num">${a[sp].n}</td>
      <td class="num"><b>${a[sp].kappa.toFixed(2)}</b></td><td class="num">${pct(a[sp].accuracy)}</td>
      <td class="num">${pct(a[sp].judge_pass_rate)}</td><td class="num">${pct(a[sp].label_pass_rate)}</td><td class="num">${a[sp].unparsed}</td></tr>`)).join("")}</tbody>
  </table></div>
  ${sel ? `<h3>${esc(judgeName)}: where it disagrees with the labels</h3>
    <div class="cms">${cm("dev")}${cm("test")}</div>
    <div class="tablewrap" style="margin-top:12px"><table>
      <thead><tr><th>Sample</th><th>Label</th><th>Judge</th><th>Task</th><th>Answer</th></tr></thead>
      <tbody>${dis.slice(0, 60).map(s => `<tr><td>${esc(s.sample_id)}<br><span class="muted">${esc(s.split)}</span></td>
        <td>${esc(s.label)}<br><span class="muted">${esc(s.labeled_by || "")}</span></td><td>${esc(s.verdicts[judgeName])}</td>
        <td>${esc(s.task_id)}</td><td class="wrap-ok">${esc(s.answer.slice(0, 420))}${s.answer.length > 420 ? "…" : ""}
        ${s.rationale ? `<br><span class="muted">Label note: ${esc(s.rationale)}</span>` : ""}</td></tr>`).join("")
        || '<tr><td colspan="5">No disagreements.</td></tr>'}</tbody></table></div>` : ""}`;
  bindRowLinks();
}

// ---------------------------------------------------------------- about
async function renderAbout() {
  const idx = await get("index.json");
  const bc = idx.suite.by_category;
  view.innerHTML = `
  <h2>How it works</h2>
  <p class="intro">The agent works inside a fictional company, Kestrel Cycles, so it cannot answer from memory: every fact
    lives in a tool. Each attempt gets a fresh copy of the world, and the environment, not the agent, records every tool
    call, so any agent framework is scored the same way.</p>
  <h3>The suite: ${idx.suite.n_tasks} tasks</h3>
  <dl class="defs">${CATS.map(c => `<dt>${CAT_LABEL[c]} (${bc[c] || 0})</dt><dd>${{
    lookup: "One fact from one tool: a SQL count, a policy in a document, a CEO's name on a web page.",
    multi_step: "Two to eight calls across tools: join tables, read invoices then convert currencies, search then fetch.",
    action: "Change the world: write a report file, send an email, schedule a meeting. Graded on the final world state.",
    trap: "No right answer exists (a missing order, a file that isn't there), or the user said not to act.",
    open_ended: "Summaries, replies and briefs graded by an LLM judge against a per-task rubric and reference answer.",
  }[c]}</dd>`).join("")}</dl>
  <h3>What a trajectory is scored on</h3>
  <dl class="defs">
    <dt>Correct</dt><dd>Deterministic checks on the answer and the world (numbers within tolerance, required strings, files, emails, meetings), or the judge for open-ended tasks.</dd>
    <dt>Invalid calls</dt><dd>Hallucinated tool names, arguments that break the JSON schema, and calls that could not be parsed. Tool errors on valid calls (a missing file) are counted separately.</dd>
    <dt>Redundant calls</dt><dd>The same tool with the same arguments, with no side effect in between: the answer cannot have changed.</dd>
    <dt>Loops</dt><dd>The same call issued three or more times in one trajectory.</dd>
    <dt>Ran past its turn</dt><dd>The model kept writing after its tool call and invented the user's reply or the tool's result. The text agent cuts the turn at that point (the invention is never fed back) and counts it.</dd>
    <dt>Efficiency</dt><dd>(gold calls + 1) / (calls + 1), capped at 1, on passed trajectories.</dd>
    <dt>Invented recipients</dt><dd>Email or meeting addresses that never appeared in the prompt or an earlier tool result. The call is schema-valid and succeeds; the email goes to a stranger.</dd>
    <dt>Unrequested side effects</dt><dd>Emails, files or meetings created on a task that did not ask for one.</dd>
    <dt>Cost</dt><dd>Prompt and completion tokens and wall-clock time per trajectory.</dd>
  </dl>
  <h3>Every task</h3>
  <div class="tablewrap"><table><thead><tr><th>Task</th><th>Category</th><th>Prompt</th></tr></thead>
    <tbody>${idx.suite.tasks.map(t => `<tr><td>${esc(t.id)}</td><td>${CAT_LABEL[t.category]}</td><td class="wrap-ok">${esc(t.prompt)}</td></tr>`).join("")}</tbody></table></div>`;
}

// ---------------------------------------------------------------- boot
(async () => {
  try {
    const idx = await get("index.json");
    let slot = 1;
    for (const r of idx.runs) if (!MODEL_SLOT[r.model]) MODEL_SLOT[r.model] = Math.min(slot++, 3);
    $("#generated").textContent = `Data exported ${idx.generated.replace("T", " ").replace("+00:00", " UTC")}.`;
  } catch { /* route() reports it */ }
  route();
})();
