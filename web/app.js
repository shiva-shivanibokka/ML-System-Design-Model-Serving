/*
 * Control panel for the shadow-deployment serving API.
 *
 * No framework and no build step: the page is served as static files by the
 * same FastAPI process that serves the API, so there is one container, one
 * origin, no CORS, and nothing in the deployment that can expire.
 *
 * Refresh policy is deliberate. There is no always-on timer, because an open
 * tab polling forever keeps a scale-to-zero instance awake and spends the free
 * tier on nobody. Data refreshes when you switch tab, when you act, and when
 * you ask; "Live" is opt-in and stops itself after two minutes.
 */

"use strict";

const STAGES = ["loading_v1", "warming_v1", "loading_v2", "warming_v2", "ready"];

const STAGE_TEXT = {
  not_started: "Waking the container…",
  loading_v1: "Loading model v1 — DistilBERT, fp32",
  warming_v1: "Warming v1 — running JIT passes",
  loading_v2: "Loading model v2 and applying int8 quantisation",
  warming_v2: "Warming v2 — running JIT passes",
  ready: "Ready",
  failed: "Model load failed",
};

// Mirrors PROGRESSION_ORDER in deployment/state_machine.py. rolled_back is
// deliberately absent: it is not a position on the path, it is having left it.
const LANE = ["shadow", "canary_5", "canary_25", "canary_50", "full"];
const LANE_LABEL = {
  shadow: "Shadow",
  canary_5: "Canary 5",
  canary_25: "Canary 25",
  canary_50: "Canary 50",
  full: "Full",
};

const $ = (id) => document.getElementById(id);
const pct = (x) => `${(x * 100).toFixed(2)}%`;
const fixed = (x, n = 1) => (x === null || x === undefined ? "—" : Number(x).toFixed(n));

let liveTimer = null;
let liveStopAt = 0;
let lastPredictText = "";

/* ───────────────────────── fetch helpers ───────────────────────── */

async function api(path, opts) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok && res.status !== 503) {
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return body;
}

function toast(message, bad) {
  const el = document.createElement("div");
  el.className = "toast" + (bad ? " bad" : "");
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3200);
}

/* ───────────────────────── boot sequence ───────────────────────── */

/*
 * Waits out the cold start. This is the only unbounded poll in the page, and
 * it is bounded in practice: it runs once, on arrival, and stops the moment
 * the service reports ready or failed.
 *
 * It polls /health rather than /ready. Both carry the load stage, but /ready
 * correctly answers 503 until warm-up finishes, and a browser logs every
 * non-2xx fetch as a console error — so polling it paints the console red for
 * seventy seconds of entirely normal startup. /health is the liveness probe:
 * 200 for as long as the process is alive, with models_stage inside. /ready
 * keeps its meaning for orchestrators, and is consulted here only to retrieve
 * the exception text when a load has actually failed.
 */
async function boot() {
  const started = Date.now();
  for (;;) {
    let h;
    try {
      h = await api("/health");
    } catch (e) {
      $("bootStage").textContent = "Cannot reach the service";
      $("bootStage").classList.add("failed");
      $("bootSub").textContent = String(e.message || e);
      await sleep(3000);
      continue;
    }

    const stage = h.models_stage || "not_started";
    $("bootStage").textContent = STAGE_TEXT[stage] || stage;

    const elapsed = Math.round((Date.now() - started) / 1000);
    if (stage !== "not_started" && stage !== "failed") {
      $("bootSub").textContent = `${elapsed}s elapsed · the page becomes usable as soon as both models are warm.`;
    }

    const at = STAGES.indexOf(stage);
    document.querySelectorAll("#bootSteps i").forEach((el) => {
      const i = STAGES.indexOf(el.dataset.stage);
      el.className = i < at ? "done" : i === at ? "now" : "";
    });

    if (stage === "failed") {
      $("bootStage").classList.add("failed");
      $("bootStage").textContent = "Model load failed";
      const detail = await api("/ready").catch(() => ({}));
      $("bootSub").textContent = detail.load_error || "No error detail returned.";
      return;
    }

    if (stage === "ready") {
      $("boot").remove();
      $("shell").hidden = false;
      await refreshAll();
      return;
    }

    await sleep(2000);
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ───────────────────────── shared chrome ───────────────────────── */

async function refreshChrome() {
  const [health, status] = await Promise.all([
    api("/health"),
    api("/deployment/status"),
  ]);

  $("lampReady").className = "lamp " + (health.status === "ok" ? "on" : "fault");
  $("lampStateText").textContent = status.state.replace(/_/g, " ");
  $("lampState").className = "lamp " + (status.state === "shadow" ? "" : "hot");

  const brk = status.circuit_breaker || {};
  const open = brk.state && brk.state !== "closed";
  $("lampBreakerText").textContent = brk.state || "—";
  $("lampBreaker").className = "lamp " + (open ? "fault" : "on");

  $("lampCacheText").textContent = health.cache_backend || "none";
  $("lampCache").className = "lamp " + (health.cache_backend === "none" ? "" : "on");

  $("uptime").textContent = formatUptime(health.uptime_seconds);
  $("lastRefresh").textContent = new Date().toLocaleTimeString();

  // Stage lane. rolled_back is off the path entirely, so no step is lit and
  // the lane says so rather than silently showing nothing.
  const at = LANE.indexOf(status.state);
  const share = Math.round((status.v2_traffic_fraction ?? 0) * 100);
  document.querySelectorAll("#stageLane .step").forEach((el) => {
    const i = LANE.indexOf(el.dataset.state);
    el.className = "step " + (i < at ? "done" : i === at ? "now" : "");
  });
  $("stageLane").classList.toggle("aborted", status.state === "rolled_back");
  $("lanePos").textContent =
    at >= 0
      ? `stage ${at + 1} of ${LANE.length} · ${share}% to v2`
      : status.state === "rolled_back"
        ? "rolled back — off the progression, all traffic on v1"
        : status.state;

  const ephemeral = status.state_durability === "ephemeral";
  $("hazard").hidden = !ephemeral;
  $("durabilityStamp").textContent = ephemeral
    ? "State ephemeral · resets on restart"
    : "State persisted to disk";

  return { health, status };
}

function formatUptime(s) {
  if (s === undefined || s === null) return "—";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

/* ───────────────────────── predict ───────────────────────── */

async function runPredict(text) {
  const body = text ?? $("predictText").value;
  if (!body.trim()) return toast("Enter some text first", true);

  $("btnPredict").disabled = true;
  try {
    const r = await api("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: body }),
    });
    lastPredictText = body;
    renderPrediction(r);
    await refreshChrome();
  } catch (e) {
    toast(e.message, true);
  } finally {
    $("btnPredict").disabled = false;
  }
}

function renderPrediction(r) {
  const negative = r.label.toUpperCase().startsWith("NEG");
  $("predictResult").innerHTML = `
    <div class="result">
      <div class="label ${negative ? "neg" : ""}">${esc(r.label)}</div>
      <div class="score">confidence ${fixed(r.score, 4)}</div>
      <div class="conf"><i style="width:${(r.score * 100).toFixed(1)}%;background:${
        negative ? "var(--alarm)" : "var(--amber)"
      }"></i></div>
      <div class="meta">
        <span>served by <b>${esc(r.model_used)}</b></span>
        <span>version <b>${esc(r.model_version)}</b></span>
        <span>latency <b>${fixed(r.latency_ms, 2)} ms</b></span>
        <span>cache <b>${r.cache_hit ? "hit" : "miss"}</b></span>
      </div>
    </div>`;

  $("predictDetail").innerHTML = `<tbody>
    ${row("label", r.label)}
    ${row("score", fixed(r.score, 6))}
    ${row("model_used", r.model_used)}
    ${row("model_version", r.model_version)}
    ${row("deployment_state", r.deployment_state)}
    ${row("latency_ms", fixed(r.latency_ms, 2))}
    ${row("cache_hit", String(r.cache_hit))}
    ${row("trace_id", r.trace_id)}
  </tbody>`;
}

const row = (k, v) => `<tr><td class="k">${esc(k)}</td><td class="n">${esc(String(v))}</td></tr>`;

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

/* ───────────────────────── rollout ───────────────────────── */

async function refreshRollout() {
  const status = await api("/deployment/status");
  const t = status.rollback_thresholds || {};
  const share = Math.round((status.v2_traffic_fraction ?? 0) * 100);
  const aborted = status.state === "rolled_back";

  $("rolloutReadouts").innerHTML = `
    ${ro("Traffic to v2", `${share}<small>%</small>`, status.state.replace(/_/g, " "), aborted ? "alarm" : "")}
    ${ro("Requests seen", fmtInt(status.total_requests), `${fmtInt(status.v2_requests)} answered by v2`)}
    ${ro("Time in state", formatUptime(status.time_in_state_seconds), "auto-progression " + (status.auto_progression_enabled ? "on" : "off"))}
    ${ro("v2 error rate", pct(status.v2_error_rate ?? 0), `limit ${pct(t.error_rate ?? 0.05)}`, (status.v2_error_rate ?? 0) >= (t.error_rate ?? 0.05) ? "alarm" : "ok")}
  `;

  // A 5% slice is too narrow to hold "v2 — 5%", and a clipped label reads as a
  // rendering fault. Below 12% the segment carries no text and the caption
  // underneath states the split instead.
  const seg = (cls, w, text) =>
    `<i class="${cls}" style="width:${w}%">${w >= 12 ? text : ""}</i>`;

  $("splitBar").innerHTML = aborted
    ? `<i class="a" style="width:100%">rolled back — v1 serving 100%</i>`
    : share === 0
      ? `<i class="a" style="width:100%">v1 — 100% · v2 shadowing every request</i>`
      : seg("a", 100 - share, `v1 — ${100 - share}%`) + seg("b", share, `v2 — ${share}%`);

  const at = LANE.indexOf(status.state);
  const next = at >= 0 && at < LANE.length - 1 ? LANE[at + 1] : null;
  $("splitCaption").textContent = aborted
    ? "Promote returns the machine to shadow and starts the progression again."
    : next
      ? `v1 ${100 - share}% · v2 ${share}% — promote advances to ${LANE_LABEL[next]}`
      : "Fully deployed. v2 is serving every request.";

  const brk = status.circuit_breaker || {};
  const bt = brk.thresholds || {};
  $("breakerTable").innerHTML = `<tbody>
    ${row("state", brk.state ?? "—")}
    ${row("failure count", `${brk.failure_count ?? 0} / ${bt.failure_threshold ?? "—"}`)}
    ${row("failure rate", pct(brk.failure_rate ?? 0))}
    ${row("calls / failures", `${fmtInt(brk.total_calls)} / ${fmtInt(brk.total_failures)}`)}
    ${row("blocked calls", fmtInt(brk.total_blocked))}
    ${row("reopen timeout", `${bt.timeout_seconds ?? "—"} s`)}
  </tbody>`;

  // v2 latency is judged relative to v1, not against a fixed ceiling: the
  // limit is v1's own p99 times latency_multiplier.
  const mult = t.latency_multiplier ?? 2.0;
  const v1p99 = status.v1_p99_latency_ms ?? 0;
  $("thresholdTable").querySelector("tbody").innerHTML = [
    thr("v2 error rate", status.v2_error_rate ?? 0, t.error_rate ?? 0.05, "rate"),
    thr(
      `v2 p99 latency (≤ ${mult}× v1)`,
      status.v2_p99_latency_ms ?? 0,
      v1p99 * mult,
      "ms"
    ),
  ].join("");
}

function thr(name, measured, limit, kind) {
  const fmt = (v) => (kind === "rate" ? pct(v) : `${fixed(v, 1)} ms`);
  // A limit of zero means the baseline has not been measured yet (no traffic
  // through v1). Reporting "Pass" then would be a false all-clear.
  if (!limit) {
    return `<tr><td>${esc(name)}</td><td class="n">${fmt(measured)}</td>
      <td class="n">—</td>
      <td class="n"><span class="flag watch">No baseline</span></td></tr>`;
  }
  const ratio = measured / limit;
  const flag = ratio >= 1 ? "alarm" : ratio >= 0.6 ? "watch" : "ok";
  const label = ratio >= 1 ? "Breach" : ratio >= 0.6 ? "Watch" : "Pass";
  return `<tr><td>${esc(name)}</td><td class="n">${fmt(measured)}</td>
    <td class="n">${fmt(limit)}</td>
    <td class="n"><span class="flag ${flag}">${label}</span></td></tr>`;
}

function ro(k, v, d, cls) {
  return `<div class="ro"><div class="k">${esc(k)}</div>
    <div class="v ${cls || ""}">${v}</div><div class="d">${esc(d || "")}</div></div>`;
}

const fmtInt = (n) => Number(n || 0).toLocaleString("en-US");

async function act(path, label) {
  try {
    const r = await api(path, { method: "POST" });
    if (r.ok === false) {
      toast(r.reason || `${label} refused`, true);
    } else {
      const moved = r.from_state && r.to_state ? ` ${r.from_state} → ${r.to_state}` : "";
      toast(`${label}${moved}${r.cache_flushed ? " · cache flushed" : ""}`);
    }
    await refreshChrome();
    await refreshRollout();
  } catch (e) {
    toast(e.message, true);
  }
}

/* ───────────────────────── disagreement ───────────────────────── */

async function refreshDisagreement() {
  const [s, recent] = await Promise.all([
    api("/monitoring/disagreement"),
    api("/monitoring/disagreement/recent?n=20"),
  ]);

  const alerting = s.disagreement_rate >= s.alert_threshold;
  $("disagreeReadouts").innerHTML = `
    ${ro("Disagreement rate", pct(s.disagreement_rate), `${s.disagreements_in_window} of ${s.window_size} · alerts at ${pct(s.alert_threshold)}`, alerting ? "alarm" : "")}
    ${ro("Comparisons", fmtInt(s.total_comparisons), "both models scored")}
    ${ro("Mean gap, all", fixed(s.avg_confidence_gap_all, 3), "confidence difference", "plain")}
    ${ro("Mean gap, on disagreement", fixed(s.avg_confidence_gap_on_disagreements, 3), "higher means confidently split", "plain")}
  `;

  $("scopeWindow").textContent = `${s.window_size} req window`;

  const cases = recent.recent_disagreements || [];
  drawScope(cases, s);

  const dir = s.direction_breakdown || {};
  const dirRows = Object.keys(dir).length
    ? Object.entries(dir).map(([k, v]) => row(k.replace(/_/g, " → "), fmtInt(v))).join("")
    : `<tr><td class="k">No disagreements recorded yet</td><td class="n">—</td></tr>`;
  $("directionTable").innerHTML = `<tbody>${dirRows}</tbody>`;

  // The API returns the labels, scores and input length but deliberately not
  // the input text itself, so nothing a user typed is echoed back out of the
  // monitoring endpoint.
  $("disagreeList").innerHTML = cases.length
    ? cases
        .map(
          (c) => `<div class="item">
            <div class="top">
              <span class="trig">v1 ${esc(c.v1_label ?? "—")} → v2 ${esc(c.v2_label ?? "—")}</span>
              <time>gap ${fixed(c.confidence_gap, 3)}</time>
            </div>
            <div class="note">
              v1 scored ${fixed(c.v1_score, 4)}, v2 scored ${fixed(c.v2_score, 4)}
              on ${fmtInt(c.input_length)} characters
            </div>
          </div>`
        )
        .join("")
    : `<div class="empty">No disagreements yet. Both models score every request, so cases
       accumulate as the service is used — try the borderline samples on the Predict tab.</div>`;
}

function drawScope(cases, stats) {
  const svg = $("disagreeScope");
  const W = 720;
  const H = 180;
  const grid = [];
  for (let y = 36; y < H; y += 36) grid.push(`<line x1="0" y1="${y}" x2="${W}" y2="${y}"/>`);
  for (let x = 90; x < W; x += 90) grid.push(`<line x1="${x}" y1="0" x2="${x}" y2="${H}"/>`);

  const yOf = (v) => H - Math.max(0, Math.min(1, v)) * (H - 16) - 8;

  let trace = "";
  let points = "";
  if (cases.length) {
    const step = cases.length > 1 ? W / (cases.length - 1) : W;
    const pts = cases.map((c, i) => [i * step, yOf(c.confidence_gap ?? 0)]);
    trace = `<path fill="none" stroke="#F0A22E" stroke-width="1.8" d="M${pts
      .map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`)
      .join(" L")}"/>`;
    points = pts
      .map((p) => `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="2.6" fill="#F0A22E"/>`)
      .join("");
  }

  const meanY = yOf(stats.avg_confidence_gap_all ?? 0);

  svg.innerHTML = `
    <g stroke="#241D12" stroke-width="1">${grid.join("")}</g>
    <line x1="0" y1="${meanY}" x2="${W}" y2="${meanY}" stroke="#7E5715" stroke-width="1.4" stroke-dasharray="5 5"/>
    ${trace}${points}
    ${
      cases.length
        ? ""
        : `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="#7E5715"
             font-family="ui-monospace, monospace" font-size="12" letter-spacing="2">NO SIGNAL</text>`
    }`;
}

/* ───────────────────────── drift ───────────────────────── */

async function refreshDrift() {
  const [d, hist] = await Promise.all([
    api("/monitoring/drift"),
    api("/monitoring/drift/history"),
  ]);

  const last = d.last_check;
  const th = d.thresholds || {};
  const frozen = d.reference_window_frozen;

  $("driftReadouts").innerHTML = `
    ${ro("Reference window", frozen ? "frozen" : "filling", `${fmtInt(d.reference_size)} records`, frozen ? "ok" : "plain")}
    ${ro("Records seen", fmtInt(d.total_records), `${fmtInt(d.checks_run)} checks run`, "plain")}
    ${ro(
      "Text length drift",
      last ? fixed(last.text_length_drift_score, 3) : "—",
      `limit ${fixed(th.text_length ?? 0.1, 2)}`,
      last ? (last.text_length_drifted ? "alarm" : "ok") : "plain"
    )}
    ${ro(
      "Confidence drift",
      last ? fixed(last.confidence_drift_score, 3) : "—",
      `limit ${fixed(th.confidence ?? 0.1, 2)}`,
      last ? (last.confidence_drifted ? "alarm" : "ok") : "plain"
    )}
  `;

  const history = hist.history || [];
  $("driftMeter").innerHTML = history.length
    ? history
        .slice(-8)
        .map((h) => {
          const bar = (label, score, limit, drifted) => {
            const w = Math.min(100, limit ? (score / limit) * 100 : 0);
            return `<div class="meter-row">
              <span class="lab">${esc(label)}</span>
              <span class="track"><i class="${drifted ? "alarm" : "ok"}" style="width:${w.toFixed(0)}%"></i></span>
              <span class="n">${fixed(score, 3)}</span>
            </div>`;
          };
          return (
            bar(`n=${h.checked_at_request_n} len`, h.text_length_drift_score, th.text_length ?? 0.1, h.text_length_drifted) +
            bar(`n=${h.checked_at_request_n} conf`, h.confidence_drift_score, th.confidence ?? 0.1, h.confidence_drifted)
          );
        })
        .join("")
    : `<div class="empty">No drift checks yet. The reference window fills and freezes first; each
       later batch is then compared against it.</div>`;

  $("driftTable").innerHTML = `<tbody>
    ${row("reference frozen", String(frozen))}
    ${row("reference size", fmtInt(d.reference_size))}
    ${row("total records", fmtInt(d.total_records))}
    ${row("checks run", fmtInt(d.checks_run))}
    ${last ? row("method", last.method) : ""}
    ${last ? row("checked at request", fmtInt(last.checked_at_request_n)) : ""}
    ${last ? row("any drift", String(last.any_drift)) : ""}
  </tbody>`;
}

/* ───────────────────────── cache ───────────────────────── */

async function refreshCache() {
  const c = await api("/monitoring/cache");
  const total = (c.hits || 0) + (c.misses || 0);

  $("cacheReadouts").innerHTML = `
    ${ro("Backend", esc(c.backend), c.available ? "answering" : "unavailable", c.backend === "none" ? "alarm" : "")}
    ${ro("Hit rate", pct(c.hit_rate || 0), `${fmtInt(total)} lookups`, "ok")}
    ${ro("Hits", fmtInt(c.hits), "inference skipped", "plain")}
    ${ro("Misses", fmtInt(c.misses), `${fmtInt(c.errors)} errors`, "plain")}
  `;

  const hit = (c.hit_rate || 0) * 100;
  $("cacheMeter").innerHTML = `
    <div class="meter-row"><span class="lab">Hits</span>
      <span class="track"><i class="ok" style="width:${hit.toFixed(1)}%"></i></span>
      <span class="n">${fmtInt(c.hits)}</span></div>
    <div class="meter-row"><span class="lab">Misses</span>
      <span class="track"><i style="width:${(100 - hit).toFixed(1)}%"></i></span>
      <span class="n">${fmtInt(c.misses)}</span></div>
    <div class="meter-row"><span class="lab">Errors</span>
      <span class="track"><i class="alarm" style="width:${total ? ((c.errors || 0) / total) * 100 : 0}%"></i></span>
      <span class="n">${fmtInt(c.errors)}</span></div>`;
}

/* ───────────────────────── audit ───────────────────────── */

async function refreshAudit() {
  const a = await api("/deployment/audit");
  const entries = (a.entries || []).slice().reverse();

  $("auditList").innerHTML = entries.length
    ? entries
        .map(
          (e) => `<div class="item">
            <div class="top">
              <time>${esc(fmtTime(e.timestamp))}</time>
              <span class="trig">${esc(e.trigger || "—")}</span>
              <span class="arrow">${esc(e.from_state || "—")} → ${esc(e.to_state || "—")}</span>
            </div>
            <div class="note">
              error rate ${pct(e.v2_error_rate_at_event ?? 0)} ·
              v2 p99 ${fixed(e.v2_p99_latency_ms, 1)} ms ·
              v1 p99 ${fixed(e.v1_p99_latency_ms, 1)} ms ·
              ${fmtInt(e.requests_seen)} requests${e.note ? ` · ${esc(e.note)}` : ""}
            </div>
          </div>`
        )
        .join("")
    : `<div class="empty">No transitions recorded.</div>`;
}

function fmtTime(ts) {
  if (!ts) return "—";
  const d = new Date(ts);
  return isNaN(d) ? ts : d.toLocaleTimeString();
}

/* ───────────────────────── wiring ───────────────────────── */

const VIEWS = {
  predict: async () => {},
  rollout: refreshRollout,
  disagreement: refreshDisagreement,
  drift: refreshDrift,
  cache: refreshCache,
  audit: refreshAudit,
};

let current = "predict";

async function refreshAll() {
  try {
    await refreshChrome();
    await VIEWS[current]();
  } catch (e) {
    toast(e.message, true);
  }
}

$("tabs").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-view]");
  if (!btn) return;
  current = btn.dataset.view;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b === btn));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("on", v.id === `view-${current}`));
  await refreshAll();
});

$("samples").addEventListener("click", (ev) => {
  const b = ev.target.closest("button[data-t]");
  if (b) $("predictText").value = b.dataset.t;
});

$("btnPredict").addEventListener("click", () => runPredict());
$("btnPredictAgain").addEventListener("click", () =>
  runPredict(lastPredictText || $("predictText").value)
);
$("btnPromote").addEventListener("click", () => act("/deployment/promote", "Promoted"));
$("btnRollback").addEventListener("click", () => act("/deployment/rollback", "Rolled back"));
$("btnResetBreaker").addEventListener("click", () => act("/circuit-breaker/reset", "Breaker reset"));
$("btnRefresh").addEventListener("click", refreshAll);

/*
 * Live mode stops itself after two minutes. An operator watching a rollout
 * wants a few minutes of live numbers; nobody wants a forgotten tab holding a
 * scale-to-zero container awake all night.
 */
$("btnLive").addEventListener("click", () => {
  if (liveTimer) {
    clearInterval(liveTimer);
    liveTimer = null;
    $("btnLive").textContent = "Live: off";
    return;
  }
  liveStopAt = Date.now() + 120000;
  liveTimer = setInterval(() => {
    if (Date.now() > liveStopAt) {
      clearInterval(liveTimer);
      liveTimer = null;
      $("btnLive").textContent = "Live: off";
      toast("Live refresh stopped after 2 minutes");
      return;
    }
    refreshAll();
  }, 5000);
  $("btnLive").textContent = "Live: on";
  refreshAll();
});

boot();
