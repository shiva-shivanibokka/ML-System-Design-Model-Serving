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

// Written for someone who has never deployed anything. The technical stage
// names are still visible on /ready for anyone who wants them.
const STAGE_TEXT = {
  not_started: "Waking the server…",
  loading_v1: "Loading the current model into memory",
  warming_v1: "Warming it up — the first few runs are always slowest",
  loading_v2: "Loading the new, compressed model",
  warming_v2: "Warming that one up too",
  ready: "Ready",
  failed: "Something went wrong starting up",
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

/*
 * Sample sentences, deliberately spread across domains — product reviews,
 * restaurants, film, books, support tickets, social posts — rather than one
 * kind of text. A model that only ever sees product reviews here looks more
 * reliable than it is; the sarcasm and mixed-feeling sets are where the two
 * models actually have room to disagree, which is the point of the demo.
 */
const SENTENCES = {
  negative: [
    "The battery died after two days and support never replied.",
    "Waited forty minutes for a table and the food arrived cold.",
    "The plot dragged for two hours and then simply stopped.",
    "Third outage this month — I am moving my account elsewhere.",
    "The instructions were wrong and I had to take the whole thing apart again.",
    "Nobody at the counter would look up, let alone help.",
    "It arrived cracked, and the return form does not load.",
  ],
  positive: [
    "Absolutely brilliant — better than I expected for the price.",
    "The staff remembered my name and the coffee was perfect.",
    "I finished the whole book in one sitting and started it again the next day.",
    "Setup took four minutes and it has not crashed once.",
    "Best decision I have made all year, genuinely.",
    "The engineer called back within the hour and fixed it on the spot.",
    "Every seat in the room was still full at the end of the last talk.",
  ],
  mixed: [
    "It works, I suppose, though I would not buy it again.",
    "Beautiful hotel, terrible breakfast, and the wifi barely worked.",
    "Great acting, but the script gave them almost nothing to do.",
    "Fast delivery. Wrong item.",
    "The app is fine once you get past the setup, which is awful.",
    "Cheap, cheerful, and it will probably fall apart by spring.",
    "I liked the first half enough to forgive the second.",
  ],
  tricky: [
    "Not bad at all, honestly.",
    "Oh good, another update that moves every button.",
    "I can't say I hated it.",
    "Well, that was certainly a decision someone made.",
    "It only broke twice, which for this brand is impressive.",
    "Sure, a two-hour queue is exactly how I wanted to spend the morning.",
    "Nothing about it is memorable, and I mean that kindly.",
  ],
};

const ALL_SENTENCES = Object.values(SENTENCES).flat();

/* Never returns what is already in the box — a random button that appears to
 * do nothing reads as broken. */
function randomSentence(kind) {
  const pool = kind && kind !== "any" ? SENTENCES[kind] : ALL_SENTENCES;
  const current = $("predictText").value.trim();
  const fresh = pool.filter((s) => s !== current);
  const from = fresh.length ? fresh : pool;
  return from[Math.floor(Math.random() * from.length)];
}

const pct = (x) => `${(x * 100).toFixed(2)}%`;
const fixed = (x, n = 1) => (x === null || x === undefined ? "—" : Number(x).toFixed(n));
const fmtInt = (n) => Number(n || 0).toLocaleString("en-US");

let liveTimer = null;
let liveStopAt = 0;
let lastPredictText = "";

/* ───────────────────────── fetch helpers ───────────────────────── */

/*
 * A 503 means the models are not loaded — either the very first start, or this
 * tab outliving the container and landing on a cold one.
 *
 * It has to throw. It once did not: /ready answers 503 by design during warm-up
 * and the boot poll wants that body, so 503 was exempted for every caller. That
 * handed /predict's error payload to the success path, where reading .label off
 * {detail: "Models not ready"} threw a TypeError and Classify appeared to do
 * nothing. Only the boot poll opts out, and it says so.
 */
async function api(path, opts, { allow503 = false } = {}) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (res.status === 503 && !allow503) {
    const err = new Error(body.detail || "The server is still starting up.");
    err.notReady = true;
    throw err;
  }
  if (!res.ok && !(res.status === 503 && allow503)) {
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

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ───────────────────────── boot sequence ───────────────────────── */

/*
 * Waits out the cold start. This is the only unbounded poll in the page, and
 * it is bounded in practice: it runs once, on arrival, and stops the moment
 * the service reports ready or failed.
 *
 * It polls /health rather than /ready. Both carry the load stage, but /ready
 * correctly answers 503 until warm-up finishes, and a browser logs every
 * non-2xx fetch as a console error — so polling it paints the console red for
 * a minute of entirely normal startup. /health is the liveness probe: 200 for
 * as long as the process is alive, with models_stage inside. /ready keeps its
 * meaning for orchestrators, and is consulted here only to retrieve the
 * exception text when a load has actually failed.
 */
let booting = false;
let wokeFromSleep = false;

const SLEEP_NOTE =
  "The server restarted, so both AI models are being loaded back into memory. That takes about a" +
  " minute. Nothing you did caused this — your sentence has not been lost, and the page reopens" +
  " by itself.";

async function boot() {
  if (booting) return;
  booting = true;
  const started = Date.now();
  try {
    await bootLoop(started);
  } finally {
    booting = false;
  }
}

/*
 * Cloud Run stops the container when nobody is using it, so a tab left open
 * long enough will find the next click landing on a cold instance. Rather than
 * failing that click, put the same starting-up screen back and resume the poll;
 * the shell reopens by itself once the models are loaded.
 */
function wakeUp() {
  const el = $("boot");
  if (!el || !el.hidden) return;
  if (liveTimer) {
    clearInterval(liveTimer);
    liveTimer = null;
    $("btnLive").textContent = "Auto-refresh: off";
  }
  wokeFromSleep = true;
  $("shell").hidden = true;
  el.hidden = false;
  $("bootStage").classList.remove("failed");
  $("bootStage").textContent = "The server restarted — loading the models again";
  $("bootSub").textContent = SLEEP_NOTE;
  boot();
}

async function bootLoop(started) {
  for (;;) {
    let h;
    try {
      h = await api("/health");
    } catch (e) {
      $("bootStage").textContent = "Cannot reach the server";
      $("bootStage").classList.add("failed");
      $("bootSub").textContent = String(e.message || e);
      await sleep(3000);
      continue;
    }

    const stage = h.models_stage || "not_started";
    $("bootStage").textContent = STAGE_TEXT[stage] || stage;

    const elapsed = Math.round((Date.now() - started) / 1000);
    if (stage !== "not_started" && stage !== "failed") {
      // Someone who arrived to a sleeping server needs the explanation to stay
      // on screen, not be replaced by a stopwatch one poll later.
      $("bootSub").textContent = wokeFromSleep
        ? `${SLEEP_NOTE} — ${elapsed} seconds so far.`
        : `${elapsed} seconds so far. The page opens by itself as soon as both models are ready.`;
    }

    const at = STAGES.indexOf(stage);
    document.querySelectorAll("#bootSteps i").forEach((el) => {
      const i = STAGES.indexOf(el.dataset.stage);
      el.className = i < at ? "done" : i === at ? "now" : "";
    });

    if (stage === "failed") {
      $("bootStage").classList.add("failed");
      const detail = await api("/ready", undefined, { allow503: true }).catch(() => ({}));
      $("bootSub").textContent = detail.load_error || "No error detail returned.";
      return;
    }

    if (stage === "ready") {
      // Hidden, not removed: the container can sleep again later and this
      // screen is what explains the wait when it does.
      $("boot").hidden = true;
      $("shell").hidden = false;
      await refreshAll();
      return;
    }

    await sleep(2000);
  }
}

/* ───────────────────────── shared chrome ───────────────────────── */

async function refreshChrome() {
  const [health, status] = await Promise.all([api("/health"), api("/deployment/status")]);

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
  document.querySelectorAll("#stageLane .step-cell").forEach((el) => {
    const i = LANE.indexOf(el.dataset.state);
    el.className = "step-cell " + (i < at ? "done" : i === at ? "now" : "");
  });
  $("stageLane").classList.toggle("aborted", status.state === "rolled_back");
  $("lanePos").textContent =
    at >= 0
      ? `step ${at + 1} of ${LANE.length} — the new model is answering ${share}% of requests`
      : status.state === "rolled_back"
        ? "rolled back — the old model is answering everything"
        : status.state;

  const ephemeral = status.state_durability === "ephemeral";
  $("hazard").hidden = !ephemeral;
  $("durabilityStamp").textContent = ephemeral ? "Resets when the server sleeps" : "Saved to disk";

  return { health, status };
}

function formatUptime(s) {
  if (s === undefined || s === null) return "—";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

/* ───────────────────────── shared renderers ───────────────────────── */

function ro(tone, k, v, d, tip) {
  const q = tip ? ` <button class="q" data-tip="${esc(tip)}">?</button>` : "";
  return `<div class="ro ${tone}"><div class="k">${esc(k)}${q}</div>
    <div class="v">${v}</div><div class="d">${esc(d || "")}</div></div>`;
}

const row = (k, v) => `<tr><td class="k">${esc(k)}</td><td class="n">${esc(String(v))}</td></tr>`;

/* ───────────────────────── predict ───────────────────────── */

async function runPredict(text) {
  const body = text ?? $("predictText").value;
  if (!body.trim()) return toast("Type a sentence first", true);

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
    if (e.notReady) wakeUp();
    else toast(e.message, true);
  } finally {
    $("btnPredict").disabled = false;
  }
}

function renderPrediction(r) {
  const negative = r.label.toUpperCase().startsWith("NEG");
  const colour = negative ? "var(--alarm)" : "var(--ok)";
  $("predictResult").innerHTML = `
    <div class="result">
      <div class="label ${negative ? "neg" : "pos"}">${esc(r.label)}</div>
      <div class="score">${(r.score * 100).toFixed(2)}% confident</div>
      <div class="conf"><i style="width:${(r.score * 100).toFixed(1)}%;background:${colour}"></i></div>
      <div class="meta">
        <span>answered by <b>${esc(r.model_used)}</b></span>
        <span>took <b>${fixed(r.latency_ms, 2)} ms</b></span>
        <span>${r.cache_hit ? "<b>from cache</b> — the model did not run" : "the model ran"}</span>
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

/* ───────────────────────── rollout ───────────────────────── */

async function refreshRollout() {
  const status = await api("/deployment/status");
  const t = status.rollback_thresholds || {};
  const share = Math.round((status.v2_traffic_fraction ?? 0) * 100);
  const aborted = status.state === "rolled_back";
  const errRate = status.v2_error_rate ?? 0;
  const errLimit = t.error_rate ?? 0.05;

  $("rolloutReadouts").innerHTML = `
    ${ro("traffic", "Traffic to the new model", `${share}<small>%</small>`,
      status.state.replace(/_/g, " "),
      "The share of real requests the new model is answering right now.")}
    ${ro("plain", "Requests handled", fmtInt(status.total_requests),
      // In shadow the new model runs on every sentence but answers nobody, so
      // "answered by the new model" here would contradict the 0% split bar
      // sitting directly beneath it.
      status.state === "shadow"
        ? `${fmtInt(status.v2_requests)} scored silently by the new model`
        : `${fmtInt(status.v2_requests)} answered by the new model`,
      "Counts each model run, so in Shadow one sentence counts twice — the old model answers you and the new one scores it silently. The rollout will not act on fewer than 20: a couple of bad requests is noise, not evidence.")}
    ${ro("speed", "Time at this stage", formatUptime(status.time_in_state_seconds),
      "auto-advance " + (status.auto_progression_enabled ? "on" : "off"),
      "How long the new model has held this share of traffic without tripping a limit.")}
    ${ro(errRate >= errLimit ? "alarm" : "ok", "New model failures", pct(errRate),
      `rolls back above ${pct(errLimit)}`,
      "How often the new model errors. Cross the limit and the system rolls itself back without waiting for a human.")}
  `;

  // A 5% slice is too narrow to hold "v2 — 5%", and a clipped label reads as a
  // rendering fault. Below 12% the segment carries no text and the caption
  // underneath states the split instead.
  const seg = (cls, w, text) => `<i class="${cls}" style="width:${w}%">${w >= 12 ? text : ""}</i>`;

  $("splitBar").innerHTML = aborted
    ? `<i class="a" style="width:100%">rolled back — old model answering everything</i>`
    : share === 0
      ? `<i class="a" style="width:100%">old model — 100% · new model watching silently</i>`
      : seg("a", 100 - share, `old — ${100 - share}%`) + seg("b", share, `new — ${share}%`);

  const at = LANE.indexOf(status.state);
  const next = at >= 0 && at < LANE.length - 1 ? LANE[at + 1] : null;
  $("splitCaption").textContent = aborted
    ? "Promote starts the whole process again from the beginning."
    : next
      ? `Promote moves the new model to ${LANE_LABEL[next]}.`
      : "Fully rolled out. The new model is answering everything.";

  const brk = status.circuit_breaker || {};
  const bt = brk.thresholds || {};
  $("breakerTable").innerHTML = `<tbody>
    ${row("state", brk.state ?? "—")}
    ${row("failures in a row", `${brk.failure_count ?? 0} of ${bt.failure_threshold ?? "—"}`)}
    ${row("failure rate", pct(brk.failure_rate ?? 0))}
    ${row("calls / failures", `${fmtInt(brk.total_calls)} / ${fmtInt(brk.total_failures)}`)}
    ${row("requests blocked", fmtInt(brk.total_blocked))}
    ${row("retry after", `${bt.timeout_seconds ?? "—"} s`)}
  </tbody>`;

  // The new model's speed is judged relative to the old one, not against a
  // fixed ceiling: the limit is v1's own p99 times latency_multiplier.
  const mult = t.latency_multiplier ?? 2.0;
  const v1p99 = status.v1_p99_latency_ms ?? 0;
  $("thresholdTable").querySelector("tbody").innerHTML = [
    thr("How often the new model errors", errRate, errLimit, "rate"),
    thr(`How slow it is (limit: ${mult}× the old model)`, status.v2_p99_latency_ms ?? 0, v1p99 * mult, "ms"),
  ].join("");
}

function thr(name, measured, limit, kind) {
  const fmt = (v) => (kind === "rate" ? pct(v) : `${fixed(v, 1)} ms`);
  // A limit of zero means the baseline has not been measured yet (no traffic
  // through v1). Reporting "Pass" then would be a false all-clear.
  if (!limit) {
    return `<tr><td>${esc(name)}</td><td class="n">${fmt(measured)}</td>
      <td class="n">—</td>
      <td class="n"><span class="flag watch">Not measured yet</span></td></tr>`;
  }
  const ratio = measured / limit;
  const flag = ratio >= 1 ? "alarm" : ratio >= 0.6 ? "watch" : "ok";
  const label = ratio >= 1 ? "Breach" : ratio >= 0.6 ? "Watch" : "Pass";
  return `<tr><td>${esc(name)}</td><td class="n">${fmt(measured)}</td>
    <td class="n">${fmt(limit)}</td>
    <td class="n"><span class="flag ${flag}">${label}</span></td></tr>`;
}

async function act(path, label) {
  try {
    const r = await api(path, { method: "POST" });
    if (r.ok === false) {
      toast(r.reason || `${label} not possible`, true);
    } else {
      const moved = r.from_state && r.to_state ? ` ${r.from_state} → ${r.to_state}` : "";
      toast(`${label}${moved}${r.cache_flushed ? " · cache cleared" : ""}`);
    }
    await refreshChrome();
    await refreshRollout();
  } catch (e) {
    if (e.notReady) wakeUp();
    else toast(e.message, true);
  }
}

/* ───────────────────────── comparison ───────────────────────── */

async function refreshComparison() {
  const [s, recent, status] = await Promise.all([
    api("/monitoring/disagreement"),
    api("/monitoring/disagreement/comparisons?n=40"),
    api("/deployment/status"),
  ]);

  /*
   * Comparisons are only recorded in shadow, because that is the only state
   * where both models are run on the same sentence. Everywhere else the router
   * picks one. Without saying so, this tab looks frozen or broken the moment
   * someone promotes — which is exactly when they are watching it hardest.
   */
  const shadowing = status.state === "shadow";
  $("comparisonNotice").innerHTML = shadowing
    ? ""
    : `<div class="notice">
         <b>Not recording right now.</b> Both models are only run on the same sentence during
         <b>Shadow</b>. You are at <b>${esc((status.state || "").replace(/_/g, " "))}</b>, where the
         router sends each request to one model or the other — so the numbers below are frozen at the
         last ${fmtInt(s.total_comparisons)} and will not move until this returns to Shadow.
       </div>`;

  const alerting = s.disagreement_rate >= s.alert_threshold;
  $("comparisonReadouts").innerHTML = `
    ${ro("gap", "Average gap", fixed(s.avg_confidence_gap_all, 4),
      "0 would mean identical answers",
      "How far apart the two models are on average. Near zero means compressing the model barely changed it — the result you want.")}
    ${ro("plain", "Sentences compared", fmtInt(s.total_comparisons),
      "both models read every one",
      "Both models score every request even when only one answers you, so they can be compared continuously at no risk.")}
    ${ro(alerting ? "alarm" : "ok", "Outright disagreements", pct(s.disagreement_rate),
      `${s.disagreements_in_window} of ${s.window_size} · alerts above ${pct(s.alert_threshold)}`,
      "How often the two models reached opposite conclusions. This is the strong signal — it rises before accuracy visibly drops, and needs no correct answers to compute.")}
    ${ro("speed", "Gap when they disagree", fixed(s.avg_confidence_gap_on_disagreements, 4),
      "higher means confidently split",
      "When they do disagree, how confident each was. A large number means one model was confidently wrong rather than both being unsure.")}
  `;

  const cases = recent.comparisons || [];
  drawGapChart(cases, s.avg_confidence_gap_all ?? 0);

  const dir = s.direction_breakdown || {};
  $("directionTable").innerHTML = `<tbody>${
    Object.keys(dir).length
      ? Object.entries(dir).map(([k, v]) => row(k.replace(/_to_/g, " → "), fmtInt(v))).join("")
      : `<tr><td class="k">No disagreements yet — the models have agreed every time</td><td class="n">—</td></tr>`
  }</tbody>`;

  /*
   * The API returns labels, scores and input length but deliberately not the
   * input text, so nothing anyone typed is echoed back out of monitoring.
   *
   * What is left has to work harder, so each row draws both confidences as
   * bars. Two numbers four decimals apart are hard to compare by eye; two bars
   * of near-identical length say "these models agree" instantly, and a row
   * where the bars point at different labels is unmissable.
   */
  const first = (s.total_comparisons || cases.length) - cases.length + 1;

  const gapWords = (g) =>
    g < 0.005 ? "practically identical" : g < 0.05 ? "slightly apart" : "far apart";

  const bar = (who, label, score, tone) => `
    <div class="cmp-bar">
      <span class="who">${who}</span>
      <span class="track"><i class="${tone}" style="width:${(score * 100).toFixed(1)}%"></i></span>
      <span class="val">${esc(label)} ${fixed(score, 4)}</span>
    </div>`;

  $("comparisonList").innerHTML = cases.length
    ? cases
        .slice()
        .reverse()
        .map((c, revIdx) => {
          const n = first + (cases.length - 1 - revIdx);
          return `<div class="item cmp ${c.agrees ? "" : "split"}">
            <div class="top">
              <span class="seq">#${n}</span>
              <span class="trig" style="color:${c.agrees ? "var(--ok)" : "var(--gap)"}">
                ${c.agrees ? "agreed" : "disagreed"}</span>
              <time>gap ${fixed(c.confidence_gap, 4)} — ${gapWords(c.confidence_gap ?? 0)}</time>
            </div>
            ${bar("old", c.v1_label, c.v1_score, "ok")}
            ${bar("new", c.v2_label, c.v2_score, c.agrees ? "speed" : "alarm")}
            <div class="note">${fmtInt(c.input_length)}-character sentence — the text itself is not
              stored, so nothing anyone types is echoed back here</div>
          </div>`;
        })
        .join("")
    : `<div class="empty">Nothing compared yet. Send a sentence on the Predict tab — in Shadow both
       models score every one, so a row appears here immediately.</div>`;
}

/*
 * One point per comparison. The y-axis is scaled to the data rather than fixed
 * at 0–1: the gap is typically around 0.002, so a fixed axis would draw a flat
 * line along the bottom and hide the very thing the chart exists to show.
 */
function drawGapChart(cases, meanGap) {
  const svg = $("gapChart");
  const W = 720;
  const H = 200;
  const PAD = 14;

  if (!cases.length) {
    svg.innerHTML = `
      <g stroke="#262B40" stroke-width="1">
        <line x1="0" y1="50" x2="${W}" y2="50"/><line x1="0" y1="100" x2="${W}" y2="100"/>
        <line x1="0" y1="150" x2="${W}" y2="150"/>
      </g>
      <text x="${W / 2}" y="${H / 2 - 6}" text-anchor="middle" fill="#A7AECB"
            font-family="ui-monospace, monospace" font-size="15">Nothing compared yet</text>
      <text x="${W / 2}" y="${H / 2 + 18}" text-anchor="middle" fill="#7E86A6"
            font-family="ui-monospace, monospace" font-size="12">
        Send a sentence on the Predict tab and a point appears here</text>`;
    return;
  }

  const gaps = cases.map((c) => c.confidence_gap ?? 0);
  const peak = Math.max(...gaps, meanGap, 1e-6);
  const yOf = (v) => H - PAD - (v / peak) * (H - PAD * 2);
  const step = cases.length > 1 ? W / (cases.length - 1) : W;
  const pts = cases.map((c, i) => [i * step, yOf(c.confidence_gap ?? 0)]);

  const line = pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" L");
  const area = `M${line} L${W},${H} L0,${H} Z`;
  const meanY = yOf(meanGap);

  const dots = cases
    .map((c, i) =>
      c.agrees
        ? `<circle cx="${pts[i][0].toFixed(1)}" cy="${pts[i][1].toFixed(1)}" r="4" fill="#22F0D4"/>`
        : `<circle cx="${pts[i][0].toFixed(1)}" cy="${pts[i][1].toFixed(1)}" r="6.5" fill="#FF6BA3"
             stroke="#FF6BA3" stroke-opacity="0.35" stroke-width="6"/>`
    )
    .join("");

  svg.innerHTML = `
    <defs>
      <linearGradient id="gapFill" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0%" stop-color="#22F0D4" stop-opacity="0.45"/>
        <stop offset="60%" stop-color="#9B82FF" stop-opacity="0.16"/>
        <stop offset="100%" stop-color="#9B82FF" stop-opacity="0"/>
      </linearGradient>
      <linearGradient id="gapLine" x1="0" x2="1" y1="0" y2="0">
        <stop offset="0%" stop-color="#22F0D4"/><stop offset="100%" stop-color="#9B82FF"/>
      </linearGradient>
    </defs>
    <g stroke="#262B40" stroke-width="1">
      <line x1="0" y1="50" x2="${W}" y2="50"/><line x1="0" y1="100" x2="${W}" y2="100"/>
      <line x1="0" y1="150" x2="${W}" y2="150"/>
    </g>
    <path fill="url(#gapFill)" d="${area}"/>
    <path fill="none" stroke="url(#gapLine)" stroke-width="2.8" d="M${line}"/>
    <line x1="0" y1="${meanY.toFixed(1)}" x2="${W}" y2="${meanY.toFixed(1)}"
          stroke="#A7AECB" stroke-width="1.4" stroke-dasharray="5 5"/>
    ${dots}
    <text x="8" y="18" fill="#7E86A6" font-family="ui-monospace, monospace" font-size="12">
      top of chart = ${peak.toFixed(4)}</text>`;
}

/* ───────────────────────── drift ───────────────────────── */

async function refreshDrift() {
  const [d, hist] = await Promise.all([api("/monitoring/drift"), api("/monitoring/drift/history")]);

  const last = d.last_check;
  const th = d.thresholds || {};
  const frozen = d.reference_window_frozen;

  $("driftReadouts").innerHTML = `
    ${ro(frozen ? "ok" : "plain", "Reference snapshot", frozen ? "frozen" : "filling",
      `${fmtInt(d.reference_size)} sentences`,
      "The first batch of traffic is frozen as the baseline. Everything later is compared against it.")}
    ${ro("plain", "Sentences seen", fmtInt(d.total_records), `${fmtInt(d.checks_run)} checks run`,
      "Checks run automatically once enough new sentences have arrived.")}
    ${ro(last ? (last.text_length_drifted ? "alarm" : "ok") : "plain", "Sentence length",
      last ? fixed(last.text_length_drift_score, 3) : "—",
      `limit ${fixed(th.text_length ?? 0.1, 2)}`,
      "Are the sentences arriving now about as long as the ones in the baseline? 0 means identical, 1 means nothing in common.")}
    ${ro(last ? (last.confidence_drifted ? "alarm" : "ok") : "plain", "Model confidence",
      last ? fixed(last.confidence_drift_score, 3) : "—",
      `limit ${fixed(th.confidence ?? 0.1, 2)}`,
      "Is the model as sure of itself as it used to be? A confidence shift often shows up before accuracy visibly drops.")}
  `;

  const history = hist.history || [];
  $("driftMeter").innerHTML = history.length
    ? history
        .slice(-8)
        .map((h) => {
          const bar = (label, score, limit, drifted, tone) => {
            const w = Math.min(100, limit ? (score / limit) * 100 : 0);
            return `<div class="meter-row">
              <span class="lab">${esc(label)}</span>
              <span class="track"><i class="${drifted ? "alarm" : tone}" style="width:${w.toFixed(0)}%"></i></span>
              <span class="n">${fixed(score, 3)}</span>
            </div>`;
          };
          return (
            bar(`#${h.checked_at_request_n} length`, h.text_length_drift_score, th.text_length ?? 0.1, h.text_length_drifted, "ok") +
            bar(`#${h.checked_at_request_n} confidence`, h.confidence_drift_score, th.confidence ?? 0.1, h.confidence_drifted, "speed")
          );
        })
        .join("")
    : `<div class="empty">No checks yet. The baseline snapshot fills first, then each later batch
       of sentences is compared against it automatically.</div>`;

  $("driftTable").innerHTML = `<tbody>
    ${row("baseline frozen", String(frozen))}
    ${row("baseline size", fmtInt(d.reference_size))}
    ${row("sentences seen", fmtInt(d.total_records))}
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
    ${ro(c.backend === "none" ? "alarm" : "cache", "Store in use", esc(c.backend),
      c.available ? "answering" : "unavailable",
      "Which store actually answered. Named explicitly so a stand-in can never quietly pass for the real thing.")}
    ${ro("ok", "Answered from cache", pct(c.hit_rate || 0), `${fmtInt(total)} lookups`,
      "The share of requests that skipped the model entirely because the same sentence had been seen before.")}
    ${ro("plain", "Cache hits", fmtInt(c.hits), "model did not run",
      "Each of these returned in well under a millisecond instead of running a transformer.")}
    ${ro("plain", "Cache misses", fmtInt(c.misses), `${fmtInt(c.errors)} errors`,
      "New sentences the model had to actually run. Errors fall through to the model rather than failing the request.")}
  `;

  const hit = (c.hit_rate || 0) * 100;
  $("cacheMeter").innerHTML = `
    <div class="meter-row"><span class="lab">From cache</span>
      <span class="track"><i class="ok" style="width:${hit.toFixed(1)}%"></i></span>
      <span class="n">${fmtInt(c.hits)}</span></div>
    <div class="meter-row"><span class="lab">Model ran</span>
      <span class="track"><i class="cache" style="width:${(100 - hit).toFixed(1)}%"></i></span>
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
              <span class="trig">${esc((e.trigger || "—").replace(/_/g, " "))}</span>
              <span class="arrow">${esc(e.from_state || "—")} → ${esc(e.to_state || "—")}</span>
            </div>
            <div class="note">
              failures ${pct(e.v2_error_rate_at_event ?? 0)} ·
              new model ${fixed(e.v2_p99_latency_ms, 1)} ms ·
              old model ${fixed(e.v1_p99_latency_ms, 1)} ms ·
              ${fmtInt(e.requests_seen)} requests${e.note ? ` · ${esc(e.note)}` : ""}
            </div>
          </div>`
        )
        .join("")
    : `<div class="empty">Nothing has changed yet. Promote or roll back on the Rollout tab and it
       will be recorded here.</div>`;
}

function fmtTime(ts) {
  if (!ts) return "—";
  const d = new Date(ts);
  return isNaN(d) ? ts : d.toLocaleTimeString();
}

/* ───────────────────────── wiring ───────────────────────── */

// Manual and Predict have nothing to fetch: one is static text, the other only
// changes when you press a button.
const VIEWS = {
  manual: async () => {},
  predict: async () => {},
  comparison: refreshComparison,
  rollout: refreshRollout,
  drift: refreshDrift,
  cache: refreshCache,
  audit: refreshAudit,
};

let current = "manual";

async function refreshAll() {
  try {
    await refreshChrome();
    await VIEWS[current]();
  } catch (e) {
    if (e.notReady) wakeUp();
    else toast(e.message, true);
  }
}

$("tabs").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-view]");
  if (!btn) return;
  current = btn.dataset.view;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b === btn));
  document
    .querySelectorAll(".view")
    .forEach((v) => v.classList.toggle("on", v.id === `view-${current}`));
  await refreshAll();
});

$("samples").addEventListener("click", (ev) => {
  const b = ev.target.closest("button[data-kind]");
  if (b) $("predictText").value = randomSentence(b.dataset.kind);
});

$("btnRandom").addEventListener("click", () => {
  $("predictText").value = randomSentence("any");
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
 * Live mode stops itself after two minutes. Someone watching a rollout wants a
 * few minutes of live numbers; nobody wants a forgotten tab holding a
 * scale-to-zero container awake all night.
 */
$("btnLive").addEventListener("click", () => {
  if (liveTimer) {
    clearInterval(liveTimer);
    liveTimer = null;
    $("btnLive").textContent = "Auto-refresh: off";
    return;
  }
  liveStopAt = Date.now() + 120000;
  liveTimer = setInterval(() => {
    if (Date.now() > liveStopAt) {
      clearInterval(liveTimer);
      liveTimer = null;
      $("btnLive").textContent = "Auto-refresh: off";
      toast("Auto-refresh stopped after 2 minutes");
      return;
    }
    refreshAll();
  }, 5000);
  $("btnLive").textContent = "Auto-refresh: on";
  refreshAll();
});

boot();
