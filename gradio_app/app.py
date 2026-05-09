"""
Gradio Deployment Control Panel.

A real-time dashboard for managing the model serving deployment lifecycle.
This is what an ML engineer would use to monitor and control a canary rollout.

Tabs:
  1. Predict          — test the API with live predictions, see which model responded
  2. Deployment       — deployment state, promote/rollback controls, progress bar
  3. Shadow Analysis  — disagreement rate charts, recent disagreement table
  4. Drift Monitor    — Evidently drift scores, confidence distribution comparison
  5. Circuit Breaker  — circuit state, failure counts, manual reset
  6. Audit Log        — full history of state transitions

Design philosophy:
  Every panel auto-refreshes so the state is always live.
  Promote and Rollback buttons give immediate visual feedback.
  The disagreement chart updates in real time as shadow requests accumulate.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import gradio as gr
import httpx
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# API base URL — can be overridden by environment variable
# ---------------------------------------------------------------------------
API_BASE = os.getenv("GATEWAY_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# API client helpers
# ---------------------------------------------------------------------------


def api_get(path: str) -> Optional[dict]:
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_post(path: str, body: Optional[dict] = None) -> Optional[dict]:
    try:
        r = httpx.post(f"{API_BASE}{path}", json=body or {}, timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tab 1: Predict
# ---------------------------------------------------------------------------


def run_prediction(text: str) -> tuple[str, str, str, str, str]:
    """Call /predict and return formatted result fields."""
    if not text.strip():
        return "—", "—", "—", "—", "—"

    result = api_post("/predict", {"text": text})
    if "error" in result:
        return f"Error: {result['error']}", "—", "—", "—", "—"

    label = result.get("label", "—")
    score = f"{result.get('score', 0):.4f}"
    model_used = result.get("model_used", "—")
    deployment_state = result.get("deployment_state", "—")
    latency = f"{result.get('latency_ms', 0):.1f} ms"
    cache_hit = "Yes" if result.get("cache_hit") else "No"

    label_display = f"{'POSITIVE' if label == 'POSITIVE' else 'NEGATIVE'}"
    return (
        label_display,
        score,
        model_used,
        deployment_state,
        f"{latency} | Cache: {cache_hit}",
    )


def predict_tab():
    with gr.Column():
        gr.Markdown(
            "### Live Inference\n"
            "Submit text and see which model version served the request. "
            "The **model_used** field shows whether v1, v2, or a fallback responded."
        )
        text_input = gr.Textbox(
            label="Input Text",
            placeholder="Enter text to classify...",
            lines=3,
        )
        submit_btn = gr.Button("Predict", variant="primary")

        with gr.Row():
            label_out = gr.Textbox(label="Label", interactive=False)
            score_out = gr.Textbox(label="Confidence Score", interactive=False)

        with gr.Row():
            model_out = gr.Textbox(label="Model Used", interactive=False)
            state_out = gr.Textbox(label="Deployment State", interactive=False)

        perf_out = gr.Textbox(label="Latency | Cache", interactive=False)

        submit_btn.click(
            fn=run_prediction,
            inputs=[text_input],
            outputs=[label_out, score_out, model_out, state_out, perf_out],
        )


# ---------------------------------------------------------------------------
# Tab 2: Deployment Control
# ---------------------------------------------------------------------------

STAGE_ORDER = ["shadow", "canary_5", "canary_25", "canary_50", "full", "rolled_back"]
STAGE_LABELS = {
    "shadow": "Shadow (0%)",
    "canary_5": "Canary 5%",
    "canary_25": "Canary 25%",
    "canary_50": "Canary 50%",
    "full": "Full (100%)",
    "rolled_back": "Rolled Back",
}


def get_deployment_html() -> str:
    """Build an HTML status card for the current deployment state."""
    data = api_get("/deployment/status")
    if not data or "error" in data:
        return f"<p style='color:red'>API error: {data.get('error', 'unknown')}</p>"

    state = data.get("state", "—")
    v2_frac = data.get("v2_traffic_fraction", 0)
    v2_reqs = data.get("v2_requests", 0)
    v2_err_rate = data.get("v2_error_rate", 0)
    v2_p99 = data.get("v2_p99_latency_ms", 0)
    v1_p99 = data.get("v1_p99_latency_ms", 0)
    total_reqs = data.get("total_requests", 0)
    time_in_state = data.get("time_in_state_seconds", 0)

    cb = data.get("circuit_breaker", {})
    cb_state = cb.get("state", "unknown")
    cb_color = {"closed": "green", "open": "red", "half_open": "orange"}.get(
        cb_state, "grey"
    )

    state_color = {
        "shadow": "#6366f1",
        "canary_5": "#f59e0b",
        "canary_25": "#f97316",
        "canary_50": "#ef4444",
        "full": "#10b981",
        "rolled_back": "#6b7280",
    }.get(state, "#6b7280")

    # Progress bar
    idx = STAGE_ORDER.index(state) if state in STAGE_ORDER else 0
    max_idx = len(STAGE_ORDER) - 2  # exclude rolled_back from progress
    progress_pct = min(int((idx / max_idx) * 100), 100)

    rollback_thresholds = data.get("rollback_thresholds", {})

    return f"""
    <div style="font-family: monospace; padding: 12px;">
      <div style="background:{state_color}; color:white; padding:8px 16px;
                  border-radius:6px; display:inline-block; font-size:1.2em;
                  font-weight:bold; margin-bottom:12px;">
        STATE: {STAGE_LABELS.get(state, state).upper()}
      </div>

      <div style="background:#f3f4f6; padding:10px; border-radius:6px; margin-bottom:10px;">
        <b>Deployment Progress</b>
        <div style="background:#e5e7eb; border-radius:4px; height:20px; margin-top:6px;">
          <div style="background:{state_color}; width:{progress_pct}%; height:100%;
                      border-radius:4px; text-align:center; line-height:20px;
                      color:white; font-size:0.85em; font-weight:bold;">
            {progress_pct}%
          </div>
        </div>
        <div style="margin-top:4px; font-size:0.8em; color:#6b7280;">
          Shadow → Canary 5% → Canary 25% → Canary 50% → Full
        </div>
      </div>

      <table style="width:100%; border-collapse:collapse;">
        <tr><td style="padding:4px 8px; color:#6b7280;">v2 Traffic Fraction</td>
            <td style="padding:4px 8px; font-weight:bold;">{v2_frac:.0%}</td></tr>
        <tr><td style="padding:4px 8px; color:#6b7280;">v2 Requests</td>
            <td style="padding:4px 8px;">{v2_reqs:,}</td></tr>
        <tr><td style="padding:4px 8px; color:#6b7280;">v2 Error Rate</td>
            <td style="padding:4px 8px; color:{"red" if v2_err_rate > rollback_thresholds.get("error_rate", 0.05) else "green"};">
              {v2_err_rate:.2%} (threshold: {rollback_thresholds.get("error_rate", 0.05):.0%})
            </td></tr>
        <tr><td style="padding:4px 8px; color:#6b7280;">v2 p99 Latency</td>
            <td style="padding:4px 8px;">{v2_p99:.1f} ms</td></tr>
        <tr><td style="padding:4px 8px; color:#6b7280;">v1 p99 Latency</td>
            <td style="padding:4px 8px;">{v1_p99:.1f} ms</td></tr>
        <tr><td style="padding:4px 8px; color:#6b7280;">Total Requests</td>
            <td style="padding:4px 8px;">{total_reqs:,}</td></tr>
        <tr><td style="padding:4px 8px; color:#6b7280;">Time in State</td>
            <td style="padding:4px 8px;">{time_in_state:.0f}s</td></tr>
        <tr><td style="padding:4px 8px; color:#6b7280;">Circuit Breaker</td>
            <td style="padding:4px 8px; color:{cb_color}; font-weight:bold;">
              {cb_state.upper()}
            </td></tr>
      </table>
    </div>
    """


def do_promote() -> str:
    result = api_post("/deployment/promote")
    if result.get("ok"):
        return f"Promoted: {result['from_state']} → {result['to_state']}"
    return f"Cannot promote: {result.get('reason', 'unknown')}"


def do_rollback() -> str:
    result = api_post("/deployment/rollback")
    if result.get("ok"):
        return f"Rolled back: {result['from_state']} → {result['to_state']} | Cache flushed: {result.get('cache_flushed', False)}"
    return f"Cannot rollback: {result.get('reason', 'unknown')}"


def deployment_tab():
    with gr.Column():
        gr.Markdown(
            "### Deployment Control Panel\n"
            "Promote v2 through stages or roll back instantly. "
            "Auto-progression advances stages on a timer if no rollback triggers fire."
        )
        status_html = gr.HTML(label="Deployment Status")

        with gr.Row():
            promote_btn = gr.Button("Promote to Next Stage", variant="primary")
            rollback_btn = gr.Button("Roll Back to Shadow", variant="stop")

        action_result = gr.Textbox(label="Last Action Result", interactive=False)
        refresh_btn = gr.Button("Refresh Status")

        def refresh_status():
            return get_deployment_html()

        refresh_btn.click(fn=refresh_status, inputs=[], outputs=[status_html])
        promote_btn.click(
            fn=lambda: (do_promote(), get_deployment_html()),
            inputs=[],
            outputs=[action_result, status_html],
        )
        rollback_btn.click(
            fn=lambda: (do_rollback(), get_deployment_html()),
            inputs=[],
            outputs=[action_result, status_html],
        )

        # Load status on tab load
        status_html.value = get_deployment_html()


# ---------------------------------------------------------------------------
# Tab 3: Shadow Analysis
# ---------------------------------------------------------------------------


def get_disagreement_chart() -> go.Figure:
    """Gauge chart showing current disagreement rate."""
    data = api_get("/monitoring/disagreement")
    if not data or "error" in data:
        rate = 0.0
        threshold = 0.3
    else:
        rate = data.get("disagreement_rate", 0.0)
        threshold = data.get("alert_threshold", 0.3)

    color = (
        "green" if rate < threshold * 0.5 else "orange" if rate < threshold else "red"
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=rate * 100,
            number={"suffix": "%", "font": {"size": 28}},
            title={"text": "v1 vs v2 Disagreement Rate", "font": {"size": 14}},
            delta={"reference": threshold * 100, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, threshold * 50], "color": "#d1fae5"},
                    {"range": [threshold * 50, threshold * 100], "color": "#fef3c7"},
                    {"range": [threshold * 100, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=10, r=10))
    return fig


def get_disagreement_table() -> str:
    """Recent disagreements as an HTML table."""
    data = api_get("/monitoring/disagreement/recent?n=15")
    if not data or "error" in data:
        return "<p>No data available</p>"

    rows = data.get("recent_disagreements", [])
    if not rows:
        return "<p>No disagreements recorded yet. Running in shadow mode...</p>"

    html = """
    <table style="width:100%; border-collapse:collapse; font-family:monospace; font-size:0.9em;">
    <tr style="background:#f3f4f6;">
      <th style="padding:6px 10px; text-align:left;">v1 Label</th>
      <th style="padding:6px 10px; text-align:left;">v2 Label</th>
      <th style="padding:6px 10px; text-align:right;">v1 Score</th>
      <th style="padding:6px 10px; text-align:right;">v2 Score</th>
      <th style="padding:6px 10px; text-align:right;">Conf Gap</th>
      <th style="padding:6px 10px; text-align:right;">Input Len</th>
    </tr>
    """
    for i, r in enumerate(rows):
        bg = "#fff" if i % 2 == 0 else "#f9fafb"
        html += f"""
        <tr style="background:{bg};">
          <td style="padding:4px 10px; color:{"#10b981" if r["v1_label"] == "POSITIVE" else "#ef4444"};">
            {r["v1_label"]}
          </td>
          <td style="padding:4px 10px; color:{"#10b981" if r["v2_label"] == "POSITIVE" else "#ef4444"};">
            {r["v2_label"]}
          </td>
          <td style="padding:4px 10px; text-align:right;">{r["v1_score"]:.4f}</td>
          <td style="padding:4px 10px; text-align:right;">{r["v2_score"]:.4f}</td>
          <td style="padding:4px 10px; text-align:right; color:#f59e0b;">
            {r["confidence_gap"]:.4f}
          </td>
          <td style="padding:4px 10px; text-align:right;">{r["input_length"]}</td>
        </tr>
        """
    html += "</table>"
    return html


def get_disagreement_summary() -> str:
    data = api_get("/monitoring/disagreement")
    if not data or "error" in data:
        return "No data"

    total = data.get("total_comparisons", 0)
    window = data.get("window_size", 0)
    disagree = data.get("disagreements_in_window", 0)
    rate = data.get("disagreement_rate", 0)
    alert = data.get("alert_active", False)
    direction = data.get("direction_breakdown", {})
    avg_gap = data.get("avg_confidence_gap_all", 0)

    alert_str = " | ALERT ACTIVE" if alert else ""
    direction_str = (
        " | ".join(f"{k}: {v}" for k, v in direction.items()) if direction else "None"
    )

    return (
        f"Total comparisons: {total:,} | Window: {window} | "
        f"Disagreements: {disagree} | Rate: {rate:.2%}{alert_str}\n"
        f"Direction breakdown: {direction_str}\n"
        f"Avg confidence gap: {avg_gap:.4f}"
    )


def shadow_tab():
    with gr.Column():
        gr.Markdown(
            "### Shadow Mode Analysis\n"
            "In shadow mode, v1 serves users and v2 runs silently on every request. "
            "This panel shows how often v1 and v2 disagree — a key signal before promoting."
        )
        disagree_chart = gr.Plot(label="Disagreement Rate Gauge")
        summary_text = gr.Textbox(label="Summary Stats", interactive=False, lines=3)
        disagree_table_html = gr.HTML(label="Recent Disagreements")
        refresh_btn = gr.Button("Refresh")

        def refresh():
            return (
                get_disagreement_chart(),
                get_disagreement_summary(),
                get_disagreement_table(),
            )

        refresh_btn.click(
            fn=refresh,
            inputs=[],
            outputs=[disagree_chart, summary_text, disagree_table_html],
        )
        # Load on start
        disagree_chart.value = get_disagreement_chart()
        summary_text.value = get_disagreement_summary()
        disagree_table_html.value = get_disagreement_table()


# ---------------------------------------------------------------------------
# Tab 4: Drift Monitor
# ---------------------------------------------------------------------------


def get_drift_status_html() -> str:
    data = api_get("/monitoring/drift")
    if not data or "error" in data:
        return f"<p style='color:red'>Error: {data.get('error', 'unknown')}</p>"

    frozen = data.get("reference_window_frozen", False)
    ref_size = data.get("reference_size", 0)
    total = data.get("total_records", 0)
    checks = data.get("checks_run", 0)
    last = data.get("last_check")
    thresholds = data.get("thresholds", {})

    if not frozen:
        return f"""
        <div style="padding:12px; background:#fef3c7; border-radius:6px;">
          <b>Building reference window...</b>
          <p>Collecting first {thresholds.get("text_length", "?")} requests to establish baseline distribution.</p>
          <p>Progress: {ref_size} / {200} records collected.</p>
        </div>
        """

    if not last:
        return f"""
        <div style="padding:12px; background:#d1fae5; border-radius:6px;">
          <b>Reference window frozen ({ref_size} records)</b>
          <p>Total requests seen: {total:,} | Drift checks run: {checks}</p>
          <p>Waiting for enough current data to run first drift check...</p>
        </div>
        """

    text_drift = last.get("text_length_drifted", False)
    conf_drift = last.get("confidence_drifted", False)
    any_drift = last.get("any_drift", False)
    text_score = last.get("text_length_drift_score", 0)
    conf_score = last.get("confidence_drift_score", 0)
    method = last.get("method", "—")

    status_color = "#fee2e2" if any_drift else "#d1fae5"
    status_text = "DRIFT DETECTED" if any_drift else "No Drift"

    return f"""
    <div style="padding:12px; background:{status_color}; border-radius:6px; font-family:monospace;">
      <b style="font-size:1.1em;">Status: {status_text}</b>
      <p style="color:#6b7280; margin:4px 0;">Detection method: {method}</p>

      <table style="width:100%; margin-top:8px; border-collapse:collapse;">
        <tr>
          <th style="text-align:left; padding:4px 8px; color:#6b7280;">Feature</th>
          <th style="text-align:right; padding:4px 8px; color:#6b7280;">JS Divergence Score</th>
          <th style="text-align:right; padding:4px 8px; color:#6b7280;">Threshold</th>
          <th style="text-align:right; padding:4px 8px; color:#6b7280;">Drifted?</th>
        </tr>
        <tr>
          <td style="padding:4px 8px;">Text Length</td>
          <td style="padding:4px 8px; text-align:right;">{text_score:.4f}</td>
          <td style="padding:4px 8px; text-align:right;">{thresholds.get("text_length", 0.1):.2f}</td>
          <td style="padding:4px 8px; text-align:right; color:{"red" if text_drift else "green"};">
            {"YES" if text_drift else "No"}
          </td>
        </tr>
        <tr>
          <td style="padding:4px 8px;">Confidence Score</td>
          <td style="padding:4px 8px; text-align:right;">{conf_score:.4f}</td>
          <td style="padding:4px 8px; text-align:right;">{thresholds.get("confidence", 0.1):.2f}</td>
          <td style="padding:4px 8px; text-align:right; color:{"red" if conf_drift else "green"};">
            {"YES" if conf_drift else "No"}
          </td>
        </tr>
      </table>

      <p style="margin-top:8px; color:#6b7280; font-size:0.85em;">
        Reference: {ref_size} records | Total: {total:,} | Checks: {checks}
      </p>
    </div>
    """


def drift_tab():
    with gr.Column():
        gr.Markdown(
            "### Input Distribution Drift Monitor\n"
            "Evidently AI compares the current input distribution against the reference baseline. "
            "Drift indicates the types of inputs arriving during canary differ from the training distribution."
        )
        drift_html = gr.HTML(label="Drift Status")
        refresh_btn = gr.Button("Refresh")
        refresh_btn.click(fn=get_drift_status_html, inputs=[], outputs=[drift_html])
        drift_html.value = get_drift_status_html()


# ---------------------------------------------------------------------------
# Tab 5: Circuit Breaker
# ---------------------------------------------------------------------------


def get_circuit_breaker_html() -> str:
    data = api_get("/circuit-breaker/status")
    if not data or "error" in data:
        return f"<p style='color:red'>Error: {data.get('error', 'unknown')}</p>"

    state = data.get("state", "unknown")
    failures = data.get("failure_count", 0)
    total_calls = data.get("total_calls", 0)
    total_failures = data.get("total_failures", 0)
    total_blocked = data.get("total_blocked", 0)
    failure_rate = data.get("failure_rate", 0)
    thresholds = data.get("thresholds", {})

    state_color = {"closed": "#d1fae5", "open": "#fee2e2", "half_open": "#fef3c7"}.get(
        state, "#f3f4f6"
    )
    state_text_color = {"closed": "green", "open": "red", "half_open": "orange"}.get(
        state, "black"
    )

    return f"""
    <div style="padding:12px; background:{
        state_color
    }; border-radius:6px; font-family:monospace;">
      <div style="font-size:1.3em; font-weight:bold; color:{state_text_color};">
        Circuit: {state.upper()}
      </div>

      <p style="margin:8px 0; color:#374151;">
        {
        "v2 is operating normally."
        if state == "closed"
        else "v2 is BLOCKED. All traffic falls back to v1 immediately (fail-fast)."
        if state == "open"
        else "One probe request allowed through to test if v2 recovered."
    }
      </p>

      <table style="width:100%; border-collapse:collapse; margin-top:8px;">
        <tr><td style="padding:4px 8px; color:#6b7280;">Consecutive failures</td>
            <td style="padding:4px 8px; font-weight:bold; color:{
        "red" if failures > 0 else "green"
    };">
              {failures} / {thresholds.get("failure_threshold", 5)}
            </td></tr>
        <tr><td style="padding:4px 8px; color:#6b7280;">Total v2 calls</td>
            <td style="padding:4px 8px;">{total_calls:,}</td></tr>
        <tr><td style="padding:4px 8px; color:#6b7280;">Total v2 failures</td>
            <td style="padding:4px 8px; color:{
        "red" if total_failures > 0 else "green"
    };">{total_failures:,}</td></tr>
        <tr><td style="padding:4px 8px; color:#6b7280;">Requests blocked</td>
            <td style="padding:4px 8px;">{total_blocked:,}</td></tr>
        <tr><td style="padding:4px 8px; color:#6b7280;">Failure rate</td>
            <td style="padding:4px 8px;">{failure_rate:.2%}</td></tr>
        <tr><td style="padding:4px 8px; color:#6b7280;">Open timeout</td>
            <td style="padding:4px 8px;">{
        thresholds.get("timeout_seconds", 30)
    }s</td></tr>
      </table>
    </div>
    """


def do_cb_reset() -> tuple[str, str]:
    result = api_post("/circuit-breaker/reset")
    if result.get("ok"):
        return "Circuit breaker reset to CLOSED.", get_circuit_breaker_html()
    return f"Error: {result.get('error', 'unknown')}", get_circuit_breaker_html()


def circuit_breaker_tab():
    with gr.Column():
        gr.Markdown(
            "### Circuit Breaker\n"
            "Protects v1 from v2 failures. When v2 errors exceed the threshold, "
            "the circuit opens and all traffic immediately falls back to v1 (fail-fast, <1ms)."
        )
        cb_html = gr.HTML(label="Circuit Breaker Status")
        with gr.Row():
            reset_btn = gr.Button("Reset Circuit Breaker (CLOSED)", variant="secondary")
            refresh_btn = gr.Button("Refresh")

        action_out = gr.Textbox(label="Action Result", interactive=False)

        reset_btn.click(fn=do_cb_reset, inputs=[], outputs=[action_out, cb_html])
        refresh_btn.click(fn=get_circuit_breaker_html, inputs=[], outputs=[cb_html])
        cb_html.value = get_circuit_breaker_html()


# ---------------------------------------------------------------------------
# Tab 6: Audit Log
# ---------------------------------------------------------------------------


def get_audit_log_html() -> str:
    data = api_get("/deployment/audit")
    if not data or "error" in data:
        return f"<p style='color:red'>Error: {data.get('error', 'unknown')}</p>"

    entries = data.get("entries", [])
    if not entries:
        return "<p>No transitions recorded yet.</p>"

    trigger_colors = {
        "startup": "#6b7280",
        "manual_promote": "#10b981",
        "auto_promote": "#3b82f6",
        "manual_rollback": "#f59e0b",
        "auto_rollback_error_rate": "#ef4444",
        "auto_rollback_latency": "#ef4444",
    }

    html = """
    <table style="width:100%; border-collapse:collapse; font-family:monospace; font-size:0.85em;">
    <tr style="background:#f3f4f6;">
      <th style="padding:6px 8px; text-align:left;">Timestamp</th>
      <th style="padding:6px 8px; text-align:left;">Transition</th>
      <th style="padding:6px 8px; text-align:left;">Trigger</th>
      <th style="padding:6px 8px; text-align:right;">v2 Err Rate</th>
      <th style="padding:6px 8px; text-align:right;">v2 p99 ms</th>
      <th style="padding:6px 8px; text-align:right;">v1 p99 ms</th>
      <th style="padding:6px 8px; text-align:right;">Requests</th>
    </tr>
    """
    for i, e in enumerate(reversed(entries)):
        bg = "#fff" if i % 2 == 0 else "#f9fafb"
        ts = e.get("timestamp", "—")[:19].replace("T", " ")
        trigger = e.get("trigger", "—")
        t_color = trigger_colors.get(trigger, "#374151")

        html += f"""
        <tr style="background:{bg};">
          <td style="padding:4px 8px; color:#6b7280;">{ts}</td>
          <td style="padding:4px 8px;">
            <span style="color:#6b7280;">{e.get("from_state", "—")}</span>
            → <b>{e.get("to_state", "—")}</b>
          </td>
          <td style="padding:4px 8px; color:{t_color}; font-weight:bold;">{trigger}</td>
          <td style="padding:4px 8px; text-align:right;">{e.get("v2_error_rate_at_event", 0):.2%}</td>
          <td style="padding:4px 8px; text-align:right;">{e.get("v2_p99_latency_ms", 0):.1f}</td>
          <td style="padding:4px 8px; text-align:right;">{e.get("v1_p99_latency_ms", 0):.1f}</td>
          <td style="padding:4px 8px; text-align:right;">{e.get("requests_seen", 0):,}</td>
        </tr>
        """
        if e.get("note"):
            html += f"""
            <tr style="background:{bg};">
              <td colspan="7" style="padding:2px 8px 6px 24px; color:#9ca3af; font-style:italic;">
                {e["note"]}
              </td>
            </tr>
            """
    html += "</table>"
    return html


def audit_tab():
    with gr.Column():
        gr.Markdown(
            "### Deployment Audit Log\n"
            "Every state transition is logged here with the metrics at the time of the event. "
            "Auto-rollbacks show the error rate or latency that triggered them."
        )
        audit_html = gr.HTML(label="Audit Log")
        refresh_btn = gr.Button("Refresh")
        refresh_btn.click(fn=get_audit_log_html, inputs=[], outputs=[audit_html])
        audit_html.value = get_audit_log_html()


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="ML Model Serving — Deployment Control Panel",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# ML System Design: Model Serving\n"
            "**Production deployment lifecycle management** — shadow mode, "
            "canary rollout, circuit breaker, drift detection, and audit trail."
        )

        with gr.Tabs():
            with gr.Tab("Predict"):
                predict_tab()
            with gr.Tab("Deployment Control"):
                deployment_tab()
            with gr.Tab("Shadow Analysis"):
                shadow_tab()
            with gr.Tab("Drift Monitor"):
                drift_tab()
            with gr.Tab("Circuit Breaker"):
                circuit_breaker_tab()
            with gr.Tab("Audit Log"):
                audit_tab()

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name=os.getenv("GRADIO_HOST", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=False,
    )
