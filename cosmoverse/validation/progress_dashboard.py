"""
Standalone HTML progress dashboard for long-running COSMOVERSE SBI and
domain-adaptation runs (Master Document Results 3 and 4).

Dependency-free (stdlib only). Each `update` rewrites a single self-contained
HTML artifact that reloads itself every two seconds, so a training loop can call
`update` from the main process while a browser watches the file.
"""

import json
import math
import os
import pathlib
import tempfile
import webbrowser
from datetime import datetime, timezone
from typing import Any


class ProgressDashboard:
    """
    Renders the latest stage/payload pair as a standalone dark-theme HTML page.

    Payload keys are optional: progress, coverage, metrics, loss_history,
    fine_tune_loss_history, posterior, device, note.
    """

    def __init__(
        self, output_path: str | pathlib.Path = "dist/dashboard/cosmoverse_progress.html"
    ) -> None:
        self.output_path = pathlib.Path(output_path)
        self.stage = ""
        self.updated_at = ""
        self.payload: dict[str, Any] = {}

    def update(self, stage: str, payload: dict[str, Any]) -> pathlib.Path:
        """Replaces the dashboard state with `payload` and writes the HTML artifact."""
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")

        self.stage = str(stage)
        self.payload = _json_safe(payload)
        self.updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        document = _TEMPLATE.replace("__PAYLOAD__", self._state_json())
        self._write(document)
        return self.output_path

    def open(self) -> None:
        """Opens the dashboard artifact in the default browser."""
        webbrowser.open(self.output_path.resolve().as_uri())

    def _state_json(self) -> str:
        state = {
            "stage": self.stage,
            "generated_at": self.updated_at,
            "payload": self.payload,
        }
        return json.dumps(state, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")

    def _write(self, document: str) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(
            dir=self.output_path.parent, prefix=self.output_path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(document)
            os.replace(temp_name, self.output_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="2">
<title>COSMOVERSE | progress</title>
<style>
:root {
  --ink: #07080a;
  --panel: #0c0f12;
  --panel-raised: #101418;
  --line: #22262d;
  --text: #e9e7e2;
  --muted: #9aa1ab;
  --faint: #6b727c;
  --amber: #ffb347;
  --mint: #79d6c0;
  --pass: #4ade80;
  --fail: #ff6b6b;
  --mono: ui-monospace, "Cascadia Mono", "SFMono-Regular", Menlo, Consolas, monospace;
  --sans: "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
html { color-scheme: dark; }
body {
  margin: 0;
  min-height: 100vh;
  background:
    radial-gradient(1100px 520px at 84% -12%, rgba(255, 179, 71, 0.055), transparent 62%),
    repeating-linear-gradient(0deg, rgba(255, 255, 255, 0.02) 0 1px, transparent 1px 64px),
    repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.02) 0 1px, transparent 1px 64px),
    var(--ink);
  color: var(--text);
  font-family: var(--sans);
  font-size: 14px;
  line-height: 1.45;
  -webkit-font-smoothing: antialiased;
}
body::before {
  content: "";
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 2px;
  background: var(--amber);
  z-index: 2;
}
.shell { max-width: 1180px; margin: 0 auto; padding: clamp(20px, 3.4vw, 44px); }
.masthead {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  justify-content: space-between;
  gap: 22px;
  padding-bottom: 18px;
  border-bottom: 1px solid var(--line);
}
.brand h1 {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 0;
  font-size: clamp(26px, 4.2vw, 46px);
  font-weight: 650;
  letter-spacing: 0.16em;
  line-height: 1;
}
.live {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--pass);
  animation: pulse 2s ease-out infinite;
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.45); }
  70% { box-shadow: 0 0 0 7px rgba(74, 222, 128, 0); }
  100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
}
.kicker {
  margin: 9px 0 0;
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: var(--faint);
}
.telemetry { display: grid; gap: 5px; min-width: 250px; max-width: 100%; }
.telemetry .row {
  display: flex;
  align-items: baseline;
  justify-content: flex-end;
  gap: 14px;
}
.telemetry .k {
  font-family: var(--mono);
  font-size: 9.5px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--faint);
}
.telemetry .v {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--text);
  text-align: right;
  overflow-wrap: anywhere;
}
.progress-block { margin-top: 24px; }
.progress-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 16px;
}
.progress-head .label {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: var(--faint);
}
.progress-head .value {
  font-family: var(--mono);
  font-size: clamp(22px, 3vw, 32px);
  font-variant-numeric: tabular-nums;
  color: var(--amber);
  line-height: 1;
}
.progress {
  position: relative;
  height: 10px;
  margin-top: 10px;
  background: #111419;
  border: 1px solid var(--line);
  border-radius: 1px;
  overflow: hidden;
}
.progress::after {
  content: "";
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(
    90deg,
    transparent 0 calc(10% - 1px),
    rgba(255, 255, 255, 0.075) calc(10% - 1px) 10%
  );
  pointer-events: none;
}
.progress-fill {
  height: 100%;
  width: 100%;
  background: var(--amber);
  transform: scaleX(0);
  transform-origin: left center;
  transition: transform 0.5s cubic-bezier(0.2, 0.7, 0.2, 1);
}
.grid {
  display: grid;
  grid-template-columns: repeat(12, minmax(0, 1fr));
  gap: 16px;
  margin-top: 22px;
}
.panel {
  grid-column: span 6;
  min-width: 0;
  padding: 18px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 2px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}
.span-7 { grid-column: span 7; }
.span-5 { grid-column: span 5; }
.panel h2,
.note h2 {
  margin: 0 0 5px;
  font-family: var(--mono);
  font-size: 10.5px;
  font-weight: 600;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--muted);
}
.hint { margin: 0 0 12px; font-size: 11.5px; color: var(--faint); }
.chart { min-height: 120px; }
.chart-svg { display: block; width: 100%; height: auto; overflow: visible; }
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  margin: 2px 0 8px;
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
}
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { display: inline-block; width: 14px; height: 3px; background: var(--mint); }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
th {
  padding: 8px;
  border-bottom: 1px solid var(--line);
  font-family: var(--mono);
  font-size: 9.5px;
  font-weight: 500;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  text-align: left;
  color: var(--faint);
  line-height: 1.25;
}
td {
  padding: 9px 8px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.045);
  font-variant-numeric: tabular-nums;
  overflow-wrap: anywhere;
}
td:first-child { white-space: nowrap; }
tr:last-child td { border-bottom: none; }
th.num, td.num { text-align: right; font-family: var(--mono); }
td.num { white-space: nowrap; }
td.gate { font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; white-space: nowrap; }
td.gate.pass { color: var(--pass); }
td.gate.fail { color: var(--fail); }
#metrics-section { display: flex; flex-direction: column; }
#metrics-section .table-wrap { flex: 1 1 auto; }
.gate-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 22px;
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid var(--line);
}
.gate-summary div { display: flex; align-items: baseline; gap: 8px; }
.gate-summary .k {
  font-family: var(--mono);
  font-size: 9.5px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--faint);
}
.gate-summary .v { font-family: var(--mono); font-size: 20px; line-height: 1; color: var(--text); }
.gate-summary .v.pass { color: var(--pass); }
.gate-summary .v.fail { color: var(--fail); }
.note {
  margin-top: 16px;
  padding: 14px 16px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-left: 2px solid var(--amber);
  border-radius: 2px;
}
.note h2 { margin-bottom: 6px; }
.note p { margin: 0; white-space: pre-wrap; color: #d4d1ca; font-size: 13px; }
footer {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 10px;
  margin-top: 26px;
  padding-top: 14px;
  border-top: 1px solid var(--line);
  font-family: var(--mono);
  font-size: 9.5px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--faint);
}
@media (max-width: 920px) {
  .panel, .span-7, .span-5 { grid-column: span 12; }
}
@media (max-width: 720px) {
  .masthead { align-items: flex-start; }
  .telemetry { width: 100%; }
  .telemetry .row { justify-content: flex-start; }
  .telemetry .v { text-align: left; }
  .panel { padding: 15px; }
  th, td { padding: 8px 6px; }
  table { font-size: 12px; }
  td.gate { font-size: 10px; }
}
@media (prefers-reduced-motion: reduce) {
  .live { animation: none; }
  .progress-fill { transition: none; }
}
</style>
</head>
<body>
<main class="shell">
  <header class="masthead">
    <div class="brand">
      <h1><span class="live" aria-hidden="true"></span>COSMOVERSE</h1>
      <p class="kicker">hybrid cosmological inference / live run telemetry</p>
    </div>
    <div class="telemetry">
      <div class="row"><span class="k">stage</span><span class="v" id="stage">—</span></div>
      <div class="row"><span class="k">updated</span><span class="v" id="stamp">—</span></div>
      <div class="row"><span class="k">device</span><span class="v" id="device">—</span></div>
    </div>
  </header>

  <section class="progress-block" aria-label="Run progress">
    <div class="progress-head">
      <span class="label">progress</span>
      <span class="value" id="progress-value">—</span>
    </div>
    <div
      class="progress"
      id="progress"
      role="progressbar"
      aria-label="Run progress"
      aria-valuemin="0"
      aria-valuemax="100"
      aria-valuenow="0"
    >
      <div class="progress-fill" id="progress-fill"></div>
    </div>
  </section>

  <div class="grid">
    <section class="panel span-7" id="coverage-section" hidden>
      <h2>empirical coverage</h2>
      <p class="hint">68% and 95% credible intervals per group. Dashed references: 0.6827 and 0.9545.</p>
      <div class="legend" id="coverage-legend"></div>
      <div class="chart" id="coverage-chart"></div>
    </section>

    <section class="panel span-5" id="metrics-section" hidden>
      <h2>transfer metrics</h2>
      <p class="hint">Residual median bias in units of σ and the TARP-ICE shift, against the calibration gate.</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>group</th>
              <th class="num">median bias</th>
              <th class="num">Δ TARP-ICE</th>
              <th>gate</th>
            </tr>
          </thead>
          <tbody id="metrics-body"></tbody>
        </table>
      </div>
      <div class="gate-summary">
        <div><span class="k">passed</span><span class="v pass" id="pass-count">0</span></div>
        <div><span class="k">failed</span><span class="v fail" id="fail-count">0</span></div>
        <div><span class="k">groups</span><span class="v" id="group-count">0</span></div>
      </div>
    </section>

    <section class="panel span-7" id="loss-section" hidden>
      <h2>training loss</h2>
      <div class="legend" id="loss-legend"></div>
      <div class="chart" id="loss-chart"></div>
    </section>

    <section class="panel span-5" id="posterior-section" hidden>
      <h2>posterior geometry</h2>
      <div class="legend" id="posterior-legend"></div>
      <div class="chart" id="posterior-chart"></div>
    </section>
  </div>

  <section class="note" id="note-section" hidden>
    <h2>run note</h2>
    <p id="note-text"></p>
  </section>

  <footer>
    <span>COSMOVERSE / validation telemetry</span>
    <span>standalone artifact / auto-refresh 2 s</span>
  </footer>
</main>

<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  var COLOR = {
    text: "#e9e7e2",
    muted: "#9aa1ab",
    faint: "#6b727c",
    line: "#22262d",
    amber: "#ffb347",
    mint: "#79d6c0",
    pass: "#4ade80",
    fail: "#ff6b6b",
    ink: "#07080a",
  };

  function readState() {
    var node = document.getElementById("payload");
    if (!node) return {};
    try {
      return JSON.parse(node.textContent || "{}");
    } catch (error) {
      return {};
    }
  }

  var state = readState();
  var payload = state.payload && typeof state.payload === "object" ? state.payload : {};

  function isNum(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function num(value) {
    return isNum(value) ? value : null;
  }

  function fixed(value, digits) {
    var valueNum = num(value);
    return valueNum === null ? "—" : valueNum.toFixed(digits);
  }

  function tickLabel(value, span) {
    var magnitude = Math.abs(value);
    if (magnitude !== 0 && (magnitude >= 10000 || magnitude < 0.001)) {
      return value.toExponential(1);
    }
    var decimals = span >= 100 ? 0 : span >= 10 ? 1 : span >= 1 ? 2 : 3;
    return value.toFixed(decimals);
  }

  function shortLabel(value, limit) {
    var text = String(value);
    return text.length > limit ? text.slice(0, limit - 1) + "…" : text;
  }

  function clamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
  }

  function chartWidth(holder, fallback, max) {
    return clamp(Math.round(holder.clientWidth || fallback), 300, max);
  }

  function svgNode(tag, attrs) {
    var node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) {
        node.setAttribute(key, String(attrs[key]));
      });
    }
    return node;
  }

  function svgText(x, y, text, attrs) {
    var node = svgNode(
      "text",
      Object.assign(
        {
          x: x,
          y: y,
          "font-family": MONO,
          "font-size": 10,
          fill: COLOR.faint,
        },
        attrs || {}
      )
    );
    node.textContent = text;
    return node;
  }

  function add(parent) {
    for (var index = 1; index < arguments.length; index += 1) {
      parent.appendChild(arguments[index]);
    }
    return parent;
  }

  function renderLegend(id, items) {
    var holder = document.getElementById(id);
    if (!holder) return;
    holder.textContent = "";
    items.forEach(function (item) {
      var entry = document.createElement("span");
      var swatch = document.createElement("i");
      swatch.style.background = item.color;
      entry.appendChild(swatch);
      entry.appendChild(document.createTextNode(item.label));
      holder.appendChild(entry);
    });
  }

  function renderHeader() {
    var stage = state.stage === undefined || state.stage === null ? "" : String(state.stage);
    if (stage.length === 0) stage = "unlabeled stage";
    document.getElementById("stage").textContent = stage;
    document.getElementById("stamp").textContent =
      state.generated_at === undefined || state.generated_at === null ? "—" : String(state.generated_at);
    document.title = "COSMOVERSE | " + stage;

    var device = payload.device && typeof payload.device === "object" ? payload.device : {};
    var parts = [device.backend, device.gpu_name]
      .filter(function (item) {
        return item !== undefined && item !== null && String(item).length > 0;
      })
      .map(String);
    document.getElementById("device").textContent = parts.length ? parts.join(" · ") : "not reported";

    var progress = num(payload.progress);
    var clamped = progress === null ? 0 : Math.max(0, Math.min(1, progress));
    document.getElementById("progress-fill").style.transform = "scaleX(" + clamped + ")";
    document.getElementById("progress-value").textContent =
      progress === null ? "—" : (clamped * 100).toFixed(1) + "%";
    document.getElementById("progress").setAttribute("aria-valuenow", String(Math.round(clamped * 100)));
  }

  function renderCoverage(rows) {
    if (!Array.isArray(rows) || rows.length === 0) return;
    document.getElementById("coverage-section").hidden = false;
    renderLegend("coverage-legend", [
      { label: "68% interval", color: COLOR.mint },
      { label: "95% interval", color: COLOR.amber },
    ]);

    var holder = document.getElementById("coverage-chart");
    holder.textContent = "";

    var width = chartWidth(holder, 720, 720);
    var narrow = width < 560;
    var height = 300;
    var margin = {
      top: 26,
      right: narrow ? 54 : 64,
      bottom: narrow ? 76 : 44,
      left: narrow ? 40 : 48,
    };
    var plotW = width - margin.left - margin.right;
    var plotH = height - margin.top - margin.bottom;
    var y = function (value) {
      return margin.top + (1 - value) * plotH;
    };

    var root = svgNode("svg", {
      viewBox: "0 0 " + width + " " + height,
      role: "img",
      "aria-label": "Empirical coverage for 68 and 95 percent intervals",
    });
    root.setAttribute("class", "chart-svg");

    [0, 0.25, 0.5, 0.75, 1].forEach(function (tick) {
      var gridY = y(tick);
      add(
        root,
        svgNode("line", {
          x1: margin.left,
          y1: gridY,
          x2: margin.left + plotW,
          y2: gridY,
          stroke: COLOR.line,
          "stroke-width": 1,
        })
      );
      add(root, svgText(margin.left - 10, gridY + 3, tick.toFixed(2), { "text-anchor": "end" }));
    });

    [
      { value: 0.6827, label: "0.6827" },
      { value: 0.9545, label: "0.9545" },
    ].forEach(function (reference) {
      var gridY = y(reference.value);
      add(
        root,
        svgNode("line", {
          x1: margin.left,
          y1: gridY,
          x2: margin.left + plotW,
          y2: gridY,
          stroke: COLOR.text,
          "stroke-width": 1,
          "stroke-dasharray": "5 5",
          opacity: 0.4,
        })
      );
      add(
        root,
        svgText(margin.left + plotW + 8, gridY + 3, reference.label, {
          "text-anchor": "start",
          fill: COLOR.muted,
        })
      );
    });

    var band = plotW / rows.length;
    var barWidth = Math.max(8, Math.min(26, band * 0.24));
    var gap = Math.max(2, Math.min(9, band * 0.06));
    var showValues = rows.length <= 6 && !narrow;

    rows.forEach(function (row, index) {
      var record = row && typeof row === "object" ? row : {};
      var label =
        record.label === undefined || record.label === null ? "group " + (index + 1) : String(record.label);
      var center = margin.left + band * (index + 0.5);
      var cov68 = num(record.cov68);
      var cov95 = num(record.cov95);

      var labelY = margin.top + plotH + (narrow ? 16 : 20);
      var labelX = center + (narrow ? 4 : 0);
      var labelNode = svgText(labelX, labelY, shortLabel(label, narrow ? 20 : 22), {
        "text-anchor": narrow ? "end" : "middle",
        fill: COLOR.muted,
      });
      if (narrow) {
        labelNode.setAttribute("transform", "rotate(-35 " + labelX + " " + labelY + ")");
      }
      add(root, labelNode);

      if (cov68 !== null) {
        var value68 = Math.max(0, Math.min(1, cov68));
        var bar68 = svgNode("rect", {
          x: center - gap - barWidth,
          y: y(value68),
          width: barWidth,
          height: value68 * plotH,
          fill: COLOR.mint,
        });
        var title68 = svgNode("title");
        title68.textContent = label + " 68% coverage: " + cov68.toFixed(4);
        bar68.appendChild(title68);
        add(root, bar68);
        if (showValues) {
          add(
            root,
            svgText(center - gap - barWidth / 2, y(value68) - 6, value68.toFixed(2), {
              "text-anchor": "middle",
              fill: COLOR.mint,
              "font-size": 9,
            })
          );
        }
      }

      if (cov95 !== null) {
        var value95 = Math.max(0, Math.min(1, cov95));
        var bar95 = svgNode("rect", {
          x: center + gap,
          y: y(value95),
          width: barWidth,
          height: value95 * plotH,
          fill: COLOR.amber,
        });
        var title95 = svgNode("title");
        title95.textContent = label + " 95% coverage: " + cov95.toFixed(4);
        bar95.appendChild(title95);
        add(root, bar95);
        if (showValues) {
          add(
            root,
            svgText(center + gap + barWidth / 2, y(value95) - 6, value95.toFixed(2), {
              "text-anchor": "middle",
              fill: COLOR.amber,
              "font-size": 9,
            })
          );
        }
      }
    });

    holder.appendChild(root);
  }

  function renderMetrics(rows) {
    if (!Array.isArray(rows) || rows.length === 0) return;
    document.getElementById("metrics-section").hidden = false;
    var body = document.getElementById("metrics-body");
    body.textContent = "";
    var passed = 0;
    var failed = 0;

    rows.forEach(function (row) {
      var record = row && typeof row === "object" ? row : {};
      var tr = document.createElement("tr");

      var labelCell = document.createElement("td");
      var labelText =
        record.label === undefined || record.label === null ? "group" : String(record.label);
      labelCell.textContent = shortLabel(labelText, 22);
      labelCell.title = labelText;
      tr.appendChild(labelCell);

      var biasCell = document.createElement("td");
      biasCell.className = "num";
      biasCell.textContent = fixed(record.median_bias, 2);
      tr.appendChild(biasCell);

      var iceCell = document.createElement("td");
      iceCell.className = "num";
      iceCell.textContent = fixed(record.delta_ice, 3);
      tr.appendChild(iceCell);

      var gate = record.gate === undefined || record.gate === null ? "" : String(record.gate).toUpperCase();
      var gateCell = document.createElement("td");
      gateCell.className = "gate";
      gateCell.textContent = gate || "—";
      if (gate === "PASS") {
        gateCell.classList.add("pass");
        passed += 1;
      }
      if (gate === "FAIL") {
        gateCell.classList.add("fail");
        failed += 1;
      }
      tr.appendChild(gateCell);

      body.appendChild(tr);
    });

    document.getElementById("pass-count").textContent = String(passed);
    document.getElementById("fail-count").textContent = String(failed);
    document.getElementById("group-count").textContent = String(rows.length);
  }

  function renderLoss() {
    var source = Array.isArray(payload.loss_history) ? payload.loss_history.filter(isNum) : [];
    var fine = Array.isArray(payload.fine_tune_loss_history)
      ? payload.fine_tune_loss_history.filter(isNum)
      : [];
    if (source.length === 0 && fine.length === 0) return;

    document.getElementById("loss-section").hidden = false;
    var series = [];
    if (source.length) series.push({ label: "source", color: COLOR.amber, data: source });
    if (fine.length) series.push({ label: "fine-tune", color: COLOR.mint, data: fine });
    renderLegend(
      "loss-legend",
      series.map(function (item) {
        return { label: item.label, color: item.color };
      })
    );

    var holder = document.getElementById("loss-chart");
    holder.textContent = "";

    var width = chartWidth(holder, 720, 720);
    var narrow = width < 480;
    var height = 290;
    var margin = {
      top: 18,
      right: narrow ? 14 : 22,
      bottom: 44,
      left: narrow ? 50 : 64,
    };
    var plotW = width - margin.left - margin.right;
    var plotH = height - margin.top - margin.bottom;

    var all = source.concat(fine);
    var lo = Math.min.apply(null, all);
    var hi = Math.max.apply(null, all);
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return;
    if (hi === lo) {
      var span = Math.abs(hi) * 0.1 || 1;
      lo -= span;
      hi += span;
    }
    var pad = (hi - lo) * 0.08;
    lo -= pad;
    hi += pad;

    var maxLen = Math.max(source.length, fine.length);
    var lastIndex = Math.max(1, maxLen - 1);
    var x = function (index) {
      if (maxLen === 1) return margin.left + plotW / 2;
      return margin.left + (index / lastIndex) * plotW;
    };
    var y = function (value) {
      return margin.top + (1 - (value - lo) / (hi - lo)) * plotH;
    };

    var root = svgNode("svg", {
      viewBox: "0 0 " + width + " " + height,
      role: "img",
      "aria-label": "Training loss history",
    });
    root.setAttribute("class", "chart-svg");

    for (var tick = 0; tick < 5; tick += 1) {
      var value = lo + ((hi - lo) * tick) / 4;
      var gridY = y(value);
      add(
        root,
        svgNode("line", {
          x1: margin.left,
          y1: gridY,
          x2: margin.left + plotW,
          y2: gridY,
          stroke: COLOR.line,
          "stroke-width": 1,
        })
      );
      add(root, svgText(margin.left - 10, gridY + 3, tickLabel(value, hi - lo), { "text-anchor": "end" }));
    }

    var seen = {};
    var tickCount = maxLen === 1 ? 1 : 5;
    for (var step = 0; step < tickCount; step += 1) {
      var index = tickCount === 1 ? 0 : Math.round((step * lastIndex) / 4);
      if (seen[index]) continue;
      seen[index] = true;
      var gridX = x(index);
      add(
        root,
        svgNode("line", {
          x1: gridX,
          y1: margin.top + plotH,
          x2: gridX,
          y2: margin.top + plotH + 4,
          stroke: COLOR.line,
          "stroke-width": 1,
        })
      );
      add(root, svgText(gridX, margin.top + plotH + 18, String(index), { "text-anchor": "middle" }));
    }

    add(
      root,
      svgText(14, margin.top + plotH / 2, "loss", {
        "text-anchor": "middle",
        fill: COLOR.muted,
        transform: "rotate(-90 14 " + (margin.top + plotH / 2) + ")",
      })
    );
    add(
      root,
      svgText(margin.left + plotW, height - 6, "step", { "text-anchor": "end", fill: COLOR.muted })
    );

    series.forEach(function (item) {
      var path = "";
      item.data.forEach(function (value, index) {
        path += (index === 0 ? "M " : "L ") + x(index).toFixed(2) + " " + y(value).toFixed(2) + " ";
      });
      add(
        root,
        svgNode("path", {
          d: path.trim(),
          fill: "none",
          stroke: item.color,
          "stroke-width": 1.8,
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
        })
      );
      if (item.data.length <= 60) {
        item.data.forEach(function (value, index) {
          add(
            root,
            svgNode("circle", {
              cx: x(index),
              cy: y(value),
              r: 2.4,
              fill: item.color,
              stroke: COLOR.ink,
              "stroke-width": 1,
            })
          );
        });
      }
    });

    holder.appendChild(root);
  }

  function starPoints(cx, cy, outer, inner) {
    var points = [];
    for (var index = 0; index < 10; index += 1) {
      var radius = index % 2 === 0 ? outer : inner;
      var angle = -Math.PI / 2 + (index * Math.PI) / 5;
      points.push((cx + radius * Math.cos(angle)).toFixed(2) + "," + (cy + radius * Math.sin(angle)).toFixed(2));
    }
    return points.join(" ");
  }

  function renderPosterior() {
    var posterior = payload.posterior;
    if (!posterior || typeof posterior !== "object") return;

    var samples = Array.isArray(posterior.samples)
      ? posterior.samples.filter(function (pair) {
          return Array.isArray(pair) && pair.length >= 2 && isNum(pair[0]) && isNum(pair[1]);
        })
      : [];
    var truth =
      Array.isArray(posterior.truth) &&
      posterior.truth.length >= 2 &&
      isNum(posterior.truth[0]) &&
      isNum(posterior.truth[1])
        ? [posterior.truth[0], posterior.truth[1]]
        : null;
    if (samples.length === 0 && !truth) return;

    document.getElementById("posterior-section").hidden = false;
    var legendItems = [];
    if (truth) legendItems.push({ label: "truth", color: COLOR.amber });
    if (samples.length) legendItems.push({ label: "samples", color: COLOR.mint });
    renderLegend("posterior-legend", legendItems);

    var holder = document.getElementById("posterior-chart");
    holder.textContent = "";

    var xs = samples.map(function (pair) {
      return pair[0];
    });
    var ys = samples.map(function (pair) {
      return pair[1];
    });
    if (truth) {
      xs.push(truth[0]);
      ys.push(truth[1]);
    }

    var domain = function (values) {
      var low = Math.min.apply(null, values);
      var high = Math.max.apply(null, values);
      if (high === low) {
        var half = Math.abs(high) * 0.05 || 0.5;
        low -= half;
        high += half;
      }
      var room = (high - low) * 0.1;
      return [low - room, high + room];
    };
    var xDomain = domain(xs);
    var yDomain = domain(ys);

    var width = chartWidth(holder, 520, 560);
    var height = Math.round(width * 0.82);
    var margin = {
      top: 20,
      right: 20,
      bottom: 52,
      left: width < 400 ? 50 : 62,
    };
    var plotW = width - margin.left - margin.right;
    var plotH = height - margin.top - margin.bottom;
    var px = function (value) {
      return margin.left + ((value - xDomain[0]) / (xDomain[1] - xDomain[0])) * plotW;
    };
    var py = function (value) {
      return margin.top + (1 - (value - yDomain[0]) / (yDomain[1] - yDomain[0])) * plotH;
    };

    var root = svgNode("svg", {
      viewBox: "0 0 " + width + " " + height,
      role: "img",
      "aria-label": "Posterior samples with truth marker",
    });
    root.setAttribute("class", "chart-svg");

    for (var tick = 0; tick < 5; tick += 1) {
      var xValue = xDomain[0] + ((xDomain[1] - xDomain[0]) * tick) / 4;
      var gridX = px(xValue);
      add(
        root,
        svgNode("line", {
          x1: gridX,
          y1: margin.top,
          x2: gridX,
          y2: margin.top + plotH,
          stroke: COLOR.line,
          "stroke-width": 1,
        })
      );
      add(root, svgText(gridX, margin.top + plotH + 18, tickLabel(xValue, xDomain[1] - xDomain[0]), { "text-anchor": "middle" }));

      var yValue = yDomain[0] + ((yDomain[1] - yDomain[0]) * tick) / 4;
      var gridY = py(yValue);
      add(
        root,
        svgNode("line", {
          x1: margin.left,
          y1: gridY,
          x2: margin.left + plotW,
          y2: gridY,
          stroke: COLOR.line,
          "stroke-width": 1,
        })
      );
      add(root, svgText(margin.left - 10, gridY + 3, tickLabel(yValue, yDomain[1] - yDomain[0]), { "text-anchor": "end" }));
    }

    var label =
      posterior.label === undefined || posterior.label === null ? "" : String(posterior.label);
    var parts = label.split(/\s+vs\.?\s+/i);
    var xLabel = parts[0] || "x";
    var yLabel = parts.length > 1 ? parts[1] : "y";
    add(
      root,
      svgText(margin.left + plotW / 2, height - 12, xLabel, {
        "text-anchor": "middle",
        fill: COLOR.muted,
      })
    );
    add(
      root,
      svgText(16, margin.top + plotH / 2, yLabel, {
        "text-anchor": "middle",
        fill: COLOR.muted,
        transform: "rotate(-90 16 " + (margin.top + plotH / 2) + ")",
      })
    );

    var stride = Math.max(1, Math.ceil(samples.length / 3000));
    samples.forEach(function (pair, index) {
      if (index % stride !== 0) return;
      add(
        root,
        svgNode("circle", {
          cx: px(pair[0]),
          cy: py(pair[1]),
          r: 2.6,
          fill: COLOR.mint,
          opacity: 0.45,
        })
      );
    });

    if (truth) {
      var star = svgNode("polygon", {
        points: starPoints(px(truth[0]), py(truth[1]), 9, 3.8),
        fill: COLOR.amber,
        stroke: COLOR.ink,
        "stroke-width": 1.4,
      });
      var title = svgNode("title");
      title.textContent = "truth: " + truth[0] + ", " + truth[1];
      star.appendChild(title);
      add(root, star);
    }

    holder.appendChild(root);
  }

  function renderNote() {
    if (payload.note === undefined || payload.note === null || String(payload.note).length === 0) return;
    document.getElementById("note-text").textContent = String(payload.note);
    document.getElementById("note-section").hidden = false;
  }

  renderHeader();
  renderCoverage(payload.coverage);
  renderMetrics(payload.metrics);
  renderLoss();
  renderPosterior();
  renderNote();
})();
</script>
</body>
</html>
"""
