"""
DeepWaterLog decision-support UI.

Flask dashboard for the Deep Learning-Enabled Hyperspectral Remote Sensing for
Agricultural Waterlogging Detection project.

Shows, for the most recent pipeline run:
  - evaluation metrics (accuracy, precision, recall, F1, IoU)
  - per-class pixel and area statistics
  - the classified waterlogging map and the severity map
  - provenance (run ID, scenario, device, timings, warnings)

Endpoints
---------
GET  /                     dashboard
GET  /api/summary          summary of the latest run
GET  /api/runs             list of recorded runs
GET  /api/run/<run_id>     full run record
POST /api/export_report    write a Markdown report for the latest run
GET  /results/<filename>   serve run artefacts (maps, reports, logs)

Theme: Smart AgriTech - Precision Farming and Agro Sustainability
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template_string, send_from_directory

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import config  # noqa: E402
from core.mapper import export_report  # noqa: E402

LOGGER = logging.getLogger("deepwaterlog.ui")
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

RESULTS_DIR = config.RESULTS_DIR
MODELS_DIR = config.MODELS_DIR
CLASS_NAMES = config.CLASSES
CLASS_COLORS = config.CLASS_COLORS

# Hex colours used by the legend (consistent with core/mapper.DEFAULT_PALETTE)
CLASS_HEX = {
    0: "#d62828",
    1: "#40916c",
    2: "#f4a261",
    3: "#2a6fb7",
    4: "#969696",
}

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeepWaterLog - Agricultural Waterlogging Detection</title>
<style>
  :root {
    --ink: #10263b; --muted: #5b7288; --line: #dfe7ee;
    --bg: #f4f7fa; --card: #ffffff;
    --waterlogged: #d62828; --healthy: #40916c; --stressed: #f4a261;
    --water: #2a6fb7; --other: #969696;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--ink); line-height: 1.5;
    padding: 24px; max-width: 1280px; margin: 0 auto;
  }
  header { margin-bottom: 20px; }
  h1 { font-size: 24px; letter-spacing: -0.3px; }
  .sub { color: var(--muted); font-size: 13px; margin-top: 4px; }
  .theme-pill {
    display: inline-block; margin-top: 8px; padding: 3px 10px;
    background: #e6f2ec; color: #1f6b4a; border-radius: 20px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.4px;
    text-transform: uppercase;
  }
  .grid { display: grid; gap: 14px; }
  .kpis { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
  .card {
    background: var(--card); border: 1px solid var(--line);
    border-radius: 10px; padding: 16px;
  }
  .kpi-label {
    font-size: 10px; font-weight: 700; letter-spacing: 0.7px;
    text-transform: uppercase; color: var(--muted);
  }
  .kpi-value {
    font-size: 26px; font-weight: 700; margin-top: 6px;
    font-variant-numeric: tabular-nums;
  }
  .kpi-unit { font-size: 12px; font-weight: 600; color: var(--muted); }
  h2 {
    font-size: 13px; text-transform: uppercase; letter-spacing: 0.7px;
    color: var(--muted); margin: 26px 0 10px;
  }
  .two-col { grid-template-columns: 1fr 1fr; }
  @media (max-width: 860px) { .two-col { grid-template-columns: 1fr; } }
  .map-img {
    width: 100%; display: block; border-radius: 6px;
    image-rendering: pixelated; background: #eef2f6;
  }
  .legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; }
  .swatch { width: 12px; height: 12px; border-radius: 3px; flex: 0 0 auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--line); }
  th { font-size: 10px; text-transform: uppercase; letter-spacing: 0.6px; color: var(--muted); }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .bar-track { background: #eef2f6; border-radius: 4px; height: 8px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; }
  .empty { color: var(--muted); font-size: 14px; }
  code {
    background: #eef2f6; padding: 2px 6px; border-radius: 4px;
    font-size: 12px; font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  pre {
    background: #10263b; color: #d7e3ee; padding: 14px; border-radius: 8px;
    font-size: 12px; overflow-x: auto; line-height: 1.6;
  }
  button {
    background: var(--ink); color: #fff; border: 0; border-radius: 6px;
    padding: 9px 16px; font-weight: 600; font-size: 13px; cursor: pointer;
  }
  button:hover { background: #1c4670; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .status { font-size: 12px; color: var(--muted); margin-top: 8px; }
  .warn {
    background: #fff8e6; border: 1px solid #f1d9a0; color: #7a5a12;
    border-radius: 8px; padding: 10px 12px; font-size: 12px; margin-top: 8px;
  }
  .pill {
    display: inline-block; padding: 1px 8px; border-radius: 10px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.4px;
  }
  .pill-ok { background: #e6f2ec; color: #1f6b4a; }
  .footer { margin-top: 30px; color: var(--muted); font-size: 11px; text-align: center; }
</style>
</head>
<body>

<header>
  <h1>DeepWaterLog</h1>
  <div class="sub">
    Deep Learning-Enabled Hyperspectral Remote Sensing for
    Agricultural Waterlogging Detection
  </div>
  <span class="theme-pill">Smart AgriTech - Precision Farming and Agro Sustainability</span>
</header>

<div id="root">
  <p class="empty">Loading latest run...</p>
</div>

<script>
const CLASS_META = [
  { id: 0, name: "Waterlogged soil",  hex: "#d62828", note: "saturated root zone" },
  { id: 1, name: "Healthy crop",      hex: "#40916c", note: "control class" },
  { id: 2, name: "Stressed crop",     hex: "#f4a261", note: "waterlogging damage" },
  { id: 3, name: "Open water",        hex: "#2a6fb7", note: "standing water" },
  { id: 4, name: "Non-agricultural",  hex: "#969696", note: "roads, built-up" }
];

function fmt(value, digits) {
  if (value === null || value === undefined) return "&mdash;";
  return Number(value).toFixed(digits === undefined ? 2 : digits);
}

function pct(value) {
  if (value === null || value === undefined) return "&mdash;";
  return (Number(value) * 100).toFixed(1) + "%";
}

function kpi(label, value, unit) {
  return `<div class="card">
    <div class="kpi-label">${label}</div>
    <div class="kpi-value">${value}<span class="kpi-unit"> ${unit || ""}</span></div>
  </div>`;
}

function legendHtml() {
  return `<div class="legend">` + CLASS_META.map(c =>
    `<div class="legend-item">
       <span class="swatch" style="background:${c.hex}"></span>
       <span>${c.name}</span>
     </div>`).join("") + `</div>`;
}

function classRows(map) {
  if (!map || !map.class_counts) return "";
  const total = Object.values(map.class_counts).reduce((a, b) => a + b, 0) || 1;
  return CLASS_META.map(c => {
    const count = map.class_counts[c.id] || 0;
    const area = (map.class_areas_ha && map.class_areas_ha[c.id]) || 0;
    const share = count / total;
    return `<tr>
      <td><span class="swatch" style="display:inline-block;background:${c.hex};margin-right:7px"></span>${c.name}</td>
      <td class="num">${count.toLocaleString()}</td>
      <td class="num">${area.toFixed(2)}</td>
      <td style="width:34%">
        <div class="bar-track"><div class="bar-fill" style="width:${(share*100).toFixed(1)}%;background:${c.hex}"></div></div>
      </td>
      <td class="num">${(share*100).toFixed(1)}%</td>
    </tr>`;
  }).join("");
}

async function loadSummary() {
  const root = document.getElementById("root");
  try {
    const response = await fetch("/api/summary");
    const data = await response.json();

    if (!data.run_id) {
      root.innerHTML = `
        <div class="card">
          <p class="empty">No pipeline run recorded yet.</p>
          <p class="status">Run the health check to generate results:</p>
          <pre>python test_pipeline.py --bands 30 --size 64 --patch-size 16 --stride 8</pre>
          <p class="status">Then refresh this page.</p>
        </div>`;
      return;
    }

    const metrics = data.metrics || {};
    const unet = metrics.unet || {};
    const cnn = metrics.cnn || {};
    const map = data.map || {};
    const artifacts = map.artifacts || {};

    const mapBlock = artifacts.class_map_png
      ? `<img class="map-img" src="/results/${artifacts.class_map_png.split(/[\\\\/]/).pop()}" alt="Classified waterlogging map">`
      : `<p class="empty">No map image stored for this run. Re-run the pipeline to generate one.</p>`;

    const sevBlock = artifacts.severity_png
      ? `<img class="map-img" src="/results/${artifacts.severity_png.split(/[\\\\/]/).pop()}" alt="Waterlogging severity map">`
      : `<p class="empty">No severity image stored for this run.</p>`;

    const warnBlock = (data.warnings && data.warnings.length)
      ? `<h2>Warnings</h2><div class="warn">${data.warnings.map(w => "&bull; " + w).join("<br>")}</div>`
      : "";

    root.innerHTML = `
      <div class="grid kpis">
        ${kpi("U-Net accuracy", fmt(unet.accuracy, 4), "")}
        ${kpi("U-Net F1 (macro)", fmt(unet.f1_macro, 4), "")}
        ${kpi("U-Net IoU (macro)", fmt(unet.iou_macro, 4), "")}
        ${kpi("CNN F1 (macro)", fmt(cnn.f1_macro, 4), "")}
        ${kpi("Waterlogged area", fmt(map.waterlogged_area_ha), "ha")}
        ${kpi("Waterlogged share", pct(map.waterlogged_fraction), "")}
      </div>

      <h2>Run provenance</h2>
      <div class="card">
        <table>
          <tr><th style="width:180px">Run ID</th><td><code>${data.run_id}</code></td></tr>
          <tr><th>Scenario</th><td>${data.scenario_id || "&mdash;"}</td></tr>
          <tr><th>Status</th><td><span class="pill pill-ok">${(data.status || "unknown").toUpperCase()}</span></td></tr>
          <tr><th>Device</th><td>${data.device || "&mdash;"}</td></tr>
          <tr><th>Processing time</th><td>${fmt(data.processing_seconds, 1)} s</td></tr>
          <tr><th>Bands</th><td>${data.bands_in || "&mdash;"} raw &rarr; ${data.bands_out || "&mdash;"} after selection</td></tr>
          <tr><th>Patches</th><td>${data.n_patches || "&mdash;"} (train ${data.train_count || 0} / val ${data.val_count || 0} / test ${data.test_count || 0})</td></tr>
        </table>
      </div>

      <h2>Waterlogging map</h2>
      <div class="grid two-col">
        <div class="card">
          <div class="kpi-label">Classified extent</div>
          ${mapBlock}
          ${legendHtml()}
        </div>
        <div class="card">
          <div class="kpi-label">Waterlogging severity (P(waterlogged soil) + P(stressed crop))</div>
          ${sevBlock}
          <div class="legend">
            <div class="legend-item"><span class="swatch" style="background:#ffeda0"></span><span>low</span></div>
            <div class="legend-item"><span class="swatch" style="background:#d62828"></span><span>high</span></div>
          </div>
        </div>
      </div>

      <h2>Per-class breakdown</h2>
      <div class="card">
        <table>
          <thead><tr>
            <th>Class</th><th class="num">Pixels</th><th class="num">Area (ha)</th>
            <th>Share</th><th class="num">%</th>
          </tr></thead>
          <tbody>${classRows(map)}</tbody>
        </table>
        <p class="status">
          Total mapped area ${fmt(map.total_area_ha)} ha at
          ${data.map_resolution_m || 10} m resolution.
          Waterlogged classes are <strong>waterlogged soil</strong> and <strong>stressed crop</strong>.
        </p>
      </div>

      <h2>Evaluation detail</h2>
      <div class="card">
        <table>
          <thead><tr><th>Model</th><th class="num">Accuracy</th><th class="num">Precision</th>
            <th class="num">Recall</th><th class="num">F1</th><th class="num">IoU</th></tr></thead>
          <tbody>
            <tr><td>CNN (patch classification)</td>
              <td class="num">${fmt(cnn.accuracy, 4)}</td>
              <td class="num">${fmt(cnn.precision_macro, 4)}</td>
              <td class="num">${fmt(cnn.recall_macro, 4)}</td>
              <td class="num">${fmt(cnn.f1_macro, 4)}</td>
              <td class="num">&mdash;</td></tr>
            <tr><td>U-Net (pixel segmentation)</td>
              <td class="num">${fmt(unet.accuracy, 4)}</td>
              <td class="num">${fmt(unet.precision_macro, 4)}</td>
              <td class="num">${fmt(unet.recall_macro, 4)}</td>
              <td class="num">${fmt(unet.f1_macro, 4)}</td>
              <td class="num">${fmt(unet.iou_macro, 4)}</td></tr>
          </tbody>
        </table>
      </div>

      ${warnBlock}

      <h2>Export</h2>
      <div class="card">
        <button id="btn-export">Generate Markdown report</button>
        <div class="status" id="export-status"></div>
      </div>

      <div class="footer">
        DeepWaterLog v${data.version || "1.0.0"} &middot;
        results are screening output, not a substitute for field inspection
      </div>
    `;

    document.getElementById("btn-export").onclick = async () => {
      const status = document.getElementById("export-status");
      status.textContent = "Generating...";
      try {
        const res = await fetch("/api/export_report", { method: "POST" });
        const payload = await res.json();
        status.textContent = payload.message || "Done.";
      } catch (error) {
        status.textContent = "Export failed: " + error.message;
      }
    };
  } catch (error) {
    root.innerHTML = `<div class="card"><p class="empty">
      Could not load results: ${error.message}</p></div>`;
  }
}

loadSummary();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _latest_run_path() -> Path | None:
    """Return the most recently written run record, or None."""
    if not RESULTS_DIR.exists():
        return None
    candidates = [
        p for p in RESULTS_DIR.glob("*.json")
        if p.name not in {"run_log.jsonl"}
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/healthz")
def healthz():
    """Liveness probe for hosting platforms."""
    return jsonify({"status": "ok", "service": "deepwaterlog-dashboard"})


@app.route("/api/summary")
def api_summary():
    """Return a flattened summary of the latest pipeline run."""
    path = _latest_run_path()
    if path is None:
        return jsonify({"run_id": None, "message": "No runs recorded yet."})

    try:
        record = _read_json(path)
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("Could not read run record %s: %s", path, error)
        return jsonify({"run_id": None, "message": f"Could not read run record: {error}"})

    metrics = record.get("metrics") or {}
    map_info = record.get("map") or {}
    dataset = record.get("dataset") or {}
    provenance = record.get("provenance") or {}
    band_sel = record.get("band_selection") or {}

    return jsonify({
        "run_id": record.get("run_id"),
        "scenario_id": record.get("scenario_id"),
        "status": record.get("status"),
        "version": provenance.get("version") or config.PROJECT_VERSION,
        "device": provenance.get("device"),
        "processing_seconds": record.get("processing_seconds"),
        "metrics": metrics,
        "map": map_info,
        "map_resolution_m": config.MAP_RESOLUTION_M,
        "bands_in": band_sel.get("n_input_bands"),
        "bands_out": band_sel.get("n_output_bands"),
        "n_patches": dataset.get("n_patches"),
        "train_count": dataset.get("train"),
        "val_count": dataset.get("val"),
        "test_count": dataset.get("test"),
        "warnings": record.get("warnings") or [],
    })


@app.route("/api/runs")
def api_runs():
    """List recorded runs, newest first."""
    if not RESULTS_DIR.exists():
        return jsonify({"runs": []})
    runs = []
    for path in RESULTS_DIR.glob("*.json"):
        try:
            record = _read_json(path)
        except Exception:  # noqa: BLE001
            continue
        runs.append({
            "run_id": record.get("run_id", path.stem),
            "scenario_id": record.get("scenario_id"),
            "status": record.get("status"),
            "processing_seconds": record.get("processing_seconds"),
            "modified": path.stat().st_mtime,
        })
    runs.sort(key=lambda r: r["modified"], reverse=True)
    return jsonify({"runs": runs})


@app.route("/api/run/<run_id>")
def api_run(run_id: str):
    """Return the full run record for a given run ID."""
    candidates = list(RESULTS_DIR.glob(f"{run_id}.json"))
    if not candidates:
        return jsonify({"error": f"Run '{run_id}' not found."}), 404
    return jsonify(_read_json(candidates[0]))


@app.route("/api/export_report", methods=["POST"])
def api_export_report():
    """Write a Markdown waterlogging report for the latest run."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "waterlogging_report.md"

    path = _latest_run_path()
    if path is None:
        message = "No run recorded yet. Run the pipeline first."
        out_path.write_text(f"# Waterlogging Detection Report\n\n{message}\n", encoding="utf-8")
        return jsonify({"message": message, "path": str(out_path)})

    record = _read_json(path)
    map_info = record.get("map") or {}
    metrics = record.get("metrics") or {}
    unet = metrics.get("unet") or {}
    cnn = metrics.get("cnn") or {}

    lines = [
        "# Waterlogging Detection Report",
        "",
        f"**Run:** `{record.get('run_id')}`  ",
        f"**Scenario:** {record.get('scenario_id')}  ",
        f"**Status:** {record.get('status')}  ",
        f"**Processing time:** {record.get('processing_seconds')} s",
        "",
        "## Evaluation",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | IoU |",
        "|---|---|---|---|---|---|",
        f"| CNN | {cnn.get('accuracy', float('nan')):.4f} | "
        f"{cnn.get('precision_macro', float('nan')):.4f} | "
        f"{cnn.get('recall_macro', float('nan')):.4f} | "
        f"{cnn.get('f1_macro', float('nan')):.4f} | - |",
        f"| U-Net | {unet.get('accuracy', float('nan')):.4f} | "
        f"{unet.get('precision_macro', float('nan')):.4f} | "
        f"{unet.get('recall_macro', float('nan')):.4f} | "
        f"{unet.get('f1_macro', float('nan')):.4f} | "
        f"{unet.get('iou_macro', float('nan')):.4f} |",
        "",
        "## Area summary",
        "",
        f"- Total mapped area: {map_info.get('total_area_ha') or 0:.2f} ha",
        f"- Waterlogged area: {map_info.get('waterlogged_area_ha') or 0:.2f} ha",
        f"- Waterlogged fraction: {(map_info.get('waterlogged_fraction') or 0):.2%}",
        "",
        "## Per-class breakdown",
        "",
        "| Class | Pixels | Area (ha) |",
        "|---|---|---|",
    ]

    class_counts = map_info.get("class_counts") or {}
    class_areas = map_info.get("class_areas_ha") or {}
    for index, name in enumerate(CLASS_NAMES):
        lines.append(
            f"| {name} | {class_counts.get(str(index), class_counts.get(index, 0))} | "
            f"{(class_areas.get(str(index)) or class_areas.get(index) or 0):.2f} |"
        )

    artifacts = map_info.get("artifacts") or {}
    if artifacts:
        lines += ["", "## Artefacts", ""]
        for key, value in artifacts.items():
            lines.append(f"- `{key}`: `{value}`")

    warnings = record.get("warnings") or []
    if warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- {w}" for w in warnings]

    lines += ["", "---", "", f"Theme: {config.THEME}"]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Report written to %s", out_path)

    return jsonify({
        "message": f"Report written to {out_path.name}",
        "path": str(out_path),
    })


@app.route("/results/<path:filename>")
def api_results_file(filename: str):
    """Serve run artefacts (maps, reports) from the results directory."""
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Starting DeepWaterLog UI on http://127.0.0.1:%s", config.UI_PORT)
    app.run(host="127.0.0.1", port=config.UI_PORT, debug=False, use_reloader=False)
