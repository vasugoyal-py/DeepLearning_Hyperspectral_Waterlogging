"""
DeepWaterLog decision-support UI.

Flask + Leaflet web application for visualising waterlogging extent and
severity maps, overlaying field boundaries, and exporting reports.

Theme: Smart AgriTech — Precision Farming and Agro Sustainability
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template_string, send_from_directory

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import config  # noqa: E402
from core.mapper import export_report, generate_maps, save_geotiff  # noqa: E402

LOGGER = logging.getLogger("deepwaterlog.ui")
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_folder="static")
app.config["JSON_SORT_KEYS"] = False

RESULTS_DIR = config.RESULTS_DIR
MODELS_DIR = config.MODELS_DIR
CLASS_NAMES = config.CLASSES
CLASS_COLORS = config.CLASS_COLORS

# ---------------------------------------------------------------------------
# HTML page (single-file Leaflet map)
# ---------------------------------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepWaterLog — Agricultural Waterlogging Detection</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css" />
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body, html { width: 100%; height: 100%; overflow: hidden; }
        #map { width: 100vw; height: 100vh; z-index: 1; }
        #sidebar {
            position: absolute; top: 12px; left: 12px; z-index: 1000;
            width: 340px; max-height: 92vh; overflow-y: auto;
            background: rgba(255, 255, 255, 0.95); padding: 16px;
            border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.25);
        }
        h1 { font-size: 17px; margin: 0 0 10px; color: #1a3c5a; }
        h2 { font-size: 14px; margin: 14px 0 6px; color: #1a3c5a; }
        p { font-size: 13px; line-height: 1.5; color: #333; margin: 0 0 10px; }
        .badge { display:inline-block; padding: 2px 8px; border-radius: 10px;
            font-size: 11px; font-weight: 700; color: #fff; margin-right: 6px; }
        .bg-waterlogged { background: #d9534f; } .bg-healthy { background: #5cb85c; }
        .bg-stressed { background: #f0ad4e; } .bg-water { background: #0275d8; }
        .bg-other { background: #777; }
        table { width: 100%; border-collapse: collapse; font-size: 12px; margin: 8px 0; }
        th, td { border: 1px solid #ddd; padding: 4px 6px; text-align: left; }
        th { background: #f5f5f5; }
        button { background: #1a3c5a; color: #fff; border: none; padding: 8px 14px;
            border-radius: 6px; cursor: pointer; font-weight: 600; width: 100%; margin-top: 8px; }
        button:hover { background: #2a5f8a; }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        .row { display: flex; gap: 8px; margin: 6px 0; }
        .col { flex: 1; }
        .muted { color: #888; font-size: 11px; }
    </style>
</head>
<body>
<div id="map"></div>
<div id="sidebar">
    <h1>DeepWaterLog</h1>
    <p style="font-size:12px;color:#666;">
        Deep Learning–Enabled Hyperspectral Remote Sensing for<br>
        Agricultural Waterlogging Detection
    </p>
    <p style="font-size:12px;color:#1a3c5a;font-weight:600;">
        Theme: Smart AgriTech — Precision Farming and Agro Sustainability
    </p>

    <h2>Map legend</h2>
    <div>
        <span class="badge bg-waterlogged">Waterlogged soil</span>
        <span class="badge bg-healthy">Healthy crop</span>
        <span class="badge bg-stressed">Stressed crop</span>
        <span class="badge bg-water">Open water</span>
        <span class="badge bg-other">Non-agricultural</span>
    </div>

    <h2>Summary</h2>
    <div id="summary">Loading…</div>

    <h2>Actions</h2>
    <div class="row">
        <div class="col"><button id="btn-refresh">Refresh maps</button></div>
        <div class="col"><button id="btn-export">Export report</button></div>
    </div>
    <p class="muted" id="status"></p>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
<script>
    const map = L.map('map', { center: [28.6, 77.2], zoom: 13 });
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    const classColors = {
        0: '#d9534f',  // waterlogged_soil
        1: '#5cb85c',  // healthy_crop
        2: '#f0ad4e',  // stressed_crop
        3: '#0275d8',  // open_water
        4: '#777777',  // non_agricultural
    };

    async function loadSummary() {
        try {
            const res = await fetch('/api/summary');
            const data = await res.json();
            const el = document.getElementById('summary');
            el.innerHTML = `
                <strong>Total area:</strong> ${data.total_area_ha?.toFixed(2) ?? '—'} ha<br>
                <strong>Waterlogged area:</strong> ${data.waterlogged_area_ha?.toFixed(2) ?? '—'} ha<br>
                <strong>Waterlogged fraction:</strong> ${data.waterlogged_fraction?.toFixed(2) ?? '—'}<br>
                <strong>Run:</strong> ${data.run_id ?? '—'}
            `;
            document.getElementById('status').textContent = 'Loaded summary.';
        } catch (e) {
            document.getElementById('summary').textContent = 'Summary unavailable.';
            document.getElementById('status').textContent = 'Error: ' + e.message;
        }
    }

    document.getElementById('btn-refresh').onclick = () => {
        document.getElementById('status').textContent = 'Refreshing…';
        loadSummary();
    };

    document.getElementById('btn-export').onclick = async () => {
        try {
            const res = await fetch('/api/export_report', { method: 'POST' });
            const data = await res.json();
            document.getElementById('status').textContent = data.message ?? 'Report exported.';
        } catch (e) {
            document.getElementById('status').textContent = 'Export failed: ' + e.message;
        }
    };

    loadSummary();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/api/summary")
def api_summary():
    """Return the latest run summary, if any."""
    log_path = RESULTS_DIR / "run_log.jsonl"
    if not log_path.exists():
        return jsonify({"run_id": None, "message": "No runs recorded yet."})

    # Read the last line
    last = None
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                last = line
    if not last:
        return jsonify({"run_id": None, "message": "No runs recorded yet."})

    record = json.loads(last)
    return jsonify({
        "run_id": record.get("run_id"),
        "status": record.get("status"),
        "processing_seconds": record.get("processing_seconds"),
        "cnn_f1": record.get("cnn_f1"),
        "unet_iou": record.get("unet_iou"),
        "waterlogged_area_ha": record.get("waterlogged_area_ha"),
        "total_area_ha": None,
        "waterlogged_fraction": None,
        "warnings": record.get("warnings", []),
    })


@app.route("/api/export_report", methods=["POST"])
def api_export_report():
    """Generate and export a waterlogging report from the latest results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "waterlogging_report.md"

    # If a map GeoTIFF exists in results, build a map result and report.
    geotiffs = list(RESULTS_DIR.glob("*.tif")) + list(RESULTS_DIR.glob("*.tiff"))
    if geotiffs:
        import numpy as np
        try:
            import rasterio
            with rasterio.open(geotiffs[0]) as src:
                arr = src.read(1)
                gt = src.transform.to_gdal() if hasattr(src.transform, "to_gdal") else None
                result = generate_maps(arr, geotransform=gt, crs=str(src.crs))
                result.output_path = str(geotiffs[0])
                report = export_report(result, path=str(out_path))
                return jsonify({"message": f"Report exported to {out_path}", "report": report})
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("Could not build report from GeoTIFF: %s", error)

    # Fallback: write a template report
    report = (
        "# Waterlogging Detection Report\n\n"
        "No classified map available yet. Run the pipeline first.\n"
        f"\nTheme: {config.THEME}\n"
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    return jsonify({"message": f"Template report written to {out_path}", "report": report})


@app.route("/results/<path:filename>")
def api_results_file(filename):
    """Serve files from the results directory (maps, reports, logs)."""
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Starting DeepWaterLog UI on port %s", config.UI_PORT)
    app.run(host="0.0.0.0", port=config.UI_PORT, debug=True)
