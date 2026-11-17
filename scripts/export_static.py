"""
Export a static snapshot of the DeepWaterLog dashboard.

GitHub Pages serves static files only, so it cannot run the Flask dashboard.
This script bakes the latest run record into a single `index.html` plus the map
images, producing a folder that can be published as-is.

Usage
-----
    python scripts/export_static.py
    python scripts/export_static.py --run-id demo_lowland_paddy_20261117T090836Z
    python scripts/export_static.py --out site

Outputs
-------
    site/index.html
    site/assets/<run_id>_map.png
    site/assets/<run_id>_severity.png
    site/.nojekyll
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import config  # noqa: E402

CLASS_META = [
    (0, "Waterlogged soil", "#d62828"),
    (1, "Healthy crop", "#40916c"),
    (2, "Stressed crop", "#f4a261"),
    (3, "Open water", "#2a6fb7"),
    (4, "Non-agricultural", "#969696"),
]


def _latest_run(results_dir: Path) -> Path | None:
    if not results_dir.exists():
        return None
    files = [p for p in results_dir.glob("*.json") if p.name != "run_log.jsonl"]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "&mdash;"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def _pct(value, digits: int = 1) -> str:
    if value is None:
        return "&mdash;"
    return f"{float(value) * 100:.{digits}f}%"


def _kpi(label: str, value: str, unit: str = "") -> str:
    unit_html = f'<span class="kpi-unit">{unit}</span>' if unit else ""
    return (
        f'<div class="card"><div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{value}{unit_html}</div></div>'
    )


def _legend() -> str:
    items = "".join(
        f'<div class="legend-item"><span class="swatch" style="background:{hex_}" '
        f'></span><span>{html.escape(name)}</span></div>'
        for _, name, hex_ in CLASS_META
    )
    return f'<div class="legend">{items}</div>'


def _class_rows(map_info: dict) -> str:
    counts = map_info.get("class_counts") or {}
    areas = map_info.get("class_areas_ha") or {}
    total = sum(int(v) for v in counts.values()) or 1

    rows = []
    for class_id, name, hex_ in CLASS_META:
        count = int(counts.get(str(class_id), counts.get(class_id, 0)) or 0)
        area = float(areas.get(str(class_id), areas.get(class_id, 0)) or 0)
        share = 100.0 * count / total
        rows.append(
            "<tr>"
            f'<td><span class="swatch inline" style="background:{hex_}"></span>'
            f"{html.escape(name)}</td>"
            f'<td class="num">{count:,}</td>'
            f'<td class="num">{area:.2f}</td>'
            f'<td class="bar-cell"><div class="bar-track"><div class="bar-fill" '
            f'style="width:{share:.1f}%;background:{hex_}"></div></div></td>'
            f'<td class="num">{share:.1f}%</td>'
            "</tr>"
        )
    return "\n".join(rows)


def build_html(record: dict, assets: dict[str, str], generated: str) -> str:
    metrics = record.get("metrics") or {}
    cnn = metrics.get("cnn") or {}
    unet = metrics.get("unet") or {}
    map_info = record.get("map") or {}
    dataset = record.get("dataset") or {}
    band_sel = record.get("band_selection") or {}
    provenance = record.get("provenance") or {}
    warnings = record.get("warnings") or []

    kpis = "".join([
        _kpi("U-Net accuracy", _fmt(unet.get("accuracy"), 4)),
        _kpi("U-Net F1 (macro)", _fmt(unet.get("f1_macro"), 4)),
        _kpi("U-Net IoU (macro)", _fmt(unet.get("iou_macro"), 4)),
        _kpi("CNN F1 (macro)", _fmt(cnn.get("f1_macro"), 4)),
        _kpi("Waterlogged area", _fmt(map_info.get("waterlogged_area_ha")), "ha"),
        _kpi("Waterlogged share", _pct(map_info.get("waterlogged_fraction"))),
    ])

    map_img = (
        f'<img class="map-img" src="assets/{assets["map"]}" '
        'alt="Classified waterlogging map">'
        if assets.get("map")
        else '<p class="empty">No class map stored for this run.</p>'
    )
    sev_img = (
        f'<img class="map-img" src="assets/{assets["severity"]}" '
        'alt="Waterlogging severity map">'
        if assets.get("severity")
        else '<p class="empty">No severity map stored for this run.</p>'
    )

    warn_block = ""
    if warnings:
        items = "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings)
        warn_block = f'<h2>Warnings</h2><div class="warn"><ul>{items}</ul></div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeepWaterLog - Agricultural Waterlogging Detection</title>
<style>
  :root {{
    --ink: #10263b; --muted: #5b7288; --line: #dfe7ee;
    --bg: #f4f7fa; --card: #ffffff;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--ink); line-height: 1.5;
    padding: 24px; max-width: 1280px; margin: 0 auto;
  }}
  header {{ margin-bottom: 18px; }}
  h1 {{ font-size: 24px; letter-spacing: -0.3px; }}
  .sub {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
  .theme-pill {{
    display: inline-block; margin-top: 8px; padding: 3px 10px;
    background: #e6f2ec; color: #1f6b4a; border-radius: 20px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.4px;
    text-transform: uppercase;
  }}
  .notice {{
    margin-top: 14px; background: #fff8e6; border: 1px solid #f1d9a0;
    color: #7a5a12; border-radius: 8px; padding: 10px 12px; font-size: 12px;
  }}
  .grid {{ display: grid; gap: 14px; }}
  .kpis {{ grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
  .card {{
    background: var(--card); border: 1px solid var(--line);
    border-radius: 10px; padding: 16px;
  }}
  .kpi-label {{
    font-size: 10px; font-weight: 700; letter-spacing: 0.7px;
    text-transform: uppercase; color: var(--muted);
  }}
  .kpi-value {{
    font-size: 26px; font-weight: 700; margin-top: 6px;
    font-variant-numeric: tabular-nums;
  }}
  .kpi-unit {{ font-size: 12px; font-weight: 600; color: var(--muted); }}
  h2 {{
    font-size: 13px; text-transform: uppercase; letter-spacing: 0.7px;
    color: var(--muted); margin: 26px 0 10px;
  }}
  .two-col {{ grid-template-columns: 1fr 1fr; }}
  @media (max-width: 860px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  .map-img {{
    width: 100%; display: block; border-radius: 6px;
    image-rendering: pixelated; background: #eef2f6;
  }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12px; }}
  .swatch {{ width: 12px; height: 12px; border-radius: 3px; flex: 0 0 auto; }}
  .swatch.inline {{ display: inline-block; margin-right: 7px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--line); }}
  th {{
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.6px;
    color: var(--muted);
  }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .bar-cell {{ width: 30%; }}
  .bar-track {{ background: #eef2f6; border-radius: 4px; height: 8px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; }}
  .empty {{ color: var(--muted); font-size: 14px; }}
  code {{
    background: #eef2f6; padding: 2px 6px; border-radius: 4px;
    font-size: 12px; font-family: ui-monospace, Menlo, Consolas, monospace;
  }}
  .warn ul {{ margin-left: 18px; }}
  .pill {{
    display: inline-block; padding: 1px 8px; border-radius: 10px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.4px;
  }}
  .pill-ok {{ background: #e6f2ec; color: #1f6b4a; }}
  .footer {{
    margin-top: 30px; color: var(--muted); font-size: 11px;
    text-align: center; line-height: 1.7;
  }}
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
  <div class="notice">
    <strong>Static snapshot.</strong> This page is a frozen export of the dashboard
    for GitHub Pages. The interactive version runs as a Flask service.
    Demonstration metrics come from a <strong>simulated five-class hyperspectral
    scene</strong>, not a drone or satellite survey.
  </div>
</header>

<div class="grid kpis">
{kpis}
</div>

<h2>Run provenance</h2>
<div class="card">
  <table>
    <tr><th style="width:180px">Run ID</th><td><code>{html.escape(str(record.get("run_id")))}</code></td></tr>
    <tr><th>Scenario</th><td>{html.escape(str(record.get("scenario_id")))}</td></tr>
    <tr><th>Status</th><td><span class="pill pill-ok">{html.escape(str(record.get("status", "unknown")).upper())}</span></td></tr>
    <tr><th>Device</th><td>{html.escape(str(provenance.get("device", "n/a")))}</td></tr>
    <tr><th>Processing time</th><td>{_fmt(record.get("processing_seconds"), 1)} s</td></tr>
    <tr><th>Bands</th><td>{band_sel.get("n_input_bands", "n/a")} raw &rarr; {band_sel.get("n_output_bands", "n/a")} after selection ({html.escape(str(band_sel.get("method", "n/a")))})</td></tr>
    <tr><th>Patches</th><td>{dataset.get("n_patches", "n/a")} (train {dataset.get("train", 0)} / val {dataset.get("val", 0)} / test {dataset.get("test", 0)})</td></tr>
  </table>
</div>

<h2>Waterlogging map</h2>
<div class="grid two-col">
  <div class="card">
    <div class="kpi-label">Classified extent</div>
    {map_img}
    {_legend()}
  </div>
  <div class="card">
    <div class="kpi-label">Waterlogging severity (P(waterlogged soil) + P(stressed crop))</div>
    {sev_img}
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
    <tbody>
{_class_rows(map_info)}
    </tbody>
  </table>
  <p class="sub" style="margin-top:8px">
    Total mapped area {_fmt(map_info.get("total_area_ha"))} ha at
    {config.MAP_RESOLUTION_M} m resolution. Waterlogged classes are
    <strong>waterlogged soil</strong> and <strong>stressed crop</strong>.
  </p>
</div>

<h2>Evaluation detail</h2>
<div class="card">
  <table>
    <thead><tr><th>Model</th><th class="num">Accuracy</th><th class="num">Precision</th>
      <th class="num">Recall</th><th class="num">F1</th><th class="num">IoU</th></tr></thead>
    <tbody>
      <tr><td>CNN (patch classification)</td>
        <td class="num">{_fmt(cnn.get("accuracy"), 4)}</td>
        <td class="num">{_fmt(cnn.get("precision_macro"), 4)}</td>
        <td class="num">{_fmt(cnn.get("recall_macro"), 4)}</td>
        <td class="num">{_fmt(cnn.get("f1_macro"), 4)}</td>
        <td class="num">&mdash;</td></tr>
      <tr><td>U-Net (pixel segmentation)</td>
        <td class="num">{_fmt(unet.get("accuracy"), 4)}</td>
        <td class="num">{_fmt(unet.get("precision_macro"), 4)}</td>
        <td class="num">{_fmt(unet.get("recall_macro"), 4)}</td>
        <td class="num">{_fmt(unet.get("f1_macro"), 4)}</td>
        <td class="num">{_fmt(unet.get("iou_macro"), 4)}</td></tr>
    </tbody>
  </table>
</div>

{warn_block}

<div class="footer">
  DeepWaterLog v{html.escape(str(provenance.get("version", config.PROJECT_VERSION)))} &middot;
  snapshot generated {html.escape(generated)} &middot;
  results are screening output, not a substitute for field inspection
</div>

</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a static dashboard snapshot.")
    parser.add_argument("--results-dir", default=str(config.RESULTS_DIR))
    parser.add_argument("--run-id", default=None, help="Run ID; defaults to the newest.")
    parser.add_argument("--out", default=str(BASE_DIR / "site"), help="Output directory.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out)

    if args.run_id:
        candidate = results_dir / f"{args.run_id}.json"
        if not candidate.exists():
            print(f"Run record not found: {candidate}")
            return 2
        record_path = candidate
    else:
        record_path = _latest_run(results_dir)

    if record_path is None:
        print(f"No run records found in {results_dir}. Run scripts/run_demo.py first.")
        return 2

    record = json.loads(record_path.read_text(encoding="utf-8"))
    run_id = record.get("run_id") or record_path.stem

    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    assets: dict[str, str] = {}
    artifacts = (record.get("map") or {}).get("artifacts") or {}
    for key, filename in (("map", "class_map_png"), ("severity", "severity_png")):
        source = artifacts.get(filename)
        if not source:
            continue
        source_path = Path(source)
        if not source_path.exists():
            # Artefact path may come from another machine; try the results dir.
            source_path = results_dir / source_path.name
        if source_path.exists():
            target = assets_dir / source_path.name
            shutil.copy2(source_path, target)
            assets[key] = target.name
        else:
            print(f"Warning: artefact missing, skipping: {source}")

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = build_html(record, assets, generated)
    (out_dir / "index.html").write_text(page, encoding="utf-8")

    # GitHub Pages runs Jekyll by default; disable it so files are served as-is.
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")

    print(f"Static snapshot written to {out_dir}")
    print(f"  index.html  ({len(page):,} bytes)")
    for key, name in assets.items():
        print(f"  assets/{name}  ({key})")
    print("")
    print("Preview locally:  python -m http.server 8000 --directory site")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
