"""
Run-record persistence for DeepWaterLog.

Each pipeline run is written to ``data/results/<run_id>.json`` and a compact
audit line is appended to ``data/results/run_log.jsonl``.  Failures are logged
rather than hidden so that a run never silently disappears.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import config

LOGGER = logging.getLogger("deepwaterlog.reporting")


def save_run_result(result: dict[str, Any]) -> Path:
    """
    Persist a pipeline run record.

    Writes the full result JSON to ``data/results/<run_id>.json`` and appends
    a single-line summary to ``data/results/run_log.jsonl``.

    Returns the path of the JSON run record.
    """
    results_dir = config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    run_id = result.get("run_id") or "unknown_run"
    run_path = results_dir / f"{run_id}.json"

    try:
        with open(run_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, default=str)
    except TypeError as error:
        LOGGER.warning("Run record not fully JSON-serialisable: %s", error)
        with open(run_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, default=str)

    # Compact audit line
    log_line = {
        "run_id": run_id,
        "status": result.get("status"),
        "scenario_id": result.get("scenario_id"),
        "processing_seconds": result.get("processing_seconds"),
        "cnn_f1": (result.get("metrics") or {}).get("cnn", {}).get("f1_macro"),
        "unet_iou": (result.get("metrics") or {}).get("unet", {}).get("iou_macro"),
        "waterlogged_area_ha": (result.get("map") or {}).get("waterlogged_area_ha"),
        "warnings": result.get("warnings", []),
    }
    log_path = results_dir / "run_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(log_line, default=str) + "\n")

    LOGGER.info("Run record saved to %s", run_path)
    return run_path
