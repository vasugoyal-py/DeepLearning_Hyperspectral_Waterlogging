# DeepWaterLog documentation index

DeepWaterLog — **Deep Learning–Enabled Hyperspectral Remote Sensing for
Agricultural Waterlogging Detection** — is a reproducible pipeline that
preprocesses hyperspectral cubes, selects waterlogging-sensitive bands, trains
CNN and U-Net models, evaluates them with accuracy, precision, recall, F1 and
IoU, and produces georeferenced waterlogging maps for decision support.

**Theme:** Smart AgriTech — Precision Farming and Agro Sustainability

## Start here

| Document | Purpose |
|---|---|
| [`../IDEATHON_2026_PROPOSAL.md`](../IDEATHON_2026_PROPOSAL.md) | IDEATHON 2026 submission format: introduction, literature review (30 APA references), research gap, objectives, methodology, hypothesis, pilot study with questionnaire, SDG contribution, limitations, references |
| [`RESEARCH_METHODOLOGY.md`](RESEARCH_METHODOLOGY.md) | Full methodology note: literature review, gap, objectives, six-stage method, hypothesis, pilot fieldwork design, contribution, limitations and APA 7th references. Source for a thesis or manuscript Methods section |
| [`PAPER_DRAFT.md`](PAPER_DRAFT.md) | Manuscript skeleton with abstract, introduction, literature review, methodology, pilot design, contribution, limitations and references. Experimental numbers are placeholders until models are trained on pilot data |
| [`../README.md`](../README.md) | Project overview, quick start and repository layout |
| [`../PROJECT_EXPLANATION.txt`](../PROJECT_EXPLANATION.txt) | Plain-language explanation for non-specialist readers |

## Pipeline at a glance

| Stage | Module | Output |
|---|---|---|
| 1. Preprocessing | `core/preprocessing.py` | Surface-reflectance cube (bad bands removed, destriped, denoised, atmospherically corrected) |
| 2. Band selection | `core/band_selection.py` | Reduced band set or PCA components retaining waterlogging-sensitive features |
| 3. Dataset preparation | `core/dataset.py` | Normalised patch tensors with majority labels (CNN) and per-pixel label patches (U-Net), split 70/15/15 |
| 4. Deep learning | `core/cnn_model.py`, `core/unet_model.py`, `core/trainer.py` | Trained CNN classifier and U-Net segmenter with early stopping and checkpoints |
| 5. Evaluation | `core/evaluator.py` | Accuracy, precision, recall, F1, IoU per class and macro-averaged, plus confusion matrices |
| 6. Mapping and decision support | `core/mapper.py`, `ui/app.py` | Georeferenced waterlogging extent/severity maps, Markdown reports, Flask + Leaflet dashboard |
| Orchestration | `core/pipeline.py` | One JSON-serialisable result with provenance and timings |
| Reporting | `core/reporting.py` | `data/results/<run_id>.json` plus a compact line in `run_log.jsonl` |

## Quick start

```bash
# Runtime dependencies (Python 3.12)
uv sync

# Offline test suite (no torch required for most tests)
uv run --with pytest pytest

# End-to-end health check on synthetic hyperspectral data
uv run python test_pipeline.py --bands 30 --size 64 --patch-size 16 --stride 8

# Train CNN and U-Net models on a labelled .npz dataset
uv run python scripts/train.py --data data/training/scene.npz --model both

# Evaluate trained models
uv run python scripts/evaluate.py --data data/testing/scene.npz --model both

# Launch the decision-support UI
uv run python ui/app.py
```

## Data layout

```
data/
  scenarios.json      four waterlogging scenarios (crop, region, window)
  hyperspectral/      raw or surface-reflectance hyperspectral cubes
  labels/             field-validated class maps (0-4)
  boundaries/         independent reference polygons for accuracy assessment
  models/             trained checkpoints (cnn_best.pt, unet_best.pt)
  results/            run records, run_log.jsonl, maps and reports
```

## Class scheme

| Index | Class | Colour |
|---|---|---|
| 0 | waterlogged_soil | red |
| 1 | healthy_crop | green |
| 2 | stressed_crop | orange |
| 3 | open_water | blue |
| 4 | non_agricultural | grey |

## Scientific contract

1. **No leakage.** Reference polygons in `data/boundaries/` are used only for
   accuracy assessment, never for training.
2. **Frozen parameters.** A configuration snapshot (`config.config_snapshot()`)
   is written into every run record.
3. **Recorded decisions.** Bad-band indices, band-selection method, split sizes,
   early-stopping epochs and all warnings are persisted with the result.
4. **Reproducibility.** Seeded splits, run IDs, UTC timestamps, device and
   timings are stored alongside the metrics.
5. **Honest placeholders.** Any unverified number in the manuscript is written as
   a placeholder, never as a finding.

## Current status and caveats

- The full six-stage pipeline runs end-to-end on synthetic hyperspectral data and
  is covered by an offline test suite.
- `scipy` and `rasterio` are optional: without `scipy` the spectral smoothing step
  passes the cube through unchanged and logs a warning; without `rasterio` maps
  are saved as `.npy` instead of GeoTIFF.
- Live data ingestion requires hyperspectral cubes in `data/hyperspectral/` and
  matching label maps in `data/labels/`.
- The decision-support UI is a screening tool, not a replacement for agronomic
  field inspection or engineering drainage design.
