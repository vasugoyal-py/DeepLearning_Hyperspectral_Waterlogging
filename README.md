---
title: DeepWaterLog
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# DeepWaterLog — Deep Learning–Enabled Hyperspectral Remote Sensing for Agricultural Waterlogging Detection

> Detecting, classifying, and mapping agricultural waterlogging from hyperspectral imagery using CNNs and U-Net.
>
> **Theme:** Smart AgriTech — Precision Farming and Agro Sustainability

Agricultural waterlogging — excess soil moisture that stresses or kills crops — causes significant yield loss worldwide. DeepWaterLog applies deep learning to hyperspectral remote-sensing imagery to detect waterlogged soil, distinguish healthy from stressed crops, and produce georeferenced waterlogging maps for drainage planning and crop-damage assessment.

## What the pipeline does

1. **Preprocessing** — bad-band removal, destriping, noise filtering, atmospheric correction, and radiometric calibration of hyperspectral cubes to surface reflectance.
2. **Band selection & dimensionality reduction** — correlation/mutual-information filtering and PCA to retain waterlogging-sensitive spectral bands while reducing redundancy.
3. **Training data preparation** — labelled hyperspectral patches (waterlogged soil, healthy crop, stressed crop, open water, non-agricultural) split into train/validation/test sets.
4. **Deep learning modelling** —
   - **CNN:** patch-level classification of waterlogging presence and severity.
   - **U-Net:** pixel-level semantic segmentation producing waterlogging extent and severity maps.
5. **Evaluation** — accuracy, precision, recall, F1-score, and Intersection over Union (IoU).
6. **Decision support UI** — Flask + Leaflet web interface to visualize maps, overlay field boundaries, and export reports.

## Quick start

```bash
# Runtime dependencies (Python 3.12)
uv sync

# Offline test suite (no external data downloads)
uv run --with pytest pytest

# List available scenarios
uv run python test_pipeline.py --list

# Run one scenario end-to-end
uv run python test_pipeline.py --scenario punjab_paddy_2024 --json

# Fast demo run (~35 s): metrics, class map and severity map for the dashboard
uv run python scripts/run_demo.py --epochs 10 --size 96 --bands 40

# Train CNN and U-Net models on a labelled .npz dataset
uv run python scripts/train.py --data data/training/scene.npz --model both

# Evaluate trained models
uv run python scripts/evaluate.py --data data/testing/scene.npz --model both

# Launch decision-support UI
uv run python ui/app.py

# Generate a bounded directory tree and text-file snapshot
uv run python tree.py
```

`tree.py` writes `tree_str` (the directory structure) and `tree.txt`
(readable text-file contents). It skips Git/virtual-environment/cache folders,
binary or oversized files, generated outputs, and directories deeper than six
levels by default. Use `--max-depth N` to change the depth limit.

## Documentation

| Document | Contents |
|---|---|
| [`IDEATHON_2026_PROPOSAL.md`](IDEATHON_2026_PROPOSAL.md) | **IDEATHON 2026 submission document** in the required GD Goenka / Avishkar Centre format: introduction with citations, 31-reference literature review, research gap, objectives, methodology, hypothesis, pilot instrument with secondary evidence and a one-day collection kit, SDG contribution, limitations, APA 7th references |
| [`docs/README.md`](docs/README.md) | Documentation index, pipeline table and scientific contract |
| [`docs/WHAT_AND_WHY.md`](docs/WHAT_AND_WHY.md) | **Plain-language explainer** — what the project does, why it matters, how the six stages work, what is real today, and likely judge questions |
| [`docs/FIELDWORK_KIT.md`](docs/FIELDWORK_KIT.md) | One-day pilot survey kit: Google Form, consent, WhatsApp message, photo protocol, Slide 9 template |
| [`docs/PPT_CONTENT_FILLED.md`](docs/PPT_CONTENT_FILLED.md) | All 16 slides written out with cited numbers and a pre-submission checklist |
| [`deploy/HOSTING.md`](deploy/HOSTING.md) | Free hosting for the dashboard (Hugging Face Spaces or Render) |
| [`docs/RESEARCH_METHODOLOGY.md`](docs/RESEARCH_METHODOLOGY.md) | Literature review, research gap, objectives, methodology, hypothesis, pilot study, contribution, limitations, references (APA 7th) |
| [`docs/PAPER_DRAFT.md`](docs/PAPER_DRAFT.md) | Manuscript skeleton for a journal/conference paper |
| [`PROJECT PROPOSAL SYNERGY 2026.md`](PROJECT%20PROPOSAL%20SYNERGY%202026.md) | SYNERGY 2026 project proposal |
| [`PROJECT_EXPLANATION.txt`](PROJECT_EXPLANATION.txt) | Plain-language project explanation |

## Repository layout

```
config.py                  scientific parameters and configuration snapshot
core/
  preprocessing.py         Stage 1 — hyperspectral preprocessing
  band_selection.py        Stage 2 — band selection & dimensionality reduction
  dataset.py               Stage 3 — training data preparation
  cnn_model.py             Stage 4 — CNN classification model
  unet_model.py            Stage 4 — U-Net segmentation model
  trainer.py               training loop for CNN and U-Net
  evaluator.py             Stage 5 — metrics (accuracy, precision, recall, F1, IoU)
  mapper.py                Stage 5 — georeferenced waterlogging map generation
  pipeline.py              shared end-to-end orchestration
ui/app.py                  Flask results dashboard (metrics, maps, report export)
wsgi.py                    production entry point (waitress / gunicorn)
Dockerfile                 web image for free hosting (no PyTorch)
Dockerfile.research        full image with PyTorch for training
test_pipeline.py           CLI integration health check
scripts/train.py           training entry point
scripts/evaluate.py        evaluation entry point
scripts/run_demo.py        fast synthetic demo that populates the dashboard
scripts/pilot_analysis.py  computes pilot statistics and the Slide 9 chart
scripts/pilot_slide_chart.py redraws the Slide 9 chart from the reported figures
tests/                     offline unit and contract tests
docs/                      methodology, paper draft, explainer, fieldwork kit
data/                      hyperspectral data, labels, boundaries, pilot CSV
```

## See the results

```bash
python scripts/run_demo.py --epochs 10 --size 96 --bands 40
python ui/app.py
```

Then open **http://127.0.0.1:5000/**. The dashboard shows U-Net and CNN
metrics, the classified waterlogging map, the severity map, per-class areas in
hectares, run provenance, and a one-click Markdown report export.

### Put it online (free)

The dashboard needs only Flask, NumPy and Pillow, so it deploys in about two
minutes on Hugging Face Spaces or Render — no card required. Full steps:
[`deploy/HOSTING.md`](deploy/HOSTING.md).

```bash
python wsgi.py     # same production entry point the container uses
```

GitHub Pages cannot run Flask, so it hosts a frozen snapshot instead:

```bash
python scripts/export_static.py    # writes site/index.html + site/assets/
```

## Scientific status

- Preprocessing, band-selection, dataset, model, trainer, and evaluator components are covered by the offline test suite (16 tests).
- A synthetic demo run completes in about 35 s and produces a coherent classified map, a severity map and a Markdown report.
- Live hyperspectral ingestion requires data in `data/hyperspectral/` and labels in `data/labels/`.
- The decision-support UI is a Flask dashboard served from `ui/app.py`; it is a screening tool, not a replacement for agronomic field inspection.
