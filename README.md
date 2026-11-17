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

# Train CNN and U-Net models
uv run python scripts/train_cnn.py --data data/training
uv run python scripts/train_unet.py --data data/training

# Evaluate models
uv run python scripts/evaluate.py --model cnn --split test
uv run python scripts/evaluate.py --model unet --split test

# Launch decision-support UI
uv run python ui/app.py
```

## Documentation

| Document | Contents |
|---|---|
| [`IDEATHON_2026_PROPOSAL.md`](IDEATHON_2026_PROPOSAL.md) | **IDEATHON 2026 submission document** in the required GD Goenka / Avishkar Centre format: introduction with citations, 30-reference literature review, research gap, objectives, methodology, hypothesis, pilot study with questionnaire and results (n = 18), SDG contribution, limitations, APA 7th references |
| [`docs/README.md`](docs/README.md) | Documentation index, pipeline table and scientific contract |
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
ui/app.py                  Flask UI and map tiles
test_pipeline.py           CLI integration health check
scripts/                   training and evaluation scripts
tests/                     offline unit and contract tests
docs/                      research methodology and paper draft
data/                      hyperspectral data, labels, boundaries
```

## Scientific status

- Preprocessing, band-selection, dataset, model, trainer, and evaluator components are covered by the offline test suite.
- Live hyperspectral ingestion requires data in `data/hyperspectral/` and labels in `data/labels/`.
- The decision-support UI is a Flask + Leaflet application served from `ui/app.py`.
