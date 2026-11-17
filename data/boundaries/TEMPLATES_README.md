# CEMS Vector Templates (`data/boundaries/`)

This directory holds **independent** reference flood extents used by Gate 5 extent validation.

## Current status

| File | Status |
|------|--------|
| `EMSR629_example.template.geojson` | Template only - not real data |
| `haridwar_2023_example.template.geojson` | Template only - not real data |
| `data/cems_registry.json` | Verified CEMS activation audit |

## CEMS audit results (verified against the Rapid Mapping API on 2026-11-05)

Only **EMSR629** ("Floods in Pakistan", 2022-08-29) is a verified flood activation.

The following codes were previously assigned but are **incorrect** - they resolve to wildfire/storm activations in other countries:
- EMSR672 → Wildfire in Attica, Greece (not a flood in India)
- EMSR580 → Forest Fire in Sierra Culebra, Spain (not a flood in India)
- EMSR305 → Forest Fire in Valencia, Spain (not a flood in India)
- EMSR395 → Tropical Cyclone in Flores Island, Portugal (not a flood in India)

## How to obtain real vectors

For EMSR629 (Pakistan 2022):
1. Query: `https://mapping.emergency.copernicus.eu/activations/api/activations/EMSR629/`
2. Download the vector product (GeoJSON or shapefile) from the Rapid Mapping portal
3. Convert to EPSG:4326 GeoJSON if needed
4. Clip to the scenario bbox: [67.90, 25.20, 68.70, 26.20]
5. Save as `data/boundaries/EMSR629.geojson` (replace the `.template.geojson` example)
6. Record the acquisition date in the file metadata

For events without a CEMS activation (Haridwar 2023, Silchar 2022, Kuttanad 2018, Patna 2019):
- Use a genuinely independent source (e.g., manually digitized Sentinel-2/NICFI extent from the same event)
- Save as `data/boundaries/<scenario_id>_reference.geojson`
- Document provenance: source, acquisition date, sensor, processing steps
- Do NOT derive from the SFIVE model output (circular validation)

## Validation behavior

- When a reference file is present: the validator computes a genuine confusion matrix
  (precision, recall, F1, IoU, omission/commission error)
- When no file is present: `available: false` is reported; no synthetic ground truth
  is invented

## How the pipeline looks up references

1. Primary: `data/boundaries/<cems_code>.geojson` (e.g., `EMSR629.geojson`)
2. Fallback: `data/boundaries/<cems_code>*.geojson` (e.g., `EMSR629_flood_extent.geojson`)
3. Scenario-based: `data/boundaries/<scenario_id>*.geojson` (e.g., `haridwar_2023_reference.geojson`)
