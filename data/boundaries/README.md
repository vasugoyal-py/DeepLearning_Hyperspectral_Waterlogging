# Field boundaries and waterlogging reference vectors

This directory holds independent reference data for validating the DeepWaterLog
waterlogging maps.

## Structure

```
data/boundaries/
  waterlogging/        place field-validated waterlogging polygons here
    <scenario_id>_reference.geojson   e.g. punjab_paddy_2024_reference.geojson
  README.md            (this file)
  TEMPLATES_README.md  GeoJSON template format for waterlogging reference layers
```

## Adding a reference vector

For each scenario in `data/scenarios.json`, you may supply an independent,
field-validated waterlogging polygon (GeoJSON) at:

```
data/boundaries/waterlogging/<scenario_id>_reference.geojson
```

The GeoJSON should contain polygon features with a `class` property indicating
the waterlogging class (e.g., `waterlogged_soil`, `healthy_crop`,
`stressed_crop`, `open_water`, `non_agricultural`).

These references are used only for accuracy assessment (precision, recall, F1,
IoU) and are never used in training, keeping the evaluation independent.

## Template format

See `TEMPLATES_README.md` for the expected GeoJSON property schema.
