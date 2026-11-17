# What We Are Doing, and Why

**Project:** DeepWaterLog — Deep Learning–Enabled Hyperspectral Remote Sensing for Agricultural Waterlogging Detection
**Theme:** Smart AgriTech — Precision Farming and Agro Sustainability
**Competition:** IDEATHON 2026, GD Goenka University / Avishkar Centre of Innovation and Entrepreneurship

This document explains the project in plain language: what problem we are
solving, why the problem is worth solving, what we are actually building, how it
works, what exists today, and what still has to be proven. It is written so that
a mentor, a judge, a teammate, or a family member can understand the project in
about ten minutes.

---

## 1. The one-paragraph version

Farmers lose a large share of their crop every year because their field gets
waterlogged — the soil stays saturated, crop roots cannot breathe, and the plant
dies slowly. The damage is usually only noticed once the leaves have already
turned yellow, which is too late. We are building a system that reads
**hyperspectral satellite or drone images** — images that record hundreds of
colours instead of just three — and uses **deep learning** to spot waterlogged
soil, healthy crops, stressed crops, and open water *before* a farmer can see the
damage. The output is a **map of the field** showing exactly which parts are
waterlogged and how badly, so that drainage can be fixed, irrigation can be
adjusted, and crop-insurance claims can be supported.

---

## 2. The problem, in plain terms

### What waterlogging actually is

When it rains heavily, or when a field is irrigated too much, or when the land is
flat and the drainage is poor, water fills the tiny spaces between soil
particles. Normally those spaces hold air, and roots use that air to breathe.
Once they fill with water, the roots suffocate. The plant stops taking up
nutrients, stops growing, and if the condition lasts long enough, it dies.

This is why waterlogging is often called a **hidden flood**. There may be no
visible standing water at all. The field simply looks wet. But underground, the
crop is already in trouble.

### Why it is expensive

| Effect | Consequence for the farmer |
|---|---|
| Root oxygen starvation | Stunted growth, yellowing, reduced grain filling |
| Nutrient lockout | Fertiliser is wasted — the plant cannot absorb it |
| Root rot and disease | Secondary fungal and bacterial infections |
| Delayed sowing or harvest | Missed crop calendar, lower yield |
| Repeated in the same spots | Low-lying areas waterlog every single season |

The last point matters most. Waterlogging is not random. It happens in the same
low-lying patches, near the same blocked drains, season after season. That
predictability is exactly what makes it a solvable remote-sensing problem.

### Why it is hard to catch in time

1. **It is invisible from the road.** No standing water, no dramatic damage — just
   a slightly duller, slightly yellower patch of crop.
2. **Farmers cannot walk every field every week.** A two-hectare plot might take
   an hour to inspect properly. A village has hundreds of plots.
3. **Visible symptoms arrive late.** By the time leaves yellow, yield has already
   been lost. You cannot un-lose it.
4. **Existing satellite indices are too blunt.** The standard tools — NDVI,
   NDWI — use only two or three broad colour bands. They see "this pixel is less
   green than last week," but they cannot tell you *why*. Drought, nutrient
   deficiency, disease, and waterlogging all look similar to them.

That fourth point is the technical heart of our project.

---

## 3. Why hyperspectral imaging

### The difference in one picture

A normal camera records three numbers per pixel: red, green, blue.
A multispectral satellite records maybe ten: a few visible bands, a couple of
near-infrared bands, a couple of shortwave-infrared bands.
A **hyperspectral** sensor records **hundreds** of narrow, continuous bands —
essentially a full spectrum for every single pixel of the image.

Think of it this way:

- A normal photo is like describing a colour as "greenish."
- A hyperspectral image is like having the actual paint formula — every
  ingredient and its exact proportion.

### Why that matters for waterlogging

Water leaves a very specific fingerprint in the spectrum:

| What we look for | Where in the spectrum | Why it signals waterlogging |
|---|---|---|
| Water absorption at ~950–970 nm | Near-infrared | Leaf water content rises in saturated crops |
| Water absorption at ~1200 nm | Near-infrared | Second water-content indicator |
| Reduced red-edge slope | ~700–750 nm | Chlorophyll breakdown from root stress |
| Lower chlorophyll absorption | ~680 nm | Yellowing, reduced photosynthesis |
| Higher shortwave-infrared reflectance from wet soil | ~1550–1750 nm | Surface and root-zone moisture |

A three-band camera cannot see any of this. A hyperspectral sensor sees all of
it, simultaneously, at every pixel. That is the whole reason we use it.

---

## 4. Why deep learning instead of a formula

You *could* try to hand-write a rule: "if band 45 is above 0.3 and band 87 is
below 0.2, call it waterlogged." This does not work well, for three reasons:

1. **The rules differ by crop, region, season, and sensor.** A rule tuned on
   Punjab paddy in July fails on Haryana wheat in February.
2. **The signature is not one number.** It is a *shape* across 200 bands, plus
   the spatial pattern of the neighbouring pixels (waterlogging clusters in
   low-lying areas).
3. **Real data is messy.** Clouds, shadows, sensor noise, mixed pixels at field
   edges. Hand-written rules break immediately.

Deep learning solves this by **learning the rules from labelled examples**. We
show the model thousands of image patches where experts have already marked
"this is waterlogged soil," "this is healthy crop," "this is stressed crop,"
"this is open water," and "this is a road." The model finds the patterns itself.

### The two models we use, and why both

| Model | Job | Output | Why we need it |
|---|---|---|---|
| **CNN** (Convolutional Neural Network) | Look at a 64×64 patch of the field and say "this patch is mainly waterlogged" | One label per patch | Fast triage — screen thousands of fields quickly |
| **U-Net** | Look at every pixel and assign it to a class | A full map — every pixel labelled | Precise extent — the farmer needs to know *which corner* of the field, not just that the field has a problem |

The CNN answers *"does this field have a problem?"* The U-Net answers *"where
exactly, and how much?"* The map is what actually gets acted on.

---

## 5. What the system produces

For a given field or scene, the system outputs:

1. **A classified map** — each pixel coloured as waterlogged soil, healthy crop,
   stressed crop, open water, or non-agricultural land.
2. **A severity map** — where the model is most confident waterlogging is
   present.
3. **Area statistics** — how many hectares are waterlogged, and what fraction of
   the total mapped area that represents.
4. **Accuracy metrics** — accuracy, precision, recall, F1-score, and
   Intersection over Union (IoU), so the numbers can be trusted or challenged.
5. **A downloadable report** — suitable for a drainage plan discussion or an
   insurance claim.

All of this is visible in the project's web dashboard.

---

## 6. How it works — the six stages

We built the system as a pipeline. Each stage feeds the next. Every stage
records what it did, so any result can be reproduced.

### Stage 1 — Preprocessing

Raw hyperspectral data is not directly usable. We clean it:

- **Remove bad bands** — some bands sit in water-vapour absorption regions
  (around 1400 nm and 1900 nm) where the signal is mostly atmospheric noise.
- **Destripe** — sensors produce thin vertical stripes; we suppress them.
- **Denoise** — smooth spectral noise while keeping the absorption features
  that actually matter.
- **Atmospheric correction** — convert "what the sensor saw" into "what the
  ground actually reflects."
- **Radiometric calibration** — convert raw digital numbers into physical
  reflectance values.

Output: a clean **surface-reflectance cube**.

### Stage 2 — Band selection and dimensionality reduction

We start with 200+ bands, but many are redundant or useless for waterlogging.
We thin them out using three complementary methods:

- **Correlation filtering** — if two bands are almost identical, drop one.
- **Mutual information** — keep the bands that are most informative about the
  class labels.
- **PCA (Principal Component Analysis)** — compress the whole cube into a small
  number of components that retain most of the variation.

This makes training faster and reduces overfitting, while keeping the
waterlogging-sensitive spectral features.

### Stage 3 — Training data preparation

We cut the cube into square patches (e.g., 64×64 pixels), assign each patch a
class label from field-validated data, normalise the values, apply augmentation
(flips, rotations, slight spectral jitter), and split everything 70% training /
15% validation / 15% test.

Each patch carries two labels: one **majority class** for the CNN, and one
**per-pixel label map** for the U-Net.

### Stage 4 — Deep learning

- The **CNN** learns to classify whole patches.
- The **U-Net** learns to segment pixels, using an encoder–decoder with skip
  connections so that fine spatial detail survives the compression.

Both are trained with early stopping and class balancing, and the best
checkpoint is saved.

### Stage 5 — Evaluation

We report, per class and averaged:

- Accuracy — how often the model is right overall.
- Precision — when it says "waterlogged," how often is it correct?
- Recall — of all the actually-waterlogged pixels, how many did it find?
- F1-score — the balance between precision and recall.
- IoU — overlap between predicted and true waterlogged regions.

Recall matters most for early warning: missing a waterlogged patch is worse than
raising a false alarm.

### Stage 6 — Mapping and decision support

The trained U-Net is run across the **entire scene** using a sliding window. We
average the predictions across overlapping windows to avoid seam artefacts, then
convert the result into a coloured map, compute areas in hectares, and display
everything in the web dashboard with a report export.

---

## 7. What is real today, and what is still planned

Being honest about status is part of doing research properly.

### Working now

- The complete six-stage pipeline runs end to end.
- Preprocessing, band selection, dataset preparation, CNN, U-Net, trainer,
  evaluator, mapper and reporting are all implemented and documented.
- The dashboard displays metrics, per-class breakdown, the classified map, the
  severity map, and exports a report.
- 16 automated tests pass, covering preprocessing, band selection, dataset
  splitting, metrics, map generation, and model forward passes.
- A full demo run on a synthetic scene completes in about 30 seconds and
  produces a coherent map (U-Net accuracy ≈ 0.92, F1 ≈ 0.79, IoU ≈ 0.72 on that
  synthetic data).

### Not yet done — and openly acknowledged

- **No real field data yet.** The demo runs on a synthetic scene built to mimic
  paddy, wheat and vegetable layouts. Real hyperspectral cubes with field labels
  are the next requirement.
- **No field validation yet.** The pilot instrument (10 unique research questions
  plus 5 demographic items) and a complete one-day fieldwork kit are ready; the
  survey itself has not been administered, so no pilot statistics are reported
  yet. Section 7.5 of the proposal says this explicitly.
- **Generalisation unproven.** A model trained on Punjab paddy has not yet been
  tested on Haryana wheat.
- **Georeferencing is optional.** Without the `rasterio` library the maps are
  saved as plain arrays rather than GeoTIFFs.

None of these are hidden. They are listed as limitations in both the proposal
and the methodology document.

---

## 8. Why this matters — who benefits

### The farmer

Gets an early warning with a map of which part of the field is at risk, weeks
before visible damage. Can open a drain, skip an irrigation cycle, or file an
insurance claim with evidence.

### The drainage planner

Gets a spatial picture of which zones waterlog repeatedly, so that limited
budget goes to the drains that will actually fix the problem, instead of being
spread evenly.

### The crop insurer

Gets an objective, dated, georeferenced record of waterlogging extent — instead
of relying on a field visit weeks after the event.

### The agricultural extension service

Gets a scalable screening tool. One analyst with satellite imagery can flag
hundreds of fields, and field officers only visit the ones that need attention.

### The research community

Gets a reproducible, open pipeline that others can extend to new crops, regions,
and sensors.

---

## 9. Fit with the theme and the SDGs

**Smart AgriTech — Precision Farming and Agro Sustainability**

- **Precision farming:** field-scale, crop-specific information instead of
  village-level guesswork. Variable-rate drainage and irrigation become
  possible.
- **Agro sustainability:** preventing yield loss avoids wasting the water,
  fertiliser, seed, and labour already invested in a crop. Protecting soil from
  prolonged saturation protects long-term soil health.

| SDG | How the project contributes |
|---|---|
| **SDG 2 — Zero Hunger** | Reduces crop loss from a major abiotic stressor |
| **SDG 6 — Clean Water and Sanitation** | Precision drainage and irrigation cut water waste and runoff |
| **SDG 12 — Responsible Consumption and Production** | Avoids wasted inputs on crops that would otherwise be lost |
| **SDG 13 — Climate Action** | Builds resilience to extreme-rainfall waterlogging |
| **SDG 15 — Life on Land** | Protects soil health and sustainable land use |

---

## 10. Questions a judge is likely to ask — and our answers

**"Isn't this just NDVI with extra steps?"**
No. NDVI uses two bands and cannot separate waterlogging from drought or
nutrient stress. We use hundreds of narrow bands specifically because the
water-absorption features at 950–970 nm and 1200 nm are what distinguish
waterlogging from other stressors. That is the entire technical justification
for choosing hyperspectral data.

**"Why not just use Sentinel-2? It's free and covers everything."**
Sentinel-2 has about 13 bands. It is excellent for crop type and general
vigour, but its spectral sampling is too coarse to isolate water-absorption
features reliably. We position hyperspectral as the accuracy benchmark, with
multispectral as the scalable fallback once the method is validated.

**"You don't have field data. Isn't this just a demo?"**
Correct, and we say so plainly. What we have built and demonstrated is the
complete working pipeline plus a rigorous, questionnaire-based fieldwork
protocol. The demo proves the software works end to end. The pilot study is the
next step, and its design is fully specified.

**"How do you know the model is right?"**
Five independent metrics — accuracy, precision, recall, F1, and IoU — plus
confusion matrices, evaluated on a held-out test set that was never used for
training. We also plan a baseline comparison against NDVI/NDWI so the advantage
is measured, not assumed.

**"What happens when it's cloudy?"**
Hyperspectral optical sensors cannot see through cloud. This is a stated
limitation. Part of the reason we report limitations up front is that judges
trust a project that knows its own boundaries.

**"Could a farmer actually use this?"**
The dashboard is browser-based, so it needs no installation, and it will be
hosted at a public URL for the demo. The intended delivery is a per-field map and
a short report, not a raw data product. Cost is the barrier our pilot instrument
is designed to measure, and it is the reason the output is built for a free
satellite-based service rather than a paid drone survey.

---

## 11. Glossary

| Term | Plain meaning |
|---|---|
| **Hyperspectral** | An image with hundreds of narrow colour bands instead of three |
| **Band** | One narrow wavelength slice of the spectrum |
| **Surface reflectance** | The fraction of sunlight a surface bounces back, after removing atmospheric effects |
| **CNN** | A neural network that learns visual patterns using small sliding filters |
| **U-Net** | A neural network shaped like a "U" that compresses an image and rebuilds it, labelling every pixel |
| **Semantic segmentation** | Assigning a class to every pixel in an image |
| **Patch** | A small square crop of the image used as one training example |
| **F1-score** | A single number balancing precision and recall |
| **IoU** | Overlap between predicted and true regions; 1.0 means perfect |
| **Waterlogging** | Soil saturation that suffocates crop roots, with or without visible standing water |
| **NDVI / NDWI** | Common two-band vegetation and water indices used as baselines |

---

## 12. Where to read more

| Document | What it contains |
|---|---|
| `IDEATHON_2026_PROPOSAL.md` | Full competition submission: literature review, gap, objectives, methodology, hypothesis, pilot instrument and plan, SDGs, references |
| `docs/RESEARCH_METHODOLOGY.md` | Detailed method, equations, sampling design and full APA reference list |
| `docs/PAPER_DRAFT.md` | Manuscript draft for journal submission |
| `docs/FIELDWORK_KIT.md` | One-day survey and photo collection kit with the Slide 9 template |
| `deploy/HOSTING.md` | How to put the dashboard online for free |
| `README.md` | How to install and run the code |
| `PROJECT_EXPLANATION.txt` | Short plain-language summary |

---

**Theme:** Smart AgriTech — Precision Farming and Agro Sustainability
**GD Goenka University · Avishkar Centre of Innovation and Entrepreneurship**
