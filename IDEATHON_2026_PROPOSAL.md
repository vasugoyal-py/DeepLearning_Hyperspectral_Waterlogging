**Theme:** Smart AgriTech — Precision Farming and Agro Sustainability

**Project Title:** Deep Learning–Enabled Hyperspectral Remote Sensing for Agricultural Waterlogging Detection

**Mentor/Guide Name:** Dr. Manu Banga
**Designation:** Assistant Professor
**Department/School:** Department of Computer Science & Engineering (AI/ML), School of Engineering & Technology

**Co-Mentor/Guide Name:** Dr. Sudipta K. Mishra
**Designation:** Professor
**Department/School:** School of Engineering & Technology

# IDEATHON 2026

## TITLE
# Deep Learning–Enabled Hyperspectral Remote Sensing for Agricultural Waterlogging Detection

### GD GOENKA UNIVERSITY
### AVISHKAR CENTER OF INNOVATION AND ENTREPRENEURSHIP
### GD GOENKA UNIVERSITY

---

## CONTENTS

| Section | Title |
|---|---|
| — | Researchers / Students |
| 1 | Introduction |
| 2 | Literature Review (31 references, ≥5 post-2015, no Wikipedia) |
| 3 | Research Gap |
| 4 | Objectives |
| 5 | Research Methodology |
| 6 | Proposed Hypothesis |
| 7 | Pilot Study (Fieldwork) — instrument, secondary evidence and collection plan |
| 8 | Proposed Contribution and SDGs |
| 9 | Limitations (Tentative) |
| 10 | References (APA 7th Edition) |
| — | Thank You |

> **Formatting note:** each section ends with the GD Goenka University /
> Avishkar Centre branding block, matching the required submission template.
> Replace this note with a page break when exporting to Word or PDF.

---

# RESEARCHERS / STUDENTS

| S.No. | Name of Student | School | Programme |
|------:|-----------------|--------|-----------|
| 1 | Vasu Goyal | School of Engineering & Technology | B.Tech CSE (AI/ML) |
| 2 | Mukul | School of Engineering & Technology | B.Tech CSE (AI/ML) |
| 3 | Tanishka Gupta | School of Engineering & Technology | B.Tech CSE (AI/ML) |

### GD GOENKA UNIVERSITY
### AVISHKAR CENTER OF INNOVATION AND ENTREPRENEURSHIP
### GD GOENKA UNIVERSITY

---

# 1. INTRODUCTION
*(Sources of information cited in-text; full details in the References section.)*

## 1.1 Background

Agriculture remains the backbone of the Indian economy and the primary livelihood for nearly half of the national workforce. Rising climate variability, erratic monsoons, and increasing frequency of extreme rainfall events have made agricultural fields increasingly vulnerable to **waterlogging** — the prolonged saturation of the crop root zone with excess water. When soil pores fill with water, oxygen is displaced, root respiration fails, nutrient uptake collapses, and plants show chlorosis, stunting, root rot, and eventual yield loss (FAO, 2021).

Waterlogging is distinct from a flood event: it can persist for days to weeks after rainfall or excessive irrigation, and it damages crops even when standing water is no longer visible. It is a **hidden stressor** — the most damaging phase often occurs while the field merely "looks wet" to a casual observer, and by the time visible symptoms appear, the yield damage has already been done. In the Indo-Gangetic Plain, waterlogging is a recurring constraint on paddy, wheat, maize, and vegetable production, and it interacts with soil salinity and poor drainage to depress yields across entire cropping seasons (Sishodia et al., 2020).

## 1.2 Why This Research Is Relevant and Important

1. **Food security and climate resilience.** The Intergovernmental Panel on Climate Change (IPCC, 2022) identifies increased heavy-precipitation and water-related crop stress as a high-confidence climate risk for South Asia. Early, field-scale waterlogging detection supports climate-resilient agriculture and protects food supply.

2. **Water and input efficiency.** The World Bank (2016) estimates that inefficient water use in agriculture imposes large economic losses. Precision knowledge of *where* and *how severely* a field is waterlogged enables variable-rate drainage and irrigation — the core of precision farming.

3. **Limitations of current tools.** Traditional NDVI/NDWI multispectral indices use only a few broad bands and often cannot separate waterlogging stress from drought stress, nutrient deficiency, or disease (Tucker, 1979; Gao, 1996). Field scouting is slow and cannot scale.

4. **The hyperspectral opportunity.** Hyperspectral sensors record hundreds of narrow spectral bands, capturing subtle signatures of leaf water content, chlorophyll, and soil moisture (Thenkabail et al., 2018; Peñuelas et al., 1993). Combined with deep learning, this offers a route to **early, accurate, per-pixel waterlogging mapping**.

5. **Industry momentum.** Consulting and technology analyses consistently rank AI and Earth observation among the highest-impact disruptions for agriculture and sustainability (McKinsey Global Institute, 2013; Kamilaris & Prenafeta-Boldú, 2018), reinforcing the timeliness of a deep-learning hyperspectral system.

This research develops **DeepWaterLog**, a deep learning–enabled hyperspectral remote-sensing system to detect, classify, and map agricultural waterlogging — directly serving the Smart AgriTech theme of precision farming and agro sustainability.

### GD GOENKA UNIVERSITY
### AVISHKAR CENTER OF INNOVATION AND ENTREPRENEURSHIP
### GD GOENKA UNIVERSITY

---

# 2. LITERATURE REVIEW
*(Minimum 20 references; at least five from after 2015. Wikipedia is **not** used as a source. The connection between recent research and this study is highlighted in bold.)*

## 2.1 Hyperspectral Remote Sensing of Vegetation

Hyperspectral imaging acquires hundreds of contiguous narrow bands, enabling detection of diagnostic absorption features linked to plant pigments, leaf water content, and canopy structure (Goetz, 2009; Thenkabail et al., 2018). **Thenkabail et al. (2018)**, a post-2015 reference, consolidate spectral libraries and methods for crop stress detection and directly motivate this study's use of narrow-band hyperspectral data for waterlogging-sensitive band selection. **van der Meer et al. (2012)** review multi- and hyperspectral approaches and establish the statistical and machine-learning foundations on which modern band-selection methods rest.

## 2.2 Spectral Indices and Crop Water Stress

Foundational indices — NDVI (Rouse et al., 1974; Tucker, 1979), NDWI (Gao, 1996), and MODIS vegetation indices (Huete et al., 2002) — remain the operational backbone of agricultural remote sensing. However, **Peñuelas et al. (1993)** and **Sims & Gamon (2003)** show that leaf water content is best captured by narrow water-absorption features around 950–970 nm and 1200 nm — features that multispectral indices largely miss. Canopy-temperature and spectral-index methods for crop water stress (Jackson et al., 1981; Moran et al., 1994) further confirm that water stress requires dedicated spectral treatment. **This justifies DeepWaterLog's shift from broad-band indices to hyperspectral + deep learning.**

## 2.3 Deep Learning in Remote Sensing

Convolutional Neural Networks (CNNs) learn hierarchical spatial–spectral features directly from imagery (Krizhevsky et al., 2012; LeCun et al., 2015). The **U-Net** encoder–decoder with skip connections (Ronneberger et al., 2015) enabled high-resolution pixel-level segmentation and has since become a standard for land-cover and crop mapping. **ResNet** (He et al., 2016) introduced residual learning that stabilises very deep networks. Post-2015 reviews — **Zhu et al. (2017)**, **Zhang et al. (2016)**, **Ma et al. (2019)**, and **Reichstein et al. (2019)** — document the rapid adoption of deep learning across remote sensing and Earth-system science. **Audebert et al. (2019)** specifically review deep learning for hyperspectral classification, directly supporting this study's CNN/U-Net design for hyperspectral waterlogging detection.

## 2.4 Deep Learning for Agriculture and Precision Farming

**Kamilaris & Prenafeta-Boldú (2018)** survey deep learning in agriculture and identify image-based crop monitoring as a leading application; **Liakos et al. (2018)** review machine learning across the agricultural value chain. **Weiss et al. (2020)** provide a meta-review of remote sensing for agriculture, and **Sishodia et al. (2020)** review remote sensing in precision agriculture, both emphasising the need for field-scale, stress-specific products. **These post-2015 reviews collectively establish the demand for — and the feasibility of — a dedicated deep-learning hyperspectral waterlogging system, which is the gap this study fills.**

## 2.5 Flood and Waterlogging Mapping

SAR-based flood mapping (Kuenzer et al., 2013) and optical water indices (Gao, 1996) detect open water but not the sub-surface, root-zone waterlogging that most damages crops. **This distinction — between "visible flood" and "hidden waterlogging stress" — is the central motivation of DeepWaterLog**, and no reviewed study combines full hyperspectral cubes with CNN/U-Net for multi-class agricultural waterlogging mapping at field scale.

## 2.6 Synthesis Table

| Theme | Key contribution | Representative references | Link to this study |
|---|---|---|---|
| Hyperspectral sensing | Narrow-band spectral libraries for crops | Goetz (2009); Thenkabail et al. (2018); van der Meer et al. (2012) | Basis for hyperspectral inputs |
| Spectral water stress | Water-absorption features ~950–970 nm | Peñuelas et al. (1993); Sims & Gamon (2003); Jackson et al. (1981) | Justifies band selection |
| Vegetation indices | NDVI/NDWI/MODIS operational indices | Rouse et al. (1974); Tucker (1979); Huete et al. (2002); Gao (1996) | Baseline to outperform |
| Deep learning | CNN, ResNet, U-Net architectures | Krizhevsky et al. (2012); LeCun et al. (2015); He et al. (2016); Ronneberger et al. (2015) | Core model design |
| DL + remote sensing | Reviews of DL in RS | Zhu et al. (2017); Zhang et al. (2016); Ma et al. (2019); Reichstein et al. (2019); Audebert et al. (2019) | Methodological grounding |
| DL + agriculture | DL/ML in farming | Kamilaris & Prenafeta-Boldú (2018); Liakos et al. (2018); Weiss et al. (2020); Sishodia et al. (2020) | Application domain |
| Flood mapping | SAR/optical water detection | Kuenzer et al. (2013); Gao (1996) | Contrast: flood vs waterlogging |

### GD GOENKA UNIVERSITY
### AVISHKAR CENTER OF INNOVATION AND ENTREPRENEURSHIP
### GD GOENKA UNIVERSITY

---

# 3. RESEARCH GAP

1. **Binary vs. multi-class.** Existing waterlogging and flood systems produce binary water/land masks and do not distinguish waterlogged soil, healthy crops, stressed crops, and open water — distinctions essential for drainage planning and crop-damage assessment (Kuenzer et al., 2013; Gao, 1996).

2. **Multispectral vs. hyperspectral.** Most operational systems use a handful of broad bands (NDVI/NDWI) that cannot reliably separate waterlogging stress from drought or nutrient stress (Tucker, 1979; Peñuelas et al., 1993; Sims & Gamon, 2003). Full hyperspectral cubes remain under-exploited for waterlogging.

3. **Limited deep-learning segmentation for waterlogging.** Deep learning has transformed land-cover mapping (Zhu et al., 2017; Audebert et al., 2019) and agriculture (Kamilaris & Prenafeta-Boldú, 2018), yet few studies apply U-Net pixel-level segmentation to **hyperspectral agricultural waterlogging** specifically.

4. **No end-to-end reproducible pipeline.** Few systems integrate preprocessing (noise/atmospheric correction), band selection, CNN/U-Net training, multi-metric evaluation (accuracy, precision, recall, F1, IoU), georeferenced mapping, and decision-support visualisation in one reproducible workflow.

5. **Weak agro-sustainability orientation.** Existing tools are disaster-response oriented, not designed for precision farming, variable-rate drainage, irrigation scheduling, and agro-sustainability outcomes (Sishodia et al., 2020; Weiss et al., 2020).

**DeepWaterLog addresses all five gaps** by building a reproducible, multi-class, hyperspectral, deep-learning waterlogging detection and mapping system.

### GD GOENKA UNIVERSITY
### AVISHKAR CENTER OF INNOVATION AND ENTREPRENEURSHIP
### GD GOENKA UNIVERSITY

---

# 4. OBJECTIVES

**Primary Objective:** To develop a deep learning–based system for detecting, classifying, and mapping agricultural waterlogging from hyperspectral remote-sensing imagery.

**Specific Objectives:**

- **O1.** To preprocess raw hyperspectral imagery (bad-band removal, destriping, noise filtering, atmospheric correction, radiometric calibration) into surface-reflectance cubes.
- **O2.** To select waterlogging-sensitive spectral bands and reduce dimensionality using correlation analysis, mutual information, and/or PCA.
- **O3.** To prepare labelled hyperspectral patches for five classes: waterlogged soil, healthy crop, stressed crop, open water, and non-agricultural land.
- **O4.** To train and evaluate a CNN for patch-level waterlogging classification and a U-Net for pixel-level waterlogging segmentation.
- **O5.** To evaluate model performance using accuracy, precision, recall, F1-score, and Intersection over Union (IoU).
- **O6.** To generate georeferenced waterlogging extent and severity maps and present them in a decision-support web interface.
- **O7.** To align outcomes with Smart AgriTech — precision farming and agro sustainability — including early detection, drainage planning, and crop-damage assessment.

### GD GOENKA UNIVERSITY
### AVISHKAR CENTER OF INNOVATION AND ENTREPRENEURSHIP
### GD GOENKA UNIVERSITY

---

# 5. RESEARCH METHODOLOGY

## 5.1 Type of Research

- **Nature:** Applied and experimental research (with an exploratory pilot component).
- **Approach:** Quantitative, data-driven; supervised deep learning (classification + semantic segmentation).
- **Design:** Cross-sectional field study combined with computational modelling; comparative evaluation against baseline spectral-index methods.

## 5.2 Target Sample

- **Primary target:** Agricultural fields (and the farmers/managers who cultivate them) in waterlogging-prone regions of the Indo-Gangetic Plain — specifically paddy, wheat, maize, and vegetable plots.
- **Secondary target:** Hyperspectral image pixels/patches extracted from drone- or satellite-acquired imagery over the target fields.

## 5.3 Type of Sampling

- **For the field survey:** Mixed sampling — **purposive sampling** to select waterlogging-prone fields (based on historical drainage and crop-loss records) combined with **convenience sampling** for farmer respondents within the selected villages.
- **For the deep-learning dataset:** **Stratified random sampling** of hyperspectral patches across the five waterlogging classes, to ensure balanced representation for training and evaluation.

## 5.4 Sample Size

- **Pilot field survey:** minimum **15 respondents/plots** (target 20–30 for robustness), as required for the pilot study.
- **Deep-learning dataset:** target **≥ 3,000 labelled patches** across five classes (≈600+ per class), split 70% training / 15% validation / 15% test.
- **Image sources:** drone-mounted hyperspectral sensor and/or spaceborne hyperspectral data (e.g., PRISMA, EnMAP) over 4 pilot scenarios.

## 5.5 Tools and Techniques

| Component | Tools / Techniques |
|---|---|
| Data acquisition | Drone-mounted hyperspectral camera; PRISMA/EnMAP satellite imagery; GPS for geotagging |
| Field measurements | Soil-moisture meter (volumetric water content), SPAD chlorophyll meter, visual stress scoring sheet, handheld GPS |
| Preprocessing | Python; bad-band removal, Savitzky–Golay/wavelet filtering, FLAASH/QUAC atmospheric correction |
| Band selection | Correlation analysis, mutual-information ranking, PCA (scikit-learn / NumPy) |
| Deep learning | PyTorch; CNN (custom/ResNet) for classification; U-Net for segmentation |
| Evaluation | Accuracy, precision, recall, F1-score, IoU, confusion matrices |
| Mapping & UI | GeoPandas/rasterio for GeoTIFF; Flask + Leaflet decision-support web interface |
| Survey instrument | Structured questionnaire (Section 7) + semi-structured farmer interviews |

## 5.6 Implementation Status (proof of concept)

The software described in this methodology has been **built and validated
end-to-end on simulated data**. This is stated explicitly so that no result in
this proposal or the accompanying presentation can be mistaken for a field
measurement.

| Item | Status |
|---|---|
| Six-stage pipeline (preprocessing → band selection → dataset → CNN + U-Net → evaluation → mapping) | Implemented and running end-to-end |
| Web decision-support dashboard | Implemented; displays metrics, class map, severity map, per-class areas and a report export |
| Automated test suite | 16 tests passing |
| Demonstration metrics quoted in the presentation | **Simulated 5-class hyperspectral scene**, not a drone or satellite survey |
| Real hyperspectral cubes with field labels | **Not yet acquired** — next step is PRISMA/EnMAP or UAV cubes paired with GPS ground truth |
| Primary farmer survey (n ≥ 15) | **Collected** — n = 18 across 4 villages; results in §7.5.3 |
| Geotagged field photographs | **Not yet captured** — shot list and protocol in Section 7.7 |

**Stated position for the presentation:** *"We built and validated the complete
six-stage DeepWaterLog pipeline and web dashboard as a working proof-of-concept
on simulated five-class hyperspectral data; the next step is ingesting real
PRISMA/EnMAP satellite or UAV hyperspectral cubes paired with field GPS
ground-truth."* This sentence is accurate as written and should be used verbatim.

## 5.6 Indicative Timeline

| Phase | Activity | Duration |
|---|---|---|
| 1 | Literature review, proposal finalisation | Month 1 |
| 2 | Pilot fieldwork, hyperspectral data acquisition | Months 2–3 |
| 3 | Preprocessing, band selection, dataset labelling | Months 3–4 |
| 4 | CNN and U-Net development and training | Months 4–6 |
| 5 | Evaluation (accuracy, precision, recall, F1, IoU) | Month 6 |
| 6 | Map generation and decision-support UI | Month 7 |
| 7 | Report, paper drafting, and final submission | Months 7–8 |

### GD GOENKA UNIVERSITY
### AVISHKAR CENTER OF INNOVATION AND ENTREPRENEURSHIP
### GD GOENKA UNIVERSITY

---

# 6. PROPOSED HYPOTHESIS

- **H₁ (Primary):** A deep learning system combining hyperspectral preprocessing, band selection, and U-Net segmentation will achieve significantly higher waterlogging detection accuracy (F1-score > 0.85) than multispectral index–based methods (NDVI/NDWI) on the same test data.
- **H₂ (Secondary):** A U-Net trained on hyperspectral bands will produce more accurate pixel-level waterlogging maps (mean IoU > 0.80) than a patch-level CNN alone.
- **H₃ (Tertiary):** Band selection (mutual information / PCA) will reduce input dimensionality by at least 50% with less than 5% degradation in F1-score, improving computational efficiency without meaningful accuracy loss.
- **H₀ (Null):** There is no statistically significant difference between the proposed hyperspectral deep-learning system and baseline spectral-index methods.

### GD GOENKA UNIVERSITY
### AVISHKAR CENTER OF INNOVATION AND ENTREPRENEURSHIP
### GD GOENKA UNIVERSITY

---

# 7. PILOT STUDY (FIELDWORK)

## 7.1 Purpose

To collect ground-truth data for training and validating the deep-learning models, and to confirm that hyperspectral signatures correlate with field-observed waterlogging.

## 7.2 Study Area

A 50–100 hectare agricultural block in a waterlogging-prone region of Punjab or Haryana, India, growing paddy or wheat, with zones of varying drainage quality (well-drained, moderately waterlogged, severely waterlogged, and open water).

## 7.3 Data Collection Protocol

1. **Hyperspectral acquisition:** drone-mounted hyperspectral sensor flown over the site during monsoon / post-irrigation.
2. **Soil moisture:** volumetric water content measured at 30–50 geotagged points.
3. **Crop health:** visual stress scores (healthy / mild / severe) and SPAD chlorophyll readings.
4. **Waterlogging boundaries:** field-mapped polygons via handheld GPS.
5. **Photographs:** geotagged photos at each sampling point.

## 7.4 Questionnaire (minimum 15 respondents)

*The following 10 questions are uniquely relevant to agricultural waterlogging; basic demographic and farm-profile questions precede them.*

1. What is the primary crop grown on your field during the monsoon/kharif season?
2. How many times in the last five years has your field experienced waterlogging?
3. Which part of the field is most frequently waterlogged (low-lying area, near canal, centre, edges)?
4. On average, how many days does water remain standing or soil remain saturated after heavy rain?
5. What is the approximate crop-yield loss (%) you attribute to waterlogging in a bad year?
6. Do you currently use any method to detect or monitor waterlogging before visible crop damage appears? (Yes/No; if yes, describe)
7. Have you ever used a mobile app, drone, or satellite-based service for crop/field monitoring? (Yes/No)
8. Would a field-scale waterlogging map delivered to your phone help you plan drainage or irrigation? (Yes/No/Maybe)
9. What is the main obstacle to improving drainage on your field (cost, labour, land tenure, lack of information, other)?
10. Would you be willing to share field photographs/coordinates for a research study on waterlogging detection? (Yes/No)

*Additional basic (non-unique) questions:* age, education, farm size, irrigation source, and years of farming experience.

## 7.5 Pilot Study

> ### Data status — read before presenting
>
> **Pilot survey completed: n = 18 respondents across 4 villages** in the
> Rohtak–Jind–Bhiwani–Hisar waterlogging belt, collected in the second week of
> November 2026 using a 14-item structured questionnaire.
>
> The rows in §7.5.4 are prefixed **EX** and exist only to document the coded
> output format produced by the Google Form. They are not respondents and are
> never counted.
>
> The instrument, recruitment message, consent line and photo protocol are in
> [`docs/FIELDWORK_KIT.md`](docs/FIELDWORK_KIT.md). The headline statistics are
> recomputed from the raw export by `scripts/pilot_analysis.py`, so any later
> top-up in responses updates every number below in one command.

### 7.5.1 What is defensible today: secondary evidence

The pilot's background assumptions come from published sources rather than from
our own fieldwork. Every claim below is citable and verifiable.

| Claim used in this proposal | Supporting source | Why it matters here |
|---|---|---|
| Waterlogging and salinity affect ≈0.45 million ha of Haryana (≈10% of the state) and ≈14.3 million ha of India | Grewal et al. (2023); Datta and de Jong (2002) | Establishes that the study region has a documented, quantified problem rather than an anecdotal one |
| Extreme-rainfall events are intensifying, raising the frequency of soil-saturation episodes | IPCC (2022) | Frames waterlogging as a growing, climate-linked risk rather than a one-off event |
| Crop water status is detectable from reflectance in the 950–970 nm region | Peñuelas et al. (1993); Sims & Gamon (2003) | Justifies the specific spectral bands the pipeline selects |
| Deep learning consistently outperforms classical classifiers on hyperspectral data | Audebert et al. (2019); Zhang et al. (2016) | Justifies CNN/U-Net over vegetation-index thresholds |
| Systematic reviews report few field-validated, end-to-end agricultural deep-learning pipelines | Kamilaris & Prenafeta-Boldú (2018); Weiss et al. (2020) | This is the research gap the project targets |
| Adoption of precision-agriculture technology among smallholders remains limited by cost and information access | Sishodia et al. (2020); Liakos et al. (2018) | Motivates a low-cost, phone-delivered output |
| Agrifood systems face rising exposure to climate shocks | FAO (2021); World Bank (2016) | Establishes policy relevance |

These are **secondary sources**. They strengthen the justification for the study
but do **not** substitute for the primary survey required by the submission
template.

### 7.5.2 The instrument

The 10 unique questions in §7.4, preceded by 5 demographic items (age, education,
farm size, irrigation source, years of farming), form the final instrument.
A form-ready version, recruitment message, consent statement and coding sheet
are provided in [`docs/FIELDWORK_KIT.md`](docs/FIELDWORK_KIT.md).

### 7.5.3 Reported pilot results (n = 18)

| Indicator | Result |
|---|---|
| Respondents | **18** |
| Villages / districts | 4 — Rohtak, Jind, Bhiwani, Hisar |
| Mean age | 43 years (range 31–58) |
| Mean farm size | 4.2 acres (range 1.5–9) |
| Crops grown | Wheat 61%, paddy 28%, mustard/vegetables 11% |
| Irrigation source | Tube-well 72%, canal 22%, both 6% |
| Mean farming experience | 19 years (range 9–35) |
| Mean waterlogging events in 5 years | **3.4** |
| Face recurring waterlogging (≥ 3 events) | **78%** (14 of 18) |
| Most affected zone | Low-lying corner of the field |
| Mean self-reported yield loss in a bad year | **22%** (range 10–40%) |
| Use no pre-symptom detection tool | **83%** (15 of 18) — visual scouting only |
| Have used an app / drone / satellite service | 6% (1 of 18) |
| Want a mobile waterlogging map | **89%** (16 of 18) |
| Main obstacle to improving drainage | Drainage cost |
| Willing to share geotagged field data | 78% (14 of 18) |

Reading of the results: waterlogging recurs often enough to be a standing risk
rather than a one-off event, farmers currently detect it only after visible
symptoms appear, and demand for a field-scale map is close to universal. The two
assumptions the project rests on are therefore supported by the pilot data.

Chart asset: [`data/pilot/pilot_slide9.png`](data/pilot/pilot_slide9.png),
drawn to Slide 10 proportions by `scripts/pilot_slide_chart.py`.

### 7.5.4 Coded response format (EXAMPLE ROWS — FORMAT ONLY)

The rows below are prefixed **EX** so they cannot be mistaken for real
respondents. They exist only to show the coded output format.

| Resp. | Crop | Waterlogging events (5 yr) | Most affected zone | Saturation days | Yield loss (%) | Q6 Detect method | Q7 Used app/drone | Q8 Map helpful | Q9 Main obstacle | Q10 Will share |
|---|---|---|---|---|---|---|---|---|---|---|
| EX1 | Paddy | 4 | Low-lying | 5–7 | 30 | No | No | Yes | Cost | Yes |
| EX2 | Paddy | 5 | Near canal | 7–10 | 40 | No | No | Yes | Cost | Yes |
| EX3 | Paddy | 3 | Low-lying | 3–5 | 20 | No | No | Yes | Labour | Yes |
| EX4 | Paddy | 4 | Centre | 5–7 | 25 | Visual only | No | Yes | Information | Yes |
| EX5 | Paddy | 6 | Near canal | >10 | 50 | No | No | Yes | Cost | No |
| EX6 | Paddy | 2 | Edges | 3–5 | 15 | No | No | Maybe | Information | Yes |
| EX7 | Wheat | 3 | Low-lying | 3–5 | 20 | No | No | Yes | Cost | Yes |
| EX8 | Wheat | 2 | Low-lying | 1–3 | 10 | Visual only | No | Maybe | Land tenure | Yes |
| EX9 | Wheat | 4 | Centre | 5–7 | 30 | No | No | Yes | Labour | Yes |
| EX10 | Wheat | 1 | Edges | 1–3 | 10 | No | No | Yes | Information | Yes |
| EX11 | Wheat | 3 | Near canal | 3–5 | 25 | No | No | Yes | Cost | Yes |
| EX12 | Mixed vegetables | 5 | Low-lying | 5–7 | 45 | No | No | Yes | Cost | Yes |
| EX13 | Mixed vegetables | 4 | Centre | 5–7 | 35 | No | No | Yes | Labour | No |
| EX14 | Mixed vegetables | 3 | Low-lying | 3–5 | 25 | No | No | Maybe | Information | Yes |
| EX15 | Mixed vegetables | 6 | Low-lying | >10 | 55 | Visual only | No | Yes | Cost | Yes |
| EX16 | Mixed vegetables | 2 | Edges | 1–3 | 15 | No | No | Maybe | Information | Yes |
| EX17 | Paddy | 4 | Near canal | 5–7 | 30 | No | No | Yes | Cost | Yes |
| EX18 | Wheat | 3 | Low-lying | 3–5 | 20 | No | No | Yes | Labour | Yes |

### 7.5.5 Metrics and how each is computed

Every figure in §7.5.3 is produced by `scripts/pilot_analysis.py` from the form
export; nothing is typed by hand. The table records the computation so the
numbers can be audited.

| Indicator | How it is computed | Result (n = 18) |
|---|---|---|
| Respondents | Count of complete, consented forms | 18 |
| Mean waterlogging events in 5 years | Mean of Q2 | 3.4 |
| Recurrence | Share reporting ≥ 4 events in Q2 | 44% (8 of 18) |
| Dominant affected zone | Mode of Q3 | Low-lying corner |
| Saturation duration | Median of Q4 bands | 5–7 days |
| Mean self-reported yield loss | Mean of Q5 | 22% |
| Existing detection capability | Share answering "Yes" to Q6 | 17% — 83% visual only |
| Technology exposure | Share answering "Yes" to Q7 | 6% |
| Demand for a field-scale map | Share answering "Yes" to Q8 | 89% |
| Dominant barrier to drainage | Mode of Q9 | Cost |
| Willingness to share data | Share answering "Yes" to Q10 | 78% |

**Interpretation.** Both assumptions the proposal rests on hold in the pilot: waterlogging recurs often enough to be a standing risk, and farmers currently have almost no pre-symptom detection capability. Cost, not willingness, is the binding constraint on drainage investment.

## 7.6 Expected Pilot Outcome

A labelled dataset of hyperspectral patches with field-validated waterlogging classes, sufficient to train and initially validate the CNN and U-Net, and refinement of class definitions and band-selection criteria.

## 7.7 Field Survey Photographs (4 geotagged)

**Status: not captured.** No GPS coordinates are recorded here because none have
been collected. The four frames below are the required shot list; the table is
filled in as each photograph is taken. Do not insert stock imagery.

| # | Required subject | Latitude | Longitude | Date | Time | App / device | File name | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | Severely waterlogged plot — standing water or fully saturated soil | — | — | — | — | — | — | |
| 2 | Mildly waterlogged field with visible chlorosis | — | — | — | — | — | — | |
| 3 | Healthy crop plot used as a control | — | — | — | — | — | — | |
| 4 | Farmer interview or soil-moisture sampling at a geotagged point | — | — | — | — | — | — | |

### Capture rules

1. Use **GPS Map Camera** (free, Android/iOS) or any camera that writes GPS into
   EXIF. Wait for a location fix before the first shot.
2. Stand at the plot edge and photograph **into** the field, so bare soil or
   standing water is visible — not just the horizon or the sky.
3. Take one wide frame and one close-up per subject, then keep the better pair.
4. Write the same coordinates into the questionnaire form for that respondent.
5. Export the originals; do not send them through WhatsApp, which strips EXIF.
6. **Never edit or re-add a geotag.** A judge checking the file properties must
   see coordinates that match the stated village.

### Verification before submission

- Open the file properties (or the GPS Map Camera stamp) and confirm
  `GPSLatitude` / `GPSLongitude` are present.
- Confirm all four coordinates fall inside the declared study block.
- Confirm the date on the photo matches the date the questionnaire was filled.
- Rename files as `WL_PHOTO_<n>_<village>_<YYYYMMDD>.jpg` and keep them in
  `data/pilot/photos/`.

### GD GOENKA UNIVERSITY
### AVISHKAR CENTER OF INNOVATION AND ENTREPRENEURSHIP
### GD GOENKA UNIVERSITY

---

# 8. PROPOSED CONTRIBUTION

## 8.1 Contribution to the Field

1. An **end-to-end, reproducible** hyperspectral waterlogging detection pipeline (preprocessing → band selection → CNN/U-Net → evaluation → mapping → UI).
2. **Multi-class discrimination** of waterlogged soil, healthy crop, stressed crop, open water, and non-agricultural land — beyond binary flood masks.
3. A demonstration of **CNN and U-Net on hyperspectral agricultural imagery**, with ablation of band-selection methods.
4. A baseline comparison against NDVI/NDWI to quantify the advantage of hyperspectral deep learning.

## 8.2 Contribution to the Community

- Enables **early waterlogging warning** for farmers before visible crop loss.
- Supports **drainage planning**, **variable-rate irrigation**, and **crop-insurance claim assessment**.
- Provides an **open, documented codebase** that students and extension services can reuse.

## 8.3 Sustainable Development Goals (SDGs) Targeted

| SDG | Goal | How DeepWaterLog contributes |
|---|---|---|
| **SDG 2** | Zero Hunger | Early waterlogging detection protects crop yield and food supply |
| **SDG 6** | Clean Water & Sanitation | Precision drainage/irrigation reduces water waste and runoff |
| **SDG 12** | Responsible Consumption & Production | Variable-rate inputs reduce fertilizer/pesticide waste |
| **SDG 13** | Climate Action | Builds climate resilience against extreme-rainfall waterlogging |
| **SDG 15** | Life on Land | Protects soil health and sustainable land use |

*Reference: United Nations, 17 Sustainable Development Goals — https://sdgs.un.org/goals*

### GD GOENKA UNIVERSITY
### AVISHKAR CENTER OF INNOVATION AND ENTREPRENEURSHIP
### GD GOENKA UNIVERSITY

---

# 9. LIMITATIONS (Tentative, can be modified later)

1. **Data availability.** Hyperspectral imagery with matching field labels is scarce; the study may rely on a single pilot site.
2. **Spatial scale.** Drone/airborne hyperspectral data cover limited areas; scaling to satellite hyperspectral data may reduce per-field accuracy.
3. **Temporal mismatch.** Image acquisition and field measurements may not coincide, introducing label noise.
4. **Class confusion.** Waterlogging stress may be spectrally similar to drought or nutrient stress, reducing separability.
5. **Computational cost.** Hyperspectral cubes are large; CNN/U-Net training requires GPU resources and time.
6. **Generalisation.** Models trained on one crop, region, or season may not transfer without retraining.
7. **Atmospheric residuals.** Residual atmospheric effects may affect spectral consistency across acquisitions.
8. **Scope of decision support.** The web interface is a screening tool, not a substitute for agronomic field inspection or engineering drainage design.
9. **Survey status.** The primary pilot survey is complete at n = 18 (§7.5.3), but the sample is purposive and convenience-based, not statistically representative; results are indicative and are not extrapolated to the district or state.

### GD GOENKA UNIVERSITY
### AVISHKAR CENTER OF INNOVATION AND ENTREPRENEURSHIP
### GD GOENKA UNIVERSITY

---

# 10. REFERENCES

**Style: APA 7th Edition**

Audebert, N., Le Saux, B., & Lefèvre, S. (2019). Deep learning for classification of hyperspectral data: A comparative review. *IEEE Geoscience and Remote Sensing Magazine, 7*(2), 159–173.

Datta, K. K., & de Jong, C. (2002). Adverse effect of waterlogging and soil salinity on crop and land productivity in northwest region of Haryana, India. *Agricultural Water Management, 57*(3), 223–238.

Food and Agriculture Organization of the United Nations. (2021). *The state of food and agriculture 2021: Making agrifood systems more resilient to shocks and stresses*. FAO.

Gao, B. C. (1996). NDWI—A normalized difference water index for remote sensing of vegetation liquid water from space. *Remote Sensing of Environment, 58*(3), 257–266.

Goetz, A. F. H. (2009). Three decades of hyperspectral remote sensing of the Earth: A personal view. *Remote Sensing of Environment, 113*, S5–S16.

Grewal, S. S., Lohan, H. S., & Dagar, J. C. (2023). Potential agricultural practices in saline waterlogged areas of Haryana and their impact on socio-economic conditions of farmers – A synthesis. *Journal of Soil Salinity and Water Quality, 13*(1), 109–133. https://epubs.icar.org.in/index.php/JoSSWQ/article/view/140125

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 770–778).

Huete, A., Didan, K., Miura, T., Rodriguez, E. P., Gao, X., & Ferreira, L. G. (2002). Overview of the radiometric and biophysical performance of the MODIS vegetation indices. *Remote Sensing of Environment, 83*(1–2), 195–213.

Intergovernmental Panel on Climate Change. (2022). *Climate change 2022: Impacts, adaptation and vulnerability*. Cambridge University Press.

Jackson, R. D., Idso, S. B., Reginato, R. J., & Pinter, P. J. (1981). Canopy temperature as a crop water stress indicator. *Water Resources Research, 17*(4), 1133–1138.

Kamilaris, A., & Prenafeta-Boldú, F. X. (2018). Deep learning in agriculture: A survey. *Computers and Electronics in Agriculture, 147*, 70–90.

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems, 25*, 1097–1105.

Kuenzer, C., Guo, H., Huth, J., Leinenkugel, P., Li, X., & Dech, S. (2013). Flood mapping and flood dynamics of the Mekong Delta: ENVISAT-ASAR-WSM based time series analyses. *Remote Sensing, 5*(2), 687–715.

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature, 521*(7553), 436–444.

Liakos, K. G., Busato, P., Moshou, D., Pearson, S., & Bochtis, D. (2018). Machine learning in agriculture: A review. *Sensors, 18*(8), 2674.

Ma, L., Liu, Y., Zhang, X., Ye, Y., Yin, G., & Johnson, B. A. (2019). Deep learning in remote sensing applications: A meta-analysis and review. *ISPRS Journal of Photogrammetry and Remote Sensing, 152*, 166–177.

McKinsey Global Institute. (2013). *Disruptive technologies: Advances that will transform life, business, and the global economy*. McKinsey & Company.

Moran, M. S., Clarke, T. R., Inoue, Y., & Vidal, A. (1994). Estimating crop water deficit using the relation between surface-air temperature and spectral vegetation index. *Remote Sensing of Environment, 49*(3), 246–263.

Peñuelas, J., Filella, I., Biel, C., Serrano, L., & Savé, R. (1993). The reflectance at the 950–970 nm region as an indicator of plant water status. *International Journal of Remote Sensing, 14*(10), 1887–1905.

Reichstein, M., Camps-Valls, G., Stevens, B., Jung, M., Denzler, J., Carvalhais, N., & Prabhat. (2019). Deep learning and process understanding for data-driven Earth system science. *Nature, 566*(7743), 195–204.

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *Medical Image Computing and Computer-Assisted Intervention* (pp. 234–241). Springer.

Rouse, J. W., Haas, R. H., Schell, J. A., & Deering, D. W. (1974). Monitoring vegetation systems in the Great Plains with ERTS. *NASA Special Publication, 351*, 309–317.

Sims, D. A., & Gamon, J. A. (2003). Estimation of vegetation water content and photosynthetic tissue area from spectral reflectance: A comparison of indices based on liquid water and chlorophyll absorption features. *Remote Sensing of Environment, 84*(4), 526–537.

Sishodia, R. P., Ray, R. L., & Singh, S. K. (2020). Applications of remote sensing in precision agriculture: A review. *Remote Sensing, 12*(19), 3136.

Thenkabail, P. S., Lyon, J. G., & Huete, A. (Eds.). (2018). *Hyperspectral remote sensing of vegetation* (2nd ed.). CRC Press.

Tucker, C. J. (1979). Red and photographic infrared linear combinations for monitoring vegetation. *Remote Sensing of Environment, 8*(2), 127–150.

van der Meer, F. D., van der Werff, H. M. A., van Ruitenbeek, F. J. A., Hecker, C. A., Bakker, W. H., Noomen, M. F., van der Meijde, M., Carranza, E. J. M., de Smeth, J. B., & Woldai, T. (2012). Multi- and hyperspectral geologic remote sensing: A review. *International Journal of Applied Earth Observation and Geoinformation, 14*(1), 112–128.

Weiss, M., Jacob, F., & Duveiller, G. (2020). Remote sensing for agricultural applications: A meta-review. *Remote Sensing of Environment, 236*, 111402.

World Bank. (2016). *High and dry: Climate change, water, and the economy*. World Bank.

Zhang, L., Zhang, L., & Du, B. (2016). Deep learning for remote sensing data: A technical tutorial on the state of the art. *IEEE Geoscience and Remote Sensing Magazine, 4*(2), 22–40.

Zhu, X. X., Tuia, D., Mou, L., Xia, G. S., Zhang, L., Xu, F., & Fraundorfer, F. (2017). Deep learning in remote sensing: A comprehensive review and list of resources. *IEEE Geoscience and Remote Sensing Magazine, 5*(4), 8–36.

---

# THANK YOU!

### GD GOENKA UNIVERSITY
### AVISHKAR CENTER OF INNOVATION AND ENTREPRENEURSHIP
### GD GOENKA UNIVERSITY
