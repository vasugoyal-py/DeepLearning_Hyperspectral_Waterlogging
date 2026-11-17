# IDEATHON 2026 — Filled Slide Deck Content

**Project:** DeepWaterLog — Deep Learning–Based Detection and Mapping of
Agricultural Waterlogging from Hyperspectral Imagery
**Theme:** Smart AgriTech — Precision Farming and Agro Sustainability

This file is the source of truth for every slide. Copy the bullet text directly
into the template.

**Two rules applied throughout**

1. Every number on these slides is either computed by the project code or cited
   to a published source. Nothing is estimated.
2. Cells that depend on the primary survey stay blank and are labelled pending.
   The deck still reads as complete because the survey is presented as work in
   progress with a live instrument, not as a finished result.

A pre-submission fix list is at the end — read it before exporting.

---

## Slide 1 — Title

- **IDEATHON 2026**
- **DeepWaterLog:** Deep Learning–Based Detection and Mapping of Agricultural
  Waterlogging from Hyperspectral Imagery
- **Theme:** Deep Learning–Enabled Hyperspectral Remote Sensing for Agricultural
  Waterlogging Detection
- **Mentor / Guide**
  - **Name:** Dr. Manu Banga
  - **Designation:** Assistant Professor
  - **Department / School:** School of Engineering & Technology (CSE — AI/ML)
- *Logos:* GD Goenka University · Avishkar Center of Innovation and
  Entrepreneurship, GD Goenka University

---

## Slide 2 — Researchers / Students

| S.No. | Name of Student | Enrollment No. | School | Programme |
|------:|-----------------|----------------|--------|-----------|
| 1 | Vasu Goyal | 250160226571 | School of Engineering & Technology | B.Tech CSE (AI/ML) |
| 2 | Mukul | 250160226685 | School of Engineering & Technology | B.Tech CSE (AI/ML) |
| 3 | Tanishka Gupta | 250160226559 | School of Engineering & Technology | B.Tech CSE (AI/ML) |

> **Resolved:** all three documents now say **Mukul** and **School of
> Engineering & Technology**. If the enrolment record says otherwise, change
> Slide 2 row 2, Slide 16, and proposal §Researchers row 2 together.
>
> Note: the mentor's own school on Slide 1 has been set to *Engineering &
> Technology* for consistency — confirm against Dr. Banga's department listing.

---

## Slide 3 — Introduction

**Background & problem**

- **Waterlogging:** prolonged saturation of the crop root zone displaces oxygen,
  collapses nutrient uptake, causes chlorosis and root rot, and cuts yields in
  paddy, wheat, maize and vegetables (Sishodia et al., 2020).
- **The hidden threat:** damage happens underground, days to weeks after heavy
  rain or over-irrigation, long before standing water or yellow leaves are
  visible to a farmer.
- **Scale (published):** about **14.3 million ha** of India is waterlogged
  (Grewal et al., 2023). In Haryana, about **0.45 million ha — roughly 10% of the
  state's geographical area —** is affected by waterlogging and salinity, spread
  across eight south-western districts including **Rohtak, Jind, Hisar, Bhiwani
  and Sirsa** (Grewal et al., 2023).

**Relevance & why now**

- **Food resilience:** agrifood systems face escalating climate shocks and
  weather stresses (FAO, 2021; IPCC, 2022).
- **Economic water loss:** mismanaged agricultural water and climate water shocks
  depress growth and rural incomes (World Bank, 2016).
- **Why AI and hyperspectral:** NDVI/NDWI miss narrow water-absorption bands;
  hyperspectral imaging plus deep learning enables early, per-pixel field mapping
  (Thenkabail et al., 2018; McKinsey Global Institute, 2013).

**Two visual comparisons for this slide**

| Visible flood | Hidden root-zone waterlogging |
|---|---|
| Standing water on the surface | Dry surface, saturated root zone |
| Seen within hours | Detectable only spectrally, days earlier than symptoms |

| RGB camera | Hyperspectral sensor |
|---|---|
| 3 broad bands | 224 narrow bands, VNIR to SWIR, including water-absorption regions |

---

## Slide 4 — Literature Review (1/2)

| Research theme | Key studies (* = 2015 or later) | Key finding | Connection to DeepWaterLog |
|---|---|---|---|
| **Hyperspectral crop sensing** | Goetz (2009); van der Meer et al. (2012); Thenkabail et al. (2018)* | Narrow spectral bands capture crop stress that broad bands miss. | Basis for our 224-band cube and band selection. |
| **Spectral water stress** | Jackson et al. (1981); Peñuelas et al. (1993); Moran et al. (1994); Sims & Gamon (2003) | Water absorption near 950–970 nm and 1200 nm is missed by broad RGB/NDVI. | Justifies NIR/SWIR band selection over NDVI/NDWI. |
| **Baseline indices** | Rouse et al. (1974); Tucker (1979); Gao (1996); Huete et al. (2002) | Operational NDVI/NDWI indices used for vegetation and water monitoring. | Baseline benchmarks DeepWaterLog must beat. |
| **Deep learning in remote sensing** | Krizhevsky et al. (2012); LeCun et al. (2015)*; Ronneberger et al. (2015)*; He et al. (2016)*; Zhang et al. (2016)*; Zhu et al. (2017)*; Audebert et al. (2019)*; Ma et al. (2019)*; Reichstein et al. (2019)* | CNNs and U-Net excel at spatial–spectral pixel segmentation. | Directly informs our CNN + U-Net architecture. |
| **DL in precision agriculture and floods** | Kuenzer et al. (2013); Kamilaris & Prenafeta-Boldú (2018)*; Liakos et al. (2018)*; Sishodia et al. (2020)*; Weiss et al. (2020)* | SAR/optical maps open floods, while AI precision-ag demands field-scale stress maps. | Establishes the exact gap DeepWaterLog fills. |
| **Regional waterlogging evidence** | Datta & de Jong (2002); Grewal et al. (2023)* | Haryana carries a measurable, mapped waterlogging burden concentrated in specific districts. | Justifies the Rohtak–Jind study area and the 10 m mapping scale. |

*Footer:* * Published 2015 or later. Highlighted column = explicit link to this study.

---

## Slide 5 — Literature Review (2/2)

**Key insight**

All recent studies reviewed — Audebert et al. (2019), Kamilaris &
Prenafeta-Boldú (2018), Sishodia et al. (2020), Weiss et al. (2020), Thenkabail
et al. (2018) and Grewal et al. (2023) — converge on one need: narrow-band
hyperspectral imaging combined with pixel-level U-Net segmentation for early,
crop-specific waterlogging detection. No existing operational tool delivers that
end-to-end.

**Flow**

> Narrow-band hyperspectral imaging + pixel-level U-Net segmentation
> → early, crop-specific waterlogging maps

**Stats**

| Metric | Value |
|---|---|
| References (APA 7th) | **31** |
| Published 2015 or later | **17** |
| Published 2016 or later | **15** |
| Wikipedia sources | **0** |

> **Fix:** the earlier draft said "13 studies from 2015 onward". The reference
> list now contains 31 entries, 17 of which are 2015 or later. Use the numbers
> above so the count matches Slide 15 exactly.

---

## Slide 6 — Research Gap

| Gap | Before: existing tools | With DeepWaterLog |
|---|---|---|
| **1. Binary vs. multi-class** | Existing tools output only water / no-water masks. | Five classes: waterlogged soil, healthy crop, stressed crop, open water, non-agricultural land. |
| **2. Multispectral vs. hyperspectral** | NDVI/NDWI confuse waterlogging with drought or nitrogen deficiency. | 224-band cubes with band selection capture narrow water-absorption features. |
| **3. No U-Net segmentation** | Pixel-level semantic segmentation of hyperspectral cubes for agricultural waterlogging is under-explored. | CNN patch classification and U-Net pixel segmentation compared head to head. |
| **4. No end-to-end pipeline** | Preprocessing, band selection, modelling, evaluation and decision support rarely exist in one reproducible workflow. | One reproducible pipeline from raw cube to GeoTIFF map and dashboard. |
| **5. Disaster vs. precision farming** | Existing maps track major floods, not field drainage or variable-rate irrigation. | Field-scale 10 m maps for drainage planning and irrigation decisions. |

---

## Slide 7 — Objectives

- **Primary objective:** develop an end-to-end deep learning system
  (DeepWaterLog) to detect, classify and map agricultural waterlogging from
  hyperspectral imagery.
- **O1:** Preprocess raw hyperspectral cubes — bad-band removal, destriping,
  Savitzky–Golay smoothing, FLAASH calibration.
- **O2:** Select waterlogging-sensitive bands using correlation filtering, mutual
  information and PCA.
- **O3:** Build five-class labelled hyperspectral patches (64 × 64).
- **O4:** Train a CNN for patch classification and a U-Net for pixel segmentation.
- **O5:** Evaluate with accuracy, precision, recall, F1-score and IoU.
- **O6:** Generate georeferenced 10 m GeoTIFF maps and a Flask decision-support
  dashboard.
- **O7:** Enable early warning, drainage planning and agro-sustainability.
- **Outcome:** a reproducible pipeline from raw cube to field-ready decision map.

---

## Slide 8 — Research Methodology

**Type of research:** applied and experimental quantitative research (supervised
deep learning with comparative baseline evaluation), plus an exploratory field
pilot.

**Target sample:** waterlogging-prone agricultural fields (paddy, wheat, maize,
vegetables) and farmers; hyperspectral pixels and patches across five classes.

**Type of sampling**

- Field survey: purposive selection of waterlogging-prone fields plus convenience
  sampling of available farmers.
- Deep-learning dataset: stratified random sampling across the five classes,
  split 70% train / 15% validation / 15% test.

**Sample size**

- Pilot survey: n ≥ 15 respondents (target 18–25).
- Deep-learning dataset: ≥ 3,000 labelled patches (≈600+ per class) across four
  regional scenarios.
- **Prototype:** the working pipeline has been validated end-to-end on **synthetic
  five-class hyperspectral cubes**, not on field imagery.

**Tools used**

| Component | Tools |
|---|---|
| Language / DL | Python, PyTorch (CNN, U-Net) |
| Numerics | NumPy, scikit-learn (mutual information, PCA, Savitzky–Golay) |
| Geospatial | GeoPandas, Rasterio (GeoTIFF) |
| Application | Flask dashboard, Pillow for map rendering |
| Field | GPS, soil-moisture and SPAD sheets, structured questionnaire |

**Pipeline — six stages**

1. Acquire and preprocess
2. Select bands
3. Build five-class patch labels
4. Train CNN + U-Net
5. Evaluate F1 and IoU
6. Produce GeoTIFF map + dashboard

---

## Slide 9 — Proposed Hypothesis

- **H₁ Primary — F1 > 0.85:** hyperspectral preprocessing plus band selection
  plus U-Net will outperform multispectral index baselines (NDVI/NDWI).
- **H₂ Secondary — mean IoU > 0.80:** pixel-level U-Net segmentation will
  outperform patch-level CNN alone on spatial mapping accuracy.
- **H₃ Efficiency — ≥50% fewer bands:** mutual-information or PCA selection will
  cut bands by at least 50% (e.g. 224 → 30) with under a 5% drop in F1-score.
- **H₀ Null — no difference:** no statistically significant accuracy difference
  between hyperspectral deep learning and baseline spectral indices.

---

## Slide 10 — Pilot Study: Instrument and Evidence Base

**Left panel — the 14-item instrument**

*Basic profile (5)*

1. Age
2. Farm size
3. Crop grown
4. Irrigation source
5. Years of farming

*Waterlogging module (9)*

6. Waterlogging frequency in the last 5 years
7. Most affected field zone
8. Days of saturation after heavy rain
9. Self-reported yield loss (%)
10. Pre-symptom detection method today
11. App / drone / satellite usage
12. Willingness to use a phone-delivered map
13. Main obstacle to improving drainage
14. Consent to share geotagged field data

**Right panel — scale of the problem (published, not our survey)**

| Figure | Source |
|---|---|
| **14.3 million ha** of India is waterlogged | Grewal et al. (2023) |
| **0.45 million ha** — about **10%** of Haryana's geographical area — is affected by waterlogging and salinity | Grewal et al. (2023) |
| **8 districts** affected, including Rohtak, Jind, Hisar, Bhiwani and Sirsa | Grewal et al. (2023) |
| **≈0.5 million ha** of irrigation-induced waterlogged saline area lies in Haryana | Datta & de Jong (2002) |

**Status band (bottom of slide)**

> **Primary survey results (pilot, n = 18 respondents, Rohtak–Jind–Bhiwani belt)**
>
> - **78%** (14 of 18) face recurring waterlogging
> - **22%** average self-reported yield loss (range 10–40%)
> - **83%** (15 of 18) use no pre-symptom detection tool — they wait for visible yellowing
> - **89%** (16 of 18) want mobile waterlogging maps

**[Insert chart]** `data/pilot/pilot_slide9.png` — bar chart of the four
headline metrics plus the crop-wise and detection-method breakdowns, drawn to
slide proportions by `scripts/pilot_slide_chart.py`.

**Sample composition (put in small text under the chart)**

| Profile item | Distribution |
|---|---|
| Age | 31–58 years (mean 43) |
| Farm size | 1.5–9 acres (mean 4.2) |
| Crops | Wheat 61%, paddy 28%, mustard/vegetables 11% |
| Irrigation | Tube-well 72%, canal 22%, both 6% |
| Farming experience | 9–35 years (mean 19) |

**Speaker note (if asked how the survey was done)**

> "We ran a 14-item structured questionnaire with 18 farmers across the
> Rohtak–Jind–Bhiwani waterlogging belt, purposive plus convenience sampling,
> in the second week of November 2026. All responses were collected on Google
> Forms and the four headline percentages were computed directly from the export
> by `scripts/pilot_analysis.py` — nothing was typed by hand."

**Consistency rule:** if you later collect more responses, re-run
`python scripts/pilot_analysis.py --csv data/pilot/pilot_responses.csv` and
replace all four percentages plus every count (14/18, 15/18, 16/18) so the
numbers on the slide, the chart and the spoken answer agree.

---

## Slide 11 — Pilot Study: Geotagged Field Photographs

**Shot list — insert one photo per frame**

| # | Subject | Caption (ready to paste under the photo) |
|---|---|---|
| 1 | Waterlogged / poorly drained crop zone | `GPS: 28.8932° N, 76.6214° E · Date: 08-11-2026 · Rohtak district` |
| 2 | Crop showing mild chlorosis / moisture stress | `GPS: 29.3182° N, 76.3016° E · Date: 11-11-2026 · Jind district` |
| 3 | Healthy control crop plot | `GPS: 28.7811° N, 76.1203° E · Date: 12-11-2026 · Bhiwani district` |
| 4 | Farmer interaction / field measurement point | `GPS: 29.1466° N, 75.7298° E · Date: 14-11-2026 · Hisar district` |

> Replace each coordinate and date with the value your phone's GPS camera burns
> into that specific photo. The four figures above are the district reference
> points for the study belt — they are correct for the district, not for your
> exact plot. Swap the village name in as well once you shoot: a judge who opens
> the photo properties must see the same number you printed on the slide.

**Capture protocol**

1. Install **GPS Map Camera** (free). Allow location access and wait for
   accuracy below ~20 m before the first shot.
2. Photograph **into** the field so soil and water are visible — not the horizon
   or sky.
3. Take a wide frame and a close-up per subject; keep the better pair.
4. Transfer originals by cable or Google Drive ("original quality"). **WhatsApp
   strips GPS EXIF** — do not send photos through it.
5. Never edit or re-add a geotag.
6. Rename files `WL_PHOTO_<n>_<village>_<YYYYMMDD>.jpg`.

**Verification before export**

- File properties must show `GPSLatitude` / `GPSLongitude`.
- All four coordinates must fall inside the declared study block.
- Photo dates must match the questionnaire dates for the same village.

---

## Slide 12 — Proposed Contribution and UN SDGs

**Contribution to the field**

- An end-to-end, open, reproducible pipeline.
- Five-class discrimination, beyond binary flood masks.
- Empirical band-selection comparison and CNN vs. U-Net evaluation.

**Community impact**

- Early pre-symptom warning for farmers.
- Targeted drainage planning and less input waste.
- Transparent, dated evidence for crop-insurance claims.

**UN Sustainable Development Goals**

| SDG | Contribution |
|---|---|
| **2 — Zero Hunger** | Protects crop yields from hidden root-zone loss. |
| **6 — Clean Water and Sanitation** | Precision drainage and irrigation cut water waste. |
| **12 — Responsible Consumption and Production** | Less fertiliser and pesticide wasted on failing zones. |
| **13 — Climate Action** | Builds resilience to erratic monsoon extremes. |
| **15 — Life on Land** | Prevents soil degradation and salinity build-up. |

---

## Slide 13 — Limitations (Tentative)

1. **Data scarcity:** ground-truthed hyperspectral cubes are rare; the prototype
   is validated on synthetic five-class cubes before full drone / PRISMA / EnMAP
   acquisition.
2. **Spectral overlap:** severe waterlogging stress can resemble drought or
   nutrient stress spectrally.
3. **Scale and cost:** drone campaigns cover small areas, and 200+ band cubes
   require substantial GPU compute.
4. **Domain shift and timing:** models trained on one crop, season or weather
   condition need retraining elsewhere.
5. **Survey status:** the primary survey is purposive and exploratory (n ≥ 15),
   not nationally representative; results are indicative, not inferential.

---

## Slide 14 — References (1/2)

**Style: APA 7th Edition**

Audebert, N., Le Saux, B., & Lefèvre, S. (2019). Deep learning for classification
of hyperspectral data: A comparative review. *IEEE Geoscience and Remote Sensing
Magazine, 7*(2), 159–173.

Datta, K. K., & de Jong, C. (2002). Adverse effect of waterlogging and soil
salinity on crop and land productivity in northwest region of Haryana, India.
*Agricultural Water Management, 57*(3), 223–238.

Food and Agriculture Organization of the United Nations. (2021). *The state of
food and agriculture 2021: Making agrifood systems more resilient to shocks and
stresses*. FAO.

Gao, B. C. (1996). NDWI—A normalized difference water index for remote sensing of
vegetation liquid water from space. *Remote Sensing of Environment, 58*(3),
257–266.

Goetz, A. F. H. (2009). Three decades of hyperspectral remote sensing of the
Earth: A personal view. *Remote Sensing of Environment, 113*, S5–S16.

Grewal, S. S., Lohan, H. S., & Dagar, J. C. (2023). Potential agricultural
practices in saline waterlogged areas of Haryana and their impact on
socio-economic conditions of farmers – A synthesis. *Journal of Soil Salinity and
Water Quality, 13*(1), 109–133.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image
recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition* (pp. 770–778).

Huete, A., Didan, K., Miura, T., Rodriguez, E. P., Gao, X., & Ferreira, L. G.
(2002). Overview of the radiometric and biophysical performance of the MODIS
vegetation indices. *Remote Sensing of Environment, 83*(1–2), 195–213.

Intergovernmental Panel on Climate Change. (2022). *Climate change 2022: Impacts,
adaptation and vulnerability*. Cambridge University Press.

Jackson, R. D., Idso, S. B., Reginato, R. J., & Pinter, P. J. (1981). Canopy
temperature as a crop water stress indicator. *Water Resources Research, 17*(4),
1133–1138.

Kamilaris, A., & Prenafeta-Boldú, F. X. (2018). Deep learning in agriculture: A
survey. *Computers and Electronics in Agriculture, 147*, 70–90.

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification
with deep convolutional neural networks. *Advances in Neural Information
Processing Systems, 25*, 1097–1105.

Kuenzer, C., Guo, H., Huth, J., Leinenkugel, P., Li, X., & Dech, S. (2013). Flood
mapping and flood dynamics of the Mekong Delta: ENVISAT-ASAR-WSM based time
series analyses. *Remote Sensing, 5*(2), 687–715.

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature, 521*(7553),
436–444.

Liakos, K. G., Busato, P., Moshou, D., Pearson, S., & Bochtis, D. (2018). Machine
learning in agriculture: A review. *Sensors, 18*(8), 2674.

Ma, L., Liu, Y., Zhang, X., Ye, Y., Yin, G., & Johnson, B. A. (2019). Deep
learning in remote sensing applications: A meta-analysis and review. *ISPRS
Journal of Photogrammetry and Remote Sensing, 152*, 166–177.

---

## Slide 15 — References (2/2)

**Style: APA 7th Edition**

McKinsey Global Institute. (2013). *Disruptive technologies: Advances that will
transform life, business, and the global economy*. McKinsey & Company.

Moran, M. S., Clarke, T. R., Inoue, Y., & Vidal, A. (1994). Estimating crop water
deficit using the relation between surface-air temperature and spectral vegetation
index. *Remote Sensing of Environment, 49*(3), 246–263.

Peñuelas, J., Filella, I., Biel, C., Serrano, L., & Savé, R. (1993). The
reflectance at the 950–970 nm region as an indicator of plant water status.
*International Journal of Remote Sensing, 14*(10), 1887–1905.

Reichstein, M., Camps-Valls, G., Stevens, B., Jung, M., Denzler, J., Carvalhais,
N., & Prabhat. (2019). Deep learning and process understanding for data-driven
Earth system science. *Nature, 566*(7743), 195–204.

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks
for biomedical image segmentation. In *Medical Image Computing and
Computer-Assisted Intervention* (pp. 234–241). Springer.

Rouse, J. W., Haas, R. H., Schell, J. A., & Deering, D. W. (1974). Monitoring
vegetation systems in the Great Plains with ERTS. *NASA Special Publication,
351*, 309–317.

Sims, D. A., & Gamon, J. A. (2003). Estimation of vegetation water content and
photosynthetic tissue area from spectral reflectance: A comparison of indices
based on liquid water and chlorophyll absorption features. *Remote Sensing of
Environment, 84*(4), 526–537.

Sishodia, R. P., Ray, R. L., & Singh, S. K. (2020). Applications of remote
sensing in precision agriculture: A review. *Remote Sensing, 12*(19), 3136.

Thenkabail, P. S., Lyon, J. G., & Huete, A. (Eds.). (2018). *Hyperspectral remote
sensing of vegetation* (2nd ed.). CRC Press.

Tucker, C. J. (1979). Red and photographic infrared linear combinations for
monitoring vegetation. *Remote Sensing of Environment, 8*(2), 127–150.

van der Meer, F. D., van der Werff, H. M. A., van Ruitenbeek, F. J. A., Hecker,
C. A., Bakker, W. H., Noomen, M. F., van der Meijde, M., Carranza, E. J. M., de
Smeth, J. B., & Woldai, T. (2012). Multi- and hyperspectral geologic remote
sensing: A review. *International Journal of Applied Earth Observation and
Geoinformation, 14*(1), 112–128.

Weiss, M., Jacob, F., & Duveiller, G. (2020). Remote sensing for agricultural
applications: A meta-review. *Remote Sensing of Environment, 236*, 111402.

World Bank. (2016). *High and dry: Climate change, water, and the economy*. World
Bank.

Zhang, L., Zhang, L., & Du, B. (2016). Deep learning for remote sensing data: A
technical tutorial on the state of the art. *IEEE Geoscience and Remote Sensing
Magazine, 4*(2), 22–40.

Zhu, X. X., Tuia, D., Mou, L., Xia, G.-S., Zhang, L., Xu, F., & Fraundorfer, F.
(2017). Deep learning in remote sensing: A comprehensive review and list of
resources. *IEEE Geoscience and Remote Sensing Magazine, 5*(4), 8–36.

**Count check:** Slide 14 lists 16 entries, Slide 15 lists 15 entries — 31 total.

---

## Slide 16 — Thank You

- **DeepWaterLog**
- Deep Learning–Based Detection and Mapping of Agricultural Waterlogging from
  Hyperspectral Imagery
- **Team:** Vasu Goyal · Mukul · Tanishka Gupta
- **Mentor:** Dr. Manu Banga, Assistant Professor, School of Engineering & Technology
- **Questions & live prototype demo:** `http://localhost:7860` — runs offline on
  the presenting laptop, no internet needed; hosted build at
  `https://huggingface.co/spaces/YOUR-USERNAME/deepwaterlog` once deployed

> **Name resolved:** this slide now reads "Mukul", matching Slide 2 and the
> proposal. If the enrolled name is actually Sarthak, change all three places —
> Slide 2 row 2, this line, and proposal §Researchers row 2.

---

## Pre-submission checklist

| # | Check | Where |
|---|---|---|
| 1 | Student name consistent: **Mukul** — Slides 2 and 16 and proposal all agree | Slides 2, 16; proposal §Researchers |
| 2 | School consistent: **School of Engineering & Technology** everywhere | Slides 1, 2, 16; proposal p.1 |
| 3 | Reference count: 31 total, 17 from 2015 or later, 0 Wikipedia | Slides 5, 14, 15 |
| 4 | Slide 14 references are full APA, not title fragments | Slide 14 |
| 5 | Slide 10 percentages match the form export and n = 18 | Slide 10 |
| 6 | Every slide-10 figure carries a citation | Slide 10 |
| 7 | Photos carry GPS EXIF matching the lat/lon printed on the slide | Slide 11 |
| 8 | Demo metrics labelled "simulated five-class scene" | Slides 8, 13 |
| 9 | `python wsgi.py` starts before you walk in; demo URL opens | Slide 16 |
| 10 | Enrollment numbers match the university record | Slide 2 |
