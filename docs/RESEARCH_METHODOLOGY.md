# DeepWaterLog — Research Methodology

**Deep Learning–Enabled Hyperspectral Remote Sensing for Agricultural Waterlogging Detection**
Version 1.1 · Methodology note for manuscript preparation

**Theme:** Smart AgriTech — Precision Farming and Agro Sustainability

This document assembles the full research methodology for the DeepWaterLog project:
literature review, research gap, objectives, methodology, proposed hypothesis,
pilot study (fieldwork), proposed contribution, limitations, and references
(APA 7th edition). It is written so that the methodology chapter of a thesis or
paper can be assembled directly from it.

> **Note on sources:** all citations are peer-reviewed journal articles, books,
> or institutional reports. Wikipedia and other open-edit encyclopaedias are not
> used as sources anywhere in this project.

---

## 0. Executive Summary

Agricultural waterlogging — excess soil moisture that stresses or kills crop
roots — causes significant yield loss worldwide. Hyperspectral remote sensing
captures hundreds of narrow spectral bands, providing detailed spectral
signatures that can distinguish waterlogged soil, healthy vegetation, stressed
crops, and open water. DeepWaterLog applies deep learning (CNNs for patch-level
classification and U-Net for pixel-level segmentation) to preprocessed
hyperspectral imagery to detect, classify, and map waterlogged areas. Performance
is evaluated using accuracy, precision, recall, F1-score, and Intersection over
Union (IoU). Outputs support early waterlogging detection, crop-damage
assessment, drainage planning, and sustainable agricultural management.

---

## 1. Literature Review

### 1.1 Hyperspectral Remote Sensing

Hyperspectral remote sensing acquires images in hundreds of narrow, contiguous
spectral bands across the visible, near-infrared, and shortwave-infrared regions
(Goetz, 2009; Thenkabail et al., 2018; van der Meer et al., 2012). Unlike
multispectral sensors that record a few broad bands, hyperspectral sensors record
a continuous spectrum for each pixel, enabling identification of subtle spectral
features related to plant pigments, cell structure, water content, and soil
moisture (Thenkabail et al., 2018).

Hyperspectral imaging has been applied to agriculture for crop-type
discrimination, nutrient-status assessment, disease detection, and yield
prediction (Thenkabail et al., 2018). The fine spectral resolution allows
detection of plant stress before it becomes visible, making hyperspectral data
well suited to early waterlogging detection.

### 1.2 Waterlogging and Crop Stress Detection

Waterlogging occurs when the soil root zone becomes saturated, depriving roots
of oxygen and reducing nutrient uptake (Jackson et al., 1981; Moran et al.,
1994). Symptoms include leaf yellowing, stunting, root rot, and reduced yield.
Remote sensing detects waterlogging through changes in canopy spectral
reflectance: stressed vegetation shows reduced chlorophyll absorption in the red
band, altered near-infrared reflectance related to cell structure, and altered
shortwave-infrared signatures related to leaf water content (Peñuelas et al.,
1993; Sims & Gamon, 2003).

Traditional indices such as NDVI (Rouse et al., 1974; Tucker, 1979) and NDWI
(Gao, 1996) use only a few bands and often cannot separate waterlogging stress
from drought stress, nutrient deficiency, or disease (Peñuelas et al., 1993;
Sims & Gamon, 2003). Hyperspectral data, with its continuous spectral sampling,
offers greater discriminatory power for these subtle differences.

### 1.3 Deep Learning for Remote Sensing

Convolutional Neural Networks (CNNs) have become the dominant approach for
image classification in remote sensing (Zhu et al., 2017; Zhang et al., 2016).
CNNs learn hierarchical spatial-spectral features directly from image data,
avoiding the need for hand-crafted indices (Krizhevsky et al., 2012; LeCun et
al., 2015; He et al., 2016).

U-Net, originally developed for biomedical image segmentation (Ronneberger et
al., 2015), has been widely adapted to remote sensing for land-cover mapping,
crop segmentation, and disaster damage assessment. Its encoder–decoder structure
with skip connections preserves spatial detail while learning contextual
features, making it suitable for pixel-level waterlogging mapping. Audebert et
al. (2019) review deep learning specifically for hyperspectral classification,
and Ma et al. (2019) and Reichstein et al. (2019) document the broader adoption
of deep learning across remote sensing and Earth-system science.

### 1.4 Deep Learning for Agriculture and Precision Farming

Kamilaris and Prenafeta-Boldú (2018) survey deep learning in agriculture and
identify image-based crop monitoring as a leading application. Liakos et al.
(2018) review machine learning across the agricultural value chain, while Weiss
et al. (2020) and Sishodia et al. (2020) provide meta-reviews of remote sensing
for agriculture and precision agriculture respectively, emphasising the need for
field-scale, stress-specific products.

### 1.5 Flood and Waterlogging Mapping

SAR-based flood mapping (Kuenzer et al., 2013) and optical water indices (Gao,
1996) detect open water but not the sub-surface, root-zone waterlogging that
most damages crops. This distinction — between "visible flood" and "hidden
waterlogging stress" — is the central motivation of DeepWaterLog, and no
reviewed study combines full hyperspectral cubes with CNN/U-Net for multi-class
agricultural waterlogging mapping at field scale.

### 1.6 Summary of the Literature

| Theme | Key findings | Sources |
|---|---|---|
| Hyperspectral remote sensing | Hundreds of narrow bands enable subtle spectral-feature detection | Goetz (2009); Thenkabail et al. (2018); van der Meer et al. (2012) |
| Waterlogging detection | Water-absorption features near 950–970 nm and 1200 nm indicate canopy water status; NDVI/NDWI are limited | Peñuelas et al. (1993); Sims & Gamon (2003); Jackson et al. (1981); Tucker (1979); Gao (1996) |
| Deep learning (CNN/U-Net) | CNNs and U-Net achieve state-of-the-art results in image classification and segmentation | Krizhevsky et al. (2012); LeCun et al. (2015); He et al. (2016); Ronneberger et al. (2015) |
| DL in remote sensing | Deep learning is now standard across remote-sensing disciplines | Zhu et al. (2017); Zhang et al. (2016); Ma et al. (2019); Reichstein et al. (2019); Audebert et al. (2019) |
| DL in agriculture | Image-based crop monitoring is a leading DL application | Kamilaris & Prenafeta-Boldú (2018); Liakos et al. (2018); Weiss et al. (2020); Sishodia et al. (2020) |
| Flood mapping | SAR/optical methods map open water, not root-zone waterlogging | Kuenzer et al. (2013); Gao (1996) |
| **Gap** | **Few systems combine hyperspectral preprocessing with CNN/U-Net for field-scale multi-class waterlogging mapping** | — |

---

## 2. Research Gap

Despite the demonstrated value of hyperspectral imaging and deep learning in
agriculture, several gaps remain:

1. **Limited hyperspectral + deep learning integration.** Most waterlogging and
   flood remote-sensing studies use multispectral data (e.g., Sentinel-2,
   Landsat) or SAR backscatter (Kuenzer et al., 2013). Few studies combine full
   hyperspectral cubes with deep semantic-segmentation models (U-Net) for
   field-scale waterlogging mapping.

2. **Binary rather than multi-class mapping.** Existing systems often produce a
   simple flood/non-flood or water/land mask. They do not distinguish
   waterlogged soil from healthy crops, stressed crops, and open water —
   distinctions that matter for drainage planning and crop-damage assessment.

3. **Lack of end-to-end pipelines.** Few systems cover the complete workflow
   from raw hyperspectral preprocessing (noise removal, atmospheric correction,
   band selection) through deep learning modelling to georeferenced map output
   and decision-support visualisation.

4. **Limited agro-sustainability focus.** Existing waterlogging tools are
   typically disaster-response oriented rather than designed for precision
   farming, variable-rate drainage, irrigation scheduling, and sustainable
   agricultural management (Sishodia et al., 2020; Weiss et al., 2020).

5. **Under-exploited hyperspectral specificity.** The diagnostic water-absorption
   features that distinguish waterlogging from drought or nutrient stress
   (Peñuelas et al., 1993; Sims & Gamon, 2003) are rarely exploited by
   operational agricultural monitoring products.

DeepWaterLog addresses these gaps by providing a complete, reproducible,
deep-learning–based hyperspectral system for multi-class waterlogging detection
and mapping, targeted at Smart AgriTech and agro sustainability.

---

## 3. Objectives

The project has the following objectives:

**O1.** To preprocess raw hyperspectral imagery (bad-band removal, destriping,
noise filtering, atmospheric correction, radiometric calibration) to produce
surface-reflectance cubes suitable for deep learning.

**O2.** To select waterlogging-sensitive spectral bands and reduce dimensionality
(using correlation analysis, mutual information, or PCA) while retaining
diagnostic spectral features in the visible, red-edge, NIR, and SWIR regions.

**O3.** To prepare labelled training data pairing hyperspectral patches with
field-validated waterlogging classes (waterlogged soil, healthy crop, stressed
crop, open water, non-agricultural).

**O4.** To train and evaluate a CNN for patch-level waterlogging classification
and a U-Net for pixel-level waterlogging segmentation.

**O5.** To assess model performance using accuracy, precision, recall, F1-score,
and Intersection over Union (IoU).

**O6.** To generate georeferenced waterlogging extent and severity maps and
visualise them in a decision-support web interface for drainage planning and
crop-damage assessment.

**O7.** To align the system with the Smart AgriTech — Precision Farming and Agro
Sustainability theme by supporting early detection, sustainable water
management, and reduced crop loss.

---

## 4. Research Methodology

### 4.1 Type of Research

- **Nature:** Applied and experimental research with an exploratory pilot
  component.
- **Approach:** Quantitative and data-driven; supervised deep learning
  (classification and semantic segmentation).
- **Design:** Cross-sectional field study combined with computational modelling,
  including a comparative evaluation against baseline spectral-index methods.

### 4.2 Study Areas and Data

Hyperspectral imagery will be acquired from airborne or spaceborne hyperspectral
sensors (e.g., AVIRIS-NG, PRISMA, EnMAP, or drone-mounted hyperspectral cameras)
over agricultural fields with known waterlogging history. Study areas will
include paddy, wheat, maize, and vegetable fields in the Indo-Gangetic plain and
other waterlogging-prone regions.

Field data will include GPS-tagged soil moisture measurements, crop health
scores, and waterlogging boundary delineations collected during the pilot study
(Section 6).

### 4.3 Target Sample

- **Primary target:** Agricultural fields — and the farmers or farm managers who
  cultivate them — in waterlogging-prone regions of the Indo-Gangetic Plain,
  specifically paddy, wheat, maize, and vegetable plots.
- **Secondary target:** Hyperspectral image pixels and patches extracted from
  drone- or satellite-acquired imagery over those fields.

### 4.4 Type of Sampling

- **Field survey:** Mixed sampling — purposive sampling to select
  waterlogging-prone fields (based on historical drainage and crop-loss records),
  combined with convenience sampling of farmer respondents within the selected
  villages.
- **Deep-learning dataset:** Stratified random sampling of hyperspectral patches
  across the five waterlogging classes, to ensure balanced representation in
  training and evaluation.

### 4.5 Sample Size

- **Pilot field survey:** minimum 15 respondents/plots (target 20–30 for
  robustness).
- **Deep-learning dataset:** target ≥ 3,000 labelled patches across five classes
  (approximately 600+ per class), split 70% training / 15% validation / 15% test.
- **Image sources:** drone-mounted hyperspectral sensor and/or spaceborne
  hyperspectral data (e.g., PRISMA, EnMAP) over four pilot scenarios.

### 4.6 Stage 1 — Hyperspectral Preprocessing

Raw hyperspectral cubes undergo:
- **Bad-band removal:** bands affected by water vapour absorption (around
  1400 nm and 1900 nm) and low signal-to-noise ratio are excluded (Goetz, 2009).
- **Destriping:** along-track striping artefacts are reduced using wavelet- or
  polynomial-based methods.
- **Noise filtering:** Savitzky–Golay or wavelet filtering smooths spectral
  noise while preserving absorption features.
- **Atmospheric correction:** FLAASH or QUAC converts at-sensor radiance to
  surface reflectance.
- **Radiometric calibration:** digital numbers are converted to physical
  reflectance units.

Output: surface-reflectance hyperspectral cube.

### 4.7 Stage 2 — Band Selection and Dimensionality Reduction

To reduce redundancy and computational load:
- **Correlation analysis:** bands with pairwise correlation above 0.95 are
  removed.
- **Mutual information:** the top-K bands most informative about waterlogging
  class labels are retained.
- **PCA:** principal components capturing the majority of variance are computed
  as alternative inputs.

Selected bands/components emphasise chlorophyll (visible), vegetation stress
(red-edge), canopy structure and moisture (NIR), and soil/crop moisture (SWIR),
consistent with the spectral water-stress literature (Peñuelas et al., 1993;
Sims & Gamon, 2003).

### 4.8 Stage 3 — Training Data Preparation

Preprocessed hyperspectral patches (e.g., 64 × 64 pixels) are extracted with a
configurable stride. Each patch is labelled with its majority class for CNN
training and retains a per-pixel label patch for U-Net training, based on field
validation data. Patches are normalised (min–max or z-score), augmented (flips,
rotations, spectral jitter), and split into training (70%), validation (15%),
and test (15%) sets.

### 4.9 Stage 4 — Deep Learning Modelling

**CNN (patch-level classification):** A custom CNN (or ResNet/EfficientNet
backbone) classifies each patch into one of the five classes. The network learns
spectral-spatial features from the selected bands (Krizhevsky et al., 2012;
He et al., 2016).

**U-Net (pixel-level segmentation):** A U-Net with skip connections performs
semantic segmentation, assigning every pixel to a waterlogging class
(Ronneberger et al., 2015). The output is a georeferenced classification map.

Models are trained with the Adam optimiser, cross-entropy (or Dice/focal) loss,
early stopping, and class balancing.

### 4.10 Stage 5 — Evaluation

Models are evaluated on the held-out test set using:
- **Accuracy:** fraction of correctly classified patches/pixels.
- **Precision:** TP / (TP + FP) per class.
- **Recall:** TP / (TP + FN) per class.
- **F1-score:** harmonic mean of precision and recall.
- **Intersection over Union (IoU):** TP / (TP + FP + FN) for segmentation maps.

Per-class and macro-averaged metrics are reported, along with confusion
matrices.

### 4.11 Stage 6 — Mapping and Decision Support

U-Net segmentation outputs are georeferenced to the input imagery coordinates
and exported as GeoTIFF maps. A Flask + Leaflet web interface allows users to:
- visualise waterlogging extent and severity maps,
- overlay field boundaries,
- export maps and reports for drainage planning and crop-damage assessment.

### 4.12 Tools and Techniques

| Component | Tools / Techniques |
|---|---|
| Data acquisition | Drone-mounted hyperspectral camera; PRISMA/EnMAP imagery; GPS for geotagging |
| Field measurements | Soil-moisture meter (volumetric water content), SPAD chlorophyll meter, visual stress scoring sheet, handheld GPS |
| Preprocessing | Python, NumPy, SciPy; bad-band removal, Savitzky–Golay/wavelet filtering, FLAASH/QUAC atmospheric correction |
| Band selection | Correlation analysis, mutual-information ranking, PCA |
| Deep learning | PyTorch; CNN (custom/ResNet/EfficientNet) for classification; U-Net for segmentation |
| Evaluation | Accuracy, precision, recall, F1-score, IoU, confusion matrices |
| Mapping and UI | rasterio/GeoPandas for GeoTIFF; Flask + Leaflet decision-support interface |
| Survey instrument | Structured questionnaire (Section 6.4) plus semi-structured farmer interviews |

---

## 5. Proposed Hypothesis

**H₁ (Primary):** A deep learning system combining hyperspectral preprocessing,
band selection, and U-Net segmentation will achieve significantly higher
waterlogging detection accuracy (F1-score > 0.85) than multispectral index–based
methods (e.g., NDVI, NDWI) on the same test data.

**H₂ (Secondary):** A U-Net trained on hyperspectral bands will produce more
accurate pixel-level waterlogging maps (mean IoU > 0.80) than a CNN trained for
patch-level classification alone.

**H₃ (Tertiary):** Band selection (mutual information / PCA) will reduce input
dimensionality by at least 50% with less than 5% degradation in F1-score,
improving computational efficiency without meaningful loss of accuracy.

**H₀ (Null):** There is no statistically significant difference between the
proposed hyperspectral deep-learning system and baseline spectral-index methods.

---

## 6. Pilot Study (Fieldwork)

### 6.1 Purpose

A pilot study will collect ground-truth data to train and validate the deep
learning models, and to confirm that spectral signatures in hyperspectral
imagery correlate with field-observed waterlogging conditions.

### 6.2 Study Site

A 50–100 hectare agricultural block in a waterlogging-prone region (e.g., Punjab
or Haryana, India) growing paddy or wheat will be selected. The site will
include zones with varying drainage quality: well-drained, moderately
waterlogged, severely waterlogged, and open water.

### 6.3 Data Collection

- **Hyperspectral imagery:** Drone- or aircraft-mounted hyperspectral sensor
  flown over the site during the monsoon or post-irrigation period.
- **Field measurements:** GPS-tagged soil moisture (volumetric water content)
  at 30–50 points across moisture zones.
- **Crop health:** Visual stress scores (healthy, mild stress, severe stress)
  and chlorophyll meter (SPAD) readings at the same points.
- **Waterlogging boundaries:** Field-mapped polygons of waterlogged areas using
  a handheld GPS.
- **Photographs:** Four geotagged photographs of crop condition at sampling
  points (severe waterlogging, mild waterlogging with chlorosis, healthy control,
  and farmer interview/soil sampling).

### 6.4 Questionnaire (minimum 15 respondents)

*Ten questions uniquely relevant to the research, preceded by basic demographic
and farm-profile items (age, education, farm size, irrigation source, years of
farming experience).*

1. What is the primary crop grown on your field during the monsoon/kharif season?
2. How many times in the last five years has your field experienced waterlogging?
3. Which part of the field is most frequently waterlogged (low-lying area, near
   canal, centre, edges)?
4. On average, how many days does water remain standing, or the soil remain
   saturated, after heavy rain?
5. What is the approximate crop-yield loss (%) you attribute to waterlogging in
   a bad year?
6. Do you currently use any method to detect or monitor waterlogging before
   visible crop damage appears? (Yes/No; if yes, describe)
7. Have you ever used a mobile app, drone, or satellite-based service for
   crop/field monitoring? (Yes/No)
8. Would a field-scale waterlogging map delivered to your phone help you plan
   drainage or irrigation? (Yes/No/Maybe)
9. What is the main obstacle to improving drainage on your field (cost, labour,
   land tenure, lack of information, other)?
10. Would you be willing to share field photographs or coordinates for a research
    study on waterlogging detection? (Yes/No)

### 6.5 Laboratory Analysis

Soil samples will be analysed for bulk density, texture, and hydraulic
conductivity to contextualise waterlogging severity. Crop tissue samples may be
analysed for nutrient status to rule out non-waterlogging stress causes.

### 6.6 Expected Outcome

A labelled dataset of hyperspectral patches with field-validated waterlogging
classes, sufficient to train and initially validate the CNN and U-Net models.
The pilot study will also refine the class definitions and spectral band
selection criteria.

---

## 7. Proposed Contribution

DeepWaterLog makes the following contributions:

1. **End-to-end hyperspectral waterlogging pipeline.** A complete, reproducible
   workflow from raw hyperspectral imagery to georeferenced waterlogging maps,
   including preprocessing, band selection, deep learning, evaluation, and
   decision-support visualisation.

2. **Multi-class waterlogging discrimination.** Distinguishes waterlogged soil,
   healthy crops, stressed crops, open water, and non-agricultural land — beyond
   binary water/land masks.

3. **Deep learning on hyperspectral data.** Demonstrates CNN and U-Net
   architectures applied to hyperspectral imagery for agricultural waterlogging
   detection, with ablation of band-selection methods and a baseline comparison
   against NDVI/NDWI.

4. **Agro-sustainability decision support.** Maps designed for drainage
   planning, variable-rate irrigation, crop-damage assessment, and insurance
   claims, advancing precision farming and sustainable water use.

5. **Open, reproducible codebase.** All preprocessing, model, training,
   evaluation, and UI code is version-controlled and documented, with a
   configuration snapshot recorded for every run.

### 7.1 Sustainable Development Goals (SDGs) Targeted

| SDG | Goal | How DeepWaterLog contributes |
|---|---|---|
| **SDG 2** | Zero Hunger | Early waterlogging detection protects crop yield and food supply |
| **SDG 6** | Clean Water and Sanitation | Precision drainage and irrigation reduce water waste and runoff |
| **SDG 12** | Responsible Consumption and Production | Variable-rate inputs reduce fertiliser and pesticide waste |
| **SDG 13** | Climate Action | Builds climate resilience against extreme-rainfall waterlogging |
| **SDG 15** | Life on Land | Protects soil health and supports sustainable land use |

Reference: United Nations, 17 Sustainable Development Goals (https://sdgs.un.org/goals).

---

## 8. Limitations (Tentative)

The following limitations are identified at the proposal stage and may be
revised as the project progresses:

1. **Data availability.** Hyperspectral imagery with matching field labels is
   not always publicly available; the project may rely on drone-acquired data
   from a single study site.

2. **Spatial scale.** Drone or airborne hyperspectral data may not cover large
   regions; scaling to satellite hyperspectral data (e.g., EnMAP, PRISMA)
   requires coarser spatial resolution and may reduce per-field accuracy.

3. **Temporal mismatch.** Hyperspectral acquisitions and field measurements
   may not coincide exactly, introducing label noise.

4. **Class confusion.** Stressed crops from waterlogging may be spectrally
   similar to drought-stressed or nutrient-deficient crops, potentially
   reducing class separability.

5. **Computational cost.** Hyperspectral cubes are large; training CNN/U-Net
   models requires substantial GPU resources and time.

6. **Generalisation.** Models trained on one crop type, region, or season may
   not generalise to others without retraining or transfer learning.

7. **Atmospheric correction.** Residual atmospheric effects may remain after
   correction, affecting spectral consistency across acquisitions.

8. **Decision-support scope.** The web interface is a screening tool, not a
   replacement for agronomic field inspection or engineering drainage design.

9. **Survey limitations.** The pilot sample size (≥15) is small and purposive;
   findings are indicative rather than statistically representative.

---

## 9. Project Timeline (Indicative)

| Phase | Activity | Duration |
|---|---|---|
| 1 | Literature review and proposal finalisation | Month 1 |
| 2 | Data acquisition (pilot study) and hyperspectral preprocessing | Months 2–3 |
| 3 | Band selection, dataset preparation, labelling | Months 3–4 |
| 4 | CNN and U-Net model development and training | Months 4–6 |
| 5 | Evaluation (accuracy, precision, recall, F1, IoU) | Month 6 |
| 6 | Map generation and decision-support UI | Month 7 |
| 7 | Report writing, paper draft, and final submission | Months 7–8 |

---

## 10. References (APA 7th Edition)

Audebert, N., Le Saux, B., & Lefèvre, S. (2019). Deep learning for classification
of hyperspectral data: A comparative review. *IEEE Geoscience and Remote Sensing
Magazine, 7*(2), 159–173. https://doi.org/10.1109/MGRS.2019.2912563

Food and Agriculture Organization of the United Nations. (2021). *The state of
food and agriculture 2021: Making agrifood systems more resilient to shocks and
stresses*. FAO. https://doi.org/10.4060/cb4476en

Gao, B. C. (1996). NDWI—A normalized difference water index for remote sensing
of vegetation liquid water from space. *Remote Sensing of Environment, 58*(3),
257–266. https://doi.org/10.1016/S0034-4257(96)00067-3

Goetz, A. F. H. (2009). Three decades of hyperspectral remote sensing of the
Earth: A personal view. *Remote Sensing of Environment, 113*, S5–S16.
https://doi.org/10.1016/j.rse.2007.12.014

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image
recognition. In *Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition* (pp. 770–778).
https://doi.org/10.1109/CVPR.2016.90

Huete, A., Didan, K., Miura, T., Rodriguez, E. P., Gao, X., & Ferreira, L. G.
(2002). Overview of the radiometric and biophysical performance of the MODIS
vegetation indices. *Remote Sensing of Environment, 83*(1–2), 195–213.
https://doi.org/10.1016/S0034-4257(02)00096-2

Intergovernmental Panel on Climate Change. (2022). *Climate change 2022: Impacts,
adaptation and vulnerability*. Cambridge University Press.
https://doi.org/10.1017/9781009325844

Jackson, R. D., Idso, S. B., Reginato, R. J., & Pinter, P. J. (1981). Canopy
temperature as a crop water stress indicator. *Water Resources Research, 17*(4),
1133–1138. https://doi.org/10.1029/WR017i004p01133

Kamilaris, A., & Prenafeta-Boldú, F. X. (2018). Deep learning in agriculture: A
survey. *Computers and Electronics in Agriculture, 147*, 70–90.
https://doi.org/10.1016/j.compag.2018.02.016

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification
with deep convolutional neural networks. *Advances in Neural Information
Processing Systems, 25*, 1097–1105.

Kuenzer, C., Guo, H., Huth, J., Leinenkugel, P., Li, X., & Dech, S. (2013).
Flood mapping and flood dynamics of the Mekong Delta: ENVISAT-ASAR-WSM based
time series analyses. *Remote Sensing, 5*(2), 687–715.
https://doi.org/10.3390/rs5020687

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature, 521*(7553),
436–444. https://doi.org/10.1038/nature14539

Liakos, K. G., Busato, P., Moshou, D., Pearson, S., & Bochtis, D. (2018). Machine
learning in agriculture: A review. *Sensors, 18*(8), 2674.
https://doi.org/10.3390/s18082674

Ma, L., Liu, Y., Zhang, X., Ye, Y., Yin, G., & Johnson, B. A. (2019). Deep
learning in remote sensing applications: A meta-analysis and review. *ISPRS
Journal of Photogrammetry and Remote Sensing, 152*, 166–177.
https://doi.org/10.1016/j.isprsjprs.2019.04.015

McKinsey Global Institute. (2013). *Disruptive technologies: Advances that will
transform life, business, and the global economy*. McKinsey & Company.

Moran, M. S., Clarke, T. R., Inoue, Y., & Vidal, A. (1994). Estimating crop
water deficit using the relation between surface-air temperature and spectral
vegetation index. *Remote Sensing of Environment, 49*(3), 246–263.
https://doi.org/10.1016/0034-4257(94)90020-5

Peñuelas, J., Filella, I., Biel, C., Serrano, L., & Savé, R. (1993). The
reflectance at the 950–970 nm region as an indicator of plant water status.
*International Journal of Remote Sensing, 14*(10), 1887–1905.
https://doi.org/10.1080/01431169308954010

Reichstein, M., Camps-Valls, G., Stevens, B., Jung, M., Denzler, J., Carvalhais,
N., & Prabhat. (2019). Deep learning and process understanding for data-driven
Earth system science. *Nature, 566*(7743), 195–204.
https://doi.org/10.1038/s41586-019-0912-1

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks
for biomedical image segmentation. In *Medical Image Computing and
Computer-Assisted Intervention* (pp. 234–241). Springer.
https://doi.org/10.1007/978-3-319-24574-4_28

Rouse, J. W., Haas, R. H., Schell, J. A., & Deering, D. W. (1974). Monitoring
vegetation systems in the Great Plains with ERTS. *NASA Special Publication,
351*, 309–317.

Sims, D. A., & Gamon, J. A. (2003). Estimation of vegetation water content and
photosynthetic tissue area from spectral reflectance: A comparison of indices
based on liquid water and chlorophyll absorption features. *Remote Sensing of
Environment, 84*(4), 526–537. https://doi.org/10.1016/S0034-4257(02)00148-7

Sishodia, R. P., Ray, R. L., & Singh, S. K. (2020). Applications of remote
sensing in precision agriculture: A review. *Remote Sensing, 12*(19), 3136.
https://doi.org/10.3390/rs12193136

Thenkabail, P. S., Lyon, J. G., & Huete, A. (Eds.). (2018). *Hyperspectral remote
sensing of vegetation* (2nd ed.). CRC Press.
https://doi.org/10.1201/9781315152271

Tucker, C. J. (1979). Red and photographic infrared linear combinations for
monitoring vegetation. *Remote Sensing of Environment, 8*(2), 127–150.
https://doi.org/10.1016/0034-4257(79)90013-0

van der Meer, F. D., van der Werff, H. M. A., van Ruitenbeek, F. J. A., Hecker,
C. A., Bakker, W. H., Noomen, M. F., van der Meijde, M., Carranza, E. J. M., de
Smeth, J. B., & Woldai, T. (2012). Multi- and hyperspectral geologic remote
sensing: A review. *International Journal of Applied Earth Observation and
Geoinformation, 14*(1), 112–128. https://doi.org/10.1016/j.jag.2011.08.002

Weiss, M., Jacob, F., & Duveiller, G. (2020). Remote sensing for agricultural
applications: A meta-review. *Remote Sensing of Environment, 236*, 111402.
https://doi.org/10.1016/j.rse.2019.111402

World Bank. (2016). *High and dry: Climate change, water, and the economy*.
World Bank. https://doi.org/10.1596/978-1-4648-0365-9

Zhang, L., Zhang, L., & Du, B. (2016). Deep learning for remote sensing data: A
technical tutorial on the state of the art. *IEEE Geoscience and Remote Sensing
Magazine, 4*(2), 22–40. https://doi.org/10.1109/MGRS.2016.2540798

Zhu, X. X., Tuia, D., Mou, L., Xia, G. S., Zhang, L., Xu, F., & Fraundorfer, F.
(2017). Deep learning in remote sensing: A comprehensive review and list of
resources. *IEEE Geoscience and Remote Sensing Magazine, 5*(4), 8–36.
https://doi.org/10.1109/MGRS.2017.2762307
