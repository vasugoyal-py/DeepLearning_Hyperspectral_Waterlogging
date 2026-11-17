# SYNERGY 2026 PROJECT PROPOSAL

### 9th Edition Interdisciplinary Techno-Fest, SGT University Gurugram

---

## 1. GENERAL INFORMATION

**Project Title:** Deep Learning–Enabled Hyperspectral Remote Sensing for Agricultural Waterlogging Detection

**Theme:** Smart AgriTech — Precision Farming and Agro Sustainability

**Project Type:** Deep Learning–Based Hyperspectral Remote Sensing Decision Support System

**Primary Track:** Climate Tech & Sustainability / Smart Agriculture

**Secondary Track:** Intelligent Systems / Emerging Technologies

**University/College/Department:** [Insert your university/college/department name]

---

## 2. TEAM INFORMATION

- **Team Leader:** Vasu Goyal | B.TECH CSE AI/ML | vasugoyal1984@gmail.com
- **Member 2:** Sarthak | B.TECH CSE AI/ML | sarthaksinghxiia1410@gmail.com
- **Member 3:** Tanishka Gupta | B.TECH CSE AI/ML | tanishkag265@gmail.com
- **Faculty Mentor/Guide:** Dr. Manu Banga | Assistant Professor | manu.banga@gdgu.org
- **Faculty Mentor/Guide:** Dr. Sudipta K. Mishra | Professor | sudipta.mishra@gdgu.org

---

## 3. ABSTRACT

Agricultural waterlogging — the temporary or prolonged saturation of crop root zones by excess water — is a major cause of yield loss worldwide. Optical hyperspectral remote sensing provides rich spectral signatures that can distinguish waterlogged soil, healthy vegetation, stressed crops, and open water, yet most operational systems still rely on binary flood masks or SAR backscatter without exploiting the full hyperspectral spectrum.

This project develops **DeepWaterLog**, a deep learning–enabled system for detecting, classifying, and mapping waterlogged areas using hyperspectral remote-sensing imagery. The pipeline applies hyperspectral preprocessing (noise removal, atmospheric correction, band selection, and dimensionality reduction) before training Convolutional Neural Networks (CNNs) and U-Net segmentation models to identify waterlogged regions and generate accurate spatial maps. Model performance is evaluated using accuracy, precision, recall, F1-score, and Intersection over Union (IoU).

The proposed system supports early waterlogging detection, crop-damage assessment, drainage planning, and sustainable agricultural management — directly advancing Smart AgriTech, precision farming, and agro sustainability.

---

## 4. PROBLEM STATEMENT

- **Yield loss from waterlogging:** Excess soil moisture deprives crop roots of oxygen, reduces nutrient uptake, and promotes root diseases, causing significant production losses in rice, wheat, maize, and vegetable crops.
- **Late detection:** Conventional field scouting and farmer reports detect waterlogging only after visible crop stress appears, by which time yield damage is already incurred.
- **Insufficient spectral discrimination:** Standard multispectral indices (e.g., NDVI, NDWI) often cannot separate waterlogged soil from healthy crops or distinguish stress caused by waterlogging from drought or nutrient deficiency.
- **Lack of deep learning hyperspectral systems:** Few operational systems combine hyperspectral imagery with deep semantic segmentation (U-Net) and classification (CNN) models to map waterlogging at field scale with per-pixel accuracy.
- **No decision-support output:** Farmers and drainage planners lack fine-grained spatial waterlogging maps that can guide targeted drainage, irrigation scheduling, and crop insurance claims.

---

## 5. SYSTEM DESIGN

DeepWaterLog implements a six-stage pipeline:

> [Hyperspectral Imagery]  
> [Stage 1: Preprocessing]  
> [Stage 2: Band Selection & Dimensionality Reduction]  
> [Stage 3: Training Data Preparation]  
> [Stage 4: Deep Learning Modelling — CNN & U-Net]  
> [Stage 5: Evaluation & Waterlogging Mapping]  
> [Stage 6: Decision Support UI]

**[Stage 1: Hyperspectral Preprocessing]** – Raw hyperspectral cubes undergo bad-band removal, destriping, noise reduction (e.g., Savitzky–Golay or wavelet filtering), atmospheric correction (e.g., FLAASH or QUAC), and radiometric calibration to produce surface-reflectance cubes.

**[Stage 2: Band Selection & Dimensionality Reduction]** – Redundant and highly correlated bands are removed using correlation analysis, mutual information, or principal component analysis (PCA). A reduced band set is selected to retain waterlogging-sensitive spectral features (visible–near-infrared and shortwave-infrared regions related to chlorophyll, water content, and soil moisture).

**[Stage 3: Training Data Preparation]** – Preprocessed hyperspectral patches are paired with field-validated waterlogging labels (waterlogged soil, healthy crop, stressed crop, open water, non-agricultural). Patches are normalized and split into training, validation, and test sets.

**[Stage 4: Deep Learning Modelling]** –
- **CNN:** Trained for patch-level classification (waterlogged vs. non-waterlogged, and stress severity classes).
- **U-Net:** Trained for pixel-level semantic segmentation to produce waterlogging extent and severity maps.

**[Stage 5: Evaluation & Mapping]** – Models are evaluated using accuracy, precision, recall, F1-score, and IoU. U-Net outputs are georeferenced to produce waterlogging maps at the scale of the input imagery.

**[Stage 6: Decision Support UI]** – A web interface (Flask + Leaflet) allows agronomists and drainage planners to visualize waterlogging maps, overlay field boundaries, and export reports for drainage planning and crop-damage assessment.

---

## 6. RELEVANCE TO SYNERGY 2026 THEME — Smart AgriTech Precision Farming and Agro Sustainability

- **Precision Farming:** Provides field-scale, crop-specific waterlogging maps that enable variable-rate drainage, targeted irrigation, and precision input application — reducing water waste and chemical runoff.
- **Agro Sustainability:** Early waterlogging detection prevents yield loss, protects soil health, and supports climate-resilient agriculture aligned with UN Sustainable Development Goal 2 (Zero Hunger) and SDG 13 (Climate Action).
- **Societal Impact:** Supports crop-damage assessment for insurance claims, drainage infrastructure planning, and extension advisories for smallholder farmers.

---

## 7. NOVELTY

- **Hyperspectral + Deep Learning fusion:** Exploits hundreds of narrow spectral bands with CNN and U-Net architectures to detect waterlogging before visible stress symptoms appear.
- **Multi-class discrimination:** Distinguishes waterlogged soil, healthy vegetation, stressed crops, and open water — beyond simple flood/water binary masks.
- **End-to-end pipeline:** Preprocessing → band selection → deep learning → evaluation → mapping → decision support, all in one reproducible system.
- **Agro-focused outputs:** Maps are designed for drainage planning and crop-damage assessment, not just inundation extent.

---

## 8. DEMO PLAN

During the 3-day event the team will demonstrate the web application using real hyperspectral examples from waterlogged agricultural regions:

- Punjab/Haryana paddy fields (monsoon waterlogging, India)
- Indo-Gangetic plain wheat fields (post-irrigation waterlogging)
- Vegetable crops in peri-urban farms (drainage failure scenarios)

**Requirements:**
- Table & 4 chairs
- Power socket (230 V AC) for laptop and external monitor
- Wi-Fi/Internet connection to stream map tiles
- Space to set up project (1 laptop, 1 external monitor, Banner/Poster)

---

## 9. REFERENCES (APA 7th Edition)

(See full APA-reference list in `docs/RESEARCH_METHODOLOGY.md`, Section 10.)
