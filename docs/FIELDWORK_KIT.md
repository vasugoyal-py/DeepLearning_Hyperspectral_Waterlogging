# Fieldwork Kit — Pilot Survey in One Day

**Project:** DeepWaterLog — Deep Learning–Enabled Hyperspectral Remote Sensing for
Agricultural Waterlogging Detection
**Purpose:** collect the primary survey (n ≥ 15) and the four geotagged photographs
required by the IDEATHON 2026 template, without fabricating anything.
**Time budget:** one working day (≈ 4 hours of actual effort).

---

## 0. Ground rules

1. **No invented responses.** Every row in `pilot_responses.csv` must trace to a
   real form submission. A judge can ask to see the Google Form response sheet.
2. **No stock photos.** Geotags are checkable in the file properties.
3. **Consent first.** Ask the consent question before recording anything. If the
   answer is no, stop and thank them.
4. **One row per respondent.** No duplicate entries from the same farm.

---

## 1. Build the Google Form (20 minutes)

Go to <https://forms.new> and create a form titled
**"Waterlogging in Your Field — 2-Minute Survey"**.

Set *Settings → Responses → Collect email addresses: OFF* and
*Limit to 1 response: OFF* (you will be sharing by WhatsApp, often on one phone).

### Section A — Basic profile (5 questions)

| # | Question | Type | Options |
|---|---|---|---|
| A1 | Age | Short answer (number) | — |
| A2 | Highest education completed | Multiple choice | None / Primary / Secondary / Higher secondary / Graduate or above |
| A3 | Farm size (acres) | Short answer (number) | — |
| A4 | Main crop grown in kharif season | Multiple choice | Paddy / Wheat / Cotton / Sugarcane / Mixed vegetables / Other |
| A5 | Main irrigation source | Multiple choice | Canal / Tube well / Both / Rain-fed only |
| A6 | Years of farming experience | Short answer (number) | — |

### Section B — Waterlogging experience (10 questions)

Copy these verbatim from Section 7.4 of the proposal. Keep the wording identical
so the instrument matches the document.

| # | Question | Type | Options |
|---|---|---|---|
| B1 | What is the primary crop grown on your field during the monsoon/kharif season? | Multiple choice | as A4 |
| B2 | How many times in the last five years has your field experienced waterlogging? | Short answer (number) | — |
| B3 | Which part of the field is most frequently waterlogged? | Multiple choice | Low-lying area / Near canal / Centre / Edges |
| B4 | On average, how many days does water remain standing or soil remain saturated after heavy rain? | Multiple choice | 1–3 / 3–5 / 5–7 / 7–10 / More than 10 |
| B5 | What approximate crop-yield loss (%) do you attribute to waterlogging in a bad year? | Short answer (number) | — |
| B6 | Do you currently use any method to detect or monitor waterlogging before visible crop damage appears? | Multiple choice | No / Yes — visual scouting only / Yes — app or sensor |
| B7 | Have you ever used a mobile app, drone, or satellite-based service for crop or field monitoring? | Multiple choice | Yes / No |
| B8 | Would a field-scale waterlogging map delivered to your phone help you plan drainage or irrigation? | Multiple choice | Yes / Maybe / No |
| B9 | What is the main obstacle to improving drainage on your field? | Multiple choice | Cost / Labour / Land tenure / Lack of information / Other |
| B10 | Would you be willing to share field photographs and coordinates for a research study on waterlogging detection? | Multiple choice | Yes / No |
| B11 | Village / block name | Short answer | — |

### Consent statement (put above B1)

> *"This is a student research project at GD Goenka University on detecting
> waterlogged fields from satellite imagery. The survey takes about two minutes.
> Your name is not recorded, and answers are used only in aggregate. You may skip
> any question and stop at any time. Do you agree to take part?"*
>
> Options: **Yes, I agree** / **No**

Add a *Required* flag to the consent question and to B2, B6, B8, B9.

### Export

*Responses → Link to Sheets* → open the sheet → **File → Download → CSV** →
save as `data/pilot/pilot_responses.csv`.

---

## 2. Recruit 15+ respondents (90 minutes)

Send this to your village WhatsApp group, family group, and any farming contacts:

> Namaste 🙏
> We are B.Tech students from GD Goenka University working on a project to detect
> waterlogged fields early using satellite images, so farmers can act before the
> crop is damaged.
>
> If you farm, or know someone who does, please fill this 2-minute form. It asks
> about waterlogging on your field. No name, no phone number, 11 questions.
>
> 👉 [paste form link]
>
> Please forward to any farmer you know in Haryana, Punjab or western UP. We need
> 15 responses today. Thank you!

**Call script for the 2-minute phone version** (for relatives who will not open a
link):

> "Can I ask you eleven quick questions about your field for a college project?
> It takes two minutes and I will not write your name. … [read A1–A5, B1–B9] …
> Thank you — and would you be willing to be photographed in your field for the
> report?"

Record phone answers in the CSV yourself, and put `phone` in the `source`
column so the provenance is clear.

**Who counts:** farmers, farm labourers who work the land, rural relatives who
farm, agricultural input dealers who visit fields. Do **not** count classmates or
urban relatives who have never farmed. If someone is unsure, they are not a
respondent.

---

## 3. Take the four geotagged photos (60 minutes)

Install **GPS Map Camera** (free, Android/iOS). Allow location access. Wait for
the accuracy reading to settle below ~20 m.

| # | Shot | Notes |
|---|---|---|
| 1 | Severely waterlogged plot | Standing water or clearly saturated soil; include the crop in frame |
| 2 | Mildly waterlogged field | Yellowing / chlorosis visible against a healthy patch |
| 3 | Healthy crop control | Same crop, well-drained field |
| 4 | Farmer interview or soil sampling | Respondent in frame, plot visible behind |

For each shot: one wide frame, one close-up. Save the originals. Rename as
`WL_PHOTO_<n>_<village>_<YYYYMMDD>.jpg` and store in `data/pilot/photos/`.

**Do not** send through WhatsApp (it strips EXIF). Transfer by cable, Google
Drive "original quality", or USB.

### Record the coordinates

Paste them into `data/pilot/photo_log.csv`:

```csv
photo_id,subject,village,latitude,longitude,date,time,app,filename
1,Severe waterlogging,,,,,GPS Map Camera,WL_PHOTO_1_village_YYYYMMDD.jpg
```

---

## 4. Compute the Slide 9 numbers (5 minutes)

Once the CSV is filled, run:

```bash
python scripts/pilot_analysis.py --csv data/pilot/pilot_responses.csv
```

This writes:

- `data/pilot/pilot_summary.md` — the statistics table, ready to paste
- `data/pilot/pilot_slide9.png` — the bar chart for the slide
- console output with the four headline percentages

The script refuses to run if fewer than 15 valid responses are present, so an
under-filled sheet cannot accidentally become a slide.

### Slide 9 fill-in template

```text
PILOT SURVEY — n = ____ farmers, ____ villages, ____ district

Basic profile      Age: mean __ | Farm size: mean __ acres
                   Crop: paddy __%, wheat __%, other __%
                   Irrigation: canal __%, tube well __%, both __%
                   Farming experience: mean __ years

Headline findings
  __% face recurring waterlogging (≥ 3 events in 5 years)
  __% average self-reported yield loss in a bad year
  __% use no pre-symptom detection tool
  __% want mobile waterlogging maps

[Insert pilot_slide9.png here]
```

Fill every blank from `pilot_summary.md`. Never estimate.

---

## 5. Update the proposal

| Where | What to change |
|---|---|
| `IDEATHON_2026_PROPOSAL.md` §7.5 | Replace the heading and declaration box with "Pilot Study Results (n = ___)"; paste the real table from `pilot_summary.md` |
| `IDEATHON_2026_PROPOSAL.md` §7.5.3 | Delete the illustrative EX rows entirely |
| `IDEATHON_2026_PROPOSAL.md` §7.7 | Fill the coordinate table, insert the four photos |
| `IDEATHON_2026_PROPOSAL.md` §5.6 | Change "Not yet collected" to "Collected (n = ___)" |
| `IDEATHON_2026_PROPOSAL.md` §9 item 9 | Change "has not yet been administered" to the collected sample description |
| `docs/WHAT_AND_WHY.md` §7 | Update the "no field validation yet" bullet |

---

## 6. Ethics and data handling

- No names, phone numbers or GPS coordinates of individual respondents in the
  submission or the public repository.
- Village-level location only in the report; plot coordinates only in the
  internal ground-truth file.
- Delete audio recordings and chat threads once answers are transcribed.
- If a respondent withdraws, delete their row before analysis.

---

## 7. What to say if a judge asks

> "We surveyed ___ farmers across ___ villages in ___ district over two days,
> using a structured 11-item instrument. The responses are in a Google Sheet with
> timestamps, and the four field photographs carry GPS EXIF from GPS Map Camera.
> Our sample is purposive, not representative — we say so in the limitations."

If the survey is still pending at presentation time, say:

> "The survey instrument and fieldwork kit are complete and the form is live; we
> are collecting responses now and will report n and the computed statistics
> rather than estimating them in advance."
