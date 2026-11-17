# Hosting the dashboard for free

The dashboard is a read-only Flask app that loads saved run records and images.
It does **not** need PyTorch, so it deploys in about two minutes and fits
comfortably inside every free tier.

Two routes are supported. Option A needs no credit card and no GitHub repository.

---

## Option A — Hugging Face Spaces (recommended, no card)

**Result:** a permanent public URL like
`https://<your-username>-deepwaterlog.hf.space`

### Steps

1. Create a free account at <https://huggingface.co/join>.
2. Go to <https://huggingface.co/new-space>.
3. Fill in:
   - **Space name:** `deepwaterlog`
   - **License:** MIT
   - **SDK:** **Docker** → **Blank**
   - **Hardware:** CPU basic (free)
   - **Visibility:** Public
4. Click **Create Space**.
5. On the new Space page choose **Files → Add file → Upload files** and upload
   the contents of this project folder. The files that matter are:

   ```
   Dockerfile
   requirements-web.txt
   wsgi.py
   config.py
   core/__init__.py
   core/mapper.py
   ui/app.py
   data/results/        (the demo run JSON + PNGs)
   README.md            (contains the required YAML header)
   ```

   `.dockerignore` and `.gitattributes` are optional. Do **not** upload `old/`,
   `tests/`, `scripts/`, or the large proposal documents.

   > Easier alternative: push this folder to GitHub, then in the Space choose
   > **Settings → Repository → Connect to GitHub**. Every push redeploys.

6. The Space builds automatically. Watch the **Logs** tab until it says
   `Running on http://0.0.0.0:7860`.
7. Your link is the Space URL. Put it in the PPT.

### Verify before you paste it into the deck

```bash
curl https://<your-space>.hf.space/healthz
# {"service":"deepwaterlog-dashboard","status":"ok"}
```

Then open `/` and confirm the KPI cards and both map images render.

### Free-tier behaviour to know

- A Space sleeps after roughly 48 hours without traffic. The first visit after
  that takes 20–60 seconds to wake. **Open the link five minutes before you
  present** so the judges do not hit a cold start.
- The filesystem is ephemeral: a report exported from the UI disappears on
  restart. That is fine; it is a demo, not a database.

---

## Option B — Render (free web service, needs GitHub)

1. Push the project to a GitHub repository.
2. <https://render.com> → **New → Web Service** → connect the repository.
3. Settings:
   - **Environment:** Docker
   - **Dockerfile path:** `./Dockerfile`
   - **Instance type:** Free
4. Deploy. Render sets `PORT` automatically; `wsgi.py` reads it.

Free instances also spin down when idle.

---

## Option C — local only (no hosting)

```bash
python wsgi.py          # production-style server on http://127.0.0.1:7860
python ui/app.py        # development server on http://127.0.0.1:5000
```

---

## Option D — GitHub Pages (static snapshot only)

GitHub Pages serves **static files**, so it cannot run the Flask app. It *can*
host a frozen snapshot, which is enough for a PPT link.

```bash
python scripts/export_static.py          # writes site/index.html + site/assets/
python -m http.server 8000 --directory site   # preview at http://127.0.0.1:8000
```

To publish:

```bash
git init
git add .
git commit -m "DeepWaterLog dashboard snapshot"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo>.git
git push -u origin main
```

Then in the repository: **Settings → Pages → Build and deployment → Source =
GitHub Actions**. The workflow in `.github/workflows/pages.yml` rebuilds `site/`
on every push to `main` and publishes it to
`https://<your-username>.github.io/<repo>/`.

| | GitHub Pages | Hugging Face Spaces |
|---|---|---|
| Runs the Flask API | No | Yes |
| Report export button | No | Yes |
| Setup | GitHub account + repo | HF account + upload |
| Best for | PPT link, archival snapshot | Live demo |

Use both if you like: Pages for the link in the deck, Spaces for the live demo.

---

## What to put on the slide

> **Live demo:** `<space-url>`
> Dashboard showing the latest pipeline run: U-Net metrics, classified
> waterlogging extent, severity map and per-class area breakdown.

Add a QR code pointing at the URL — free generators exist, and judges can scan
it from the projected slide.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Build fails at `pip install` | `requirements-web.txt` missing | Upload it next to the Dockerfile |
| Page loads but shows "No pipeline run recorded" | `data/results/*.json` not uploaded | Upload the demo run JSON and PNGs |
| Images 404 but KPIs load | PNGs missing from `data/results/` | Upload `<run_id>_map.png` and `<run_id>_severity.png` |
| Space stuck at "Building" | Docker SDK not selected | Recreate the Space with SDK = Docker |
| 502 after deploy | App bound to 127.0.0.1 | `wsgi.py` must bind `0.0.0.0` (it does by default) |
