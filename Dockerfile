# DeepWaterLog dashboard — production image
#
# Serves the results dashboard only (Flask + NumPy + Pillow).
# Model training does not happen at runtime; use Dockerfile.research for that.
#
# Build:
#   docker build -t deepwaterlog-web .
#
# Run:
#   docker run --rm -p 7860:7860 deepwaterlog-web
#   -> http://127.0.0.1:7860

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

COPY requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt

# Dashboard code (no torch, no training run at runtime).
# core/__init__.py imports the numeric modules eagerly, so the whole package is
# copied; the torch-dependent models are wrapped in a try/except there.
COPY config.py ./
COPY wsgi.py ./
COPY core ./core
COPY ui ./ui
COPY data/results ./data/results

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/healthz', timeout=3)"

CMD ["python", "wsgi.py"]
