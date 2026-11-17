# DeepWaterLog - Deep Learning–Enabled Hyperspectral Remote Sensing for
# Agricultural Waterlogging Detection
#
# Build:
#   docker build -t deepwaterlog:latest .
#
# Run the offline health check (no Earth Engine credentials needed):
#   docker run --rm deepwaterlog:latest python test_pipeline.py
#
# Start the decision-support UI:
#   docker run --rm -p 5000:5000 deepwaterlog:latest python ui/app.py

FROM python:3.12-slim

WORKDIR /app

# System dependencies (GDAL, GEOS for raster/vector GeoTIFF output)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && \
    uv sync --frozen --no-dev --no-group paper --no-group docs && \
    pip install numpy torch torchvision flask scipy rasterio pytest ruff

# Application code
COPY config.py ./
COPY core ./core
COPY data ./data
COPY scripts ./scripts
COPY test_pipeline.py ./
COPY tests ./tests
COPY ui ./ui

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command: run the offline test suite (no credentials needed)
CMD ["python", "-m", "pytest", "-q"]
