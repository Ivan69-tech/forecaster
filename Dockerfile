FROM python:3.11-slim

# LightGBM requiert libgomp (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (layer cache)
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies (no dev extras)
RUN uv pip install --system --no-cache .

# Données synthétiques pour le mode démo
COPY tests/fixtures/load_history_2025.csv ./data/load_history_2025.csv

# Scripts d'initialisation
COPY scripts/ ./scripts/

# Create models directory
RUN mkdir -p /data/models

COPY alembic.ini ./
COPY alembic/ ./alembic/

# Utilisateur non-root
RUN useradd -u 1000 -m forecaster \
    && chown -R forecaster:forecaster /app /data
USER 1000:1000

CMD ["python", "-m", "forecaster.main"]
