FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (layer cache)
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies (no dev extras)
RUN uv pip install --system --no-cache .

# Create models directory
RUN mkdir -p /data/models

COPY alembic.ini ./
COPY alembic/ ./alembic/

CMD ["python", "-m", "forecaster.main"]
