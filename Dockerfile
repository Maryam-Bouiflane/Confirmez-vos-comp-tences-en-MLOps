# Dockerfile for Credit Scoring API
# Using uv for dependency management
# Multi-stage build for smaller final image

# Stage 1: Build stage with uv
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv system-wide so it is on PATH for the next RUN steps
# (--user would install it in /root/.local/bin, which is not on PATH)
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock .

# Sync dependencies into /app/.venv
RUN uv sync --frozen

# Stage 2: Runtime stage
FROM python:3.12-slim

WORKDIR /app

# System library required by LightGBM (OpenMP runtime), missing from slim images
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r apiuser && useradd -r -g apiuser apiuser

# Copy the virtualenv with all installed dependencies
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY api/ ./api/

# Writable directory for the MLflow tracking sqlite
# (default tracking URI: sqlite:///tracking/mlflow.db, relative to /app)
RUN mkdir -p /app/tracking && chown apiuser:apiuser /app/tracking

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/api/models/credit_scoring_model.pkl

# Switch to non-root user
USER apiuser

# Expose port
EXPOSE 8000

# Health check (uses the venv python directly: no uv at runtime)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5).raise_for_status()" || exit 1

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
