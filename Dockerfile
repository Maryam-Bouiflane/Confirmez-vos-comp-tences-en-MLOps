# Dockerfile for Credit Scoring API
# Using uv for dependency management
# Multi-stage build for smaller final image

# Stage 1: Build stage with uv
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir --user uv

# Copy project files
COPY pyproject.toml uv.lock .

# Sync dependencies with uv
RUN uv sync --frozen

# Stage 2: Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Create non-root user for security
RUN groupadd -r apiuser && useradd -r -g apiuser apiuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/apiuser/.local

# Copy project files
COPY pyproject.toml uv.lock .
COPY api/ ./api/
COPY api/models/ ./api/models/

# Make sure scripts in .local are usable
ENV PATH=/home/apiuser/.local/bin:$PATH

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

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD uv run python -c "import requests; requests.get('http://localhost:8000/health', timeout=5).raise_for_status()" || exit 1

# Run the application
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
