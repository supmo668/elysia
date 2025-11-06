# Use Python 3.12 slim image for smaller size
FROM python:3.12-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY MANIFEST.in ./
COPY README.md ./

# Copy source code needed for editable install
COPY elysia ./elysia

# Install dependencies using uv
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install -e .

# Download spaCy model
RUN . /app/.venv/bin/activate && \
    python -m spacy download en_core_web_sm

# Production stage
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV=/app/.venv

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 elysia && \
    mkdir -p /app /app/logs /app/data && \
    chown -R elysia:elysia /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=elysia:elysia /app/.venv /app/.venv

# Copy application code
COPY --chown=elysia:elysia . .

# Switch to non-root user
USER elysia

# Expose the application port
EXPOSE 8005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8005/api/health || exit 1

# Run the application
CMD ["sh", "-c", "uvicorn elysia.api.app:app --host 0.0.0.0 --port ${PORT:-8005}"]
