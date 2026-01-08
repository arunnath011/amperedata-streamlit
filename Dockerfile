# Multi-stage Dockerfile for AmpereData Platform
# Optimized for production deployment

# Stage 1: Base Python image with common dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stage 2: Builder - Install Python dependencies
FROM base as builder

# Copy requirements
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 3: Production image
FROM base as production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r amperedata && \
    useradd -r -g amperedata -u 1000 amperedata && \
    mkdir -p /app /app/logs /app/data && \
    chown -R amperedata:amperedata /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=amperedata:amperedata . .

# Switch to non-root user
USER amperedata

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - FastAPI
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Stage 4: Streamlit image (alternative)
FROM production as streamlit

EXPOSE 8501

# Override command for Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

# Stage 5: Celery worker (alternative)
FROM production as celery-worker

# Override command for Celery
CMD ["celery", "-A", "backend.celery_app", "worker", "--loglevel=info", "--concurrency=4"]

# Stage 6: Development image with additional tools
FROM production as development

USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    flake8 \
    mypy \
    ipython

USER amperedata

# Development command with auto-reload
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
