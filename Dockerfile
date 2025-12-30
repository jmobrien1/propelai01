# PropelAI Production Dockerfile
# Multi-stage build for smaller, secure images

# =============================================================================
# Stage 1: Build dependencies
# =============================================================================
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt


# =============================================================================
# Stage 2: Production image
# =============================================================================
FROM python:3.11-slim as production

# Labels
LABEL maintainer="PropelAI Team" \
    version="4.1.0" \
    description="PropelAI - Autonomous Proposal Operating System"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROPELAI_ENV=production \
    PORT=8000 \
    # Application paths
    APP_HOME=/app \
    DATA_DIR=/data

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 propelai && \
    useradd --uid 1000 --gid propelai --shell /bin/bash --create-home propelai

# Create application directories
RUN mkdir -p ${APP_HOME} ${DATA_DIR}/uploads ${DATA_DIR}/outputs ${DATA_DIR}/chroma && \
    chown -R propelai:propelai ${APP_HOME} ${DATA_DIR}

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR ${APP_HOME}

# Copy application code
COPY --chown=propelai:propelai . .

# Switch to non-root user
USER propelai

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health/live || exit 1

# Default command
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
