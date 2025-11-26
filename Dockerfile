# PropelAI Autonomous Proposal Operating System
# Production Dockerfile (Railway compatible)

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROPELAI_ENV=production \
    PORT=8000

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Create data directories
RUN mkdir -p /app/data/uploads /app/data/outputs /app/data/chroma

# Expose port (Railway overrides this)
EXPOSE 8000

# Run the application - use shell form to expand $PORT
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
