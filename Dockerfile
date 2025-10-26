# Multi-stage Dockerfile for the gold news sentiment analysis system

# Stage 1: Base image with Python and system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Copy application code
COPY . .

# Stage 2: Production image for API service
FROM base as api

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start command
CMD ["python", "main.py"]

# Stage 3: Production image for dashboard service
FROM base as dashboard

# Expose port
EXPOSE 8501

# Install additional dependencies for dashboard
RUN pip install --user streamlit plotly altair

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start command
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Stage 4: Worker image for Celery tasks
FROM base as worker

# Install additional dependencies for workers
USER root
RUN apt-get update && apt-get install -y \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*
USER app

# Start command for worker
CMD ["celery", "-A", "app.tasks.celery_config", "worker", "--loglevel=info"]

# Stage 5: Beat scheduler image
FROM worker as beat

# Start command for beat scheduler
CMD ["celery", "-A", "app.tasks.celery_config", "beat", "--loglevel=info", "--scheduler=django_celery_beat.schedulers:DatabaseScheduler"]
