FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.3
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root --only main

# Copy application
COPY app/ ./app/
COPY gunicorn.conf.py ./

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with gunicorn (multi-worker)
CMD ["gunicorn", "app.main:app", "-c", "gunicorn.conf.py"]
