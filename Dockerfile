# IoMT IDS - Dockerfile

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies (Java for PySpark, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    build-essential \
    curl \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . .

# Default port for API if used
EXPOSE 8000

# Default command prints help; override with `docker run ... <command>`
CMD ["bash", "-lc", "echo 'Container ready. Examples:' && \
  echo '  python scripts/test_pipeline_spark.py' && \
  echo '  python scripts/test_pipeline_compare.py' && \
  echo '  python scripts/train.py' && \
  echo '  python service/api/main.py' && \
  sleep infinity"]
