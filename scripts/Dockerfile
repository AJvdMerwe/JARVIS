# ─────────────────────────────────────────────────────────────────────────────
#  Virtual Personal Assistant – Dockerfile
#
#  Multi-stage build:
#    builder  – installs Python deps
#    runtime  – slim image without build tools
#
#  Build:
#    docker build -t virtual-assistant .
#
#  Run:
#    docker run -it --rm \
#      -v $(pwd)/data:/app/data \
#      -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
#      virtual-assistant
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for audio, PDF, and document processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libsndfile1-dev \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip wheel setuptools

# Install Python dependencies (cached layer if requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Only runtime system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create data directories
RUN mkdir -p data/uploads data/vector_store data/logs/sessions

# Non-root user for security
RUN useradd -m -u 1000 assistant && chown -R assistant:assistant /app
USER assistant

# Environment defaults (override at runtime via -e or .env)
ENV LLM_BACKEND=ollama \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    VOICE_ENABLED=false \
    LOG_LEVEL=INFO \
    VECTOR_STORE_PATH=/app/data/vector_store \
    UPLOADS_PATH=/app/data/uploads \
    LOG_PATH=/app/data/logs/assistant.log

# Data volume for persistence
VOLUME ["/app/data"]

# Expose nothing (this is a CLI app)
# Use docker run -it for interactive mode

ENTRYPOINT ["python", "cli.py"]
CMD ["chat"]
