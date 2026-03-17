FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .

# ── Training ─────────────────────────────────────────────────
FROM base AS train
CMD ["python", "-m", "src.training.trainer"]

# ── API Server ───────────────────────────────────────────────
FROM base AS serve
EXPOSE 8000
CMD ["python", "-m", "src.serving.api"]

# ── Dashboard ────────────────────────────────────────────────
FROM base AS dashboard
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
