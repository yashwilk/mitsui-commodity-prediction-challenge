# ============================================================
# Dockerfile
# ============================================================
# Build:  docker build -t mitsui .
# Run:    docker run mitsui python train.py --model lgbm
# ============================================================

# ── BASE IMAGE ───────────────────────────────────────────────
# python:3.11-slim is a minimal Python image
# -slim means no extra OS packages — keeps image small (~150MB
# vs ~900MB for the full python:3.11 image)
FROM python:3.11-slim

# ── WORKING DIRECTORY ────────────────────────────────────────
# all subsequent commands run from /app inside the container
# this is where your project lives inside the container
WORKDIR /app

# ── SYSTEM DEPENDENCIES ──────────────────────────────────────
# lightgbm and xgboost need these C libraries to compile
# --no-install-recommends keeps the image lean
RUN apt-get update && apt-get install -y \
    libgomp1 \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# ── PYTHON DEPENDENCIES ──────────────────────────────────────
# copy requirements.txt FIRST before copying the rest of the code
# why: Docker caches each layer — if requirements.txt hasn't changed,
# Docker skips the pip install step on rebuilds (much faster)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ── PROJECT CODE ─────────────────────────────────────────────
# copy source code AFTER installing dependencies
# so a code change doesn't invalidate the pip install cache
COPY src/        ./src/
COPY config.py   .
COPY train.py    .
COPY predict.py  .
COPY evaluate.py .

# ── DIRECTORIES ──────────────────────────────────────────────
# create directories the app writes to at runtime
# data/ models/ logs/ assets/ are mounted as volumes at run time
# so files persist on your machine after the container stops
RUN mkdir -p data models logs assets mlruns

# ── ENVIRONMENT VARIABLES ────────────────────────────────────
# tell MLflow where to store run data inside the container
# this gets overridden by docker-compose.yml in practice
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# tells Python not to buffer stdout/stderr
# so logs appear in real time in the terminal
ENV PYTHONUNBUFFERED=1

# ── DEFAULT COMMAND ──────────────────────────────────────────
# what runs if you do just: docker run mitsui
# can be overridden: docker run mitsui python predict.py
CMD ["python", "train.py", "--model", "lgbm"]