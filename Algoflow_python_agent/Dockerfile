# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    git \
    chromium \
    chromium-driver \
    fonts-liberation \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./

RUN mkdir -p src

RUN uv sync

COPY . .

RUN uv run python scripts/download_local_audio_models.py --stt-model turbo --tts-language hi
RUN uv run python scripts/build_native_stt.py

RUN chown -R appuser:appuser /app

USER appuser

RUN uv run agent.py download-files

CMD ["uv", "run", "agent.py", "start"]
