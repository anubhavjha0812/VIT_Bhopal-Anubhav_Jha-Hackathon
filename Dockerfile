# syntax=docker/dockerfile:1

# Stage 1: build the dashboard (web/dist). Only this folder is copied into the final image.
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.hf \
    PATH="/app/.venv/bin:$PATH"

# Dependencies first (CPU-only torch via the index in pyproject.toml), so code edits don't reinstall them.
# uv's download cache lives in a build-cache mount, so it is never saved into the image.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Bake one copy of the model weights in, so the container needs no network at run time
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('ProsusAI/finbert', allow_patterns=['*.json', '*.txt', 'pytorch_model.bin'])"
ENV HF_HUB_OFFLINE=1

COPY . .
COPY --from=web /web/dist web/dist
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# The live feed and the ledger write under data/; a host may run the container as another user.
RUN mkdir -p data/live && chmod -R a+rwX data

# One server for API + dashboard. HF Spaces expects 7860; locally: -e PORT=8000 or compose below.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn riskengine.site:app --host 0.0.0.0 --port ${PORT}"]
