# syntax=docker/dockerfile:1
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
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

EXPOSE 8000
CMD ["python", "run_demo.py"]
