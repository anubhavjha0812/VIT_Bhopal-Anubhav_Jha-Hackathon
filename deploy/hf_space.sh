#!/usr/bin/env bash
# Publish the app as a Hugging Face Space (Gradio SDK wrapper; Docker Spaces need a paid plan), without touching this repo's git history.
#   deploy/hf_space.sh <hf-user>/<space-name> [--public]
# Needs `hf auth login` (or ~/.cache/huggingface/token). Builds a clean copy first, so the Space
# only gets what the image needs: no raw data, kaggle model dumps, private notes or git history.
set -euo pipefail
SPACE="${1:?usage: deploy/hf_space.sh <hf-user>/<space-name> [--public]}"
VISIBILITY="--private"; [ "${2:-}" = "--public" ] && VISIBILITY=""

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# The same exclusions as .dockerignore, plus dev-only folders.
rsync -a --exclude-from=<(sed '/^#/d;/^$/d' "$ROOT/.dockerignore") \
  --exclude='.agents' --exclude='.aws' --exclude='.claude' --exclude='.codex' --exclude='.github' \
  --exclude='.uv-cache' --exclude='tests' --exclude='kaggle' --exclude='notebooks' \
  --exclude=':memory:.ses' --exclude='data/cache' --exclude='data/live' \
  --exclude='final-comp*.md' --exclude='INSTRUCTIONS.MD' --exclude='skills-lock.json' \
  "$ROOT/" "$STAGE/"

# The dashboard is built here, so the Space needs no Node.
(cd "$ROOT/web" && npm ci --silent && npm run build --silent)
rsync -a --delete "$ROOT/web/dist/" "$STAGE/web/dist/"

# Free Spaces cannot run Docker: declare a Gradio app whose app.py starts our own server.
cp "$ROOT/deploy/hf_app.py" "$STAGE/app.py"
rm -f "$STAGE/Dockerfile" "$STAGE/docker-compose.yml" "$STAGE/requirements.txt" "$STAGE/uv.lock"
cat > "$STAGE/requirements.txt" <<'REQ'
--extra-index-url https://download.pytorch.org/whl/cpu
fastapi>=0.142.2
httpx>=0.28.1
joblib>=1.6.0
onnx>=1.23.2
onnxruntime>=1.30.0
pandas>=3.0.6
scikit-learn>=1.9.1
scipy>=1.18.1
tabulate>=0.10.0
torch>=2.14.1
transformers>=5.18.0
uvicorn>=0.54.0
yfinance>=1.7.0
REQ

# Spaces read their settings from the README's front matter.
cat > "$STAGE/README.md" <<'HEADER'
---
title: AI Risk Engine
emoji: 📉
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
---

News and social sentiment turned into portfolio and credit risk. Dashboard at `/`, API under `/api`.
HEADER

# DRY_RUN=1 keeps the staged copy and stops here, to test it locally first.
if [ -n "${DRY_RUN:-}" ]; then trap - EXIT; echo "staged: $STAGE"; exit 0; fi

hf repos create "$SPACE" --repo-type space --space-sdk gradio $VISIBILITY --exist-ok
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy"
echo "https://huggingface.co/spaces/$SPACE"
