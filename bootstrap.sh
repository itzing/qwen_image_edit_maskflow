#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "$(dirname "$0")"
APP="$PWD"
CACHE_ROOT="${MASKFLOW_CACHE_ROOT:-/runpod-volume/maskflow-cache}"
mkdir -p "$CACHE_ROOT"
export UV_CACHE_DIR="$CACHE_ROOT/uv-cache"
export UV_PYTHON_INSTALL_DIR="$CACHE_ROOT/python"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
UPSTREAM_REV=$(python3 -c 'import json; print(json.load(open("pins.json"))["maskflow_code"])')
RUNTIME_HASH=$(sha256sum "$APP/REQUIREMENTS.runtime.txt" | cut -c1-16)
export MASKFLOW_SOURCE="$CACHE_ROOT/source-$UPSTREAM_REV-$RUNTIME_HASH"
# Serialize initialization across the two workers sharing a model cache.
exec 9>"$CACHE_ROOT/bootstrap.lock"
flock 9
python3 -m pip install --disable-pip-version-check --quiet uv==0.13.9
if [ ! -d "$MASKFLOW_SOURCE" ]; then
  SOURCE_STAGE=$(mktemp -d "$CACHE_ROOT/source-stage.XXXXXX")
  curl --fail --location --retry 3 "https://codeload.github.com/ReyChiaro/MaskFlow/tar.gz/$UPSTREAM_REV" | tar xz --strip-components=1 -C "$SOURCE_STAGE"
  mv "$SOURCE_STAGE" "$MASKFLOW_SOURCE"
fi
cd "$MASKFLOW_SOURCE"
if [ ! -f .runtime-ready ]; then
  uv sync --frozen --python 3.12 --no-dev
  uv pip install --python "$MASKFLOW_SOURCE/.venv/bin/python" -r "$APP/REQUIREMENTS.runtime.txt"
  touch .runtime-ready
fi
flock -u 9
cd "$APP"
exec "$MASKFLOW_SOURCE/.venv/bin/python" handler.py
