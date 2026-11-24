# syntax=docker/dockerfile:1
FROM runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_CACHE_DIR=/tmp/uv-cache \
    MASKFLOW_SOURCE=/opt/maskflow MASKFLOW_MODEL_ROOT=/opt/models \
    HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_XET_CACHE=/tmp/hf-xet
WORKDIR /opt/engui-maskflow
COPY pins.json REQUIREMENTS.runtime.txt ./
RUN python3 -m pip install --no-cache-dir uv==0.13.9 && \
    mkdir -p /opt/maskflow && \
    REV=$(python3 -c 'import json; print(json.load(open("pins.json"))["maskflow_code"])') && \
    curl -fsSL --retry 3 "https://codeload.github.com/ReyChiaro/MaskFlow/tar.gz/$REV" | tar xz --strip-components=1 -C /opt/maskflow && \
    cd /opt/maskflow && uv sync --frozen --python 3.12 --no-dev && \
    uv pip install --python /opt/maskflow/.venv/bin/python -r /opt/engui-maskflow/REQUIREMENTS.runtime.txt && \
    rm -rf /tmp/uv-cache /root/.cache/pip
ENV PATH=/opt/maskflow/.venv/bin:$PATH
COPY weights.lock.json bake_models.py ./
RUN python bake_models.py configs && rm -rf /tmp/hf-xet
RUN python bake_models.py encoder-1 && rm -rf /tmp/hf-xet
RUN python bake_models.py encoder-2 && rm -rf /tmp/hf-xet
RUN python bake_models.py encoder-3 && rm -rf /tmp/hf-xet
RUN python bake_models.py encoder-4 && rm -rf /tmp/hf-xet
RUN python bake_models.py transformer-1 && rm -rf /tmp/hf-xet
RUN python bake_models.py transformer-2 && rm -rf /tmp/hf-xet
RUN python bake_models.py transformer-3 && rm -rf /tmp/hf-xet
RUN python bake_models.py transformer-4 && rm -rf /tmp/hf-xet
RUN python bake_models.py transformer-5 && rm -rf /tmp/hf-xet
RUN python bake_models.py vae && rm -rf /tmp/hf-xet
RUN python bake_models.py adapter && rm -rf /tmp/hf-xet
COPY secure.py handler.py engine.py runtime.py ./
COPY tests ./tests
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/huggingface
RUN python bake_models.py verify && python -m unittest discover -s tests -v && \
    python -c "import torch, diffusers, transformers, runpod, boto3, cryptography; import sys; sys.path.insert(0, '/opt/maskflow'); from trainer.lora_utils import load_inference_loras" 
ENTRYPOINT ["/opt/maskflow/.venv/bin/python", "/opt/engui-maskflow/handler.py"]
