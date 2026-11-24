"""Resolve baked model assets without allowing network fallback."""
import os
from pathlib import Path
from bake_models import verify_all,MODEL_ROOT

def baked_paths():
    verify_all()
    base=MODEL_ROOT/"qwen2511"
    adapter=MODEL_ROOT/"adapter/latest.safetensors"
    source=Path(os.environ.get("MASKFLOW_SOURCE","/opt/maskflow"))
    if not (source/"configs/pipeline/qwenimage_maskflow.yaml").is_file():raise RuntimeError("Baked MaskFlow source is missing")
    return base,adapter,source
