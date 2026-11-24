"""Pinned official MaskFlow baseline; no ComfyUI/Lightning substitution."""
import sys
from PIL import Image
from runtime import baked_paths

def compatible_adapter_metadata(adapter):
    """Preserve all active settings; omit only disabled future-version fields."""
    import inspect
    import json
    from peft import LoraConfig
    from safetensors import safe_open
    with safe_open(str(adapter), framework="pt", device="cpu") as f:
        metadata = json.loads(f.metadata()["lora_adapter_metadata"])
    supported = inspect.signature(LoraConfig).parameters
    unknown_active = [key for key, value in metadata.items() if key not in supported and value is not None]
    if unknown_active:
        raise ValueError("Adapter enables unsupported PEFT settings")
    compatible = {key: value for key, value in metadata.items() if key in supported}
    LoraConfig(**compatible)  # Validate before materializing adapter tensors.
    return compatible

def load_baked_adapter(transformer, adapter):
    from safetensors.torch import load_file
    metadata = compatible_adapter_metadata(adapter)
    # A local state dict avoids offline filename inference and preserves the
    # explicitly validated metadata (Diffusers re-reads metadata for file paths).
    state = load_file(str(adapter), device="cpu")
    transformer.load_lora_adapter(state, adapter_name="maskflow", prefix=None, metadata=metadata)
    transformer.set_adapter("maskflow")
    transformer.fuse_lora(lora_scale=1.0, safe_fusing=True, adapter_names=["maskflow"])
    transformer.unload_lora()

class MaskFlowEngine:
    def __init__(self):
        import torch
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        base, adapter, upstream = baked_paths()
        sys.path.insert(0, str(upstream))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if torch.cuda.get_device_properties(0).total_memory < 75_000_000_000:
            raise RuntimeError("MaskFlow baseline requires an 80GB-class GPU")
        cfg = OmegaConf.load(upstream / "configs/pipeline/qwenimage_maskflow.yaml")
        cfg.pretrained_model = str(base)
        self.pipe = instantiate(cfg, device=torch.device("cuda"), dtype=torch.bfloat16)
        self.pipe.load_pretrained_weights()
        load_baked_adapter(self.pipe.transformer, adapter)
        self.pipe.vae.eval(); self.pipe.text_pipeline.text_encoder.eval(); self.pipe.transformer.eval()
        torch.cuda.synchronize()

    def __call__(self, source, mask, prompt, settings):
        import torch
        import torchvision.transforms.functional as T
        size = (settings["width"], settings["height"])
        # Apply the same geometry to both inputs; no independent crop or random transform.
        source = source.resize(size, Image.Resampling.LANCZOS)
        mask = mask.resize(size, Image.Resampling.NEAREST).convert("RGB")
        with torch.inference_mode():
            result = self.pipe.eval_step({"prompt": [prompt], "negative_prompt": [""],
                "conditions": {"source": T.to_tensor(source).unsqueeze(0), "mask": T.to_tensor(mask).unsqueeze(0)}},
                num_inference_steps=settings["steps"], text_cfg_scale=settings["guidance"], mask_cfg_scale=1.0,
                interaction_cfg_scale=None, seed=settings["seed"])
            return T.to_pil_image(result["output"][0].float().cpu().clamp(0, 1))
