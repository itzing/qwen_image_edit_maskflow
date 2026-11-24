"""Pinned official MaskFlow baseline; no ComfyUI/Lightning substitution."""
import json
import os
import sys
from pathlib import Path
from PIL import Image
import time
from runtime import baked_paths

class MaskFlowEngine:
    def __init__(self):
        import torch
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        started = time.perf_counter()
        base, adapter, upstream = baked_paths()
        sys.path.insert(0, str(upstream))
        from trainer.lora_utils import load_inference_loras
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if torch.cuda.get_device_properties(0).total_memory < 75_000_000_000:
            raise RuntimeError("MaskFlow baseline requires an 80GB-class GPU")
        cfg = OmegaConf.load(upstream / "configs/pipeline/qwenimage_maskflow.yaml")
        cfg.pretrained_model = str(base)
        self.pipe = instantiate(cfg, device=torch.device("cuda"), dtype=torch.bfloat16)
        self.pipe.load_pretrained_weights()
        checkpoint = OmegaConf.create({"sft_path": str(adapter), "sft_weight_name": None, "sft_adapter_name": "maskflow", "dmd_path": None, "dmd_weight_name": None, "dmd_adapter_name": "dmd"})
        load_inference_loras(self.pipe.transformer, checkpoint, 1.0)
        self.pipe.vae.eval(); self.pipe.text_pipeline.text_encoder.eval(); self.pipe.transformer.eval()
        torch.cuda.synchronize()
        self.load_ms = round((time.perf_counter() - started) * 1000)
        print(json.dumps({"event": "model_loaded", "modelLoadMs": self.load_ms}), flush=True)

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
