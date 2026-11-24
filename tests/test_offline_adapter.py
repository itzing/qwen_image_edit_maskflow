import json
import tempfile
import unittest
from pathlib import Path

class OfflineAdapterTest(unittest.TestCase):
    def test_pinned_adapter_metadata_accepted_by_installed_peft(self):
        from peft import LoraConfig
        from engine import compatible_adapter_metadata
        path = Path("/opt/models/adapter/latest.safetensors")
        self.assertTrue(path.is_file(), "Run offline integration inside the baked image")
        config = LoraConfig(**compatible_adapter_metadata(path))
        self.assertEqual(config.r, 256)
        self.assertEqual(config.lora_alpha, 256)
        self.assertIn("attn.to_q", config.target_modules)

    def test_real_offline_adapter_load_fuse_and_unload(self):
        import torch
        from safetensors.torch import save_file
        from peft import LoraConfig
        from diffusers import ModelMixin
        from diffusers.loaders import PeftAdapterMixin
        from engine import load_baked_adapter
        class Tiny(ModelMixin, PeftAdapterMixin):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(3, 2, bias=False)
        model = Tiny()
        before = model.proj.weight.detach().clone()
        metadata = LoraConfig(r=1, lora_alpha=1, target_modules=["proj"]).to_dict()
        metadata.update(monteclora_config=None, velora_config=None)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "latest.safetensors"
            save_file({"proj.lora_A.weight": torch.ones(1, 3), "proj.lora_B.weight": torch.ones(2, 1)}, str(path),
                metadata={"lora_adapter_metadata": json.dumps(metadata, default=list)})
            load_baked_adapter(model, path)
        self.assertTrue(torch.allclose(model.proj.weight, before + 1))
        self.assertIsInstance(model.proj, torch.nn.Linear)

    def test_unknown_active_adapter_setting_fails_closed(self):
        import torch
        from safetensors.torch import save_file
        from engine import compatible_adapter_metadata
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "latest.safetensors"
            save_file({"a": torch.ones(1)}, str(path), metadata={"lora_adapter_metadata": json.dumps({"future_option": {"enabled": True}})})
            with self.assertRaisesRegex(ValueError, "unsupported PEFT"):
                compatible_adapter_metadata(path)
