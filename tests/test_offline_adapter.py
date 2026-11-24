import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

class OfflineAdapterTest(unittest.TestCase):
    def test_baked_adapter_resolves_with_network_disabled(self):
        import torch
        from safetensors.torch import save_file
        from diffusers.loaders import lora_base
        from engine import adapter_checkpoint
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "latest.safetensors"
            save_file({"test.lora_A.weight": torch.ones((2, 3))}, str(path))
            config = adapter_checkpoint(path)
            args = dict(pretrained_model_name_or_path_or_dict=config.sft_path,
                weight_name=config.sft_weight_name, use_safetensors=True,
                local_files_only=True, cache_dir=None, force_download=False,
                proxies=None, token=None, revision=None, subfolder=None,
                user_agent=None, allow_pickle=False)
            with patch.object(lora_base, "HF_HUB_OFFLINE", True):
                with self.assertRaisesRegex(ValueError, "weight_name"):
                    lora_base._fetch_state_dict(**{**args, "weight_name": None})
                state, _ = lora_base._fetch_state_dict(**args)
            self.assertTrue(torch.equal(state["test.lora_A.weight"], torch.ones((2, 3))))
