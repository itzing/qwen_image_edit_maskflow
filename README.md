# Qwen Image Edit 2511 + MaskFlow — Engui secure endpoint

Secure-only RunPod queue endpoint for masked image editing. Model ID: `qwen-image-edit-maskflow`.

## Runtime: no custom image build

Use the published CUDA/PyTorch image digest in `pins.json`. Download this repository at an immutable commit into local worker storage and run `bootstrap.sh`. It installs the official MaskFlow locked Python 3.12 runtime on a persistent cache; no Docker build is needed.

Use an 80 GB-class GPU or larger. RTX PRO 6000 Blackwell Server Edition (96 GB) passed a real secure inference test. Deployment uses minimum 0 and maximum 2 workers. First launch downloads approximately 60 GB of model/adapters plus runtime; reserve at least 100 GB of persistent cache. Cache storage is billed separately from GPU execution.

Pinned upstream [MaskFlow](https://github.com/ReyChiaro/MaskFlow), Qwen 2511 and latest general-scene SFT adapter revisions are in `pins.json`. The engine uses the official mask-conditioned scheduler and Soft-Poisson, with 50 steps / CFG 4 by default. This is not ComfyUI or Lightning 4. Adapter weights are not included in this repository. Upstream code/weights are MIT; the Qwen base retains its own license.

## Secret environment

- `FIELD_ENC_KEY_B64`: Engui shared 32-byte master key, base64 (or a provider-managed secret resolved into this environment variable).
- `ENGUI_S3_ENDPOINT`, `ENGUI_S3_BUCKET`, `ENGUI_S3_REGION`, `ENGUI_S3_ACCESS_KEY`, `ENGUI_S3_SECRET_KEY`: existing encrypted transport bucket. HTTPS required. It may be in a different region from the model cache; only ciphertext is transferred.
- `MASKFLOW_CACHE_ROOT`: optional persistent model/environment cache, default `/runpod-volume/maskflow-cache`.

Never put secret values in the repository, request examples, startup command or image. Provider environment is managed outside version control.

## Wire contract

Input `_secure` is Engui AES-256-GCM v1 containing exactly `prompt` and `generation` (`width`, `height`, `seed`, `steps`, `guidance`). It authenticates job/model/attempt/direction. `media_inputs` has exactly two encrypted descriptors: `source_image` and `edit_mask`, each additionally authenticated by role/kind.

The mask must be opaque, aligned grayscale: white edits, black preserves; empty masks are rejected. No automatic text-to-mask segmentation. Input images are limited to 25 MiB / 40 MP; output to 1 MP, with sides 256..2048 in multiples of 64.

Both inputs undergo the same resize to requested output dimensions. Match the source aspect ratio to avoid distortion. Output `transport_result.result_media` contains only the encrypted result descriptor. Plaintext prompts, URLs, paths and base64 images are rejected. File keys are confined to the authenticated job/attempt; arbitrary writes are not accepted.

Plaintext exists only inside worker memory. Persistent transport contains ciphertext; the model cache contains public weights/code only. Generation errors do not echo upstream exception text. Each worker runs one job at a time. RunPod cancellation semantics apply; closing a client connection is not cancellation.

## MCP integration

Engui exposes this as a dedicated MCP-only model; ordinary Qwen and UI are unchanged. Prepare a plan with one source reference and an explicit image mask or normalized rectangles, inspect the frozen plan, then start. Mask/source identities and hashes are captured in frozen plans and rechecked before dispatch. Engui enforces workspace ownership; arbitrary remote media URLs are not accepted here.

## Validation

`python3 -m unittest discover -s tests -v` requires cryptography and Pillow. The 19 CPU tests use an in-memory fake engine/storage, never GPU or paid inference. They check encrypted roundtrip, authentication tampering, wrong key/role/attempt, path traversal, opaque/grayscale/aligned masks and resource bounds.

A separate real GPU test on 2026-11-24 completed one encrypted 512x512 synthetic edit using revision `c6d5815`, the same pinned model/runtime/engine as the current handler. Provider execution was 94.9 seconds after 53.2 seconds of queue delay. The output decrypted successfully. Background more than 25 pixels outside the mask was pixel-identical to the input. The model also added an unwanted white shape inside the editable region: this verifies operation and background preservation in that sample, not perfect instruction following or a general benchmark.

Revision `501930b` additionally rejects transparent palette masks before inference; the updated 19-test CPU suite passed. No second GPU inference was needed for that validation-only change.
