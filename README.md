# Qwen Image Edit 2511 + MaskFlow — Engui secure endpoint

Secure-only RunPod queue endpoint for precise masked image editing. Model ID: `qwen-image-edit-maskflow`.

## Runtime (no custom image build)
Use the published CUDA/PyTorch image digest in `pins.json`. Download this repository at an immutable commit into local worker storage and run `bootstrap.sh`. The script installs the official MaskFlow locked Python3.12 runtime on a persistent cache; no Docker build is needed. Require an80GB-class GPU; RTX PRO6000 Blackwell96GB is a supported validation target, not yet a benchmark result. minWorkers0, maxWorkers2. First launch downloads ~60GB of model/adapters plus runtime; reserve100GB+ persistent cache. No sample output is claimed until GPU verification is recorded.

Pinned upstream [MaskFlow](https://github.com/ReyChiaro/MaskFlow), Qwen2511 and latest general-scene SFT adapter revisions are in `pins.json`. Uses official mask-conditioned scheduler and Soft-Poisson,50steps/CFG4 by default. This is not ComfyUI or Lightning4. The adapter weights are not included in this repo. Upstream code/weights MIT; Qwen base retains its own license.

## Secret environment
- `FIELD_ENC_KEY_B64`: Engui shared32-byte master key, base64.
- `ENGUI_S3_ENDPOINT`, `ENGUI_S3_BUCKET`, `ENGUI_S3_REGION`, `ENGUI_S3_ACCESS_KEY`, `ENGUI_S3_SECRET_KEY`: existing encrypted transport bucket. HTTPS required. It may be in a different region than the model cache; only ciphertext is transferred.
- `MASKFLOW_CACHE_ROOT`: optional persistent model/environment cache (default `/runpod-volume/maskflow-cache`).
Never put secret values in repository, request examples, startup command or image. Provider environment is managed outside version control.

## Wire contract
Input `_secure` is Engui AES256-GCM v1 containing exactly `prompt` and `generation` (`width`,`height`,`seed`,`steps`,`guidance`). It authenticates job_id/model_id/attempt_id/direction. `media_inputs` has exactly2 encrypted descriptors: `source_image` and `edit_mask`, each additionally authenticated by role/kind. Mask must be aligned grayscale: white edits, black preserves; empty masks rejected. No automatic text-to-mask segmentation. Input images up to25MiB/40MP, output up to1MP, sides256..2048 multiples64.

Both inputs undergo the same resize to requested output dimensions. Match the source aspect ratio to avoid distortion. Output `transport_result.result_media` contains only the encrypted result descriptor. Plaintext prompts, URLs, paths and base64 images are rejected. File keys are confined to the authenticated job/attempt; no arbitrary writes. Plaintext exists only inside worker memory; persistent transport contains ciphertext, model cache contains public weights/code only. Generation errors do not echo upstream exception text. One job per worker. RunPod cancellation semantics apply; killing a client connection is not cancellation.

## MCP integration
Engui exposes this as a dedicated MCP-only model; ordinary Qwen and UI are unchanged. Prepare a plan with source reference and explicit mask, review it, then start. Masks and source identities/hashes must be captured in frozen plans and rechecked before dispatch. Asset ownership is enforced by Engui, not by accepting arbitrary remote URLs here.

## Local contract tests
`python3 -m unittest discover -s tests -v` (cryptography + Pillow). Tests use an in-memory fake engine/storage, never GPU or paid inference. They check encrypted roundtrip, authentication tampering, wrong key/role/attempt, path traversal, mask alignment/content and resource bounds. CPU tests do not prove model inference.
