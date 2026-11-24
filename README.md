# Qwen Image Edit 2511 + MaskFlow — Engui secure endpoint

Secure-only RunPod queue endpoint for masked image editing. Model ID: `qwen-image-edit-maskflow`.

## Runtime: self-contained Docker image

Build with `docker build -t itzing/qwen-image-edit-maskflow:v1 .` and deploy the resulting immutable registry digest. The image contains the pinned Qwen 2511 base, MaskFlow adapter and source, Python 3.12 environment and every runtime dependency. No model download, pip installation, Git checkout or network model-cache volume is needed at worker startup. The compact Python base avoids a duplicate PyTorch installation; the pinned official lock supplies PyTorch 2.13.0 / CUDA 13.0 and its CUDA libraries. The upstream pyproject declares an older torch version, but `uv sync --frozen` follows the pinned lock. Use a CUDA 13-capable driver; do not infer runtime compatibility from the older pyproject or base-image label.

`weights.lock.json` records 34 files (approximately 60.17 GB) with immutable revisions, sizes and checksum metadata. Downloads and checksum verification happen at image build time, in separate model-shard layers. Runtime enforces `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` and fails if baked files are absent or incomplete; there is no download fallback. Only the existing encrypted S3 transport requires network access.

Use an 80 GB-class GPU or larger; RTX PRO 6000 Blackwell Server Edition (96 GB) is the deployment target. Minimum 0 and maximum 2 workers. Image pulling and reading weights into RAM/VRAM still take time on a cold worker; baking eliminates startup weight downloads from Hugging Face or a network model disk, not all cold-start work.

Pinned upstream [MaskFlow](https://github.com/ReyChiaro/MaskFlow), Qwen 2511 and latest general-scene SFT adapter revisions are in `pins.json`. The engine uses the official mask-conditioned scheduler and Soft-Poisson, with 50 steps / CFG 4 by default. This is not ComfyUI or Lightning 4. Adapter weights are not included in this repository. Upstream code/weights are MIT; the Qwen base retains its own license.

## Secret environment

- `FIELD_ENC_KEY_B64`: Engui shared 32-byte master key, base64 (or a provider-managed secret resolved into this environment variable).
- `ENGUI_S3_ENDPOINT`, `ENGUI_S3_BUCKET`, `ENGUI_S3_REGION`, `ENGUI_S3_ACCESS_KEY`, `ENGUI_S3_SECRET_KEY`: existing encrypted transport bucket. HTTPS required. It may be in a different region from the model cache; only ciphertext is transferred.
- Model files are baked under `/opt/models`; source/runtime under `/opt/maskflow`. No persistent model-cache mount is required. Keep default local paths unless constructing a separately verified image.

Never put secret values in the repository, request examples, startup command or image. Provider environment is managed outside version control.

## Wire contract

Input `_secure` is Engui AES-256-GCM v1 containing exactly `prompt` and `generation` (`width`, `height`, `seed`, `steps`, `guidance`). It authenticates job/model/attempt/direction. `media_inputs` has exactly two encrypted descriptors: `source_image` and `edit_mask`, each additionally authenticated by role/kind.

The mask must be opaque, aligned grayscale: white edits, black preserves; empty masks are rejected. No automatic text-to-mask segmentation. Input images are limited to 25 MiB / 40 MP; output to 1 MP, with sides 256..2048 in multiples of 64.

Both inputs undergo the same resize to requested output dimensions. Match the source aspect ratio to avoid distortion. Output `transport_result.result_media` contains only the encrypted result descriptor. Plaintext prompts, URLs, paths and base64 images are rejected. File keys are confined to the authenticated job/attempt; arbitrary writes are not accepted.

Plaintext exists only inside worker memory. Persistent transport contains ciphertext; the image contains public weights/code only. Generation errors do not echo upstream exception text. Each worker runs one job at a time. RunPod cancellation semantics apply; closing a client connection is not cancellation.

## MCP integration

Engui exposes this as a dedicated MCP-only model; ordinary Qwen and UI are unchanged. Prepare a plan with one source reference and an explicit image mask or normalized rectangles, inspect the frozen plan, then start. Mask/source identities and hashes are captured in frozen plans and rechecked before dispatch. Engui enforces workspace ownership; arbitrary remote media URLs are not accepted here.

## Validation

`python3 -m unittest discover -s tests -v` requires cryptography and Pillow. The 21 CPU tests use an in-memory fake engine/storage, never GPU or paid inference. They check encrypted roundtrip, authentication tampering, wrong key/role/attempt, path traversal, opaque/grayscale/aligned masks and resource bounds.

A separate real GPU test on 2026-11-24 completed one encrypted 512x512 synthetic edit using revision `c6d5815`, the same pinned model/runtime/engine as the current handler. Provider execution was 94.9 seconds after 53.2 seconds of queue delay. The output decrypted successfully. Background more than 25 pixels outside the mask was pixel-identical to the input. The model also added an unwanted white shape inside the editable region: this verifies operation and background preservation in that sample, not perfect instruction following or a general benchmark.

Revision `501930b` additionally rejects transparent palette masks before inference; the updated 19-test CPU suite passed. No second GPU inference was needed for that validation-only change.

## Timings

Successful output includes `transport_result.timings`: `modelLoadMs`, `inferenceMs`, `handlerMs` and `modelWasWarm`. Model initialization is measured separately from the edit. Inference includes the engine's image preprocessing, pipeline and conversion to an output image. Handler time also includes encrypted media I/O. RunPod `delayTime` and `executionTime` remain separate provider-level measurements; neither is a pure model-download benchmark.

The older 94.9-second test above measured total provider execution with lazy model initialization and inference together. It must not be read as 94.9 seconds of either downloading or warmed-up inference. The former 100 GB network model-cache volume was deleted at the owner's request on 2026-11-24.
