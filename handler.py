"""Secure-only RunPod queue handler. Importable without GPU for contract tests."""
import io
import json
import threading
from PIL import Image, ImageOps
from secure import MODEL, ROOT, InvalidRequest, require, key_from_env, binding_from, decrypt, encrypt, storage_key, S3Store, MAX_IMAGE

Image.MAX_IMAGE_PIXELS = 40_000_000
_ENGINE = None
_LOCK = threading.Lock()

def decode_image(data, mask=False):
    require(len(data) <= MAX_IMAGE, "Image exceeds the input limit")
    try:
        image = Image.open(io.BytesIO(data))
        require(image.format in ("PNG", "JPEG", "WEBP") and getattr(image, "n_frames", 1) == 1, "Only static PNG/JPEG/WebP images are supported")
        require(image.width * image.height <= 40_000_000, "Image exceeds pixel limit")
        image.load()
        image = ImageOps.exif_transpose(image)
        if mask:
            # Alpha is never implicitly interpreted as the edit region.
            require(image.mode in ("1", "L", "LA", "RGB", "RGBA", "P"), "Invalid mask format")
            if "A" in image.getbands():
                require(image.getchannel("A").getextrema() == (255, 255), "Mask must be opaque")
            rgb = image.convert("RGB")
            r, g, b = rgb.split()
            from PIL import ImageChops
            require(ImageChops.difference(r, g).getbbox() is None and ImageChops.difference(r, b).getbbox() is None, "Mask must be grayscale: white edits, black preserves")
            image = rgb.convert("L")
            require(image.getextrema()[1] > 0, "Mask has no editable pixels")
        else:
            image = image.convert("RGB")
        return image
    except InvalidRequest:
        raise
    except Exception:
        raise InvalidRequest("Image cannot be decoded") from None

def settings_from(payload):
    require(isinstance(payload, dict) and set(payload) == {"prompt", "generation"}, "Prompt and authenticated generation settings are required")
    prompt, s = payload["prompt"], payload["generation"]
    require(isinstance(prompt, str) and 0 < len(prompt.strip()) <= 64000, "Invalid prompt")
    require(isinstance(s, dict) and set(s) == {"width", "height", "seed", "steps", "guidance"}, "Invalid generation settings")
    for key in ("width", "height"):
        require(type(s[key]) is int and 256 <= s[key] <= 2048 and s[key] % 64 == 0, "Dimensions must be multiples of64 within256..2048")
    require(s["width"] * s["height"] <= 1_048_576, "MaskFlow baseline supports at most1megapixel")
    require(type(s["seed"]) is int and 0 <= s["seed"] <= 2147483647, "Invalid seed")
    require(type(s["steps"]) is int and 1 <= s["steps"] <= 50, "Invalid steps")
    require(type(s["guidance"]) in (int, float) and 1 <= s["guidance"] <= 8, "Invalid guidance")
    return prompt, s

def process(job, engine, store, master):
    data = job.get("input")
    require(isinstance(data, dict), "Missing secure input")
    require(not any(k in data for k in ("prompt", "negative_prompt", "negativePrompt", "image_url", "image_path", "image_base64", "mask", "mask_url")), "Plaintext input is not accepted")
    binding = binding_from(data.get("_secure"))
    try:
        payload = json.loads(decrypt(master, data["_secure"], binding))
    except (UnicodeError, json.JSONDecodeError):
        raise InvalidRequest("Invalid authenticated payload") from None
    prompt, settings = settings_from(payload)
    descriptors = data.get("media_inputs")
    require(isinstance(descriptors, list) and len(descriptors) == 2, "Source image and edit mask are required")
    require(all(isinstance(d, dict) for d in descriptors) and {d.get("role") for d in descriptors} == {"source_image", "edit_mask"}, "Exactly one source image and one edit mask are required")
    images = {}
    for d in descriptors:
        role = d["role"]
        require(d.get("kind") == "image", "Invalid media kind")
        expected = {**binding, "role": role, "kind": "image"}
        path = storage_key(d.get("storage_path"), binding, "input")
        images[role] = decode_image(decrypt(master, d.get("envelope"), expected, store.read(path)), mask=role == "edit_mask")
    require(images["source_image"].size == images["edit_mask"].size, "Source and mask dimensions must match")
    tr = data.get("transport_request", {})
    require(isinstance(tr, dict) and tr.get("output_dir", "").rstrip("/") == ROOT.rstrip("/"), "Invalid output directory")
    output_path = ROOT + tr.get("output_file_name", "")
    out_key = storage_key(output_path, binding, "output")
    # A single worker executes one generation at a time, preventing model/state races.
    with _LOCK:
        result = engine(images["source_image"], images["edit_mask"], prompt, settings)
    require(isinstance(result, Image.Image) and result.size == (settings["width"], settings["height"]), "Invalid engine output")
    output = io.BytesIO()
    result.convert("RGB").save(output, format="PNG")
    out_binding = {**binding, "direction": "endpoint_to_engui", "role": "result", "kind": "image"}
    env, ciphertext = encrypt(master, out_binding, output.getvalue())
    store.write(out_key, ciphertext)
    return {"transport_result": {"status": "completed", "result_media": {"kind": "image", "mime": "image/png", "storage_path": output_path, "envelope": env}}}

def handler(job):
    global _ENGINE
    try:
        def run_engine(*args):
            global _ENGINE
            if _ENGINE is None:
                from engine import MaskFlowEngine
                _ENGINE = MaskFlowEngine()
            return _ENGINE(*args)
        return process(job, run_engine, S3Store(), key_from_env())
    except InvalidRequest as e:
        return {"transport_result": {"status": "failed", "error": {"code": "INVALID_SECURE_REQUEST", "message": str(e)}}}
    except Exception as error:
        print("MaskFlow failure class:", type(error).__name__, flush=True)
        # Third-party exception messages can contain prompts/credentials. Never echo them.
        return {"transport_result": {"status": "failed", "error": {"code": "MASKFLOW_FAILED", "message": "MaskFlow execution failed; inspect secret-free worker health."}}}

if __name__ == "__main__":
    import runpod
    key_from_env()
    runpod.serverless.start({"handler": handler})
