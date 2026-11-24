"""Engui AES-256-GCM v1. No plaintext transport or caller-controlled remote URLs."""
import base64
import json
import os
import re
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MODEL = "qwen-image-edit-maskflow"
ROOT = "/runpod-volume/secure-jobs/"
MAX_IMAGE = 25 * 1024 * 1024

class InvalidRequest(ValueError):
    pass

def require(condition, message):
    if not condition:
        raise InvalidRequest(message)

def decode(value):
    try:
        return base64.b64decode(value, validate=True)
    except Exception:
        raise InvalidRequest("Invalid secure envelope") from None

def key_from_env():
    key = decode(os.environ.get("FIELD_ENC_KEY_B64", ""))
    require(len(key) == 32, "Encryption key is not configured")
    return key

def aad(binding):
    return json.dumps(binding, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def binding_from(envelope):
    require(isinstance(envelope, dict) and envelope.get("v") == 1, "Secure v1 envelope is required")
    b = envelope.get("binding", {})
    require(set(b) == {"job_id", "model_id", "attempt_id", "direction"}, "Invalid prompt binding")
    require(b["model_id"] == MODEL and b["direction"] == "engui_to_endpoint", "Invalid model or direction")
    for name in ("job_id", "attempt_id"):
        require(isinstance(b[name], str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", b[name]), "Invalid request identity")
    return b

def decrypt(master, envelope, expected, ciphertext=None):
    require(isinstance(envelope, dict) and envelope.get("v") == 1 and envelope.get("binding") == expected, "Secure binding mismatch")
    wrapped = envelope.get("wrapped_key", "")
    require(isinstance(wrapped, str) and wrapped.startswith("v1:"), "Invalid wrapped key")
    try:
        w = decode(wrapped[3:])
        dek = AESGCM(master).decrypt(w[:12], w[12:], b"engui:wrapped-key:v1")
        nonce = decode(envelope.get("nonce", ""))
        require(len(nonce) == 12 and len(dek) == 32, "Invalid secure nonce or key")
        body = ciphertext if ciphertext is not None else decode(envelope.get("ciphertext", ""))
        return AESGCM(dek).decrypt(nonce, body, aad(expected))
    except InvalidRequest:
        raise
    except Exception:
        raise InvalidRequest("Secure authentication failed") from None

def encrypt(master, binding, plaintext, structured=False):
    dek, nonce, wrap_nonce = os.urandom(32), os.urandom(12), os.urandom(12)
    ciphertext = AESGCM(dek).encrypt(nonce, plaintext, aad(binding))
    wrapped = wrap_nonce + AESGCM(master).encrypt(wrap_nonce, dek, b"engui:wrapped-key:v1")
    envelope = {"v": 1, "binding": binding, "wrapped_key": "v1:" + base64.b64encode(wrapped).decode(), "nonce": base64.b64encode(nonce).decode()}
    if structured:
        envelope["ciphertext"] = base64.b64encode(ciphertext).decode()
    return envelope, ciphertext

def storage_key(path, binding, direction):
    require(isinstance(path, str), "Missing storage path")
    prefix = ROOT + binding["job_id"] + "__" + binding["attempt_id"] + "__" + direction + "__"
    require(path.startswith(prefix), "Storage path does not belong to this attempt")
    suffix = path[len(prefix):]
    require(bool(re.fullmatch(r"[a-zA-Z0-9_.-]{1,160}", suffix)) and ".." not in suffix, "Invalid storage filename")
    return path[len("/runpod-volume/"):]

class S3Store:
    """Only encrypted bytes cross the provider boundary; no local volume aliasing."""
    def __init__(self):
        import boto3
        from botocore.config import Config
        endpoint = os.environ["ENGUI_S3_ENDPOINT"]
        require(endpoint.startswith("https://"), "Secure storage endpoint must use HTTPS")
        self.bucket = os.environ["ENGUI_S3_BUCKET"]
        self.client = boto3.client("s3", endpoint_url=endpoint,
            aws_access_key_id=os.environ["ENGUI_S3_ACCESS_KEY"], aws_secret_access_key=os.environ["ENGUI_S3_SECRET_KEY"],
            region_name=os.environ.get("ENGUI_S3_REGION", "us-east-1"),
            config=Config(signature_version="s3v4", connect_timeout=15, read_timeout=60, retries={"max_attempts": 3}, s3={"addressing_style": "path"}))

    def read(self, key):
        r = self.client.get_object(Bucket=self.bucket, Key=key)
        body = r["Body"]
        try:
            require(r.get("ContentLength", MAX_IMAGE + 17) <= MAX_IMAGE + 16, "Encrypted input is too large")
            result = body.read(MAX_IMAGE + 17)
            require(len(result) <= MAX_IMAGE + 16, "Encrypted input is too large")
            return result
        finally:
            body.close()

    def write(self, key, data):
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType="application/octet-stream")
