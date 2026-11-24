"""Build-time downloads only; runtime uses verified local paths."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).parent
MODEL_ROOT=Path(os.environ.get("MASKFLOW_MODEL_ROOT","/opt/models"))
def verify_file(entry,path,full=False):
    if not path.is_file() or path.stat().st_size!=entry["size"]:raise RuntimeError("Missing or incomplete baked model file: "+entry["target"])
    if full:
        sha=hashlib.sha256() if entry["sha256"] else hashlib.sha1(b"blob "+str(entry["size"]).encode()+b"\0")
        with path.open("rb") as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b""):sha.update(chunk)
        if sha.hexdigest()!=(entry["sha256"] or entry["gitBlob"]):raise RuntimeError("Model checksum mismatch: "+entry["target"])
def verify_all(full=False):
    lock=json.loads((ROOT/"weights.lock.json").read_text())
    for entry in lock["files"]:verify_file(entry,MODEL_ROOT/entry["target"],full)
    return len(lock["files"])
if __name__=="__main__":
    if sys.argv[1]=="verify":print("Verified baked model files:",verify_all("--full" in sys.argv));sys.exit(0)
    from huggingface_hub import hf_hub_download
    entries=[e for e in json.loads((ROOT/"weights.lock.json").read_text())["files"] if e["group"]==sys.argv[1]]
    assert entries,"Unknown model group"
    for e in entries:
        directory=MODEL_ROOT/e["target"].split("/")[0]
        path=Path(hf_hub_download(e["repo"],e["path"],revision=e["revision"],local_dir=str(directory)))
        verify_file(e,path,True)
        print("Baked and verified",e["target"],e["size"],flush=True)
