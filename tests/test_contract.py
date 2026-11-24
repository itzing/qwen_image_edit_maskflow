import copy
import io
import json
import os
import unittest
from PIL import Image
from handler import process
from secure import MODEL, ROOT, InvalidRequest, encrypt, decrypt

class Store:
    def __init__(self): self.objects = {}
    def read(self, key): return self.objects[key]
    def write(self, key, value): self.objects[key] = value

def png(color, mode="RGB", size=(256,256)):
    out=io.BytesIO(); Image.new(mode, size, color).save(out,format="PNG");return out.getvalue()

class ContractTest(unittest.TestCase):
    def setUp(self):
        self.key=os.urandom(32); self.store=Store();self.calls=0
        self.binding={"job_id":"test-job","model_id":MODEL,"attempt_id":"attempt-1","direction":"engui_to_endpoint"}
        self.payload={"prompt":"Replace the selected region with a red circle.","generation":{"width":256,"height":256,"seed":42,"steps":50,"guidance":4}}
        self.job={"input":{"_secure":encrypt(self.key,self.binding,json.dumps(self.payload).encode(),True)[0],"media_inputs":[],"transport_request":{"output_dir":ROOT,"output_file_name":"test-job__attempt-1__output__result.bin"}}}
        self.add("source_image",png("blue"));self.add("edit_mask",png(255,"L"))
    def add(self, role, data):
        b={**self.binding,"role":role,"kind":"image"};env,cipher=encrypt(self.key,b,data)
        key=f"secure-jobs/test-job__attempt-1__input__{role}.bin";self.store.write(key,cipher)
        self.job["input"]["media_inputs"]=[x for x in self.job["input"]["media_inputs"] if x["role"]!=role]
        self.job["input"]["media_inputs"].append({"role":role,"kind":"image","mime":"image/png","storage_path":"/runpod-volume/"+key,"envelope":env})
    def engine(self, source, mask, prompt, settings):
        self.calls+=1; self.assertEqual(prompt,self.payload["prompt"]);self.assertEqual(mask.mode,"L")
        return Image.new("RGB",(settings["width"],settings["height"]),"red")
    def run_job(self): return process(self.job,self.engine,self.store,self.key)
    def reject(self):
        with self.assertRaises(InvalidRequest):self.run_job()
        self.assertEqual(self.calls,0)
    def test_encrypted_roundtrip(self):
        out=self.run_job()["transport_result"]["result_media"]
        body=self.store.read(out["storage_path"].removeprefix("/runpod-volume/"))
        self.assertFalse(body.startswith(b"\x89PNG"));binding={**self.binding,"direction":"endpoint_to_engui","role":"result","kind":"image"}
        plain=decrypt(self.key,out["envelope"],binding,body)
        self.assertEqual(Image.open(io.BytesIO(plain)).getpixel((0,0)),(255,0,0))
        self.assertEqual(self.calls,1)
    def test_plaintext_rejected(self):self.job["input"]["prompt"]="secret";self.reject()
    def test_tampered_prompt(self):self.job["input"]["_secure"]["binding"]["attempt_id"]="other";self.reject()
    def test_wrong_key(self):self.key=os.urandom(32);self.reject()
    def test_swapped_media_roles(self):self.job["input"]["media_inputs"][0]["envelope"]["binding"]["role"]="edit_mask";self.reject()
    def test_foreign_path(self):self.job["input"]["media_inputs"][0]["storage_path"]=ROOT+"foreign.bin";self.reject()
    def test_traversal_path(self):self.job["input"]["media_inputs"][0]["storage_path"]=ROOT+"test-job__attempt-1__input__../x";self.reject()
    def test_empty_mask(self):self.add("edit_mask",png(0,"L"));self.reject()
    def test_transparent_rgba_mask(self):self.add("edit_mask",png((255,255,255,0),"RGBA"));self.reject()
    def test_transparent_palette_mask(self):
        image=Image.new("P",(256,256),1)
        image.putpalette([0,0,0,255,255,255]+[0]*762)
        out=io.BytesIO();image.save(out,format="PNG",transparency=1)
        self.add("edit_mask",out.getvalue());self.reject()
    def test_opaque_palette_mask(self):
        image=Image.new("P",(256,256),1)
        image.putpalette([0,0,0,255,255,255]+[0]*762)
        out=io.BytesIO();image.save(out,format="PNG")
        self.add("edit_mask",out.getvalue());self.run_job();self.assertEqual(self.calls,1)
    def test_colored_mask(self):self.add("edit_mask",png("red"));self.reject()
    def test_misaligned_mask(self):self.add("edit_mask",png(255,"L",(128,128)));self.reject()
    def test_missing_mask(self):self.job["input"]["media_inputs"].pop();self.reject()
    def test_duplicate_role(self):self.job["input"]["media_inputs"][1]["role"]="source_image";self.reject()
    def test_output_traversal(self):self.job["input"]["transport_request"]["output_file_name"]="../secret";self.reject()
    def test_output_wrong_attempt(self):self.job["input"]["transport_request"]["output_file_name"]="test-job__attempt-2__output__result.bin";self.reject()
    def test_dimension_limit(self):
        self.payload["generation"].update(width=2048,height=2048)
        self.job["input"]["_secure"]=encrypt(self.key,self.binding,json.dumps(self.payload).encode(),True)[0];self.reject()
    def test_tampered_media(self):
        k="secure-jobs/test-job__attempt-1__input__edit_mask.bin";self.store.objects[k]=self.store.objects[k][:-1]+bytes([self.store.objects[k][-1]^1]);self.reject()

if __name__=="__main__": unittest.main()
