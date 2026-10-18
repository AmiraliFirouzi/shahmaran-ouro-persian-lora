import os, sys, argparse, json
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, create_repo

p = argparse.ArgumentParser()
p.add_argument("--base", default="ByteDance/Ouro-1.4B")
p.add_argument("--adapter", default="/marimo/ouro-lora")
p.add_argument("--merged-dir", default="/marimo/ouro-merged")
p.add_argument("--repo", default="amiralifirouzi/Shahmaran-Ouro-1.4B-persian-lora")
p.add_argument("--push", action="store_true")
a = p.parse_args()

os.makedirs(a.merged_dir, exist_ok=True)
print("[merge] loading base", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    a.base, trust_remote_code=True, torch_dtype=torch.bfloat16
)
print("[merge] attaching adapter", flush=True)
model = PeftModel.from_pretrained(model, a.adapter)
model = model.merge_and_unload()
model.save_pretrained(a.merged_dir, safe_serialization=True, max_shard_size="4GB")
tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
tok.save_pretrained(a.merged_dir)
print("[merge] saved ->", a.merged_dir, flush=True)
print("files:", sorted(os.listdir(a.merged_dir)), flush=True)

if a.push:
    api = HfApi()
    create_repo(a.repo, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(repo_id=a.repo, folder_path=a.merged_dir,
                      commit_message="Add merged Ouro-1.4B Persian SFT (LoRA merged, bf16)")
    api.upload_folder(repo_id=a.repo, folder_path=a.adapter, path_in_repo="adapter",
                      commit_message="Add LoRA adapter files")
    for extra in ["train.log", "results"]:
        fp = "/marimo/" + extra
        if os.path.exists(fp):
            try:
                if os.path.isdir(fp):
                    api.upload_folder(repo_id=a.repo, folder_path=fp, path_in_repo=extra)
                else:
                    api.upload_file(repo_id=a.repo, path_or_fileobj=fp, path_in_repo=extra)
            except Exception as e:
                print("skip", extra, e)
    print("PUSHED", f"https://huggingface.co/{a.repo}", flush=True)
print("UPLOAD_DONE", flush=True)
