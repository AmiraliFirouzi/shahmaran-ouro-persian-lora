import os, sys, glob, shutil, argparse
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, create_repo

REPO = "amiralifirouzi/Shahmaran-Ouro-1.4B-persian-lora"
BASE = "ByteDance/Ouro-1.4B"
ADAPTER = "/marimo/ouro-lora-2e"
MERGED = "/marimo/ouro-merged-2e"

for d in glob.glob(ADAPTER + "/checkpoint-*"):
    shutil.rmtree(d, ignore_errors=True)
    print("[clean] rm", d, flush=True)

print("[merge] base", BASE, flush=True)
model = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()
model.save_pretrained(MERGED, safe_serialization=True, max_shard_size="4GB")
AutoTokenizer.from_pretrained(BASE, trust_remote_code=True).save_pretrained(MERGED)
print("[merge] ->", MERGED, sorted(os.listdir(MERGED)), flush=True)

api = HfApi()
create_repo(REPO, repo_type="model", exist_ok=True, private=False)

# drop stale artifacts from the previous (1-epoch) push
for pat in ["adapter/checkpoint-*", "results/dry.json", "results/smoke.json"]:
    try:
        api.delete_pattern(repo_id=REPO, patterns=pat, repo_type="model")
        print("[clean-hub]", pat, flush=True)
    except Exception as e:
        print("[clean-hub] skip", pat, e, flush=True)

api.upload_folder(repo_id=REPO, folder_path=MERGED,
                  commit_message="Merged bf16 model (2-epoch Persian SFT)")
api.upload_folder(repo_id=REPO, folder_path=ADAPTER, path_in_repo="adapter",
                  commit_message="LoRA adapter (2 epochs)")
api.upload_file(repo_id=REPO, path_or_fileobj="/marimo/README.md", path_in_repo="README.md",
                commit_message="Add model card with honest evaluation")
for f in ["train.log", "train2.log"]:
    if os.path.exists("/marimo/" + f):
        api.upload_file(repo_id=REPO, path_or_fileobj="/marimo/" + f, path_in_repo=f)
api.upload_folder(repo_id=REPO, folder_path="/marimo/results", path_in_repo="results")

files = sorted(api.list_repo_files(REPO))
print("[hub] root:", [f for f in files if "/" not in f], flush=True)
print("[hub] n_files:", len(files), flush=True)
print("FINALIZE_DONE https://huggingface.co/" + REPO, flush=True)
