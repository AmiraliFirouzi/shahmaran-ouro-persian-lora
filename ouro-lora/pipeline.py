import os, sys, subprocess, time

env = dict(os.environ)
env.update(USE_TF="0", TRANSFORMERS_NO_TF="1", USE_TORCH="1", TOKENIZERS_PARALLELISM="false")

LOG = "/marimo/train.log"
PY = sys.executable


def sh(cmd):
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env).returncode


print("[pipe] waiting for training...", flush=True)
deadline = time.time() + 3 * 3600
while time.time() < deadline:
    if os.path.exists(LOG):
        txt = open(LOG, errors="replace").read()
        if "ALL_DONE" in txt:
            print("[pipe] training done", flush=True)
            break
        if "Traceback" in txt and "ALL_DONE" not in txt and txt.rstrip().endswith("Error"):
            pass
    time.sleep(30)
else:
    print("[pipe] timeout waiting", flush=True)

JOBS = [
    ("ouro_base", "ByteDance/Ouro-1.4B", None),
    ("ouro_ft", "ByteDance/Ouro-1.4B", "/marimo/ouro-lora"),
    ("qwen3_4b", "Qwen/Qwen3-4B", None),
    ("cassandane", "Qwen/Qwen3-4B", "amiralifirouzi/Cassandane-qwen3-4b-persian-lora"),
]
for tag, mid, ad in JOBS:
    cmd = [PY, "/marimo/eval_bench.py", "--model", mid, "--tag", tag, "--limit", "500"]
    if ad:
        cmd += ["--adapter", ad]
    sh(cmd)

print("[pipe] merge + push", flush=True)
sh([PY, "/marimo/upload_ouro.py", "--push"])
print("PIPELINE_DONE", flush=True)
