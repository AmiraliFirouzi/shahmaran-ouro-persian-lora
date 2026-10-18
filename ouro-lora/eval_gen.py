import os, sys, json, re, argparse, time
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench_data import load_bench, BENCHES

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--adapter", default=None)
p.add_argument("--tag", required=True)
p.add_argument("--limit", type=int, default=500)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--maxnew", type=int, default=8)
p.add_argument("--maxlen", type=int, default=4096)
p.add_argument("--trunc-side", default="left", choices=["left", "right"])
p.add_argument("--benches", default="khayyam,percul,parsinlu_mc,parsinlu_entail,medqa,musr")
p.add_argument("--no-cache", action="store_true", help="disable KV cache (required for Ouro custom code)")
a = p.parse_args()

dev = "cuda"
print(f"[load] {a.tag} model={a.model} adapter={a.adapter}", flush=True)
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
tok.truncation_side = a.trunc_side
model = AutoModelForCausalLM.from_pretrained(
    a.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
    device_map={"": 0}, attn_implementation="sdpa",
)
model.config.use_cache = not a.no_cache
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
model.eval()

SYS = ("شما یک دستیار دقیق فارسی هستید. به سوال چندگزینه‌ای پاسخ می‌دهید و "
       "فقط شماره گزینه صحیح را می‌نویسید.")


def build_prompt(it):
    lines = [f"{i+1}) {o}" for i, o in enumerate(it["options"])]
    user = (f"{it['prompt']}\n\nگزینه‌ها:\n" + "\n".join(lines) +
            "\n\nفقط شماره گزینه صحیح را بنویس. پاسخ:")
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse_ans(text, k):
    m = re.search(r"\d+", text)
    if m:
        v = int(m.group())
        if 1 <= v <= k:
            return v - 1
    for ch, idx in zip("ABCD", range(k)) if k <= 4 else []:
        if ch in text.upper():
            return idx
    return -1


def evaluate(name, items):
    items = items[: a.limit]
    t0 = time.time()
    prompts = [build_prompt(it) for it in items]
    correct = 0
    preds = []
    for i in range(0, len(prompts), a.batch):
        chunk = prompts[i:i + a.batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=a.maxlen, add_special_tokens=False).to(dev)
        with torch.inference_mode():
            out = model.generate(**enc, do_sample=False, num_beams=1,
                                 max_new_tokens=a.maxnew, pad_token_id=tok.pad_token_id,
                                 use_cache=not a.no_cache)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for j, t in enumerate(texts):
            it = items[i + j]
            pred = parse_ans(t, len(it["options"]))
            preds.append(pred)
            correct += int(pred == it["gold"])
    acc = correct / len(items)
    print(f"[{name}] n={len(items)} acc={acc:.4f} ({correct}/{len(items)}) {time.time()-t0:.0f}s", flush=True)
    return {"n": len(items), "acc": acc}


results = {}
os.makedirs("/marimo/results", exist_ok=True)
for name in a.benches.split(","):
    name = name.strip()
    if not name:
        continue
    try:
        results[name] = evaluate(name, load_bench(name))
    except Exception as e:
        import traceback
        print(f"[{name}] ERROR {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        results[name] = {"error": f"{type(e).__name__}: {e}"}
    with open(f"/marimo/results/gen_{a.tag}.json", "w") as f:
        json.dump({"model": a.model, "adapter": a.adapter, "mode": "generation",
                   "results": results}, f, ensure_ascii=False, indent=2)
print("RESULTS", json.dumps(results, ensure_ascii=False), flush=True)
print("GEN_DONE", a.tag, flush=True)
