import os, sys, json, ast, argparse, time
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--adapter", default=None)
p.add_argument("--tag", required=True)
p.add_argument("--limit", type=int, default=500)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--maxlen", type=int, default=1024)
p.add_argument("--benches", default="percor,khayyam,percul,parsinlu_mc,parsinlu_entail,medqa,musr")
p.add_argument("--data-only", action="store_true")
a = p.parse_args()

dev = "cuda"
tok = None
model = None
PAD = 0
if not a.data_only:
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    PAD = tok.pad_token_id
    print(f"[load] {a.tag} model={a.model} adapter={a.adapter}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()


def score_pairs(pairs):
    """pairs: list of (prompt_ids, comp_ids) -> list of sum-logprob (float)."""
    seqs, plens = [], []
    for pids, cids in pairs:
        if not cids:
            cids = [PAD]
        if len(pids) + len(cids) > a.maxlen:
            pids = pids[-(a.maxlen - len(cids)):]
        seqs.append(pids + cids)
        plens.append(len(pids))
    L = max(len(s) for s in seqs)
    B = len(seqs)
    ids = torch.full((B, L), PAD, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s)
        att[i, L - len(s):] = 1
    with torch.inference_mode():
        out = model(input_ids=ids.to(dev), attention_mask=att.to(dev))
        logits = out.logits.float()
        logp = torch.log_softmax(logits, dim=-1)
    res = []
    for i, pl in enumerate(plens):
        start = L - len(seqs[i])
        cs = start + pl
        if cs >= L:
            res.append(float("-inf"))
            continue
        idx = torch.arange(cs, L)
        res.append(logp[i, (idx - 1).to(dev), ids[i, idx].to(dev)].sum().item())
    return res


def enc(s):
    return tok(s, add_special_tokens=False).input_ids


def evaluate(name, items):
    items = items[: a.limit]
    correct = 0
    t0 = time.time()
    pairs = []
    metas = []
    for it in items:
        pids = enc(it["prompt"])
        for oi, opt in enumerate(it["options"]):
            pairs.append((pids, enc(opt)))
            metas.append((it["gold"], oi))
    if not pairs:
        return {"n": 0, "acc": 0.0}
    scores = []
    for i in range(0, len(pairs), a.batch):
        scores.extend(score_pairs(pairs[i:i + a.batch]))
    nopt = [len(it["options"]) for it in items]
    pos = 0
    for it, k in zip(items, nopt):
        ss = scores[pos:pos + k]
        pred = max(range(k), key=lambda j: ss[j])
        correct += int(pred == it["gold"])
        pos += k
    acc = correct / len(items)
    print(f"[{name}] n={len(items)} acc={acc:.4f} ({correct}/{len(items)}) {time.time()-t0:.0f}s", flush=True)
    return {"n": len(items), "acc": acc}


B = {}


def load_bench(name):
    if name == "percor":
        d = load_dataset("MCINext/PerCoR", split="validation")
        out = []
        for r in d:
            out.append({"prompt": r["text_1"] + " " + r["text"],
                        "options": list(r["choices"]), "gold": int(r["true_choice"]) - 1})
        return out
    if name == "khayyam":
        d = load_dataset("raia-center/khayyam-challenge", "default", split="train")
        out = []
        for r in d:
            if r["Key"] is None:
                continue
            out.append({"prompt": r["Question Body"],
                        "options": [r["Choice 1"], r["Choice 2"], r["Choice 3"], r["Choice 4"]],
                        "gold": int(float(r["Key"])) - 1})
        return out
    if name == "percul":
        d = load_dataset("teias-ai/percul", split="original_persian")
        out = []
        for r in d:
            out.append({"prompt": r["Story"],
                        "options": [r["Choice 1"], r["Choice 2"], r["Choice 3"], r["Choice 4"]],
                        "gold": int(r["Correct Choice"]) - 1})
        return out
    if name == "parsinlu_mc":
        d = load_dataset("PartAI/ParsiNLU-multiple-choice", split="test")
        out = []
        for r in d:
            out.append({"prompt": r["context"] + "\n" + r["question"],
                        "options": [r["ending0"], r["ending1"], r["ending2"], r["ending3"]],
                        "gold": int(r["label"])})
        return out
    if name == "parsinlu_entail":
        d = load_dataset("ParsBench/parsinlu-entailment-alpaca-style", split="train")
        opts = ["c", "e", "n"]
        out = []
        for r in d:
            o = r["output"].split(":")[-1].strip()
            if o not in opts:
                continue
            out.append({"prompt": r["instruction"] + "\n" + r["input"], "options": opts,
                        "gold": opts.index(o)})
        return out
    if name == "medqa":
        d = load_dataset("MohammadJRanjbar/PersianMedQA", split="test")
        out = []
        for r in d:
            ans = r["answer"]
            if isinstance(ans, str):
                ans = ast.literal_eval(ans)
            ordered = [ans[k] for k in sorted(ans.keys(), key=lambda k: int(k))]
            gold = int(r["correct answer"]) - 1
            if not (0 <= gold < len(ordered)):
                continue
            out.append({"prompt": r["question"], "options": ordered, "gold": gold})
        return out
    if name == "musr":
        out = []
        for sub in ["murder_mysteries", "object_placements", "team_allocation"]:
            try:
                d = load_dataset("ParsBench/Persian-MuSR", split=sub)
            except Exception:
                continue
            for r in d:
                out.append({"prompt": r["narrative"] + "\n" + r["question"],
                            "options": list(r["choices"]), "gold": int(r["answer_index"])})
        return out
    raise SystemExit("unknown bench " + name)


results = {}
for name in a.benches.split(","):
    name = name.strip()
    try:
        items = load_bench(name)
        if a.data_only:
            ex = items[0] if items else None
            print(f"[data] {name}: n={len(items)} gold={ex['gold']} nopts={len(ex['options'])} "
                  f"prompt[:60]={ex['prompt'][:60]!r}", flush=True)
            results[name] = {"n": len(items)}
        else:
            results[name] = evaluate(name, items)
    except Exception as e:
        import traceback
        print(f"[{name}] ERROR {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        results[name] = {"error": f"{type(e).__name__}: {e}"}

os.makedirs("/marimo/results", exist_ok=True)
with open(f"/marimo/results/{a.tag}.json", "w") as f:
    json.dump({"model": a.model, "adapter": a.adapter, "results": results}, f, ensure_ascii=False, indent=2)
print("RESULTS", json.dumps(results, ensure_ascii=False), flush=True)
print("EVAL_DONE", a.tag, flush=True)
