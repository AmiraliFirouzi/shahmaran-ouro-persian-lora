import ast
from datasets import load_dataset

BENCHES = ["percor", "khayyam", "percul", "parsinlu_mc", "parsinlu_entail", "medqa", "musr"]


def load_bench(name):
    if name == "percor":
        d = load_dataset("MCINext/PerCoR", split="validation")
        return [{"prompt": r["text_1"] + " " + r["text"],
                 "options": list(r["choices"]), "gold": int(r["true_choice"]) - 1} for r in d]
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
        return [{"prompt": r["Story"],
                 "options": [r["Choice 1"], r["Choice 2"], r["Choice 3"], r["Choice 4"]],
                 "gold": int(r["Correct Choice"]) - 1} for r in d]
    if name == "parsinlu_mc":
        d = load_dataset("PartAI/ParsiNLU-multiple-choice", split="test")
        return [{"prompt": r["context"] + "\n" + r["question"],
                 "options": [r["ending0"], r["ending1"], r["ending2"], r["ending3"]],
                 "gold": int(r["label"])} for r in d]
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
