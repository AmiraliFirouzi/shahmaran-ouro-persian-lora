import os, sys, json, time
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

MODEL_ID = os.environ.get("MODEL_ID", "ByteDance/Ouro-1.4B")
DATASET_ID = os.environ.get("DATASET_ID", "amiralifirouzi/tiny-persian-sft-15k")
OUT = os.environ.get("OUT", "/marimo/ouro-lora")
MAX_LEN = int(os.environ.get("MAX_LEN", "1024"))
EPOCHS = float(os.environ.get("EPOCHS", "1"))
BS = int(os.environ.get("BS", "8"))
GA = int(os.environ.get("GA", "4"))
LR = float(os.environ.get("LR", "2e-4"))
EVAL_N = int(os.environ.get("EVAL_N", "200"))
GC = os.environ.get("GC", "1") == "1"
SEED = 42

print("[cfg]", json.dumps(dict(MODEL_ID=MODEL_ID, DATASET_ID=DATASET_ID, MAX_LEN=MAX_LEN,
      EPOCHS=EPOCHS, BS=BS, GA=GA, LR=LR, EVAL_N=EVAL_N, GC=GC)), flush=True)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

ASSIST_HEADER = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
print("[tok] assist_header_ids", ASSIST_HEADER, "im_end_id", IM_END, flush=True)


def build(msgs):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(ids)
    h = len(ASSIST_HEADER)
    i = 0
    while i <= len(ids) - h:
        if ids[i:i + h] == ASSIST_HEADER:
            j = i + h
            while j < len(ids) and ids[j] != IM_END:
                j += 1
            end = min(j, len(ids) - 1)
            for k in range(i + h, end + 1):
                labels[k] = ids[k]
            i = j + 1
        else:
            i += 1
    return {"input_ids": ids, "labels": labels}


def truncate(x):
    x["input_ids"] = x["input_ids"][:MAX_LEN]
    x["labels"] = x["labels"][:MAX_LEN]
    return x


ds = load_dataset(DATASET_ID, split="train")
ds = ds.filter(lambda x: any(m["role"] == "assistant" for m in x["messages"]), desc="filter-assistant")
ds = ds.select_columns(["messages"])
ds = ds.map(lambda x: build(x["messages"]), remove_columns=["messages"], desc="tokenize")
ds = ds.filter(lambda x: max(x["labels"]) != -100, desc="drop-empty")
ds = ds.map(truncate, desc="truncate")
lens = [len(x["input_ids"]) for x in ds]
sup = [sum(1 for t in x["labels"] if t != -100) for x in ds]
print(f"[data] n={len(ds)} len_mean={sum(lens)/len(lens):.0f} len_p95={sorted(lens)[int(len(lens)*0.95)]} "
      f"sup_mean={sum(sup)/len(sup):.0f}", flush=True)
_i = next(i for i, s in enumerate(sup) if s > 20)
_sp = [t for t in ds[_i]["labels"] if t != -100]
print("[data] sample supervised span:", repr(tok.decode(_sp[:60]))[:300], flush=True)

ds = ds.train_test_split(test_size=min(EVAL_N, max(1, len(ds) // 20)), seed=SEED)
train_ds, eval_ds = ds["train"], ds["test"]
print(f"[data] train={len(train_ds)} eval={len(eval_ds)}", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": 0},
    attn_implementation="sdpa",
)
model.config.use_cache = False
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=OUT,
    per_device_train_batch_size=BS,
    gradient_accumulation_steps=GA,
    per_device_eval_batch_size=BS,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.0,
    bf16=True,
    logging_steps=10,
    logging_first_step=True,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=500,
    save_total_limit=2,
    gradient_checkpointing=GC,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    seed=SEED,
    dataloader_num_workers=0,
    optim="adamw_torch_fused",
    tf32=True,
)

collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                  data_collator=collator)

print("[train] start", time.strftime("%H:%M:%S"), flush=True)
trainer.train()
print("[train] done", time.strftime("%H:%M:%S"), flush=True)

model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("[save] adapter ->", OUT, flush=True)
for f in os.listdir(OUT):
    print("  ", f, flush=True)
print("ALL_DONE", flush=True)
