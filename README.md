---
base_model: ByteDance/Ouro-1.4B
library_name: peft
license: apache-2.0
language:
- fa
pipeline_tag: text-generation
tags:
- persian
- farsi
- ouro
- looped-transformer
- lora
- peft
- sft
datasets:
- amiralifirouzi/tiny-persian-sft-15k
---

<div align="center">
  <img src="./Gemini_Generated_Image_na5ymzna5ymzna5y.jfif" alt="Shahmaran — Persian fine-tuned Ouro-1.4B" width="600">
</div>

# Shahmaran — Ouro-1.4B Persian LoRA (SFT)

**Shahmaran** is a Persian supervised fine-tune of [`ByteDance/Ouro-1.4B`](https://huggingface.co/ByteDance/Ouro-1.4B),
a 1.4B **looped / Universal Transformer** language model, trained with LoRA on
[`amiralifirouzi/tiny-persian-sft-15k`](https://huggingface.co/datasets/amiralifirouzi/tiny-persian-sft-15k).

The repository contains:
- the **merged bf16 model** at the repo root (`model.safetensors` + custom `modeling_ouro.py`), and
- the **LoRA adapter** under [`adapter/`](./adapter).

## What this model is (and is not)

The base Ouro-1.4B is **not instruction-tuned**: it cannot reliably answer multiple-choice
questions in Persian (it does not follow the "output the option number" format). After this
Persian SFT the model becomes **instruction-following and usable**, but it **remains clearly
weaker than 4B-class models** such as Qwen3-4B / Cassandane on the same benchmarks.

So the honest claim is: *"a working 1.4B Persian chat/MCQ model, a large improvement over its own
base"* — **not** *"better than Qwen3-4B"*.

## Evaluation

Zero-shot, **greedy decoding** generation. The prompt lists the options as `1) … N)` in a chat
template and the model's first integer answer is parsed. 500 items per benchmark.

| Benchmark | Ouro-1.4B (base) | **Shahmaran (this, 2 ep)** | Cassandane (Qwen3-4B) | Qwen3-4B (base) |
|---|---:|---:|---:|---:|
| Khayyam-Challenge | 0.012 | **0.304** | 0.336 | 0.212 |
| PerCul | 0.254 | 0.174 | 0.278 | 0.294 |
| ParsiNLU-multiple-choice | 0.028 | **0.330** | 0.444 | 0.408 |
| ParsiNLU-entailment | 0.008 | **0.154** | 0.298 | 0.392 |
| PersianMedQA | 0.014 | **0.220** | 0.364 | 0.390 |
| Persian-MuSR | 0.166 | **0.328** | 0.488 | 0.512 |
| **Mean** | **0.080** | **0.252** | 0.368 | 0.368 |

The fine-tune improves **5 of 6** benchmarks over its own base (mean **0.080 → 0.252**). PerCul
is the only regression (near-random-baseline, high variance). Results are single-run at n=500
and should be treated as indicative, not as a leaderboard.

> Metrics above were produced with a local harness and are **not** directly comparable to
> evaluation numbers obtained with a different protocol (e.g. cloze log-likelihood) or a
> different PerCoR formulation.

## Training

| Parameter | Value |
|---|---|
| Base model | ByteDance/Ouro-1.4B (looped Universal Transformer) |
| Method | LoRA SFT (`peft`) |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | q,k,v,o,gate,up,down `_proj` |
| Trainable params | 15,138,816 (1.04%) |
| Epochs | 2 |
| Global batch | 8 × 4 = 32 |
| Learning rate | 2e-4 (cosine, 3% warmup) |
| Sequence length | 1024 |
| Precision | bf16 (sdpa, fused AdamW, gradient checkpointing) |
| Steps | 924 |
| Final eval loss | ~0.75 |

Assistant-only loss masking is applied manually (Ouro's chat template has no `{% generation %}`
block). `transformers==4.54.1` is required (Ouro needs `<4.56`).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

mid = "amiralifirouzi/Shahmaran-Ouro-1.4B-persian-lora"
tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    mid, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
)

messages = [{"role": "user", "content": "سلام! خودت را کوتاه معرفی کن."}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")

out = model.generate(
    ids,
    max_new_tokens=128,
    do_sample=False,
    use_cache=False,                     # see note below
    attention_mask=torch.ones_like(ids),
)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

> **`use_cache=False` is required.** With `transformers>=4.5x` the cached-decoding path of
> Ouro's custom `modeling_ouro.py` raises
> `AttributeError: property 'key_cache' of 'UniversalTransformerCache' object has no setter`.
> This affects the **base model too** and is unrelated to this fine-tune.

## Limitations

- 1.4B parameters: factual accuracy is limited; expect hallucinations (e.g. wrong capitals).
- Repetition/degeneration can appear on open-ended prompts under greedy decoding; use a
  repetition penalty or sampling for free-form generation.
- Trained on ~15K Persian SFT samples for 2 epochs — not an encyclopedic model.
- Below Qwen3-4B-class models on every benchmark.
- Benchmarks are log-probability-free generation accuracy at n=500, single seed.

## Links

- **Base:** [`ByteDance/Ouro-1.4B`](https://huggingface.co/ByteDance/Ouro-1.4B)
- **Dataset:** [`amiralifirouzi/tiny-persian-sft-15k`](https://huggingface.co/datasets/amiralifirouzi/tiny-persian-sft-15k)
- **Reference Persian model:** [`amiralifirouzi/Cassandane-qwen3-4b-persian-lora`](https://huggingface.co/amiralifirouzi/Cassandane-qwen3-4b-persian-lora)

Apache-2.0.
