---
name: minicpm-finetune
description: Use when fine-tuning any OpenBMB MiniCPM model, including LoRA, QLoRA, full SFT, or DPO, when choosing between TRL, LLaMA-Factory, ms-swift, Unsloth or XTuner for MiniCPM, when preparing or debugging a MiniCPM training dataset, or when a MiniCPM fine-tune is failing to converge or losing capability.
---

# Fine-tuning a MiniCPM

Plan first, then generate, then validate. Do not hand-write a config before
knowing the template, the memory budget and the dataset shape.

```
minicpm_plan(modelId: "...", goal: "lora", gpuCount: 1, vramGb: 24)
minicpm_check_dataset(path: "data/train.jsonl", modelId: "...")
minicpm_recipe(modelId: "...", toolchain: "trl", method: "lora")
```

## Choosing a framework

| Situation | Use |
| --- | --- |
| Text model, want minimal Python, no YAML | **TRL + PEFT** |
| Text or VLM, want YAML and a WebUI | **LLaMA-Factory** |
| Text or VLM, want a CLI that runs out of the box, or DPO | **ms-swift** |
| Any VLM | **LLaMA-Factory or ms-swift** — there is no documented TRL VLM path for MiniCPM-V/o |
| DPO / KTO / ORPO | ms-swift or TRL |

The adapter format (`adapter_model.safetensors` + `adapter_config.json`) is
identical across all three, so the choice is not a commitment.

## Dataset shapes

Text SFT (messages):

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

LLaMA-Factory alpaca:

```json
{"instruction": "...", "input": "...", "output": "..."}
```

VLM (LLaMA-Factory / ms-swift):

```json
{"messages": [{"role": "user", "content": "<image>\nWhat is this?"},
              {"role": "assistant", "content": "A cat."}],
 "images": ["/abs/path/cat.jpg"]}
```

DPO:

```json
{"instruction": "...", "input": "...", "chosen": "...", "rejected": "...", "history": [["q", "a"]]}
```

KTO scores rather than compares — `{"messages": [...], "label": true}` per record.

Run `minicpm_check_dataset` before training. It catches the four failures that
cost the most time: no assistant turn (record trains nothing), unexpanded
`*.jpg` globs (loader reads a literal path), records ending on a user turn, and
silent token overflow.

## The settings that actually matter

**Assistant-only loss.** By default the loss includes the prompt, which dilutes
signal and often makes loss *rise* on a small adapter. In TRL, set
`assistant_only_loss=True` **and** patch the chat template to wrap assistant
content in `{% generation %}` blocks. The generated TRL recipe does both.

**Never save the training template.** The patched template exists only to place
the generation mask. Saving it with the tokenizer strips think-mode and tool-call
rendering at inference. Reload the original tokenizer when you merge or serve.

**Do not pack.** OpenBMB documents packed training as broken with the current
`transformers` on these models. `packing=False` / `--packing false`.

**VLM: `freeze_vision_tower=false`.** Freezing the vision tower is the most
common reason a VLM fine-tune learns the answer style but loses grounding — it
can still talk about images, just not about *the* image.

**VLM: pick the visual token budget.** `DOWNSAMPLE_MODE=4x` keeps detail and is
right for OCR/counting; `16x` cuts visual tokens roughly fourfold at a measurable
accuracy cost. It is an environment variable, not a YAML key.

**VLM: coordinates are normalised to 0–1000.** Point, box and grounding targets
must be rescaled or the model learns a wrong scale.

## Hyperparameter starting points

| Method | LR | Batch × accum | Epochs |
| --- | --- | --- | --- |
| LoRA (text) | 2e-4 | 4 × 4 | 2 |
| QLoRA (text) | 2e-4 | 1 × 8 | 2 |
| Full SFT (text) | 1e-5 | 1 × 8 | 2–3 |
| LoRA / full (VLM) | 5e-6 | 1 × 16 | 3–4 |

LoRA rank 16 / alpha 32 targets `q,k,v,o,gate,up,down` and lands near **1%**
trainable parameters. Raise rank before raising epochs when the model underfits.

## Hardware floors OpenBMB publishes

- MiniCPM4-0.5B: LoRA ≥8 GB; full >12 GB.
- MiniCPM4.1-8B: LoRA ≥2×24 GB; full ≥4×24 GB.
- VLM 4.6/4.7: official walkthroughs use 8 GPUs at batch 1 with ZeRO-2.

`minicpm_plan` returns a computed ESTIMATE alongside these. Treat the official
number as the safe floor and the estimate as the lower bound to verify.

## Merging an adapter

```python
merged = model.merge_and_unload()
merged.save_pretrained("./merged")
tok.save_pretrained("./merged")   # ORIGINAL tokenizer, not the training one
```

For VLMs, merge only the language stack; the vision tower is not a LoRA target
and merging it through a generic path can corrupt the projector.

## When loss will not fall

1. Is `assistant_only_loss` on **and** does the template contain `{% generation %}`?
2. Is the framework template string the one in the catalog (`cpm3` for MiniCPM4.x,
   `minicpm5`, `minicpmv4_6`)? A wrong template trains on the wrong token layout.
3. Is the dataset actually reaching the trainer — check `train_samples` in the first log line.
4. Is the learning rate appropriate for the method? 2e-4 on a full fine-tune destroys the model.
