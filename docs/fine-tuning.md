# Fine-tuning MiniCPM

English · [中文](zh/fine-tuning.md)

Everything needed to go from a dataset to a working adapter, including the settings that decide whether a run converges.

---

Sources are cited inline. Where a number is an estimate rather than a
measurement, it says so.

## 1. Framework matrix

| Framework | Text models | VLM | DPO/KTO | Notes |
| --- | --- | --- | --- | --- |
| TRL + PEFT | ✅ documented | ❌ no documented path | ✅ native | Lowest surface area; assistant-only loss via `SFTConfig(assistant_only_loss=True)` |
| LLaMA-Factory | ✅ `cpm3` / `minicpm5` templates | ✅ `minicpm_v_4_6`, `minicpmv4_6`, `minicpmv4_7` | ✅ | YAML driven; the V4.6 walkthrough is the most complete official fine-tuning doc |
| ms-swift | ✅ | ✅ `--model_type minicpmv4_6` | ✅ (`swift rlhf --rlhf_type dpo`) | CLI driven; handles the visual token budget natively |
| Unsloth | ✅ (MiniCPM5) | — | — | Documented in `docs/finetune/unsloth.md` |
| XTuner | ✅ (MiniCPM5) | — | — | Documented in `docs/finetune/xtuner.md` |
| Official OpenBMB code | ✅ | ✅ | — | `OpenBMB/MiniCPM/finetune/`; DeepSpeed launcher with `--use_lora` / `--qlora` |

Sources: [docs/finetune/](https://github.com/OpenBMB/MiniCPM/tree/main/docs/finetune),
[skills/](https://github.com/OpenBMB/MiniCPM/tree/main/skills),
[MiniCPM4.1 finetune guide](https://openbmb.cn/docs/en/guides/minicpm-text/minicpm4.1-and-minicpm4/finetune),
[MiniCPM-V 4.6 finetune guide](https://openbmb.cn/docs/en/guides/minicpm-v/minicpm-v-4.6/finetune).

## 2. Dataset schemas

Text SFT — the shape OpenBMB's official finetune code consumes:

```json
[{"messages": [
  {"role": "system", "content": "optional"},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}]
```

LLaMA-Factory alpaca:

```json
{"instruction": "...", "input": "...", "output": "..."}
```

LLaMA-Factory / ms-swift VLM:

```json
{"messages": [
  {"role": "user", "content": "<image>\nCarefully observe the image. ..."},
  {"role": "assistant", "content": "<think>\n\n</think>\n\n... <point>236 469</point> ..."}
],
 "images": ["/path/to/img.jpg"]}
```

The V4.6 walkthrough adds two requirements a naive conversion misses:

- **Coordinates are normalised to 0–1000.** Raw pixel values teach a wrong scale.
- **Assistant responses benefit from the non-thinking prefix** `<think>\n\n</think>\n\n`
  for non-reasoning tasks; `<think>\n` for reasoning tasks.

DPO (LLaMA-Factory):

```json
{"instruction": "...", "input": "...", "chosen": "...", "rejected": "...", "history": [["q", "a"]]}
```

KTO scores rather than compares: `{"messages": [...], "label": true}`.

## 3. The settings that decide whether a run converges

### Assistant-only loss

TRL's `assistant_only_loss=True` masks every token outside a `{% generation %}`
block. Without it, loss covers the prompt too, which dilutes signal on small
adapters and can make loss *rise*.

The mechanism requires a training-only chat template. The token sequence is
identical to the model's real template, so the resulting adapter remains
compatible with it — but **the patched template must never be saved with the
tokenizer**. Saving it strips think-mode and tool-call rendering at inference.

Source: [docs/finetune/trl.md](https://github.com/OpenBMB/MiniCPM/blob/main/docs/finetune/trl.md),
[skills/minicpm5-finetune-trl](https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-finetune-trl/SKILL.md).

### Packing

OpenBMB states plainly that packed training is broken with the current
`transformers` on these models. `packing=False` / `--packing false`.

### Vision tower

`freeze_vision_tower: false` (LLaMA-Factory) / `--freeze_vit false` (ms-swift).
Every official VLM walkthrough unfreezes it. A frozen tower is the usual cause of
a model that discusses images generically but not *the* image.

### Visual token budget

V4.6/V4.7 support two compression ratios, `4x` and `16x`, selected by the
`DOWNSAMPLE_MODE` environment variable — not a YAML key. The published counting
benchmark shows the difference: 51.8 → 83.1 (4x, fine-tuned) versus 46.5 → 78.4
(16x, fine-tuned) Acc@0.

## 4. Hyperparameters, as documented

| Setting | MiniCPM-V 4.6 (official walkthrough) | MiniCPM5 TRL (official recipe) |
| --- | --- | --- |
| Method | full | LoRA |
| Learning rate | 5e-6 | 2e-4 |
| Batch × accum | 1 × 16 | 4 × 4 |
| Epochs | 4 | 2 |
| Sequence length | 4096 | 2048 |
| LoRA rank / alpha | n/a (full) | 16 / 32 |
| Target modules | n/a | `q,k,v,o,gate,up,down` |
| Scheduler | cosine, 5% warmup | cosine, 3% warmup |
| Precision | bf16 | bf16 |

MiniCPM4.1's official shell example uses `learning_rate 5e-5`, batch 16,
`model_max_length 1024`, `warmup_steps 100`, `max_steps 1000`, DeepSpeed ZeRO-2
with CPU offload.

Sources: [V4.6 finetune](https://openbmb.cn/docs/en/guides/minicpm-v/minicpm-v-4.6/finetune),
[TRL recipe](https://github.com/OpenBMB/MiniCPM/blob/main/docs/finetune/trl.md),
[MiniCPM4.1 finetune](https://openbmb.cn/docs/en/guides/minicpm-text/minicpm4.1-and-minicpm4/finetune).

## 5. Memory

OpenBMB publishes these floors:

| Model | LoRA | Full |
| --- | --- | --- |
| MiniCPM4-0.5B | ≥8 GB (20-series or newer) | >12 GB |
| MiniCPM4.1-8B | ≥2 × 24 GB | ≥4 × 24 GB |

VLM walkthroughs use 8 GPUs at per-device batch 1 with ZeRO-2.

This project computes its own estimate from an explicit formula
(`plugin/lib/estimate.js`) rather than storing numbers, because memory depends on
sequence length, batch, optimizer, checkpointing and offload policy. The computed
figure is labelled ESTIMATE and the official floor is reported alongside it as the
safe bound.

## 6. Merging

```python
merged = model.merge_and_unload()
merged.save_pretrained(out)
tok.save_pretrained(out)     # the ORIGINAL tokenizer
```

For a VLM, merge only the language stack. The vision tower is not a LoRA target,
and a generic merge path can corrupt the projector.

## 7. Known failure modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `SFTConfig has no attribute 'max_length'` | `trl < 0.21` | Upgrade TRL |
| Loss does not fall | `assistant_only_loss` off, or template has no `{% generation %}` | Apply the patched template |
| Loss rises on a small adapter | Loss computed over prompt tokens | Same as above |
| `ImportError` on model load | Missing `trust_remote_code=True` | Set it; MiniCPM4.x, V and o need it |
| Training silently slow / OOM | Packing enabled | `packing=False` |
| VLM learns style, not grounding | Vision tower frozen | `freeze_vision_tower: false` |
| VLM accuracy collapses on OCR | `DOWNSAMPLE_MODE=16x` | Use `4x` |
| Coordinates wrong after fine-tune | Raw pixels instead of 0–1000 normalisation | Rescale targets |
| `transformers` version reverts | Installing a training stack after the pinned version | Reinstall the pinned `transformers` last |
| DeepSpeed ZeRO-3 hang on small models | ZeRO-3 on a model that fits | Use ZeRO-2 |

## 8. Evaluation

Official benchmark suites used in the model cards: C-Eval, CMMLU, MMLU, GSM8K,
HumanEval for text; MMMU, OCRBench, MathVista for vision.

For a fine-tune, public benchmarks are a smoke test, not evidence. The practical
protocol:

1. Freeze 50–200 held-out prompts from your own distribution **before** training.
2. Score base and fine-tune with identical decoding parameters.
3. Read the worst 10 output diffs by hand — the score hides regressions that users notice.
4. Re-run 20 general instructions to catch catastrophic forgetting.
