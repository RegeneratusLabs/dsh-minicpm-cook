---
name: minicpm-serve
description: Use when deploying, serving or running inference with a MiniCPM model or MiniCPM LoRA adapter, when choosing between vLLM, SGLang, llama.cpp, Ollama, LM Studio, MLX or transformers for MiniCPM, when quantizing a MiniCPM model, or when a MiniCPM deployment is out of memory or producing repetitive or degraded output.
---

# Serving a MiniCPM

```
minicpm_serve(modelId: "...", backend: "vllm", gpuCount: 1, vramGb: 24, contextTokens: 8192)
minicpm_serve(modelId: "...", backend: "sglang", toolCalling: true)
minicpm_serve(modelId: "...", backend: "ollama", contextTokens: 4096)
```

The tool emits the exact command, a curl smoke test, and a memory estimate. Run
the smoke test before wiring the endpoint into anything else — a server that
answers is not the same as a server that answers correctly.

## Backend by situation

| Situation | Backend | Why |
| --- | --- | --- |
| Throughput, batching, an OpenAI-compatible API | **vLLM** | Best default for text; shards with `--tensor-parallel-size` |
| Tool calling | **SGLang** | Native `--tool-call-parser minicpm5` produces OpenAI `tool_calls` |
| CPU, laptop, phone | **llama.cpp / Ollama** | GGUF; `--ngl 99` for partial GPU offload |
| Apple Silicon | **MLX** | 4-bit MLX weights published upstream |
| One-off script, debugging | **transformers** | No server, easiest to inspect |

## Memory: know the two numbers

**Weights** dominate and scale with parameter count and precision: 8B at bf16 ≈ 16 GB,
at int4 ≈ 4.5 GB. **KV cache** scales with context and concurrency:
`2 × layers × kv_heads × head_dim × tokens × batch × 2 bytes`.

`minicpm_serve` returns both, per token and total. Two practical consequences:

- Raising `--max-model-len` from 8k to 128k costs roughly 16× the KV memory, even though weights are unchanged.
- Concurrency multiplies KV, not weights. A server sized for one sequence OOMs at twenty.

Set `--gpu-memory-utilization 0.9` and let the engine fail loudly rather than
silently truncating context.

## Quirks that cost quality, not uptime

**MiniCPM4.1 on vLLM needs `extra_body={"add_special_tokens": true}`.** vLLM's
chat API defaults to `False`, so BOS is dropped and the prompt is mis-framed.
Nothing errors; the answers are just worse.

**llama.cpp `min_p`.** Default `0.05` filters tokens below 5% of the top
probability, which for small models removes exactly the tokens that break a
repetition loop. Pass `min_p=0.0`.

**Sampling defaults.** MiniCPM5 recommends `temperature=1.0, top_p=0.95, min_p=0.0`;
add `repetition_penalty=1.05` if output loops. MiniCPM4.1 uses `temperature=0.6,
top_p=0.95` for reasoning.

**Vision models.** The visual token budget is chosen at load time for V4.6/4.7
(`DOWNSAMPLE_MODE`). Serving at `16x` on an OCR workload will lose accuracy that
no prompt can recover.

**SALA has no vLLM path.** It needs the OpenBMB SGLang fork with
`--attention-backend minicpm_flashinfer --dense-as-sparse`, or CPM.cu.

## Quantization

| Format | Use for | Notes |
| --- | --- | --- |
| GGUF (Q4_K_M) | CPU/edge | Official GGUFs exist for most releases; check `variants` first |
| AWQ / GPTQ (W4A16) | GPU serving | Published upstream for 4.7 and 4.1; needs a calibration set if you quantize yourself |
| int4 (bitsandbytes) | Quick local test | Slow at inference; fine for a demo, not for serving |
| FP8 | Hopper+ | Near-lossless, published for V4.7 35B-A3B |

Keep the **vision tower and projector in higher precision** when quantizing a VLM.
Quantizing them is the fastest way to a model that runs and cannot read.

## Serving an adapter

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base, "./adapter_final")
model = model.merge_and_unload()          # then save and serve the merged dir
```

vLLM can serve a PEFT adapter without merging via `--enable-lora --lora-modules`,
which is preferable when you serve one base with several task adapters.

## Regression check before you ship

Serve the base and the fine-tune, run the same 20–50 prompts through both, and
read the diffs. A fine-tune that improved your task but lost basic chat is
over-trained, and the benchmark you ran probably did not notice.
