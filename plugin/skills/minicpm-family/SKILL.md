---
name: minicpm-family
description: Use when choosing which OpenBMB MiniCPM model to use, when asked what MiniCPM models or versions exist, when checking MiniCPM licences and commercial-use terms, or when a task mentions MiniCPM5, MiniCPM4.1, MiniCPM-V, MiniCPM-o, MiniCPM-SALA or BitCPM and the current release state matters.
---

# Choosing a MiniCPM model

The family is large, ships every few months, and the names do not sort
usefully. Resolve the choice from the catalog rather than from memory:

```
minicpm_catalog(detail: "summary")
minicpm_catalog(family: "minicpm-v", modality: "vision")
minicpm_catalog(maxParamsB: 8, framework: "vllm")
```

## What each line is for

| Line | Sizes | Reach for it when |
| --- | --- | --- |
| MiniCPM5 | 1B, 2B | On-device or CPU-adjacent assistant, coding/tool agent, long context. Standard `LlamaForCausalLM`, so **no `trust_remote_code`** and no custom kernels. The easiest target to fine-tune. |
| MiniCPM-SALA | 8B | Million-token context. Sparse+linear hybrid; needs the OpenBMB SGLang fork and CUDA kernels for the sparse path. Not a casual fine-tuning target. |
| MiniCPM4.1 / 4 | 0.5B, 8B | Trainable sparse attention with hybrid reasoning (`enable_thinking`). 4.1 supersedes 4 unless you are reproducing 4.0 results. Needs `trust_remote_code`. |
| MiniCPM-V 4.7 | 35B-A3B, 1B | Best quality-per-cost multimodal. MoE at 35B-A3B; the 1B is the cheap credible VLM. |
| MiniCPM-V 4.6 | 8B | The best-documented VLM **fine-tuning** path: official LLaMA-Factory and ms-swift walkthroughs, official Ollama tag. |
| MiniCPM-V 2.6 | 8B | Legacy, but the widest third-party ecosystem (quantizations, demos, OCR pipelines). |
| MiniCPM-o | 9B, 8.7B | Omni: continuous video+audio in, speech+text out, full duplex. Different problem class from VLM. |
| MiniCPM-OCR | 8B + verifier | Document parsing with a paired correctness checker. |

## Version facts that change the answer

- **MiniCPM5 needs `transformers>=5.6`**, `vllm>=0.21`, `sglang>=0.5.16`.
- **MiniCPM-V 4.6 needs `transformers>=5.7.0`** — installing it after your training stack silently downgrades it. Always reinstall the pinned version last.
- **MiniCPM4.1 on vLLM drops BOS** in the chat API by default; requests must send `extra_body={"add_special_tokens": true}` or quality degrades with no error.
- **MiniCPM4.1 / 4 only get sparse attention in Transformers and CPM.cu.** vLLM and SGLang run the dense path, which is slower on long context but identical in output.
- **MiniCPM5 uses `min_p=0.0`.** llama.cpp defaults to `0.05`, which traps small models in repetition loops.

## Licences

`MiniCPM` and `MiniCPM-V` both state **Apache-2.0** for code and model weights.
That means commercial use and redistribution are permitted, with attribution and a
copy of the licence.

Before publishing anything derived from a model, confirm the specific entry:

```
minicpm_publish(modelId: "<id>", repoId: "...")
```

The tool returns `compliance.blocked: true` when the catalog records the model's
licence as unverified. Treat that as a stop, not a warning: an unverified licence
is a redistribution risk for whoever receives the weights, not a formality.

## Do not

- Do not pick a model by parameter count alone. A 2B MiniCPM5 beats many 4B models
  on tool use and long context, and a 1B VLM is better than a 4B one for OCR on a phone.
- Do not assume an official GGUF exists for every model. Check `variants` in the
  catalog entry before planning a llama.cpp deployment.
- Do not fine-tune MiniCPM-SALA or BitCPM expecting the standard recipe to carry
  over; both have custom kernels on the inference side.
