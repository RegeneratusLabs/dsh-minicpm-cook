# What OpenBMB publishes today

English · [中文](zh/whats-available.md)

What exists, what it covers, and where the gaps are. Read this first if you are new to the family.

---

**Yes — several, as of November 2026. None of them is a catalog, and none is built
for an agent to act on.** This document records what exists, what it covers, and
the gap this project fills.

## What exists

| Surface | What it is | Coverage | Format |
| --- | --- | --- | --- |
| [`openbmb.cn/docs`](https://openbmb.cn/docs/) — "MiniCPM Wiki" | Official knowledge base, Chinese with an `/en/` mirror | Per-model fine-tuning guides, hardware tables, deployment | Web prose |
| [`OpenBMB/MiniCPM` `docs/finetune/`](https://github.com/OpenBMB/MiniCPM/tree/main/docs/finetune) | 5 files: `trl.md`, `llamafactory.md`, `ms_swift.md`, `unsloth.md`, `xtuner.md` | Fine-tuning for MiniCPM5 | Markdown with runnable snippets |
| [`OpenBMB/MiniCPM` `docs/deployment/`](https://github.com/OpenBMB/MiniCPM/tree/main/docs/deployment) | 10 files | transformers, vLLM, SGLang, llama.cpp, Ollama, LM Studio, MLX, ArcLight, LiteRT-LM, vLLM-Ascend | Markdown |
| [`OpenBMB/MiniCPM` `skills/`](https://github.com/OpenBMB/MiniCPM/tree/main/skills) | **18 Agent Skills** | `minicpm5-finetune-{trl,llamafactory,ms-swift,unsloth,xtuner,gguf-lora}`, `minicpm5-deploy-{vllm,sglang,transformers,llama-cpp,ollama,lmstudio,mlx,arclight,litert,vllm-ascend}`, plus index skills | `SKILL.md` + frontmatter |
| [`OpenSQZ/MiniCPM-V-CookBook`](https://github.com/OpenSQZ/MiniCPM-V-CookBook) | A cookbook for V and o, developed by OpenBMB **and** OpenSQZ | Inference, fine-tuning, quantization, web demo, iOS | Repo + [docs site](https://opensqz.github.io/MiniCPM-V-CookBook/) |
| [vLLM recipes](https://recipes.vllm.com.cn/openbmb/MiniCPM-V-4.6) | Per-model launch pages | vLLM only | Markdown |
| [SGLang cookbook](https://github.com/sgl-project/sglang/tree/main/docs_new/cookbook/autoregressive/OpenBMB) | Per-model launch pages | SGLang only | Markdown |
| [minicpm-o.readthedocs.io](https://minicpm-o.readthedocs.io/en/latest/index.html) | Documentation site for o | Omni models | ReadTheDocs |

There is **no** repository named like `MiniCPM-cookbook` or `MiniCPM-recipes` from
OpenBMB covering the whole family, and no official "cookbook" analogue to Meta's
`llama-cookbook` for the text models.

## The important finding: upstream already uses SKILL.md

OpenBMB's `skills/` directory contains skills in exactly the format the DeepSeek
Harness reads: a `SKILL.md` per directory with `name` and `description`
frontmatter, a decision table, steps, verification expectations, pitfalls, and a
link back to the corresponding `docs/` page.

That changes the right design for a DSH plugin. **Vendoring them would be a
mistake** — they are versioned per model, they change monthly, and a stale copy
silently teaches the wrong template. The bundled skills here instead cover the
cross-model questions upstream cannot answer per model, and the README tells you
how to mount upstream's corpus alongside them.

## The five gaps this project fills

1. **No cross-model catalog.** Every surface is organised per model or per
   framework. Answering "which 8B vision model has a vLLM path and a permissive
   licence" means reading four repositories.
2. **No compatibility gate.** Requirements are prose scattered across pages:
   `transformers>=5.6` for MiniCPM5, `>=5.7.0` for V4.6, `trl>=0.21`, `vLLM>=0.21`,
   `sglang>=0.5.16`. Nothing compares an installed environment against them.
3. **No validation.** Dataset schema errors, unexpanded image globs, records that
   end on a user turn, and silent token overflow are all discovered at training
   time, after the GPU bill.
4. **No distribution layer.** Nothing upstream covers model cards, NOTICE files,
   adapter-versus-merged decisions, or Hub/ModelScope upload for a *derivative*.
   Apache-2.0 makes redistribution easy, but only if the attribution is right.
5. **No drift detection.** Five releases in 2026 (SALA in February, MiniCPM5-1B in
   May, V4.6 in May, MiniCPM5-2B in September, V4.7 + OCR in October). Any static
   cookbook is wrong within a quarter.

## What this project does about it

- `data/models.json` — one machine-readable catalog, 17 models, each with the URL
  its facts came from and a `confidence` flag. VRAM is computed, never asserted.
- Seven tools — catalog, plan, recipe, dataset check, host doctor, serve, publish.
- Four skills — the cross-model judgements that a per-model doc cannot express.
- `scripts/check-upstream.mjs` — turns upstream drift into a failing check instead
  of into quietly wrong advice.

## What is deliberately not claimed

- No training or serving was executed while building this. The host has no GPU and
  no PyTorch; recipes are verified by parsing and by schema assertions only.
- Five of seventeen catalog entries are marked `unverified`, with a note on what
  to confirm. An unverified licence blocks publication by design.
