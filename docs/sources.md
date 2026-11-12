# Sources

Every claim in `docs/` traces to one of these. Accessed 2026-11-12.

## Primary: OpenBMB repositories

| Source | Used for |
| --- | --- |
| https://github.com/OpenBMB/MiniCPM | Model family, release dates, architecture, deployment commands, licence |
| https://raw.githubusercontent.com/OpenBMB/MiniCPM/main/README.md | MiniCPM5-1B/2B, SALA, 4.1, 4, 3, legacy tables; sampling parameters; vLLM/SGLang/CPM.cu/llama.cpp/Ollama commands |
| https://github.com/OpenBMB/MiniCPM/tree/main/docs/deployment | 10 backend pages |
| https://github.com/OpenBMB/MiniCPM/tree/main/docs/finetune | `trl.md`, `llamafactory.md`, `ms_swift.md`, `unsloth.md`, `xtuner.md` |
| https://raw.githubusercontent.com/OpenBMB/MiniCPM/main/docs/finetune/trl.md | Assistant-only loss, chat-template patch, LoRA config, merge procedure |
| https://github.com/OpenBMB/MiniCPM/tree/main/skills | 18 Agent Skills in SKILL.md format |
| https://raw.githubusercontent.com/OpenBMB/MiniCPM/main/skills/minicpm5-finetune-trl/SKILL.md | Skill structure; dependency pins (`trl>=0.21`, `transformers>=5.6,<6`) |
| https://github.com/OpenBMB/MiniCPM/blob/main/docs/README-legacy.md | BitCPM4, MiniCPM4 applications |
| https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE | Apache-2.0 |
| https://github.com/OpenBMB/MiniCPM-V | V and o families, weights matrix, licence statement |
| https://raw.githubusercontent.com/OpenBMB/MiniCPM-V/main/README.md | V4.7 35B-A3B/1B, o 4.5, quantized variants, DSpark, LES, SRFT |
| https://github.com/OpenBMB/MiniCPM-OCR | OCR suite (Oct 2026) |
| https://github.com/OpenBMB/CPM.cu | Sparse-attention inference runtime |
| https://github.com/OpenBMB/infllmv2_cuda_impl | InfLLM-V2 kernels for sparse inference |
| https://github.com/OpenBMB/llama.cpp-omni | Lookahead Expert Scheduling for on-device MoE |

## Primary: OpenBMB documentation site

| Source | Used for |
| --- | --- |
| https://openbmb.cn/docs/ | Knowledge-base index (CN; `/en/` mirrors) |
| https://openbmb.cn/docs/en/guides/minicpm-text/minicpm4.1-and-minicpm4/finetune | Hardware floors, official finetune script, LoRA/QLoRA flags, LLaMA-Factory `cpm3` template |
| https://openbmb.cn/docs/en/guides/minicpm-v/minicpm-v-4.6/finetune | V4.6 LLaMA-Factory and ms-swift walkthroughs, `DOWNSAMPLE_MODE`, coordinate normalisation, dependency pins |
| https://openbmb.cn/docs/en/guides/minicpm-v/minicpm-v-4/finetune | V4 finetune guide |
| https://openbmb.cn/docs/guides/minicpm-text/minicpm-sala/finetune | SALA LoRA/DeepSpeed |
| https://openbmb.cn/en/blog/self-replay-fine-tuning | Self-Replay Fine-Tuning (V4.7) |

## Cookbooks and framework documentation

| Source | Used for |
| --- | --- |
| https://github.com/OpenSQZ/MiniCPM-V-CookBook | V/o cookbook; deployment, finetune, quantization, demo recipes |
| https://opensqz.github.io/MiniCPM-V-CookBook/ | Rendered cookbook |
| https://recipes.vllm.com.cn/openbmb/MiniCPM-V-4.6 | vLLM recipe page |
| https://github.com/sgl-project/sglang/tree/main/docs_new/cookbook/autoregressive/OpenBMB | SGLang cookbook pages |
| https://github.com/hiyouga/LLaMA-Factory | Templates, `dataset_info.json` schema |
| https://github.com/modelscope/ms-swift | `--model_type`, dataset formats, `swift rlhf` |
| https://minicpm-o.readthedocs.io/en/latest/index.html | Omni documentation site |
| https://ollama.com/library/minicpm-v4.6 | Official Ollama tag |
| https://ollama.com/openbmb/minicpm4.1 | Official Ollama tag |

## Papers

| Source | Used for |
| --- | --- |
| https://arxiv.org/abs/2506.07900 | MiniCPM4 technical report |
| https://arxiv.org/abs/2509.18154 | MiniCPM-V 4.5: architecture, data and training recipe |
| https://arxiv.org/abs/2604.27393 | MiniCPM-o 4.5: real-time full-duplex omni-modal interaction |
| https://arxiv.org/abs/2509.24663 | InfLLM-V2 sparse attention |
| https://arxiv.org/pdf/2602.09003 | UltraData tiered data management |
| https://arxiv.org/abs/2408.01800 | MiniCPM-V: a GPT-4V level MLLM on your phone |

## Datasets referenced by the official walkthroughs

| Source | Used for |
| --- | --- |
| https://huggingface.co/datasets/allenai/pixmo-count | The counting task used in the V4.6 LLaMA-Factory / ms-swift walkthroughs |
| https://huggingface.co/datasets/sqrti/SPA-VL | The safety-alignment pairs used in the V4.6 DPO walkthrough |
| https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609 | Agent SFT data released with MiniCPM5-2B |
| https://huggingface.co/datasets/openbmb/UltraData-RL-2609 | RL data released with MiniCPM5-2B |

## Installed harness references (for the plugin's shape, not for MiniCPM facts)

| Source | Used for |
| --- | --- |
| `@deepseek-ai/dsh-agent-preset/skills/dsh-doc` | Bundle manifest, Host plugin exports, display metadata |
| `@deepseek-ai/dsh-skill` type definitions | `ctx.skills.register()` contract |
| `@deepseek-ai/dsh-tools` type definitions | `ToolDefinition` shape, `render`/`execute` contract |
| `@deepseek-ai/dsh-shell` type definitions | `ctx.shell.resolve()/execute()/result()` |
| `/home/josh/dsh-plugins/dsh-web-search` | Reference for a dependency-free Host bundle |

## How to re-verify

```bash
node scripts/check-upstream.mjs     # has OpenBMB's repo moved?
node scripts/validate-data.mjs      # is the catalog still internally consistent?
node scripts/render-catalog.mjs     # regenerate docs/models.md
```

Anything the catalog could not confirm is marked `confidence: "unverified"` with a
`confidenceNote` saying what to check. Five entries carry that flag; none of them
is used to justify a redistribution decision.
