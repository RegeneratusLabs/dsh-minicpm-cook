# Publishing a fine-tune

English · [中文](zh/publishing.md)

What to ship, what the licence requires, and the mistakes that are expensive to fix after release.

---

This is the part of the MiniCPM lifecycle with the least upstream coverage and the
highest consequence of getting it wrong.

## 1. The licences

| Repository | Statement | Source |
| --- | --- | --- |
| `OpenBMB/MiniCPM` | "This repository and MiniCPM models are released under the Apache-2.0 License." | [README § LICENSE](https://github.com/OpenBMB/MiniCPM/blob/main/README.md) |
| `OpenBMB/MiniCPM-V` | "Code and model weights are released under Apache-2.0." | [README](https://github.com/OpenBMB/MiniCPM-V/blob/main/README.md) |
| `OpenSQZ/MiniCPM-V-CookBook` | Apache-2.0 | [LICENSE](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/LICENSE) |

Apache-2.0 means, concretely:

- **Commercial use is permitted.**
- **Redistribution of weights and derivatives is permitted.**
- **Attribution is required**, and the `NOTICE` file must be preserved.
- **Modified files must be marked.** Fine-tuning is a modification; say so.
- **No trademark or endorsement rights** are granted.

The catalog records the licence per model with the URL it was read from, plus a
`licenseVerified` flag. A model whose licence has not been read against its own
`LICENSE` is treated as **not redistributable** by `minicpm_publish`, because the
cost of being wrong is borne by whoever receives the weights.

## 2. Adapter vs merged vs GGUF

| Artifact | Size (8B, r=16) | Pros | Cons |
| --- | --- | --- | --- |
| LoRA adapter | ~100–200 MB | Small; base provenance explicit; disableable; several adapters per base | Consumer needs `peft`; not loadable by every engine |
| Merged weights | ~16 GB (bf16) | Works anywhere a plain model works | Duplicates the base; provenance only visible in the card |
| GGUF | ~4–8 GB (Q4) | Runs on CPU/edge | One file per quant level; vision projector handling adds risk |

Default to the adapter. Publish merged weights when the audience genuinely cannot
run PEFT, and say so in the card. Do not publish a second GGUF when an official one
exists upstream — it splits users across two artefacts with different bugs.

## 3. Required artifacts

### `LICENSE`

The verbatim Apache-2.0 text. Do not paraphrase, truncate, or link in place of it:

```bash
curl -fsSL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE
```

### `NOTICE`

```text
MiniCPM5-2B derivative
Copyright 2026 <author>

This product includes model weights derived from MiniCPM5-2B
(openbmb/MiniCPM5-2B), developed by OpenBMB, THUNLP (Tsinghua University)
and the Gaoling School of Artificial Intelligence at Renmin University of
China, and licensed under Apache-2.0.

Original repository: https://github.com/OpenBMB/MiniCPM
Modifications: fine-tuned on additional data. See README.md.
```

### `README.md` frontmatter

```yaml
license: apache-2.0
base_model: openbmb/MiniCPM5-2B
library_name: peft          # transformers for merged weights
pipeline_tag: text-generation
tags:
  - minicpm
  - lora
  - text-generation
```

`base_model` must name the real base. For a verified handoff, pin the revision in
the body — `main` is a moving target and a consumer loading the model a year later
should get what you evaluated.

## 4. The model card sections that carry weight

- **What this is** — base model, artifact kind, licence.
- **Training** — method, hardware, sequence length, epochs, learning rate, base revision.
- **Data** — source, size, languages, licence. If it cannot be released, say so explicitly; a consumer needs to know whether the behaviour is reproducible.
- **Evaluation** — a table of base / fine-tune / delta with measured numbers.
- **Limitations** — observed failure modes. This is the section that prevents an incident.
- **Usage** — a runnable snippet, including how to load the adapter and which tokenizer to use.

## 5. The tokenizer trap, restated because it is easy to ship wrong

If training used the assistant-only-loss chat-template patch, that template is a
training artefact. It must not be saved with the tokenizer and must not ship.
Consumers must load the **original** tokenizer from the base model. An adapter
trained this way is compatible with the base template, because the token sequence
is identical — but only if the patched template is not the one distributed.

## 6. Upload commands

```bash
# Hugging Face
huggingface-cli login                     # or export HF_TOKEN
huggingface-cli upload <user>/<repo> ./adapter_final . --repo-type model

# ModelScope
modelscope login                          # or export MS_TOKEN
modelscope upload <user>/<repo> . --repo-type model
```

ModelScope ids are case-sensitive and follow `Owner/Model-Name`.

## 7. Release checklist

- [ ] Full `LICENSE` text present
- [ ] `NOTICE` names OpenBMB, THUNLP, ModelBest
- [ ] `base_model` correct, revision pinned
- [ ] Evaluation table has measured numbers
- [ ] Limitations section has observed failure modes
- [ ] Data provenance described, or its absence stated
- [ ] Adapter published rather than merged (or the choice justified in the card)
- [ ] Licence verified for this exact model
- [ ] Tokenizer is the base model's, not the training-patched one
- [ ] Smoke test: load the published artefact from a clean environment and generate once

## 8. Versioning a release

Minimum viable practice:

- Tag the training run: base revision + dataset hash + config file.
- Record the adapter's `adapter_config.json` unchanged; it is the provenance.
- Note the framework versions used (`trl`, `peft`, `transformers`), because a
  future merge with a different `peft` can produce different weights.
- If you retrain, publish a new repo or a new revision — do not overwrite weights
  that someone may have evaluated and approved.
