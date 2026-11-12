---
name: minicpm-publish
description: Use when distributing, publishing or releasing a fine-tuned MiniCPM model or LoRA adapter, when writing a Hugging Face or ModelScope model card for a MiniCPM derivative, when checking whether MiniCPM licence terms permit redistribution or commercial use, or when deciding between publishing an adapter and merged weights.
---

# Publishing a MiniCPM fine-tune

```
minicpm_publish(modelId: "...", repoId: "alice/minicpm5-2b-sql", author: "...",
                evalResults: [{"benchmark": "task-accuracy", "baseScore": 0.61, "score": 0.84}])
```

The tool returns the model card, the NOTICE, the licence instructions, the upload
commands and a checklist. It returns `blocked: true` and withholds the upload
commands when the catalogue records the base licence as unverified — treat that
as a gate, not a suggestion.

## Adapter or merged?

**Default to the adapter.** It is smaller, makes the base licence obvious, keeps
the provenance legible, and consumers who want merged weights can merge in one
line. It also lets a consumer disable your behaviour if it misfires.

Publish merged weights only when the audience cannot run PEFT — a mobile app
team, an inference server with no PEFT support — and say so in the card.

Do not publish a second GGUF when an official one exists upstream; you add a
maintenance burden and split users across two artefacts.

## The licence obligations, concretely

Apache-2.0 permits commercial use and redistribution. It requires:

1. **Ship the full licence text** in a file named `LICENSE`. The generator
   deliberately does not paste the text; fetch it from
   `https://www.apache.org/licenses/LICENSE-2.0.txt` so it is verbatim and complete.
2. **Ship `NOTICE`** naming OpenBMB, THUNLP and ModelBest as the origin of the
   base weights.
3. **State your changes.** Fine-tuning is a modification; say what you changed.
4. **Do not imply endorsement** by OpenBMB for your derivative.

## Model card: the parts that matter

```yaml
license: apache-2.0
base_model: openbmb/MiniCPM5-2B
library_name: peft          # transformers for merged weights
pipeline_tag: text-generation
```

Then, in the body:

- **What this is** — base model, artifact kind, licence.
- **Training** — method, hardware, sequence length, epochs, learning rate. Pin the base revision.
- **Data** — source, size, languages, licence. If it cannot be released, say that explicitly.
- **Evaluation** — a table with base vs fine-tune vs delta. A card with `_not yet measured_` is a claim, not a record.
- **Limitations** — observed failure modes. This is the section that saves a consumer an incident.
- **Usage** — a runnable snippet.

## The tokenizer trap

If you trained with the assistant-only-loss chat-template patch, that template
must **not** be saved with the tokenizer and must not ship. Instruct consumers to
load the original tokenizer from the base model. The generated adapter card says
this; do not delete it when editing.

## Upload

```bash
huggingface-cli login                       # or export HF_TOKEN
huggingface-cli upload <repo> ./adapter_final . --repo-type model

modelscope login                            # or export MS_TOKEN
modelscope upload <repo> . --repo-type model
```

Before the first push, confirm the checklist: `LICENSE` present, `NOTICE`
present, `base_model` correct, evaluation numbers measured, limitations written,
data provenance stated.

## After publishing

Record the adapter in the card and keep the base pinned by revision, not by
`main`. A consumer who loads your model a year from now should get the weights
you evaluated, not whatever the base repository has become.
