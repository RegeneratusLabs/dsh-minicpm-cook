# MiniCPM cook

**English** · [中文](README.zh.md)

A cookbook for OpenBMB's **MiniCPM** model family, built for AI coding agents.

Ask your agent "which MiniCPM should I fine-tune for this, and on this GPU?" and it
answers with a plan, a runnable config, and the licence terms — instead of five
tabs of documentation.

---

## Why this exists

OpenBMB publishes a lot: a documentation site, a fine-tuning guide per model,
deployment pages for ten different inference servers, and a separate cookbook for
the vision models. It is good material.

The problem is that it is organised by model. Your questions are not.

- *Which 8B vision model works with vLLM and lets me redistribute what I train?*
- *Will this fit on one 24 GB card, or do I need four?*
- *Is my training file actually usable, or will it waste six hours?*
- *What has to ship with a fine-tune so the licence is satisfied?*

Answering those means reading across four repositories and remembering which
version of each library supports which model. This project compiles that into one
catalog and seven tools your agent can call.

## What you can ask it

Once installed, these are normal prompts:

> Fine-tune MiniCPM5-2B for SQL generation. I have one RTX 4090.

→ Picks LoRA, estimates ~7 GB of VRAM, emits a TRL script with the correct loss
masking, and lists the version floors you need.

> Is this dataset usable?

→ Checks the JSONL: missing assistant turns, broken role order, image paths that
don't exist, records that silently overflow the context window.

> Serve MiniCPM-V 4.6 on vLLM.

→ Gives you the launch command, a memory estimate for your context length, and a
`curl` to prove it answers.

> Can I publish this fine-tune on Hugging Face?

→ Checks the licence, then generates the model card, the `NOTICE` file and the
upload command — or blocks and tells you what is unverified.

## Install

The plugin is a local folder, so you do not need a package registry.

1. Clone this repository:

   ```bash
   git clone https://github.com/RegeneratusLabs/dsh-minicpm-cook.git
   ```

2. In the DeepSeek Harness, open **Plugins → Add plugin** and select the
   `dsh-minicpm-cook/plugin` folder.

3. Start a new session and ask something in scope.

To install from the command line instead, use the plugin manager's
`install_bundle` action with the absolute path to that folder.

You can also use this as a plain reference: every fact lives in
[`data/models.json`](data/models.json), and the guides in [`docs/`](docs/) stand on
their own.

## The seven tools

| Tool | What it does for you |
| --- | --- |
| `minicpm_catalog` | Lists the family and filters by size, modality, licence or backend |
| `minicpm_plan` | Turns a goal and your GPU into an ordered plan with a memory estimate |
| `minicpm_recipe` | Writes a runnable TRL, LLaMA-Factory or ms-swift config |
| `minicpm_check_dataset` | Finds the problems that would waste your GPU hours |
| `minicpm_doctor` | Checks your machine against a model's requirements |
| `minicpm_serve` | Produces the launch command and a smoke test |
| `minicpm_publish` | Prepares the model card, attribution files and upload commands |

There are also four skills the agent loads on demand — one each for choosing a
model, fine-tuning, serving and publishing.

## Supported models

Seventeen models, from a 1B phone model to a 35B-A3B vision model.

| Family | Sizes | Good for |
| --- | --- | --- |
| MiniCPM5 | 1B, 2B | Small on-device assistants. The easiest to fine-tune |
| MiniCPM4.1 / 4 | 0.5B, 8B | Reasoning at the edge, with sparse attention |
| MiniCPM-SALA | 8B | Very long documents (a million tokens) |
| MiniCPM-V 4.7 / 4.6 | 1B, 8B, 35B-A3B | Images, documents and video |
| MiniCPM-o 4.5 | 9B | Speech and vision together, in real time |
| MiniCPM-OCR | 8B | Document parsing with a correctness checker |

The full table — licence, context length, which servers support it — is in
[`docs/models.md`](docs/models.md).

## Guides

| Guide | What it covers |
| --- | --- |
| [What's available](docs/whats-available.md) | What OpenBMB publishes today, and where the gaps are |
| [Models](docs/models.md) | The full catalog, generated from the data file |
| [Fine-tuning](docs/fine-tuning.md) | Frameworks, dataset formats, settings, failure modes |
| [Deployment](docs/deployment.md) | Serving, quantization, memory arithmetic |
| [Publishing](docs/publishing.md) | Licences, model cards, release checklist |
| [Sources](docs/sources.md) | Every link used, and what it supported |

## Honest limits

- **It does not train or deploy anything itself.** It generates plans and
  configurations; you run them. Nothing touches your GPU unless you ask.
- **Memory numbers are estimates.** They come from a published formula, and every
  screen labels them as estimates. Your real usage will differ.
- **Five of seventeen catalog entries are marked unverified**, each saying what to
  check. Where a licence could not be confirmed, publishing is blocked on purpose.
- **It tracks a fast-moving upstream.** OpenBMB shipped five models in 2026. Run
  `npm run drift` to see whether this repo has fallen behind.

## For contributors

```bash
npm run validate   # catalog schema and integrity
npm test           # 40 tests, no network, no GPU
npm run render     # regenerate docs/models.md from the catalog
npm run drift      # compare against OpenBMB's repository
```

Every fact about a model lives in `data/models.json` and carries the URL it came
from. Add a model by adding an entry; the validator will tell you what is missing.

## Licence

This project is Apache-2.0. It contains no model weights — each model's own licence
is recorded in the catalog.

MiniCPM is built by [OpenBMB](https://github.com/OpenBMB), THUNLP at Tsinghua
University, and ModelBest. This project is an independent cookbook and is not
affiliated with or endorsed by them.
