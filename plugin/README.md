# MiniCPM cook

A harness bundle that turns the OpenBMB MiniCPM documentation into something an
agent can act on: seven tools and four skills covering choose → plan → tune →
validate → serve → publish.

## What it is not

It does not train, download, or deploy anything itself. Every tool is a pure
function of its arguments plus a bundled catalog, so nothing can run a GPU job or
write to your disk without you asking for it. Execution still goes through the
harness's own shell and file tools.

## Tools

| Tool | Answers |
| --- | --- |
| `minicpm_catalog` | Which models exist, what they cost, what they support |
| `minicpm_plan` | Which framework and method fit your GPUs, and in what order |
| `minicpm_recipe` | A runnable TRL / LLaMA-Factory / ms-swift config |
| `minicpm_check_dataset` | Is this JSONL actually trainable |
| `minicpm_doctor` | Does this host meet the version and VRAM floors |
| `minicpm_serve` | The launch command, memory estimate and smoke test |
| `minicpm_publish` | Model card, NOTICE, licence gate, upload commands |

## Skills

`minicpm-family`, `minicpm-finetune`, `minicpm-serve`, `minicpm-publish`.

## Data

`data/models.json` is the single source of truth. It records, per model: ids on
Hugging Face and ModelScope, parameter count, modality, context length, licence
**with the URL it was read from**, per-framework minimum versions and template
strings, and a `confidence` flag. Entries whose facts could not be confirmed are
marked `unverified` and say what needs checking.

VRAM figures are never stored as facts. `lib/estimate.js` computes them from an
explicit formula and every surface labels the result an ESTIMATE.

## Installing

The bundle is a local directory. In the harness UI use **Plugins → Add plugin**
and point it at this directory, or install it with the plugin manager's
`install_bundle` action and this absolute path as the target.

After installing, open a new session and ask for something in scope, for example:

> fine-tune MiniCPM5-2B for SQL generation on one 24 GB GPU

The agent should call `minicpm_plan`, then `minicpm_recipe`, then
`minicpm_check_dataset`.

## Verifying

```bash
node --test plugin/test/            # unit and integration tests
node scripts/validate-data.mjs      # catalog schema and referential integrity
node scripts/check-upstream.mjs     # drift against OpenBMB's repository
```

## Licence

The bundle's code is Apache-2.0. Model weights are not distributed here; the
catalog records each upstream model's licence instead.
