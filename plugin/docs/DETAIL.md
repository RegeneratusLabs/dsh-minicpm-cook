# MiniCPM cook — plugin detail

## Why this exists

OpenBMB publishes a lot: a docs site, per-model fine-tuning guides, deployment
pages for ten backends, sixteen Agent Skills, and a separate vision cookbook.
None of it is a catalog. The facts a user needs before starting — which
framework version supports this model, what template string it wants, whether
its licence allows redistribution — are spread across pages that are organized
per model rather than per question.

This bundle answers the questions in the order they come up.

## The shape of a run

1. **Choose.** `minicpm_catalog` filters the family by size, modality, licence, or
   backend support. The result is one line per model, with the backend that
   actually works for it.
2. **Plan.** `minicpm_plan` resolves a model (by id, or from a family and
   modality), estimates memory for the stated GPUs, chooses a framework and
   method, lists version floors, and returns an ordered path. When the requested
   configuration does not fit, it returns the nearest one that does instead of a
   refusal.
3. **Validate.** `minicpm_check_dataset` reads the training file and reports
   per-record problems with fixes: a record with no assistant turn, an unexpanded
   `*.jpg` glob, a message list that ends on a user turn, or a record that
   silently exceeds the token budget.
4. **Generate.** `minicpm_recipe` emits a complete config — a TRL script with the
   assistant-only-loss template patch, a LLaMA-Factory YAML with the right
   `template` string, or an ms-swift launch script with the right `model_type`.
5. **Check the host.** `minicpm_doctor` runs one bounded, read-only probe and
   compares the result against the model's recorded version floors and VRAM.
6. **Serve.** `minicpm_serve` emits the launch command with the per-model quirks
   that cause silent quality loss: `add_special_tokens` on vLLM for MiniCPM4.1,
   `min_p=0.0` on llama.cpp, the tool-call parser on SGLang.
7. **Publish.** `minicpm_publish` produces the model card, the NOTICE, and the
   licence instructions, and withholds the upload commands when the base licence
   is not verified for redistribution.

## What it deliberately does not do

- **It does not write files.** Generated recipes are returned as text so the
  harness's own file tools decide where they land and under which policy.
- **It does not vendor OpenBMB's skills.** Upstream's sixteen skills are already
  in a compatible format; copying them here would guarantee drift. Point a
  filesystem skill provider at a clone instead.
- **It does not store VRAM as fact.** The numbers depend on sequence length,
  batch, optimizer and offload policy, so the bundle computes and labels them.
- **It does not hide uncertainty.** A catalog entry whose facts could not be
  confirmed is marked `unverified`, says what to check, and causes
  `minicpm_publish` to block.

## Extending the catalog

`data/models.json` is the only place a model is described. Adding one means
adding an entry with its `sources` and `confidence`; `scripts/validate-data.mjs`
enforces the shape, the licence reference, and that at least one framework is
marked supported. `scripts/check-upstream.mjs` compares the recorded upstream
revision against the live repository so drift shows up as a failing check rather
than as quietly wrong advice.
