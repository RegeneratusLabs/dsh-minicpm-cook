/**
 * Publication artifacts: the model card, the attribution files, and the upload
 * commands, plus the licence gate that decides whether to print them at all.
 *
 * This is the part of the MiniCPM lifecycle that upstream documents least, and
 * the part most likely to go wrong legally: a fine-tune redistributed without a
 * NOTICE or with a stale base-model field is a real problem for whoever receives
 * it, not a formatting nit.
 */
import { licenseRules } from './catalog.js'

/** What is being published. */
export const ARTIFACT_KINDS = ['adapter', 'merged', 'gguf']

/** Where it is going. */
export const TARGETS = ['hf', 'modelscope', 'both']

/**
 * Build the publication set.
 *
 * @param options.catalog loaded catalog.
 * @param options.model catalog entry.
 * @param options.artifactKind adapter | merged | gguf.
 * @param options.target hf | modelscope | both.
 * @param options.repoId destination repo, e.g. `alice/minicpm5-2b-sql`.
 * @param options.baseModelId base model the adapter applies to; defaults to the catalog id.
 * @param options.evalResults optional [{benchmark, score, baseScore}] rows.
 * @param options.author attribution line for the NOTICE.
 * @param options.adapterPath local directory holding adapter files.
 * @returns { files, commands, checklist, compliance, blocked }.
 */
export function buildPublish(options) {
  const {
    catalog,
    model,
    artifactKind = 'adapter',
    target = 'hf',
    repoId,
    baseModelId,
    evalResults = [],
    author,
    adapterPath = './runs/minicpm/adapter_final',
  } = options

  if (!ARTIFACT_KINDS.includes(artifactKind)) {
    return { error: `unknown artifactKind "${artifactKind}"`, alternatives: ARTIFACT_KINDS }
  }
  if (!TARGETS.includes(target)) {
    return { error: `unknown target "${target}"`, alternatives: TARGETS }
  }

  const rules = licenseRules(catalog, model)
  const licenseUnknown = model.licenseVerified !== true
  const blocked = licenseUnknown || !rules.redistributable

  const compliance = {
    license: model.license,
    licenseVerified: model.licenseVerified,
    redistributable: rules.redistributable,
    commercialUse: rules.commercialUse,
    attributionRequired: rules.attributionRequired,
    blocked,
    notes: rules.notes,
    sourceUrl: rules.sourceUrl ?? model.licenseSourceUrl,
    reason: blocked
      ? licenseUnknown
        ? 'The base model licence has not been verified against its own LICENSE file, so redistribution cannot be asserted.'
        : `Licence ${model.license} does not permit redistribution of the weights.`
      : 'Apache-2.0 permits redistribution, including commercially, with attribution and a copy of the licence.',
  }

  const base = baseModelId ?? model.hfId
  const card = modelCard({ model, artifactKind, base, repoId, evalResults, author, license: slugLicense(model.license) })
  const notice = noticeFile({ model, author, base })

  const files = [
    { path: 'README.md', content: card },
    { path: 'NOTICE', content: notice },
    {
      path: 'LICENSE-APACHE-2.0-HOWTO.md',
      content: licenseHowto(model),
    },
  ]

  if (artifactKind === 'adapter') {
    files.push({
      path: 'ADAPTER-USAGE.md',
      content: adapterUsage(model, base),
    })
  }

  const commands = blocked ? [] : uploadCommands({ target, repoId, artifactKind, adapterPath })
  const checklist = publicationChecklist({ model, artifactKind, blocked })

  return {
    model: model.id,
    artifactKind,
    target,
    repoId,
    baseModelId: base,
    blocked,
    compliance,
    files,
    commands,
    checklist,
    warnings: blocked
      ? [compliance.reason, 'Upload commands are withheld until the licence is verified for this model.']
      : [
        'Publish the adapter by default: it is smaller, keeps the base licence obvious, and lets consumers disable it.',
        'Merged weights are for consumers who cannot run PEFT; re-save the ORIGINAL tokenizer when merging, not the training-patched one.',
      ],
  }
}

/** Hugging Face model-card frontmatter wants the licence slug, not the display name. */
function slugLicense(license) {
  if (typeof license !== 'string') return 'other'
  if (license.toLowerCase() === 'apache-2.0') return 'apache-2.0'
  return license.toLowerCase()
}

/** The model card. */
function modelCard({ model, artifactKind, base, repoId, evalResults, author, license }) {
  const name = repoId ?? `${model.displayName} fine-tune`
  const library = artifactKind === 'adapter' ? 'peft' : 'transformers'
  const tags = [
    'minicpm',
    artifactKind === 'adapter' ? 'lora' : 'fine-tune',
    model.modality === 'text' ? 'text-generation' : `${model.modality}`,
  ]

  const evalTable = evalResults.length === 0
    ? '| Benchmark | Base | Fine-tune | Delta |\n| --- | --- | --- | --- |\n| _not yet measured_ | — | — | — |\n\n> Replace this table with measured numbers. An unmeasured card is a claim, not a record.'
    : [
      '| Benchmark | Base | Fine-tune | Delta |',
      '| --- | --- | --- | --- |',
      ...evalResults.map((row) => {
        const delta = typeof row.score === 'number' && typeof row.baseScore === 'number'
          ? (row.score - row.baseScore).toFixed(2)
          : '—'
        return `| ${row.benchmark} | ${row.baseScore ?? '—'} | ${row.score ?? '—'} | ${delta} |`
      }),
    ].join('\n')

  return `---
license: ${license}
base_model: ${base}
library_name: ${library}
pipeline_tag: ${model.modality === 'text' ? 'text-generation' : 'image-text-to-text'}
tags:
${tags.map((tag) => `  - ${tag}`).join('\n')}
---

# ${name}

${artifactKind === 'adapter' ? 'A LoRA adapter' : artifactKind === 'merged' ? 'Fully merged weights' : 'GGUF weights'} for
[${model.displayName}](https://huggingface.co/${base}) (${model.paramsB}B, ${model.modality}, context ${model.contextTokens.toLocaleString('en-US')} tokens).

Fine-tuned by ${author ?? 'an unnamed author'}. Trained ${new Date().toISOString().slice(0, 10)}.

## What this is

- **Base model:** [\`${base}\`](https://huggingface.co/${base})
- **Artifact:** ${artifactKind}
- **Licence of the base:** ${model.license}
${artifactKind === 'adapter' ? '- **Loading:** requires the base model plus `peft`\n' : ''}
## Training

| Setting | Value |
| --- | --- |
| Method | ${artifactKind === 'merged' ? 'full fine-tune or merged adapter' : 'LoRA'} |
| Base revision | \`main\` (pin a commit for reproducibility) |
| Hardware | _fill in: GPU model and count_ |
| Sequence length | _fill in_ |
| Epochs | _fill in_ |
| Learning rate | _fill in_ |
| Batch (per device × accumulation) | _fill in_ |

## Data

_Describe the training data: source, size, languages, licence, and how it was filtered._
If the data cannot be released, say so explicitly here; a consumer needs to know
whether the behaviour you ship can be reproduced.

## Evaluation

${evalTable}

## Usage

${artifactKind === 'adapter' ? adapterUsage(model, base) : mergedUsage(model)}

## Limitations

- Fine-tuned for a narrow task. Off-distribution behaviour may be worse than the base model.
- Not safety-tuned. Review outputs before any user-facing deployment.
- _Add observed failure modes here: this section is what saves a consumer an incident._

## Licence and attribution

Base model and this derivative are distributed under **${model.license}**.
${model.license.toLowerCase() === 'apache-2.0'
    ? 'You must retain the licence text and the NOTICE file, and indicate any changes made.'
    : 'Check the base model licence before redistributing or using this commercially.'}

The base model was produced by OpenBMB / THUNLP / ModelBest. See \`NOTICE\`.

## Citation

\`\`\`bibtex
@misc{${model.id}${model.releaseDate.slice(0, 4)},
  title  = {${model.displayName}},
  author = {OpenBMB and THUNLP and ModelBest},
  year   = {${model.releaseDate.slice(0, 4)}},
  url    = {https://huggingface.co/${base}}
}
\`\`\`
`
}

/** Adapter loading instructions, including the tokenizer trap. */
function adapterUsage(model, base) {
  return `# Loading the adapter

\`\`\`python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("${base}", torch_dtype="auto", device_map="auto"${model.trustRemoteCode ? ', trust_remote_code=True' : ''})
model = PeftModel.from_pretrained(base, "<this-repo>").eval()
# Load the ORIGINAL tokenizer: the training template exists only to mask loss.
tok = AutoTokenizer.from_pretrained("${base}"${model.trustRemoteCode ? ', trust_remote_code=True' : ''})

messages = [{"role": "user", "content": "Your prompt here"}]
inputs = tok.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
).to(model.device)
out = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=1.0, top_p=0.95)
print(tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
\`\`\`

> The training recipe patches the chat template to add \`{% generation %}\` blocks for
> assistant-only loss. That template must never be saved with the tokenizer, and this
> adapter relies on the base model's own template at inference time.
`
}

/** Merged-weight usage. */
function mergedUsage(model) {
  return `# Usage

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "<this-repo>"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "Your prompt here"}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=1.0, top_p=0.95)
print(tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
\`\`\`

Or serve it directly:

\`\`\`bash
vllm serve <this-repo> --port 8000${model.trustRemoteCode ? ' --trust-remote-code' : ''}
\`\`\`
`
}

/** NOTICE with the attribution Apache-2.0 requires. */
function noticeFile({ model, author, base }) {
  return `${model.displayName} derivative
Copyright ${new Date().getFullYear()} ${author ?? 'the publishing author'}

This product includes model weights and/or code derived from
${model.displayName} (${base}), developed by OpenBMB, THUNLP (Tsinghua University)
and the Gaoling School of Artificial Intelligence at Renmin University of China,
and licensed under ${model.license}.

Original repository: https://github.com/OpenBMB/MiniCPM
Base model:          https://huggingface.co/${base}

Modifications: this derivative was fine-tuned on additional data. The base weights
are otherwise unmodified. Training code and data provenance are described in README.md.

The full text of the ${model.license} licence is available at
https://www.apache.org/licenses/LICENSE-2.0.txt and must be shipped alongside
these weights.
`
}

/** Explain how to complete the licence file rather than fabricating its text. */
function licenseHowto(model) {
  return `# Licence file: one manual step

The base model is ${model.license}. To redistribute this derivative you must ship the
full licence text in a file named \`LICENSE\`. This generator does not embed the
licence body, because a paraphrased or truncated licence is worse than none.

\`\`\`bash
curl -fsSL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE
\`\`\`

Then commit \`LICENSE\` next to \`README.md\` and \`NOTICE\`.

Keep \`NOTICE\`: ${model.license} requires attribution, and the file names OpenBMB,
THUNLP and ModelBest as the origin of the base weights.
`
}

/** Upload commands for the selected target. */
function uploadCommands({ target, repoId, artifactKind, adapterPath }) {
  if (typeof repoId !== 'string' || repoId.trim() === '') {
    return [{
      step: 'set repoId',
      command: '# pass repoId (e.g. "alice/minicpm5-2b-sql") to get upload commands',
    }]
  }

  const commands = []
  if (target === 'hf' || target === 'both') {
    commands.push({
      target: 'huggingface',
      command: artifactKind === 'adapter'
        ? `huggingface-cli upload ${repoId} ${adapterPath} . --repo-type model`
        : `huggingface-cli upload ${repoId} . --repo-type model`,
      note: 'Log in first with `huggingface-cli login`, or set HF_TOKEN.',
    })
  }
  if (target === 'modelscope' || target === 'both') {
    commands.push({
      target: 'modelscope',
      command: `modelscope upload ${repoId} . --repo-type model`,
      note: 'Export MS_TOKEN or run `modelscope login` first. ModelScope ids are case-sensitive.',
    })
  }
  return commands
}

/** What the publisher must confirm before pushing. */
function publicationChecklist({ model, artifactKind, blocked }) {
  return [
    { item: 'Full licence text present as LICENSE', required: true },
    { item: 'NOTICE names OpenBMB / THUNLP / ModelBest', required: true },
    { item: 'README frontmatter `base_model` points at the real base', required: true },
    { item: 'Evaluation table contains measured numbers, not placeholders', required: true },
    { item: 'Limitations section lists observed failure modes', required: true },
    { item: 'Training data provenance described, or its absence stated', required: true },
    {
      item: 'Adapter published rather than merged weights',
      required: false,
      note: artifactKind === 'adapter' ? 'Satisfied.' : 'You are publishing merged or GGUF weights; confirm that is deliberate.',
    },
    {
      item: 'Base model revision pinned in the card',
      required: false,
    },
    {
      item: 'Licence verified for this exact model',
      required: true,
      note: blocked ? `BLOCKED: ${model.licenseVerified ? 'licence terms forbid it' : 'licence not verified'}.` : 'Satisfied.',
    },
  ]
}
