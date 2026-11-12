/**
 * End-to-end planning: turn a goal plus a machine profile into one ordered path.
 *
 * This is the module that answers "what should I actually do". It never guesses
 * silently: every choice carries its reason, and a configuration that does not
 * fit the stated hardware comes back with the nearest workable alternative
 * instead of a bare refusal.
 */
import { bestBackend, licenseRules } from './catalog.js'
import { estimateInference, estimateTraining } from './estimate.js'
import { toolchainSupports } from './recipes.js'

/** Goals a caller can plan for. */
export const GOALS = ['sft', 'lora', 'qlora', 'dpo', 'serve', 'publish', 'evaluate']

/**
 * Build a plan.
 *
 * @param options.catalog loaded catalog (for license rules).
 * @param options.model catalog entry.
 * @param options.goal one of GOALS.
 * @param options.gpuCount device count.
 * @param options.vramGb per-device VRAM.
 * @param options.datasetFormat dataset layout the user has.
 * @param options.maxSeqLen sequence length.
 * @param options.preferToolchain optional toolchain override.
 * @returns a plan object with recommendation, pins, estimates, and steps.
 */
export function buildPlan(options) {
  const {
    catalog,
    model,
    goal = 'sft',
    gpuCount = 1,
    vramGb = 24,
    datasetFormat,
    maxSeqLen,
    preferToolchain,
  } = options

  if (!GOALS.includes(goal)) {
    return { error: `unknown goal "${goal}"`, alternatives: GOALS }
  }

  const warnings = []
  const isVision = model.modality !== 'text'
  const seqLen = maxSeqLen ?? (isVision ? 4096 : Math.min(4096, model.contextTokens))

  if (goal === 'publish') return publishPlan({ catalog, model, warnings })
  if (goal === 'serve') return servePlan({ model, gpuCount, vramGb, seqLen, warnings })
  if (goal === 'evaluate') return evaluatePlan({ model, warnings })

  const toolchain = pickToolchain(model, preferToolchain, warnings)
  const method = pickMethod({ goal, model, gpuCount, vramGb, seqLen, warnings })
  const training = estimateTraining({
    model,
    method,
    seqLen,
    batch: 1,
    gpuCount,
    vramGb,
    gradientCheckpointing: true,
  })

  // A configuration that cannot fit is still a useful answer: report the
  // smallest method that would fit, so the user has an action rather than a wall.
  let fallback
  if (!training.fits) {
    for (const candidate of ['qlora', 'lora']) {
      if (candidate === method) continue
      const attempt = estimateTraining({
        model, method: candidate, seqLen, batch: 1, gpuCount, vramGb, gradientCheckpointing: true,
      })
      if (attempt.fits) {
        fallback = { method: candidate, perGpuGb: attempt.perGpuGb }
        break
      }
    }
    warnings.push(
      fallback === undefined
        ? `Not even QLoRA fits at ${seqLen} tokens on ${gpuCount}×${vramGb} GB. Shorten the sequence, add devices, or pick a smaller model.`
        : `${method} does not fit at ${seqLen} tokens on ${gpuCount}×${vramGb} GB; ${fallback.method} fits at ~${fallback.perGpuGb} GB/GPU.`,
    )
  }

  const pins = versionPins(model, { needsTraining: true })
  if (model.trustRemoteCode) {
    warnings.push(
      `${model.displayName} ships custom modeling code: load it with trust_remote_code=True. ` +
      'The generated configs set it; a hand-written loader that omits it fails at import.',
    )
  }
  if (isVision) {
    warnings.push(
      `Keep freeze_vision_tower=false. Freezing the vision tower is the most common cause of a VLM fine-tune that ` +
      'learns the answer style but loses the grounding.',
    )
    const modes = model.vision?.downsampleModes ?? []
    if (modes.length > 0) {
      warnings.push(
        `Choose the visual token budget explicitly (DOWNSAMPLE_MODE=${modes.join('|')}). ` +
        `4x keeps more visual detail, 16x costs ~4x fewer visual tokens and measurably lower OCR accuracy.`,
      )
    }
  }
  if (datasetFormat !== undefined && isVision && datasetFormat === 'messages') {
    warnings.push(
      'Vision fine-tuning needs an images field beside messages. Use the vlm-swift format ' +
      '(`{"messages":[...],"images":["a.jpg"]}`) or an LLaVA-style sharegpt record.',
    )
  }
  if (!isVision && datasetFormat === 'vlm-swift') {
    warnings.push('vlm-swift records carry images; this model is text-only and ignores them.')
  }

  const steps = [
    {
      step: 1,
      action: 'Check the environment against the recorded version floor',
      tool: 'minicpm_doctor',
      why: pins.length > 0 ? `Required: ${pins.map((pin) => `${pin.framework}>=${pin.version}`).join(', ')}` : 'Confirms torch/transformers/toolchain alignment.',
    },
    {
      step: 2,
      action: 'Validate the dataset before spending GPU time',
      tool: 'minicpm_check_dataset',
      why: 'Catches missing assistant turns, unexpanded image globs, and silent token overflow.',
    },
    {
      step: 3,
      action: `Generate the ${toolchain} recipe`,
      tool: 'minicpm_recipe',
      args: { toolchain, method, maxSeqLen: seqLen },
      why: `Emits the correct template (${templateFor(model, toolchain) ?? 'model-specific'}) and target modules.`,
    },
    {
      step: 4,
      action: 'Run training, then verify loss falls and sample outputs are on-task',
      why: 'A flat loss usually means assistant-only masking is off or the template was not patched.',
    },
    {
      step: 5,
      action: 'Serve the adapter or merged model for a regression check',
      tool: 'minicpm_serve',
      why: 'Confirms the adapter loads against the base and the chat template still renders.',
    },
  ]

  if (goal === 'dpo') {
    steps.splice(3, 0, {
      step: 3,
      action: 'SFT first, then run DPO on preference pairs',
      why: 'DPO on a base model without a task SFT stage mostly teaches formatting, not preference.',
    })
  }

  return {
    model: model.id,
    displayName: model.displayName,
    goal,
    recommendation: {
      toolchain,
      method,
      reason: recommendationReason({ model, toolchain, method, goal }),
    },
    hardware: { gpuCount, vramGb, seqLen },
    training: {
      estimate: training.estimate,
      perGpuGb: training.perGpuGb,
      totalGb: training.totalGb,
      headroomGb: training.headroomGb,
      fits: training.fits,
      tight: training.tight,
      breakdownGb: training.breakdownGb,
      trainableParams: training.utilizations.trainableParams,
      fallback,
    },
    publishedHardware: model.publishedHardware ?? null,
    defaults: defaultHyperparameters({ model, method, toolchain, seqLen }),
    pins,
    datasetFormat: datasetFormat ?? (isVision ? 'vlm-swift' : 'messages'),
    steps,
    warnings,
    caveats: [
      'Memory figures are ESTIMATEs from an explicit formula, not measurements on your devices.',
      model.publishedHardware === undefined
        ? 'No official hardware floor is recorded for this model.'
        : `Official floor: ${JSON.stringify(model.publishedHardware)}.`,
    ],
  }
}

/** Choose a toolchain, honouring an explicit preference only if it can work. */
function pickToolchain(model, preferToolchain, warnings) {
  // For a text model TRL is the smallest documented surface; for a VLM it has no
  // path at all, so LLaMA-Factory leads.
  const defaults = model.modality === 'text'
    ? ['trl', 'llamafactory', 'ms-swift']
    : ['llamafactory', 'ms-swift']
  const candidates = preferToolchain === undefined
    ? defaults
    : [preferToolchain, ...defaults]

  for (const candidate of candidates) {
    const support = toolchainSupports(model, candidate)
    if (support.ok && model.frameworks?.[candidate] !== undefined) return candidate
    if (preferToolchain === candidate && !support.ok) warnings.push(support.reason)
  }
  return 'llamafactory'
}

/** Choose a training method for the goal and the hardware. */
function pickMethod({ goal, model, gpuCount, vramGb, seqLen, warnings }) {
  if (goal === 'qlora') return 'qlora'
  if (goal === 'lora') return 'lora'

  // For a plain SFT goal, let the hardware decide, then fall back downward.
  for (const method of ['lora', 'qlora']) {
    const estimate = estimateTraining({
      model, method, seqLen, batch: 1, gpuCount, vramGb, gradientCheckpointing: true,
    })
    if (estimate.fits) {
      if (method === 'qlora') {
        warnings.push(`LoRA does not fit at ${seqLen} tokens here; planning QLoRA instead.`)
      }
      return method
    }
  }
  return 'qlora'
}

/** Human-readable reason for the chosen path. */
function recommendationReason({ model, toolchain, method, goal }) {
  const why = {
    trl: 'the smallest documented surface: one Python file, assistant-only loss built in',
    llamafactory: model.modality === 'text'
      ? 'a YAML-driven path with a WebUI'
      : 'a documented VLM path that handles image tokens and dynamic resolution',
    'ms-swift': 'a CLI that runs out of the box and covers DPO as well as SFT',
  }[toolchain] ?? 'a documented path for this model'
  const parts = [
    `${model.displayName} is a ${model.paramsB}B ${model.modality} model released ${model.releaseDate}.`,
    `${toolchain} is ${why}.`,
    `${method.toUpperCase()} keeps the ${goal} run inside the stated memory budget.`,
  ]
  return parts.join(' ')
}

/** Framework template string for a model and toolchain, if recorded. */
export function templateFor(model, toolchain) {
  if (toolchain === 'llamafactory') return model.frameworks?.llamafactory?.template
  if (toolchain === 'ms-swift') return model.frameworks?.ms_swift?.template
  return model.chatTemplate?.name
}

/** Version pins that matter for this model. */
export function versionPins(model, { needsTraining = false } = {}) {
  const pins = []
  for (const framework of ['transformers', 'torch', 'vllm', 'sglang', 'trl', 'peft']) {
    const version = model.frameworks?.[framework]?.minVersion
    if (typeof version === 'string') pins.push({ framework, version })
  }
  if (needsTraining && !pins.some((pin) => pin.framework === 'trl') && model.modality === 'text') {
    pins.push({ framework: 'trl', version: '0.21' })
  }
  return pins
}

/** Method-appropriate starting hyperparameters. */
function defaultHyperparameters({ model, method, toolchain, seqLen }) {
  const isVision = model.modality !== 'text'
  return {
    maxSeqLen: seqLen,
    epochs: isVision ? 3 : 2,
    learningRate: isVision ? 5e-6 : method === 'full' ? 1e-5 : 2e-4,
    perDeviceBatchSize: isVision ? 1 : 4,
    gradientAccumulation: isVision ? 16 : 4,
    loraRank: method === 'full' ? null : 16,
    loraAlpha: method === 'full' ? null : 32,
    loraDropout: method === 'full' ? null : 0.05,
    lrScheduler: 'cosine',
    warmupRatio: 0.03,
    bf16: true,
    packing: false,
    assistantOnlyLoss: toolchain === 'trl' && !isVision,
    freezeVisionTower: isVision ? false : null,
    downsample: isVision ? model.vision?.defaultDownsample ?? null : null,
  }
}

/** Serving plan. */
function servePlan({ model, gpuCount, vramGb, seqLen, warnings }) {
  const { backend } = bestBackend(model)
  const inference = estimateInference({ model, quant: 'bf16', seqLen, gpuCount, vramGb })

  if (model.frameworks?.vllm?.supported !== true) {
    warnings.push(
      `${model.displayName} has no documented vLLM path; the plan uses ${backend ?? 'transformers'}.`,
    )
  }
  if (!inference.fits) {
    warnings.push(
      `BF16 does not fit ${gpuCount}×${vramGb} GB at ${seqLen} tokens (~${inference.perGpuGb} GB/GPU). ` +
      'Use the quantized variant, shorten --max-model-len, or add devices.',
    )
  }

  return {
    model: model.id,
    goal: 'serve',
    recommendation: { backend: backend ?? 'transformers', notice: model.frameworks?.[backend ?? '']?.notes },
    inference,
    variants: model.variants.filter((variant) => ['gguf', 'awq', 'gptq', 'fp8', 'int4', 'marlin'].includes(variant.kind)),
    steps: [
      { step: 1, action: 'Generate the launch command', tool: 'minicpm_serve' },
      { step: 2, action: 'Smoke-test /v1/chat/completions with the sampling parameters from the catalog' },
      { step: 3, action: 'Compare base vs fine-tuned outputs on a fixed prompt set before shipping' },
    ],
    sampling: model.sampling ?? null,
    warnings,
  }
}

/** Publication plan, including the compliance gate. */
function publishPlan({ catalog, model, warnings }) {
  const rules = licenseRules(catalog, model)
  const blocked = !rules.redistributable || model.licenseVerified !== true

  if (blocked) {
    warnings.push(
      model.licenseVerified !== true
        ? `The license for ${model.displayName} is not recorded as verified; redistribution is gated until it is.`
        : `License ${model.license} does not permit redistribution.`,
    )
  }
  if (model.variants.some((variant) => variant.kind === 'gguf')) {
    warnings.push('An official GGUF exists upstream. Publishing a second GGUF adds a maintenance burden for little reach.')
  }

  return {
    model: model.id,
    goal: 'publish',
    compliance: {
      license: model.license,
      licenseVerified: model.licenseVerified,
      redistributable: rules.redistributable,
      commercialUse: rules.commercialUse,
      attributionRequired: rules.attributionRequired,
      blocked,
      notes: rules.notes,
      sourceUrl: rules.sourceUrl ?? model.licenseSourceUrl,
    },
    steps: [
      { step: 1, action: 'Run the regression check: base vs fine-tune on a fixed prompt set' },
      { step: 2, action: 'Generate the model card and attribution files', tool: 'minicpm_publish' },
      { step: 3, action: 'Push the adapter (default) or the merged model to the Hub' },
      { step: 4, action: 'Record the adapter in the model card so consumers can disable it' },
    ],
    warnings,
  }
}

/** Evaluation plan. */
function evaluatePlan({ model, warnings }) {
  warnings.push(
    'Benchmarks are only a smoke test for a fine-tune: a task-specific eval set you build beats a public score ' +
    'that the base model already saturated.',
  )
  return {
    model: model.id,
    goal: 'evaluate',
    recommended: {
      taskEval: 'Build 50–200 held-out prompts from your own distribution; report exact-match or rubric scores.',
      regression: 'Run the same prompts against the unmodified base and diff outputs, not just scores.',
      publicBenchmarks: model.modality === 'text'
        ? ['C-Eval', 'CMMLU', 'MMLU', 'GSM8K']
        : ['OCRBench', 'MMMU', 'MathVista'],
      catastrophicForgetting: 'Re-run 20 general instructions; a fine-tune that loses basic chat is over-trained.',
    },
    steps: [
      { step: 1, action: 'Freeze a held-out prompt set before training' },
      { step: 2, action: 'Score base and fine-tune with the same harness and decoding parameters' },
      { step: 3, action: 'Inspect the worst 10 regressions by hand' },
    ],
    warnings,
  }
}
