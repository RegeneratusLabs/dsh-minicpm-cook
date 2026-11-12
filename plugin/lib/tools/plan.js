/**
 * `minicpm_plan` — turn a goal and a machine profile into one ordered path.
 */
import { resolveModel } from '../catalog.js'
import { buildPlan, GOALS } from '../plan.js'
import { bulletList, bool, failure, int, num, outputSchema, pick, renderSummary, str } from '../toolkit.js'

export function createPlanTool({ catalog }) {
  return {
    name: 'minicpm_plan',
    description:
      'Plan a MiniCPM task end to end: pick the model, the fine-tuning framework and method that fit the ' +
      'stated GPUs, estimate memory, list version pins, and return the ordered steps. Use before writing ' +
      'any config by hand.',

    parameters: {
      type: 'object',
      properties: {
        modelId: str('Catalog id, e.g. "minicpm5-2b". Omit to resolve from family/modality.'),
        family: str('Model line to resolve when modelId is omitted.'),
        modality: pick('Modality to resolve when modelId is omitted.', ['any', 'text', 'vision', 'omni', 'ocr']),
        goal: pick('What you are trying to do.', GOALS),
        gpuCount: int('Number of GPUs available.', { minimum: 1 }),
        vramGb: num('Usable VRAM per GPU, in GB.', { minimum: 1 }),
        datasetFormat: pick('Shape of the training data you already have.', ['messages', 'alpaca', 'dpo', 'kto', 'vlm-swift']),
        maxSeqLen: int('Sequence length to train at.', { minimum: 128 }),
        preferToolchain: pick('Force a framework if you have a reason to.', ['any', 'llamafactory', 'ms-swift', 'trl']),
        includeServe: bool('Also include the serving recommendation (default true).'),
      },
      additionalProperties: false,
    },

    output: {
      schema: outputSchema(
        {
          model: str('Resolved catalog id.'),
          goal: str('Goal that was planned.'),
          recommendation: { type: 'object', additionalProperties: true },
          training: { type: 'object', additionalProperties: true },
          steps: { type: 'array', items: { type: 'object', additionalProperties: true } },
          warnings: { type: 'array', items: { type: 'string' } },
        },
        ['model', 'recommendation', 'steps'],
      ),
      render(_args, value) {
        const lines = [
          `${value.displayName ?? value.model} — goal: ${value.goal}`,
          `Framework: ${value.recommendation.toolchain}   Method: ${value.recommendation.method}`,
          value.recommendation.reason,
          '',
        ]
        if (value.training !== undefined) {
          lines.push(
            `Memory: ${value.training.perGpuGb} GB/GPU (${value.training.fits ? 'fits' : 'DOES NOT FIT'}` +
            `${value.training.tight ? ', tight' : ''}) — ${value.training.estimate ? 'ESTIMATE' : 'measured'}`,
            `  ${Object.entries(value.training.breakdownGb ?? {}).map(([key, gb]) => `${key} ${gb}GB`).join(' · ')}`,
            '',
          )
        }
        lines.push('Steps:')
        lines.push(...value.steps.map((step) => `${step.step}. ${step.action}${step.tool ? `  [${step.tool}]` : ''}`))
        if (value.pins?.length > 0) {
          lines.push('', `Version floor: ${value.pins.map((pin) => `${pin.framework}>=${pin.version}`).join(', ')}`)
        }
        if (value.warnings?.length > 0) {
          lines.push('', 'Watch out:', ...bulletList(value.warnings))
        }
        return renderSummary('', lines)
      },
    },

    async execute(args) {
      const filter = args ?? {}
      const resolved = resolveModel(catalog, filter)
      if (resolved.error !== undefined) throw failure(resolved.error, resolved.hint)

      const plan = buildPlan({
        catalog,
        model: resolved.model,
        goal: filter.goal ?? 'sft',
        gpuCount: filter.gpuCount ?? 1,
        vramGb: filter.vramGb ?? 24,
        datasetFormat: filter.datasetFormat,
        maxSeqLen: filter.maxSeqLen,
        preferToolchain: filter.preferToolchain === 'any' ? undefined : filter.preferToolchain,
      })
      if (plan.error !== undefined) throw failure(plan.error, `Try one of: ${plan.alternatives?.join(', ')}`)

      return { ...plan, resolvedBy: resolved.how, alternatives: resolved.alternatives }
    },
  }
}
