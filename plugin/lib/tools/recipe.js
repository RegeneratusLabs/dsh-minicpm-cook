/**
 * `minicpm_recipe` — emit a runnable fine-tuning recipe.
 *
 * Files are returned as text; the tool does not write them. The agent saves them
 * with whatever file tool the profile provides, which keeps this bundle free of
 * sandbox and path-policy assumptions.
 */
import { resolveModel } from '../catalog.js'
import { buildRecipe, METHODS, TOOLCHAINS } from '../recipes.js'
import { bulletList, failure, int, num, outputSchema, pick, renderSummary, str } from '../toolkit.js'

export function createRecipeTool({ catalog }) {
  return {
    name: 'minicpm_recipe',
    description:
      'Generate a complete MiniCPM fine-tuning recipe as files you can save and run: a TRL+PEFT script, ' +
      'a LLaMA-Factory YAML plus dataset entry, or an ms-swift launch script. Carries the model-specific ' +
      'template, target modules and assistant-only loss settings so the common failure modes are avoided.',

    parameters: {
      type: 'object',
      properties: {
        modelId: str('Catalog id, e.g. "minicpm5-2b". Omit to resolve from family/modality.'),
        family: str('Model line to resolve when modelId is omitted.'),
        modality: pick('Modality to resolve when modelId is omitted.', ['any', 'text', 'vision', 'omni', 'ocr']),
        toolchain: pick('Framework to generate for.', TOOLCHAINS),
        method: pick('Training method.', METHODS),
        dataPath: str('Training file the generated config should point at.'),
        outputDir: str('Directory for checkpoints and adapters.'),
        maxSeqLen: int('Sequence length to train at.', { minimum: 128 }),
        epochs: num('Epoch count.', { minimum: 0.1 }),
        lr: num('Learning rate; omitted picks a method-appropriate default.', { minimum: 0 }),
        batch: int('Per-device batch size.', { minimum: 1 }),
        gradAccum: int('Gradient accumulation steps.', { minimum: 1 }),
        loraRank: int('LoRA rank.', { minimum: 1 }),
        downsample: pick('VLM visual token budget.', ['4x', '16x']),
      },
      additionalProperties: false,
    },

    output: {
      schema: outputSchema(
        {
          model: str('Resolved catalog id.'),
          toolchain: str('Framework the recipe targets.'),
          files: {
            type: 'array',
            items: {
              type: 'object',
              properties: { path: { type: 'string' }, content: { type: 'string' } },
              required: ['path', 'content'],
              additionalProperties: false,
            },
          },
          commands: { type: 'array', items: { type: 'string' } },
          warnings: { type: 'array', items: { type: 'string' } },
        },
        ['model', 'toolchain', 'files'],
      ),
      render(_args, value) {
        const lines = [
          `${value.displayName ?? value.model} — ${value.toolchain} (${value.defaults?.method})`,
          '',
          `Files: ${value.files.map((file) => file.path).join(', ')}`,
          '',
          'Run:',
          ...value.commands.map((command) => `  ${command}`),
        ]
        if (value.warnings?.length > 0) lines.push('', 'Before you run:', ...bulletList(value.warnings))
        if (value.notes?.length > 0) lines.push('', 'Notes:', ...bulletList(value.notes))
        return renderSummary('', lines)
      },
    },

    async execute(args) {
      const filter = args ?? {}
      const resolved = resolveModel(catalog, filter)
      if (resolved.error !== undefined) throw failure(resolved.error, resolved.hint)

      const recipe = buildRecipe({
        model: resolved.model,
        toolchain: filter.toolchain ?? 'llamafactory',
        method: filter.method ?? 'lora',
        dataPath: filter.dataPath,
        outputDir: filter.outputDir,
        maxSeqLen: filter.maxSeqLen,
        epochs: filter.epochs,
        lr: filter.lr,
        batch: filter.batch,
        gradAccum: filter.gradAccum,
        loraRank: filter.loraRank,
        downsample: filter.downsample,
      })
      if (recipe.error !== undefined) {
        throw failure(recipe.error, `Compatible toolchains: ${recipe.alternatives?.join(', ') ?? 'none'}`)
      }

      return {
        model: resolved.model.id,
        displayName: resolved.model.displayName,
        toolchain: filter.toolchain ?? 'llamafactory',
        ...recipe,
      }
    },
  }
}
