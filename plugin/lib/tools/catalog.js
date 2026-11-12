/**
 * `minicpm_catalog` — the entry point into the MiniCPM family.
 *
 * Read-only over the bundled catalog. Answers "what exists, which one should I
 * use, what does it cost me" before any other tool is worth calling.
 */
import { bestBackend, knownIds, queryModels } from '../catalog.js'
import { bulletList, int, num, outputSchema, pick, renderSummary, str } from '../toolkit.js'

const FAMILIES = ['any', 'minicpm5', 'minicpm41', 'minicpm4', 'sala', 'minicpm-v', 'minicpm-o', 'ocr', 'bitcpm']
const MODALITIES = ['any', 'text', 'vision', 'omni', 'ocr']

export function createCatalogTool({ catalog }) {
  return {
    name: 'minicpm_catalog',
    description:
      'List and filter the OpenBMB MiniCPM model family: ids, size, modality, context, licence, ' +
      'recommended backend, and which fine-tuning frameworks each model supports. Use this first to ' +
      'choose a model; the result feeds the other minicpm_* tools.',

    parameters: {
      type: 'object',
      properties: {
        query: str('Free-text match against id, display name, tagline and architecture.'),
        family: pick('Restrict to one model line.', FAMILIES),
        modality: pick('Restrict to one input modality.', MODALITIES),
        maxParamsB: num('Only models at or below this total parameter count, in billions.'),
        license: str('Require an exact licence string, for example "Apache-2.0".'),
        framework: str('Only models whose catalog records this backend as supported, for example "vllm".'),
        detail: pick('summary returns one line per model; full returns every recorded field.', ['summary', 'full']),
      },
      additionalProperties: false,
    },

    output: {
      schema: outputSchema(
        {
          matched: int('How many models matched.'),
          models: { type: 'array', items: { type: 'object', additionalProperties: true } },
          notes: { type: 'array', items: { type: 'string' } },
        },
        ['matched', 'models'],
      ),
      render(args, value) {
        if (value.matched === 0) {
          return renderSummary('No MiniCPM model matches that filter.', [
            'Relax family/modality, or drop the free-text query.',
            `Known ids: ${knownIds(catalog).join(', ')}`,
          ])
        }
        const lines = value.models.map((model) => {
          const head = `${model.id} — ${model.displayName} (${model.paramsB}B ${model.modality}, ${model.releaseDate})`
          if (args.detail === 'full') {
            return [
              head,
              `  hf: ${model.hfId}   licence: ${model.license}${model.licenseVerified ? '' : ' (unverified)'}`,
              `  context: ${model.contextTokens.toLocaleString('en-US')}   trust_remote_code: ${model.trustRemoteCode}`,
              `  best for: ${model.bestFor}`,
              model.bestBackend === undefined ? undefined : `  backend: ${model.bestBackend}`,
              `  sources: ${model.sources.length} recorded`,
            ].filter(Boolean).join('\n')
          }
          return `${head}\n  ${model.bestFor}${model.bestBackend ? `  [${model.bestBackend}]` : ''}`
        })
        return renderSummary(
          `${value.matched} model(s):`,
          value.models.flatMap((model, index) => [lines[index], '']).slice(0, -1),
          value.notes?.length ? `\n${bulletList(value.notes).join('\n')}` : undefined,
        )
      },
    },

    async execute(args) {
      const models = queryModels(catalog, args ?? {})
      return {
        matched: models.length,
        models: models.map((model) => {
          const summary = {
            id: model.id,
            displayName: model.displayName,
            family: model.family,
            modality: model.modality,
            paramsB: model.paramsB,
            contextTokens: model.contextTokens,
            hfId: model.hfId,
            license: model.license,
            licenseVerified: model.licenseVerified,
            releaseDate: model.releaseDate,
            bestFor: model.bestFor,
            confidence: model.confidence,
            bestBackend: bestBackend(model).backend,
          }
          if (args?.detail !== 'full') return summary
          return {
            ...summary,
            msId: model.msId,
            architecture: model.architecture,
            trustRemoteCode: model.trustRemoteCode,
            activeParamsB: model.activeParamsB,
            chatTemplate: model.chatTemplate,
            variants: model.variants,
            frameworks: model.frameworks,
            sources: model.sources,
            confidenceNote: model.confidenceNote,
          }
        }),
        notes: catalog.notes ?? [],
      }
    },
  }
}
