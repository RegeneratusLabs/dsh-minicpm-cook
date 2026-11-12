/**
 * `minicpm_publish` — the distribution layer.
 *
 * Generates the model card, NOTICE, licence instructions, and upload commands for
 * a fine-tuned MiniCPM, and withholds the upload path when the licence is not
 * verified for redistribution.
 */
import { resolveModel } from '../catalog.js'
import { buildPublish, ARTIFACT_KINDS, TARGETS } from '../publish.js'
import { bulletList, failure, outputSchema, pick, renderSummary, str } from '../toolkit.js'

export function createPublishTool({ catalog }) {
  return {
    name: 'minicpm_publish',
    description:
      'Prepare a fine-tuned MiniCPM for distribution: Hugging Face model card with correct frontmatter, ' +
      'NOTICE and licence instructions for the Apache-2.0 attribution, adapter-vs-merged guidance, and ' +
      'upload commands. Blocks the upload path when the licence is not verified.',

    parameters: {
      type: 'object',
      properties: {
        modelId: str('Catalog id of the base model, e.g. "minicpm5-2b".'),
        family: str('Model line to resolve when modelId is omitted.'),
        modality: pick('Modality to resolve when modelId is omitted.', ['any', 'text', 'vision', 'omni', 'ocr']),
        artifactKind: pick('What you are publishing.', ARTIFACT_KINDS),
        target: pick('Where it is going.', TARGETS),
        repoId: str('Destination repo, e.g. "alice/minicpm5-2b-sql".'),
        baseModelId: str('Override the base model id recorded in the card.'),
        author: str('Name to put in the NOTICE and card.'),
        adapterPath: str('Local directory holding the adapter files.'),
        evalResults: {
          type: 'array',
          description: 'Measured results, so the card carries numbers instead of placeholders.',
          items: {
            type: 'object',
            properties: {
              benchmark: { type: 'string' },
              score: { type: 'number' },
              baseScore: { type: 'number' },
            },
            required: ['benchmark'],
            additionalProperties: false,
          },
        },
      },
      additionalProperties: false,
    },

    output: {
      schema: outputSchema(
        {
          model: str('Resolved catalog id.'),
          blocked: { type: 'boolean', description: 'True when redistribution is not permitted.' },
          compliance: { type: 'object', additionalProperties: true },
          files: { type: 'array', items: { type: 'object', additionalProperties: true } },
          commands: { type: 'array', items: { type: 'object', additionalProperties: true } },
          checklist: { type: 'array', items: { type: 'object', additionalProperties: true } },
        },
        ['model', 'blocked', 'compliance'],
      ),
      render(_args, value) {
        const lines = [
          `${value.model} — ${value.artifactKind} → ${value.repoId ?? '(no repoId given)'}`,
          '',
          `Licence: ${value.compliance.license}${value.compliance.licenseVerified ? ' (verified)' : ' (UNVERIFIED)'}`,
          `Redistributable: ${value.compliance.redistributable}   Commercial: ${value.compliance.commercialUse}`,
        ]
        if (value.blocked) {
          lines.push('', `BLOCKED — ${value.compliance.reason}`)
        } else {
          lines.push('', `Files: ${value.files.map((file) => file.path).join(', ')}`)
          if (value.commands.length > 0) {
            lines.push('', 'Upload:')
            for (const command of value.commands) {
              lines.push(`  ${command.command}`)
              if (command.note) lines.push(`    ${command.note}`)
            }
          }
        }
        const required = value.checklist.filter((item) => item.required && item.note?.startsWith('BLOCKED') !== true)
        if (required.length > 0) {
          lines.push('', 'Before you push:', ...bulletList(required.map((item) => item.item)))
        }
        return renderSummary('Publication set', lines, value.warnings?.length ? `\n${bulletList(value.warnings).join('\n')}` : undefined)
      },
    },

    async execute(args) {
      const filter = args ?? {}
      const resolved = resolveModel(catalog, filter)
      if (resolved.error !== undefined) throw failure(resolved.error, resolved.hint)

      const published = buildPublish({
        catalog,
        model: resolved.model,
        artifactKind: filter.artifactKind ?? 'adapter',
        target: filter.target ?? 'hf',
        repoId: filter.repoId,
        baseModelId: filter.baseModelId,
        evalResults: filter.evalResults ?? [],
        author: filter.author,
        adapterPath: filter.adapterPath,
      })
      if (published.error !== undefined) {
        throw failure(published.error, `Try: ${published.alternatives?.join(', ')}`)
      }
      return published
    },
  }
}
