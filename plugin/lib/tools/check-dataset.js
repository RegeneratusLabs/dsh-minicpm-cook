/**
 * `minicpm_check_dataset` — validate training data before spending GPU time.
 *
 * Reads one file (bounded) or inline records, then reports per-record issues and
 * token statistics. Never writes.
 */
import { resolveModel } from '../catalog.js'
import { parseDataset, validateRecords, FORMATS } from '../validate.js'
import { bulletList, failure, int, outputSchema, pick, renderSummary, str } from '../toolkit.js'

export function createCheckDatasetTool({ catalog, readText, fileExists }) {
  return {
    name: 'minicpm_check_dataset',
    description:
      'Validate a MiniCPM fine-tuning dataset against the expected schema: message roles, assistant turns, ' +
      'image references, and token length against the training budget. Returns per-record issues with fixes.',

    parameters: {
      type: 'object',
      properties: {
        path: str('Dataset file to read (JSONL or a JSON array).'),
        records: {
          type: 'array',
          description: 'Inline records, for small checks that are already in context.',
          items: { type: 'object', additionalProperties: true },
        },
        format: pick('Record shape to validate against.', FORMATS),
        modelId: str('Model the data is for; used for the context length and image rules.'),
        maxSeqLen: int('Token budget per record.', { minimum: 128 }),
        limit: int('Maximum records to examine.', { minimum: 1 }),
      },
      additionalProperties: false,
    },

    output: {
      schema: outputSchema(
        {
          ok: { type: 'boolean', description: 'True when no record has an error.' },
          total: int('Records seen.'),
          invalid: int('Records with at least one error.'),
          issues: { type: 'array', items: { type: 'object', additionalProperties: true } },
          stats: { type: 'object', additionalProperties: true },
        },
        ['ok', 'total', 'issues'],
      ),
      render(_args, value) {
        if (value.error !== undefined) return renderSummary('Cannot read that dataset.', [value.error])

        const lines = [
          `${value.total} record(s) — ${value.valid} valid, ${value.invalid} with errors`,
        ]
        if (value.stats) {
          const p = value.stats.estTokens
          lines.push(`Tokens (estimate): p50 ${p.p50}, p95 ${p.p95}, max ${p.max}${value.stats.maxSeqLen ? ` / budget ${value.stats.maxSeqLen}` : ''}`)
          if (value.stats.overLimit > 0) lines.push(`${value.stats.overLimit} record(s) exceed the budget.`)
        }
        const byCode = Object.entries(value.issueCounts ?? {}).sort((a, b) => b[1] - a[1])
        if (byCode.length > 0) {
          lines.push('', 'Issue counts:', ...byCode.map(([code, count]) => `  ${code} × ${count}`))
        }
        if (value.issues?.length > 0) {
          lines.push('', 'First issues:')
          for (const issue of value.issues.slice(0, 12)) {
            lines.push(`  [${issue.index}] ${issue.code}: ${issue.message}`)
            if (issue.fix) lines.push(`        fix: ${issue.fix}`)
          }
          if (value.issues.length > 12) lines.push(`  … ${value.issues.length - 12} more in the full result`)
        }
        return renderSummary(value.ok ? 'Dataset looks usable.' : 'Dataset has problems.', lines)
      },
    },

    async execute(args) {
      const filter = args ?? {}
      let records = filter.records

      if (records === undefined) {
        if (typeof filter.path !== 'string' || filter.path === '') {
          throw failure('give either `path` or `records`')
        }
        let text
        try {
          text = await readText(filter.path)
        } catch (error) {
          throw failure(`cannot read ${filter.path}`, error.message)
        }
        const parsed = parseDataset(text)
        if (parsed.error !== undefined) throw failure(`cannot parse ${filter.path}`, parsed.error)
        records = parsed.records
      }

      if (!Array.isArray(records)) throw failure('`records` must be an array')

      let model
      if (filter.modelId !== undefined) {
        const resolved = resolveModel(catalog, { modelId: filter.modelId })
        if (resolved.error !== undefined) throw failure(resolved.error, resolved.hint)
        model = resolved.model
      }

      const report = validateRecords({
        records,
        format: filter.format ?? (model?.modality !== undefined && model.modality !== 'text' ? 'vlm-swift' : 'messages'),
        maxSeqLen: filter.maxSeqLen ?? (model === undefined ? undefined : Math.min(4096, model.contextTokens)),
        limit: filter.limit ?? 5000,
        fileExists,
      })
      if (report.error !== undefined) throw failure(report.error, `Try: ${report.alternatives?.join(', ')}`)

      return {
        ...report,
        model: model?.id,
        notes: [
          ...(report.notes ?? []),
          model === undefined ? undefined : `Validated against ${model.displayName}.`,
          model?.modality !== 'text' && model !== undefined
            ? `Visual token budget is set at runtime (DOWNSAMPLE_MODE=${model.vision?.downsampleModes?.join('|') ?? 'n/a'}); token counts here exclude image tokens.`
            : undefined,
        ].filter(Boolean),
        hints: report.ok ? [] : bulletList(distinctFixes(report.issues)),
      }
    },
  }
}

/** Distinct fix strings, so the summary does not repeat itself. */
function distinctFixes(issues) {
  const seen = new Set()
  const fixes = []
  for (const issue of issues) {
    if (issue.fix && !seen.has(issue.fix)) {
      seen.add(issue.fix)
      fixes.push(issue.fix)
    }
  }
  return fixes.slice(0, 8)
}
