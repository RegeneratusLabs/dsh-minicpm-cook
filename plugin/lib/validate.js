/**
 * Dataset validation for MiniCPM fine-tuning.
 *
 * The point of this module is to fail loudly before a run wastes GPU hours: a
 * missing `messages` key, a dataset that ends on a user turn, an unexpanded
 * `*.jpg` glob, or a record that silently exceeds the context window are all
 * cheap to catch here and expensive to discover at epoch 2.
 *
 * Nothing is written; large inputs are streamed and bounded by `limit`.
 */
import { estimateTokens } from './estimate.js'

/** Accepted record shapes. */
export const FORMATS = ['messages', 'alpaca', 'dpo', 'kto', 'vlm-swift']

const ROLES = new Set(['system', 'user', 'assistant', 'tool', 'ipython'])

/**
 * Validate a set of records.
 *
 * @param options.records parsed records.
 * @param options.format one of FORMATS.
 * @param options.maxSeqLen token budget per record; omitted skips the check.
 * @param options.fileExists predicate used for image path checks.
 * @param options.limit cap on records examined.
 * @returns a report with per-record issues and aggregate statistics.
 */
export function validateRecords(options) {
  const {
    records,
    format = 'messages',
    maxSeqLen,
    fileExists = () => true,
    limit = 5000,
  } = options

  if (!FORMATS.includes(format)) {
    return { error: `unknown format "${format}"`, alternatives: FORMATS }
  }

  const examined = records.slice(0, limit)
  const issues = []
  const tokenLengths = []
  const issueCounts = Object.create(null)

  examined.forEach((record, index) => {
    const report = validateOne(record, { format, maxSeqLen, fileExists })
    issues.push(...report.issues.map((issue) => ({ index, ...issue })))
    for (const issue of report.issues) {
      issueCounts[issue.code] = (issueCounts[issue.code] ?? 0) + 1
    }
    if (report.tokens > 0) tokenLengths.push(report.tokens)
  })

  const overLimit = maxSeqLen === undefined
    ? 0
    : tokenLengths.filter((tokens) => tokens > maxSeqLen).length

  const badIndexes = new Set(issues.filter((issue) => issue.severity !== 'warn').map((issue) => issue.index))

  return {
    ok: badIndexes.size === 0,
    format,
    total: records.length,
    examined: examined.length,
    valid: examined.length - badIndexes.size,
    invalid: badIndexes.size,
    truncated: records.length > examined.length,
    issues: issues.slice(0, 200),
    issueCounts,
    stats: {
      estTokens: percentileSummary(tokenLengths),
      overLimit,
      maxSeqLen: maxSeqLen ?? null,
      imageRecords: examined.filter((record) => Array.isArray(record?.images) && record.images.length > 0).length,
    },
    notes: [
      'Token counts are a character-based estimate, not a tokenizer result; expect ±10%.',
      issues.length > 200 ? `Showing the first 200 of ${issues.length} issues.` : undefined,
    ].filter(Boolean),
  }
}

/** Validate a single record. */
function validateOne(record, { format, maxSeqLen, fileExists }) {
  const issues = []

  if (record === null || typeof record !== 'object' || Array.isArray(record)) {
    return {
      tokens: 0,
      issues: [{
        code: 'NOT_AN_OBJECT',
        severity: 'error',
        message: 'record is not a JSON object',
        fix: 'Each line of a JSONL dataset must be one object.',
      }],
    }
  }

  let text = ''

  if (format === 'alpaca') {
    if (typeof record.instruction !== 'string' || record.instruction.trim() === '') {
      issues.push(issue('MISSING_FIELD', 'instruction is missing or empty', 'Add the instruction text.'))
    }
    if (typeof record.output !== 'string' || record.output.trim() === '') {
      issues.push(issue('MISSING_FIELD', 'output is missing or empty', 'Add the target response as `output`.'))
    }
    text = [record.instruction, record.input, record.output].filter((part) => typeof part === 'string').join('\n')
    return finish(text, issues, maxSeqLen)
  }

  if (format === 'dpo') {
    for (const key of ['chosen', 'rejected']) {
      if (typeof record[key] !== 'string' || record[key].trim() === '') {
        issues.push(issue('MISSING_FIELD', `${key} is missing or empty`, 'DPO needs both a chosen and a rejected response.'))
      }
    }
    if (record.instruction === undefined && record.prompt === undefined && record.messages === undefined) {
      issues.push(issue('MISSING_FIELD', 'no prompt field (instruction/prompt/messages)', 'Add the prompt the pair responds to.'))
    }
    text = [record.instruction, record.input, record.chosen, record.rejected]
      .filter((part) => typeof part === 'string').join('\n')
    return finish(text, issues, maxSeqLen)
  }

  // messages / kto / vlm-swift all carry a messages array.
  const messages = record.messages
  if (!Array.isArray(messages) || messages.length === 0) {
    return {
      tokens: 0,
      issues: [issue(
        'MISSING_MESSAGES',
        '`messages` is missing, empty, or not an array',
        format === 'vlm-swift'
          ? 'Use [{"messages":[...],"images":["a.jpg"]}] — the VLM schema nests messages under a top-level array entry.'
          : 'Each record needs a non-empty `messages` array.',
      )],
    }
  }

  if (format === 'kto' && typeof record.label !== 'boolean') {
    issues.push(issue('MISSING_FIELD', '`label` must be true or false', 'KTO scores each exchange as good (true) or bad (false).'))
  }

  let lastRole
  let sawAssistant = false

  messages.forEach((message, position) => {
    if (message === null || typeof message !== 'object') {
      issues.push(issue('BAD_MESSAGE', `messages[${position}] is not an object`, 'Each message needs role and content.'))
      return
    }
    if (!ROLES.has(message.role)) {
      issues.push(issue(
        'BAD_ROLE',
        `messages[${position}].role is ${JSON.stringify(message.role)}`,
        `Use one of: ${[...ROLES].join(', ')}.`,
      ))
    }
    if (typeof message.content !== 'string' || message.content.trim() === '') {
      issues.push(issue(
        'EMPTY_CONTENT',
        `messages[${position}].content is empty`,
        'Empty turns waste tokens and teach the model to answer nothing.',
      ))
    } else {
      text += `${message.content}\n`
    }
    if (message.role === 'assistant') sawAssistant = true
    if (lastRole !== undefined && lastRole === message.role && message.role !== 'tool') {
      issues.push(issue(
        'ROLES_NOT_ALTERNATING',
        `messages[${position}] repeats role "${message.role}"`,
        'Merge the consecutive turns or insert the missing reply.',
      ))
    }
    lastRole = message.role
  })

  if (!sawAssistant) {
    issues.push(issue(
      'NO_ASSISTANT',
      'record has no assistant turn',
      'A supervised record must contain at least one assistant message; otherwise it contributes no loss.',
    ))
  }
  if (lastRole === 'user') {
    issues.push(issue(
      'ROLES_NOT_ALTERNATING',
      'record ends on a user turn',
      'A trailing user turn means no target for the generation; the record trains nothing.',
      'warn',
    ))
  }

  if (format === 'vlm-swift' || Array.isArray(record.images)) {
    issues.push(...checkImages(record.images, fileExists))
    if (format === 'vlm-swift' && !String(record.messages?.[0]?.content ?? '').includes('<image>')) {
      issues.push(issue(
        'IMAGE_TOKEN_MISSING',
        'no <image> placeholder in the first user turn',
        'The VLM template needs an <image> token where the image is injected.',
        'warn',
      ))
    }
  }

  return finish(text, issues, maxSeqLen)
}

/** Check that image references are concrete, existing paths. */
function checkImages(images, fileExists) {
  const issues = []
  if (images === undefined) return issues
  if (!Array.isArray(images)) {
    return [issue('BAD_IMAGES', '`images` is not an array', 'Use an array of paths or {path} objects.')]
  }
  images.forEach((image, position) => {
    if (typeof image === 'string') {
      if (/[*?[\]]/u.test(image)) {
        issues.push(issue(
          'IMAGE_GLOB_UNEXPANDED',
          `images[${position}] is a glob pattern: ${image}`,
          'Expand globs before training; loaders read literal paths.',
        ))
      } else if (!fileExists(image)) {
        issues.push(issue(
          'IMAGE_MISSING',
          `images[${position}] does not exist: ${image}`,
          'Fix the path or mount the dataset directory.',
        ))
      }
    } else if (image !== null && typeof image === 'object') {
      const path = image.path
      if (typeof path !== 'string') {
        issues.push(issue('BAD_IMAGES', `images[${position}] has no path`, 'Use {"path": "..."} or bytes.'))
      } else if (!fileExists(path)) {
        issues.push(issue('IMAGE_MISSING', `images[${position}].path does not exist: ${path}`, 'Fix the path.'))
      }
    } else {
      issues.push(issue('BAD_IMAGES', `images[${position}] is ${typeof image}`, 'Use a path string or {path}.'))
    }
  })
  return issues
}

function finish(text, issues, maxSeqLen) {
  const tokens = estimateTokens(text)
  if (maxSeqLen !== undefined && tokens > maxSeqLen) {
    issues.push(issue(
      'TOO_LONG',
      `estimated ${tokens} tokens exceeds maxSeqLen ${maxSeqLen}`,
      'Shorten the record or raise cutoff_len; silent truncation loses the answer most of the time.',
      'warn',
    ))
  }
  return { tokens, issues }
}

function issue(code, message, fix, severity = 'error') {
  return { code, severity, message, fix }
}

/** Percentile summary for a list of token lengths. */
function percentileSummary(values) {
  if (values.length === 0) return { p50: 0, p95: 0, max: 0 }
  const sorted = [...values].sort((a, b) => a - b)
  return {
    p50: sorted[Math.floor(sorted.length * 0.5)],
    p95: sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))],
    max: sorted[sorted.length - 1],
  }
}

/**
 * Split JSONL text into records, reporting the first parse failure.
 * @returns { records } or { error }.
 */
export function parseJsonl(text) {
  const records = []
  const lines = text.split(/\r?\n/u)
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index].trim()
    if (line === '') continue
    try {
      records.push(JSON.parse(line))
    } catch (error) {
      return { error: `line ${index + 1} is not valid JSON: ${error.message}` }
    }
  }
  return { records }
}

/**
 * Parse a file that is either JSONL or a JSON array.
 * @returns { records } or { error }.
 */
export function parseDataset(text) {
  const trimmed = text.trim()
  if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
    try {
      const parsed = JSON.parse(trimmed)
      if (Array.isArray(parsed)) return { records: parsed }
      // A single object is a one-record dataset.
      return { records: [parsed] }
    } catch {
      // fall through to JSONL: a JSONL file often starts with "{" per line
    }
  }
  return parseJsonl(text)
}
