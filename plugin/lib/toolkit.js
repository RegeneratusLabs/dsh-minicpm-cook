/**
 * Shared helpers for tool definitions.
 *
 * Deliberately minimal: this bundle registers raw tool objects rather than
 * importing the harness's tool helpers, so it keeps working across harness
 * versions without an npm dependency on `@deepseek-ai/*`.
 */

/** One plain-text content block. */
export function text(value) {
  return [{ type: 'text', text: String(value) }]
}

/**
 * Render a bounded, human-readable summary.
 *
 * Long generated files are returned in the canonical value and summarised here,
 * so a model sees the shape of the result without a 400-line dump on every call.
 */
export function renderSummary(title, lines, footer) {
  const body = lines.filter((line) => line !== undefined && line !== null).join('\n')
  return text([title, body, footer].filter(Boolean).join('\n'))
}

/** A tool failure that reads as guidance rather than a stack trace. */
export function failure(message, hint) {
  const error = new Error(hint === undefined ? message : `${message}\n${hint}`)
  error.name = 'MinicpmCookError'
  return error
}

/** JSON Schema fragment: a string with a description. */
export function str(description, extra = {}) {
  return { type: 'string', description, ...extra }
}

/** JSON Schema fragment: an integer with a description. */
export function int(description, extra = {}) {
  return { type: 'integer', description, ...extra }
}

/** JSON Schema fragment: a number with a description. */
export function num(description, extra = {}) {
  return { type: 'number', description, ...extra }
}

/** JSON Schema fragment: a boolean with a description. */
export function bool(description, extra = {}) {
  return { type: 'boolean', description, ...extra }
}

/** JSON Schema fragment: an enumeration with a description. */
export function pick(description, values, extra = {}) {
  return { type: 'string', description, enum: values, ...extra }
}

/** A permissive output schema: the tool declares its real fields and allows more. */
export function outputSchema(properties, required = []) {
  return { type: 'object', properties, required, additionalProperties: true }
}

/** Standard bounds on string list lines. */
export function bulletList(items, marker = '-') {
  return items.filter(Boolean).map((item) => `${marker} ${item}`)
}

/** Truncate a long string for a summary without hiding that it happened. */
export function clip(value, max = 1200) {
  const text = String(value)
  return text.length <= max ? text : `${text.slice(0, max)}\n… [${text.length - max} more characters in the full result]`
}

/**
 * Make a renderer total.
 *
 * The harness calls `render` on a live card and on a replayed one; a renderer
 * that throws blanks the display and loses the result for the user. A guard that
 * returns an empty block degrades to "nothing to draw" instead.
 */
export function guardRender(render) {
  return (args, value) => {
    try {
      const blocks = render(args ?? {}, value ?? {})
      if (Array.isArray(blocks) && blocks.length > 0) return blocks
    } catch {
      // fall through to the empty block
    }
    return text('')
  }
}
