/**
 * `minicpm_serve` — emit a copy-pasteable launch command plus a smoke test.
 */
import { resolveModel } from '../catalog.js'
import { buildServe, BACKENDS, QUANTS } from '../serve.js'
import { bulletList, bool, failure, int, num, outputSchema, pick, renderSummary, str } from '../toolkit.js'

export function createServeTool({ catalog }) {
  return {
    name: 'minicpm_serve',
    description:
      'Generate the exact command to serve a MiniCPM model on a chosen backend (vLLM, SGLang, llama.cpp, ' +
      'Ollama, transformers, LM Studio, MLX), with a memory estimate and a curl smoke test. Includes the ' +
      'per-model quirks that otherwise cause silent quality loss.',

    parameters: {
      type: 'object',
      properties: {
        modelId: str('Catalog id, e.g. "minicpm-v-4.6". Omit to resolve from family/modality.'),
        family: str('Model line to resolve when modelId is omitted.'),
        modality: pick('Modality to resolve when modelId is omitted.', ['any', 'text', 'vision', 'omni', 'ocr']),
        backend: pick('Serving runtime.', BACKENDS),
        gpuCount: int('Number of GPUs to serve on.', { minimum: 1 }),
        vramGb: num('Usable VRAM per GPU, in GB.', { minimum: 1 }),
        contextTokens: int('Context length to serve; capped to the model maximum.', { minimum: 512 }),
        quant: pick('Weight precision. "none" means the model default.', QUANTS),
        toolCalling: bool('Emit a tool-call parser where the backend supports one (SGLang).'),
        port: int('Port to bind.', { minimum: 1, maximum: 65535 }),
      },
      additionalProperties: false,
    },

    output: {
      schema: outputSchema(
        {
          model: str('Resolved catalog id.'),
          backend: str('Serving runtime.'),
          command: str('Launch command.'),
          smokeTest: str('Request that proves the server answers.'),
          memory: { type: 'object', additionalProperties: true },
          warnings: { type: 'array', items: { type: 'string' } },
        },
        ['model', 'backend', 'command'],
      ),
      render(_args, value) {
        const lines = [`${value.model} on ${value.backend}`, '', value.command, '']
        if (value.memory) {
          lines.push(
            `Memory: ${value.memory.perGpuGb} GB/GPU at ${value.memory.contextTokens} tokens ` +
            `(${value.memory.quant}, ${value.memory.kvKbPerToken} KB KV per token) — ESTIMATE`,
          )
        }
        if (value.files?.length > 0) lines.push('', `Also generated: ${value.files.map((file) => file.path).join(', ')}`)
        lines.push('', 'Smoke test:', value.smokeTest)
        if (value.warnings?.length > 0) lines.push('', 'Before you rely on it:', ...bulletList(value.warnings))
        return renderSummary('', lines)
      },
    },

    async execute(args) {
      const filter = args ?? {}
      const resolved = resolveModel(catalog, filter)
      if (resolved.error !== undefined) throw failure(resolved.error, resolved.hint)

      const served = buildServe({
        model: resolved.model,
        backend: filter.backend ?? 'vllm',
        gpuCount: filter.gpuCount ?? 1,
        vramGb: filter.vramGb ?? 24,
        contextTokens: filter.contextTokens,
        quant: filter.quant ?? 'none',
        toolCalling: filter.toolCalling ?? false,
        port: filter.port,
      })
      if (served.error !== undefined) {
        throw failure(served.error, `Supported here: ${served.alternatives?.join(', ') ?? 'none'}`)
      }
      return { ...served, displayName: resolved.model.displayName }
    },
  }
}
