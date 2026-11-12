/**
 * Serving command generation.
 *
 * The output is one copy-pasteable launch plus a smoke test, with memory-safe
 * defaults derived from the requested context rather than from a guess. Where a
 * backend has a documented quirk for this family (add_special_tokens on vLLM,
 * min_p on llama.cpp, the tool-call parser on SGLang) the generated command
 * carries it and a warning explains why.
 */
import { estimateInference } from './estimate.js'

/** Supported serving backends. */
export const BACKENDS = ['transformers', 'vllm', 'sglang', 'llama_cpp', 'ollama', 'lmstudio', 'mlx']

/** Quantization levels accepted for the `quant` option. */
export const QUANTS = ['none', 'fp8', 'int4', 'awq', 'gptq', 'gguf']

/**
 * Build a serve command.
 *
 * @param options.model catalog entry.
 * @param options.backend one of BACKENDS.
 * @param options.gpuCount devices.
 * @param options.vramGb per-device VRAM.
 * @param options.contextTokens context to serve; capped to the trained length.
 * @param options.quant one of QUANTS.
 * @param options.toolCalling emit a tool-call parser where supported.
 * @returns { command, files, smokeTest, memory, warnings } or { error }.
 */
export function buildServe(options) {
  const {
    model,
    backend = 'vllm',
    gpuCount = 1,
    vramGb = 24,
    contextTokens,
    quant = 'none',
    toolCalling = false,
    port,
  } = options

  if (!BACKENDS.includes(backend)) {
    return { error: `unknown backend "${backend}"`, alternatives: BACKENDS }
  }

  const entry = model.frameworks?.[backend]
  if (entry === undefined || entry.supported !== true) {
    return {
      error: `${model.displayName} has no documented ${backend} path`,
      alternatives: BACKENDS.filter((candidate) => model.frameworks?.[candidate]?.supported === true),
    }
  }

  const warnings = []
  const ctxLen = Math.min(contextTokens ?? Math.min(8192, model.contextTokens), model.contextTokens)
  const quantKey = quant === 'none' ? 'bf16' : quant
  const memory = estimateInference({ model, quant: quantKey, seqLen: ctxLen, gpuCount, vramGb })

  if (!memory.fits) {
    warnings.push(
      `Estimated ${memory.perGpuGb} GB/GPU at ${ctxLen} tokens with ${quant}; ${gpuCount}×${vramGb} GB is not enough. ` +
      'Lower --max-model-len, move to a quantized variant, or add devices.',
    )
  }
  if (gpuCount > 1 && !['vllm', 'sglang'].includes(backend)) {
    warnings.push(`${backend} does not shard across devices the way vLLM/SGLang do; the command assumes one device.`)
  }
  if (entry.notes !== undefined) warnings.push(entry.notes)
  if (model.confidence === 'unverified') {
    warnings.push(`Catalog entry for ${model.displayName} is unverified: ${model.confidenceNote ?? 'confirm the launch flags.'}`)
  }

  const recipe = renderBackend({ model, backend, entry, ctxLen, gpuCount, vramGb, quant, toolCalling, port })
  if (Array.isArray(recipe.warnings)) warnings.push(...recipe.warnings)
  return {
    model: model.id,
    backend,
    command: recipe.command,
    files: recipe.files ?? [],
    smokeTest: recipe.smokeTest,
    sampling: model.sampling ?? { temperature: 1.0, top_p: 0.95, min_p: 0.0 },
    memory: {
      estimate: true,
      perGpuGb: memory.perGpuGb,
      totalGb: memory.totalGb,
      breakdownGb: memory.breakdownGb,
      kvKbPerToken: memory.kvKbPerToken,
      contextTokens: ctxLen,
      quant,
    },
    warnings,
    notes: [
      'Memory figures are ESTIMATEs; measure with your own load before committing capacity.',
      'Run the smoke test before wiring the endpoint into anything else.',
    ],
  }
}

/** Render the command and auxiliary files for one backend. */
function renderBackend({ model, backend, entry, ctxLen, gpuCount, vramGb, quant, toolCalling, port }) {
  const needsTrust = model.trustRemoteCode ? ' --trust-remote-code' : ''
  const hfId = model.hfId

  if (backend === 'vllm') {
    const flags = [
      `vllm serve ${hfId}`,
      `--port ${port ?? 8000}`,
      `--max-model-len ${ctxLen}`,
      `--gpu-memory-utilization 0.9`,
    ]
    if (gpuCount > 1) flags.push(`--tensor-parallel-size ${gpuCount}`)
    if (model.trustRemoteCode) flags.push('--trust-remote-code')
    const command = flags.join(' \\\n  ')

    const warnings = []
    if (model.id.startsWith('minicpm4')) {
      warnings.push(
        'vLLM chat requests drop BOS by default. Send extra_body={"add_special_tokens": true} or the prompt is ' +
        'mis-framed and quality drops without an error.',
      )
    }
    return {
      command,
      warnings,
      smokeTest: `curl -s http://localhost:${port ?? 8000}/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"${hfId}","messages":[{"role":"user","content":"Who are you?"}],"max_tokens":128,"temperature":1.0,"top_p":0.95}'`,
    }
  }

  if (backend === 'sglang') {
    const flags = [
      `python -m sglang.launch_server`,
      `--model-path ${hfId}`,
      `--port ${port ?? 30000}`,
      `--context-length ${ctxLen}`,
      `--mem-fraction-static 0.9`,
    ]
    if (gpuCount > 1) flags.push(`--tp ${gpuCount}`)
    if (model.trustRemoteCode) flags.push('--trust-remote-code')
    if (toolCalling && entry.toolCallParser !== undefined) flags.push(`--tool-call-parser ${entry.toolCallParser}`)
    return {
      command: flags.join(' \\\n  '),
      smokeTest: `curl -s http://localhost:${port ?? 30000}/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"${hfId}","messages":[{"role":"user","content":"Who are you?"}],"max_tokens":128}'`,
    }
  }

  if (backend === 'llama_cpp') {
    const gguf = model.variants.find((variant) => variant.kind === 'gguf')?.hfId ?? 'openbmb/REPLACE_WITH_GGUF_REPO'
    return {
      command: `llama-server -m ${shortName(model)}-Q4_K_M.gguf \\\n  -a ${model.displayName} --port ${port ?? 8080} -ngl 99 -c ${ctxLen} --jinja`,
      warnings: [
        `Download the GGUF from ${gguf} and substitute the exact filename; quantization suffixes differ per release.`,
        'llama.cpp defaults min_p=0.05, which can trap small models in repetition. Pass min_p=0.0 per request.',
      ],
      smokeTest: `curl -s http://localhost:${port ?? 8080}/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"${model.displayName}","messages":[{"role":"user","content":"1+1=?"}],"temperature":1.0,"top_p":0.95,"min_p":0.0}'`,
    }
  }

  if (backend === 'ollama') {
    const modelfile = `# Generated by dsh-minicpm-cook
FROM ${model.variants.find((variant) => variant.kind === 'gguf')?.hfId ?? 'hf.co/openbmb/REPLACE_WITH_GGUF'}

PARAMETER temperature 1.0
PARAMETER top_p 0.95
PARAMETER min_p 0.0
PARAMETER num_ctx ${ctxLen}
`
    return {
      command: 'ollama create minicpm-custom -f Modelfile && ollama run minicpm-custom',
      files: [{ path: 'Modelfile', content: modelfile }],
      warnings: [
        'Check whether an official library tag already exists (for example minicpm-v4.6) before building a custom one.',
        'num_ctx above the model default multiplies KV memory; raise it only as far as you need.',
      ],
      smokeTest: `curl -s http://localhost:11434/api/chat -d '{"model":"minicpm-custom","messages":[{"role":"user","content":"hi"}],"stream":false}'`,
    }
  }

  if (backend === 'lmstudio') {
    return {
      command: '# In LM Studio: search the model, download the GGUF, then Start Server on port 1234',
      warnings: [
        'LM Studio serves an OpenAI-compatible endpoint; set the context length in the model load settings.',
      ],
      smokeTest: `curl -s http://localhost:1234/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"${model.displayName}","messages":[{"role":"user","content":"hi"}]}'`,
    }
  }

  if (backend === 'mlx') {
    return {
      command: `mlx_lm.server --model ${model.variants.find((variant) => variant.kind === 'mlx')?.hfId ?? `mlx-community/${model.displayName}`} --port ${port ?? 8080}`,
      warnings: ['MLX runs on Apple Silicon only; the published MLX weights are 4-bit.'],
      smokeTest: `curl -s http://localhost:${port ?? 8080}/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"${model.displayName}","messages":[{"role":"user","content":"hi"}]}'`,
    }
  }

  // transformers: a runnable snippet, because there is no server to launch.
  const python = `from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "${hfId}"
tok = AutoTokenizer.from_pretrained(model_id${model.trustRemoteCode ? ', trust_remote_code=True' : ''})
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",${model.trustRemoteCode ? '\n    trust_remote_code=True,' : ''}
)
messages = [{"role": "user", "content": "Who are you? Please briefly introduce yourself."}]
inputs = tok.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
${model.chatTemplate?.applyArgs?.enable_thinking === true ? '    enable_thinking=True,\n' : ''}).to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=1.0, top_p=0.95)
print(tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
`
  return {
    command: 'python serve_transformers.py',
    files: [{ path: 'serve_transformers.py', content: python }],
    warnings: model.trustRemoteCode
      ? ['trust_remote_code=True executes modeling code shipped in the repo; pin a revision for reproducible serving.']
      : [],
    smokeTest: 'Run the script; it prints one sampled answer.',
  }
}

/** Short model name used as a local filename stem. */
function shortName(model) {
  return model.displayName.replace(/\s+/gu, '-')
}
