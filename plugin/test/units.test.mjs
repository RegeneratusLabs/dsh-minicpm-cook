/**
 * Unit tests for the pure functions: the estimator, the validator, the probe
 * parser and the frontmatter reader. No harness, no I/O.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'

import { estimateInference, estimateTokens, estimateTraining, kvBytesPerToken, loraFraction } from '../lib/estimate.js'
import { validateRecords, parseDataset, parseJsonl } from '../lib/validate.js'
import { buildChecks, compareVersions, parseProbe } from '../lib/tools/doctor.js'
import { parseFrontmatter } from '../lib/skills.js'
import { loadCatalog, getModel, resolveModel, queryModels, bestBackend, licenseRules } from '../lib/catalog.js'

const catalog = loadCatalog()
const textModel = getModel(catalog, 'minicpm5-2b')
const vlm = getModel(catalog, 'minicpm-v-4.6')

test('memory ordering holds: QLoRA < LoRA < full', () => {
  const base = { model: textModel, seqLen: 2048, batch: 1, gpuCount: 1, vramGb: 24 }
  const qlora = estimateTraining({ ...base, method: 'qlora' })
  const lora = estimateTraining({ ...base, method: 'lora' })
  const full = estimateTraining({ ...base, method: 'full' })

  assert.ok(qlora.perGpuGb < lora.perGpuGb, 'QLoRA must be cheaper than LoRA')
  assert.ok(lora.perGpuGb < full.perGpuGb, 'LoRA must be cheaper than full fine-tuning')
  // Full fine-tuning pays optimizer state on every parameter, LoRA on ~1%.
  assert.ok(full.breakdownGb.optimizer > lora.breakdownGb.optimizer * 10)
})

test('estimates are labelled and expose their breakdown', () => {
  const estimate = estimateTraining({ model: textModel, method: 'lora', seqLen: 2048 })
  assert.equal(estimate.estimate, true)
  const sum = Object.values(estimate.breakdownGb).reduce((total, part) => total + part, 0)
  assert.ok(Math.abs(sum - estimate.totalGb) < 0.5, 'breakdown should sum to the total')
  assert.ok(estimate.utilizations.trainableFraction > 0.005)
  assert.ok(estimate.utilizations.trainableFraction < 0.02)
})

test('MoE weights use total parameters, activations use active ones', () => {
  const moe = getModel(catalog, 'minicpm-v-4.7-35b-a3b')
  const estimate = estimateTraining({ model: moe, method: 'lora', seqLen: 2048 })
  // 35B at bf16 is ~70 GB of weights; active params are 3B, so activations stay small.
  assert.ok(estimate.breakdownGb.weights > 60)
  assert.ok(estimate.breakdownGb.activations < 5)
})

test('KV cache matches the published architecture', () => {
  // 2 × 42 layers × 2 KV heads × 128 head_dim × 2 bytes
  assert.equal(kvBytesPerToken(textModel), 43_008)
  const inference = estimateInference({ model: textModel, seqLen: 32768, vramGb: 24 })
  assert.ok(inference.breakdownGb.kvCache > 1)
  assert.equal(inference.fits, true)
})

test('loraFraction grows with rank and stays bounded', () => {
  assert.ok(loraFraction(8) < loraFraction(16))
  assert.ok(loraFraction(16) > 0)
  assert.ok(loraFraction(100_000) <= 0.05)
  assert.equal(loraFraction(0), 0)
})

test('token estimate distinguishes scripts and never returns zero for real text', () => {
  const latin = estimateTokens('The quick brown fox jumps over the lazy dog.')
  const cjk = estimateTokens('这是一个用于测试分词估算的中文句子')
  assert.ok(latin > 0)
  assert.equal(estimateTokens(''), 0)
  // CJK packs more tokens per character than Latin text.
  assert.ok(cjk / 20 > latin / 44)
})

test('dataset validation reports every documented code', () => {
  const report = validateRecords({
    format: 'messages',
    records: [
      { messages: [{ role: 'user', content: 'q' }] },
      { messages: [{ role: 'wizard', content: 'q' }, { role: 'assistant', content: 'a' }] },
      { messages: [{ role: 'user', content: '' }, { role: 'assistant', content: 'a' }] },
      { messages: 'nope' },
      { messages: [{ role: 'user', content: 'q' }, { role: 'assistant', content: 'a' }], images: ['missing.jpg'] },
    ],
    fileExists: () => false,
  })
  const codes = new Set(report.issues.map((issue) => issue.code))
  for (const code of ['NO_ASSISTANT', 'BAD_ROLE', 'EMPTY_CONTENT', 'MISSING_MESSAGES', 'IMAGE_MISSING']) {
    assert.ok(codes.has(code), `expected ${code}`)
  }
  assert.equal(report.ok, false)
})

test('token overflow is a warning, not a hard error', () => {
  const report = validateRecords({
    format: 'messages',
    maxSeqLen: 5,
    records: [{ messages: [{ role: 'user', content: 'a'.repeat(500) }, { role: 'assistant', content: 'b'.repeat(500) }] }],
  })
  const overflow = report.issues.find((issue) => issue.code === 'TOO_LONG')
  assert.ok(overflow)
  assert.equal(overflow.severity, 'warn')
  assert.equal(report.ok, true)
})

test('alpaca and dpo formats are checked by their own rules', () => {
  const alpaca = validateRecords({ format: 'alpaca', records: [{ instruction: 'q' }] })
  assert.ok(alpaca.issues.some((issue) => issue.code === 'MISSING_FIELD'))

  const dpo = validateRecords({ format: 'dpo', records: [{ instruction: 'q', chosen: 'good' }] })
  assert.ok(dpo.issues.some((issue) => issue.message.includes('rejected')))
})

test('JSONL parsing reports the offending line', () => {
  const ok = parseJsonl('{"a":1}\n\n{"a":2}\n')
  assert.equal(ok.records.length, 2)

  const bad = parseJsonl('{"a":1}\nnot json\n')
  assert.match(bad.error, /line 2/)
})

test('a JSON array and a JSONL file both parse', () => {
  assert.equal(parseDataset('[{"a":1},{"a":2}]').records.length, 2)
  assert.equal(parseDataset('{"a":1}\n{"a":2}\n').records.length, 2)
})

test('probe parsing tolerates a missing tool and repeated GPU lines', () => {
  const facts = parseProbe([
    'SHELL_OK=1',
    'PY_BIN=/usr/bin/python3',
    'PKG_torch=2.8.0',
    'PKG_trl=none',
    'NVSMI_BIN=none',
    'GPU=NVIDIA RTX 4090, 24564 MiB',
    'GPU=NVIDIA RTX 4090, 24564 MiB',
    'DISK_FREE_KB=104857600',
  ].join('\n'))

  assert.equal(facts.python, '/usr/bin/python3')
  assert.equal(facts.packages.torch, '2.8.0')
  assert.equal(facts.packages.trl, 'none')
  assert.equal(facts.gpus.length, 2)
  assert.equal(facts.gpus[0].vramGb, 24)
  assert.equal(facts.diskFreeGb, 100)
})

test('doctor flags a host with no GPU and no torch', () => {
  const checks = buildChecks({
    facts: { packages: { torch: 'none' }, gpus: [], python: '/usr/bin/python3' },
    model: textModel,
  })
  const gpu = checks.find((check) => check.id === 'gpu')
  assert.equal(gpu.status, 'fail')
  assert.match(gpu.fix, /GGUF/)
  const torch = checks.find((check) => check.id.startsWith('torch'))
  assert.equal(torch.status, 'fail')
})

test('version comparison is numeric, not lexical', () => {
  assert.ok(compareVersions('5.10.0', '5.9.0') > 0)
  assert.ok(compareVersions('5.7.0', '5.7.0') === 0)
  assert.ok(compareVersions('4.57.3', '5.7.0') < 0)
  assert.ok(compareVersions('0.21', '0.5.16') > 0) // 21 > 5, which is why pinning matters
})

test('frontmatter parsing returns attributes and a clean body', () => {
  const { attributes, body } = parseFrontmatter('---\nname: demo\ndescription: A demo skill.\n---\n\n# Body\n')
  assert.equal(attributes.name, 'demo')
  assert.equal(attributes.description, 'A demo skill.')
  assert.ok(body.startsWith('# Body'))
})

test('a skill file without frontmatter keeps its whole body', () => {
  const { attributes, body } = parseFrontmatter('# Just markdown\n')
  assert.deepEqual(attributes, {})
  assert.equal(body, '# Just markdown\n')
})

test('catalog resolution prefers verified data, then the newest release', () => {
  const resolved = resolveModel(catalog, { family: 'minicpm-v' })
  assert.equal(resolved.model.confidence, 'verified')
  assert.equal(resolved.how, 'filter')

  const missing = resolveModel(catalog, { modelId: 'nope' })
  assert.match(missing.error, /unknown model id/)
  assert.ok(missing.available.includes('minicpm5-2b'))
})

test('backend preference falls back to whatever the model actually supports', () => {
  assert.equal(bestBackend(textModel).backend, 'vllm')
  // SALA has no vLLM path but does have one through the OpenBMB SGLang fork.
  const sala = getModel(catalog, 'minicpm-sala')
  assert.equal(bestBackend(sala).backend, 'sglang')
})

test('licence rules come from the catalog, with a blocking fallback', () => {
  assert.equal(licenseRules(catalog, textModel).redistributable, true)
  const unknown = licenseRules(catalog, { license: 'Proprietary-Untested' })
  assert.equal(unknown.redistributable, false)
})

test('filters compose', () => {
  const visionVllm = queryModels(catalog, { modality: 'vision', framework: 'vllm' })
  assert.ok(visionVllm.length > 0)
  assert.ok(visionVllm.every((model) => model.modality === 'vision'))

  assert.equal(queryModels(catalog, { maxParamsB: 0.6 }).every((model) => model.paramsB <= 0.6), true)
})
