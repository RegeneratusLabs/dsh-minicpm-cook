/**
 * Integration tests: activate the bundle against a fake context and exercise
 * every tool the way the model would.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { activate, renderTool, runTool } from './harness.mjs'

const EXPECTED_TOOLS = [
  'minicpm_catalog',
  'minicpm_plan',
  'minicpm_recipe',
  'minicpm_check_dataset',
  'minicpm_doctor',
  'minicpm_serve',
  'minicpm_publish',
]

const EXPECTED_SKILLS = ['minicpm-family', 'minicpm-finetune', 'minicpm-serve', 'minicpm-publish']

test('registers the seven tools and four skills', () => {
  const { tools, skills, errors } = activate()
  assert.deepEqual(errors, [])
  assert.deepEqual([...tools.keys()].sort(), [...EXPECTED_TOOLS].sort())
  assert.deepEqual([...skills.keys()].sort(), [...EXPECTED_SKILLS].sort())
})

test('every tool has a usable definition', () => {
  const { tools } = activate()
  for (const [name, tool] of tools) {
    assert.equal(tool.name, name)
    assert.equal(typeof tool.description, 'string')
    assert.ok(tool.description.length > 40, `${name} description is too thin`)
    assert.equal(tool.parameters.type, 'object', `${name} parameters must be an object schema`)
    assert.equal(typeof tool.parameters.properties, 'object')
    assert.equal(typeof tool.execute, 'function')
    assert.equal(typeof tool.output.render, 'function')
    assert.equal(tool.output.schema.type, 'object', `${name} output schema must be an object`)
    // A renderer must return content blocks, not a bare string.
    const blocks = tool.output.render({}, {})
    assert.ok(Array.isArray(blocks))
    assert.ok(blocks.every((block) => block.type === 'text' && typeof block.text === 'string'))
  }
})

test('every skill carries the frontmatter fields the registry requires', () => {
  const { skills } = activate()
  for (const [skillName, skill] of skills) {
    assert.equal(skill.name, skillName)
    assert.ok(skill.description.length > 40, `${skillName} needs a routing-quality description`)
    assert.ok(skill.content.length > 400, `${skillName} body is too thin to be useful`)
    assert.deepEqual(skill.invocation, { modelInvocable: true, userInvocable: true })
    assert.ok(!skill.content.includes('---\nname:'), `${skillName} body still contains frontmatter`)
  }
})

test('catalog lists the family and filters by modality', async () => {
  const { tools } = activate()
  const all = await runTool(tools, 'minicpm_catalog', {})
  assert.ok(all.matched >= 15)

  const vision = await runTool(tools, 'minicpm_catalog', { modality: 'vision' })
  assert.ok(vision.matched >= 3)
  assert.ok(vision.models.every((model) => model.modality === 'vision'))

  const tiny = await runTool(tools, 'minicpm_catalog', { maxParamsB: 2 })
  assert.ok(tiny.models.every((model) => model.paramsB <= 2))

  const none = await runTool(tools, 'minicpm_catalog', { query: 'definitely-not-a-minicpm' })
  assert.equal(none.matched, 0)
  assert.match(renderTool(tools, 'minicpm_catalog', {}, none)[0].text, /No MiniCPM model matches/)
})

test('plan fits a 24 GB card and refuses to silently overcommit', async () => {
  const { tools } = activate()

  const roomy = await runTool(tools, 'minicpm_plan', {
    modelId: 'minicpm5-2b', goal: 'lora', gpuCount: 1, vramGb: 24,
  })
  assert.equal(roomy.model, 'minicpm5-2b')
  assert.equal(roomy.training.fits, true)
  assert.ok(roomy.steps.length >= 4)

  // 8B LoRA does not fit an 8 GB card, so the plan drops to QLoRA and says why.
  const cramped = await runTool(tools, 'minicpm_plan', {
    modelId: 'minicpm4.1-8b', goal: 'sft', gpuCount: 1, vramGb: 8,
  })
  assert.equal(cramped.recommendation.method, 'qlora')
  assert.equal(cramped.training.fits, true)
  assert.ok(cramped.warnings.some((warning) => warning.includes('LoRA does not fit')))

  // Nothing fits this one, and the plan says so instead of pretending.
  const impossible = await runTool(tools, 'minicpm_plan', {
    modelId: 'minicpm-v-4.7-35b-a3b', goal: 'sft', gpuCount: 1, vramGb: 8,
  })
  assert.equal(impossible.training.fits, false)
  assert.ok(impossible.warnings.some((warning) => warning.includes('Not even QLoRA fits')))
})

test('plan resolves a model from family and modality when no id is given', async () => {
  const { tools } = activate()
  const plan = await runTool(tools, 'minicpm_plan', { family: 'minicpm-v', gpuCount: 8, vramGb: 80 })
  assert.equal(plan.model, 'minicpm-v-4.7-35b-a3b')
  assert.equal(plan.resolvedBy, 'filter')
})

test('an unknown model id is an actionable failure, not a crash', async () => {
  const { tools } = activate()
  await assert.rejects(
    () => runTool(tools, 'minicpm_plan', { modelId: 'minicpm9' }),
    (error) => {
      assert.match(error.message, /unknown model id/)
      assert.match(error.message, /minicpm_catalog/)
      return true
    },
  )
})

test('TRL recipe carries assistant-only loss and compiles as Python', async () => {
  const { tools } = activate()
  const recipe = await runTool(tools, 'minicpm_recipe', {
    modelId: 'minicpm5-2b', toolchain: 'trl', method: 'lora',
    dataPath: '/tmp/train.jsonl', outputDir: '/tmp/out',
  })

  const script = recipe.files.find((file) => file.path === 'train_lora.py')
  assert.ok(script, 'expected train_lora.py')
  assert.match(script.content, /assistant_only_loss=True/)
  assert.match(script.content, /\{%- generation %\}/)
  assert.match(script.content, /target_modules=\["q_proj"/)
  assert.equal(recipe.defaults.loraRank, 16)

  // Syntax-level verification only: this host has no torch, but a script that
  // does not parse would fail before it ever reached an import.
  const directory = mkdtempSync(join(tmpdir(), 'minicpm-cook-'))
  const file = join(directory, 'train_lora.py')
  writeFileSync(file, script.content)
  execFileSync('python3', ['-m', 'py_compile', file], { stdio: 'pipe' })
})

test('a vision model is steered away from TRL, which has no VLM path', async () => {
  const { tools } = activate()
  await assert.rejects(
    () => runTool(tools, 'minicpm_recipe', { modelId: 'minicpm-v-4.6', toolchain: 'trl' }),
    (error) => {
      assert.match(error.message, /VLM/)
      assert.match(error.message, /llamafactory/)
      return true
    },
  )
})

test('LLaMA-Factory vision recipe unfreezes the vision tower', async () => {
  const { tools } = activate()
  const recipe = await runTool(tools, 'minicpm_recipe', {
    modelId: 'minicpm-v-4.6', toolchain: 'llamafactory', method: 'lora', downsample: '4x',
  })
  const yaml = recipe.files.find((file) => file.path.endsWith('.train.yaml'))
  assert.ok(yaml)
  assert.match(yaml.content, /freeze_vision_tower: false/)
  assert.match(yaml.content, /template: minicpm_v_4_6/)
  assert.match(yaml.content, /finetuning_type: lora/)

  const dataset = recipe.files.find((file) => file.path.endsWith('.dataset_info.json'))
  assert.ok(dataset)
  assert.equal(JSON.parse(dataset.content).minicpm_sft.formatting, 'sharegpt')
  assert.ok(recipe.warnings.some((warning) => warning.includes('DOWNSAMPLE_MODE')))
})

test('ms-swift recipe uses the catalog model_type', async () => {
  const { tools } = activate()
  const recipe = await runTool(tools, 'minicpm_recipe', {
    modelId: 'minicpm-v-4.6', toolchain: 'ms-swift',
  })
  const script = recipe.files[0]
  assert.match(script.content, /--model_type minicpmv4_6/)
  assert.match(script.content, /--packing false/)
  assert.match(script.content, /--freeze_vit false/)
})

test('dataset check accepts a valid messages file', async () => {
  const { tools } = activate()
  const report = await runTool(tools, 'minicpm_check_dataset', {
    records: [
      { messages: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'hello' }] },
      { messages: [{ role: 'system', content: 'be terse' }, { role: 'user', content: 'q' }, { role: 'assistant', content: 'a' }] },
    ],
    format: 'messages',
  })
  assert.equal(report.ok, true)
  assert.equal(report.total, 2)
  assert.equal(report.invalid, 0)
})

test('dataset check names the failures that waste GPU hours', async () => {
  const { tools } = activate()
  const report = await runTool(tools, 'minicpm_check_dataset', {
    records: [
      { messages: [{ role: 'user', content: 'no answer here' }] },
      { messages: [{ role: 'user', content: '<image>\nwhat?' }, { role: 'assistant', content: 'x' }], images: ['*.jpg'] },
      { messages: [{ role: 'user', content: 'a' }, { role: 'assistant', content: 'b' }, { role: 'assistant', content: 'c' }] },
      { nope: true },
    ],
    format: 'messages',
  })
  assert.equal(report.ok, false)
  const codes = new Set(report.issues.map((issue) => issue.code))
  assert.ok(codes.has('NO_ASSISTANT'))
  assert.ok(codes.has('IMAGE_GLOB_UNEXPANDED'))
  assert.ok(codes.has('ROLES_NOT_ALTERNATING'))
  assert.ok(codes.has('MISSING_MESSAGES'))
})

test('dataset check reads a JSONL file from disk', async () => {
  const { tools } = activate()
  const directory = mkdtempSync(join(tmpdir(), 'minicpm-cook-data-'))
  const file = join(directory, 'train.jsonl')
  writeFileSync(file, [
    JSON.stringify({ messages: [{ role: 'user', content: 'q' }, { role: 'assistant', content: 'a' }] }),
    JSON.stringify({ messages: [{ role: 'user', content: 'q2' }] }),
  ].join('\n'))

  const report = await runTool(tools, 'minicpm_check_dataset', { path: file, format: 'messages' })
  assert.equal(report.total, 2)
  assert.equal(report.invalid, 1)
})

test('doctor degrades honestly when no shell is mounted', async () => {
  const { tools } = activate()
  const report = await runTool(tools, 'minicpm_doctor', {})
  assert.equal(report.available, false)
  assert.match(report.reason, /shell/i)
  assert.match(renderTool(tools, 'minicpm_doctor', {}, report)[0].text, /Cannot probe this host/)
})

test('serve emits trust_remote_code only where the catalog requires it', async () => {
  const { tools } = activate()

  const plain = await runTool(tools, 'minicpm_serve', { modelId: 'minicpm5-2b', backend: 'vllm' })
  assert.ok(!plain.command.includes('--trust-remote-code'))
  assert.match(plain.command, /vllm serve openbmb\/MiniCPM5-2B/)
  assert.match(plain.smokeTest, /v1\/chat\/completions/)

  const custom = await runTool(tools, 'minicpm_serve', { modelId: 'minicpm4.1-8b', backend: 'vllm' })
  assert.match(custom.command, /--trust-remote-code/)
  assert.ok(custom.warnings.some((warning) => warning.includes('add_special_tokens')))
})

test('serve adds the SGLang tool-call parser and an Ollama Modelfile', async () => {
  const { tools } = activate()

  const sglang = await runTool(tools, 'minicpm_serve', { modelId: 'minicpm5-2b', backend: 'sglang', toolCalling: true })
  assert.match(sglang.command, /--tool-call-parser minicpm5/)

  const ollama = await runTool(tools, 'minicpm_serve', { modelId: 'minicpm-v-4.6', backend: 'ollama' })
  assert.equal(ollama.files[0].path, 'Modelfile')
  assert.match(ollama.files[0].content, /PARAMETER min_p 0.0/)
})

test('serve rejects a backend the model does not support', async () => {
  const { tools } = activate()
  await assert.rejects(
    () => runTool(tools, 'minicpm_serve', { modelId: 'minicpm-sala', backend: 'vllm' }),
    (error) => {
      assert.match(error.message, /no documented vllm path/)
      assert.match(error.message, /sglang/)
      return true
    },
  )
})

test('publish refuses to print upload commands for an unverified licence', async () => {
  const { tools } = activate()
  const report = await runTool(tools, 'minicpm_publish', {
    modelId: 'minicpm-ocr-1.0', repoId: 'alice/ocr-sql',
  })
  assert.equal(report.blocked, true)
  assert.equal(report.compliance.licenseVerified, false)
  assert.deepEqual(report.commands, [])
  assert.match(renderTool(tools, 'minicpm_publish', {}, report)[0].text, /BLOCKED/)
})

test('publish emits a card, a NOTICE and upload commands for Apache-2.0', async () => {
  const { tools } = activate()
  const report = await runTool(tools, 'minicpm_publish', {
    modelId: 'minicpm5-2b',
    repoId: 'alice/minicpm5-2b-sql',
    author: 'Alice',
    evalResults: [{ benchmark: 'sql-exec', baseScore: 0.61, score: 0.84 }],
  })
  assert.equal(report.blocked, false)

  const paths = report.files.map((file) => file.path)
  assert.ok(paths.includes('README.md'))
  assert.ok(paths.includes('NOTICE'))
  assert.ok(paths.includes('ADAPTER-USAGE.md'))

  const card = report.files.find((file) => file.path === 'README.md').content
  assert.match(card, /license: apache-2\.0/)
  assert.match(card, /base_model: openbmb\/MiniCPM5-2B/)
  assert.match(card, /library_name: peft/)
  assert.match(card, /0\.23/) // measured delta, not a placeholder
  assert.match(card, /NOTICE/)

  const notice = report.files.find((file) => file.path === 'NOTICE').content
  assert.match(notice, /OpenBMB/)
  assert.match(notice, /THUNLP/)

  assert.ok(report.commands.some((command) => command.target === 'huggingface'))
  assert.match(report.commands[0].command, /huggingface-cli upload alice\/minicpm5-2b-sql/)
})
