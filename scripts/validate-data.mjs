#!/usr/bin/env node
/**
 * Validate data/models.json against data/schema.json plus the referential rules
 * a plain JSON Schema cannot express.
 *
 * Dependency-free on purpose: this runs on a host with no npm install step.
 *
 *   node scripts/validate-data.mjs
 *
 * Exit codes: 0 valid, 1 validation failure, 2 unreadable input.
 */
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const root = join(here, '..')

/** Known framework keys, so a typo cannot silently create an unusable row. */
const FRAMEWORKS = new Set([
  'transformers', 'vllm', 'sglang', 'llama_cpp', 'ollama', 'lmstudio', 'mlx',
  'arcLight', 'arclight', 'litert', 'cpmcu', 'tgi', 'lmdeploy', 'xinference',
  'trl', 'llamafactory', 'ms_swift', 'unsloth', 'xtuner', 'align_anything',
])

/** Chat-template names the generators understand. */
const TEMPLATES = new Set(['chatml', 'cpm3', 'minicpm-v', 'minicpm-o', 'custom'])

/**
 * Minimal JSON Schema validator for the subset this repo uses.
 * Supports: type, required, properties, additionalProperties, items, enum,
 * minimum, maximum, minItems, pattern.
 */
function validate(node, value, path, errors) {
  if (node === undefined || node === null) return

  const types = Array.isArray(node.type) ? node.type : node.type === undefined ? [] : [node.type]
  if (types.length > 0 && !types.some((type) => matchesType(type, value))) {
    errors.push(`${path}: expected ${types.join('|')}, got ${describe(value)}`)
    return
  }

  if (Array.isArray(node.enum) && !node.enum.some((candidate) => candidate === value)) {
    errors.push(`${path}: ${JSON.stringify(value)} is not one of ${JSON.stringify(node.enum)}`)
  }

  if (typeof value === 'number') {
    if (typeof node.minimum === 'number' && value < node.minimum) {
      errors.push(`${path}: ${value} is below minimum ${node.minimum}`)
    }
    if (typeof node.maximum === 'number' && value > node.maximum) {
      errors.push(`${path}: ${value} is above maximum ${node.maximum}`)
    }
  }

  if (typeof value === 'string') {
    if (node.pattern !== undefined && !new RegExp(node.pattern, 'u').test(value)) {
      errors.push(`${path}: ${JSON.stringify(value)} does not match ${node.pattern}`)
    }
    // `minItems` on a string is a schema bug; treat empty strings as unacceptable
    // wherever a non-empty string is clearly intended.
    if (value.trim().length === 0 && (node.minItems !== undefined || node.pattern === undefined)) {
      errors.push(`${path}: must not be an empty string`)
    }
  }

  if (Array.isArray(value)) {
    if (typeof node.minItems === 'number' && value.length < node.minItems) {
      errors.push(`${path}: needs at least ${node.minItems} item(s), has ${value.length}`)
    }
    if (node.items !== undefined) {
      value.forEach((item, index) => validate(node.items, item, `${path}[${index}]`, errors))
    }
  }

  if (isPlainObject(value)) {
    if (Array.isArray(node.required)) {
      for (const key of node.required) {
        if (!Object.hasOwn(value, key)) errors.push(`${path}: missing required property "${key}"`)
      }
    }
    const properties = node.properties ?? {}
    for (const [key, child] of Object.entries(value)) {
      if (Object.hasOwn(properties, key)) {
        validate(properties[key], child, `${path}.${key}`, errors)
      } else if (node.additionalProperties === false) {
        errors.push(`${path}: unexpected property "${key}"`)
      } else if (isPlainObject(node.additionalProperties)) {
        validate(node.additionalProperties, child, `${path}.${key}`, errors)
      }
    }
  }
}

function matchesType(type, value) {
  switch (type) {
    case 'object': return isPlainObject(value)
    case 'array': return Array.isArray(value)
    case 'string': return typeof value === 'string'
    case 'number': return typeof value === 'number' && Number.isFinite(value)
    case 'integer': return typeof value === 'number' && Number.isInteger(value)
    case 'boolean': return typeof value === 'boolean'
    case 'null': return value === null
    default: return false
  }
}

function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function describe(value) {
  if (value === null) return 'null'
  if (Array.isArray(value)) return 'array'
  return typeof value
}

/** Rules a JSON Schema cannot state. */
function checkReferences(catalog, errors) {
  const seen = new Set()
  for (const model of catalog.models ?? []) {
    const where = `models[${model.id ?? '?'}]`

    if (seen.has(model.id)) errors.push(`${where}: duplicate id`)
    seen.add(model.id)

    if (!Object.hasOwn(catalog.licenses ?? {}, model.license)) {
      errors.push(`${where}.license: "${model.license}" has no entry in licenses{}`)
    }
    if (model.licenseVerified === true && !model.licenseSourceUrl) {
      errors.push(`${where}: licenseVerified is true but licenseSourceUrl is empty`)
    }
    if (model.confidence === 'unverified' && !model.confidenceNote) {
      errors.push(`${where}: confidence is unverified but confidenceNote is missing`)
    }

    for (const key of Object.keys(model.frameworks ?? {})) {
      if (!FRAMEWORKS.has(key)) errors.push(`${where}.frameworks: unknown framework "${key}"`)
    }

    if (!TEMPLATES.has(model.chatTemplate?.name)) {
      errors.push(`${where}.chatTemplate.name: "${model.chatTemplate?.name}" is not a known template`)
    }

    if (model.modality !== 'text' && model.vision === undefined && model.family !== 'minicpm-o') {
      errors.push(`${where}: non-text modality should declare a vision{} block`)
    }

    // A model advertised as installable must name at least one working backend.
    const usable = Object.values(model.frameworks ?? {}).some((entry) => entry?.supported === true)
    if (!usable) errors.push(`${where}: no framework is marked supported`)

    for (const variant of model.variants ?? []) {
      if (!variantsAreKnown(model, variant.kind)) {
        errors.push(`${where}.variants: unknown variant kind "${variant.kind}"`)
      }
    }
  }
}

const VARIANT_KINDS = new Set([
  'sft', 'base', 'midtrain', 'gguf', 'mlx', 'gptq', 'awq', 'marlin', 'int4',
  'fp8', 'qat', 'draft', 'ternary', 'verifier',
])

function variantsAreKnown(_model, kind) {
  return VARIANT_KINDS.has(kind)
}

async function main() {
  let schema
  let catalog
  try {
    schema = JSON.parse(await readFile(join(root, 'data', 'schema.json'), 'utf8'))
    catalog = JSON.parse(await readFile(join(root, 'data', 'models.json'), 'utf8'))
  } catch (error) {
    console.error(`cannot read catalog: ${error.message}`)
    process.exit(2)
  }

  const errors = []
  validate(schema, catalog, 'catalog', errors)
  if (errors.length === 0) checkReferences(catalog, errors)

  if (errors.length > 0) {
    console.error(`data/models.json FAILED validation (${errors.length} problem(s)):`)
    for (const error of errors) console.error(`  - ${error}`)
    process.exit(1)
  }

  // The installable package carries a copy of the catalog; a stale copy means the
  // bundle ships advice the repository already corrected.
  try {
    const shipped = await readFile(join(root, 'plugin', 'data', 'models.json'), 'utf8')
    if (shipped !== await readFile(join(root, 'data', 'models.json'), 'utf8')) {
      console.error('plugin/data/models.json is stale; run: node scripts/sync-data.mjs')
      process.exit(1)
    }
  } catch (error) {
    if (error.code === 'ENOENT') {
      console.error('plugin/data/models.json is missing; run: node scripts/sync-data.mjs')
        process.exit(1)
    }
    throw error
  }

  const verified = catalog.models.filter((model) => model.confidence === 'verified').length
  console.log(
    `data/models.json OK: ${catalog.models.length} models ` +
    `(${verified} verified, ${catalog.models.length - verified} unverified), ` +
    `${Object.keys(catalog.licenses).length} license rule(s).`,
  )
}

await main()
