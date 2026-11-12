/**
 * Catalog access: load the bundled model catalog once and answer queries over it.
 *
 * The catalog is the single source of truth for every tool in this bundle. It is
 * frozen on load so a tool cannot accidentally mutate shared state between calls.
 */
import { readFileSync } from 'node:fs'

/** Path of the bundled catalog, resolved relative to this module. */
export const CATALOG_URL = new URL('../data/models.json', import.meta.url)

/**
 * Load and freeze the catalog.
 * @param url override, used by tests.
 * @returns the frozen catalog object.
 */
export function loadCatalog(url = CATALOG_URL) {
  const raw = readFileSync(url, 'utf8')
  return deepFreeze(JSON.parse(raw))
}

/**
 * Find one model by exact id.
 * @param catalog loaded catalog.
 * @param id model id, e.g. `minicpm5-2b`.
 * @returns the model, or undefined.
 */
export function getModel(catalog, id) {
  return catalog.models.find((model) => model.id === id)
}

/** Every model id, in catalog order. */
export function knownIds(catalog) {
  return catalog.models.map((model) => model.id)
}

/**
 * Resolve a model from an explicit id or from a family/modality pair.
 *
 * The tools accept either form so an agent that only knows "a vision model" does
 * not have to guess an id first. When several models match, the newest release
 * wins, which is what a user asking for "a MiniCPM VLM" almost always means.
 *
 * @returns { model, alternatives, how } or { error, available }.
 */
export function resolveModel(catalog, filter = {}) {
  const { modelId, family, modality } = filter

  if (modelId !== undefined && modelId !== null && modelId !== '') {
    const model = getModel(catalog, modelId)
    if (model === undefined) {
      return {
        error: `unknown model id "${modelId}"`,
        available: knownIds(catalog),
        hint: 'call minicpm_catalog to list models',
      }
    }
    return { model, alternatives: [], how: 'id' }
  }

  const candidates = catalog.models.filter((model) => {
    if (family !== undefined && family !== 'any' && model.family !== family) return false
    if (modality !== undefined && modality !== 'any' && model.modality !== modality) return false
    return true
  })

  if (candidates.length === 0) {
    return {
      error: `no model matches family="${family ?? 'any'}" modality="${modality ?? 'any'}"`,
      available: knownIds(catalog),
      hint: 'relax the filter or call minicpm_catalog',
    }
  }

  const ranked = [...candidates].sort((a, b) => {
    // Verified data beats inferred data, then newest release wins.
    if (a.confidence !== b.confidence) return a.confidence === 'verified' ? -1 : 1
    return String(b.releaseDate).localeCompare(String(a.releaseDate))
  })

  return {
    model: ranked[0],
    alternatives: ranked.slice(1).map((model) => model.id),
    how: 'filter',
  }
}

/**
 * Query the catalog with a filter.
 * @returns matching models, newest first within a family.
 */
export function queryModels(catalog, filter = {}) {
  const {
    query,
    family = 'any',
    modality = 'any',
    maxParamsB,
    license,
    framework,
  } = filter
  const needle = typeof query === 'string' ? query.trim().toLowerCase() : ''

  return catalog.models
    .filter((model) => {
      if (family !== 'any' && model.family !== family) return false
      if (modality !== 'any' && model.modality !== modality) return false
      if (typeof maxParamsB === 'number' && model.paramsB > maxParamsB) return false
      if (license !== undefined && model.license !== license) return false
      if (framework !== undefined && model.frameworks?.[framework]?.supported !== true) return false
      if (needle.length > 0) {
        const haystack = [
          model.id, model.displayName, model.family, model.modality,
          model.hfId, model.bestFor, model.architecture,
        ].join(' ').toLowerCase()
        if (!haystack.includes(needle)) return false
      }
      return true
    })
    .sort((a, b) => String(b.releaseDate).localeCompare(String(a.releaseDate)))
}

/**
 * Pick the best backend for a model given a preference order.
 * @returns { backend, entry } of the first supported backend found.
 */
export function bestBackend(model, preference = ['vllm', 'sglang', 'transformers', 'llama_cpp', 'ollama']) {
  for (const backend of preference) {
    const entry = model.frameworks?.[backend]
    if (entry?.supported === true) return { backend, entry }
  }
  const fallback = Object.entries(model.frameworks ?? {}).find(([, entry]) => entry?.supported === true)
  return fallback === undefined
    ? { backend: undefined, entry: undefined }
    : { backend: fallback[0], entry: fallback[1] }
}

/** License rules for a model, with an explicit unknown fallback. */
export function licenseRules(catalog, model) {
  return catalog.licenses?.[model.license] ?? catalog.licenses?.other ?? {
    redistributable: false,
    commercialUse: false,
    attributionRequired: true,
    notes: 'Unknown license; treat redistribution as blocked.',
    sourceUrl: null,
  }
}

/** Recursively freeze an object graph. */
function deepFreeze(value) {
  if (value === null || typeof value !== 'object' || Object.isFrozen(value)) return value
  Object.freeze(value)
  for (const child of Object.values(value)) deepFreeze(child)
  return value
}
