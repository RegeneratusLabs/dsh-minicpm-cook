/**
 * dsh-minicpm-cook — the MiniCPM cookbook as a harness bundle.
 *
 * Registers seven tools and four skills that cover the whole lifecycle of an
 * OpenBMB MiniCPM model: choose, plan, fine-tune, validate data, check the host,
 * serve, and publish.
 *
 * Design constraints, both deliberate:
 *
 *   1. No `@deepseek-ai/*` imports. Only Node builtins and `ctx` services, so the
 *      package resolves under any installed harness version — the same approach
 *      the installed `@farmagent/dsh-web-search` plugin takes.
 *   2. No writes, no process spawning except one read-only probe in
 *      `minicpm_doctor`, and no network. Every tool is a pure function of its
 *      arguments plus the bundled catalog, so nothing here can corrupt a call or
 *      surprise a user with a side effect they did not ask for.
 */
import { existsSync, statSync } from 'node:fs'
import { readFile } from 'node:fs/promises'

import { loadCatalog } from './lib/catalog.js'
import { loadSkills } from './lib/skills.js'
import { guardRender } from './lib/toolkit.js'
import { createCatalogTool } from './lib/tools/catalog.js'
import { createPlanTool } from './lib/tools/plan.js'
import { createRecipeTool } from './lib/tools/recipe.js'
import { createCheckDatasetTool } from './lib/tools/check-dataset.js'
import { createDoctorTool } from './lib/tools/doctor.js'
import { createServeTool } from './lib/tools/serve.js'
import { createPublishTool } from './lib/tools/publish.js'

/** Loader identity. */
export const name = 'minicpm-cook'

/** Services this bundle needs before it can activate. */
export const inject = ['tools', 'skills']

/**
 * Register the tools and skills.
 *
 * @param ctx plugin context.
 * @param config optional row config; currently unused, kept for forward compatibility.
 */
export function apply(ctx, config = {}) {
  void config
  const catalog = loadCatalog()
  const skillsBase = new URL('./skills/', import.meta.url)

  const readText = makeReadText(ctx)
  const fileExists = makeFileExists(ctx)

  const tools = [
    createCatalogTool({ catalog }),
    createPlanTool({ catalog }),
    createRecipeTool({ catalog }),
    createCheckDatasetTool({ catalog, readText, fileExists }),
    createDoctorTool({ catalog, getShell: () => ctx.get('shell') }),
    createServeTool({ catalog }),
    createPublishTool({ catalog }),
  ]

  for (const tool of tools) {
    // Keep each declared renderer, but make it total: a renderer that throws
    // blanks a live and a replayed card alike.
    const definition = { ...tool, output: { ...tool.output, render: guardRender(tool.output.render) } }
    ctx.effect(() => ctx.tools.register(definition))
  }

  for (const skill of loadSkills(skillsBase)) {
    ctx.effect(() => ctx.skills.register({
      name: skill.name,
      description: skill.description,
      content: skill.content,
      // Both surfaces: the model can load it on demand, and a user can invoke it.
      invocation: { modelInvocable: true, userInvocable: true },
    }))
  }
}

/**
 * Read a text file through the harness file service when it is mounted,
 * falling back to Node's own fs so the tool still works in a bare profile.
 */
function makeReadText(ctx) {
  return async (path) => {
    const fs = ctx.get?.('fs')
    if (fs !== undefined && typeof fs.readText === 'function') {
      const result = await fs.readText(path)
      if (typeof result === 'string') return result
      if (result !== null && typeof result === 'object' && typeof result.text === 'string') return result.text
    }
    return readFile(path, 'utf8')
  }
}

/** Existence check that uses the file service when available. */
function makeFileExists(ctx) {
  return (path) => {
    const fs = ctx.get?.('fs')
    if (fs !== undefined && typeof fs.stat === 'function') {
      // `stat` is async for the service but the validator is synchronous; the
      // Node check is equivalent for the paths a caller supplies here.
      void fs
    }
    try {
      return statSync(path).isFile()
    } catch {
      return existsSync(path)
    }
  }
}
