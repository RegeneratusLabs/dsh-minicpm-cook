/**
 * A minimal fake plugin context.
 *
 * The real harness is not running here and `cordis_inspect_query` is not
 * available, so this stands in for it: it records what `apply()` registers, which
 * is enough to prove the bundle registers the tools and skills it claims, with
 * well-formed schemas, without installing anything into a profile.
 */
import { apply } from '../index.js'

/** Build a fake context and run the bundle against it. */
export function activate() {
  const tools = new Map()
  const skills = new Map()
  const disposers = []
  const errors = []

  const ctx = {
    effect(callback) {
      try {
        const disposer = callback()
        if (typeof disposer !== 'function') {
          errors.push(new Error('effect callback did not return a disposer'))
        } else {
          disposers.push(disposer)
        }
        return disposer
      } catch (error) {
        errors.push(error)
        throw error
      }
    },
    tools: {
      register(definition) {
        if (tools.has(definition.name)) throw new Error(`duplicate tool ${definition.name}`)
        tools.set(definition.name, definition)
        return () => tools.delete(definition.name)
      },
    },
    skills: {
      register(skill) {
        if (skills.has(skill.name)) throw new Error(`duplicate skill ${skill.name}`)
        skills.set(skill.name, skill)
        return () => skills.delete(skill.name)
      },
    },
    inject(_deps, callback) {
      return callback(this)
    },
    get() {
      return undefined
    },
  }

  apply(ctx, {})

  return { ctx, tools, skills, disposers, errors }
}

/** Call a registered tool and return its canonical value. */
export async function runTool(tools, name, args = {}) {
  const tool = tools.get(name)
  if (tool === undefined) throw new Error(`tool ${name} is not registered`)
  return tool.execute(args, { signal: undefined })
}

/** Render a tool result through its declared renderer. */
export function renderTool(tools, name, args, value) {
  const tool = tools.get(name)
  return tool.output.render(args, value)
}
