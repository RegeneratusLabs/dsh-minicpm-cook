/**
 * Embedded skill loading.
 *
 * Skills ship as `SKILL.md` files inside this package and are registered with the
 * harness skill registry at activation, so a user gets them without cloning
 * anything or configuring a skill directory.
 */
import { readFileSync, readdirSync } from 'node:fs'

/** Skill directories, in the order they should appear in a catalog. */
export const SKILL_DIRS = ['minicpm-family', 'minicpm-finetune', 'minicpm-serve', 'minicpm-publish']

/**
 * Parse YAML frontmatter from a SKILL.md body.
 *
 * Only the two scalar keys the skill format requires are read here; a general
 * YAML parser would be a dependency for no benefit, since the body is opaque
 * markdown and the frontmatter is flat.
 *
 * @returns { attributes, body }.
 */
export function parseFrontmatter(source) {
  const normalized = source.replace(/^\uFEFF/u, '')
  if (!normalized.startsWith('---')) return { attributes: {}, body: normalized }

  const end = normalized.indexOf('\n---', 3)
  if (end === -1) return { attributes: {}, body: normalized }

  const header = normalized.slice(3, end)
  const body = normalized.slice(end + 4).replace(/^(?:\r?\n)+/u, '')
  const attributes = {}

  for (const line of header.split(/\r?\n/u)) {
    const match = /^([A-Za-z][A-Za-z0-9_-]*):\s*(.*)$/u.exec(line)
    if (match === null) continue
    const [, key, rawValue] = match
    attributes[key] = rawValue.replace(/^['"]|['"]$/gu, '').trim()
  }

  return { attributes, body }
}

/**
 * Load every bundled skill.
 *
 * @param baseUrl directory URL containing the skill folders.
 * @returns array of { name, description, content }.
 * @throws when a skill file is missing or lacks required frontmatter, because a
 *         silently absent skill is indistinguishable from a broken install.
 */
export function loadSkills(baseUrl) {
  const skills = []
  for (const directory of SKILL_DIRS) {
    const url = new URL(`${directory}/SKILL.md`, baseUrl)
    const source = readFileSync(url, 'utf8')
    const { attributes, body } = parseFrontmatter(source)

    if (typeof attributes.name !== 'string' || attributes.name === '') {
      throw new Error(`skill ${directory} is missing a "name" in its frontmatter`)
    }
    if (typeof attributes.description !== 'string' || attributes.description === '') {
      throw new Error(`skill ${directory} is missing a "description" in its frontmatter`)
    }

    skills.push({
      name: attributes.name,
      description: attributes.description,
      content: body,
    })
  }
  return skills
}

/** List the skill files present, used by the package self-test. */
export function listSkillFiles(baseUrl) {
  const root = new URL('.', baseUrl)
  return readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
}
