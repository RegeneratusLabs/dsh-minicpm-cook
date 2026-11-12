#!/usr/bin/env node
/**
 * Detect drift between this catalog and OpenBMB's repository.
 *
 * MiniCPM ships faster than any cookbook can track — five releases in 2026 alone.
 * The failure mode is not a broken build, it is advice that quietly stops being
 * true. This script turns that into a failing check.
 *
 *   node scripts/check-upstream.mjs           # compare against the lock file
 *   node scripts/check-upstream.mjs --write   # accept current upstream as the new lock
 *
 * Exit codes: 0 in sync, 1 drift, 2 network or API failure.
 */
import { readFile, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const root = join(here, '..')
const lockPath = join(root, 'data', 'upstream-lock.json')
const write = process.argv.includes('--write')

const API = 'https://api.github.com/repos/OpenBMB/MiniCPM'
const HEADERS = {
  accept: 'application/vnd.github+json',
  'user-agent': 'dsh-minicpm-cook-drift-check',
  ...(process.env.GITHUB_TOKEN === undefined ? {} : { authorization: `Bearer ${process.env.GITHUB_TOKEN}` }),
}

/** Fetch JSON with a hard per-request timeout. */
async function getJson(url) {
  const response = await fetch(url, { headers: HEADERS, signal: AbortSignal.timeout(20000) })
  if (!response.ok) throw new Error(`${url} returned ${response.status} ${response.statusText}`)
  return response.json()
}

/** List a directory's entries as {type, name, sha}. */
async function listDir(path) {
  const entries = await getJson(`${API}/contents/${path}`)
  if (!Array.isArray(entries)) throw new Error(`${path} is not a directory`)
  return entries.map((entry) => ({ name: entry.name, type: entry.type, sha: entry.sha }))
}

async function main() {
  let current
  try {
    const repo = await getJson(API)
    const [skills, deployment, finetune] = await Promise.all([
      listDir('skills'),
      listDir('docs/deployment'),
      listDir('docs/finetune'),
    ])
    current = {
      repo: 'OpenBMB/MiniCPM',
      defaultBranch: repo.default_branch,
      pushedAt: repo.pushed_at,
      skills: skills.map((entry) => entry.name).sort(),
      deployment: deployment.map((entry) => entry.name).sort(),
      finetune: finetune.map((entry) => entry.name).sort(),
    }
  } catch (error) {
    console.error(`cannot reach GitHub: ${error.message}`)
    console.error('Set GITHUB_TOKEN if you are rate limited; this check is optional and needs network.')
    process.exit(2)
  }

  if (write) {
    await writeFile(lockPath, `${JSON.stringify(current, null, 2)}\n`)
    console.log(`lock written: ${current.skills.length} skills, ${current.deployment.length} deployment docs, ${current.finetune.length} finetune docs`)
    return
  }

  let previous
  try {
    previous = JSON.parse(await readFile(lockPath, 'utf8'))
  } catch {
    console.error('no data/upstream-lock.json yet; run: node scripts/check-upstream.mjs --write')
    process.exit(2)
  }

  const drift = []
  for (const key of ['skills', 'deployment', 'finetune']) {
    const added = current[key].filter((name) => !previous[key].includes(name))
    const removed = previous[key].filter((name) => !current[key].includes(name))
    if (added.length > 0) drift.push(`${key}: added ${added.join(', ')}`)
    if (removed.length > 0) drift.push(`${key}: removed ${removed.join(', ')}`)
  }

  if (current.pushedAt !== previous.pushedAt) {
    drift.push(`upstream HEAD moved: ${previous.pushedAt} -> ${current.pushedAt}`)
  }

  if (drift.length === 0) {
    console.log(`in sync with OpenBMB/MiniCPM (${current.skills.length} skills, ${current.deployment.length} deployment docs, ${current.finetune.length} finetune docs)`)
    return
  }

  console.error('upstream drift detected:')
  for (const line of drift) console.error(`  - ${line}`)
  console.error('\nReview the new material, update data/models.json, then: node scripts/check-upstream.mjs --write')
  process.exit(1)
}

await main()
