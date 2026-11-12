#!/usr/bin/env node
/**
 * Copy the canonical catalog into the installable package.
 *
 * `data/models.json` at the repository root is the file humans edit and review.
 * `plugin/data/models.json` is what the bundle reads at runtime. Two files means
 * they can drift, so `scripts/validate-data.mjs` fails when they differ and this
 * script is the only supported way to move one to the other.
 *
 *   node scripts/sync-data.mjs
 */
import { copyFile, readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const root = join(here, '..')

const source = join(root, 'data', 'models.json')
const target = join(root, 'plugin', 'data', 'models.json')

await copyFile(source, target)

const [a, b] = await Promise.all([readFile(source, 'utf8'), readFile(target, 'utf8')])
if (a !== b) {
  console.error('sync failed: copies differ after copy')
  process.exit(1)
}

console.log(`synced data/models.json -> plugin/data/models.json (${(a.length / 1024).toFixed(1)} KiB)`)
