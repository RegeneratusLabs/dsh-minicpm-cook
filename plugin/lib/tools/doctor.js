/**
 * `minicpm_doctor` — compare this host against a model's recorded requirements.
 *
 * Read-only: one bounded shell probe, no installs, no mutations. When no shell
 * executor is mounted the tool still answers, and says why it cannot.
 */
import { resolveModel } from '../catalog.js'
import { versionPins } from '../plan.js'
import { bulletList, failure, int, outputSchema, pick, renderSummary, str } from '../toolkit.js'

/**
 * One probe command. Kept to a single invocation so a doctor call costs one
 * process, not ten, and so partial failure is reported rather than thrown.
 */
const PROBE = `set +e
echo "SHELL_OK=1"
echo "PY_BIN=$(command -v python3 2>/dev/null || echo none)"
if command -v python3 >/dev/null 2>&1; then
python3 - <<'PY' 2>/dev/null || echo "PY_PROBE=failed"
from importlib.metadata import version, PackageNotFoundError
for name in ["torch","transformers","trl","peft","datasets","accelerate","vllm","sglang","bitsandbytes","flash-attn","huggingface_hub","modelscope"]:
    try:
        print("PKG_" + name + "=" + version(name))
    except PackageNotFoundError:
        print("PKG_" + name + "=none")
    except Exception:
        print("PKG_" + name + "=unknown")
PY
fi
echo "NVSMI_BIN=$(command -v nvidia-smi 2>/dev/null || echo none)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/GPU=/'
fi
DF_FREE=$(df -Pk . 2>/dev/null | tail -1 | awk '{print $4}')
echo "DISK_FREE_KB=\${DF_FREE:-unknown}"
`

export function createDoctorTool({ catalog, getShell }) {
  return {
    name: 'minicpm_doctor',
    description:
      'Inspect this machine against MiniCPM requirements: Python and package versions, GPU model and VRAM, ' +
      'and free disk. Reports pass/warn/fail per check with a concrete fix. Read-only; installs nothing.',

    parameters: {
      type: 'object',
      properties: {
        modelId: str('Catalog id to check requirements against. Omit for a generic report.'),
        family: str('Model line to resolve when modelId is omitted.'),
        modality: pick('Modality to resolve when modelId is omitted.', ['any', 'text', 'vision', 'omni', 'ocr']),
        workdir: str('Directory to probe and report free space for.'),
        timeoutMs: int('Probe timeout in milliseconds.', { minimum: 1000 }),
      },
      additionalProperties: false,
    },

    output: {
      schema: outputSchema(
        {
          available: { type: 'boolean', description: 'False when no shell executor is mounted.' },
          checks: { type: 'array', items: { type: 'object', additionalProperties: true } },
          verdict: str('One-word summary: ready, degraded, or blocked.'),
        },
        ['available'],
      ),
      render(_args, value) {
        if (value.available === false) {
          return renderSummary('Cannot probe this host.', [
            value.reason,
            'No shell executor is mounted in this profile.',
          ])
        }
        const icon = { pass: 'ok  ', warn: 'warn', fail: 'FAIL' }
        const lines = (value.checks ?? []).map((check) => {
          const found = check.found === undefined ? '' : ` (found ${check.found}${check.required ? `, need ${check.required}` : ''})`
          return `${icon[check.status] ?? check.status} ${check.id}${found}${check.fix ? `\n       ${check.fix}` : ''}`
        })
        return renderSummary(`Environment: ${value.verdict}`, lines, value.note)
      },
    },

    async execute(args, exec) {
      const filter = args ?? {}
      const shell = getShell()
      if (shell === undefined) {
        return {
          available: false,
          reason: 'No shell executor is available, so package and GPU versions cannot be read.',
          checks: [],
          verdict: 'unknown',
        }
      }

      let model
      if (filter.modelId !== undefined || filter.family !== undefined || filter.modality !== undefined) {
        const resolved = resolveModel(catalog, filter)
        if (resolved.error !== undefined) throw failure(resolved.error, resolved.hint)
        model = resolved.model
      }

      const spec = shell.resolve({
        command: PROBE,
        ...(filter.workdir === undefined ? {} : { workdir: filter.workdir }),
        timeoutMs: filter.timeoutMs ?? 20000,
        stdoutMaxBytes: 256 * 1024,
        signal: exec?.signal,
      })
      const execution = await shell.execute(spec)
      const result = await execution.result()
      const stdout = result.stdout?.text ?? ''

      if (result.timedOut) {
        return {
          available: true,
          checks: [{ id: 'probe', status: 'warn', fix: 'The probe timed out; raise timeoutMs or check for a slow shell init.' }],
          verdict: 'degraded',
          note: `Timed out after ${result.timeoutMs} ms.`,
        }
      }

      const facts = parseProbe(stdout)
      const checks = buildChecks({ facts, model })
      const verdict = checks.some((check) => check.status === 'fail')
        ? 'blocked'
        : checks.some((check) => check.status === 'warn') ? 'degraded' : 'ready'

      return {
        available: true,
        model: model?.id,
        facts,
        checks,
        verdict,
        note: model === undefined
          ? 'No model given: checked only for a usable Python/GPU environment.'
          : `Requirements taken from the catalog entry for ${model.displayName}.`,
        stderr: result.stderr?.text?.slice(0, 2000) || undefined,
      }
    },
  }
}

/** Parse `KEY=value` probe lines, keeping repeated GPU lines. */
export function parseProbe(stdout) {
  const facts = { packages: {}, gpus: [] }
  for (const line of stdout.split(/\r?\n/u)) {
    if (line.startsWith('GPU=')) {
      const [name, memory] = line.slice(4).split(',').map((part) => part.trim())
      const gb = Number.parseInt(String(memory ?? '').replace(/[^0-9]/gu, ''), 10)
      facts.gpus.push({ name, memoryMb: Number.isFinite(gb) ? gb : null, vramGb: Number.isFinite(gb) ? Math.round((gb / 1024) * 10) / 10 : null })
      continue
    }
    const index = line.indexOf('=')
    if (index <= 0) continue
    const key = line.slice(0, index)
    const value = line.slice(index + 1).trim()
    if (key.startsWith('PKG_')) facts.packages[key.slice(4)] = value
    else if (key === 'PY_BIN') facts.python = value
    else if (key === 'NVSMI_BIN') facts.nvidiaSmi = value
    else if (key === 'DISK_FREE_KB') facts.diskFreeGb = value === 'unknown' ? null : Math.round(Number(value) / 1024 / 1024 * 10) / 10
  }
  return facts
}

/** Turn facts into pass/warn/fail checks. */
export function buildChecks({ facts, model }) {
  const checks = []

  checks.push({
    id: 'python',
    status: facts.python && facts.python !== 'none' ? 'pass' : 'fail',
    found: facts.python ?? 'none',
    required: 'python3',
    fix: facts.python === 'none' || facts.python === undefined
      ? 'Install Python 3.10+; every MiniCPM training and serving path needs it.'
      : undefined,
  })

  checks.push({
    id: 'gpu',
    status: facts.gpus.length > 0 ? 'pass' : 'fail',
    found: facts.gpus.length > 0
      ? facts.gpus.map((gpu) => `${gpu.name} (${gpu.vramGb ?? '?'} GB)`).join(', ')
      : 'none detected',
    fix: facts.gpus.length === 0
      ? 'No NVIDIA GPU detected. Fine-tuning and CUDA serving need one; CPU-only GGUF inference is the fallback.'
      : undefined,
  })

  if (model !== undefined) {
    const totalVram = facts.gpus.reduce((sum, gpu) => sum + (gpu.vramGb ?? 0), 0)

    // PyTorch is mandatory for training whatever the version floor says, so check
    // its presence even when the catalog records no minimum for it.
    const torchVersion = facts.packages.torch
    if (torchVersion === undefined || torchVersion === 'none') {
      checks.push({
        id: 'torch',
        status: 'fail',
        found: torchVersion ?? 'not installed',
        required: 'any',
        fix: 'Install PyTorch matched to your CUDA driver, then the stack: pip install torch transformers trl peft datasets accelerate',
      })
    }

    for (const pin of versionPins(model, { needsTraining: true })) {
      const found = facts.packages[pin.framework === 'torch' ? 'torch' : pin.framework]
      if (found === undefined) continue
      if (found === 'none') {
        checks.push({
          id: `${pin.framework}>=${pin.version}`,
          status: pin.framework === 'vllm' || pin.framework === 'sglang' ? 'warn' : 'fail',
          found: 'not installed',
          required: pin.version,
          fix: `pip install "${pin.framework}>=${pin.version}"`,
        })
        continue
      }
      const ok = compareVersions(found, pin.version) >= 0
      checks.push({
        id: `${pin.framework}>=${pin.version}`,
        status: ok ? 'pass' : 'fail',
        found,
        required: pin.version,
        fix: ok ? undefined : `Upgrade: pip install -U "${pin.framework}>=${pin.version}"`,
      })
    }

    const floorGb = model.publishedHardware === undefined ? undefined : inferFloorGb(model)
    if (floorGb !== undefined) {
      checks.push({
        id: 'vram',
        status: totalVram >= floorGb ? 'pass' : 'warn',
        found: `${totalVram} GB total`,
        required: `${floorGb} GB`,
        fix: totalVram >= floorGb
          ? undefined
          : 'Below the documented floor for fine-tuning this model; expect to reduce sequence length or use a smaller model.',
      })
    }
  }

  if (typeof facts.diskFreeGb === 'number') {
    checks.push({
      id: 'disk',
      status: facts.diskFreeGb >= 30 ? 'pass' : facts.diskFreeGb >= 10 ? 'warn' : 'fail',
      found: `${facts.diskFreeGb} GB free`,
      required: '30 GB',
      fix: facts.diskFreeGb < 30
        ? 'Model weights plus checkpoints need tens of GB; free space or point HF_HOME at a larger volume.'
        : undefined,
    })
  }

  return checks
}

/**
 * A conservative VRAM floor from the documented prose, so the check has a
 * number to compare against. Returns undefined when no floor is recorded.
 */
function inferFloorGb(model) {
  const published = model.publishedHardware
  if (published === undefined) return undefined
  // Published values are prose like "at least two 20-series GPUs with 24GB VRAM".
  const match = /(\d+)\s*GB/iu.exec(String(published.lora ?? published.full ?? ''))
  return match === null ? undefined : Number(match[1])
}

/** Compare dotted numeric versions; non-numeric suffixes are ignored. */
export function compareVersions(found, required) {
  const parse = (value) => String(value).split('.').map((part) => Number.parseInt(part, 10) || 0)
  const a = parse(found)
  const b = parse(required)
  const length = Math.max(a.length, b.length)
  for (let index = 0; index < length; index += 1) {
    const left = a[index] ?? 0
    const right = b[index] ?? 0
    if (left !== right) return left - right
  }
  return 0
}
