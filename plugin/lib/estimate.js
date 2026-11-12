/**
 * Memory and token estimation for MiniCPM models.
 *
 * Everything here is an ESTIMATE and is labelled as one wherever it surfaces.
 * The formulas are deliberately explicit so a reader can check or recalibrate
 * them instead of trusting a magic number:
 *
 *   weights   = paramsB × 1e9 × bytesPerWeight(method)
 *   trainable = paramsB × loraFraction        (LoRA only; 1% at rank 16)
 *   optimizer = trainable × 12 bytes          (AdamW fp32 m + v + master copy)
 *   gradients = trainable × 2 bytes           (bf16)
 *   activations = batch × seqLen × hidden × layers × 2 bytes × recomputeFactor
 *   kvCache   = 2 × layers × kvHeads × headDim × seqLen × batch × dtypeBytes
 *   overhead  = 2 GB (CUDA context, fragmentation, framework buffers)
 *
 * Weights use TOTAL parameters because every expert must be resident; compute
 * and activation terms use ACTIVE parameters, which is what actually runs.
 */

/** Conservative default for models whose shape the catalog does not record. */
const FALLBACK_SHAPE = { layers: 32, hiddenSize: 4096, kvHeads: 8, headDim: 128 }

/** Bytes resident per parameter, by storage strategy. */
export const BYTES_PER_WEIGHT = Object.freeze({
  bf16: 2,
  fp16: 2,
  fp8: 1,
  int8: 1,
  int4: 0.55, // nf4/int4 weights plus quantization constants
  awq: 0.55,
  gptq: 0.55,
  gguf_q4: 0.58,
  gguf_q5: 0.7,
  gguf_q8: 1.05,
})

/** Fraction of parameters that become trainable, by LoRA rank. */
export function loraFraction(rank) {
  if (!Number.isFinite(rank) || rank <= 0) return 0
  return Math.min(0.05, 0.01 * (rank / 16))
}

/** Dtype byte width for a KV cache. */
function dtypeBytes(dtype) {
  if (dtype === 'fp8') return 1
  if (dtype === 'fp32') return 4
  return 2
}

/**
 * Estimate resident bytes for the KV cache.
 * @param model catalog entry.
 * @param seqLen context length in tokens.
 * @param batch concurrent sequences.
 * @param dtype KV cache dtype.
 * @returns bytes.
 */
export function kvCacheBytes(model, seqLen, batch = 1, dtype = 'bf16') {
  const shape = {
    layers: model.layers ?? FALLBACK_SHAPE.layers,
    kvHeads: model.kvHeads ?? FALLBACK_SHAPE.kvHeads,
    headDim: model.headDim ?? FALLBACK_SHAPE.headDim,
  }
  return 2 * shape.layers * shape.kvHeads * shape.headDim * seqLen * batch * dtypeBytes(dtype)
}

/** KV bytes per token, a stable number worth quoting to users. */
export function kvBytesPerToken(model, dtype = 'bf16') {
  return kvCacheBytes(model, 1, 1, dtype)
}

/**
 * Estimate training memory for one configuration.
 *
 * @param options.model catalog entry.
 * @param options.method 'full' | 'lora' | 'qlora'.
 * @param options.seqLen sequence length.
 * @param options.batch per-device batch size.
 * @param options.loraRank LoRA rank (lora/qlora only).
 * @param options.dtype training dtype; 'bf16' typical.
 * @param options.gradientCheckpointing recompute activations to save memory.
 * @param options.gpuCount devices the run is sharded over.
 * @param options.vramGb per-device usable VRAM.
 * @returns a breakdown with an explicit total and fit decision.
 */
export function estimateTraining(options) {
  const {
    model,
    method = 'lora',
    seqLen = 2048,
    batch = 1,
    loraRank = 16,
    dtype = 'bf16',
    gradientCheckpointing = true,
    gpuCount = 1,
    vramGb = 24,
  } = options

  const shape = {
    layers: model.layers ?? FALLBACK_SHAPE.layers,
    hiddenSize: model.hiddenSize ?? FALLBACK_SHAPE.hiddenSize,
  }
  const totalParams = model.paramsB * 1e9
  const activeParams = (model.activeParamsB ?? model.paramsB) * 1e9

  const quantization = method === 'qlora' ? BYTES_PER_WEIGHT.int4
    : method === 'lora' ? BYTES_PER_WEIGHT.bf16
    : BYTES_PER_WEIGHT[dtype] ?? BYTES_PER_WEIGHT.bf16

  const weights = totalParams * (method === 'full' ? (BYTES_PER_WEIGHT[dtype] ?? 2) : quantization)

  // Full fine-tuning keeps optimizer state for every parameter; LoRA only for
  // the adapter, because the frozen base stays frozen.
  const trainable = method === 'full' ? totalParams : totalParams * loraFraction(loraRank)
  const optimizer = trainable * 12
  const gradients = trainable * 2

  // Activation memory scales with sequence, batch, width, depth. Without
  // checkpointing every layer's activations stay resident; with it, roughly the
  // boundary layers do.
  const bytesPerElement = dtype === 'fp32' ? 4 : 2
  const recomputeFactor = gradientCheckpointing ? 0.5 : 8
  const activations =
    batch * Math.min(seqLen, 32768) * shape.hiddenSize * shape.layers *
    bytesPerElement * recomputeFactor

  const overhead = 2 * 1024 ** 3

  const totalBytes = weights + optimizer + gradients + activations + overhead
  const perGpuBytes = totalBytes / Math.max(1, gpuCount)
  const availableBytes = vramGb * 1024 ** 3
  const headroomBytes = availableBytes - perGpuBytes

  return {
    estimate: true,
    method,
    seqLen,
    batch,
    gpuCount,
    vramGb,
    breakdownGb: {
      weights: roundGb(weights),
      optimizer: roundGb(optimizer),
      gradients: roundGb(gradients),
      activations: roundGb(activations),
      overhead: roundGb(overhead),
    },
    totalGb: roundGb(totalBytes),
    perGpuGb: roundGb(perGpuBytes),
    headroomGb: roundGb(headroomBytes),
    fits: headroomBytes >= 0,
    // Below ~1.5 GB of slack a run usually OOMs on fragmentation, not on the
    // model itself, so call this tight rather than fitting.
    tight: headroomBytes >= 0 && headroomBytes < 1.5 * 1024 ** 3,
    utilizations: {
      totalParams,
      activeParams,
      trainableParams: Math.round(trainable),
      trainableFraction: trainable / totalParams,
    },
  }
}

/**
 * Estimate inference memory: weights plus KV cache for the requested context.
 */
export function estimateInference(options) {
  const {
    model,
    quant = 'bf16',
    seqLen = 8192,
    batch = 1,
    gpuCount = 1,
    vramGb = 24,
    dtype = 'bf16',
  } = options

  const bytesPerWeight = BYTES_PER_WEIGHT[quant] ?? BYTES_PER_WEIGHT.bf16
  const weights = model.paramsB * 1e9 * bytesPerWeight
  const kv = kvCacheBytes(model, seqLen, batch, dtype)
  const overhead = 1.5 * 1024 ** 3
  const totalBytes = weights + kv + overhead
  const perGpuBytes = totalBytes / Math.max(1, gpuCount)
  const availableBytes = vramGb * 1024 ** 3

  return {
    estimate: true,
    quant,
    seqLen,
    batch,
    gpuCount,
    breakdownGb: {
      weights: roundGb(weights),
      kvCache: roundGb(kv),
      overhead: roundGb(overhead),
    },
    totalGb: roundGb(totalBytes),
    perGpuGb: roundGb(perGpuBytes),
    headroomGb: roundGb(availableBytes - perGpuBytes),
    fits: availableBytes - perGpuBytes >= 0,
    kvBytesPerToken: kvBytesPerToken(model, dtype),
    kvKbPerToken: Math.round(kvBytesPerToken(model, dtype) / 1024) / 1,
  }
}

/**
 * Heuristic token count for text with no tokenizer available.
 *
 * Latin text tokenizes near 4 characters per token; CJK runs closer to 1.6.
 * The estimate blends the two by detected script ratio and always over-reports
 * slightly, because under-reporting causes silent truncation.
 */
export function estimateTokens(text) {
  if (typeof text !== 'string' || text.length === 0) return 0
  const cjk = (text.match(/[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]/gu) ?? []).length
  const ratio = cjk / text.length
  const charsPerToken = 4 * (1 - ratio) + 1.6 * ratio
  return Math.ceil(text.length / charsPerToken)
}

/** Format bytes as a human string. */
export function formatGb(bytes) {
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`
}

function roundGb(bytes) {
  return Math.round((bytes / 1024 ** 3) * 10) / 10
}
