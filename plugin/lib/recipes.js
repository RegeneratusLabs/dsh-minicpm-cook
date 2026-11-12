/**
 * Recipe generation for TRL + PEFT, LLaMA-Factory, and ms-swift.
 *
 * Generated files are text. This module never writes to disk, never spawns a
 * process, and never downloads anything: the agent shows or saves the result.
 *
 * Every emitted config carries the model-specific values that the upstream
 * documentation insists on and that a naive template gets wrong:
 *   - the framework's exact template / model_type string for that model
 *   - trust_remote_code when the architecture ships custom modeling code
 *   - freeze_vision_tower: false for VLMs (freezing it silently caps quality)
 *   - the DOWNSAMPLE_MODE that controls V4.6/V4.7 visual token budget
 *   - assistant-only loss, which is the difference between converging and not
 */

/** Toolchains this module can emit. */
export const TOOLCHAINS = ['trl', 'llamafactory', 'ms-swift']

/** Training methods. */
export const METHODS = ['full', 'lora', 'qlora']

/**
 * Whether a toolchain officially covers a model's modality.
 *
 * TRL has no documented VLM path for MiniCPM-V/o, so steering a vision user
 * there would produce a config that trains on image tokens as ordinary text.
 */
export function toolchainSupports(model, toolchain) {
  const isVision = model.modality !== 'text'
  if (!isVision) return { ok: true }
  if (toolchain === 'trl') {
    return {
      ok: false,
      reason:
        `${model.displayName} is a ${model.modality} model and OpenBMB does not document a TRL VLM path. ` +
        'Use llamafactory or ms-swift, which handle image tokens, dynamic resolution ' +
        `and the ${model.vision?.downsampleModes?.join('/') ?? 'visual'} token budget.`,
    }
  }
  return { ok: true }
}

/**
 * Build a complete recipe.
 *
 * @param options.model catalog entry.
 * @param options.toolchain one of TOOLCHAINS.
 * @param options.method one of METHODS.
 * @param options.dataPath training file the generated config should point at.
 * @param options.outputDir where checkpoints land.
 * @param options.maxSeqLen sequence length.
 * @param options.epochs epoch count.
 * @param options.lr learning rate; omitted uses a method-appropriate default.
 * @param options.batch per-device batch size.
 * @param options.gradAccum gradient accumulation steps.
 * @param options.loraRank LoRA rank.
 * @param options.downsample VLM visual token budget ('4x' | '16x').
 * @returns { files, commands, warnings, defaults } or { error, alternatives }.
 */
export function buildRecipe(options) {
  const {
    model,
    toolchain,
    method = 'lora',
    dataPath = './data/train.jsonl',
    outputDir = './runs/minicpm',
    maxSeqLen,
    epochs = 2,
    lr,
    batch = 1,
    gradAccum = 8,
    loraRank = 16,
    downsample,
  } = options

  if (!TOOLCHAINS.includes(toolchain)) {
    return { error: `unknown toolchain "${toolchain}"`, alternatives: TOOLCHAINS }
  }
  if (!METHODS.includes(method)) {
    return { error: `unknown method "${method}"`, alternatives: METHODS }
  }

  const support = toolchainSupports(model, toolchain)
  if (!support.ok) {
    return {
      error: support.reason,
      alternatives: TOOLCHAINS.filter((candidate) => toolchainSupports(model, candidate).ok),
    }
  }

  const warnings = []
  const seqLen = maxSeqLen ?? defaultSeqLen(model)
  if (seqLen > model.contextTokens) {
    warnings.push(
      `maxSeqLen ${seqLen} exceeds ${model.displayName}'s trained context of ${model.contextTokens}. ` +
      'Exceeding it without RoPE scaling degrades long-context quality.',
    )
  }
  if (method === 'full' && model.paramsB > 9) {
    warnings.push(
      `${model.displayName} is ${model.paramsB}B; full fine-tuning needs several 80 GB devices. ` +
      'LoRA or QLoRA is almost always the right call at this size.',
    )
  }
  if (model.confidence === 'unverified') {
    warnings.push(
      `Catalog entry for ${model.displayName} is marked unverified` +
      `${model.confidenceNote ? `: ${model.confidenceNote}` : ''}. Confirm template and ids against the model card.`,
    )
  }

  const resolved = {
    method,
    seqLen,
    epochs,
    lr: lr ?? defaultLearningRate(model, method),
    batch,
    gradAccum,
    loraRank,
    downsample: downsample ?? model.vision?.defaultDownsample,
  }

  if (toolchain === 'trl') return trlRecipe({ model, resolved, dataPath, outputDir, warnings })
  if (toolchain === 'llamafactory') return llamafactoryRecipe({ model, resolved, dataPath, outputDir, warnings })
  return msSwiftRecipe({ model, resolved, dataPath, outputDir, warnings })
}

/** Sequence length a model's own docs use for fine-tuning. */
function defaultSeqLen(model) {
  if (model.modality !== 'text') return 4096
  return Math.min(4096, model.contextTokens)
}

/**
 * Learning rate defaults.
 *
 * LoRA tolerates a much larger rate than full fine-tuning, and the VLM recipe
 * uses a smaller one again because the vision tower is unfrozen.
 */
function defaultLearningRate(model, method) {
  if (model.modality !== 'text') return 5e-6
  if (method === 'full') return 1e-5
  return 2e-4
}

/** LoRA target modules for a decoder-only transformer. */
function targetModules(model) {
  if (model.modality !== 'text') {
    return ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
  }
  return ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
}

/** TRL + PEFT bare-metal recipe (text models). */
function trlRecipe({ model, resolved, dataPath, outputDir, warnings }) {
  const base = model.hfId
  const trainableFraction = resolved.loraRank === 16 ? '~1%' : 'model-size dependent'
  const usesFullFinetune = resolved.method === 'full'

  if (usesFullFinetune) {
    warnings.push(
      'TRL recipe emitted for full fine-tuning: no adapter is created, so trainable params are the whole model. ' +
      'Expect optimizer state for every parameter.',
    )
  }
  if (!model.trustRemoteCode) {
    // Nothing to warn about; recorded so the template stays explicit.
  } else {
    warnings.push(
      `The catalog marks ${model.displayName} as needing trust_remote_code for transformers. ` +
      'The TRL recipe sets it; keep it set or loading fails.',
    )
  }

  const script = `# Generated by dsh-minicpm-cook for ${model.displayName}.
# Recipe adapted from OpenBMB's documented TRL path
# (https://github.com/OpenBMB/MiniCPM/blob/main/docs/finetune/trl.md, Apache-2.0).
#
# Assistant-only loss requires the {% generation %} blocks below. The patched
# template is for TRAINING ONLY: re-load the original tokenizer for inference, or
# you lose think-mode and tool-call rendering.
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

BASE = os.environ.get("BASE_MODEL", "${base}")
DATA = os.environ.get("DATA", "${dataPath}")
OUT = os.environ.get("OUTPUT_DIR", "${outputDir}")

TRAIN_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{{- '<|im_start|>system\\\\n' + message['content'] + '<|im_end|>\\\\n' }}"
    "{%- elif message['role'] == 'user' %}"
    "{{- '<|im_start|>user\\\\n' + message['content'] + '<|im_end|>\\\\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\\\\n' }}"
    "{%- generation %}"
    "{{- message['content'] + '<|im_end|>' }}"
    "{%- endgeneration %}"
    "{{- '\\\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\\\n' }}"
    "{%- endif %}"
)

set_seed(42)

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=${model.trustRemoteCode ? 'True' : 'False'}, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.chat_template = TRAIN_CHAT_TEMPLATE

rows = [json.loads(line) for line in open(DATA, encoding="utf-8") if line.strip()]
ds = Dataset.from_list([{"messages": row["messages"]} for row in rows])

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    trust_remote_code=${model.trustRemoteCode ? 'True' : 'False'},
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

${usesFullFinetune ? '# Full fine-tuning: no LoRA wrapper.\nmodel = model' : `lora = LoraConfig(
    r=${resolved.loraRank},
    lora_alpha=${resolved.loraRank * 2},
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=${JSON.stringify(targetModules(model))},
)
model = get_peft_model(model, lora)`}
model.print_trainable_parameters()

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=OUT,
        num_train_epochs=${resolved.epochs},
        per_device_train_batch_size=${resolved.batch},
        gradient_accumulation_steps=${resolved.gradAccum},
        learning_rate=${resolved.lr},
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        max_length=${resolved.seqLen},
        packing=False,
        assistant_only_loss=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to=["tensorboard"],
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=42,
    ),
    train_dataset=ds,
    processing_class=tok,
)
trainer.train()
trainer.model.save_pretrained(os.path.join(OUT, "adapter_final") if ${usesFullFinetune ? 'False' : 'True'} else OUT)
print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad), "(${trainableFraction})")
`

  const files = [{ path: 'train_lora.py', content: script }]
  const commands = [
    `BASE_MODEL=${base} DATA=${dataPath} OUTPUT_DIR=${outputDir} python train_lora.py`,
  ]

  return {
    files,
    commands,
    warnings,
    defaults: resolved,
    notes: [
      'Requires trl>=0.21 for SFTConfig.max_length; older TRL raises AttributeError.',
      'assistant_only_loss=True masks everything outside {% generation %} blocks, so the template above is load-bearing.',
      'The adapter is compatible with the base chat template at inference: reload the ORIGINAL tokenizer.',
    ],
  }
}

/** LLaMA-Factory recipe: dataset_info entry plus a train YAML. */
function llamafactoryRecipe({ model, resolved, dataPath, outputDir, warnings }) {
  const template = model.frameworks?.llamafactory?.template
  if (template === undefined) {
    warnings.push(`No LLaMA-Factory template is recorded for ${model.displayName}; confirm it before training.`)
  }

  const isVision = model.modality !== 'text'
  const datasetName = 'minicpm_sft'

  const datasetInfo = {
    [datasetName]: {
      file_name: dataPath,
      formatting: 'sharegpt',
      columns: isVision
        ? { messages: 'messages', images: 'images' }
        : { messages: 'messages' },
      tags: {
        role_tag: 'role',
        content_tag: 'content',
        user_tag: 'user',
        assistant_tag: 'assistant',
        system_tag: 'system',
      },
    },
  }

  const lines = [
    '### model',
    `model_name_or_path: ${model.hfId}`,
    `trust_remote_code: ${model.trustRemoteCode}`,
    '',
    '### method',
    'stage: sft',
    'do_train: true',
    `finetuning_type: ${resolved.method === 'full' ? 'full' : 'lora'}`,
    `lora_rank: ${resolved.loraRank}`,
    `lora_target: ${targetModules(model).join(',')}`,
  ]

  if (isVision) {
    lines.push('freeze_vision_tower: false')
    lines.push('image_max_pixels: 12845056')
  }
  if (resolved.method === 'qlora') {
    lines.push('quantization_bit: 4')
    lines.push('quantization_method: bitsandbytes')
  }

  lines.push(
    '',
    '### dataset',
    `dataset: ${datasetName}`,
    `dataset_dir: ${dirnameOf(dataPath)}`,
    `template: ${template ?? 'REPLACE_ME'}`,
    `cutoff_len: ${resolved.seqLen}`,
    'preprocessing_num_workers: 8',
    'overwrite_cache: true',
    '',
    '### output',
    `output_dir: ${outputDir}`,
    'logging_steps: 10',
    'save_steps: 200',
    'save_total_limit: 2',
    'plot_loss: true',
    '',
    '### train',
    `per_device_train_batch_size: ${resolved.batch}`,
    `gradient_accumulation_steps: ${resolved.gradAccum}`,
    `learning_rate: ${resolved.lr}`,
    `num_train_epochs: ${resolved.epochs}`,
    'lr_scheduler_type: cosine',
    'warmup_ratio: 0.05',
    'bf16: true',
    'max_grad_norm: 1.0',
  )

  if (isVision && resolved.downsample !== undefined) {
    warnings.push(
      `Set DOWNSAMPLE_MODE=${resolved.downsample} in the environment before invoking llamafactory-cli; ` +
      'it selects the visual token budget and is not a YAML key.',
    )
  }
  warnings.push(
    'merge the dataset_info.json entry into LLaMA-Factory/data/dataset_info.json; it is not read from an arbitrary path.',
  )

  return {
    files: [
      { path: `${datasetName}.dataset_info.json`, content: `${JSON.stringify(datasetInfo, null, 2)}\n` },
      { path: `${model.id}.train.yaml`, content: `${lines.join('\n')}\n` },
    ],
    commands: [
      `FORCE_TORCHRUN=1 llamafactory-cli train ${model.id}.train.yaml`,
    ],
    warnings,
    defaults: resolved,
  }
}

/** ms-swift recipe: one shell script with the documented flags. */
function msSwiftRecipe({ model, resolved, dataPath, outputDir, warnings }) {
  const modelType = model.frameworks?.ms_swift?.modelType ?? 'REPLACE_ME'
  const template = model.frameworks?.ms_swift?.template ?? 'REPLACE_ME'
  const isVision = model.modality !== 'text'

  const flags = [
    `--model "\${MODEL_PATH}"`,
    `--model_type ${modelType}`,
    `--template ${template}`,
    `--dataset "\${TRAIN_DATA}"`,
    `--deepspeed zero2`,
    `--tuner_type ${resolved.method === 'full' ? 'full' : resolved.method === 'qlora' ? 'lora' : 'lora'}`,
    '--torch_dtype bfloat16',
    '--packing false',
    `--max_length ${resolved.seqLen}`,
    `--num_train_epochs ${resolved.epochs}`,
    `--per_device_train_batch_size ${resolved.batch}`,
    `--gradient_accumulation_steps ${resolved.gradAccum}`,
    `--learning_rate ${resolved.lr}`,
    '--warmup_ratio 0.05',
    '--logging_steps 1',
    '--save_steps 200',
    '--save_total_limit 2',
    '--load_from_cache_file false',
    '--dataset_num_proc 8',
    '--dataloader_num_workers 8',
    '--attn_impl flash_attn',
    `--output_dir "\${OUTPUT_DIR}"`,
  ]

  if (isVision) {
    flags.push('--freeze_vit false')
    if (resolved.downsample !== undefined) {
      flags.push('--add_non_thinking_prefix false')
    }
  }
  if (resolved.method === 'qlora') {
    flags.push('--quant_bits 4')
    flags.push('--quant_method bnb')
  }

  const script = `#!/usr/bin/env bash
# Generated by dsh-minicpm-cook for ${model.displayName}.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-0}"
${isVision && resolved.downsample !== undefined ? `export DOWNSAMPLE_MODE="\${DOWNSAMPLE_MODE:-${resolved.downsample}}"\n` : ''}MODEL_PATH="\${MODEL_PATH:-${model.hfId}}"
TRAIN_DATA="\${TRAIN_DATA:-${dataPath}}"
OUTPUT_DIR="\${OUTPUT_DIR:-${outputDir}}"

swift sft \\
${flags.map((flag) => `  ${flag}`).join(' \\\n')}

echo "adapter/checkpoint written to \${OUTPUT_DIR}"
`

  warnings.push(
    'The ms-swift dataset must already be in ms-swift format (messages + images); ' +
    'convert with `swift export --to swift` or validate it with minicpm_check_dataset first.',
  )
  if (model.modality === 'text') {
    warnings.push(
      'Do not enable --packing: OpenBMB documents packed training as broken on the current transformers release.',
    )
  }

  return {
    files: [{ path: 'run_swift.sh', content: script }],
    commands: ['bash run_swift.sh'],
    warnings,
    defaults: resolved,
  }
}

function dirnameOf(path) {
  const index = path.lastIndexOf('/')
  return index === -1 ? '.' : path.slice(0, index)
}
