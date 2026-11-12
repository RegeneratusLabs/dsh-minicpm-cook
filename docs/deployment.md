# Serving and packaging MiniCPM

English · [中文](zh/deployment.md)

How to run these models, how much memory that takes, and what changes when you quantize.

---

## 1. Backend support by family

| Model | transformers | vLLM | SGLang | llama.cpp | Ollama | Other |
| --- | --- | --- | --- | --- | --- | --- |
| MiniCPM5-1B/2B | ≥5.6 | ≥0.21 | ≥0.5.16, `--tool-call-parser minicpm5` | ✅ official GGUF | ✅ | MLX, LM Studio |
| MiniCPM4.1-8B | ✅ | ✅ dense only, `--trust-remote-code` | OpenBMB fork, `--chat-template chatml` | ✅ | ✅ `openbmb/minicpm4.1` | CPM.cu (recommended for speed), EAGLE3 |
| MiniCPM-SALA | ✅ (sparse needs `infllmv2_cuda_impl`) | ❌ | fork, `--attention-backend minicpm_flashinfer --dense-as-sparse` | ❌ | ❌ | — |
| MiniCPM-V 4.7 | ✅ | ✅ | ✅ | ✅ (LES in the OpenBMB fork for MoE on-device) | ✅ | MLX, LiteRT-LM, ArcLight |
| MiniCPM-V 4.6 | **≥5.7.0** | ✅ | ✅ | ✅ | ✅ official library tag | — |
| MiniCPM-o 4.5 | ✅ | ✅ | ✅ | via `llama.cpp-omni` | ✅ | — |

Sources: [MiniCPM README](https://github.com/OpenBMB/MiniCPM/blob/main/README.md),
[MiniCPM-V README](https://github.com/OpenBMB/MiniCPM-V/blob/main/README.md),
[docs/deployment/](https://github.com/OpenBMB/MiniCPM/tree/main/docs/deployment),
[SGLang cookbook](https://github.com/sgl-project/sglang/tree/main/docs_new/cookbook/autoregressive/OpenBMB).

## 2. Launch commands that work

MiniCPM5-2B on vLLM (standard `LlamaForCausalLM`, no remote code):

```bash
pip install "vllm>=0.21"
vllm serve openbmb/MiniCPM5-2B --port 8000
```

Same on SGLang, with tool calling:

```bash
pip install "sglang[srt]>=0.5.16"
python -m sglang.launch_server --model-path openbmb/MiniCPM5-2B --port 30000 \
  --tool-call-parser minicpm5
```

MiniCPM4.1-8B on vLLM needs remote code and an explicit BOS:

```bash
vllm serve openbmb/MiniCPM4.1-8B --trust-remote-code
```

```python
client.chat.completions.create(..., extra_body=dict(add_special_tokens=True))
```

vLLM's chat API defaults `add_special_tokens=False`, so BOS is dropped and the
prompt is mis-framed. Nothing errors; answers are just worse.

GGUF on llama.cpp:

```bash
llama-server -m MiniCPM5-2B-F16.gguf -a MiniCPM5-2B --port 8080 -ngl 99 -c 8192 --jinja
```

llama.cpp defaults `min_p=0.05`, which removes the low-probability tokens that
break repetition loops in small models. Send `min_p=0.0`.

Ollama:

```
FROM hf.co/openbmb/MiniCPM5-2B-GGUF
PARAMETER temperature 1.0
PARAMETER top_p 0.95
PARAMETER min_p 0.0
PARAMETER num_ctx 8192
```

## 3. Sampling parameters

| Model | Temperature | top_p | Other |
| --- | --- | --- | --- |
| MiniCPM5-1B/2B | 1.0 | 0.95 | `min_p=0.0`; add `repetition_penalty=1.05` if it loops |
| MiniCPM4.1 (reasoning) | 0.6 | 0.95 | `max_tokens` up to 32768 |

Source: [MiniCPM README](https://github.com/OpenBMB/MiniCPM/blob/main/README.md).

## 4. Quantization

| Format | Tooling | Where it exists | Caveat |
| --- | --- | --- | --- |
| GGUF | llama.cpp `quantize` | Official GGUFs for most releases | Quantizing the vision projector separately; not every quant level keeps OCR |
| GPTQ / AWQ (W4A16) | `llm-compressor`, `autoawq` | Official for V4.7 and MiniCPM4.1 | Needs a calibration set if you do it yourself |
| int4 (bitsandbytes) | `bitsandbytes` | Official V4.7 1B variant | Slow at inference; fine for a demo, not a server |
| FP8 | built into vLLM | Official V4.7 35B-A3B variant | Hopper or newer |
| Ternary (BitCPM4) | OpenBMB kernels | BitCPM4-0.5B/1B | Research artifact; needs its own runtime |

**Keep the vision tower and projector at higher precision** when quantizing a VLM.
It is the fastest route to a model that runs and cannot read.

## 5. Memory arithmetic

Weights: `paramsB × 1e9 × bytesPerWeight`. At bf16, 8B ≈ 16 GB; at int4, ≈ 4.5 GB.

KV cache: `2 × layers × kv_heads × head_dim × tokens × batch × dtypeBytes`.

For MiniCPM5-2B (42 layers, 2 KV heads, 128 head_dim, bf16): **43,008 bytes per
token ≈ 42 KB**. At 32k context that is ~1.3 GB; at 128k, ~5.4 GB.

Consequences worth internalising:

- Raising `--max-model-len` 8k → 128k costs ~16× the KV memory, not 16× the weights.
- Concurrency multiplies KV only. A server sized for one sequence OOMs at twenty.
- For an MoE model, weights are the *total* parameter count (all experts resident) while activations follow the *active* count.

## 6. Packaging and distribution

### Adapter or merged

Publish the adapter by default: smaller, licence-provenance obvious, loadable with
`PeftModel.from_pretrained`, and disableable by a consumer. Publish merged weights
only for consumers who cannot run PEFT.

### Upload

```bash
huggingface-cli login
huggingface-cli upload <user>/<repo> ./adapter_final . --repo-type model

modelscope login
modelscope upload <user>/<repo> . --repo-type model
```

### What a credible release contains

- `LICENSE` — the full Apache-2.0 text, verbatim.
- `NOTICE` — naming OpenBMB, THUNLP and ModelBest as the origin of the base weights.
- `README.md` — card with `license`, `base_model`, `library_name`, `pipeline_tag` frontmatter.
- A measured evaluation table, base vs fine-tune vs delta. A placeholder is a claim, not a record.
- A limitations section naming observed failure modes.
- Training data provenance, or an explicit statement that it cannot be released.

### Docker pattern

```dockerfile
FROM vllm/vllm-openai:v0.21.0
ENV HF_HOME=/models
COPY ./merged /models/minicpm
EXPOSE 8000
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", \
            "--model", "/models/minicpm", "--port", "8000"]
```

Multi-stage builds help little — the CUDA runtime dominates the image. Keep the
model out of the image and mount it, so an image update does not require a
150 GB push.

## 7. Hardware reality by task

| Task | Practical minimum |
| --- | --- |
| QLoRA 0.5B | 8 GB |
| LoRA 2B | 12–16 GB |
| QLoRA 8B | 16–24 GB |
| LoRA 8B | 24 GB+ |
| Full SFT 8B | 4 × 24 GB (official floor) |
| Inference 1–2B bf16 | 8 GB |
| Inference 8B bf16 | 24 GB |
| Inference 8B int4 | 8–12 GB |
| VLM 4.6/4.7 fine-tune | 8 GPUs per the official walkthrough; fewer works at small sequence length, unverified |

These are conventions, not guarantees. `minicpm_doctor` measures the host;
`minicpm_plan` estimates the fit and labels it an estimate.
