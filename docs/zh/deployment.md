# 部署与打包 MiniCPM

[English](../deployment.md) · **中文**

怎么把这些模型跑起来、需要多少显存，以及量化之后哪些东西会变。

---

## 一、各系列支持的推理框架

| 模型 | transformers | vLLM | SGLang | llama.cpp | Ollama | 其他 |
| --- | --- | --- | --- | --- | --- | --- |
| MiniCPM5-1B/2B | ≥5.6 | ≥0.21 | ≥0.5.16，`--tool-call-parser minicpm5` | ✅ 官方 GGUF | ✅ | MLX、LM Studio |
| MiniCPM4.1-8B | ✅ | ✅ 仅 dense，需 `--trust-remote-code` | OpenBMB 分支，`--chat-template chatml` | ✅ | ✅ `openbmb/minicpm4.1` | CPM.cu（速度推荐）、EAGLE3 |
| MiniCPM-SALA | ✅（稀疏需 `infllmv2_cuda_impl`） | ❌ | 分支，`--attention-backend minicpm_flashinfer --dense-as-sparse` | ❌ | ❌ | — |
| MiniCPM-V 4.7 | ✅ | ✅ | ✅ | ✅（MoE 端侧需 OpenBMB 分支的 LES） | ✅ | MLX、LiteRT-LM、ArcLight |
| MiniCPM-V 4.6 | **≥5.7.0** | ✅ | ✅ | ✅ | ✅ 官方库 | — |
| MiniCPM-o 4.5 | ✅ | ✅ | ✅ | 需 `llama.cpp-omni` | ✅ | — |

来源：[MiniCPM README](https://github.com/OpenBMB/MiniCPM/blob/main/README.md)、
[MiniCPM-V README](https://github.com/OpenBMB/MiniCPM-V/blob/main/README.md)、
[docs/deployment/](https://github.com/OpenBMB/MiniCPM/tree/main/docs/deployment)、
[SGLang cookbook](https://github.com/sgl-project/sglang/tree/main/docs_new/cookbook/autoregressive/OpenBMB)。

## 二、可以直接用的启动命令

MiniCPM5-2B 用 vLLM（标准 `LlamaForCausalLM`，不需要远程代码）：

```bash
pip install "vllm>=0.21"
vllm serve openbmb/MiniCPM5-2B --port 8000
```

同样的模型用 SGLang，并开启工具调用：

```bash
pip install "sglang[srt]>=0.5.16"
python -m sglang.launch_server --model-path openbmb/MiniCPM5-2B --port 30000 \
  --tool-call-parser minicpm5
```

MiniCPM4.1-8B 用 vLLM 时需要远程代码，并且要显式补上 BOS：

```bash
vllm serve openbmb/MiniCPM4.1-8B --trust-remote-code
```

```python
client.chat.completions.create(..., extra_body=dict(add_special_tokens=True))
```

vLLM 的 chat 接口默认 `add_special_tokens=False`，BOS 会被丢掉，整个提示词结构就不对了。
**不会报错**，只是回答质量变差 —— 这是最难排查的一类问题。

llama.cpp 跑 GGUF：

```bash
llama-server -m MiniCPM5-2B-F16.gguf -a MiniCPM5-2B --port 8080 -ngl 99 -c 8192 --jinja
```

llama.cpp 默认 `min_p=0.05`，会把那些"概率不高但恰好能打破重复循环"的 token 过滤掉，
小模型上很容易陷入复读。请求里带上 `min_p=0.0`。

Ollama：

```
FROM hf.co/openbmb/MiniCPM5-2B-GGUF
PARAMETER temperature 1.0
PARAMETER top_p 0.95
PARAMETER min_p 0.0
PARAMETER num_ctx 8192
```

## 三、采样参数

| 模型 | temperature | top_p | 其他 |
| --- | --- | --- | --- |
| MiniCPM5-1B/2B | 1.0 | 0.95 | `min_p=0.0`；若复读可加 `repetition_penalty=1.05` |
| MiniCPM4.1（推理模式） | 0.6 | 0.95 | `max_tokens` 可到 32768 |

来源：[MiniCPM README](https://github.com/OpenBMB/MiniCPM/blob/main/README.md)。

## 四、量化

| 格式 | 工具 | 官方提供情况 | 注意 |
| --- | --- | --- | --- |
| GGUF | llama.cpp `quantize` | 大多数版本都有官方 GGUF | 视觉投影层要单独处理；不是每一档量化都保得住 OCR |
| GPTQ / AWQ（W4A16） | `llm-compressor`、`autoawq` | V4.7 与 MiniCPM4.1 有官方版 | 自己做需要校准集 |
| int4（bitsandbytes） | `bitsandbytes` | V4.7 1B 有官方版 | 推理慢，适合演示不适合服务 |
| FP8 | vLLM 内置 | V4.7 35B-A3B 有官方版 | 需要 Hopper 及以后 |
| 三值（BitCPM4） | OpenBMB 自有内核 | BitCPM4-0.5B/1B | 研究性质，要专用运行时 |

**量化多模态模型时，视觉塔和投影层要保持较高精度。** 否则最容易得到一个"能跑、但读不出字"
的模型。

## 五、显存怎么算

权重：`参数量 × 1e9 × 每参数字节数`。8B 在 bf16 下约 16 GB，int4 下约 4.5 GB。

KV 缓存：`2 × 层数 × KV 头数 × 头维度 × token 数 × 并发数 × 每元素字节数`。

以 MiniCPM5-2B 为例（42 层、2 个 KV 头、头维度 128、bf16）：**每 token 43,008 字节，
约 42 KB**。32k 上下文约 1.3 GB；128k 约 5.4 GB。

三条值得记住的结论：

- `--max-model-len` 从 8k 提到 128k，KV 占用涨约 16 倍，但权重不变。
- 并发只放大 KV，不放大权重。按单条序列配好的服务，二十条并发就会 OOM。
- MoE 模型的权重按**总参数**算（所有专家都要驻留），激活按**激活参数**算。

## 六、打包与分发

### 发 adapter 还是合并权重

默认发 adapter：体积小、基座来源清晰、用 `PeftModel.from_pretrained` 就能加载，使用方也
可以随时关掉。只有当对方完全无法使用 PEFT 时，才发合并权重。

### 上传

```bash
huggingface-cli login
huggingface-cli upload <用户名>/<仓库名> ./adapter_final . --repo-type model

modelscope login
modelscope upload <用户名>/<仓库名> . --repo-type model
```

### 一次可信的发布应该包含什么

- `LICENSE` —— Apache-2.0 完整原文，逐字不改。
- `NOTICE` —— 写明基座权重来自 OpenBMB、THUNLP 与 ModelBest。
- `README.md` —— frontmatter 里有 `license`、`base_model`、`library_name`、`pipeline_tag`。
- 一张**实测**的评估表：基座 / 微调 / 差值。占位符不算记录，只是声明。
- 局限说明，写清观察到的失效场景。
- 训练数据来源；如果数据不能公开，就明确写出来。

### Docker 示例

```dockerfile
FROM vllm/vllm-openai:v0.21.0
ENV HF_HOME=/models
COPY ./merged /models/minicpm
EXPOSE 8000
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", \
            "--model", "/models/minicpm", "--port", "8000"]
```

多阶段构建帮助有限 —— 镜像里主要就是 CUDA 运行时。建议把模型放在镜像外挂载，否则每次
更新镜像都要推上百 GB。

## 七、按任务看的显存门槛

| 任务 | 实际下限 |
| --- | --- |
| QLoRA 0.5B | 8 GB |
| LoRA 2B | 12–16 GB |
| QLoRA 8B | 16–24 GB |
| LoRA 8B | 24 GB 以上 |
| 全参 8B | 4 × 24 GB（官方下限） |
| 推理 1–2B bf16 | 8 GB |
| 推理 8B bf16 | 24 GB |
| 推理 8B int4 | 8–12 GB |
| 多模态 4.6/4.7 微调 | 官方示例用 8 卡；序列短时可以用更少，但未经验证 |

这些是经验值，不是承诺。`minicpm_doctor` 负责测你的机器，`minicpm_plan` 负责估算能否装下，
并明确标注是估算。
