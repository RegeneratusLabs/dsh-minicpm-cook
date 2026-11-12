# 微调 MiniCPM

[English](../fine-tuning.md) · **中文**

从一份数据集走到一个可用 adapter 所需要的全部内容，包括那些决定训练能否收敛的关键设置。
文中引用来源；凡属估算而非实测的数字，都会说明。

---

## 一、框架怎么选

| 框架 | 文本模型 | 多模态 | DPO/KTO | 说明 |
| --- | --- | --- | --- | --- |
| TRL + PEFT | ✅ 有文档 | ❌ 无官方路径 | ✅ 原生支持 | 抽象最少；用 `SFTConfig(assistant_only_loss=True)` 做助手掩码 |
| LLaMA-Factory | ✅ `cpm3` / `minicpm5` 模板 | ✅ `minicpm_v_4_6`、`minicpmv4_6`、`minicpmv4_7` | ✅ | YAML 驱动；V4.6 那份是官方写得最完整的微调文档 |
| ms-swift | ✅ | ✅ `--model_type minicpmv4_6` | ✅（`swift rlhf --rlhf_type dpo`） | 命令行驱动；原生处理视觉 token 预算 |
| Unsloth | ✅（MiniCPM5） | — | — | 见 `docs/finetune/unsloth.md` |
| XTuner | ✅（MiniCPM5） | — | — | 见 `docs/finetune/xtuner.md` |
| OpenBMB 官方代码 | ✅ | ✅ | — | `OpenBMB/MiniCPM/finetune/`，DeepSpeed 启动，`--use_lora` / `--qlora` |

一点提醒：**多模态模型不要走 TRL**。OpenBMB 没有为 MiniCPM-V/o 提供 TRL 路径，照搬文本
配方会把图像 token 当普通文本训练。

来源：[docs/finetune/](https://github.com/OpenBMB/MiniCPM/tree/main/docs/finetune)、
[skills/](https://github.com/OpenBMB/MiniCPM/tree/main/skills)、
[MiniCPM4.1 微调指南](https://openbmb.cn/docs/en/guides/minicpm-text/minicpm4.1-and-minicpm4/finetune)、
[MiniCPM-V 4.6 微调指南](https://openbmb.cn/docs/en/guides/minicpm-v/minicpm-v-4.6/finetune)。

## 二、数据格式

文本 SFT —— OpenBMB 官方微调代码吃的就是这个结构：

```json
[{"messages": [
  {"role": "system", "content": "可选"},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}]
```

LLaMA-Factory 的 alpaca 格式：

```json
{"instruction": "...", "input": "...", "output": "..."}
```

LLaMA-Factory / ms-swift 的多模态格式：

```json
{"messages": [
  {"role": "user", "content": "<image>\n仔细观察这张图……"},
  {"role": "assistant", "content": "<think>\n\n</think>\n\n…… <point>236 469</point> ……"}
],
 "images": ["/path/to/img.jpg"]}
```

V4.6 的官方示例额外强调了两点，随手转换很容易漏掉：

- **坐标要归一化到 0–1000。** 直接喂像素值，模型学到的是错误的尺度。
- **非推理任务建议给助手回复加前缀** `<think>\n\n</think>\n\n`；推理任务用 `<think>\n`。

DPO（LLaMA-Factory）：

```json
{"instruction": "...", "input": "...", "chosen": "...", "rejected": "...", "history": [["q", "a"]]}
```

KTO 是"打分"而不是"对比"：每条记录用 `{"messages": [...], "label": true}` 标记好坏。

## 三、决定训练成败的几个设置

### 只在助手回复上算损失

TRL 的 `assistant_only_loss=True` 会屏蔽所有不在 `{% generation %}` 块内的 token。不开这个
开关，损失会把提问也算进去，小 adapter 上的信号被稀释，损失甚至可能不降反升。

这个机制依赖一份"仅训练用"的对话模板。它的 token 序列与模型自带模板完全一致，所以训完
的 adapter 在推理时依然兼容 —— 但**这份改过的模板绝对不能随 tokenizer 一起保存**，
否则会把思考模式和工具调用渲染一起弄丢。

来源：[docs/finetune/trl.md](https://github.com/OpenBMB/MiniCPM/blob/main/docs/finetune/trl.md)、
[skills/minicpm5-finetune-trl](https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-finetune-trl/SKILL.md)。

### 不要开启 packing

OpenBMB 明确写了：在当前 `transformers` 版本下，packed 训练在这些模型上是坏的。用
`packing=False` / `--packing false`。

### 视觉塔要解冻

`freeze_vision_tower: false`（LLaMA-Factory）或 `--freeze_vit false`（ms-swift）。官方每一份
多模态示例都是解冻的。冻住视觉塔，是"模型会泛泛谈图片、但看不清**这张**图"的最常见原因。

### 视觉 token 预算

V4.6/V4.7 支持 `4x` 和 `16x` 两档压缩比，通过环境变量 `DOWNSAMPLE_MODE` 选择（它不是
YAML 里的键）。差别是实测的：官方计数基准上，微调后 4x 从 51.8 提升到 83.1，16x 从 46.5
提升到 78.4（Acc@0）。

## 四、官方给定的超参数

| 设置 | MiniCPM-V 4.6（官方示例） | MiniCPM5 TRL（官方配方） |
| --- | --- | --- |
| 训练方式 | 全参 | LoRA |
| 学习率 | 5e-6 | 2e-4 |
| batch × 累积 | 1 × 16 | 4 × 4 |
| 轮数 | 4 | 2 |
| 序列长度 | 4096 | 2048 |
| LoRA rank / alpha | 不适用（全参） | 16 / 32 |
| 目标模块 | 不适用 | `q,k,v,o,gate,up,down` |
| 调度器 | cosine，预热 5% | cosine，预热 3% |
| 精度 | bf16 | bf16 |

MiniCPM4.1 的官方脚本用 `learning_rate 5e-5`、batch 16、`model_max_length 1024`、
`warmup_steps 100`、`max_steps 1000`，配合 ZeRO-2（含 CPU offload）。

来源：[V4.6 微调](https://openbmb.cn/docs/en/guides/minicpm-v/minicpm-v-4.6/finetune)、
[TRL 配方](https://github.com/OpenBMB/MiniCPM/blob/main/docs/finetune/trl.md)、
[MiniCPM4.1 微调](https://openbmb.cn/docs/en/guides/minicpm-text/minicpm4.1-and-minicpm4/finetune)。

## 五、显存

OpenBMB 公布的硬件下限：

| 模型 | LoRA | 全参 |
| --- | --- | --- |
| MiniCPM4-0.5B | ≥8 GB（20 系及以后） | >12 GB |
| MiniCPM4.1-8B | ≥2 × 24 GB | ≥4 × 24 GB |

多模态的官方示例统一用 8 卡、单卡 batch 1、ZeRO-2。

本项目自己按公开公式计算（`plugin/lib/estimate.js`），因为实际占用取决于序列长度、batch、
优化器、梯度检查点以及是否 offload。计算结果一律标注为"估算"，同时并列给出官方下限作为
安全边界。

## 六、合并 adapter

```python
merged = model.merge_and_unload()
merged.save_pretrained(out)
tok.save_pretrained(out)     # 用原本的 tokenizer
```

多模态模型只合并语言部分。视觉塔不是 LoRA 的目标模块，用通用的合并流程有可能把投影层弄坏。

## 七、常见故障对照表

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `SFTConfig has no attribute 'max_length'` | TRL 低于 0.21 | 升级 TRL |
| 损失不下降 | 没开 `assistant_only_loss`，或模板里没有 `{% generation %}` | 用打好补丁的模板 |
| 小 adapter 上损失反而上升 | 损失把提示词也算进去了 | 同上 |
| 加载模型报 `ImportError` | 少了 `trust_remote_code=True` | 补上；MiniCPM4.x、V、o 都需要 |
| 训练异常慢或 OOM | 开了 packing | `packing=False` |
| 多模态只学到说话风格，不看图 | 视觉塔被冻住 | `freeze_vision_tower: false` |
| OCR 效果明显掉 | `DOWNSAMPLE_MODE=16x` | 改用 `4x` |
| 微调后坐标全错 | 用了原始像素而不是 0–1000 归一化 | 重新缩放标注 |
| `transformers` 版本被改回去 | 装训练依赖时装了别的版本 | 最后再装回指定版本 |
| DeepSpeed ZeRO-3 卡住 | 小模型用 ZeRO-3 | 改用 ZeRO-2 |

## 八、评估

模型卡里官方使用的基准：文本看 C-Eval、CMMLU、MMLU、GSM8K、HumanEval；多模态看 MMMU、
OCRBench、MathVista。

但对一次微调来说，公开基准只能当"冒烟测试"，不能当证据。实际做法：

1. **训练前**先留出 50–200 条来自你自己业务分布的样本。
2. 用完全相同的解码参数，分别跑基座和微调版本。
3. 人工读一遍差异最大的 10 条 —— 分数看不见的退化，用户一眼就能看见。
4. 再跑 20 条通用指令，确认没有灾难性遗忘。
