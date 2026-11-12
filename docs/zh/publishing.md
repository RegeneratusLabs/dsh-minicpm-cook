# 发布微调模型

[English](../publishing.md) · **中文**

该交付什么、许可证要求什么，以及那些"发布之后再改很贵"的坑。

---

这是 MiniCPM 全流程里上游覆盖最少、后果却最重的一环。

## 一、许可证

| 仓库 | 原文表述 | 出处 |
| --- | --- | --- |
| `OpenBMB/MiniCPM` | "This repository and MiniCPM models are released under the Apache-2.0 License." | [README § LICENSE](https://github.com/OpenBMB/MiniCPM/blob/main/README.md) |
| `OpenBMB/MiniCPM-V` | "Code and model weights are released under Apache-2.0." | [README](https://github.com/OpenBMB/MiniCPM-V/blob/main/README.md) |
| `OpenSQZ/MiniCPM-V-CookBook` | Apache-2.0 | [LICENSE](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/LICENSE) |

Apache-2.0 具体意味着：

- **允许商业使用。**
- **允许再分发权重及其衍生作品。**
- **必须署名**，并且要保留 `NOTICE` 文件。
- **修改过的地方要标注。** 微调本身就是修改，要写清楚。
- **不授予商标权和背书权。**

目录里每个模型都记录了许可证，以及当初读到它的链接，外加一个"是否已核实"的标记。凡是
没有对着该模型自己的 `LICENSE` 文件核对过的，`minicpm_publish` 一律按**不可再分发**处理
—— 因为判断错的代价由拿到权重的人承担。

## 二、adapter、合并权重、还是 GGUF

| 形式 | 体积（8B，r=16） | 优点 | 缺点 |
| --- | --- | --- | --- |
| LoRA adapter | 约 100–200 MB | 小；基座来源清楚；可以随时关掉；一个基座可挂多个 adapter | 使用方需要 `peft`；不是所有推理框架都支持 |
| 合并权重 | 约 16 GB（bf16） | 任何能加载普通模型的地方都能用 | 重复一份基座；来源只体现在模型卡里 |
| GGUF | 约 4–8 GB（Q4） | CPU 和端侧可跑 | 每档量化一个文件；视觉投影层处理有风险 |

默认发 adapter。确实有使用方用不了 PEFT 时再发合并权重，并在模型卡里写明原因。上游已经有
官方 GGUF 时，不建议再发一份 —— 那只会把用户分散到两个各有各的 bug 的文件上。

## 三、必须交付的文件

### `LICENSE`

Apache-2.0 的完整原文。不要改写、不要截断、也不要用链接代替：

```bash
curl -fsSL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE
```

### `NOTICE`

```text
MiniCPM5-2B derivative
Copyright 2026 <作者>

This product includes model weights derived from MiniCPM5-2B
(openbmb/MiniCPM5-2B), developed by OpenBMB, THUNLP (Tsinghua University)
and the Gaoling School of Artificial Intelligence at Renmin University of
China, and licensed under Apache-2.0.

Original repository: https://github.com/OpenBMB/MiniCPM
Modifications: fine-tuned on additional data. See README.md.
```

### `README.md` 的 frontmatter

```yaml
license: apache-2.0
base_model: openbmb/MiniCPM5-2B
library_name: peft          # 合并权重则写 transformers
pipeline_tag: text-generation
tags:
  - minicpm
  - lora
  - text-generation
```

`base_model` 必须是真实的基座。想做得更稳，在正文里把基座的 revision 也固定下来 ——
`main` 是会变的，一年之后别人加载到的应该还是你评估过的那份权重。

## 四、真正有分量的几节内容

- **这是什么** —— 基座模型、产物形式、许可证。
- **训练** —— 方法、硬件、序列长度、轮数、学习率、基座 revision。
- **数据** —— 来源、规模、语言、许可证。不能公开就明确说，使用者需要知道行为可不可复现。
- **评估** —— 基座 / 微调 / 差值的实测表格。
- **局限** —— 观察到的失效场景。这一节是防止别人出事的关键。
- **用法** —— 一段能直接跑的示例，包括怎么加载 adapter、该用哪个 tokenizer。

## 五、tokenizer 这个坑，值得单独说一遍

如果训练时用了"只在助手回复上算损失"的对话模板补丁，那份模板是训练产物：**不能随
tokenizer 保存，也不能发出去**。使用方必须从基座模型加载**原本的** tokenizer。

这样训出来的 adapter 与基座模板是兼容的（token 序列一致），但前提是发出去的不是那份打了
补丁的模板。

## 六、上传命令

```bash
# Hugging Face
huggingface-cli login                     # 或 export HF_TOKEN
huggingface-cli upload <用户名>/<仓库名> ./adapter_final . --repo-type model

# ModelScope
modelscope login                          # 或 export MS_TOKEN
modelscope upload <用户名>/<仓库名> . --repo-type model
```

ModelScope 的仓库名区分大小写，格式是 `Owner/Model-Name`。

## 七、发布检查清单

- [ ] 完整的 `LICENSE` 原文已放入
- [ ] `NOTICE` 写明了 OpenBMB、THUNLP、ModelBest
- [ ] `base_model` 正确，且 revision 已固定
- [ ] 评估表是实测数字
- [ ] 局限一节写了实际观察到的失效场景
- [ ] 数据来源有说明，或明确说明不能公开
- [ ] 发布的是 adapter 而非合并权重（若选合并，模型卡里说明了原因）
- [ ] 该模型的许可证已核实
- [ ] tokenizer 用的是基座的，不是训练时打补丁的那份
- [ ] 冒烟测试：在干净环境里加载已发布产物并生成一次

## 八、版本管理

最低限度的做法：

- 记录一次训练：基座 revision + 数据哈希 + 配置文件。
- adapter 的 `adapter_config.json` 原样保留，它就是来源证明。
- 记下当时用的框架版本（`trl`、`peft`、`transformers`），因为换一个 `peft` 版本再合并，
  得到的权重可能不同。
- 重新训练就发新仓库或新 revision —— 不要覆盖别人已经评估并认可过的权重。
