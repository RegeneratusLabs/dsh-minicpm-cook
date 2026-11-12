# OpenBMB 目前公开了什么

[English](../whats-available.md) · **中文**

已经存在的材料、各自覆盖的范围，以及缺口在哪里。第一次接触这个模型系列的话，先看这篇。

---

## 结论先说

截至 2026 年 11 月，**OpenBMB 已经发布了不止一份"手册"**：一个文档站、十份部署说明、
五份微调指南、**18 个采用 `SKILL.md` 格式的 Agent Skills**、一份与 OpenSQZ 合作的
多模态手册，以及 vLLM / SGLang 各自的模型配方页。

**缺的不是手册，而是一份目录。** 所有材料都是按模型或按框架组织的，而用户的问题是跨
模型的。本项目的价值就在这里。

## 现有材料一览

| 材料 | 是什么 | 覆盖范围 | 形式 |
| --- | --- | --- | --- |
| [openbmb.cn/docs](https://openbmb.cn/docs/) | 官方知识库（MiniCPM Wiki），中文，带 `/en/` 英文镜像 | 每个模型的微调指南、硬件要求、部署 | 网页 |
| [OpenBMB/MiniCPM `docs/finetune/`](https://github.com/OpenBMB/MiniCPM/tree/main/docs/finetune) | 5 个文件：`trl.md`、`llamafactory.md`、`ms_swift.md`、`unsloth.md`、`xtuner.md` | MiniCPM5 的微调 | Markdown，含可运行片段 |
| [OpenBMB/MiniCPM `docs/deployment/`](https://github.com/OpenBMB/MiniCPM/tree/main/docs/deployment) | 10 个文件 | transformers、vLLM、SGLang、llama.cpp、Ollama、LM Studio、MLX、ArcLight、LiteRT-LM、vLLM-Ascend | Markdown |
| [OpenBMB/MiniCPM `skills/`](https://github.com/OpenBMB/MiniCPM/tree/main/skills) | **18 个 Agent Skills** | `minicpm5-finetune-{trl,llamafactory,ms-swift,unsloth,xtuner,gguf-lora}`、`minicpm5-deploy-{vllm,sglang,transformers,llama-cpp,ollama,lmstudio,mlx,arclight,litert,vllm-ascend}` 等 | `SKILL.md` + frontmatter |
| [OpenSQZ/MiniCPM-V-CookBook](https://github.com/OpenSQZ/MiniCPM-V-CookBook) | V 系列与 o 系列的手册，由 OpenBMB **与 OpenSQZ 共同**维护 | 推理、微调、量化、Web Demo、iOS | 仓库 + [文档站](https://opensqz.github.io/MiniCPM-V-CookBook/) |
| [vLLM recipes](https://recipes.vllm.com.cn/openbmb/MiniCPM-V-4.6) | 每个模型一页 | 仅 vLLM | Markdown |
| [SGLang cookbook](https://github.com/sgl-project/sglang/tree/main/docs_new/cookbook/autoregressive/OpenBMB) | 每个模型一页 | 仅 SGLang | Markdown |
| [minicpm-o.readthedocs.io](https://minicpm-o.readthedocs.io/en/latest/index.html) | o 系列文档站 | 全模态模型 | ReadTheDocs |

OpenBMB 名下**没有**一个类似 `MiniCPM-cookbook` / `MiniCPM-recipes`、覆盖全系列的仓库，
也没有对应 Meta `llama-cookbook` 的文本模型统一手册。

## 一个重要发现：上游已经在用 SKILL.md

OpenBMB 的 `skills/` 目录，格式与 DeepSeek Harness 读取的技能完全一致：每个目录一个
`SKILL.md`，frontmatter 里有 `name` 和 `description`，正文包含选择表、步骤、验证要求、
常见坑，以及指回 `docs/` 对应页面的链接。

这直接改变了插件该怎么做。**把它们复制进插件是错的** —— 它们按模型版本迭代，几乎每月
更新，复制一份就等于埋下一个会悄悄教错模板的隐患。所以本项目的技能只覆盖"跨模型"的
判断，而安装说明会告诉你怎么把上游那 18 个技能一起挂载进来。

## 五个真实缺口

1. **没有跨模型目录。** 所有材料都是按模型或框架组织的。想回答"哪个 8B 多模态模型既有
   vLLM 支持、许可证又宽松"，得读四个仓库。
2. **没有版本兼容检查。** 要求散落在各页正文里：MiniCPM5 要 `transformers>=5.6`，V4.6
   要 `>=5.7.0`，还有 `trl>=0.21`、`vLLM>=0.21`、`sglang>=0.5.16`。没有任何工具拿你的
   环境去比对。
3. **没有数据校验。** 字段写错、图片路径用了通配符、最后一条是用户提问、样本悄悄超出
   上下文长度 —— 这些问题都要等到开训之后、账单出来才发现。
4. **没有发布环节。** 上游完全没有讲微调模型怎么发布：模型卡、`NOTICE` 文件、发布
   adapter 还是合并权重、怎么上传到 Hub 和 ModelScope。Apache-2.0 本身很宽松，但前提
   是署名做对了。
5. **没有版本漂移检测。** 仅 2026 年就发布了五个模型（2 月 SALA、5 月 MiniCPM5-1B、
   5 月 V4.6、9 月 MiniCPM5-2B、10 月 V4.7 与 OCR）。任何静态手册，一个季度后就会过时。

## 本项目做了什么

- `data/models.json` —— 一份机器可读的目录，17 个模型，每条事实都带来源链接和
  `confidence`（是否已核实）标记。显存占用是**算出来的**，不是抄来的。
- 七个工具 —— 选型、方案、配方、数据校验、环境体检、部署、发布。
- 四个技能 —— 那些"按模型拆分的文档无法表达"的跨模型判断。
- `scripts/check-upstream.mjs` —— 把上游变化变成一次失败的检查，而不是一条悄悄过时的建议。

## 本项目明确不声称的部分

- **构建过程中没有实际训练或部署过。** 这台机器没有显卡，也没有装 PyTorch；配方只做了
  语法解析和结构断言。
- **17 条目录记录中有 5 条标注为"未核实"**，并写明了要确认什么。许可证未核实的模型，
  发布流程会被直接拦截。
