# 这个仓库里每个文件是干什么的

> 235 个跟踪文件。按"你会想找什么"分组，不按字母表。
>
> **这份文档是生成的**：`python src/by1files.py --write`。
> 文件清单、每组的个数、每个文件的一句话都从目录树和 docstring 来 ——
> 所以它不会过期。要改措辞，改 `src/by1files.py` 里的 `GROUPS`。

## 先读哪三份

| 想干什么 | 读哪份 |
|---|---|
| **它到底是什么东西** | `README.md` |
| **当初为什么这么做、错过什么** | `history/ir.md` |
| **每个文件干嘛的** | 就是这份 |

`1.md` 是立项时的宣言，写的是**要做什么**。`history/ir.md` 写的是
**做的时候撞上了什么** —— 后者比前者值钱，因为成功的部分哪份 README
都能写，失败的部分只有日志有。

---

## 根目录（12 个）

| 文件 | 用途 |
|---|---|
| `.gitattributes` | 行尾一律 LF。没有它的后果见文件里的注释 |
| `.gitignore` | 产物、权重、缓存、临时探针 |
| `1.md` | 立项宣言：这门语言要解决什么、不解决什么 |
| `FILES.md` | **这份文档本身**，由 `src/by1files.py` 生成 |
| `LICENSE` | MIT。里面另有一节说明 `refs/` 不是本项目的作品 |
| `README.md` | 门面。是什么、怎么跑、数据从哪来 |
| `TESTING.md` | 给**想验它的人**看：三档命令、每档耗时、“通过/没验/失败”各长什么样、反馈模板 |
| `VERSION` | 版本号。**由 `by1ver.py --write` 生成**，不要手改 |
| `ir-spec.md` | **生成物**：IR 规格（字段、必填、语义、闭集）。`by1ir.py --spec` |
| `models.md` | **生成物**：所有模型的架构对比表。`by1cmp.py --md` |
| `models.tsv` | **手写的模型清单**：长名 / 短名 / HF id / 说明。命名规矩写在文件头 |
| `pyproject.toml` | 打包元数据。`py-modules` + `data-files`（没有 package） |

生成物有两份判卷人看着（`by1docs.py`），所以改了 schema 忘了重新生成会被抓出来。

---

## `src/` —— 64 个模块 + 1 个非 .py

### 底座（留在 src/ 顶层）

**被别的模块 import 的那 12 个。**

它们必须在 `src/` 顶层，因为 `by1paths.py` 自己也在这里 ——
它是引导，搬它就得先有引导，而引导又得先找得到它。

| 被谁 import |
|---|
| `by1io` 39 个模块 · `by1paths` 33 个 · `by1codegen` 17 个 · `by1skip` 11 个 |
| `by1ir` 8 个 · `by1ver` 6 个 · `by1exec` 5 个 · `by1c` 4 个 |
| `by1refs` 3 个 · `by1check` 2 个 · `by1boot` 2 个 · `by1ext` 2 个 |

**其余 44 个是叶子**（没人 import 它们），所以能进子目录。

每个入口脚本开头那几行引导是**生成的**（判据在 `by1paths.check_boot`）——
重复的东西必须能被检查，否则新加一个脚本忘了加，症状是
`No module named 'by1paths'`，看起来像"文件丢了"。

| 文件 | 用途 |
|---|---|
| `by1paths.py` | 仓库布局的唯一真相源 |
| `by1io.py` | 文件读写的每一种意图，各有一个名字 |
| `by1codegen.py` | 把 .by1 编译成**后端无关的 IR**，再由后端生成可训练模型 |
| `by1skip.py` | 一次检查有五种结论，不是两种 |
| `by1ir.py` | 规范化的 by1 IR：schema、校验、JSON 往返 |
| `by1ver.py` | 版本号的唯一真相源 |
| `by1exec.py` | 第二个后端：纯 NumPy 执行器 |
| `by1c.py` | 第三个后端：从 IR 生成 **C 代码**，编译，跑起来，和 NumPy 后端对拍 |
| `by1refs.py` | refs/ 的路径规则，只此一处 |
| `by1check.py` | 不变式检查器 v0.1 |
| `by1boot.py` | 从一个已发布的 checkpoint **反推**一份 .by1 草稿 |
| `by1ext.py` | 逃生舱第二层：**引用一个外部符号** |

### `lang/` —— 语言本身

把 `.by1` 变成一份能算的东西。

`by1export` 是**导出策略层** —— 把算好的语义变成 HF config 的字段。
它原来是 `by1check.check()` 里嵌着的 13 个 `gen_*` / `name_*`
（那个函数因此有 1243 行）。**劈开的理由写在那个文件的头注里。**

| 文件 | 用途 |
|---|---|
| `by1.py` | 用一份描述，写一个语言模型 |
| `by1contract.py` | 从机制声明**推出**张量契约 |
| `by1emit.py` | 从 IR 生成 **ggml 侧的后端需求 + 声明式图**，并和 GGUF manifest 双向对拍 |
| `by1name.py` | 模型命名的唯一执行者 |
| `by1vocab.py` | 真实数据的词汇表。模式从这里来，不从想象来 |
| `by1export.py` | 导出策略层：把算好的语义变成 HF config 的字段 |
| `by1blocks.py` | 读 `.by1` 里那些"块"，把不一致报出来 |
| `by1tens.py` | 张量契约实例化：把契约按等价类求值出形状 |
| `by1lower.py` | emit lowering：把 emit 块降成 `字段名 -> 规则` |
| `by1hp.py` | hparams 块：模型有多宽、多少层、词表多大 |
| `by1sched.py` | schedule：命名子序列的多趟展开 |
| `by1stacks.py` | stacks：栈怎么排、每层挂什么机制 |
| `by1state.py` | 按选择器把 state 声明落到具体的层上 |

### `checks/` —— 判卷人 · 神谕 · 跑检查

这个项目最核心的资产。

判卷人回答"**我怎么知道它是对的**"；神谕回答"**是不是一起错了**"——
三个后端互拍只能证明自洽，而它们可以一致地错（见 `history/ir.md`）。

跑检查的两个入口也在这里（`by1run` · `by1fast`）。

| 文件 | 用途 |
|---|---|
| `by1all.py` | 一次跑完全部验证，失败就非零退出 |
| `by1verify.py` | 差分验证：把 .by1 的展开结果与官方产物对拍 |
| `by1diff.py` | 把 by1 生成的前向 与 transformers 的参考实现**逐位对拍** |
| `by1opdiff.py` | 逐个算子**比 NumPy 和 PyTorch |
| `by1e2e.py` | 端到端：真产物 -> IR -> 三个后端 -> 对官方实现 |
| `by1irentry.py` | 三个后端必须能从 IR 入口跑通**，而不是只从 .by1 入口 |
| `by1gate.py` | 取值门的**可证伪对照** |
| `by1docs.py` | """文档里提到的每一个路径，都还在吗？ |
| `by1files.py` | 生成并核对 `FILES.md` |
| `by1lint.py` | 只盯这个项目真正怕的那几类 |
| `by1blind.py` | 量"看不见的东西" |
| `by1debt.py` | 没干完的事，量出来，不凭印象列 |
| `by1pat.py` | 哪些词是数据里的，哪些是我的想象 |
| `by1smoke.py` | """冒烟测：把每个 by1*.py 都单独跑一遍 |
| `by1pack.py` | 打出要上传到云端的那一包 |
| `by1rebuild.py` | 删掉数据，能不能重建 |
| `by1bootir.py` | 两条路的判卷人**：从 .by1 出来的 IR，和从产物反推出来的 IR |
| `by1oracles.py` | 10 个神谕：每个都有独立的真相源 |
| `by1oracle.py` | 执行神谕：把 .by1 声明的状态策略与**真实前向的行为**对拍 |
| `by1run.py` | 推送前的门，以及每四次一次数值档 |
| `by1fast.py` | 快速检查：30 秒内告诉你有没有改坏 |
| `by1gpu.py` | 只有显卡能做的那几件验证 |

### `modelcheck/` —— 一个模型一个判卷人

每个都用自己的官方产物当真相源。

| 文件 | 用途 |
|---|---|
| `by1gpt2.py` | GPT-2 前向的差分验证，**用真的 GPT-2** |
| `by1instella.py` | Instella-3B 前向的差分验证，**用真实维度** |
| `by1mla.py` | MLA 的差分验证 |
| `by1moe.py` | noaux_tc 分组路由的差分验证 |
| `by1rope.py` | llama3 式 RoPE 缩放的差分验证 |
| `by1real.py` | 真权重、真维度，跑一个 27B 模型的前向 |
| `by1load.py` | 把**真权重**装进 by1 生成的模型 |
| `by1mem.py` | 从 IR 的 state 声明推出**内存计划**，并可选地用真实模型核它 |
| `by1train.py` | 按 .by1 的描述搭一个模型，训练它，然后让它写字 |
| `by1dev.py` | 设备无关性**的判卷人 |

### `data/` —— 抓数据 / 推模型

从一个已发布的 checkpoint 反推描述，以及把 `refs/` 抓回来。

| 文件 | 用途 |
|---|---|
| `by1fetch.py` | 一条命令，把 refs/ 全部重抓回来 |
| `by1index.py` | 抓张量名清单。**免费，而且这是"能不能描述"的真正判据 |
| `by1boot.py` | 从一个已发布的 checkpoint **反推**一份 .by1 草稿 |
| `by1triage.py` | 把一批新 config 过一遍 by1，看谁需要新原语 |
| `by1cmp.py` | 把所有模型的架构摆成一张表 |
| `by1cloud.py` | 驱动一台租来的显卡机器**（AutoDL 或任何 SSH 可达的） |

### `escape/` —— 逃生舱

语言表达不了的东西有两个出口，**两个都不是"绕过检查"**：
张量契约、三后端对拍、取值门照旧生效。

| 文件 | 用途 |
|---|---|
| `by1raw.py` | 逃生舱的**判卷人** |
| `by1extdemo.py` | 逃生舱第二层的**判卷人** |
| `ext-demo.c` | 逃生舱第二层的示例：两个语言里没有的机制。`by1extdemo` 编它 |

---

## `docs/` —— 7 个

活的文档 —— 还没做完的事记在这里，做完的挪去 `history/`。

| 文件 | 用途 |
|---|---|
| `INVENTORY.txt` | 验证清单：每个模型验到了什么、判卷人是谁 |
| `check-map.md` | `check()` 的结构图 |
| `criteria-review.md` | **请人重算**的任务卡：判据本身没人验过，这一份是那份活。含"refs 判 by1 不是外部判卷"那一节 |
| `ggml-plan.md` | 接 llama.cpp：第一个外部判卷人 |
| `pattern-design-review.md` | 对上面那份草案的审查 —— **照做之前先读它** |
| `pattern-design.md` | 设计草案 v0，**状态：待审**。它挡着 4 个模型 |
| `which-gpu.md` | > 从 `README.md` 挪出来的。它是一次**具体的调查** —— 这台开发机上 |

## `docs/delete-test-1/` —— 4 个

| 文件 | 用途 |
|---|---|
| `README.md` | 门面。是什么、怎么跑、数据从哪来 |
| `agent-report.md` | 搞定了。结果在 build：9 个路径全部就位，字节数和 MANIFEST.txt 逐条对上，合计 1234405 |
| `rebuild-record.md` | 重建记录：MANIFEST.txt 的 9 个路径 |
| `rebuild_manifest.py` | 复现脚本 |

## `docs/delete-test-1/test-definition/` —— 2 个

| 文件 | 用途 |
|---|---|
| `MANIFEST.txt` | 被拿走的东西 —— 按原来的相对路径列在这里 |
| `README.md` | 门面。是什么、怎么跑、数据从哪来 |

## `drafts/` —— 3 个

还没进 `models/` 的草稿。**没有判卷人，所以不算数。**

| 文件 | 用途 |
|---|---|
| `README.md` | 门面。是什么、怎么跑、数据从哪来 |
| `deepseek-v4.1-flash.by1` | 验收 #5（最难的一个）。写它的目的就是逼出语法缺口，所以每一处当前草案 |
| `qwen3.8-flash-next.by1` | model Qwen3.8-Flash-Next { |

## `gpu/` —— 4 个

只有显卡上能跑的那部分。

| 文件 | 用途 |
|---|---|
| `README.md` | 门面。是什么、怎么跑、数据从哪来 |
| `by1kda.py` | KDA（Kimi Delta Attention）的验证。它在 `by1check` 里是独立种类，不是 GDN |
| `requirements.txt` | 依赖（和本地不同：要 CUDA 版的 torch） |
| `run.sh` | 入口 |

## `history/` —— 3 个

以前的东西。留着的理由是：能看出当时是怎么想的。

| 文件 | 用途 |
|---|---|
| `README.md` | 门面。是什么、怎么跑、数据从哪来 |
| `ir.md` | Block 0 · 中间表示（IR）规格 |
| `plan.md` | by1 · 接下来的计划 |

## `models/` —— 27 个

命名规矩：**文件名 = 长名 = HF 的模型名（去掉 owner）**，
`model` 声明 = 短名（手写的）。两列的对照见 `models.tsv`。

| 文件 | 用途 |
|---|---|
| `GLM-5.3-Flash.by1` | 真实模型 · HF id `zai-org/GLM-5.3-Flash` |
| `Instella-3B.by1` | 真实模型 · HF id `amd/Instella-3B` |
| `Laguna-XS-2_1.by1` | 真实模型 · HF id `poolside/Laguna-XS-2_1` |
| `Ling-3.0-tiny.by1` | 真实模型 · HF id `inclusionAI/Ling-3.0-tiny` |
| `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.by1` | 真实模型 · HF id `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` |
| `Qwen3.6-35B-A3B.by1` | 真实模型 · HF id `Qwen/Qwen3.6-35B-A3B` |
| `Qwen3.8-27B.by1` | 真实模型 · HF id `Qwen/Qwen3.8-27B` |
| `Step-3.7-Flash.by1` | 真实模型 · HF id `stepfun-ai/Step-3.7-Flash` |
| `Step-3_7-Flash-180B-LynnStyle-GLM52-SFT-GPT55-RL.by1` | 真实模型 · HF id `nerkyor/Step-3_7-Flash-180B-LynnStyle-GLM52-SFT-GPT55-RL` |
| `clef-tiny.by1` | 目的：证明"对着真 checkpoint 验过的那份描述"**同时是能跑的** |
| `clef.by1` | 真实模型 · HF id `Cloudflare/clef` |
| `gate-probe.by1` | 取值门的**反例**：故意写一个 codegen 没实现的取值 |
| `gemma-4-31B.by1` | 真实模型 · HF id `google/gemma-4-31B` |
| `gpt-oss-120b.by1` | 真实模型 · HF id `openai/gpt-oss-120b` |
| `gpt-oss-shaped.by1` | Block 1.2 的第二个判卷对象：GPT-OSS 的注意力 + MoE |
| `gpt2-tiny.by1` | 由 gpt2.by1 机械缩小维度得来，结构一个字没改 |
| `gpt2.by1` | 真实模型 · HF id `openai-community/gpt2` |
| `hello.by1` | 一个最小的语言模型。改数字就能改大小 —— 改完重新跑一次就行 |
| `llama-shaped.by1` | 与 transformers 的 LlamaForCausalLM 结构完全一致的最小模型： |
| `llama3-shaped.by1` | llama3 式 RoPE 缩放的最小可跑模型 |
| `minimind-3.by1` | 真实模型 · HF id `jingyaogong/minimind-3` |
| `mixtral-shaped.by1` | Block 1.1 的判卷对象：MoE |
| `mla-shaped.by1` | MLA 的最小可跑模型 |
| `qwen3-next-shaped.by1` | Block 1.3 的判卷对象：**混合栈 + 递归状态** |
| `raw-escape.by1` | 逃生舱的**可证伪对照** |
| `raw.py` | **不是模型，是逃生舱的工厂函数实现**。必须和 `raw-escape.by1` 挨着（`by1codegen` 按名字找它） |
| `selftest.by1` | 检查器自检。这个文件里每一处都是故意的错，用来证明检查不是空过 |

## `refs/` —— 73 个

从 HuggingFace / ModelScope 抓的**模型元数据**（配置 + 张量名/形状清单），
**不是权重**。版权属各发布方，各自适用各自的许可证。

| 种类 | 个数 | 是什么 |
|---|---|---|
| `*.config.json` | 35 | |
| `*.gguf.json` | 3 | |
| `*.tensors.json` | 34 | |
| `SOURCES.tsv` | 1 | |


## `src/checks/` —— 22 个

| 文件 | 用途 |
|---|---|
| `by1all.py` | 一次跑完全部验证，失败就非零退出 |
| `by1blind.py` | 量"看不见的东西" |
| `by1bootir.py` | 两条路的判卷人**：从 .by1 出来的 IR，和从产物反推出来的 IR |
| `by1debt.py` | 没干完的事，量出来，不凭印象列 |
| `by1diff.py` | 把 by1 生成的前向 与 transformers 的参考实现**逐位对拍** |
| `by1docs.py` | """文档里提到的每一个路径，都还在吗？ |
| `by1e2e.py` | 端到端：真产物 -> IR -> 三个后端 -> 对官方实现 |
| `by1fast.py` | 快速检查：30 秒内告诉你有没有改坏 |
| `by1files.py` | 生成并核对 `FILES.md` |
| `by1gate.py` | 取值门的**可证伪对照** |
| `by1gpu.py` | 只有显卡能做的那几件验证 |
| `by1irentry.py` | 三个后端必须能从 IR 入口跑通**，而不是只从 .by1 入口 |
| `by1lint.py` | 只盯这个项目真正怕的那几类 |
| `by1opdiff.py` | 逐个算子**比 NumPy 和 PyTorch |
| `by1oracle.py` | 执行神谕：把 .by1 声明的状态策略与**真实前向的行为**对拍 |
| `by1oracles.py` | 10 个神谕：每个都有独立的真相源 |
| `by1pack.py` | 打出要上传到云端的那一包 |
| `by1pat.py` | 哪些词是数据里的，哪些是我的想象 |
| `by1rebuild.py` | 删掉数据，能不能重建 |
| `by1run.py` | 推送前的门，以及每四次一次数值档 |
| `by1smoke.py` | """冒烟测：把每个 by1*.py 都单独跑一遍 |
| `by1verify.py` | 差分验证：把 .by1 的展开结果与官方产物对拍 |

## `src/data/` —— 5 个

| 文件 | 用途 |
|---|---|
| `by1cloud.py` | 驱动一台租来的显卡机器**（AutoDL 或任何 SSH 可达的） |
| `by1cmp.py` | 把所有模型的架构摆成一张表 |
| `by1fetch.py` | 一条命令，把 refs/ 全部重抓回来 |
| `by1index.py` | 抓张量名清单。**免费，而且这是"能不能描述"的真正判据 |
| `by1triage.py` | 把一批新 config 过一遍 by1，看谁需要新原语 |

## `src/escape/` —— 3 个

| 文件 | 用途 |
|---|---|
| `by1extdemo.py` | 逃生舱第二层的**判卷人** |
| `by1raw.py` | 逃生舱的**判卷人** |
| `ext-demo.c` | 逃生舱第二层的示例：两个语言里没有的机制。`by1extdemo` 编它 |

## `src/lang/` —— 13 个

| 文件 | 用途 |
|---|---|
| `by1.py` | 用一份描述，写一个语言模型 |
| `by1blocks.py` | 读 `.by1` 里那些"块"，把不一致报出来 |
| `by1contract.py` | 从机制声明**推出**张量契约 |
| `by1emit.py` | 从 IR 生成 **ggml 侧的后端需求 + 声明式图**，并和 GGUF manifest 双向对拍 |
| `by1export.py` | 导出策略层：把算好的语义变成 HF config 的字段 |
| `by1hp.py` | hparams 块：模型有多宽、多少层、词表多大 |
| `by1lower.py` | emit lowering：把 emit 块降成 `字段名 -> 规则` |
| `by1name.py` | 模型命名的唯一执行者 |
| `by1sched.py` | schedule：命名子序列的多趟展开 |
| `by1stacks.py` | stacks：栈怎么排、每层挂什么机制 |
| `by1state.py` | 按选择器把 state 声明落到具体的层上 |
| `by1tens.py` | 张量契约实例化：把契约按等价类求值出形状 |
| `by1vocab.py` | 真实数据的词汇表。模式从这里来，不从想象来 |

## `src/modelcheck/` —— 10 个

| 文件 | 用途 |
|---|---|
| `by1dev.py` | 设备无关性**的判卷人 |
| `by1gpt2.py` | GPT-2 前向的差分验证，**用真的 GPT-2** |
| `by1instella.py` | Instella-3B 前向的差分验证，**用真实维度** |
| `by1load.py` | 把**真权重**装进 by1 生成的模型 |
| `by1mem.py` | 从 IR 的 state 声明推出**内存计划**，并可选地用真实模型核它 |
| `by1mla.py` | MLA 的差分验证 |
| `by1moe.py` | noaux_tc 分组路由的差分验证 |
| `by1real.py` | 真权重、真维度，跑一个 27B 模型的前向 |
| `by1rope.py` | llama3 式 RoPE 缩放的差分验证 |
| `by1train.py` | 按 .by1 的描述搭一个模型，训练它，然后让它写字 |

## `tools/remote/` —— 34 个

| 文件 | 用途 |
|---|---|
| `README.md` | 门面。是什么、怎么跑、数据从哪来 |
| `remote-asan.sh` | 用 sanitizer 找 C 后端在 Linux 上算错的原因 |
| `remote-clef-dl.sh` | 把 clef 的 55 GB 下到 /dev/shm（它是 RAM 盘，60 GB） |
| `remote-clef-info.sh` | 看清 clef 那个仓库有多大、什么格式，再决定下不下 |
| `remote-dl-check.sh` | echo "=== clef 下载进度 ===" |
| `remote-e2e.sh` | cd /root/by1 |
| `remote-fc.sh` | cd /root/by1 |
| `remote-fix2.sh` | 收掉远端剩下的两个失败。都是环境，不是代码 |
| `remote-go.sh` | 启动 + 当场量**总**速度。单个分片 14.7 MB/s 只是六条流里的一条 |
| `remote-home.sh` | HOME 设了吗？缓存找得到吗？ |
| `remote-launch.sh` | 只启动，不量。** 量是下一个脚本的事 —— 合在一起会阻塞 |
| `remote-m.sh` | 只量，不启动。** 30 秒 |
| `remote-m20.sh` | A=$(du -sm /dev/shm/qwen38 2>/dev/null | cut -f1); A=${A:-0} |
| `remote-mem.sh` | echo "=== 内存 cgroup 上限（这才是真正管用的那个）===" |
| `remote-mirror.sh` | 限时 60 秒：镜像那个 URL 到底给字节，还是 302 到别处？ |
| `remote-more.sh` | 加流：8 -> 32。已下完的分片会复用，不重下 |
| `remote-ms.sh` | 换源：不走 HF/Xet，走 ModelScope（阿里的，Qwen 是阿里的模型） |
| `remote-ms2.sh` | 装 modelscope + 用真权重分片测速。**全程限时 |
| `remote-net.sh` | pkill -f dl1.py 2>/dev/null |
| `remote-noXet.sh` | Xet 是 HF 的新存储后端（cas-bridge.xethub.hf.co）， |
| `remote-ps.sh` | PID=$(pgrep -f by1real | head -1) |
| `remote-qwen-dl.sh` | 从 ModelScope 下 Qwen3.8-27B 的 55.6 GB 到 /dev/shm |
| `remote-rate.sh` | A=$(du -sm /dev/shm/clef 2>/dev/null | cut -f1) |
| `remote-rate2.sh` | 量**总**速度 —— 单个分片 14.7 MB/s，但那是六条流里的一条 |
| `remote-real.sh` | 在 A800 上跑真模型的前向。**这是第一次能做这件事 |
| `remote-real2.sh` | 真权重 27B 前向。**后台跑** —— 装 55 GB 要几分钟，不能占着连接 |
| `remote-real3.sh` | 用 -u 重启：非 TTY 时 Python 会缓冲 stdout，日志看起来"卡住了"， |
| `remote-redl.sh` | 关掉 Xet 重下 |
| `remote-run.sh` | 在显卡机器上跑一遍，只输出要紧的行 |
| `remote-speed.sh` | echo "=== 量 30 秒的下载速度 ===" |
| `remote-timed.sh` | 全程限时。** 卸载 + 测速，5 分钟封顶。到点就停，不磨 |
| `remote-weight.sh` | 限时 100 秒：直接 curl 几个**权重**文件的前几 MB，看真实速度 |
| `remote-which.sh` | 限时 90 秒：到底走没走 hf-mirror？ |
| `remote-xet.sh` | 限时 90 秒：这 14 个模型，哪些不在 Xet 上（不在的才下得动） |


---

## 磁盘上但没被跟踪的

| 目录 | 是什么 | 为什么没跟踪 |
|---|---|---|
| `llamacpp/` | 从 llama.cpp 拉的参照源码 | 可重新下载；`by1instella.py` 会读它 |
| `__delete_test/files/` | 删数据测试那 5% 的副本 | 和仓库里的原件是同一批字节，跟踪会存两遍 |
| `__pycache__/` | Python 字节码 | 每次运行重新生成 |

---

## 两个约定

**一、同一个意思不要两处实现。**
路径规则在 `src/by1refs.py`，仓库布局在 `src/by1paths.py`，版本号在
`src/by1ver.py`，文件读写在 `src/by1io.py`。"顺手再写一遍"是这个仓库里
反复出现过的那类 bug。

**二、能描述 ≠ 能算。**
三个后端对拍只能证明自洽。真正的判据在 `src/by1oracles.py` 那 10 个神谕里，
以及 `src/by1verify.py` 对着官方产物的对拍里。
