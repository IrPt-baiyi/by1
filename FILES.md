# 这个仓库里每个文件是干什么的

> 217 个跟踪文件。按"你会想找什么"分组，不按字母表。

## 先读哪三份

| 想干什么 | 读哪份 |
|---|---|
| **它到底是什么东西** | `README.md` |
| **当初为什么这么做、错过什么** | `history/ir.md`（73 节） |
| **每个文件干嘛的** | 就是这份 |

`1.md` 是立项时的宣言，写的是**要做什么**。`history/ir.md` 写的是
**做的时候撞上了什么** —— 后者比前者值钱，因为成功的部分哪份 README
都能写，失败的部分只有日志有。

---

## 根目录（10 个）

| 文件 | 用途 |
|---|---|
| `README.md` | 门面。是什么、怎么跑、数据从哪来 |
| `1.md` | 立项宣言：这门语言要解决什么、不解决什么 |
| `LICENSE` | MIT。里面另有一节说明 `refs/` 不是本项目的作品 |
| `VERSION` | 版本号。**由 `by1ver.py --write` 生成**，不要手改 |
| `pyproject.toml` | 打包元数据。`py-modules` + `data-files`（没有 package） |
| `.gitattributes` | 行尾一律 LF。没有它的后果见文件里的注释 |
| `.gitignore` | 产物、权重、缓存、临时探针 |
| `models.tsv` | **手写的模型清单**：长名 / 短名 / HF id / 说明。命名规矩写在文件头 |
| `ir-spec.md` | **生成物**：IR 规格（字段、必填、语义、闭集）。`by1ir.py --spec` |
| `models.md` | **生成物**：所有模型的架构对比表。`by1cmp.py --md` |

生成物有两份判卷人看着（`by1docs.py` 检查它们和源头一致），
所以改了 schema 忘了重新生成会被抓出来。

---

## `src/` —— 53 个模块 + 1 个 C 示例

### 语言本身

| 文件 | 用途 |
|---|---|
| `by1check.py` | 解析 `.by1` + 不变式检查。**最大的一个文件**（89 KB） |
| `by1ir.py` | 规范化 IR：schema、校验、JSON 往返。`KIND_ATTRS` 是合法 IR 的唯一定义 |
| `by1emit.py` | 从 IR 生成 ggml 侧的后端需求 + 声明式图，和 GGUF manifest 双向对拍 |
| `by1contract.py` | 从机制声明**推出**张量契约（不靠手写） |
| `by1codegen.py` | 把 `.by1` 编译成后端无关 IR，再生成可训练的 `nn.Module` |
| `by1vocab.py` | 真实数据的词汇表。**模式从这里来，不从想象来** |
| `by1name.py` | 模型命名的唯一执行者（长名 → 文件名，短名 → `model` 声明） |

### 三个后端

| 文件 | 用途 |
|---|---|
| `by1codegen.py` | PyTorch：生成真的 `nn.Module` |
| `by1exec.py` | NumPy：纯手工执行器，不依赖 torch |
| `by1c.py` | C：从 IR 生成 C 代码、编译、跑起来，和 NumPy 对拍 |

**三个后端互相印证只能证明自洽，不能证明正确** —— 它们可以"一致地错"。
所以真相源在别处（见下面"神谕"和"对拍"）。

### 工程底座

| 文件 | 用途 |
|---|---|
| `by1paths.py` | **仓库布局的唯一真相源**。找 gcc、找仓库根、找模型目录都问它 |
| `by1io.py` | 文件读写。`read_text` / `head_text` / `iter_lines` / `write_json` … |
| `by1refs.py` | `refs/` 的路径规则，只此一处 |
| `by1ver.py` | 版本号的唯一真相源 |
| `by1skip.py` | **"这一次没验"是一个独立的结论** —— 三态：通过 / 失败 / 跳过 |
| `by1lint.py` | 只盯这个项目真正怕的那几类（静默失败），十类规则各有反例自检 |

### 判卷人（这项目的核心资产）

| 文件 | 用途 |
|---|---|
| `by1all.py` | **一次跑完全部验证**，失败就非零退出。最大的那个判卷人 |
| `by1verify.py` | 差分验证：`.by1` 的展开结果 vs 官方产物 |
| `by1diff.py` | 把 by1 生成的前向与 transformers 参考实现**逐位对拍** |
| `by1opdiff.py` | 逐个算子比 NumPy 和 PyTorch |
| `by1e2e.py` | 端到端：真产物 → IR → 三个后端 → 对官方实现 |
| `by1irentry.py` | 三个后端必须能从 **IR 入口**跑通，而不是只从 `.by1` 入口 |
| `by1gate.py` | 取值门的**可证伪对照** —— 门的判据本身也要能被证伪 |
| `by1contract.py` | （另见上）张量契约 |
| `by1docs.py` | 文档里提到的每一个路径、每一个数字，都还对吗 |
| `by1blind.py` | 量"看不见的东西"：结构性盲区 |
| `by1debt.py` | 没干完的事，量出来，不凭印象列 |
| `by1pat.py` | 哪些识别词是数据里的，哪些是我的想象 |
| `by1smoke.py` | 冒烟测：把每个脚本单独跑一遍，只看有没有 Traceback |
| `by1pack.py` | 打出要上传到云端的那一包（有两个判卷人盯着别漏文件） |
| `by1rebuild.py` | 删掉数据，能不能重建。数据/代码的账 |
| `by1bootir.py` | 两条路的判卷人：从 `.by1` 出来的 IR，和从产物反推出来的 IR |

### 神谕 —— 独立的真相源

| 文件 | 用途 |
|---|---|
| `by1oracles.py` | **10 个神谕，每个都有独立的真相源**。这是"正确"的判据，不是"自洽" |
| `by1oracle.py` | 执行神谕：把 `.by1` 声明的状态策略与真实前向的行为对拍 |

为什么单独一类：三个后端对拍能发现"写岔了"，**发现不了"一起错了"**。
神谕是拿一个完全独立的实现（或者一条数学性质）来判。

### 具体模型的验证

| 文件 | 用途 |
|---|---|
| `by1gpt2.py` | GPT-2 前向的差分验证，用**真的** GPT-2 |
| `by1instella.py` | Instella-3B 前向的差分验证，用真实维度（临时生成 N 层的 `.by1`） |
| `by1mla.py` | MLA 的差分验证 |
| `by1moe.py` | `noaux_tc` 分组路由的差分验证 |
| `by1rope.py` | llama3 式 RoPE 缩放的差分验证 |
| `by1real.py` | 真权重、真维度，跑一个 27B 的前向 |
| `by1load.py` | 把**真权重**装进 by1 生成的模型 |
| `by1mem.py` | 从 IR 的 state 声明推出**内存计划**，可选地用真实模型核它 |
| `by1train.py` | 按 `.by1` 搭一个模型，训练它，然后让它写字 |
| `by1dev.py` | **设备无关性**的判卷人（同一份 IR 在 CPU / GPU 上应当一致） |
| `by1gpu.py` | 只有显卡能做的那几件验证 |

### 逃生舱

语言表达不了的东西有两个出口，两个都**不是"绕过检查"**：
张量契约、三后端对拍、取值门照旧生效。

| 文件 | 用途 |
|---|---|
| `by1raw.py` | 逃生舱第一层的判卷人（`impl = "名字"` → `raw.py` 里的工厂函数） |
| `by1ext.py` | 逃生舱第二层：引用一个外部符号（`.so` + ABI） |
| `by1extdemo.py` | 第二层的判卷人 |
| `src/ext-demo.c` | 示例：两个语言里没有的机制。`by1extdemo` 编它 |

### 抓数据 / 推模型

| 文件 | 用途 |
|---|---|
| `by1fetch.py` | **一条命令把 `refs/` 全部重抓回来**。种子是 `models.tsv` |
| `by1index.py` | 抓张量名清单。"能不能描述"的真正判据就看它 |
| `by1boot.py` | 从一个已发布的 checkpoint **反推**一份 `.by1` 草稿 |
| `by1triage.py` | 把一批新 config 过一遍 by1，看谁需要新原语。分成"认得出但不会展开"和"认不出" |
| `by1cmp.py` | 把所有模型的架构摆成一张表（`models.md` 就是它生成的） |
| `by1cloud.py` | 驱动一台租来的显卡机器（AutoDL 或任何 SSH 可达的） |

---

## `models/` —— 27 个 `.by1`

命名规矩：**文件名 = 长名 = HF 的模型名（去掉 owner）**，
`model` 声明 = 短名（手写的）。两列的对照见 `models.tsv`。

### 真实模型（对着官方产物验过）

| 文件 | 短名 | HF id |
|---|---|---|
| `clef.by1` | clef | Cloudflare/clef |
| `gpt2.by1` | gpt2 | openai-community/gpt2 |
| `GLM-5.3-Flash.by1` | GLM-5.3-Flash | zai-org/GLM-5.3-Flash |
| `Instella-3B.by1` | Instella-3B | amd/Instella-3B |
| `Laguna-XS-2_1.by1` | Laguna-XS-2_1 | poolside/Laguna-XS-2_1 |
| `Ling-3.0-tiny.by1` | Ling-3.0-tiny | inclusionAI/Ling-3.0-tiny |
| `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.by1` | 同左 | nvidia/… |
| `Qwen3.6-35B-A3B.by1` | Qwen3.6-35B-A3B | Qwen/Qwen3.6-35B-A3B |
| `Qwen3.8-27B.by1` | Qwen3.8-27B | Qwen/Qwen3.8-27B |
| `Step-3.7-Flash.by1` | Step-3.7-Flash | stepfun-ai/Step-3.7-Flash |
| `Step-3_7-Flash-180B-LynnStyle-….by1` | 同左 | nerkyor/…（剪枝微调版，**同名冲突所以带 owner**） |
| `gemma-4-31B.by1` | gemma-4-31B | google/gemma-4-31B |
| `gpt-oss-120b.by1` | gpt-oss-120b | openai/gpt-oss-120b |
| `minimind-3.by1` | minimind-3 | jingyaogong/minimind-3 |

### 形状测试（结构照抄某个真实模型，维度缩小到跑得动）

| 文件 | 用途 |
|---|---|
| `clef-tiny.by1` | 由 `clef.by1` 机械缩小维度得到，**结构一个字没改** |
| `gpt2-tiny.by1` | 同上，缩小版 GPT-2 |
| `llama-shaped.by1` | Llama 结构，随机权重也能跑 |
| `llama3-shaped.by1` | 带 llama3 式 RoPE 缩放 |
| `mixtral-shaped.by1` | MoE：路由 + top-k + 逐专家 SwiGLU |
| `gpt-oss-shaped.by1` | gpt-oss 的结构 |
| `qwen3-next-shaped.by1` | Qwen3-Next 的结构 |
| `mla-shaped.by1` | MLA（多头潜在注意力） |

### 合成 / 探针

| 文件 | 用途 |
|---|---|
| `hello.by1` | 最小可跑的例子。**没有 HF 对照**，所以不进前向对拍那一步 |
| `gate-probe.by1` | 专门用来探取值门：故意放非法值，看门会不会红 |
| `raw-escape.by1` | 逃生舱第一层的示例（配 `by1raw.py` 和 `models/raw.py`） |
| `selftest.by1` | 自检用 |
| `raw.py` | **不是模型，是逃生舱的工厂函数实现**。放在这里是因为它必须和 `raw-escape.by1` 挨着 —— `by1codegen` 按名字找它 |

---

## `refs/` —— 73 个文件，**不是本项目的作品**

从 HuggingFace / ModelScope 抓的**模型元数据**（配置 + 张量名/形状清单），
**不是权重**。版权属各发布方，各自适用各自的许可证。

| 种类 | 个数 | 是什么 |
|---|---|---|
| `*.config.json` | 35 | 官方 `config.json`，有的原样、有的转过行尾 |
| `*.tensors.json` | 34 | 张量名 + dtype + shape 清单。从 safetensors 头导出，只读文件头不下载正文 |
| `*.gguf-tensors.json` | 3 | 从 GGUF 头导出（gemma-4-31B / gpt-oss-120b / Laguna） |
| `SOURCES.tsv` | 1 | **没有对应 `.by1` 的那些产物的出处** —— 抓了但还没描述的模型记在这里，免得忘了它们是哪来的 |

**删掉整个 `refs/` 可以一条命令重建**：`python src/by1fetch.py`。
种子是 `models.tsv`。

命名规则：`<owner>__<Repo>.<kind>.json`，owner 和 Repo 从对应 `.by1`
头部的 `# by1-repo:` 推出来（规则在 `by1refs.py`，只此一处）。

---

## `docs/` —— 活文档

| 文件 | 用途 |
|---|---|
| `pattern-design.md` | 设计草案 v0，**状态：待审**。它挡着 4 个模型，因为现有 `pattern` 表达不了"同机制不同模式" |
| `INVENTORY.txt` | 验证清单：每个模型验到了什么、判卷人是谁 |
| `delete-test-1/README.md` | 删数据测试的索引和结论 |
| `delete-test-1/agent-report.md` | 那个无记忆 agent 自己的报告 |
| `delete-test-1/rebuild-record.md` | **逐字节的逆推记录** —— 它怎么试出四种 JSON writer 的规则 |
| `delete-test-1/rebuild_manifest.py` | 复现脚本 |
| `delete-test-1/test-definition/` | 测试的任务书和清单（不含副本，副本 1.2 MB 被 gitignore 挡着） |

---

## `history/` —— 以前的东西

| 文件 | 用途 |
|---|---|
| `ir.md` | **开发日志，73 节，137 KB**。记的是"错过什么、为什么当时没发现" |
| `plan.md` | 2026-10-08 的计划。看得出当时的优先级，以及哪些猜对了 |
| `1.md.bak` | git 接管之前的备份。纯粹纪念，想删就删 |
| `README.md` | 说明这一目录为什么留着 |

`ir.md` 里值得单独找出来的几条：`eps` 写死同一个 bug 出现**六次**；
三个后端"一致地错"而互拍全绿；一个并行化带出的 race 症状像"未定义行为"；
一次工具报了假绿（"45 项 0 失败"，实际静默跳过 11 项）。

---

## `drafts/` —— 还没进 `models/` 的草稿

| 文件 | 用途 |
|---|---|
| `deepseek-v4.1-flash.by1` | 卡在 `pattern-design.md` 上。20 层全是同一个 `CSA2`，只有 `mode` 在变，现有语法表达不了 |
| `qwen3.8-flash-next.by1` | 同上 |
| `README.md` | 说明为什么它们是草稿 |

---

## `gpu/` —— 只有显卡上能跑的那部分

| 文件 | 用途 |
|---|---|
| `README.md` | 怎么用、需要什么 |
| `run.sh` | 入口 |
| `requirements.txt` | 依赖（和本地不同：要 CUDA 版的 torch） |
| `by1kda.py` | KDA（Kimi Delta Attention）的验证。它在 `by1check` 里是独立种类，不是 GDN |

---

## `tools/remote/` —— 34 个脚本，驱动那台租来的 A800

`remote-launch.sh`（只启动）/ `remote-m.sh`（只量）/ `remote-xet.sh`（查走没走 Xet）
之类。**没有任何代码引用它们**，留着是因为它们记着那趟是怎么跑的。

那台机器是按小时租的，脚本里"只启动不量"和"只量不启动"是分开的 ——
合在一起会阻塞，而阻塞的时候机器在计费。

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
路径规则在 `by1refs.py`，仓库布局在 `by1paths.py`，版本号在 `by1ver.py`，
文件读写在 `by1io.py`。"顺手再写一遍"是这个仓库里反复出现过的那类 bug。

**二、能描述 ≠ 能算。**
三个后端对拍只能证明自洽。真正的判据在 `by1oracles.py` 那 10 个神谕里，
以及 `by1verify.py` 对着官方产物的对拍里。
