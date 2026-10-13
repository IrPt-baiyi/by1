# by1

**用声明式语法描述大模型架构，并由机器验证描述是否正确。**
**A declarative language for describing LLM architectures, with machine verification.**

```
by1 0.9.0 · by1-ir 1.0
```

[English quick start](#english) · [中文入门](#中文) · [License](#license)

Start with the CPU demo below: run a model, change one number, and see the result.
先跑下面的 CPU 小模型，再改一个数字，看到架构实际发生变化。

**Complete model definitions:** six examples now include actual parameter tying,
projection bias differences and, for Qwen3.8-27B, vision plus MTP.
They use the **Transformers backend**; see [complete models](docs/complete-models.md).
Real-checkpoint GPU results are documented [here](docs/gpu-validation-results.md).

**完整模型定义：** 已提供六份包含实际权重共享、投影 bias 差异的声明；
Qwen3.8-27B 包含视觉与 MTP。完整模型使用 **Transformers 后端**，
示例、用法及验证范围见 [完整模型说明](docs/complete-models.md)。
真实权重 GPU 对拍结果见 [GPU 验证记录](docs/gpu-validation-results.md)。

---

## English

**Start here: run a small model on your CPU, then change its architecture.**
This demo needs Python 3.9+ and NumPy. It uses random weights and requires no
GPU, model download, PyTorch, or API key. The output shows that the declared
architecture executes; it does not demonstrate trained text generation.

Download and extract this repository, or clone it:

```bash
git clone https://github.com/IrPt-baiyi/by1.git
cd by1
```

If you downloaded a ZIP, open a terminal in the extracted directory containing
`pyproject.toml`. All commands below run from that directory.

**1. Install and run.** Use the block for your operating system. The virtual
environment keeps these dependencies separate; activation is not required.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[run]"
.\.venv\Scripts\by1 check models/hello.by1
.\.venv\Scripts\by1 run models/hello.by1 --seq 8
```

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[run]"
.venv/bin/by1 check models/hello.by1
.venv/bin/by1 run models/hello.by1 --seq 8
```

The checker should report `0 错误 / 0 警告` (no errors or warnings).
The runner prints `4 层 · 1,049,728 个参数` (4 layers, 1,049,728 parameters)
and `跑通：输出 (1, 8, 256)` (forward pass completed). Current CLI reports
are in Chinese; the numbers above identify the expected result.

**2. Change one number.** Open [`models/hello.by1`](models/hello.by1) in a text
editor. This is the complete architecture:

```text
model Hello {
  hparams { d_model = 128, vocab = 256 }
  stack main {
    pattern = 4 * Attention(q = 4, kv = 2, head_dim = 32)
    main[:] >> FFN(hidden = 512)
  }
}
```

Change only `pattern = 4 * Attention(...)` to `pattern = 2 * Attention(...)`
and save. The FFN attachment follows the new layer count automatically.
Rerun the last two commands from your operating system's block.

| Result | Before: `4 * Attention(...)` | After: `2 * Attention(...)` |
|---|---:|---:|
| Decoder layers | 4 | 2 |
| Parameters allocated by the runner | 1,049,728 | 557,696 |
| Output shape | `(1, 8, 256)` | `(1, 8, 256)` |

You changed the model's depth and parameter count without editing model code.
The output shape stays the same because sequence length and vocabulary size
stay the same. Change `2` back to `4` to restore the example. Logit values are
from random weights and are not a quality comparison.

This path was checked on Windows with Python 3.12.10 and NumPy 2.5.4 in an
isolated environment without PyTorch or Transformers, including installation
and both layer counts. A first-time external user has not tried it yet.

**3. Where to go next.** `by1 build models/hello.by1 -o hello_model.py`
generates a trainable PyTorch model with `build()`. Use the same environment's
`by1` executable as above; executing that generated model additionally needs
PyTorch (`.[train]`). Published-model examples with parameter tying, vision,
and MTP use a separate [Transformers backend](docs/complete-models.md)
(`.[full]`). The CPU demo above runs on the native NumPy backend.

Built-in mechanisms can be called directly; explicit `mech` declarations
remain available for named variants. Defaults for this demo: context 256,
pre-RMSNorm (eps 1e-5), causal attention with RoPE (base 10000, interleaved
pairing), gated SiLU FFN, and an untied output head. Loading a published
checkpoint requires its tensor naming/export rules.

If installation fails, keep the complete error and your `python --version`
output. If a command reports missing NumPy, rerun installation with the
virtual environment's Python shown above. For a first trial, useful feedback
is simply which step worked or failed, your OS, and the complete error.

**The problem.** Adding one architecture to llama.cpp takes ~800 lines of
graph-building C++ and 15 edits across 3 files. But mainstream models keep
reusing the same mechanisms — GQA, MLA, sliding window, Mamba, Gated DeltaNet,
MoE. What is actually new is *how they are composed*. Qwen3-Next's 48 layers
are one line:

```
48 layers = 12 × (3 GatedDeltaNet + 1 GatedAttention)
```

That should not need a thousand lines of code.

**What works today.**

| | |
|---|---|
| **59** `.by1` files in `models/` | declarative architecture descriptions |
| **15** real models verified | config field-by-field, tensor name-by-name and shape-by-shape, against the publisher's own artifacts |
| **6** mechanisms computable | `Attention` · `FFN` · `KDA` · `MLA` · `MoE` · `SSM` |
| **4** independent backends | NumPy · PyTorch · generated C · **generated ggml**, cross-checked |
| **6** models, forward numerics | compared bitwise against official implementations |

**Verify the project.** Tier 1 needs no dependencies at all; the core is pure standard library.

```bash
python src/checks/by1all.py --tier 1     # no dependencies
pip install -e .                         # optional: the `by1` CLI
```

**What does not work.**

- **No external users. Zero.** No evidence anyone wants this.
- **llama.cpp is half-wired.** `by1ggml.py` now generates ggml code from IR:
  three models match the NumPy backend to **~1e-7**, and the real
  gemma-4-31B IR compiles. But **not one line has been generated *into
  llama.cpp itself*** (arch table / `build_graph`), and MoE / sink / YaRN
  have no ggml mapping yet. Plan: [`docs/llamacpp-plan.md`](docs/llamacpp-plan.md).
- **The criteria themselves are unaudited.** The `1e-4` threshold, the
  definition of "covered", the 15 chosen models — all written here.
  See [`docs/criteria-review.md`](docs/criteria-review.md).
- The native NumPy/C/ggml path does not execute vision towers or MTP.
  The complete Transformers backend executes these branches for Qwen3.8-27B;
  its validation and remaining limits are documented in [complete models](docs/complete-models.md).

**The most expensive lesson.**

> **Describable ≠ computable.**

`gpt2.by1` once compiled to byte-identical IR as `llama-shaped.by1`. Three
backends, the verifier and the whole check suite all said *pass* — because the
tensor contract checks names and shapes, not what is computed. The rule since:
**implement it or reject it, no third path.**

**How this was made.** Nearly all code, docs and commit messages were written by
an AI agent (DeepSeek Harness) in a few days. A human set direction and decided
what counted as done. Stated first to raise scrutiny, not to raise trust.

**Read next.** [`1.md`](1.md) — why · [`history/ir.md`](history/ir.md) — what
went wrong · [`TESTING.md`](TESTING.md) — how to verify it ·
[`FILES.md`](FILES.md) — every file.

---

## 中文

**从这里开始：用 CPU 跑一个小模型，再亲手改它的架构。**
需要 Python 3.9+，只安装 NumPy。这个示例使用随机权重，无需显卡、下载模型、
PyTorch 或 API key。跑通表示声明的架构确实能执行；生成有意义的文本需要训练。

下载并解压本仓库，或使用 Git：

```bash
git clone https://github.com/IrPt-baiyi/by1.git
cd by1
```

如果下载的是 ZIP，请在解压后包含 `pyproject.toml` 的目录打开终端。
下面的命令都在这个目录运行。

**1. 安装并运行。** 根据系统选一组命令。虚拟环境把依赖单独存放，无需激活。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[run]"
.\.venv\Scripts\by1 check models/hello.by1
.\.venv\Scripts\by1 run models/hello.by1 --seq 8
```

macOS / Linux：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[run]"
.venv/bin/by1 check models/hello.by1
.venv/bin/by1 run models/hello.by1 --seq 8
```

检查结果应包含 `0 错误 / 0 警告`。运行结果应包含：

```text
架构：4 层 · 1,049,728 个参数 · 随机权重
跑通：输出 (1, 8, 256)
```

**2. 只改一个数字。** 用文本编辑器打开
[`models/hello.by1`](models/hello.by1)。完整架构如下：

```text
model Hello {
  hparams { d_model = 128, vocab = 256 }
  stack main {
    pattern = 4 * Attention(q = 4, kv = 2, head_dim = 32)
    main[:] >> FFN(hidden = 512)
  }
}
```

把 `pattern = 4 * Attention(...)` 中的 `4` 改成 `2`，保存文件。
每层挂载的 FFN 会自动跟着层数变化。重新执行对应系统命令组里的最后两条命令。

| 运行结果 | 修改前：`4 * Attention(...)` | 修改后：`2 * Attention(...)` |
|---|---:|---:|
| 解码层数 | 4 | 2 |
| 执行器实际分配的参数量 | 1,049,728 | 557,696 |
| 输出形状 | `(1, 8, 256)` | `(1, 8, 256)` |

只改架构声明，就改变了模型深度和参数量。序列长度和词表没有改变，所以输出形状
保持一致。把 `2` 改回 `4` 即可恢复示例。随机权重产生的数值不用于比较模型质量。

这条流程已在 Windows、Python 3.12.10、NumPy 2.5.4 的隔离环境中实测，
包括首次安装和两种层数；环境没有 PyTorch 或 Transformers。尚未经过外部新用户试用。

**3. 接下来能做什么。** 使用同一虚拟环境中的 `by1` 执行
`by1 build models/hello.by1 -o hello_model.py`，可生成带 `build()` 的可训练
PyTorch 模型源码；执行生成模型还需安装 PyTorch（`.[train]`）。包含权重共享、
视觉与 MTP 的发布模型示例使用另一个 [Transformers 后端](docs/complete-models.md)
（`.[full]`）。上面的入门示例使用原生 NumPy 后端。

内置机制可直接调用；自定义名称和变体仍可用 `mech` 声明。这个示例默认上下文 256、
前置 RMSNorm（eps 1e-5）、因果注意力与 RoPE（base 10000、interleaved 配对）、
门控 SiLU FFN、独立输出头。加载已发布 checkpoint 还需对应的张量命名与导出规则。

如果安装失败，请保留完整报错和 `python --version` 的输出；如果提示缺少 NumPy，
请使用上面虚拟环境中的 Python 重新安装。第一次试用只需反馈：在哪一步走通或卡住、
使用的系统，以及完整报错。

**要解决的问题。** 今天给 llama.cpp 加一个新架构，要在 3 个文件里改 15 处、
写约 800 行建图 C++。而主流模型反复用的是同一批机制 —— GQA、MLA、滑动窗口、
Mamba、Gated DeltaNet、MoE。**真正新的只是它们怎么组合。**
Qwen3-Next 的 48 层是一句话：

```
48 层 = 12 × (3 层 GatedDeltaNet + 1 层 GatedAttention)
```

这句话不该需要上千行代码才能表达。

**现在能做什么。**

| | |
|---|---|
| `models/` 里 **59** 份 `.by1` | 声明式的架构描述 |
| **15** 个真实模型验过 | config 逐字段 · 张量逐名字逐形状，判卷人是发布方的官方产物 |
| **6** 种机制算得了 | `Attention` · `FFN` · `KDA` · `MLA` · `MoE` · `SSM` |
| **4** 个独立后端 | NumPy · PyTorch · 生成的 C · **生成的 ggml**，互相交叉验证 |
| **6** 个模型比过前向数值 | 对着官方实现逐位对拍 |

**验证项目。** 第一档一个依赖都不用装，核心是纯标准库。

```bash
python src/checks/by1all.py --tier 1     # 不要任何依赖
pip install -e .                         # 可选：装 `by1` 命令行
```

**还没做到的。**

- **没有外部用户。零。** 没有任何证据表明别人想要它。
- **llama.cpp 只接了一半。** `by1ggml.py` 已经能从 IR 生成 ggml 代码：
  三个模型与 NumPy 后端对拍到 **1e-7 量级**，真实 gemma-4-31B 的 IR
  编译通过。但**还没有一行生成进 llama.cpp 本体**（arch 表 /
  `build_graph`），MoE / sink / YaRN 也还没有 ggml 映射。
  计划见 [`docs/llamacpp-plan.md`](docs/llamacpp-plan.md)。
- **判据本身没人复核。** `1e-4` 那个阈值、"覆盖"的定义、选定的这 15 个模型
  —— 全是这里写的。见 [`docs/criteria-review.md`](docs/criteria-review.md)。
- 原生 NumPy/C/ggml 路径尚不执行视觉塔和 MTP。完整 Transformers 后端已支持
  Qwen3.8-27B 的这些分支；验证范围和剩余限制见 [完整模型说明](docs/complete-models.md)。

**最贵的一课。**

> **能描述 ≠ 能算。**

`gpt2.by1` 曾经和 `llama-shaped.by1` 编译出**一模一样的 IR**。三个后端、
`by1verify`、整套检查**全说"过"** —— 因为张量契约查的是名字和形状，
不是算了什么。此后确立的规矩：**要么实现，要么拒绝，没有第三条路。**

**这份仓库是怎么来的。** 代码、文档、提交信息几乎全部由一个 AI agent
（DeepSeek Harness）在几天内写出；人定方向、判定什么算做完。
**写在最前，是为了提高审慎程度，而非增加信任。**

**接着读。** [`1.md`](1.md) 为什么做 · [`history/ir.md`](history/ir.md) 撞上了什么 ·
[`TESTING.md`](TESTING.md) 怎么验它 · [`FILES.md`](FILES.md) 每个文件干嘛的。

---

## License

**GPL-3.0-or-later** —— 全文见 [`LICENSE`](LICENSE)，逐字未改。

`refs/` 里的模型元数据**不是本项目的作品**，版权属各自发布方，
各自适用各自的许可证 —— 详见 [`NOTICE`](NOTICE)。

---

> 本次新增的入门说明由 AI 生成并翻译。The new quick-start instructions were generated and translated by AI.

> 数字为 2026-10-12 现量。**"多少项"随机器与档位变化，不是常量** ——
> 以你自己那次跑出来的为准，见 [`TESTING.md`](TESTING.md)。
