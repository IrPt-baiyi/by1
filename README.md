> **本项目由 AI 生成。** / **This project is AI-generated.**

# by1

**用声明式语法描述大模型架构，并由机器验证描述是否正确。**
**A declarative language for describing LLM architectures, with machine verification.**

```
by1 1.0.0-rc4 · by1-ir 1.4
```

[English](#english) · [中文](#中文) · [License](#license)

---

## English

**Start here: run a small model on your CPU, then change its architecture.**
Python 3.9+ and NumPy only — no GPU, no download, no PyTorch, no API key. Random
weights: passing proves the declared architecture executes, not that it writes text.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[run]"
.\.venv\Scripts\by1 check models/hello.by1
.\.venv\Scripts\by1 run models/hello.by1 --seq 8
```

macOS / Linux: use `.venv/bin/python -m` and `.venv/bin/by1` instead.
Expected: `0 错误 / 0 警告`, `4 层 · 1,049,728 个参数`, `跑通：输出 (1, 8, 256)`.
Then open [`models/hello.by1`](models/hello.by1) and change `4 * Attention(...)`
to `2 * Attention(...)`: 4 layers / 1,049,728 params become 2 layers / 557,696,
with no model code touched. (A hosted-Linux `ensurepip` failure happens before by1
is installed; see the external reproduction in [history/external-repro/](history/external-repro/).)

**The problem.** Supporting a new model still means touching converters, tensor
loading, graph construction and hardware paths. by1 describes the architecture
**and the checkpoint contract** once, then validates both against the publisher's
own artifacts and established runtimes. It is not another inference engine:
deployment stays upstream-first.

**What works today.**

| | |
|---|---|
| **71** `.by1` files in `models/` | declarative architecture descriptions |
| **31** real models verified | config field-by-field, tensor name-by-name and shape-by-shape, against the publisher's own artifacts |
| **6** mechanisms computable | `Attention` · `FFN` · `KDA` · `MLA` · `MoE` · `SSM` |
| **4** independent backends | NumPy · PyTorch · generated C · generated ggml, cross-checked |
| **5** models, forward numerics | same weights and inputs against the official implementation (1e-3 family) |

**2026-10-19/20:** `by1-ir` moved 1.2 → 1.4 — multibranch `stacks`/`wiring`/`shared`,
vision entry/exit operators, and `causal`/`rope_pos` execution semantics.
`complete-Qwen3-0.6B` runs IR → exec backend and matches the official transformers
forward at 4.301e-06 (tier 5); browser graphs show real IR branch structure.

**What does not work.**

- **External reproduction: 1. Standing users: 0.** One first-time user reproduced the
  CPU quickstart on Colab (2026-10-16) and found a real gap (report in
  [history/external-repro/](history/external-repro/)). No one has used by1 for real work yet.
- **No complete native llama.cpp adapter.** Deployment uses upstream features and
  standard GGUF; generated ggml stays a validation backend. See
  [llamacpp strategy](docs/llamacpp-plan.md).
- **The criteria themselves are unaudited.** The thresholds, the definition of
  "covered", the chosen models — all written here. See
  [criteria review](docs/criteria-review.md).
- **The native NumPy/C/ggml path does not execute vision towers or MTP.** The complete
  Transformers backend does, for Qwen3.8-27B; limits are in
  [complete models](docs/complete-models.md).

**The most expensive lesson.**

> **Describable ≠ computable.**

`gpt2.by1` once compiled to byte-identical IR as `llama-shaped.by1`, and every check
said *pass* — the contract verifies names and shapes, not what is computed. The rule
since: **implement it or reject it, no third path.**

**How this was made.** Nearly all code, docs and commit messages were written by an AI
agent (DeepSeek Harness) in a few days. A human set direction and decided what counted
as done. Stated first to raise scrutiny, not to raise trust.

**Verify it.** `python src/by1/checks/all.py --tier 1` needs no dependencies at all.
Then read [TESTING.md](TESTING.md) (five-state verdicts, tier semantics),
[FILES.md](FILES.md) (what every file is), [1.md](1.md) (why), and
[history/ir.md](history/ir.md) (what went wrong).

---

## 中文

**从这里开始：用 CPU 跑一个小模型，再亲手改它的架构。**
只需 Python 3.9+ 和 NumPy —— 无需显卡、无需下载模型、无需 PyTorch、无需 API key。
随机权重：跑通证明声明的架构能执行，不证明它会写文章。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[run]"
.\.venv\Scripts\by1 check models/hello.by1
.\.venv\Scripts\by1 run models/hello.by1 --seq 8
```

macOS / Linux 把可执行文件换成 `.venv/bin/python -m` 与 `.venv/bin/by1`。
应看到 `0 错误 / 0 警告`、`4 层 · 1,049,728 个参数`、`跑通：输出 (1, 8, 256)`。
然后打开 [`models/hello.by1`](models/hello.by1)，把 `4 * Attention(...)` 改成
`2 * Attention(...)`：4 层 / 1,049,728 个参数 变成 2 层 / 557,696，不用碰任何模型代码。
（托管 Linux 上 `ensurepip` 的失败发生在安装 by1 **之前**；外部复现见
[history/external-repro/](history/external-repro/)。）

**要解决的问题。** 现在支持一个新模型仍要动转换器、张量加载、建图和硬件路径。
by1 把架构**和 checkpoint 合同**描述一次，再对发布方官方产物与成熟运行时做验证。
它不是又一个推理引擎：部署走上游优先。

**现在能做什么。**

| | |
|---|---|
| `models/` 里 **71** 份 `.by1` | 声明式的架构描述 |
| **31** 个真实模型验过 | config 逐字段 · 张量逐名字逐形状，判卷人是发布方的官方产物 |
| **6** 种机制算得了 | `Attention` · `FFN` · `KDA` · `MLA` · `MoE` · `SSM` |
| **4** 个独立后端 | NumPy · PyTorch · 生成的 C · 生成的 ggml，互相交叉验证 |
| **5** 个模型比过前向数值 | 与官方实现同权重同输入对拍（1e-3 族） |

**2026-10-19/20：** `by1-ir` 从 1.2 走到 1.4 —— 多分支 `stacks`/`wiring`/`shared`、
视觉入口/出口算子、`causal`/`rope_pos` 执行语义。`complete-Qwen3-0.6B` 走通
IR → exec 后端，与 transformers 官方前向 4.301e-06（档 5）；浏览器图换成真 IR 分支结构。

**还没做到的。**

- **外部复现 1 次，持续用户 0。** 一位首次使用者在 Colab（2026-10-16）复现了 CPU
  入门流程并抓到一处真实缺口（报告见 [history/external-repro/](history/external-repro/)）。
  还没有人把 by1 用于真实工作。
- **没有完整的原生 llama.cpp 适配器。** 部署用上游能力与标准 GGUF；生成的 ggml 只做
  验证后端。见[集成策略](docs/llamacpp-plan.md)。
- **判据本身没人复核。** 阈值、"覆盖"的定义、选定的模型 —— 全是这里写的。见
  [判据复核](docs/criteria-review.md)。
- **原生 NumPy/C/ggml 路径不执行视觉塔和 MTP。** 完整 Transformers 后端支持
  Qwen3.8-27B 的这些分支；范围与限制见[完整模型说明](docs/complete-models.md)。

**最贵的一课。**

> **能描述 ≠ 能算。**

`gpt2.by1` 曾经和 `llama-shaped.by1` 编译出一模一样的 IR，而所有检查都说"过" ——
因为契约查的是名字和形状，不是算了什么。此后立的规矩：**要么实现，要么拒绝，没有第三条路。**

**这份仓库是怎么来的。** 代码、文档、提交信息几乎全部由一个 AI agent（DeepSeek Harness）
在几天内写出；人定方向、判定什么算做完。**写在最前，是为了提高审慎程度，而非增加信任。**

**怎么验它。** `python src/by1/checks/all.py --tier 1` 一个依赖都不用装。然后看
[TESTING.md](TESTING.md)（五态判定与档位）、[FILES.md](FILES.md)（每个文件干嘛的）、
[1.md](1.md)（为什么做）、[history/ir.md](history/ir.md)（撞上了什么）。

---

## 关于这个项目 / About

**中文**：我是十年级学生，正在准备中国的高考 —— 它占去了我大部分时间。这是项目推进的
主要约束，不是对项目缺乏兴趣。尽管如此，我会继续做下去。

**English**：I am in Grade 10 and preparing for the National College Entrance Examination
(gaokao) in China — it takes most of my time. That is the main constraint, not a lack of
interest. Nevertheless, I will continue to work on this project.

## License

**GPL-3.0-or-later** —— 全文见 [`LICENSE`](LICENSE)，逐字未改。

`refs/` 里的模型元数据**不是本项目的作品**，版权属各自发布方 —— 详见 [`NOTICE`](NOTICE)。

---

> 数字为 2026-10-20 现量。**"多少项"随机器与档位变化，不是常量** —— 以你自己那次跑出来的为准，见 [`TESTING.md`](TESTING.md)。
