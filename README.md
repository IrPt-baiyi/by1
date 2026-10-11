# by1

**用声明式语法描述大模型架构，并由机器验证描述是否正确。**
**A declarative language for describing LLM architectures, with machine verification.**

```
by1 0.9.0 · by1-ir 1.0
```

[English](#english) · [中文](#中文) · [License](#license)

---

## English

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
| **53** `.by1` files in `models/` | declarative architecture descriptions |
| **15** real models verified | config field-by-field, tensor name-by-name and shape-by-shape, against the publisher's own artifacts |
| **6** mechanisms computable | `Attention` · `FFN` · `KDA` · `MLA` · `MoE` · `SSM` |
| **3** independent backends | NumPy · PyTorch · generated C, cross-checked |
| **6** models, forward numerics | compared bitwise against official implementations |

**Run it.** Tier 1 needs no dependencies at all; the core is pure standard library.

```bash
python src/checks/by1all.py --tier 1     # no dependencies
pip install -e .                         # optional: the `by1` CLI
```

**What does not work.**

- **No external users. Zero.** No evidence anyone wants this.
- **llama.cpp is not connected.** Graph and tensor names were cross-checked
  against `qwen3next.cpp`, but not one line of ggml has been generated.
- **The criteria themselves are unaudited.** The `1e-4` threshold, the
  definition of "covered", the 15 chosen models — all written here.
  See [`docs/criteria-review.md`](docs/criteria-review.md).
- Vision towers are *described* but not *computed* — deliberately, for lack of
  an official implementation to compare against.

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
| `models/` 里 **53** 份 `.by1` | 声明式的架构描述 |
| **15** 个真实模型验过 | config 逐字段 · 张量逐名字逐形状，判卷人是发布方的官方产物 |
| **6** 种机制算得了 | `Attention` · `FFN` · `KDA` · `MLA` · `MoE` · `SSM` |
| **3** 个独立后端 | NumPy · PyTorch · 生成的 C，互相交叉验证 |
| **6** 个模型比过前向数值 | 对着官方实现逐位对拍 |

**跑一下。** 第一档一个依赖都不用装，核心是纯标准库。

```bash
python src/checks/by1all.py --tier 1     # 不要任何依赖
pip install -e .                         # 可选：装 `by1` 命令行
```

**还没做到的。**

- **没有外部用户。零。** 没有任何证据表明别人想要它。
- **llama.cpp 没接。** 图和张量名与 `qwen3next.cpp` 逐行核对过，
  但一行 ggml 代码都没生成过。
- **判据本身没人复核。** `1e-4` 那个阈值、"覆盖"的定义、选定的这 15 个模型
  —— 全是这里写的。见 [`docs/criteria-review.md`](docs/criteria-review.md)。
- 视觉塔**描述得住、算不了** —— 决定不做，因为本地找不到可比的官方实现。

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

> 数字为 2026-10-12 现量。**"多少项"随机器与档位变化，不是常量** ——
> 以你自己那次跑出来的为准，见 [`TESTING.md`](TESTING.md)。
