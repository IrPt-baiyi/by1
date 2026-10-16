# 复现指南：从零到绿的一条路

> **这份文档不是教程，是复核清单。**
> 判据是**命令的退出码与判定行**，不是我的叙述。每条命令后面写的是
> **你应该看到什么**；对不上就是我们的问题 —— 把输出贴回来，我们改。
>
> 本文里所有"你应该看到"的数字，都是 **2026-10-16 在本机重跑过一遍**的
> （提交 `3bd127b` 之后）。陈旧的话，那是本文的错。

## 〇、覆盖什么、不覆盖什么

| | |
|---|---|
| **覆盖** | 档 1（静态）· 档 3（结构 + 判据 + refs 契约）· 档 4（完整模型）· 两条**真权重**判据（双 loader 前向 · 增量 == 全量）· 产物线（要 llama.cpp） |
| **不覆盖** | 档 5 的显卡腿（要 CUDA）· 真尺寸大模型（要几十 GB 显存）· **Step-5 的合同**（没有判卷人，我们没做 —— [判断](step5-contract-plan.md)） |

## 一、环境

| 项 | 要求 | 为什么 |
|---|---|---|
| Python | **≥ 3.9** | CI 上真跑过 3.9 与 3.12 两版（`requires-python` 不是写着玩的） |
| 磁盘 | 约 2.5 GB（不含权重） | 第 ③ 步要 `pip install -e ".[full]"`，拉 torch + transformers |
| 内存 | 8 GB 够走到第 ④ 步 | 0.6B 与 130M 都在 **CPU** 上跑；**不需要显卡** |
| git | 任意版本 | 档 1 里的 `by1files` 要它（不是 git 检出时**如实跳过**，不报失败） |

## 二、第 ① 步：档 1（零依赖，约 5 秒）

```bash
git clone https://github.com/IrPt-baiyi/by1.git && cd by1
python3 -m venv --without-pip .venv && python3 -m pip --python .venv install -e ".[run]"
.venv/bin/python src/checks/by1all.py --tier 1
```

**你应该看到**（最后两行）：

```
  14 项，0 项失败，13 项跳过
  [全过（13 项没验）]
```

- 退出码 **0**。
- "13 项跳过"不是失败 —— 那是"这一档没验"，档 1 不做档 2 以后的事。
- 这一步**故意不装任何依赖**：它同时是"档 1 只要标准库"这句话的证明。

> `python3 -m venv` 在 `ensurepip` 那一步挂掉过（某些托管 Linux 上见过），
> 所以上面用的是 `--without-pip` 的写法。README 里也有这一段。

## 三、第 ② 步：档 3（只装 numpy，约 1 分钟）

```bash
.venv/bin/python -m pip install numpy
.venv/bin/python src/checks/by1all.py --tier 3
```

**你应该看到**：

```
  101 项，0 项失败，23 项跳过
  [全过（23 项没验）]
```

档是**累积**的：这一步把档 1、2、3 都跑了。其中包含 26 份 `.by1` 检查器、
**变异反例库**（每条规则配一个必红反例）、取值门、采样链，
以及 refs 语料的 config 逐字段与张量名/形状。

## 四、第 ③ 步：档 4 完整模型（要 torch + transformers，约 1 分钟）

```bash
.venv/bin/python -m pip install -e ".[full]"
.venv/bin/python src/checks/by1complete.py
```

**你应该看到**（最后一行）：

```
  [PASS] 完整模型：全部通过
```

它验的是：**7 个完整声明**各自与"独立写出的参考配置"逐位对拍（前向 + 反向）、
官方 config 逐字段、MTP 与视觉那条链、以及 **8 条必红反例**（错声明必须被拒）。

## 五、第 ④ 步：两条真权重判据（要下权重）

这一步才是这个项目的主判据：**同一份声明，两条独立的路，数值必须对上**。
两个模型都在 CPU 上跑，权重加起来不到 2 GB。

| 模型 | 权重 | 在哪 |
|---|---|---|
| Qwen3-0.6B | 1.40 GB | `Qwen/Qwen3-0.6B` |
| mamba-130m-hf | 0.48 GB | `state-spaces/mamba-130m-hf` |
| granite-4.0-h-350m（**混合栈**） | 0.64 GB | `ibm-granite/granite-4.0-h-350m` |

```bash
# A) 双 loader 前向逐位（Qwen3-0.6B）
.venv/bin/python src/modelcheck/by1shadow.py --smoke-refs \
    --by1 complete-Qwen3-0.6B.by1 --weights-dir /path/to/qwen3-0.6b

# B) 增量 == 全量（KV cache）
.venv/bin/python src/modelcheck/by1shadow.py --smoke-state \
    --by1 complete-Qwen3-0.6B.by1 --weights-dir /path/to/qwen3-0.6b

# C) 双 loader 前向逐位（mamba-130m）
.venv/bin/python src/modelcheck/by1shadow.py --smoke-refs \
    --by1 complete-mamba-130m-hf.by1 --weights-dir /path/to/mamba-130m-hf

# D) 增量 == 全量（**递归状态**，不是 KV cache）
.venv/bin/python src/modelcheck/by1shadow.py --smoke-state-ssm \
    --by1 complete-mamba-130m-hf.by1 --weights-dir /path/to/mamba-130m-hf

# E) 双 loader 前向逐位（granite 混合栈：28 个 Mamba2 层 + 4 个注意力层）
.venv/bin/python src/modelcheck/by1shadow.py --smoke-refs \
    --by1 complete-granite-4.0-h-350m.by1 --weights-dir /path/to/granite-4.0-h-350m

# F) 增量 == 全量（**混合状态**：两本账在同一个 cache 里）
.venv/bin/python src/modelcheck/by1shadow.py --smoke-state-hybrid \
    --by1 complete-granite-4.0-h-350m.by1 --weights-dir /path/to/granite-4.0-h-350m
```

**六条各自的实测值**（2026-10-16 重跑）：

| | 你应该看到 |
|---|---|
| A | `真权重 …：311 个真名` · `源二 e2e OK：双 loader 零 missing/unexpected，前向 max\|Δ\| = 0.000e+00` |
| B | `官方 loader：max\|Δ\| = 2.289e-05` · `by1 模型：max\|Δ\| = 2.289e-05` · `[PASS] 增量 == 全量` |
| C | `真权重 …：242 个真名` · `前向 max\|Δ\| = 0.000e+00` |
| D | `官方 loader：max\|Δ\| = 1.497e-03` · `by1 模型：max\|Δ\| = 1.497e-03` · `[PASS] 递归状态：增量 == 全量` |
| E | `真权重 …：370 个真名` · `前向 max\|Δ\| = 0.000e+00` |
| F | `cache 里的两本账：递归层 28 个 · 注意力层 4 个（DynamicCache）` · `官方 loader：max\|Δ\| = 9.918e-05` · `by1 模型：max\|Δ\| = 9.918e-05` · `[PASS] 混合状态：增量 == 全量` |

**D 的那个数字要单独说**：判据是 2e-3，而实测 **1.497e-03 —— 余量只有 1.3 倍**，
是[精度预算表](precision-budget.md)里最薄的一格。原因不是接线错：递归状态每步乘 A
再累加，16 步下来 fp32 的归约序差异会被放大；**两个 loader 给出同一个数**，
说明差来自"增量 vs 全量"两条路径本身。B 的余量是 87 倍，两者不可类比。
**F 也不能拿 D 的余量类推**：混合栈实测 9.918e-05（余量 20 倍），
比纯递归好 15 倍——"递归就一定更差"不是规律。

## 六、第 ⑤ 步（可选）：产物线 —— 让上游加载 by1 出的文件

这一步最重（要检出并编译 llama.cpp），但它是最硬的判据：
**未经修改的上游 `llama.cpp` 加载 by1 生成的 GGUF，logits 逐位相同。**
命令与出处见 [llama.cpp 基线](llama.cpp-baseline.md)；实测（mamba-130m，同一条提示词）：

```
  [PASS] logits 逐位相同（30604 个值）
```

## 七、对不上时先看哪儿

| 症状 | 先看 | 常见原因 |
|---|---|---|
| `ModuleNotFoundError` | 装的是哪个 extra | 档 3 要 numpy、档 4 要 `.[full]` |
| 找不到 `*.safetensors` | `--weights-dir` 是不是 HF 那一层 | 它要的是含 `config.json` 的那一层目录 |
| `by1files` 说"不是 git 检出" | 你是不是从**发布包**解出来的 | 那是**跳过**（30），不是失败；`FILES.md` 的一致性只在检出里成立 |
| 档 3 一路红 | 先跑 `--tier 1` | 档 3 是累积的，档 1 红了档 3 一定红 |
| 显卡相关的项"跳过" | 你有没有 CUDA | 那些项在 CPU 机器上**本来就该跳过** |

## 八、给第三方的邀请（可直接粘贴）

> by1 是一个"用声明描述 LLM 架构、并让机器验证"的项目。我想请人**独立跑一遍**，
> 不是为了让人相信，而是想确认"照文档敲能不能得到同样的数"。
>
> 环境：任何 Linux/macOS/Windows，Python ≥ 3.9，**不需要显卡**，8 GB 内存够，
> 磁盘约 3 GB（含两个小模型的权重）。全程大约 **15 分钟**，其中 10 分钟在下载。
>
> 步骤在这份指南里（**贴给外人时用仓库地址**，不要用相对链接）：
> <https://github.com/IrPt-baiyi/by1/blob/main/docs/repro-guide.md> ——
> 档 1（5 秒）→ 档 3（1 分钟）→ 完整模型（1 分钟）→ 四条真权重判据（几分钟）。
> 每一步都写了**应该看到什么**。
>
> 我会想知道的是：**哪一步的数字和文档对不上**，以及输出原文。
> 对上了也请说一声 —— "有人照做成功"这件事，比我自己说一百遍有用。

---

**这份文档自己的状态**：数字取自本机重跑（提交 `3bd127b` 之后），
不是从别处抄的。发现对不上，那是文档的错 —— 请把它当 bug 报。
