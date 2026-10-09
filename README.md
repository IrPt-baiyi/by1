# by1

**一门描述大模型架构的语言 —— 而它的接口不是这门语言，是一份 IR 规格。**

```
by1 0.9.0 · by1-ir 1.0
```

---

## 这份仓库是怎么来的

**代码、文档、提交信息 —— 几乎全部是 AI 写的。**

写它的是一个 agent，跑在 **DeepSeek Harness** 里：读产物、改代码、
跑 `by1all`、把失败和错误写进提交信息，也把"我这次错在哪"一节一节
留在 `history/ir.md` 里。人做的是另外两件事 —— **定方向、判定什么算做完**，
包括反复说"不对，重来"。

**说清楚这个，是因为这个仓库的卖点是「可验证」，不是「谁写的」。**
判据不依赖作者：`refs/` 是各模型发布方的公开元数据，`.by1` 和
`models.tsv` 互相交叉验证，前向数值对着官方实现比。
**这些不因为作者是 AI 而变可信或不可信 —— 你自己跑一遍就知道。**

`history/ir.md` 里那些"我犯的错"是第一人称的，因为写它的就是那个 agent。

---

## 数据从哪来

仓库里的文件分三种，**删掉任何一种都能重建**（这是设计目标，不是巧合）：

| | 从哪来 | 怎么重建 |
|---|---|---|
| `by1*.py` | 手写的代码 | —— 这就是源码本身 |
| `refs/*.json` | **从 HuggingFace / ModelScope 抓的模型元数据** | `python src/by1fetch.py`（种子：`models.tsv` + `refs/SOURCES.tsv`） |
| `models/*.by1` · `models.tsv` | **手写的知识** | 手写的；可交叉验证（`.by1` 头部的 `by1-repo` 和 `models.tsv` 互为参照） |
| `models.md` · `ir-spec.md` · `VERSION` | 生成的 | `by1cmp.py --md` · `by1ir.py --spec` · `by1ver.py --write` |

`refs/` 里的东西**不是这个项目的作品** —— 它们是各个模型发布方的公开
元数据（配置 + 张量名/形状清单，**不是权重**），各自适用各自的许可证。
对应关系见 `models.tsv`。

**删数据测试**：有人真的拿走过 5%（按字节）的数据，
交给一个完全无记忆的 agent 重建。结果、以及它是怎么把
四种 JSON writer 的规则逐字节试出来的，记在 `docs/delete-test-1/`。

## 三十秒

今天给 llama.cpp 添加一个新架构，要在五个地方改代码、写上千行 C++ 建图函数、
在两处重复写张量命名。但主流模型翻来覆去用的是同一批机制 ——
GQA、MLA、滑动窗口、Mamba、Gated DeltaNet、MoE ——
**真正新的东西只是这些机制怎么组合、按什么比例排列、各自需要什么状态。**

Qwen3-Next 的 48 层本质上是一句话：

```
48 层 = 12 × (3 层 GatedDeltaNet + 1 层 GatedAttention)
```

by1 让它变成一句话，而且**这句话能被验证**。

---

## 现在能做什么

### ① 把一份已发布的 checkpoint 描述出来，并**证明**描述是对的

```bash
python src/by1verify.py models/clef.by1 refs/Cloudflare__clef.config.json --config \
                    --tensors refs/Cloudflare__clef.tensors.json --backend torch.module
```

**14 个真实模型**，config 逐字段 + 张量逐名字逐形状，判卷人是官方产物。
其中 **6 个整模型还对着外部参考比过前向数值**（最好 0.000e+00，最差 1.058e-06），
另有 2 个只验了一个机制（单层 MLA / 只验 rope）。

### ② 从产物**反推**一份草稿，不用先懂这门语言

```bash
python src/by1boot.py <config.json> <tensors.json> --run
```

```
推得出   pos_kind · norm_kind · norm_eps · 每层的机制种类 · out_dim · hidden
推不出   qk_norm · q_gate · sink · head_gate · routing · ...
         -> 按规格填默认值，**并且报出来**
```

> **它不是"帮你写"，是把"你必须懂 73 个属性"换成"你改到验过为止"：
> 草稿对不对不需要你判断 —— 判卷人判断。**

### ③ 端到端：真产物 → IR → 三个后端 → 对官方实现

```bash
python src/by1e2e.py
```

```
by1boot 的猜测版    1.678e-02  [不一致]
改对 gate + act     1.160e-07  [一致]
NumPy（只拿 IR）    1.308e-07  [一致]
C                   编译成功
```

### ④ 一条命令看整个项目还活着没有

```bash
python src/by1all.py            # 或 by1-all（装过的话）
```

```
71 项，0 项失败，1 项跳过，另有 1 项已知缺口
[全过（1 项没验）]
```

**这三种结论，不是两种。**

| | |
|---|---|
| `ok` | 验过了 |
| `--` 跳过 | **这台机器上没验** —— 不算失败，但也不是通过，**报告里单独列出来** |
| `!!` 失败 | 验了，不对 |

| 这台机器上缺什么 | 哪几项变成"跳过" |
|---|---|
| 显卡 | `by1gpu.py`（真的搬到 CUDA 上跑） |
| `gcc` | C 后端那 9 项 + `by1extdemo.py` |
| HF 缓存里的 `sshleifer/tiny-gpt2` | `by1e2e.py`（真产物 → IR → 三后端） |
| 加了 `--quick` | C 后端整段 |

**上面那个 `71 / 0 / 1` 是这台开发机的数**（装了 torch + transformers
加 WinLibs 的 gcc，没有显卡）。**数字跟着机器走** —— 所以这里把前提写出来，
而不是只写一个数：缺哪一样，那一项就是"没验"，而不是"通过"。

---

## 想参与的人**不需要学 by1**

**IR 才是接口。** `.by1` 只是前端之一。

```bash
python src/by1ir.py --spec          # 生成规格（字段、必填、语义、闭集）
python src/by1ir.py --emit models/hello.by1    # 出一份规范化 JSON
python src/by1ir.py --check x.json  # 校验
```

**写第四个后端**：读规格 → 读懂 JSON → 实现。
三个现有后端都能**只拿 IR 跑**，这条路由判卷人守着：

```bash
python src/by1irentry.py    # 三个后端从 IR 入口跑 + JSON 往返
python src/by1opdiff.py     # 逐算子比 NumPy 和 PyTorch
```

---

## 一句话说清楚它现在**不行**在哪

**描述得了 14 个模型，算得了 6 种机制。**

| | |
|---|---|
| **没有外部用户** | 工具是硬的，但**没有任何证据表明别人想要它** |
| **真正的目标后端（llama.cpp）没接** | 图结构和 `qwen3next.cpp` 逐行核对过，张量名双向 612=612，但**没有生成过一行 ggml 代码** |
| **覆盖率追不上描述** | KDA / SSM / 稀疏索引器 / mHC **只有契约，算不了** |
| **C 后端还不全** | 只拒三样：Attention 的 `kv_tie`/`head_gate`、gpt-oss 那种 FFN 激活、以及未知机制。**它会明确拒绝**，不会静默按别的算法算。（Linear / MLA 已经实现 —— 这句话以前是旧的。） |

---

## 装

```bash
pip install -e .           # 开发装
by1 --version
```

不需要 GPU。CPU 上跑得动 8 个缩小模型和 `tiny-gpt2`。

**核心不需要 torch**：读规格、写第四个后端、跑检查器 —— 只用标准库。
**但 NumPy 后端和 C 后端要 numpy**（`by1exec.py` / `by1c.py` 在模块级 import 它），
所以那一条路是「标准库 + numpy」。对拍那一套（torch / transformers / safetensors）
走 `pip install -e '.[verify]'`。

> **一个环境上的坑，不是本项目的。**
> 这台机器上 `pip install -e .` 会失败，报 `No module named 'kernels.lockfile'`。
> 原因是环境里那个 `kernels 0.17.0` 注册了一个坏掉的 entry point：
>
> ```
> egg_info.writers  kernels.lock -> kernels.lockfile:write_egg_lockfile
>                                    ^^^^^^^^^^^^^^^^ 这个模块不存在
> ```
>
> `setuptools` 的 `egg_info` 会加载**所有** `egg_info.writers`，
> 所以**一个空包在那台机器上也装不上** —— 验过。
> **但那只在全局解释器上。** 这个仓库自己的 venv（`~/.venvs/by1`）里
> 没有 `kernels`，所以 `pip install -e .` 在那里是好的 ——
> 实测 `Successfully installed by1-0.9.0`，`by1 --version` 能跑。
> （它以前跑不了：`pyproject.toml` 的 `py-modules` 漏了 39 个模块，
> 连 `by1paths` 都没列 —— 见 `history/ir.md` 第 71 节。）
>
> 不经过 pip 的分发方式是打包：`python src/by1pack.py`。

---

## 显卡

**这台开发机有一张独立显卡，但这条路用不上**：**AMD Radeon RX 5600 XT**
（6 GB，RDNA1 / gfx1010），而 `torch` 是 `2.14.1+cpu`。

**A 卡为什么不行** —— 三条都查过：

```
ROCm / HIP        Windows 上没有消费级 Radeon 的 PyTorch 轮子；
                  而且 ROCm 不支持 RDNA1（gfx1010 不在支持列表里）
torch-directml    **没有 Python 3.12 的发行版**（最后停在 3.8–3.11 /
                  torch 2.3.1，之后没再更新）—— pip 报 No matching distribution
onnxruntime-dml   在（1.24.4），但那要先**有一个 ONNX 后端**：
                  那是新写一个后端，不是把设备名换一行
```

所以 `by1gpu.py` 在这台机器上是**跳过**（它自己会说"没有可用的 CUDA 设备"），
**不是失败** —— 而"这一项没验"现在会印在每一轮报告里，
不再只活在文档的叙述里（见 `history/ir.md` 第 67 节）。

**换一张 N 卡就能验**，而且装得下谁是可以先算的：

```bash
python src/by1gpu.py --plan 10.8     # 12 GB 的卡
```

```
--plan 5.4     11 个 —— 都是小模型；**instella-3b 要 7.8 GB，装不下**
--plan 10.8    12 个 —— 多出来的正是 **instella-3b**（真实维度那个）
```

**（这一段原来写的是"这台开发机没有独立显卡（Intel Iris Xe）"——
对这台机器是假的：它有独显，只是 A 卡。理由比"没有卡"具体得多，
而具体的那部分才是有用的部分。）**

但有一件事**不需要任何显卡**就能验，而且它正是显卡上最先炸的那一类：

```bash
python src/by1dev.py
```

```
ok clef-tiny.by1          前向里没有裸的创建
ok gpt2-tiny.by1          前向里没有裸的创建
...
[PASS] 设备无关性 前向不依赖 CPU 默认
```

它把 `torch.zeros/ones/arange/tensor/…` 这些 **CPU 默认的工厂设成陷阱** ——
前向期间任何**不带 `device=`** 的创建都会被记下来 ✓
在显卡上，那就是 `Expected all tensors to be on the same device`。

**这条规则是可证伪的**：拆掉任何一处 `device=`，对应的模型当场变红 ✓
（第一次验的时候挑错了模型 —— `llama-shaped` 不走 MLA 那条路，
所以拆了也不红。**一个碰不到被测东西的测试不是测试。**）

**它证明什么，不证明什么：**

| | |
|---|---|
| ✓ 证 | 前向不依赖「张量默认建在 CPU 上」 |
| ✗ 不证 | 数值和显卡一致（浮点归约顺序不同，会有差异） |
| ✗ 不证 | 显存放得下 |
| ✗ 不证 | 快 |

**「设备无关」和「显卡上对」是两件事。** 后者要有卡才能做 ——
而这个脚本把前一件做完了，**它是后一件的必要条件**。

另外两个后端按构造上不了显卡：`by1exec` 是 NumPy、`by1c` 是 C ✓
**那不是缺陷 —— 它们是参照实现，不是运行时。**

真要在显卡上验，需要的是：一块 NVIDIA 卡 · `torch` 换成 cu 版 ·
然后把 `by1dev` 换成真跑、把 `by1diff` 对着 HF 的 CUDA 版对拍。

### 用哪张卡 —— **门槛在哪**

不是"越快越好"，是**哪几个真模型突然装得下**。这张表**是算出来的，不是手写的**：

```bash
python src/by1gpu.py --plan 16     # 假装有一张 16 GB 的卡
python src/by1gpu.py --plan 80
```

它按「bf16 权重 + 2 GB 余量」挑，**每一个没被选的都会带理由列出来**。

> 这一段以前是一张手写表，而它和代码对不上：`by1gpu` 里有一行
> `if len(ir['layers']) > 8: continue`，把那 5 个大模型**静默丢掉**了 ——
> 表上写着它们装得下，代码里它们连候选都不是。见 `history/ir.md` 第 70 节。

**实测（`--plan`）：**

| 卡 | 跑得了 | 谁 |
|---|---|---|
| **T4 16 GB** | **12 个** | 9 个合成/教学模型 + **`minimind-3` · `gpt2` · `instella-3b`（7.8 GB）** |
| **A800 80 GB** | **17 个** | 上面 12 个 + **`clef` 52.1 · `qwen38` 52.8 · `gemma-4-31b` 61.8 · `laguna-xs-2.1` 64.3 · `qwen36` 68.1 GB** |
| A100/H100 80 GB | 同上 | |
| — | — | `gpt-oss-120b` 要 **219.6 GB**、`step-3.7` 要 **368.9 GB** —— 单卡都装不下 |

**那 5 个多出来的，正是最该跑的**：它们的**契约早就验过了**
（config 逐字段 + 张量逐名逐形状 —— `laguna` 的 **30430 个张量全中**），
**却从来没跑过前向。**

所以 `by1gpu.py` **按显存选模型**，不写死阈值 ——
显存不够的会**列出来**，而不是静默跳过。

**T4 的 16 GB 装得下 `instella-3b`** —— 那是唯一一个有真实维度、
验过前向的模型（对官方 `0.000e+00`）。**T4 正好把它装进去。**

**A800 是 Ampere（算力 8.0）**，有 bf16 张量核 —— 而 **T4 是 Turing（7.5），没有**
（bf16 能存，算不快）。所以 T4 上要用 fp32 或 fp16。

**镜像**：约束不在 torch，在 `transformers`。

```
by1 自己用的 torch：rsqrt / tril / atan2 / outer / finfo …
                   **全是老 API，任何 torch >= 2.0 都行**
判卷人要           GptOssConfig / Qwen3NextConfig   -> >= 4.55
**5.x 改过因果掩码的行为**                            -> 必须钉住
  （HF 在 attention_mask=None 时不再自动做因果 ——
    这个坑在 MLA 和 GPT-2 上各踩过一次）
```

```bash
pip install transformers==5.15.1 safetensors     # 和本机一致
```

> **判卷人换了就不是同一个判卷人。**
> 版本不一致的话，改的可能是参考，不是 by1。

---

## 目录里有什么

```
规格         ir-spec.md       **从 src/by1ir.py 生成的**，不是手写的
            src/by1ir.py     schema + 校验 + JSON 往返
入口         src/by1boot.py   产物 -> IR（不用 .by1）
            src/by1check.py  .by1 -> 检查 + IR
后端         src/by1codegen.py PyTorch   src/by1exec.py NumPy   src/by1c.py C
判卷人       src/by1all.py    一次跑完全部（ok / 跳过 / 失败 三态）
            src/by1verify.py 对着官方产物验
            src/by1refs.py   refs 语料：能解析、有出处
            src/by1docs.py   .md 里写的路径还在吗
            src/by1e2e.py · src/by1irentry.py · src/by1opdiff.py
            src/by1bootir.py
            src/by1gate.py   取值门的可证伪对照
            src/by1raw.py · src/by1extdemo.py   逃生舱的两层
            src/by1skip.py   "这次没验"是第三种结论（协议在这里）
描述         models/*.by1     26 份，其中 14 份对真实 checkpoint 验过
                             而且**14 份都在上面那个清单里**
```

---

## 逃生舱：两层

语言表达不了的机制，可以下探去写 —— **但约束不松**。

```
第一层  Raw       写在 raw.py 里          **要改编译器**
第二层  External  IR 引用外部符号         **不用改编译器** —— 只要一个 .so
                  (.so + 固定 ABI)        和一份 IR
```

第二层是接着 llama.cpp 最近的一步：**一个 ggml 后端就是一个 `.so`** ——
它不需要编译器先认识 Mamba，**可以一个一个机制地长出来**。

---

## 最贵的一课

> **能描述 ≠ 能算。**

`gpt2.by1` 曾经和 `llama-shaped.by1` 编译出**一模一样的 IR**，
而三个后端、`by1verify`、`by1all` **全说"过"**。
因为张量契约查的是参数的名字和形状，**不是算了什么**。

**所以现在的规矩是：要么实现，要么拒绝，没有第三条路。**

这条规矩自己也有判卷人（`by1gate.py` —— 三个必须被拒的反例、
三个必须通过的正例）。**一条只会通过的规则不是规则。**

---

深处的账在 [`history/ir.md`](history/ir.md)（3200+ 行）· 主张与现实的对照在 [`1.md`](1.md)。

**语言名称暂定 by1。**
