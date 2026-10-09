# by1

**一门描述大模型架构的语言 —— 而它的接口不是这门语言，是一份 IR 规格。**

```
by1 0.9.0 · by1-ir 1.0
```

---

## 数据从哪来

仓库里的文件分三种，**删掉任何一种都能重建**（这是设计目标，不是巧合）：

| | 从哪来 | 怎么重建 |
|---|---|---|
| `by1*.py` | 手写的代码 | —— 这就是源码本身 |
| `refs/*.json` | **从 HuggingFace / ModelScope 抓的模型元数据** | `python by1fetch.py`（种子是 `models.tsv`） |
| `*.by1` · `models.tsv` | **手写的知识** | 手写的；可交叉验证（`.by1` 头部的 `by1-repo` 和 `models.tsv` 互为参照） |
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
python by1verify.py clef.by1 refs/Cloudflare__clef.config.json --config \
                    --tensors refs/Cloudflare__clef.tensors.json --backend torch.module
```

**14 个真实模型**，config 逐字段 + 张量逐名字逐形状，判卷人是官方产物。
其中 **9 个还对着外部参考比过前向数值**（最好 0.000e+00，最差 1.058e-06）。

### ② 从产物**反推**一份草稿，不用先懂这门语言

```bash
python by1boot.py <config.json> <tensors.json> --run
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
python by1e2e.py
```

```
by1boot 的猜测版    1.678e-02  [不一致]
改对 gate + act     1.160e-07  [一致]
NumPy（只拿 IR）    1.308e-07  [一致]
C                   编译成功
```

### ④ 一条命令看整个项目还活着没有

```bash
python by1all.py
```

```
61 项，0 项失败，另有 1 项已知缺口
[全过]
```

---

## 想参与的人**不需要学 by1**

**IR 才是接口。** `.by1` 只是前端之一。

```bash
python by1ir.py --spec          # 生成规格（字段、必填、语义、闭集）
python by1ir.py --emit x.by1    # 出一份规范化 JSON
python by1ir.py --check x.json  # 校验
```

**写第四个后端**：读规格 → 读懂 JSON → 实现。
三个现有后端都能**只拿 IR 跑**，这条路由判卷人守着：

```bash
python by1irentry.py    # 三个后端从 IR 入口跑 + JSON 往返
python by1opdiff.py     # 逐算子比 NumPy 和 PyTorch
```

---

## 一句话说清楚它现在**不行**在哪

**描述得了 14 个模型，算得了 6 种机制。**

| | |
|---|---|
| **没有外部用户** | 工具是硬的，但**没有任何证据表明别人想要它** |
| **真正的目标后端（llama.cpp）没接** | 图结构和 `qwen3next.cpp` 逐行核对过，张量名双向 612=612，但**没有生成过一行 ggml 代码** |
| **覆盖率追不上描述** | KDA / SSM / 稀疏索引器 / mHC **只有契约，算不了** |
| **C 后端还不全** | 没有 Linear（KDA/GDN）、没有 MLA、没有 `kv_tie`/`head_gate`。**它会明确拒绝**，不会静默按别的算法算 |

---

## 装

```bash
pip install -e .           # 开发装
by1 --version
```

不需要 GPU。CPU 上跑得动 8 个缩小模型和 `tiny-gpt2`。

**核心不需要 torch**：读规格、写第四个后端、跑检查器和 C 后端 ——
只用标准库。`pip install -e '.[verify]'` 才拉对拍那一套。

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
> 所以**一个空包在这台机器上也装不上** —— 验过。
> `pyproject.toml` 本身是对的，只是在这台机器上没法证。
>
> 能用的分发方式是打包：`python by1pack.py`（不需要 pip）。

---

## 显卡

**这台开发机没有独立显卡**（Intel Iris Xe，`torch 2.13.0+cpu`），
所以**没在显卡上验过**。但有一件事在这里就能验，而且它正是显卡上
最先炸的那一类：

```bash
python by1dev.py
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

不是"越快越好"，是**哪几个真模型突然装得下**。算过（bf16 权重 +
logits + 激活，留 10% 余量）：

| 卡 | 装得下 | 多出来的是谁 |
|---|---|---|
| **T4 16 GB** | 11 个 | 8 个 shaped + `minimind-3` + `gpt2` + **`instella-3b`** |
| **A800 80 GB** | **16 个** | **`clef` 26.9B · `qwen38` 27.3B · `gemma-4-31b` 32.1B · `laguna-xs-2.1` 33.4B · `qwen36` 35.5B** |
| A100/H100 80 GB | 同上 | |
| — | — | `gpt-oss-120b` 要 **218 GB**，`step-3.7` 要 **337 GB** —— 单卡都装不下 |

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
规格         ir-spec.md      **从 by1ir.py 生成的**，不是手写的
            by1ir.py        schema + 校验 + JSON 往返
入口         by1boot.py      产物 -> IR（不用 .by1）
            by1check.py     .by1 -> 检查 + IR
后端         by1codegen.py   PyTorch      by1exec.py  NumPy      by1c.py  C
判卷人       by1all.py       一次跑完全部
            by1verify.py    对着官方产物验
            by1e2e.py · by1irentry.py · by1opdiff.py · by1bootir.py
            by1gate.py      取值门的可证伪对照
            by1raw.py · by1extdemo.py   逃生舱的两层
描述         *.by1           26 份，其中 14 份对真实 checkpoint 验过
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

深处的账在 [`ir.md`](ir.md)（1600+ 行）· 主张与现实的对照在 [`1.md`](1.md)。

**语言名称暂定 by1。**
