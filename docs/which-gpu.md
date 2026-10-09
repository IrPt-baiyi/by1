# 该租哪张卡

> 从 `README.md` 挪出来的。它是一次**具体的调查** —— 这台开发机上
> 的 AMD RX 5600 XT 为什么用不上、换一张 N 卡能多验什么、
> 哪些结论是实测的哪些是读来的。值得留，但不该占着门面。

## 显卡

**这台开发机有一张独立显卡，但这条路用不上**：**AMD Radeon RX 5600 XT**
（6 GB，RDNA1 / gfx1010），而 `torch` 是 `2.14.1+cpu`。

**A 卡为什么不行** —— 三条都查过：

```
ROCm / HIP        官方 ROCm 的 PyTorch 轮子只有 Linux。AMD 现在有
                  "PyTorch on Windows Edition"，但公开的 Windows ROCm
                  补丁针对的是 RDNA2（gfx1030-1036）；**RDNA1（gfx1010）
                  不在官方支持列表里**。社区有非官方的 RDNA1 ROCm 构建
                  （TheTrustedComputer/ROCm-RDNA1），但那是 Linux。
torch-directml    **本机实测**：pip 报 No matching distribution ——
                  Python 3.12 没有发行版（最后停在 3.8-3.11 / torch 2.3.1）
onnxruntime-dml   在（1.24.4），但那要先**有一个 ONNX 后端**：
                  那是新写一个后端，不是把设备名换一行
```

（只有 `torch-directml` 那条是本机实测的；前两条来自公开资料 ——
这个开发沙箱取不到 `amd.com`，所以标清楚哪条是测的、哪条是读的。）

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
**它们本来就是参照实现，不是运行时** —— 所以这条路不通，
不是"以后会补上"，是"它不该走这条路"。

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
