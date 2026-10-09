# by1

**一份用声明式语法描述大模型架构的实验。它的接口不是那门语法，是一份 IR 规格。**

```
by1 0.9.0 · by1-ir 1.0
```

---

## 先说清楚：这几乎全是 AI 写的

**代码、文档、提交信息 —— 一个人 + 一个 agent，但写的是那个 agent。**

跑在 **DeepSeek Harness** 里：读产物、改代码、跑检查、把失败写进提交信息，
也把"我这次错在哪"一节一节留在 [`history/ir.md`](history/ir.md) 里。
人做的是另外两件事 —— **定方向**、**判定什么算做完**，
包括反复说"不对，重来"。

**为什么要把这一条放在最前面：因为它应该让你更怀疑，而不是更信。**
一个 AI 写的东西，最容易出的错不是"不会写"，是**写得很像对的** ——
这个仓库里真发生过：三个后端"一致地错"、互拍全绿；一份工具报了假的绿
（"45 项 0 失败"，实际静默跳过了 11 项）。

**这个项目的全部办法，就是让判据不依赖作者：**

| | |
|---|---|
| `refs/` | 各模型发布方的**公开元数据**，不是这个项目产的 |
| `.by1` ↔ `models.tsv` | 互相交叉验证（`.by1` 头部的 `by1-repo` 指向 `models.tsv` 那一行） |
| 前向数值 | 对着**官方实现**比，不是自己和自己比 |
| 神谕 | 10 条**不依赖 by1 的**期望值 |

**所以"这是 AI 写的"这件事，不改变结论 —— 你自己跑一遍就知道。**
下面所有数字都是"验过多少"，不是"有多少人在用"。

---

## 还有多大

**一个人 + 一个 agent，几个月，没有外部用户，没被任何项目采用过。**

它不是一个语言产品，是一个**实验**：试试"把架构描述成声明，再让独立的
判卷人判它对不对"这条路走不走得通。目前走到"14 个模型描述得出来、
6 个机制算得出来"。

---

## 想解决什么

今天给 llama.cpp 加一个新架构，要在五个地方改代码、写上千行 C++ 建图函数、
在两处重复写张量命名。但主流模型翻来覆去用的是同一批机制 ——
GQA、MLA、滑动窗口、Mamba、Gated DeltaNet、MoE ——
**新的东西往往是这些机制怎么组合、按什么比例排列、各自需要什么状态。**

Qwen3-Next 的 48 层是一句话：

```
48 层 = 12 × (3 层 GatedDeltaNet + 1 层 GatedAttention)
```

by1 想把这句话写成声明 —— 而它对不对，**交给判卷人，不交给写的人**。

---

## 试过什么

**① 把一份已发布的 checkpoint 描述出来，对着官方产物核对**

```bash
python src/checks/by1verify.py models/clef.by1 refs/Cloudflare__clef.config.json \
    --config --tensors refs/Cloudflare__clef.tensors.json --backend torch.module
```

**14 个真实模型**：config 逐字段 + 张量逐名字逐形状，判卷人是官方产物。
其中 **6 个整模型还对着外部参考比过前向数值**（最好 0.000e+00，最差 1.058e-06）。

**② 从产物反推一份草稿，不用先懂这门语言**

```bash
python src/data/by1boot.py <config.json> <tensors.json> --run
```

推得出的：`pos_kind` · `norm_kind` · `norm_eps` · 每层的机制种类 · `out_dim` · `hidden`。
推不出的按规格填默认值，**并且报出来**。

**③ 端到端：真产物 → IR → 三个后端 → 对官方实现**

```bash
python src/checks/by1e2e.py
```

**④ 一条命令看还活着没有 —— 分六档**

```bash
python src/checks/by1all.py --tier 3    # 跑某一档
python src/checks/by1run.py --status    # 推送的节奏现在轮到哪一档
```

档的轴是**每加一档多要一样东西**：静态 → +numpy → +refs/ → **+torch** → +gcc/下载 → +显卡。
日常用 **2~4**；完整表在 [`FILES.md`](FILES.md)。

> **"多少项"跟着机器和档位走，别当成常数。** 缺什么、或者换一档，
> 项数就变 —— 所以每次跑都要重新读，而不是引用文档里的一个数。

**三种结论，不是两种**：`ok` 验过了 · `--` **这台机器上没验**（不算失败，但也不是通过）· `!!` 验了不对。

---

## 不行在哪

**描述得了 14 个模型，算得了 6 种机制。**

| | |
|---|---|
| **没有外部用户** | 工具是硬的，但**没有任何证据表明别人想要它** |
| **真正的目标后端（llama.cpp）没接** | 图结构和 `qwen3next.cpp` 逐行核对过，张量名双向 612=612，但**没有生成过一行 ggml 代码** |
| **覆盖率追不上描述** | KDA / SSM / 稀疏索引器 / mHC **只有契约，算不了** |
| **C 后端还不全** | 只拒三样：Attention 的 `kv_tie`/`head_gate`、gpt-oss 那种 FFN 激活、未知机制。**它会明确拒绝**，不会静默按别的算法算 |
| **反向传播没在真权重上验过** | `by1train.py` 用的是 PyTorch 的 autograd，而它只在缩小模型上跑过 |
| **数值分布不是真的** | 对拍时用的是随机权重，不是真权重的分布 |

---

## 装

```bash
pip install -e .
by1 --version
```

**核心不需要 torch**：读规格、写第四个后端、跑检查器 —— 只用标准库。
NumPy 后端和 C 后端要 numpy；对拍那一套（torch / transformers / safetensors）
走 `pip install -e '.[verify]'`。

**IR 才是接口，`.by1` 只是前端之一。** 想写第四个后端的人不需要先学 `.by1`：

```bash
python src/by1ir.py --spec                      # 生成规格（字段、必填、语义、闭集）
python src/by1ir.py --emit models/hello.by1     # 出一份规范化 JSON
```

---

## 显卡

这台开发机有一张 **AMD RX 5600 XT**（6 GB，RDNA1），而 `torch` 是 CPU 版 ——
三条路都查过，都用不上。所以 `by1gpu.py` 在这里是**跳过**，
每一轮报告里都印着「没有可用的 CUDA 设备」，而不是只活在文档的叙述里。

换一张 N 卡能多验什么、哪些结论是实测的哪些是读来的：**[`docs/which-gpu.md`](docs/which-gpu.md)**。

---

## 目录

```
src/            12 个底座模块（被 import 的）+ 5 个子目录
  checks/       判卷人 · 神谕 · 跑检查
  modelcheck/   一个模型一个判卷人
  lang/         语言本身
  data/         抓数据 / 推模型
  escape/       逃生舱
models/         26 份 .by1，其中 14 份对真实 checkpoint 验过
refs/           从 HF / ModelScope 抓的**公开元数据**（不是权重）
docs/ history/ drafts/ gpu/ tools/
```

**每个文件的用途在 [`FILES.md`](FILES.md)** —— 那份是**生成的**（`by1files.py --write`），
所以不会过期；没归组的文件会被报出来。

---

## 最贵的一课

> **能描述 ≠ 能算。**

`gpt2.by1` 曾经和 `llama-shaped.by1` 编译出**一模一样的 IR**，
而三个后端、`by1verify`、`by1all` **全说"过"** ——
因为张量契约查的是参数的名字和形状，**不是算了什么**。

**所以现在的规矩是：要么实现，要么拒绝，没有第三条路。**
这条规矩自己也有判卷人（`by1gate.py`：三个必须被拒的反例、三个必须通过的正例）。

**三个后端互拍只能证明自洽 —— 它们可以一致地错。**
真正的判据在 `by1oracles.py` 那 10 个神谕里（每个都有独立的真相源），
以及 `by1verify.py` 对着官方产物的对拍里。

---

深处的账在 [`history/ir.md`](history/ir.md) · 每个文件在 [`FILES.md`](FILES.md) ·
该租哪张卡在 [`docs/which-gpu.md`](docs/which-gpu.md) · 主张与现实的对照在 [`1.md`](1.md)

**语言名称暂定 by1。作者是 AI（见最上面那一节）。**
