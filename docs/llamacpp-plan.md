# 接 llama.cpp：完整计划（执行版）

> 写于 2026-10-12。**这是执行计划**；立论和测量在 [`ggml-plan.md`](ggml-plan.md)，
> 那份是量出来的（`by1emit` 的三样产物、GGUF manifest 的三个判卷人、
> gcc 在场），本篇不重复。
>
> 排法沿用 [`../history/plan.md`](../history/plan.md) 的铁律：
> **每一个做成的块都有判卷人。写不出判卷人的，不排进计划。**

---

## 一句话：要做什么

```
.by1  →  IR  →  by1ggml.py  →  ggml C  →  gcc + libggml  →  跑起来
                                                        ↘ 和 NumPy 后端逐位对拍
```

让 README 里那句 **「一行 ggml 都没生成过」** 成为历史，并且
**生成的那一行是有判卷人的** —— ggml 是第二个独立实现，
两个独立实现同时错在同一个地方的概率比一个低。

---

## 现状（都是实测，不是设想）

| | |
|---|---|
| `by1emit.py` | 算子→原语表 · 声明式图 · GGUF manifest 双向对拍（gemma-4-31B **783=783**）|
| `by1c.py` | 第三个后端（手写循环的 C），harness：`w.bin/ids.bin/logits.bin`，阈值 1e-4 相对 |
| `by1exec.py` | NumPy 后端 —— 已经和 PyTorch、HF 对过，**现成的判卷人** |
| 工具链 | gcc 16.2.0 · g++ · cmake 4.4.2 · ninja —— 全在 |
| ggml 本体 | `llamacpp/llama.cpp/`（master，可重下）· 构建产物 `llamacpp/build-ggml/` |
| 没有的 | ~~一行 ggml 代码都没生成过~~ **2026-10-12 已成历史** —— 见文末执行记录 |

三个 GGUF 判卷人（`by1emit` 已 [PASS]，不许退）：

| 模型 | 算子种类 | 原语 | manifest |
|---|---|---|---|
| **gemma-4-31B** | 4（Add/Attention/FFN/Norm） | 12 | 783=783 |
| gpt-oss-120b | 4 | 16 | ✓ |
| Laguna-XS-2_1 | 5 | 18 | ✓ |

---

## M0 · ggml 本体落地

**做什么**：llama.cpp 源码进 `llamacpp/llama.cpp/`（`llamacpp/` 本来就是
"可重新下载"的参照区，不进 git）；cmake 构建 CPU 版静态库到
`llamacpp/build-ggml/`；一个十行的 ggml 自检程序。

**判卷人**：编译器 + 手算。自检跑一个已知小矩阵的 `ggml_mul_mat`，
和手算逐位比 —— **判的是"ggml 在这台机器上算的是这个数学"**，
不是"编译过了"。

**验收**：`libggml*.a` 存在；自检 [PASS]。

**不做**：CUDA / Vulkan / SYCL / 任何加速后端。**判数学用 CPU f32 就够**，
加速后端只会引入对拍以外的变量。

---

## M1 · `by1ggml.py` —— 第一行 ggml

**做什么**：`src/by1ggml.py`，只吃 IR（和 `by1c.emit_c_ir` 同一个入口），
生成一份调 ggml 原语的 `model.c`，gcc 链接 libggml 编出来，跑，
出 `logits.bin`。

**判卷人**：
1. **gcc** —— 编得过才算生成了代码；
2. **by1c 冻结的 harness 协议** —— `w.bin` 的扁平布局一字不改
   （它就是 ggml 的列主序视角：by1 的 `[n, d]` 行主序 ≡ ggml 的 `[d, n]`，
   零转换直接 `ggml_set_data`）。**同一个 `w.bin` 要能喂两个后端** ——
   这是"张量绑定一致"的判据，不是方便。

**验收**：`python src/by1ggml.py models/llama-shaped.by1` 生成 + 编译 + 运行。

**不做**：算子集只做 Norm / Add / FFN / Attention 的**已实现属性**。
用不了的属性**报错拒绝**（by1c 的规矩原样搬：要么实现，要么拒绝，
不许悄悄生成一个算错的模型）。

---

## M2 · 数值判卷：ggml ↔ NumPy

**这一步完成，「llama.cpp 没接」才变成假话。**

**判卷人**：`by1exec`（NumPy 后端）。它是现成的、已经被 PyTorch 和 HF
判过的实现 —— **ggml 只需要追平它**。

**验收三条**：

| # | 对象 | 判的是什么 |
|---|---|---|
| ① | `llama-shaped` | 基础四算子 · pairing=half · 相对差 < 1e-4 |
| ② | gemma 机制孪生 | **qk_norm=per_head + pairing=interleaved + window** 的组合 |
| ③ | 真实 `gemma-4-31B.by1` | 生成的 C **编译过** + manifest 783=783 不退 |

**为什么要孪生**：真实 gemma-4-31B 的 vocab 262144、87 层，
随机权重全模型是 **121.7 GB f32** —— 数值对拍跑不动。孪生**只缩维度**，
机制组合逐项相同（`qk_norm=per_head`、`rope pairing=interleaved`、
`window`、eps=1e-6、silu 门控 FFN）。"31B 不等于要跑 31B" ——
判的是算得对不对，不是装不装得下。

**已知的坑（写在前面，别踩第二遍）**：

1. **rope 的 pairing 在两边不是同一个约定。** by1 的 `half` =
   ggml 的 NEOX 模式，直接对应。by1 的 `interleaved` = **奇偶进、拼接着出**，
   而 ggml 的 NORM 模式是**奇偶进、奇偶出** —— 输入配对相同、
   输出布局差一个 q/k **共同**置换，点积不变。
   **这条必须靠 ② 的数值对拍来证，不许当成假设**（见 history：
   "变量名相同不代表含义相同"，rope 是重灾区）。
2. **eps 不许写死。** 这个 bug 仓库里出过六次。gemma 是 1e-6，llama 是 1e-5。
3. **qk_norm 的 `off` 是真值字符串。** 直接布尔判断会永远为真
   （by1c 的 `_qk_on` 就是为此存在，照搬）。
4. **window 的边界**：by1 的规则是 `j <= i - window` 丢掉（保留窗口
   内 `window` 个 key），掩码要逐位一致，别用"约等于滑窗"的近似。
5. **ggml 的 rope 是 fp32 算的，by1 的参照是 fp64。** 误差会从 1e-7 量级
   涨到 1e-5 量级 —— 这正是阈值 1e-4 存在的意义，但**要盯着相对差的
   实际值**，不许只看 PASS。

---

## M3 · 补齐三个 GGUF 判卷人的算子集

**做什么**：
- **gpt-oss-120b**：MoE（两种 routing / score 函数 / expert_bias /
  共享专家 / shared_gate）+ `sink` + YaRN + `gptoss` 激活 + 交替滑窗
- **Laguna-XS-2_1**：第 5 种算子 + `exp_bias` 那 39 个命名缺口
- **MLA / Raw**：给出映射或者**显式拒绝**。拒绝是合法交付 ——
  仓库的规矩是「要么实现，要么拒绝，没有第三条路」，
  `by1emit` 的 `GGML.get(k, ['?'])` 已经保证缺口看得见。

**判卷人**：各自的 GGUF manifest 双向对拍（不许退）+ NumPy 孪生对拍。
MoE 逐专家那段的对照是 `by1moe.py` 的现成结论。

**不做**：MoE 的融合 kernel / 性能。**逐专家 `ggml_mul_mat` 慢就慢** ——
这一阶段判的是数学。

---

## M4 · 接进 llama.cpp 本体 —— 「800 行」变生成物

**这是 README 那句话的终点**：给 llama.cpp 加一个架构不再手写
823 行建图 + 15 处编辑，而是从 `.by1` 生成。

**做什么**：从 IR 生成 llama.cpp 的接入面 ——
`llama-arch` 的枚举与名字表 · `llama-model` 的加载与超参 ·
`models/<arch>.cpp` 的 `build_graph`。**判物是 `qwen3next.cpp`**
（图结构、张量绑定、状态形状与它逐行核对，这件事 3a 做过一半）。

**判卷人**：
1. **llama.cpp 自己编译过**（生成的代码是它的一部分，编不过是零分）；
2. **有 GGUF 的模型：`llama-cli` 的输出 logits vs by1** ——
   真权重、真格式、真运行时。这是全链条唯一的"端到端"判卷人。

**验收**：演示模型 = llama.cpp **还不支持**的架构（`Instella-3B.by1`
的材料早备好了），生成 → 编译 → 跑 GGUF → logits 对上。

**不做**：量化 kernel、CUDA、批处理、KV cache 复用的性能。prefill + 逐位。

---

## M5 · 门禁与叙事

**做什么**：ggml 对拍进 `by1all` 的 gcc 档；`TESTING.md` 的档位表更新；
README 的「llama.cpp 没接」改成事实；`FILES.md` 重生成。

**判卷人**：`by1all` 全档 + `by1files` 的「覆盖不降」硬判据。
**叙事跟着测量走** —— 做到哪写到哪，不许提前。

---

## 一张图

```
M0  ggml 本体          ──  判卷人：手算的矩阵乘
 │
M1  by1ggml.py 第一行  ──  判卷人：gcc + by1c 冻结的 w.bin 协议
 │
M2  数值对拍           ──  判卷人：by1exec（NumPy，已与 PyTorch/HF 对过）
 │                        ① llama-shaped ② gemma 孪生 ③ 31B 编译 + manifest
M3  算子补齐           ──  判卷人：三份 GGUF manifest + NumPy 孪生
 │
M4  接 llama.cpp 本体  ──  判卷人：llama.cpp 编译 + llama-cli 逐位
 │
M5  门禁与叙事         ──  判卷人：by1all / by1files
```

**M0–M2 是一条链，做完它，「llama.cpp 没接」这句话就死了。**
M3–M4 是量变到质变：M4 做完，README 的立论（"800 行"变生成物）
才有证据。M5 是纪律：**没有门禁的完成会过期**，这个仓库已经吃过
"写进行文里的前提比行文先死"的亏。

---

## 明确不做的（这一整个计划期间）

| | 为什么 |
|---|---|
| GPU / 量化运行时 | 判数学用 CPU f32。加速和量化是另一条线，混进来只会让对拍的变量变多。 |
| 性能优化 | 慢就慢。逐专家循环、无融合 —— 都允许。判的是算得对不对。 |
| 多栈 / 跨栈投影（DeepSeek） | **依旧没有判卷人**（history/plan.md 的结论没变），不排。 |
| 新检查器 / 新工具 | 已经 20+ 个。`by1ggml.py` 是**第四个后端**，不是检查器 —— 它进 `by1diff` 的对拍矩阵，不另立山头。 |

---

## 执行记录（2026-10-12，M0–M2 完成）

| 里程碑 | 状态 | 判卷人的裁决 |
|---|---|---|
| M0 ggml 本体 | ✅ | `tools/ggml-smoke.c`：mul_mat / rms_norm / silu 对**手算**，[PASS] |
| M1 `by1ggml.py` | ✅ | gcc 编过；`w.bin` 布局复用 `by1c.emit_c`（同一份权重喂两个后端） |
| M2 数值判卷 | ✅ | 见下表，全部 < 1e-4，实测全在 **1e-7 量级** |
| M5（部分） | ✅ | `by1all` 档 5 新增 `ggml 后端` 段 —— ggml 不在 = 跳过（30），不算失败 |

**M2 三条判据的实测值**（判卷人：by1exec NumPy，阈值 1e-4）：

| # | 对象 | 相对差 | 备注 |
|---|---|---|---|
| ① | `llama-shaped` | **2.962e-07** | pairing=half（= ggml NEOX，逐位对应） |
| ①b | `gpt2-tiny` | **1.379e-07** | layernorm / 学习位置 / gelu_new / 无门控 |
| ② | gemma 孪生 | **1.390e-07** | qk_norm=per_head + interleaved + window + kv_tie + partial rope |
| ③ | 真实 `gemma-4-31B` | 编译过 | 5914 行 C · 950 张量 · manifest **783=783** 不退 |
| 拒绝 | `qwen3-next-shaped` | [PASS] | Linear/MoE/q_gate 落在支持集外，明确 `[不支持]` |

**顺带修掉的三个 bug（都有判卷人）：**

1. **最终 norm/head 发射在层之前** —— ggml 是惰性图，张量在**构建那一刻**
   把输入钉死：norm/head 吃到的是 embed 输出，层算了等于没算
   （定位：和"零层模型"逐位相关 0.99999）。**C 的语句顺序就是图的结构。**
2. **三个 gelu 同名不同义，且 ggml 的 GELU 不能直接用。**
   `gelu_new` 先被映到 `GELU_QUICK`（那是 x·σ(1.702x)，gptoss 专用，
   差 4.041e-03）；改成 `ggml_gelu` 后还剩 1.3e-4 —— 因为 **ggml 的
   CPU f32 GELU 走 fp16 查表**（`vec.h` 无条件 `#define GGML_GELU_FP16`）。
   最后 tanh 近似按公式用**原语拼**，1.379e-07。
3. **`shapes_of` 的全局宽度 bug**：多宽度模型（gemma-4-31B：60 层文本
   5376 + 27 层视觉塔 1152）的全局张量按**尾层的 `in_dim`** 推 ——
   `embed.weight` 推成 (262144, 1152)，而 GGUF 判卷人说 `output_norm`
   是 [5376]。单宽度模型永远碰不到（循环变量恰好等于 d_model）。

**一个被数值证明（而不是假设）的映射**：by1 的 rope `interleaved`
（奇偶进、**拼接**着出）对上 ggml 的 NORMAL 模式（奇偶进、奇偶出）——
输出布局差一个 q/k **共同**置换，点积不变。孪生模型的 1.390e-07
就是这条的判据。`half` 那支是 NEOX 的逐位对应，无需论证。

**M4 的第一块拦路石**（记下来，别再撞）：llama.cpp 的 `libllama`
在 MinGW 下编译失败（`llama-model.cpp`），而 ggml 三个静态库完好 ——
M4 开工前先解决它。另外 `git clone` 的 https 在这台机器上不通，
拉源码走的是 codeload 的 tarball。

**下一步**：M3（MoE / sink / YaRN / gptoss 激活的映射，判卷人 =
gpt-oss-120b 与 Laguna 的 GGUF manifest + NumPy 孪生）→ M4（生成进
llama.cpp 本体，判卷人 = llama.cpp 编译 + llama-cli 逐位）。
