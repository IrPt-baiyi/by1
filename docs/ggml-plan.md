# 接 llama.cpp：第一个外部判卷人

> 这份是**量出来的计划**，不是设想。数字来自 `by1emit.py`、
> `by1check`、`refs/*.gguf-tensors.json` 和 `gcc --version`，都是这一轮实测的。
>
> **它要解决的那件事**：`refs/` 判得了名字和形状，判不了数学 —— 这是
> `eps` 第六次、`apply_rope` 交错、`gpt2` 与 `llama-shaped` 编译出同一份 IR
> 这三次数值错能全绿通过的原因。**一个独立实现能判。**

---

## 一、`by1emit.py` 的头注已经过期了

它现在写的是：

> 背景：这个环境里**没有 C 编译器**、没有 llama.cpp、github 不通。
> 所以「生成 ggml 代码」这一步**无法验证**，写出来只能看起来对 —— **那不写**。

**这句的前提现在不成立**（实测）：

```
gcc.exe (MinGW-W64 x86_64-ucrt-posix-seh) 16.2.0     在 PATH-adjacent 位置
llamacpp/                                            13 个参考文件
   llama-arch.cpp · llama-arch.h · llama-hparams.cpp · llama-model.cpp
   delta-net-base.cpp · laguna_modeling.py · modeling_bailing_moe_v3.py …
```

**所以"不写"的理由没了** —— 而 `by1emit` 自己的结论正是
"[PASS] 图与 manifest 双向一致 —— 3b 剩下的只是把原语写成 C"。

---

## 二、地基比看上去完整

`by1emit.py <model>` 已经产出三样（169 行，跑得通）：

**① 后端必须实现的算子** —— 种类 + 个数 + 每个算子要哪些 ggml 原语

```
Add          72 个    ggml_add
Attention    36 个    ggml_mul_mat(qkv), ggml_rope_ext, ggml_mul_mat(qk),
                      ggml_soft_max_ext(掩码, sink), ggml_mul_mat(av), ggml_mul_mat(o)
MoE          36 个    ggml_mul_mat(router), ggml_top_k, ggml_soft_max,
                      ggml_mul_mat(专家, 逐专家), ggml_mul_mat(down), ggml_add
Norm         72 个    ggml_rms_norm, ggml_mul
```

**② 声明式图** —— 逐层逐步的数据流，每步标了调哪个原语

```
L0  op0  hidden            -> op0.out   Norm       [2 步]
L0  op1  op0.out           -> op1.out   Attention  [6 步]
L0  op2  hidden , op1.out  -> op2.out   Add        [1 步]
```

**③ 与 GGUF manifest 双向对拍** —— `612 = 612`，两个方向都是 0 差

---

## 三、谁都能接吗 —— 不

**所有 26 份 `.by1` 都只用 4~5 种算子**（Add / Attention 或 FFN 或 MLA 或 Raw / MoE / Norm）——
差别只在个数，不在种类。这是好消息：**算子种类是收敛的。**

**但 ggml 映射只覆盖五种**：

| 算子 | ggml 原语 |
|---|---|
| `Add` | ✓ |
| `Attention` | ✓ |
| `FFN` | ✓ |
| `MoE` | ✓ |
| `Norm` | ✓ |
| `MLA` | **`?`（没有）** |
| `Raw` | **`?`（没有）** |

（`by1emit` 的源码就是 `GGML.get(k, ['?'])` —— 缺的会**看得见**，
不会静默按别的算。）

而 `MLA` / `Raw` 不在下面那三个模型里，所以**第一个目标不受影响**。

---

## 四、第一个目标：`gemma-4-31B`

只有三个模型有 GGUF manifest 当判卷人：

| 模型 | 算子种类 | ggml 原语 | GGUF 对拍 |
|---|---|---|---|
| **`gemma-4-31B.by1`** | **4** | **12** | **[PASS]** |
| `gpt-oss-120b.by1` | 4 | 16 | [PASS] |
| `Laguna-XS-2_1.by1` | 5 | 18 | [PASS] |

**取 `gemma-4-31B`** —— 算子集最小，而且它要的原语全是已映射的。

**要写的 C 是 12 个原语**：

```
ggml_add · ggml_mul_mat · ggml_rms_norm · ggml_mul
ggml_rope_ext · ggml_soft_max_ext
ggml_silu · …（FFN 那一路）
```

**31B 不等于要跑 31B** —— 数值对拍用**小维度 + 随机权重**，
和 `by1diff` / `by1opdiff` 现在做的事一样。判的是**算得对不对**，
不是跑得动。

---

## 五、判据（这一步的"完成"是什么）

```
① 生成的 C 用 gcc 编得过                        ← 现在能做到
② 小维度随机权重下，数值和 torch 后端一致         ← 阈值待定，参照现有对拍
③ 张量名和 GGUF manifest 双向一致                ← 已经 PASS，不许退
④ 覆盖不降：编译 N · import N · 行尾 N           ← 门里的硬判据
```

**②是新的东西** —— 它是这个仓库里**第一个不由自己写的数学判据**。

---

## 六、这一步不依赖 `check()` 劈不劈

`by1emit` 读的是 **IR**，不是 `check()` 的内部状态：

```
.by1  →  by1check  →  IR  →  by1emit  →  ggml 原语 + 图
```

所以"劈 `check()`"和"接 llama.cpp"是**两件可以分开做的事**。
目标里把它们串起来，是因为分层是**手段**、外部判卷人是**目的** ——
而手段碰壁四轮之后，目的这条路仍然是通的。
