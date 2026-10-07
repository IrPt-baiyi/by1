# 云端 runbook

**要租卡是为了三件事**，按"现在就能做 / 必须有 CUDA"分开列。

---

## 0. 先确认这一包是自洽的

```bash
pip install -r gpu/requirements.txt
python by1all.py --quick      # 跳过 C 后端（要 gcc）
```

期望：**34 项里除了一步之外全过**。那一步是：

```
已知缺口（不算失败，但仍然存在）:
  - tensors torch.module step-3.7-flash.by1
    那 25 个张量在 HF 的 model-00009 分片里，而那个分片的头是全零 ——
    镜像的问题，不是 by1 的
```

**如果这里就有别的失败，先停下** —— 说明这一包本身有问题，别在 GPU 上查。

---

## 1. 不需要 GPU，但这一包里还没做的

**`instella-3b` 的前向。** 它稠密、3B，是六个真实模型里唯一在这台机器上也跑得动的。

```bash
python by1instella.py
```

判卷人是官方仓库里的 `modeling_instella.py`。**这一步在租卡之前就该在这里做完** ——
如果它过了，六个真实模型里就有一个是**行为上**验过的，而不只是命名上。

（它依赖 `qk_norm = full` —— Instella 的 QK-norm 是整宽的 `[2560]`，不是按头的 `[80]`。
by1 在此之前是**明确拒绝**它的。）

---

## 2. 必须有 CUDA：KDA

**这是租卡最主要的理由。**

KDA 是 Ling-3.0-tiny 里那 18 层线性注意力。它的参考实现在
**`fla`（flash-linear-attention）** 里：

```python
from fla.ops.kda import chunk_kda, fused_recurrent_kda
```

而 `fla` 要 **Triton** —— 没有 CUDA 就 import 不了。

```bash
python gpu/by1kda.py
```

这个脚本做三件事：

1. 拉 `inclusionAI/Ling-3.0-tiny` 的 `modeling_bailing_moe_v3.py`
2. 用 `fla` 把 **`BailingMoeV3KimiDeltaAttention`** 和 **by1 生成的 KDA**
   在同样的权重下各跑一遍
3. 比输出

**期望**：差在 1e-6 量级。**对不上是正常的** —— 这一族从来没有在能跑的判卷人下验过，
第一次跑大概率有出入，而**有判卷人在旁边调试**是租卡的核心价值。

---

## 3. 必须有显存：真实模型的前向

按显存从小到大：

| 模型 | 参数量 | bf16 权重 | 建议 |
|---|---|---|---|
| `instella-3b` | 3B | 6 GB | 其实 CPU 也能跑（见 §1） |
| `gemma-4-31b` | 31B | 62 GB | 1×80G |
| `ling-3.0-tiny` | ~16B(A3B) | 32 GB | 1×80G，但 KDA 那部分要先过 §2 |
| `laguna-xs-2.1` | ? | ? | 看 config |
| `step-3.7 文本主干` | 180B(剪枝后) | ~360 GB | 多卡，或只跑前 N 层 |
| `gpt-oss-120b` | 130B(MXFP4) | ~65 GB | 1×80G，但要 mxfp4 反量化 |

```bash
python gpu/by1realfwd.py --model gemma-4-31b --layers 4
```

**`--layers N` 是关键**：真实维度、真实头数、真实 FFN，只跑前 N 层。
`by1diff.py` 对合成维度做到了这一步；这个脚本把维度换成**真实的**。

**为什么值得**：现在"六个真实模型全中"的准确版本是
**"config 与命名全中，数值行为只在合成维度上验过"**。
这一步把后半句去掉。

---

## 4. 这包里**没有**的东西

- **模型权重** —— 在云端从 HF 下，下多少取决于跑哪个
- **`drafts/`** —— 没有判卷人的草稿，故意不装
- **llama.cpp 的参考源码** —— 可重新下载；而且 `by1emit.py` 需要的只是图，
  它已经在 `refs/` 的 GGUF 张量表里了

---

## 5. 一件和 GPU 无关、但比 GPU 更缺的事

**这个项目没有外部用户。** 上表里每一行都是"和某个官方产物对上了"，
**没有一行是"有人用它做成了什么"。**

租卡能让验证更硬，但不会让这件事发生变化。
