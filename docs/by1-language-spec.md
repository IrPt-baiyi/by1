# `.by1` 语言合同（T2 · 可执行规范）

> 判据：语法、错误模型、版本策略和最小示例形成一份可执行规范；**示例、解析器和错误输出相互一致**。
> 一致性判卷人：`src/checks/by1langspec.py`（把本文的示例喂给 `by1check`，绿例必须绿、红例必须红且信息含指定子串）。
> 本文不重复 [ir-spec.md](../ir-spec.md)（IR 是编译目标，这里是前端语言）。

## 1. 语法骨架

```by1
model <名字> {
  backend transformers | numpy | c | ggml
  arch <名字>
  ctx <整数>

  hparams {
    d_model = <整数>
    vocab = <整数>
    rms_eps = <浮点>
    tie_word_embeddings = true | false
    bos_token_id = <整数>
    eos_token_id = <整数>
  }

  stack <名字> {
    pattern = <N> * <机制>(<参数表>)     # 层排布；N 为展开层数
    <栈名>[:] >> <机制>(<参数表>)        # 通配调度
  }

  position {
    default = rope(base = <浮点>, pairing = half | full)
  }

  state { <名> : <机制> ... }            # 可选：状态/缓存合同
  optimizer { except(<符号表>) ... }     # 可选：训练排除表
}
```

**机制（mechanism）** 名单与参数表是唯一事实源，见 `by1codegen` 的机制键表与
[models.md](../models.md) 的对照；写不出名字的机制必须被拒绝（"要么实现要么拒绝"）。

## 2. 错误模型

- 每条诊断一行：`[E] <行号> <信息>`（错误）/ `[W] <行号> <信息>`（警告）。
- **有 `[E]` 则退出码非零**；警告不改变退出码。
- 信息必须指出**位置（行号）**与**可行动的原因**（例：`state 条目 'Missing' 解析不到任何已声明的机制`）。
- 静默回退被禁止：未知机制/未知 dtype/矛盾配置必须报错，不许"猜一个相近的"。

负例基线：`models/selftest.by1` 固定报 **9 个错误**（解析器改动后数目变化必须显式改基线并说明）。

## 3. 版本策略

- **IR_VERSION**（`src/by1ver.py`）：只在 IR 结构变化时升**主版本**；加 `.by1` 语法不动 IR_VERSION，只动 TOOL_VERSION。
- **HISTORY 追加不改写**：每条记录"版本 + 何时 + 何处 + 影响"。
- 旧描述"要么继续有效，要么由可重复的迁移工具升级"（路线图阶段 A）。
- 1.0 版的 1.0 文件继续可读：版本检查只对主版本硬（T22 判例：1.0 文件在 IR 1.1 下有效）。

## 4. 最小示例（绿）

```by1
model TinyGreen {
  backend transformers
  arch qwen2
  ctx 512

  hparams {
    d_model = 64
    vocab = 128
    rms_eps = 1e-6
    tie_word_embeddings = true
    bos_token_id = 1
    eos_token_id = 2
  }

  stack main {
    pattern = 2 * Attention(q = 2, kv = 2, head_dim = 32, qkv_bias = true, out_bias = false)
    main[:] >> FFN(hidden = 128)
  }

  position {
    default = rope(base = 10000.0, pairing = half)
  }
}
```

## 5. 最小负例（红）

```by1
model TinyRed {
  backend transformers
  arch qwen2
  ctx 512

  hparams {
    d_model = 64
    vocab = 128
    rms_eps = 1e-6
    tie_word_embeddings = true
    bos_token_id = 1
    eos_token_id = 2
  }

  stack main {
    pattern = 2 * Ghost(q = 2)         # 未声明机制 -> [E] 必须指出名字
  }

  position {
    default = rope(base = 10000.0, pairing = half)
  }
}
```

期望诊断（判卷人核对子串）：`[E]` 含 `Ghost`；退出码非零。

## 6. 一致性自检

| 检查 | 判据 |
|---|---|
| 绿例 | `by1check` 退出码 0，`0 错误` |
| 红例 | `by1check` 退出码非零，诊断含 `Ghost` |
| selftest 基线 | `models/selftest.by1` 恰好 9 错（防漂移） |
| 行尾/编码 | LF · UTF-8（`by1fast` ⑤） |

判卷人命令：`python src/checks/by1langspec.py`。
