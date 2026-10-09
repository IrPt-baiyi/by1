# by1 支持的模型 —— 架构对比

> **从 IR 生成的**（`python src/by1cmp.py --md`），不是手写的。
> 手写的对比表会过期，而过期的表比没有表更坏。
>
> **只列在模型之间真的不一样的属性。** 全都一样的列出来是噪音，
> 而噪音会把信号淹掉 —— 这个项目里反复吃过这个亏。
>
> 表里是**短名**（可读）；**文件名是 HF 的模型名**（可追溯）。
> 两者都由 `models.tsv` + `by1name.py` 生成，不会漂。

### 命名对照（`models.tsv` 是唯一真相源）

| 短名（表里用的） | 长名（= 文件名） | HF 仓库 |
|---|---|---|
| `gpt-oss-120b` | `gpt-oss-120b` | [openai/gpt-oss-120b](https://hf-mirror.com/openai/gpt-oss-120b) |
| `gemma-4-31B` | `gemma-4-31B` | [google/gemma-4-31B](https://hf-mirror.com/google/gemma-4-31B) |
| `Laguna-XS-2_1` | `Laguna-XS-2_1` | [poolside/Laguna-XS-2_1](https://hf-mirror.com/poolside/Laguna-XS-2_1) |
| `Instella-3B` | `Instella-3B` | [amd/Instella-3B](https://hf-mirror.com/amd/Instella-3B) |
| `nerkyor/Step-3_7-Flash` | `Step-3_7-Flash-180B-LynnStyle-GLM52-SFT-GPT55-RL` | [nerkyor/Step-3_7-Flash-180B-LynnStyle-GLM52-SFT-GPT55-RL](https://hf-mirror.com/nerkyor/Step-3_7-Flash-180B-LynnStyle-GLM52-SFT-GPT55-RL) |
| `Step-3.7-Flash` | `Step-3.7-Flash` | [stepfun-ai/Step-3.7-Flash](https://hf-mirror.com/stepfun-ai/Step-3.7-Flash) |
| `Ling-3.0-tiny` | `Ling-3.0-tiny` | [inclusionAI/Ling-3.0-tiny](https://hf-mirror.com/inclusionAI/Ling-3.0-tiny) |
| `minimind-3` | `minimind-3` | [jingyaogong/minimind-3](https://hf-mirror.com/jingyaogong/minimind-3) |
| `clef` | `clef` | [Cloudflare/clef](https://hf-mirror.com/Cloudflare/clef) |
| `Qwen3.8-27B` | `Qwen3.8-27B` | [Qwen/Qwen3.8-27B](https://hf-mirror.com/Qwen/Qwen3.8-27B) |
| `Qwen3.6-35B-A3B` | `Qwen3.6-35B-A3B` | [Qwen/Qwen3.6-35B-A3B](https://hf-mirror.com/Qwen/Qwen3.6-35B-A3B) |
| `gpt2` | `gpt2` | [openai-community/gpt2](https://hf-mirror.com/openai-community/gpt2) |
| `Nemotron-3.5-Lightning` | `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://hf-mirror.com/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) |
| `GLM-5.3-Flash` | `GLM-5.3-Flash` | [zai-org/GLM-5.3-Flash](https://hf-mirror.com/zai-org/GLM-5.3-Flash) |

## 一、谁是谁

| 模型 | 参数 | 层 | d_model | 词表 | 机制 |
|---|---|---|---|---|---|
| `minimind-3` | 69M | 8 | 768 | 6400 | Attn×8+FFN×8 |
| `gpt2` | 163M | 12 | 768 | 50257 | Attn×12+FFN×12 |
| `Instella-3B` | 3.1B | 36 | 2560 | 50304 | Attn×36+FFN×36 |
| `clef` | 26.9B | 64 | 5120 | 248320 | Attn×16+FFN×64+Lin×48 |
| `Qwen3.8-27B` | 27.3B | 65 | 5120 | 248320 | Attn×17+FFN×65+Lin×48 |
| `gemma-4-31B` | 32.1B | 60 | 5376 | 262144 | Attn×60+FFN×60 |
| `Laguna-XS-2_1` | 33.4B | 40 | 2048 | 100352 | Attn×40+FFN×1+MoE×39 |
| `Qwen3.6-35B-A3B` | 35.5B | 41 | 2048 | 248320 | Attn×11+Lin×30+MoE×41 |
| `gpt-oss-120b` | 116.8B | 36 | 2880 | 201088 | Attn×36+MoE×36 |
| `nerkyor/Step-3_7-Flash` | 180.8B | 45 | 4096 | 128896 | Attn×45+FFN×3+MoE×42 |
| `Step-3.7-Flash` | 197.0B | 45 | 4096 | 128896 | Attn×45+FFN×3+MoE×42 |

## 二、**独苗** —— 只有它这样的

**独苗才是需要新原语的地方。** 别的都只是参数不同。

| 属性 | 取值 | 只有 |
|---|---|---|
| `Attention.qk_norm` | `'full'` | **Instella-3B** |
| `Attention.rope` | `False` | **gpt2** |
| `Attention.rope_scale` | `1.3465735902799727` | **Laguna-XS-2_1** |
| `Attention.sink` | `True` | **gpt-oss-120b** |
| `Attention.window` | `1024` | **gemma-4-31B** |
| `Attention.window` | `128` | **gpt-oss-120b** |
| `Attention.yarn` | `{'type': 'yarn', 'factor': 32.0, 'original': 4096, 'beta_fast': 32.0, 'beta_slow': 1.0, 'high_freq': 4.0, 'low_freq': 1.0, 'truncate': False}` | **gpt-oss-120b** |
| `Attention.yarn` | `{'type': 'yarn', 'factor': 32.0, 'original': 8192, 'beta_fast': 64.0, 'beta_slow': 1.0, 'high_freq': 4.0, 'low_freq': 1.0, 'truncate': True}` | **Laguna-XS-2_1** |
| `FFN.act` | `'gelu_new'` | **gpt2** |
| `FFN.bias` | `True` | **gpt2** |
| `FFN.gate` | `False` | **gpt2** |
| `MoE.act` | `'gptoss'` | **gpt-oss-120b** |
| `MoE.expert_bias` | `True` | **gpt-oss-120b** |
| `MoE.limit` | `7.0` | **gpt-oss-120b** |
| `MoE.routing` | `'topk_softmax'` | **gpt-oss-120b** |
| `MoE.score_bias` | `True` | **Laguna-XS-2_1** |
| `MoE.shared` | `0` | **gpt-oss-120b** |
| `MoE.shared_gate` | `True` | **Qwen3.6-35B-A3B** |
| `norm_kind` | `'layer'` | **gpt2** |
| `pos_kind` | `'learned'` | **gpt2** |

**11 个模型里，5 个一个独苗都没有**：minimind-3, clef, Qwen3.8-27B, Step-3_7-Flash-180B-LynnStyle-GLM52-SFT-GPT55-RL, Step-3.7-Flash

> 这就是"原语集合收敛"这句话的可数形式 ——
> 说"收敛了"是感觉，说"这几个模型贡献 0 个新原语"是账。

> 而独苗集中在 **gpt2**（另一个时代）和 **gpt-oss-120b**
> （最特殊的一个）—— **新原语是从"另一个时代"和"最特殊的那个"
> 来的，不是从"又一个 Llama 变体"来的。**

## 三、逐属性的全部取值

### `Attention.bias`

- `False` — minimind-3, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B, Laguna-XS-2_1, Qwen3.6-35B-A3B, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `True` — gpt2, gpt-oss-120b

### `Attention.head_dim`

- `256` — clef, Qwen3.8-27B, gemma-4-31B, Qwen3.6-35B-A3B
- `128` — Laguna-XS-2_1, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `64` — gpt2, gpt-oss-120b
- `96` — minimind-3 ← **独苗**
- `80` — Instella-3B ← **独苗**

### `Attention.head_gate`

- `'off'` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3.6-35B-A3B, gpt-oss-120b
- `'per_head'` — Laguna-XS-2_1, nerkyor/Step-3_7-Flash, Step-3.7-Flash

### `Attention.kv`

- `8` — Laguna-XS-2_1, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `4` — minimind-3, clef, Qwen3.8-27B
- `12` — gpt2 ← **独苗**
- `32` — Instella-3B ← **独苗**
- `16` — gemma-4-31B ← **独苗**
- `2` — Qwen3.6-35B-A3B ← **独苗**

### `Attention.norm_eps`

- `1e-06` — minimind-3, clef, Qwen3.8-27B, gemma-4-31B, Laguna-XS-2_1, Qwen3.6-35B-A3B
- `1e-05` — gpt2, Instella-3B, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash

### `Attention.out_dim`

- `6144` — clef, Qwen3.8-27B, Laguna-XS-2_1
- `8192` — gemma-4-31B, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `768` — minimind-3, gpt2
- `4096` — Qwen3.6-35B-A3B, gpt-oss-120b
- `2560` — Instella-3B ← **独苗**

### `Attention.q`

- `64` — gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `32` — Instella-3B, gemma-4-31B
- `24` — clef, Qwen3.8-27B
- `8` — minimind-3 ← **独苗**
- `12` — gpt2 ← **独苗**
- `48` — Laguna-XS-2_1 ← **独苗**
- `16` — Qwen3.6-35B-A3B ← **独苗**

### `Attention.q_gate`

- `False` — minimind-3, gpt2, Instella-3B, gemma-4-31B, Laguna-XS-2_1, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `True` — clef, Qwen3.8-27B, Qwen3.6-35B-A3B

### `Attention.qk_norm`

- `'per_head'` — minimind-3, clef, Qwen3.8-27B, gemma-4-31B, Laguna-XS-2_1, Qwen3.6-35B-A3B, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `'off'` — gpt2, gpt-oss-120b
- `'full'` — Instella-3B ← **独苗**

### `Attention.rope`

- `True` — minimind-3, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B, Laguna-XS-2_1, Qwen3.6-35B-A3B, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `False` — gpt2 ← **独苗**

### `Attention.rope_base`

- `10000` — gpt2, Instella-3B, gemma-4-31B
- `10000000` — clef, Qwen3.8-27B, Qwen3.6-35B-A3B
- `5000000` — nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `1000000` — minimind-3 ← **独苗**
- `500000` — Laguna-XS-2_1 ← **独苗**
- `150000` — gpt-oss-120b ← **独苗**

### `Attention.rope_pairing`

- `'half'` — minimind-3, Instella-3B, clef, Qwen3.8-27B, Laguna-XS-2_1, Qwen3.6-35B-A3B, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `'interleaved'` — gpt2, gemma-4-31B, gpt-oss-120b

### `Attention.rope_partial`

- `1.0` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3.6-35B-A3B, gpt-oss-120b
- `0.5` — Laguna-XS-2_1, nerkyor/Step-3_7-Flash, Step-3.7-Flash

### `Attention.rope_scale`

- `1.0` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3.6-35B-A3B, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `1.3465735902799727` — Laguna-XS-2_1 ← **独苗**

### `Attention.sink`

- `False` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B, Laguna-XS-2_1, Qwen3.6-35B-A3B, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `True` — gpt-oss-120b ← **独苗**

### `Attention.window`

- `None` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, Laguna-XS-2_1, Qwen3.6-35B-A3B, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `1024` — gemma-4-31B ← **独苗**
- `128` — gpt-oss-120b ← **独苗**

### `Attention.yarn`

- `None` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3.6-35B-A3B
- `{'type': 'llama3', 'factor': 2.0, 'original': 131072, 'beta_fast': 32.0, 'beta_slow': 1.0, 'high_freq': 32.0, 'low_freq': 1.0, 'truncate': True}` — nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `{'type': 'yarn', 'factor': 32.0, 'original': 8192, 'beta_fast': 64.0, 'beta_slow': 1.0, 'high_freq': 4.0, 'low_freq': 1.0, 'truncate': True}` — Laguna-XS-2_1 ← **独苗**
- `{'type': 'yarn', 'factor': 32.0, 'original': 4096, 'beta_fast': 32.0, 'beta_slow': 1.0, 'high_freq': 4.0, 'low_freq': 1.0, 'truncate': False}` — gpt-oss-120b ← **独苗**

### `FFN.act`

- `'silu'` — minimind-3, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B, Laguna-XS-2_1, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `'—'` — Qwen3.6-35B-A3B, gpt-oss-120b
- `'gelu_new'` — gpt2 ← **独苗**

### `FFN.alpha`

- `1.702` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B, Laguna-XS-2_1, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `'—'` — Qwen3.6-35B-A3B, gpt-oss-120b

### `FFN.bias`

- `False` — minimind-3, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B, Laguna-XS-2_1, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `'—'` — Qwen3.6-35B-A3B, gpt-oss-120b
- `True` — gpt2 ← **独苗**

### `FFN.gate`

- `True` — minimind-3, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B, Laguna-XS-2_1, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `'—'` — Qwen3.6-35B-A3B, gpt-oss-120b
- `False` — gpt2 ← **独苗**

### `FFN.hidden`

- `17408` — clef, Qwen3.8-27B
- `'—'` — Qwen3.6-35B-A3B, gpt-oss-120b
- `11264` — nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `2432` — minimind-3 ← **独苗**
- `3072` — gpt2 ← **独苗**
- `6912` — Instella-3B ← **独苗**
- `21504` — gemma-4-31B ← **独苗**
- `8192` — Laguna-XS-2_1 ← **独苗**

### `FFN.limit`

- `None` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B, Laguna-XS-2_1, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `'—'` — Qwen3.6-35B-A3B, gpt-oss-120b

### `Linear.act`

- `'—'` — minimind-3, gpt2, Instella-3B, gemma-4-31B, Laguna-XS-2_1, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `'silu'` — clef, Qwen3.8-27B, Qwen3.6-35B-A3B

### `Linear.conv_kernel`

- `'—'` — minimind-3, gpt2, Instella-3B, gemma-4-31B, Laguna-XS-2_1, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `4` — clef, Qwen3.8-27B, Qwen3.6-35B-A3B

### `Linear.k_dim`

- `'—'` — minimind-3, gpt2, Instella-3B, gemma-4-31B, Laguna-XS-2_1, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `128` — clef, Qwen3.8-27B, Qwen3.6-35B-A3B

### `Linear.k_heads`

- `'—'` — minimind-3, gpt2, Instella-3B, gemma-4-31B, Laguna-XS-2_1, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `16` — clef, Qwen3.8-27B, Qwen3.6-35B-A3B

### `Linear.l2_eps`

- `'—'` — minimind-3, gpt2, Instella-3B, gemma-4-31B, Laguna-XS-2_1, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `1e-06` — clef, Qwen3.8-27B, Qwen3.6-35B-A3B

### `Linear.norm_eps`

- `'—'` — minimind-3, gpt2, Instella-3B, gemma-4-31B, Laguna-XS-2_1, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `1e-06` — clef, Qwen3.8-27B, Qwen3.6-35B-A3B

### `Linear.out_dim`

- `'—'` — minimind-3, gpt2, Instella-3B, gemma-4-31B, Laguna-XS-2_1, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `6144` — clef, Qwen3.8-27B
- `4096` — Qwen3.6-35B-A3B ← **独苗**

### `Linear.v_dim`

- `'—'` — minimind-3, gpt2, Instella-3B, gemma-4-31B, Laguna-XS-2_1, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `128` — clef, Qwen3.8-27B, Qwen3.6-35B-A3B

### `Linear.v_heads`

- `'—'` — minimind-3, gpt2, Instella-3B, gemma-4-31B, Laguna-XS-2_1, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `48` — clef, Qwen3.8-27B
- `32` — Qwen3.6-35B-A3B ← **独苗**

### `MoE.act`

- `'—'` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B
- `'silu'` — Laguna-XS-2_1, Qwen3.6-35B-A3B, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `'gptoss'` — gpt-oss-120b ← **独苗**

### `MoE.alpha`

- `'—'` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B
- `1.702` — Laguna-XS-2_1, Qwen3.6-35B-A3B, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash

### `MoE.expert_bias`

- `'—'` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B
- `False` — Laguna-XS-2_1, Qwen3.6-35B-A3B, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `True` — gpt-oss-120b ← **独苗**

### `MoE.experts`

- `'—'` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B
- `256` — Laguna-XS-2_1, Qwen3.6-35B-A3B
- `128` — gpt-oss-120b ← **独苗**
- `255` — nerkyor/Step-3_7-Flash ← **独苗**
- `288` — Step-3.7-Flash ← **独苗**

### `MoE.hidden`

- `'—'` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B
- `512` — Laguna-XS-2_1, Qwen3.6-35B-A3B
- `1280` — nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `2880` — gpt-oss-120b ← **独苗**

### `MoE.limit`

- `'—'` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B
- `None` — Laguna-XS-2_1, Qwen3.6-35B-A3B, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `7.0` — gpt-oss-120b ← **独苗**

### `MoE.limit_shared`

- `'—'` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B
- `None` — Laguna-XS-2_1, Qwen3.6-35B-A3B, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash

### `MoE.n_group`

- `'—'` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B
- `0` — Laguna-XS-2_1, Qwen3.6-35B-A3B, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash

### `MoE.routed_scale`

- `'—'` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B
- `1.0` — Qwen3.6-35B-A3B, gpt-oss-120b
- `3.0` — nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `2.5` — Laguna-XS-2_1 ← **独苗**

### `MoE.router_bias`

- `'—'` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B
- `True` — gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `False` — Laguna-XS-2_1, Qwen3.6-35B-A3B

### `MoE.routing`

- `'—'` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B
- `'softmax_topk'` — Laguna-XS-2_1, Qwen3.6-35B-A3B
- `'sigmoid_topk'` — nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `'topk_softmax'` — gpt-oss-120b ← **独苗**

### `MoE.score_bias`

- `'—'` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B
- `False` — Qwen3.6-35B-A3B, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `True` — Laguna-XS-2_1 ← **独苗**

### `MoE.shared`

- `'—'` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B
- `1` — Laguna-XS-2_1, Qwen3.6-35B-A3B, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `0` — gpt-oss-120b ← **独苗**

### `MoE.shared_gate`

- `'—'` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B
- `False` — Laguna-XS-2_1, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `True` — Qwen3.6-35B-A3B ← **独苗**

### `MoE.shared_hidden`

- `'—'` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B
- `512` — Laguna-XS-2_1, Qwen3.6-35B-A3B
- `1280` — nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `0` — gpt-oss-120b ← **独苗**

### `MoE.top_k`

- `'—'` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B
- `8` — Laguna-XS-2_1, Qwen3.6-35B-A3B, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `4` — gpt-oss-120b ← **独苗**

### `MoE.topk_group`

- `'—'` — minimind-3, gpt2, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B
- `0` — Laguna-XS-2_1, Qwen3.6-35B-A3B, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash

### `ctx`

- `262144` — clef, Qwen3.8-27B, gemma-4-31B, Laguna-XS-2_1, Qwen3.6-35B-A3B, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `32768` — minimind-3 ← **独苗**
- `1024` — gpt2 ← **独苗**
- `4096` — Instella-3B ← **独苗**
- `131072` — gpt-oss-120b ← **独苗**

### `d_model`

- `768` — minimind-3, gpt2
- `5120` — clef, Qwen3.8-27B
- `2048` — Laguna-XS-2_1, Qwen3.6-35B-A3B
- `4096` — nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `2560` — Instella-3B ← **独苗**
- `5376` — gemma-4-31B ← **独苗**
- `2880` — gpt-oss-120b ← **独苗**

### `n_layer`

- `36` — Instella-3B, gpt-oss-120b
- `45` — nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `8` — minimind-3 ← **独苗**
- `12` — gpt2 ← **独苗**
- `64` — clef ← **独苗**
- `65` — Qwen3.8-27B ← **独苗**
- `60` — gemma-4-31B ← **独苗**
- `40` — Laguna-XS-2_1 ← **独苗**
- `41` — Qwen3.6-35B-A3B ← **独苗**

### `norm_eps`

- `1e-06` — minimind-3, clef, Qwen3.8-27B, gemma-4-31B, Laguna-XS-2_1, Qwen3.6-35B-A3B
- `1e-05` — gpt2, Instella-3B, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash

### `norm_kind`

- `'rms'` — minimind-3, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B, Laguna-XS-2_1, Qwen3.6-35B-A3B, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `'layer'` — gpt2 ← **独苗**

### `pos_kind`

- `'rope'` — minimind-3, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B, Laguna-XS-2_1, Qwen3.6-35B-A3B, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `'learned'` — gpt2 ← **独苗**

### `vocab`

- `248320` — clef, Qwen3.8-27B, Qwen3.6-35B-A3B
- `128896` — nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `6400` — minimind-3 ← **独苗**
- `50257` — gpt2 ← **独苗**
- `50304` — Instella-3B ← **独苗**
- `262144` — gemma-4-31B ← **独苗**
- `100352` — Laguna-XS-2_1 ← **独苗**
- `201088` — gpt-oss-120b ← **独苗**

### `参数`

- `68827392` — minimind-3 ← **独苗**
- `163037184` — gpt2 ← **独苗**
- `3112497280` — Instella-3B ← **独苗**
- `26895998464` — clef ← **独苗**
- `27268253696` — Qwen3.8-27B ← **独苗**
- `32105986304` — gemma-4-31B ← **独苗**
- `33442617088` — Laguna-XS-2_1 ← **独苗**
- `35496856704` — Qwen3.6-35B-A3B ← **独苗**
- `116829156672` — gpt-oss-120b ← **独苗**
- `180845807680` — nerkyor/Step-3_7-Flash ← **独苗**
- `196956130368` — Step-3.7-Flash ← **独苗**

### `机制`

- `'Attention×45+FFN×3+MoE×42'` — nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `'Attention×8+FFN×8'` — minimind-3 ← **独苗**
- `'Attention×12+FFN×12'` — gpt2 ← **独苗**
- `'Attention×36+FFN×36'` — Instella-3B ← **独苗**
- `'Attention×16+FFN×64+Linear×48'` — clef ← **独苗**
- `'Attention×17+FFN×65+Linear×48'` — Qwen3.8-27B ← **独苗**
- `'Attention×60+FFN×60'` — gemma-4-31B ← **独苗**
- `'Attention×40+FFN×1+MoE×39'` — Laguna-XS-2_1 ← **独苗**
- `'Attention×11+Linear×30+MoE×41'` — Qwen3.6-35B-A3B ← **独苗**
- `'Attention×36+MoE×36'` — gpt-oss-120b ← **独苗**

