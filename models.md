# by1 支持的模型 —— 架构对比

> **从 IR 生成的**（`python by1cmp.py --md`），不是手写的。
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
| `gptoss120b` | `gpt-oss-120b` | [openai/gpt-oss-120b](https://hf-mirror.com/openai/gpt-oss-120b) |
| `gemma31b` | `gemma-4-31B` | [google/gemma-4-31B](https://hf-mirror.com/google/gemma-4-31B) |
| `laguna` | `Laguna-XS-2_1` | [poolside/Laguna-XS-2_1](https://hf-mirror.com/poolside/Laguna-XS-2_1) |
| `instella3b` | `Instella-3B` | [amd/Instella-3B](https://hf-mirror.com/amd/Instella-3B) |
| `step37-pruned` | `Step-3_7-Flash-180B-LynnStyle-GLM52-SFT-GPT55-RL` | [nerkyor/Step-3_7-Flash-180B-LynnStyle-GLM52-SFT-GPT55-RL](https://hf-mirror.com/nerkyor/Step-3_7-Flash-180B-LynnStyle-GLM52-SFT-GPT55-RL) |
| `step37-official` | `Step-3.7-Flash` | [stepfun-ai/Step-3.7-Flash](https://hf-mirror.com/stepfun-ai/Step-3.7-Flash) |
| `lingtiny` | `Ling-3.0-tiny` | [inclusionAI/Ling-3.0-tiny](https://hf-mirror.com/inclusionAI/Ling-3.0-tiny) |
| `minimind3` | `minimind-3` | [jingyaogong/minimind-3](https://hf-mirror.com/jingyaogong/minimind-3) |
| `clef` | `clef` | [Cloudflare/clef](https://hf-mirror.com/Cloudflare/clef) |
| `qwen38` | `Qwen3.8-27B` | [Qwen/Qwen3.8-27B](https://hf-mirror.com/Qwen/Qwen3.8-27B) |
| `qwen36` | `Qwen3.6-35B-A3B` | [Qwen/Qwen3.6-35B-A3B](https://hf-mirror.com/Qwen/Qwen3.6-35B-A3B) |
| `gpt2` | `gpt2` | [openai-community/gpt2](https://hf-mirror.com/openai-community/gpt2) |
| `nemotron` | `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://hf-mirror.com/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) |
| `glm53` | `GLM-5.3-Flash` | [zai-org/GLM-5.3-Flash](https://hf-mirror.com/zai-org/GLM-5.3-Flash) |

## 一、谁是谁

| 模型 | 参数 | 层 | d_model | 词表 | 机制 |
|---|---|---|---|---|---|
| `minimind3` | 69M | 8 | 768 | 6400 | Attn×8+FFN×8 |
| `gpt2` | 163M | 12 | 768 | 50257 | Attn×12+FFN×12 |
| `instella3b` | 3.1B | 36 | 2560 | 50304 | Attn×36+FFN×36 |
| `clef` | 26.9B | 64 | 5120 | 248320 | Attn×16+FFN×64+Lin×48 |
| `qwen38` | 27.3B | 65 | 5120 | 248320 | Attn×17+FFN×65+Lin×48 |
| `gemma31b` | 32.1B | 60 | 5376 | 262144 | Attn×60+FFN×60 |
| `laguna` | 33.4B | 40 | 2048 | 100352 | Attn×40+FFN×1+MoE×39 |
| `qwen36` | 35.5B | 41 | 2048 | 248320 | Attn×11+Lin×30+MoE×41 |
| `gptoss120b` | 116.8B | 36 | 2880 | 201088 | Attn×36+MoE×36 |
| `step37-pruned` | 180.8B | 45 | 4096 | 128896 | Attn×45+FFN×3+MoE×42 |
| `step37-official` | 197.0B | 45 | 4096 | 128896 | Attn×45+FFN×3+MoE×42 |

## 二、**独苗** —— 只有它这样的

**独苗才是需要新原语的地方。** 别的都只是参数不同。

| 属性 | 取值 | 只有 |
|---|---|---|
| `Attention.qk_norm` | `'full'` | **instella3b** |
| `Attention.rope` | `False` | **gpt2** |
| `Attention.rope_scale` | `1.3465735902799727` | **laguna** |
| `Attention.sink` | `True` | **gptoss120b** |
| `Attention.window` | `1024` | **gemma31b** |
| `Attention.window` | `128` | **gptoss120b** |
| `Attention.yarn` | `{'type': 'yarn', 'factor': 32.0, 'original': 4096, 'beta_fast': 32.0, 'beta_slow': 1.0, 'high_freq': 4.0, 'low_freq': 1.0, 'truncate': False}` | **gptoss120b** |
| `Attention.yarn` | `{'type': 'yarn', 'factor': 32.0, 'original': 8192, 'beta_fast': 64.0, 'beta_slow': 1.0, 'high_freq': 4.0, 'low_freq': 1.0, 'truncate': True}` | **laguna** |
| `FFN.act` | `'gelu_new'` | **gpt2** |
| `FFN.bias` | `True` | **gpt2** |
| `FFN.gate` | `False` | **gpt2** |
| `MoE.act` | `'gptoss'` | **gptoss120b** |
| `MoE.expert_bias` | `True` | **gptoss120b** |
| `MoE.limit` | `7.0` | **gptoss120b** |
| `MoE.routing` | `'topk_softmax'` | **gptoss120b** |
| `MoE.score_bias` | `True` | **laguna** |
| `MoE.shared` | `0` | **gptoss120b** |
| `MoE.shared_gate` | `True` | **qwen36** |
| `norm_kind` | `'layer'` | **gpt2** |
| `pos_kind` | `'learned'` | **gpt2** |

**11 个模型里，10 个一个独苗都没有**：minimind-3, Instella-3B, clef, Qwen3.8-27B, gemma-4-31B, Laguna-XS-2_1, Qwen3.6-35B-A3B, gpt-oss-120b, Step-3_7-Flash-180B-LynnStyle-GLM52-SFT-GPT55-RL, Step-3.7-Flash

> 这就是"原语集合收敛"这句话的可数形式 ——
> 说"收敛了"是感觉，说"这几个模型贡献 0 个新原语"是账。

> 而独苗集中在 **gpt2**（另一个时代）和 **gpt-oss-120b**
> （最特殊的一个）—— **新原语是从"另一个时代"和"最特殊的那个"
> 来的，不是从"又一个 Llama 变体"来的。**

## 三、逐属性的全部取值

### `Attention.bias`

- `False` — minimind3, instella3b, clef, qwen38, gemma31b, laguna, qwen36, step37-pruned, step37-official
- `True` — gpt2, gptoss120b

### `Attention.head_dim`

- `256` — clef, qwen38, gemma31b, qwen36
- `128` — laguna, step37-pruned, step37-official
- `64` — gpt2, gptoss120b
- `96` — minimind3 ← **独苗**
- `80` — instella3b ← **独苗**

### `Attention.head_gate`

- `'off'` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b, qwen36, gptoss120b
- `'per_head'` — laguna, step37-pruned, step37-official

### `Attention.kv`

- `8` — laguna, gptoss120b, step37-pruned, step37-official
- `4` — minimind3, clef, qwen38
- `12` — gpt2 ← **独苗**
- `32` — instella3b ← **独苗**
- `16` — gemma31b ← **独苗**
- `2` — qwen36 ← **独苗**

### `Attention.norm_eps`

- `1e-06` — minimind3, clef, qwen38, gemma31b, laguna, qwen36
- `1e-05` — gpt2, instella3b, gptoss120b, step37-pruned, step37-official

### `Attention.out_dim`

- `6144` — clef, qwen38, laguna
- `8192` — gemma31b, step37-pruned, step37-official
- `768` — minimind3, gpt2
- `4096` — qwen36, gptoss120b
- `2560` — instella3b ← **独苗**

### `Attention.q`

- `64` — gptoss120b, step37-pruned, step37-official
- `32` — instella3b, gemma31b
- `24` — clef, qwen38
- `8` — minimind3 ← **独苗**
- `12` — gpt2 ← **独苗**
- `48` — laguna ← **独苗**
- `16` — qwen36 ← **独苗**

### `Attention.q_gate`

- `False` — minimind3, gpt2, instella3b, gemma31b, laguna, gptoss120b, step37-pruned, step37-official
- `True` — clef, qwen38, qwen36

### `Attention.qk_norm`

- `'per_head'` — minimind3, clef, qwen38, gemma31b, laguna, qwen36, step37-pruned, step37-official
- `'off'` — gpt2, gptoss120b
- `'full'` — instella3b ← **独苗**

### `Attention.rope`

- `True` — minimind3, instella3b, clef, qwen38, gemma31b, laguna, qwen36, gptoss120b, step37-pruned, step37-official
- `False` — gpt2 ← **独苗**

### `Attention.rope_base`

- `10000` — gpt2, instella3b, gemma31b
- `10000000` — clef, qwen38, qwen36
- `5000000` — step37-pruned, step37-official
- `1000000` — minimind3 ← **独苗**
- `500000` — laguna ← **独苗**
- `150000` — gptoss120b ← **独苗**

### `Attention.rope_pairing`

- `'half'` — minimind3, instella3b, clef, qwen38, laguna, qwen36, step37-pruned, step37-official
- `'interleaved'` — gpt2, gemma31b, gptoss120b

### `Attention.rope_partial`

- `1.0` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b, qwen36, gptoss120b
- `0.5` — laguna, step37-pruned, step37-official

### `Attention.rope_scale`

- `1.0` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b, qwen36, gptoss120b, step37-pruned, step37-official
- `1.3465735902799727` — laguna ← **独苗**

### `Attention.sink`

- `False` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b, laguna, qwen36, step37-pruned, step37-official
- `True` — gptoss120b ← **独苗**

### `Attention.window`

- `None` — minimind3, gpt2, instella3b, clef, qwen38, laguna, qwen36, step37-pruned, step37-official
- `1024` — gemma31b ← **独苗**
- `128` — gptoss120b ← **独苗**

### `Attention.yarn`

- `None` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b, qwen36
- `{'type': 'llama3', 'factor': 2.0, 'original': 131072, 'beta_fast': 32.0, 'beta_slow': 1.0, 'high_freq': 32.0, 'low_freq': 1.0, 'truncate': True}` — step37-pruned, step37-official
- `{'type': 'yarn', 'factor': 32.0, 'original': 8192, 'beta_fast': 64.0, 'beta_slow': 1.0, 'high_freq': 4.0, 'low_freq': 1.0, 'truncate': True}` — laguna ← **独苗**
- `{'type': 'yarn', 'factor': 32.0, 'original': 4096, 'beta_fast': 32.0, 'beta_slow': 1.0, 'high_freq': 4.0, 'low_freq': 1.0, 'truncate': False}` — gptoss120b ← **独苗**

### `FFN.act`

- `'silu'` — minimind3, instella3b, clef, qwen38, gemma31b, laguna, step37-pruned, step37-official
- `'—'` — qwen36, gptoss120b
- `'gelu_new'` — gpt2 ← **独苗**

### `FFN.alpha`

- `1.702` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b, laguna, step37-pruned, step37-official
- `'—'` — qwen36, gptoss120b

### `FFN.bias`

- `False` — minimind3, instella3b, clef, qwen38, gemma31b, laguna, step37-pruned, step37-official
- `'—'` — qwen36, gptoss120b
- `True` — gpt2 ← **独苗**

### `FFN.gate`

- `True` — minimind3, instella3b, clef, qwen38, gemma31b, laguna, step37-pruned, step37-official
- `'—'` — qwen36, gptoss120b
- `False` — gpt2 ← **独苗**

### `FFN.hidden`

- `17408` — clef, qwen38
- `'—'` — qwen36, gptoss120b
- `11264` — step37-pruned, step37-official
- `2432` — minimind3 ← **独苗**
- `3072` — gpt2 ← **独苗**
- `6912` — instella3b ← **独苗**
- `21504` — gemma31b ← **独苗**
- `8192` — laguna ← **独苗**

### `FFN.limit`

- `None` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b, laguna, step37-pruned, step37-official
- `'—'` — qwen36, gptoss120b

### `Linear.act`

- `'—'` — minimind3, gpt2, instella3b, gemma31b, laguna, gptoss120b, step37-pruned, step37-official
- `'silu'` — clef, qwen38, qwen36

### `Linear.conv_kernel`

- `'—'` — minimind3, gpt2, instella3b, gemma31b, laguna, gptoss120b, step37-pruned, step37-official
- `4` — clef, qwen38, qwen36

### `Linear.k_dim`

- `'—'` — minimind3, gpt2, instella3b, gemma31b, laguna, gptoss120b, step37-pruned, step37-official
- `128` — clef, qwen38, qwen36

### `Linear.k_heads`

- `'—'` — minimind3, gpt2, instella3b, gemma31b, laguna, gptoss120b, step37-pruned, step37-official
- `16` — clef, qwen38, qwen36

### `Linear.l2_eps`

- `'—'` — minimind3, gpt2, instella3b, gemma31b, laguna, gptoss120b, step37-pruned, step37-official
- `1e-06` — clef, qwen38, qwen36

### `Linear.norm_eps`

- `'—'` — minimind3, gpt2, instella3b, gemma31b, laguna, gptoss120b, step37-pruned, step37-official
- `1e-06` — clef, qwen38, qwen36

### `Linear.out_dim`

- `'—'` — minimind3, gpt2, instella3b, gemma31b, laguna, gptoss120b, step37-pruned, step37-official
- `6144` — clef, qwen38
- `4096` — qwen36 ← **独苗**

### `Linear.v_dim`

- `'—'` — minimind3, gpt2, instella3b, gemma31b, laguna, gptoss120b, step37-pruned, step37-official
- `128` — clef, qwen38, qwen36

### `Linear.v_heads`

- `'—'` — minimind3, gpt2, instella3b, gemma31b, laguna, gptoss120b, step37-pruned, step37-official
- `48` — clef, qwen38
- `32` — qwen36 ← **独苗**

### `MoE.act`

- `'—'` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b
- `'silu'` — laguna, qwen36, step37-pruned, step37-official
- `'gptoss'` — gptoss120b ← **独苗**

### `MoE.alpha`

- `'—'` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b
- `1.702` — laguna, qwen36, gptoss120b, step37-pruned, step37-official

### `MoE.expert_bias`

- `'—'` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b
- `False` — laguna, qwen36, step37-pruned, step37-official
- `True` — gptoss120b ← **独苗**

### `MoE.experts`

- `'—'` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b
- `256` — laguna, qwen36
- `128` — gptoss120b ← **独苗**
- `255` — step37-pruned ← **独苗**
- `288` — step37-official ← **独苗**

### `MoE.hidden`

- `'—'` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b
- `512` — laguna, qwen36
- `1280` — step37-pruned, step37-official
- `2880` — gptoss120b ← **独苗**

### `MoE.limit`

- `'—'` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b
- `None` — laguna, qwen36, step37-pruned, step37-official
- `7.0` — gptoss120b ← **独苗**

### `MoE.limit_shared`

- `'—'` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b
- `None` — laguna, qwen36, gptoss120b, step37-pruned, step37-official

### `MoE.n_group`

- `'—'` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b
- `0` — laguna, qwen36, gptoss120b, step37-pruned, step37-official

### `MoE.routed_scale`

- `'—'` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b
- `1.0` — qwen36, gptoss120b
- `3.0` — step37-pruned, step37-official
- `2.5` — laguna ← **独苗**

### `MoE.router_bias`

- `'—'` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b
- `True` — gptoss120b, step37-pruned, step37-official
- `False` — laguna, qwen36

### `MoE.routing`

- `'—'` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b
- `'softmax_topk'` — laguna, qwen36
- `'sigmoid_topk'` — step37-pruned, step37-official
- `'topk_softmax'` — gptoss120b ← **独苗**

### `MoE.score_bias`

- `'—'` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b
- `False` — qwen36, gptoss120b, step37-pruned, step37-official
- `True` — laguna ← **独苗**

### `MoE.shared`

- `'—'` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b
- `1` — laguna, qwen36, step37-pruned, step37-official
- `0` — gptoss120b ← **独苗**

### `MoE.shared_gate`

- `'—'` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b
- `False` — laguna, gptoss120b, step37-pruned, step37-official
- `True` — qwen36 ← **独苗**

### `MoE.shared_hidden`

- `'—'` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b
- `512` — laguna, qwen36
- `1280` — step37-pruned, step37-official
- `0` — gptoss120b ← **独苗**

### `MoE.top_k`

- `'—'` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b
- `8` — laguna, qwen36, step37-pruned, step37-official
- `4` — gptoss120b ← **独苗**

### `MoE.topk_group`

- `'—'` — minimind3, gpt2, instella3b, clef, qwen38, gemma31b
- `0` — laguna, qwen36, gptoss120b, step37-pruned, step37-official

### `ctx`

- `262144` — clef, qwen38, gemma31b, laguna, qwen36, step37-pruned, step37-official
- `32768` — minimind3 ← **独苗**
- `1024` — gpt2 ← **独苗**
- `4096` — instella3b ← **独苗**
- `131072` — gptoss120b ← **独苗**

### `d_model`

- `768` — minimind3, gpt2
- `5120` — clef, qwen38
- `2048` — laguna, qwen36
- `4096` — step37-pruned, step37-official
- `2560` — instella3b ← **独苗**
- `5376` — gemma31b ← **独苗**
- `2880` — gptoss120b ← **独苗**

### `n_layer`

- `36` — instella3b, gptoss120b
- `45` — step37-pruned, step37-official
- `8` — minimind3 ← **独苗**
- `12` — gpt2 ← **独苗**
- `64` — clef ← **独苗**
- `65` — qwen38 ← **独苗**
- `60` — gemma31b ← **独苗**
- `40` — laguna ← **独苗**
- `41` — qwen36 ← **独苗**

### `norm_eps`

- `1e-06` — minimind3, clef, qwen38, gemma31b, laguna, qwen36
- `1e-05` — gpt2, instella3b, gptoss120b, step37-pruned, step37-official

### `norm_kind`

- `'rms'` — minimind3, instella3b, clef, qwen38, gemma31b, laguna, qwen36, gptoss120b, step37-pruned, step37-official
- `'layer'` — gpt2 ← **独苗**

### `pos_kind`

- `'rope'` — minimind3, instella3b, clef, qwen38, gemma31b, laguna, qwen36, gptoss120b, step37-pruned, step37-official
- `'learned'` — gpt2 ← **独苗**

### `vocab`

- `248320` — clef, qwen38, qwen36
- `128896` — step37-pruned, step37-official
- `6400` — minimind3 ← **独苗**
- `50257` — gpt2 ← **独苗**
- `50304` — instella3b ← **独苗**
- `262144` — gemma31b ← **独苗**
- `100352` — laguna ← **独苗**
- `201088` — gptoss120b ← **独苗**

### `参数`

- `68827392` — minimind3 ← **独苗**
- `163037184` — gpt2 ← **独苗**
- `3112497280` — instella3b ← **独苗**
- `26895998464` — clef ← **独苗**
- `27268253696` — qwen38 ← **独苗**
- `32105986304` — gemma31b ← **独苗**
- `33442617088` — laguna ← **独苗**
- `35496856704` — qwen36 ← **独苗**
- `116829156672` — gptoss120b ← **独苗**
- `180845807680` — step37-pruned ← **独苗**
- `196956130368` — step37-official ← **独苗**

### `机制`

- `'Attention×45+FFN×3+MoE×42'` — step37-pruned, step37-official
- `'Attention×8+FFN×8'` — minimind3 ← **独苗**
- `'Attention×12+FFN×12'` — gpt2 ← **独苗**
- `'Attention×36+FFN×36'` — instella3b ← **独苗**
- `'Attention×16+FFN×64+Linear×48'` — clef ← **独苗**
- `'Attention×17+FFN×65+Linear×48'` — qwen38 ← **独苗**
- `'Attention×60+FFN×60'` — gemma31b ← **独苗**
- `'Attention×40+FFN×1+MoE×39'` — laguna ← **独苗**
- `'Attention×11+Linear×30+MoE×41'` — qwen36 ← **独苗**
- `'Attention×36+MoE×36'` — gptoss120b ← **独苗**

