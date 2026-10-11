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
| `gpt-oss-20b` | `gpt-oss-20b` | [openai/gpt-oss-20b](https://hf-mirror.com/openai/gpt-oss-20b) |
| `Qwen3-VL-2B-Thinking` | `Qwen3-VL-2B-Thinking` | [Qwen/Qwen3-VL-2B-Thinking](https://hf-mirror.com/Qwen/Qwen3-VL-2B-Thinking) |
| `Qwen3-VL-30B-A3B-Thinking` | `Qwen3-VL-30B-A3B-Thinking` | [Qwen/Qwen3-VL-30B-A3B-Thinking](https://hf-mirror.com/Qwen/Qwen3-VL-30B-A3B-Thinking) |
| `Qwen3-VL-235B-A22B-Thinking` | `Qwen3-VL-235B-A22B-Thinking` | [Qwen/Qwen3-VL-235B-A22B-Thinking](https://hf-mirror.com/Qwen/Qwen3-VL-235B-A22B-Thinking) |
| `Qwen3-Omni-30B-A3B-Thinking` | `Qwen3-Omni-30B-A3B-Thinking` | [Qwen/Qwen3-Omni-30B-A3B-Thinking](https://hf-mirror.com/Qwen/Qwen3-Omni-30B-A3B-Thinking) |
| `DeepSeek-R1-Distill-Qwen-1.5B` | `DeepSeek-R1-Distill-Qwen-1.5B` | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://hf-mirror.com/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |
| `Qwen3.5-0.8B` | `Qwen3.5-0.8B` | [Qwen/Qwen3.5-0.8B](https://hf-mirror.com/Qwen/Qwen3.5-0.8B) |
| `gemma-4-12B` | `gemma-4-12B` | [google/gemma-4-12B](https://hf-mirror.com/google/gemma-4-12B) |
| `mamba-130m-hf` | `mamba-130m-hf` | [state-spaces/mamba-130m-hf](https://hf-mirror.com/state-spaces/mamba-130m-hf) |
| `Qwen3.5-9B` | `Qwen3.5-9B` | [Qwen/Qwen3.5-9B](https://hf-mirror.com/Qwen/Qwen3.5-9B) |
| `Qwen3.5-397B-A17B` | `Qwen3.5-397B-A17B` | [Qwen/Qwen3.5-397B-A17B](https://hf-mirror.com/Qwen/Qwen3.5-397B-A17B) |
| `Qwen3-Coder-Next` | `Qwen3-Coder-Next` | [Qwen/Qwen3-Coder-Next](https://hf-mirror.com/Qwen/Qwen3-Coder-Next) |
| `DeepSeek-R1-Distill-Qwen-7B` | `DeepSeek-R1-Distill-Qwen-7B` | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://hf-mirror.com/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) |
| `Qwen3.8-Flash-Next` | `Qwen3.8-Flash-Next` | [Qwen/Qwen3.8-Flash-Next](https://hf-mirror.com/Qwen/Qwen3.8-Flash-Next) |
| `gemma-4-26B-A4B` | `gemma-4-26B-A4B` | [google/gemma-4-26B-A4B](https://hf-mirror.com/google/gemma-4-26B-A4B) |
| `GLM-4.7-Flash` | `GLM-4.7-Flash` | [zai-org/GLM-4.7-Flash](https://hf-mirror.com/zai-org/GLM-4.7-Flash) |

## 一、谁是谁

| 模型 | 参数 | 层 | d_model | 词表 | 机制 |
|---|---|---|---|---|---|
| `AttnFusedShaped` | 361856 | 2 | 128 | 256 | Attn×2+FFN×2 |
| `MoERoutingShaped` | 363152 | 2 | 128 | 256 | Attn×2+MoE×2 |
| `KDA-lowrank-shaped` | 686800 | 4 | 128 | 512 | FFN×4+KDA×4 |
| `KDA-GVA-shaped` | 693712 | 4 | 128 | 512 | FFN×4+KDA×4 |
| `SSM-shaped` | 695472 | 4 | 128 | 512 | FFN×4+SSM×4 |
| `KDA-shaped` | 727504 | 4 | 128 | 512 | FFN×4+KDA×4 |
| `GLM-text-tiny` | 1M | 4 | 256 | 512 | FFN×3+KDA×3+MLA×1+MoE×1 |
| `Ling-tiny` | 1M | 4 | 256 | 512 | FFN×1+KDA×3+MLA×1+MoE×3 |
| `MLA-sparse-shaped` | 3M | 4 | 256 | 512 | FFN×4+MLA×4 |
| `minimind-3` | 69M | 8 | 768 | 6400 | Attn×8+FFN×8 |
| `gpt2` | 163M | 12 | 768 | 50257 | Attn×12+FFN×12 |
| `Qwen3.5-0.8B` | 983M | 37 | 1024 | 248320 | Attn×19+FFN×37+Lin×18 |
| `DeepSeek-R1-Distill-Qwen-1.5B` | 1.8B | 28 | 1536 | 151936 | Attn×28+FFN×28 |
| `Qwen3-VL-2B-Thinking` | 2.0B | 52 | 2048 | 151936 | Attn×52+FFN×52 |
| `Instella-3B` | 3.1B | 36 | 2560 | 50304 | Attn×36+FFN×36 |
| `DeepSeek-R1-Distill-Qwen-7B` | 7.6B | 28 | 3584 | 152064 | Attn×28+FFN×28 |
| `Ling-3.0-tiny` | 7.9B | 24 | 1536 | 157184 | FFN×1+KDA×18+MLA×6+MoE×23 |
| `Qwen3.5-9B` | 8.1B | 60 | 4096 | 248320 | Attn×36+FFN×60+Lin×24 |
| `gemma-4-12B` | 12.9B | 48 | 3840 | 262144 | Attn×48+FFN×48 |
| `gpt-oss-20b` | 20.9B | 24 | 2880 | 201088 | Attn×24+MoE×24 |
| `gemma-4-26B-A4B` | 25.1B | 57 | 2816 | 262144 | Attn×57+FFN×27+MoE×30 |
| `clef` | 25.3B | 91 | 5120 | 248320 | Attn×43+FFN×91+Lin×48 |
| `Qwen3.8-27B` | 25.7B | 92 | 5120 | 248320 | Attn×44+FFN×92+Lin×48 |
| `gemma-4-31B` | 30.4B | 87 | 5376 | 262144 | Attn×87+FFN×87 |
| `GLM-4.7-Flash` | 30.6B | 48 | 2048 | 154880 | FFN×1+MLA×48+MoE×47 |
| `Qwen3-VL-30B-A3B-Thinking` | 30.7B | 75 | 2048 | 151936 | Attn×75+FFN×27+MoE×48 |
| `Qwen3-Omni-30B-A3B-Thinking` | 31.3B | 107 | 2048 | 152064 | Attn×107+FFN×59+MoE×48 |
| `Laguna-XS-2_1` | 33.4B | 40 | 2048 | 100352 | Attn×40+FFN×1+MoE×39 |
| `Qwen3.6-35B-A3B` | 35.5B | 68 | 2048 | 248320 | Attn×38+FFN×27+Lin×30+MoE×41 |
| `Nemotron-3.5-Lightning` | 48.5B | 54 | 2688 | 131072 | Attn×7+MoE×24+SSM×23 |
| `Qwen3-Coder-Next` | 79.7B | 48 | 2048 | 151936 | Attn×12+Lin×36+MoE×48 |
| `gpt-oss-120b` | 116.8B | 36 | 2880 | 201088 | Attn×36+MoE×36 |
| `Qwen3.8-Flash-Next` | 127.3B | 76 | 2560 | 248320 | Attn×40+FFN×27+Lin×36+MoE×49 |
| `nerkyor/Step-3_7-Flash` | 180.8B | 45 | 4096 | 128896 | Attn×45+FFN×3+MoE×42 |
| `Step-3.7-Flash` | 198.8B | 95 | 4096 | 128896 | Attn×95+FFN×53+MoE×42 |
| `Qwen3-VL-235B-A22B-Thinking` | 234.6B | 121 | 4096 | 151936 | Attn×121+FFN×27+MoE×94 |
| `GLM-5.3-Flash` | 320.1B | 70 | 4096 | 154880 | Attn×24+FFN×27+KDA×34+MLA×12+MoE×43 |
| `Qwen3.5-397B-A17B` | 401.9B | 88 | 4096 | 248320 | Attn×43+FFN×27+Lin×45+MoE×61 |

## 二、**独苗** —— 只有它这样的

**独苗才是需要新原语的地方。** 别的都只是参数不同。

| 属性 | 取值 | 只有 |
|---|---|---|
| `Attention.qk_norm` | `'full'` | **Instella-3B** |
| `Attention.rope` | `False` | **gpt2** |
| `Attention.rope_scale` | `1.3465735902799727` | **Laguna-XS-2_1** |
| `Attention.yarn` | `{'type': 'yarn', 'factor': 32.0, 'original': 8192, 'beta_fast': 64.0, 'beta_slow': 1.0, 'high_freq': 4.0, 'low_freq': 1.0, 'truncate': True}` | **Laguna-XS-2_1** |
| `FFN.act` | `'gelu_new'` | **gpt2** |
| `KDA.gate_rank` | `8` | **GLM-text-tiny** |
| `KDA.k_heads` | `16` | **Ling-3.0-tiny** |
| `KDA.k_heads` | `2` | **KDA-GVA-shaped** |
| `KDA.k_heads` | `64` | **GLM-5.3-Flash** |
| `KDA.v_heads` | `16` | **Ling-3.0-tiny** |
| `KDA.v_heads` | `64` | **GLM-5.3-Flash** |
| `MLA.head_dim` | `192` | **Ling-3.0-tiny** |
| `MLA.head_dim` | `32` | **GLM-text-tiny** |
| `MLA.index_dim` | `128` | **GLM-5.3-Flash** |
| `MLA.index_dim` | `16` | **GLM-text-tiny** |
| `MLA.index_dim` | `32` | **MLA-sparse-shaped** |
| `MLA.index_heads` | `32` | **GLM-5.3-Flash** |
| `MLA.index_topk` | `2048` | **GLM-5.3-Flash** |
| `MLA.kv_lora` | `128` | **MLA-sparse-shaped** |
| `MLA.kv_lora` | `16` | **GLM-text-tiny** |
| `MLA.kv_lora` | `32` | **Ling-tiny** |
| `MLA.out_dim` | `16384` | **GLM-5.3-Flash** |
| `MLA.out_dim` | `2048` | **Ling-3.0-tiny** |
| `MLA.out_dim` | `5120` | **GLM-4.7-Flash** |
| `MLA.out_dim` | `64` | **Ling-tiny** |
| `MLA.q` | `16` | **Ling-3.0-tiny** |
| `MLA.q` | `20` | **GLM-4.7-Flash** |
| `MLA.q` | `64` | **GLM-5.3-Flash** |
| `MLA.q_lora` | `1536` | **GLM-5.3-Flash** |
| `MLA.q_lora` | `256` | **Ling-3.0-tiny** |
| `MLA.q_lora` | `48` | **GLM-text-tiny** |
| `MLA.q_lora` | `64` | **Ling-tiny** |
| `MLA.q_lora` | `768` | **GLM-4.7-Flash** |
| `MLA.q_lora` | `96` | **MLA-sparse-shaped** |
| `MLA.qk_nope` | `128` | **Ling-3.0-tiny** |
| `MLA.qk_nope` | `192` | **GLM-4.7-Flash** |
| `MLA.qk_nope` | `256` | **GLM-5.3-Flash** |
| `MLA.qk_nope` | `40` | **MLA-sparse-shaped** |
| `MLA.qk_rope` | `16` | **Ling-tiny** |
| `MLA.qk_rope` | `8` | **MLA-sparse-shaped** |
| `MLA.rope_base` | `1000000` | **GLM-4.7-Flash** |
| `MLA.v_dim` | `128` | **Ling-3.0-tiny** |
| `MLA.v_dim` | `16` | **Ling-tiny** |
| `MoE.act` | `'relu2'` | **Nemotron-3.5-Lightning** |
| `MoE.n_group` | `4` | **Ling-tiny** |
| `MoE.topk_group` | `2` | **Ling-tiny** |
| `MoE.topk_group` | `4` | **Ling-3.0-tiny** |
| `SSM.chunk_size` | `128` | **Nemotron-3.5-Lightning** |
| `SSM.chunk_size` | `8` | **SSM-shaped** |
| `SSM.head_dim` | `16` | **SSM-shaped** |
| `SSM.head_dim` | `64` | **Nemotron-3.5-Lightning** |
| `SSM.heads` | `4` | **SSM-shaped** |
| `SSM.heads` | `64` | **Nemotron-3.5-Lightning** |
| `SSM.n_groups` | `2` | **SSM-shaped** |
| `SSM.n_groups` | `8` | **Nemotron-3.5-Lightning** |
| `SSM.ssm_state` | `128` | **Nemotron-3.5-Lightning** |
| `SSM.ssm_state` | `32` | **SSM-shaped** |
| `norm_kind` | `'layer'` | **gpt2** |
| `pos_kind` | `'learned'` | **gpt2** |

**38 个模型里，32 个一个独苗都没有**：attn-fused-shaped, moe-routing-shaped, kda-lowrank-shaped, kda-gva-shaped, ssm-shaped, kda-shaped, glm-text-tiny, ling-tiny, mla-sparse-shaped, minimind-3, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Qwen3.6-35B-A3B, NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, Step-3_7-Flash-180B-LynnStyle-GLM52-SFT-GPT55-RL, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B

> 这就是"原语集合收敛"这句话的可数形式 ——
> 说"收敛了"是感觉，说"这几个模型贡献 0 个新原语"是账。

> 而独苗集中在 **gpt2**（另一个时代）和 **gpt-oss-120b**
> （最特殊的一个）—— **新原语是从"另一个时代"和"最特殊的那个"
> 来的，不是从"又一个 Llama 变体"来的。**

## 三、逐属性的全部取值

### `Attention.bias`

- `False` — MoERoutingShaped, minimind-3, Qwen3.5-0.8B, Qwen3-VL-2B-Thinking, Instella-3B, Qwen3.5-9B, gemma-4-12B, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash
- `True` — AttnFusedShaped, gpt2, DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, gpt-oss-20b, gpt-oss-120b, GLM-5.3-Flash

### `Attention.gate_act`

- `'softplus'` — AttnFusedShaped, MoERoutingShaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash

### `Attention.head_dim`

- `256` — Qwen3.5-0.8B, Qwen3.5-9B, gemma-4-12B, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3.6-35B-A3B, Qwen3-Coder-Next, Qwen3.8-Flash-Next, Qwen3.5-397B-A17B
- `128` — DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, DeepSeek-R1-Distill-Qwen-7B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Nemotron-3.5-Lightning, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash
- `64` — gpt2, gpt-oss-20b, gpt-oss-120b, GLM-5.3-Flash
- `32` — AttnFusedShaped, MoERoutingShaped
- `96` — minimind-3 ← **独苗**
- `80` — Instella-3B ← **独苗**

### `Attention.head_gate`

- `'off'` — AttnFusedShaped, MoERoutingShaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash
- `'per_head'` — Laguna-XS-2_1, nerkyor/Step-3_7-Flash, Step-3.7-Flash

### `Attention.kv`

- `2` — AttnFusedShaped, MoERoutingShaped, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, Qwen3.8-Flash-Next, Qwen3.5-397B-A17B
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash
- `4` — minimind-3, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, clef, Qwen3.8-27B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Qwen3-VL-235B-A22B-Thinking
- `8` — Qwen3-VL-2B-Thinking, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, Laguna-XS-2_1, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `16` — gemma-4-31B, GLM-5.3-Flash
- `12` — gpt2 ← **独苗**
- `32` — Instella-3B ← **独苗**

### `Attention.kv_tie`

- `False` — AttnFusedShaped, MoERoutingShaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash

### `Attention.norm_eps`

- `1e-06` — minimind-3, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next, Qwen3.8-Flash-Next, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `1e-05` — AttnFusedShaped, MoERoutingShaped, gpt2, Instella-3B, gpt-oss-20b, Nemotron-3.5-Lightning, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash, GLM-5.3-Flash
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash

### `Attention.norm_one_plus`

- `False` — AttnFusedShaped, MoERoutingShaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash

### `Attention.out_dim`

- `4096` — Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash
- `8192` — gemma-4-31B, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `6144` — clef, Qwen3.8-27B, Laguna-XS-2_1, Qwen3.8-Flash-Next
- `128` — AttnFusedShaped, MoERoutingShaped
- `768` — minimind-3, gpt2
- `2048` — Qwen3.5-0.8B, Qwen3-VL-2B-Thinking
- `1536` — DeepSeek-R1-Distill-Qwen-1.5B ← **独苗**
- `2560` — Instella-3B ← **独苗**
- `3584` — DeepSeek-R1-Distill-Qwen-7B ← **独苗**
- `1024` — GLM-5.3-Flash ← **独苗**

### `Attention.q`

- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash
- `16` — Qwen3-VL-2B-Thinking, Qwen3.5-9B, gemma-4-12B, gemma-4-26B-A4B, Qwen3.6-35B-A3B, Qwen3-Coder-Next, GLM-5.3-Flash
- `32` — Instella-3B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Nemotron-3.5-Lightning, Qwen3.5-397B-A17B
- `64` — gpt-oss-20b, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking
- `24` — clef, Qwen3.8-27B, Qwen3.8-Flash-Next
- `4` — AttnFusedShaped, MoERoutingShaped
- `8` — minimind-3, Qwen3.5-0.8B
- `12` — gpt2, DeepSeek-R1-Distill-Qwen-1.5B
- `28` — DeepSeek-R1-Distill-Qwen-7B ← **独苗**
- `48` — Laguna-XS-2_1 ← **独苗**

### `Attention.q_gate`

- `False` — AttnFusedShaped, MoERoutingShaped, minimind-3, gpt2, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Nemotron-3.5-Lightning, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash
- `True` — Qwen3.5-0.8B, Qwen3.5-9B, clef, Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3-Coder-Next, Qwen3.8-Flash-Next, Qwen3.5-397B-A17B

### `Attention.qk_norm`

- `'per_head'` — minimind-3, Qwen3.5-0.8B, Qwen3-VL-2B-Thinking, Qwen3.5-9B, gemma-4-12B, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `'off'` — AttnFusedShaped, MoERoutingShaped, gpt2, DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, gpt-oss-20b, Nemotron-3.5-Lightning, gpt-oss-120b, GLM-5.3-Flash
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash
- `'full'` — Instella-3B ← **独苗**

### `Attention.qkv_fused`

- `False` — MoERoutingShaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash
- `True` — AttnFusedShaped, GLM-5.3-Flash

### `Attention.rope`

- `True` — AttnFusedShaped, MoERoutingShaped, minimind-3, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash
- `False` — gpt2 ← **独苗**

### `Attention.rope_base`

- `10000` — AttnFusedShaped, MoERoutingShaped, gpt2, DeepSeek-R1-Distill-Qwen-1.5B, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, gemma-4-12B, gemma-4-26B-A4B, gemma-4-31B, Nemotron-3.5-Lightning, GLM-5.3-Flash
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash
- `10000000` — Qwen3.5-0.8B, Qwen3.5-9B, clef, Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3.8-Flash-Next, Qwen3.5-397B-A17B
- `5000000` — Qwen3-VL-2B-Thinking, Qwen3-VL-30B-A3B-Thinking, Qwen3-Coder-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking
- `1000000` — minimind-3, Qwen3-Omni-30B-A3B-Thinking
- `150000` — gpt-oss-20b, gpt-oss-120b
- `500000` — Laguna-XS-2_1 ← **独苗**

### `Attention.rope_pairing`

- `'half'` — AttnFusedShaped, MoERoutingShaped, minimind-3, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, clef, Qwen3.8-27B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash
- `'interleaved'` — gpt2, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, gemma-4-31B, gpt-oss-120b

### `Attention.rope_partial`

- `1.0` — AttnFusedShaped, MoERoutingShaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash
- `0.5` — Laguna-XS-2_1, nerkyor/Step-3_7-Flash, Step-3.7-Flash

### `Attention.rope_scale`

- `1.0` — AttnFusedShaped, MoERoutingShaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash
- `1.3465735902799727` — Laguna-XS-2_1 ← **独苗**

### `Attention.sink`

- `False` — AttnFusedShaped, MoERoutingShaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash
- `True` — gpt-oss-20b, gpt-oss-120b

### `Attention.window`

- `None` — AttnFusedShaped, MoERoutingShaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, clef, Qwen3.8-27B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash
- `1024` — gemma-4-12B, gemma-4-26B-A4B, gemma-4-31B
- `128` — gpt-oss-20b, gpt-oss-120b

### `Attention.yarn`

- `None` — AttnFusedShaped, MoERoutingShaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, Qwen3.8-Flash-Next, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'—'` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash
- `{'type': 'yarn', 'factor': 32.0, 'original': 4096, 'beta_fast': 32.0, 'beta_slow': 1.0, 'high_freq': 4.0, 'low_freq': 1.0, 'truncate': False}` — gpt-oss-20b, gpt-oss-120b
- `{'type': 'llama3', 'factor': 2.0, 'original': 131072, 'beta_fast': 32.0, 'beta_slow': 1.0, 'high_freq': 32.0, 'low_freq': 1.0, 'truncate': True}` — nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `{'type': 'yarn', 'factor': 32.0, 'original': 8192, 'beta_fast': 64.0, 'beta_slow': 1.0, 'high_freq': 4.0, 'low_freq': 1.0, 'truncate': True}` — Laguna-XS-2_1 ← **独苗**

### `FFN.act`

- `'silu'` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'—'` — MoERoutingShaped, gpt-oss-20b, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b
- `'gelu_new'` — gpt2 ← **独苗**

### `FFN.alpha`

- `1.702` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'—'` — MoERoutingShaped, gpt-oss-20b, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b

### `FFN.bias`

- `False` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Laguna-XS-2_1, nerkyor/Step-3_7-Flash, Step-3.7-Flash, GLM-5.3-Flash
- `True` — gpt2, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Qwen3.6-35B-A3B, Qwen3.8-Flash-Next, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `'—'` — MoERoutingShaped, gpt-oss-20b, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b

### `FFN.gate`

- `True` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Laguna-XS-2_1, nerkyor/Step-3_7-Flash, Step-3.7-Flash, GLM-5.3-Flash
- `False` — gpt2, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Qwen3.6-35B-A3B, Qwen3.8-Flash-Next, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `'—'` — MoERoutingShaped, gpt-oss-20b, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b

### `FFN.hidden`

- `4304` — gemma-4-26B-A4B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Qwen3.6-35B-A3B, Qwen3.8-Flash-Next, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `256` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped
- `'—'` — MoERoutingShaped, gpt-oss-20b, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b
- `128` — GLM-text-tiny, Ling-tiny
- `12288` — Qwen3.5-9B, GLM-5.3-Flash
- `17408` — clef, Qwen3.8-27B
- `11264` — nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `512` — MLA-sparse-shaped ← **独苗**
- `2432` — minimind-3 ← **独苗**
- `3072` — gpt2 ← **独苗**
- `3584` — Qwen3.5-0.8B ← **独苗**
- `8960` — DeepSeek-R1-Distill-Qwen-1.5B ← **独苗**
- `6144` — Qwen3-VL-2B-Thinking ← **独苗**
- `6912` — Instella-3B ← **独苗**
- `18944` — DeepSeek-R1-Distill-Qwen-7B ← **独苗**
- `4608` — Ling-3.0-tiny ← **独苗**
- `15360` — gemma-4-12B ← **独苗**
- `21504` — gemma-4-31B ← **独苗**
- `10240` — GLM-4.7-Flash ← **独苗**
- `8192` — Laguna-XS-2_1 ← **独苗**

### `FFN.limit`

- `None` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'—'` — MoERoutingShaped, gpt-oss-20b, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b

### `KDA.act`

- `'—'` — AttnFusedShaped, MoERoutingShaped, SSM-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `'silu'` — KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, Ling-3.0-tiny, GLM-5.3-Flash

### `KDA.conv_bias`

- `'—'` — AttnFusedShaped, MoERoutingShaped, SSM-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `False` — KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, Ling-3.0-tiny, GLM-5.3-Flash

### `KDA.conv_kernel`

- `'—'` — AttnFusedShaped, MoERoutingShaped, SSM-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `4` — KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, Ling-3.0-tiny, GLM-5.3-Flash

### `KDA.gate_lower`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `-5.0` — KDA-shaped, GLM-text-tiny, Ling-tiny, Ling-3.0-tiny, GLM-5.3-Flash

### `KDA.gate_lowrank`

- `'—'` — AttnFusedShaped, MoERoutingShaped, SSM-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `False` — KDA-GVA-shaped, KDA-shaped, Ling-tiny, Ling-3.0-tiny
- `True` — KDA-lowrank-shaped, GLM-text-tiny, GLM-5.3-Flash

### `KDA.gate_out_bias`

- `'—'` — AttnFusedShaped, MoERoutingShaped, SSM-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `True` — KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, Ling-tiny, Ling-3.0-tiny
- `False` — GLM-text-tiny, GLM-5.3-Flash

### `KDA.gate_rank`

- `'—'` — AttnFusedShaped, MoERoutingShaped, SSM-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `16` — KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, Ling-tiny
- `128` — Ling-3.0-tiny, GLM-5.3-Flash
- `8` — GLM-text-tiny ← **独苗**

### `KDA.k_dim`

- `'—'` — AttnFusedShaped, MoERoutingShaped, SSM-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `16` — KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny
- `128` — Ling-3.0-tiny, GLM-5.3-Flash

### `KDA.k_heads`

- `'—'` — AttnFusedShaped, MoERoutingShaped, SSM-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `4` — KDA-lowrank-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny
- `2` — KDA-GVA-shaped ← **独苗**
- `16` — Ling-3.0-tiny ← **独苗**
- `64` — GLM-5.3-Flash ← **独苗**

### `KDA.l2_eps`

- `'—'` — AttnFusedShaped, MoERoutingShaped, SSM-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `1e-06` — KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, Ling-3.0-tiny, GLM-5.3-Flash

### `KDA.norm_eps`

- `'—'` — AttnFusedShaped, MoERoutingShaped, SSM-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `1e-05` — KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, GLM-5.3-Flash
- `1e-06` — Ling-tiny, Ling-3.0-tiny

### `KDA.v_dim`

- `'—'` — AttnFusedShaped, MoERoutingShaped, SSM-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `16` — KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny
- `128` — Ling-3.0-tiny, GLM-5.3-Flash

### `KDA.v_heads`

- `'—'` — AttnFusedShaped, MoERoutingShaped, SSM-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `4` — KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny
- `16` — Ling-3.0-tiny ← **独苗**
- `64` — GLM-5.3-Flash ← **独苗**

### `Linear.act`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Nemotron-3.5-Lightning, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash
- `'silu'` — Qwen3.5-0.8B, Qwen3.5-9B, clef, Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3-Coder-Next, Qwen3.8-Flash-Next, Qwen3.5-397B-A17B

### `Linear.conv_kernel`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Nemotron-3.5-Lightning, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash
- `4` — Qwen3.5-0.8B, Qwen3.5-9B, clef, Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3-Coder-Next, Qwen3.8-Flash-Next, Qwen3.5-397B-A17B

### `Linear.k_dim`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Nemotron-3.5-Lightning, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash
- `128` — Qwen3.5-0.8B, Qwen3.5-9B, clef, Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3-Coder-Next, Qwen3.8-Flash-Next, Qwen3.5-397B-A17B

### `Linear.k_heads`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Nemotron-3.5-Lightning, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash
- `16` — Qwen3.5-0.8B, Qwen3.5-9B, clef, Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3-Coder-Next, Qwen3.8-Flash-Next, Qwen3.5-397B-A17B

### `Linear.l2_eps`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Nemotron-3.5-Lightning, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash
- `1e-06` — Qwen3.5-0.8B, Qwen3.5-9B, clef, Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3-Coder-Next, Qwen3.8-Flash-Next, Qwen3.5-397B-A17B

### `Linear.norm_eps`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Nemotron-3.5-Lightning, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash
- `1e-06` — Qwen3.5-0.8B, Qwen3.5-9B, clef, Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3-Coder-Next, Qwen3.8-Flash-Next, Qwen3.5-397B-A17B

### `Linear.out_dim`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Nemotron-3.5-Lightning, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash
- `4096` — Qwen3.5-9B, Qwen3.6-35B-A3B, Qwen3-Coder-Next
- `6144` — clef, Qwen3.8-27B, Qwen3.8-Flash-Next
- `2048` — Qwen3.5-0.8B ← **独苗**
- `8192` — Qwen3.5-397B-A17B ← **独苗**

### `Linear.v_dim`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Nemotron-3.5-Lightning, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash
- `128` — Qwen3.5-0.8B, Qwen3.5-9B, clef, Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3-Coder-Next, Qwen3.8-Flash-Next, Qwen3.5-397B-A17B

### `Linear.v_heads`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Nemotron-3.5-Lightning, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash
- `32` — Qwen3.5-9B, Qwen3.6-35B-A3B, Qwen3-Coder-Next
- `48` — clef, Qwen3.8-27B, Qwen3.8-Flash-Next
- `16` — Qwen3.5-0.8B ← **独苗**
- `64` — Qwen3.5-397B-A17B ← **独苗**

### `MLA.bias`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `False` — GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash, GLM-5.3-Flash

### `MLA.gate_act`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `'sigmoid'` — GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash, GLM-5.3-Flash

### `MLA.head_dim`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `48` — Ling-tiny, MLA-sparse-shaped
- `256` — GLM-4.7-Flash, GLM-5.3-Flash
- `32` — GLM-text-tiny ← **独苗**
- `192` — Ling-3.0-tiny ← **独苗**

### `MLA.head_gate`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `'off'` — GLM-text-tiny, GLM-4.7-Flash, GLM-5.3-Flash
- `'per_head'` — Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny

### `MLA.index_dim`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, Ling-tiny, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `16` — GLM-text-tiny ← **独苗**
- `32` — MLA-sparse-shaped ← **独苗**
- `128` — GLM-5.3-Flash ← **独苗**

### `MLA.index_heads`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, Ling-tiny, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `4` — GLM-text-tiny, MLA-sparse-shaped
- `32` — GLM-5.3-Flash ← **独苗**

### `MLA.index_kpool`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `1` — GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash, GLM-5.3-Flash

### `MLA.index_kpool_tail`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `False` — GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash, GLM-5.3-Flash

### `MLA.index_topk`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, Ling-tiny, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `4` — GLM-text-tiny, MLA-sparse-shaped
- `2048` — GLM-5.3-Flash ← **独苗**

### `MLA.kv_lora`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `512` — Ling-3.0-tiny, GLM-4.7-Flash, GLM-5.3-Flash
- `16` — GLM-text-tiny ← **独苗**
- `32` — Ling-tiny ← **独苗**
- `128` — MLA-sparse-shaped ← **独苗**

### `MLA.norm_eps`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `1e-05` — GLM-text-tiny, GLM-4.7-Flash, GLM-5.3-Flash
- `1e-06` — Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny

### `MLA.norm_one_plus`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `False` — GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash, GLM-5.3-Flash

### `MLA.out_dim`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `128` — GLM-text-tiny, MLA-sparse-shaped
- `64` — Ling-tiny ← **独苗**
- `2048` — Ling-3.0-tiny ← **独苗**
- `5120` — GLM-4.7-Flash ← **独苗**
- `16384` — GLM-5.3-Flash ← **独苗**

### `MLA.pairing`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `'interleaved'` — Ling-tiny, MLA-sparse-shaped, Ling-3.0-tiny, GLM-4.7-Flash
- `'half'` — GLM-text-tiny, GLM-5.3-Flash

### `MLA.q`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `4` — GLM-text-tiny, Ling-tiny, MLA-sparse-shaped
- `16` — Ling-3.0-tiny ← **独苗**
- `20` — GLM-4.7-Flash ← **独苗**
- `64` — GLM-5.3-Flash ← **独苗**

### `MLA.q_lora`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `48` — GLM-text-tiny ← **独苗**
- `64` — Ling-tiny ← **独苗**
- `96` — MLA-sparse-shaped ← **独苗**
- `256` — Ling-3.0-tiny ← **独苗**
- `768` — GLM-4.7-Flash ← **独苗**
- `1536` — GLM-5.3-Flash ← **独苗**

### `MLA.qk_nope`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `32` — GLM-text-tiny, Ling-tiny
- `40` — MLA-sparse-shaped ← **独苗**
- `128` — Ling-3.0-tiny ← **独苗**
- `192` — GLM-4.7-Flash ← **独苗**
- `256` — GLM-5.3-Flash ← **独苗**

### `MLA.qk_rope`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `0` — GLM-text-tiny, GLM-5.3-Flash
- `64` — Ling-3.0-tiny, GLM-4.7-Flash
- `16` — Ling-tiny ← **独苗**
- `8` — MLA-sparse-shaped ← **独苗**

### `MLA.rope_base`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `10000` — GLM-text-tiny, MLA-sparse-shaped, GLM-5.3-Flash
- `6000000` — Ling-tiny, Ling-3.0-tiny
- `1000000` — GLM-4.7-Flash ← **独苗**

### `MLA.v_dim`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `32` — GLM-text-tiny, MLA-sparse-shaped
- `256` — GLM-4.7-Flash, GLM-5.3-Flash
- `16` — Ling-tiny ← **独苗**
- `128` — Ling-3.0-tiny ← **独苗**

### `MoE.act`

- `'—'` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, clef, Qwen3.8-27B, gemma-4-31B
- `'silu'` — MoERoutingShaped, GLM-text-tiny, Ling-tiny, Ling-3.0-tiny, gemma-4-26B-A4B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'gptoss'` — gpt-oss-20b, gpt-oss-120b
- `'relu2'` — Nemotron-3.5-Lightning ← **独苗**

### `MoE.alpha`

- `1.702` — MoERoutingShaped, GLM-text-tiny, Ling-tiny, Ling-3.0-tiny, gpt-oss-20b, gemma-4-26B-A4B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'—'` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, clef, Qwen3.8-27B, gemma-4-31B

### `MoE.expert_bias`

- `'—'` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, clef, Qwen3.8-27B, gemma-4-31B
- `False` — MoERoutingShaped, GLM-text-tiny, Ling-tiny, Ling-3.0-tiny, gemma-4-26B-A4B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `True` — gpt-oss-20b, gpt-oss-120b

### `MoE.experts`

- `'—'` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, clef, Qwen3.8-27B, gemma-4-31B
- `128` — Ling-3.0-tiny, gemma-4-26B-A4B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Nemotron-3.5-Lightning, gpt-oss-120b, Qwen3-VL-235B-A22B-Thinking
- `8` — MoERoutingShaped, GLM-text-tiny, Ling-tiny
- `512` — Qwen3-Coder-Next, Qwen3.8-Flash-Next, Qwen3.5-397B-A17B
- `256` — Laguna-XS-2_1, Qwen3.6-35B-A3B
- `288` — Step-3.7-Flash, GLM-5.3-Flash
- `32` — gpt-oss-20b ← **独苗**
- `64` — GLM-4.7-Flash ← **独苗**
- `255` — nerkyor/Step-3_7-Flash ← **独苗**

### `MoE.hidden`

- `'—'` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, clef, Qwen3.8-27B, gemma-4-31B
- `512` — Ling-3.0-tiny, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next
- `32` — MoERoutingShaped, GLM-text-tiny, Ling-tiny
- `2880` — gpt-oss-20b, gpt-oss-120b
- `1536` — GLM-4.7-Flash, Qwen3-VL-235B-A22B-Thinking
- `768` — Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking
- `1280` — nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `704` — gemma-4-26B-A4B ← **独苗**
- `1856` — Nemotron-3.5-Lightning ← **独苗**
- `640` — Qwen3.8-Flash-Next ← **独苗**
- `2048` — GLM-5.3-Flash ← **独苗**
- `1024` — Qwen3.5-397B-A17B ← **独苗**

### `MoE.limit`

- `'—'` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, clef, Qwen3.8-27B, gemma-4-31B
- `None` — MoERoutingShaped, Ling-tiny, Ling-3.0-tiny, gemma-4-26B-A4B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `10.0` — GLM-text-tiny, GLM-5.3-Flash
- `7.0` — gpt-oss-20b, gpt-oss-120b

### `MoE.limit_shared`

- `None` — MoERoutingShaped, GLM-text-tiny, Ling-tiny, Ling-3.0-tiny, gpt-oss-20b, gemma-4-26B-A4B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'—'` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, clef, Qwen3.8-27B, gemma-4-31B

### `MoE.n_group`

- `'—'` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, clef, Qwen3.8-27B, gemma-4-31B
- `0` — MoERoutingShaped, gpt-oss-20b, gemma-4-26B-A4B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `1` — GLM-text-tiny, GLM-4.7-Flash, GLM-5.3-Flash
- `8` — Ling-3.0-tiny, Nemotron-3.5-Lightning
- `4` — Ling-tiny ← **独苗**

### `MoE.routed_scale`

- `'—'` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, clef, Qwen3.8-27B, gemma-4-31B
- `1.0` — gpt-oss-20b, gemma-4-26B-A4B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `2.5` — MoERoutingShaped, GLM-text-tiny, Ling-tiny, Ling-3.0-tiny, Laguna-XS-2_1, GLM-5.3-Flash
- `3.0` — nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `1.8` — GLM-4.7-Flash ← **独苗**

### `MoE.router_bias`

- `'—'` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, clef, Qwen3.8-27B, gemma-4-31B
- `False` — MoERoutingShaped, GLM-text-tiny, Ling-tiny, Ling-3.0-tiny, gemma-4-26B-A4B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, Qwen3.8-Flash-Next, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `True` — gpt-oss-20b, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash

### `MoE.routing`

- `'—'` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, clef, Qwen3.8-27B, gemma-4-31B
- `'softmax_topk'` — gemma-4-26B-A4B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next, Qwen3.8-Flash-Next, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `'sigmoid_group_topk'` — GLM-text-tiny, Ling-tiny, Ling-3.0-tiny, GLM-4.7-Flash, Nemotron-3.5-Lightning, GLM-5.3-Flash
- `'sigmoid_topk'` — MoERoutingShaped, nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `'topk_softmax'` — gpt-oss-20b, gpt-oss-120b

### `MoE.score_bias`

- `'—'` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, clef, Qwen3.8-27B, gemma-4-31B
- `False` — GLM-text-tiny, gpt-oss-20b, gemma-4-26B-A4B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Qwen3.6-35B-A3B, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `True` — MoERoutingShaped, Ling-tiny, Ling-3.0-tiny, GLM-4.7-Flash, Laguna-XS-2_1, Nemotron-3.5-Lightning

### `MoE.shared`

- `'—'` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, clef, Qwen3.8-27B, gemma-4-31B
- `1` — GLM-text-tiny, Ling-tiny, Ling-3.0-tiny, GLM-4.7-Flash, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `0` — MoERoutingShaped, gpt-oss-20b, gemma-4-26B-A4B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, gpt-oss-120b, Qwen3-VL-235B-A22B-Thinking

### `MoE.shared_gate`

- `'—'` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, clef, Qwen3.8-27B, gemma-4-31B
- `False` — MoERoutingShaped, GLM-text-tiny, Ling-tiny, Ling-3.0-tiny, gpt-oss-20b, gemma-4-26B-A4B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Nemotron-3.5-Lightning, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash
- `True` — Qwen3.6-35B-A3B, Qwen3-Coder-Next, Qwen3.8-Flash-Next, Qwen3.5-397B-A17B

### `MoE.shared_hidden`

- `'—'` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, clef, Qwen3.8-27B, gemma-4-31B
- `0` — MoERoutingShaped, gpt-oss-20b, gemma-4-26B-A4B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, gpt-oss-120b, Qwen3-VL-235B-A22B-Thinking
- `512` — Ling-3.0-tiny, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next
- `32` — GLM-text-tiny, Ling-tiny
- `1280` — nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `1536` — GLM-4.7-Flash ← **独苗**
- `3712` — Nemotron-3.5-Lightning ← **独苗**
- `640` — Qwen3.8-Flash-Next ← **独苗**
- `2048` — GLM-5.3-Flash ← **独苗**
- `1024` — Qwen3.5-397B-A17B ← **独苗**

### `MoE.top_k`

- `'—'` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, clef, Qwen3.8-27B, gemma-4-31B
- `8` — Ling-3.0-tiny, gemma-4-26B-A4B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash
- `2` — MoERoutingShaped, GLM-text-tiny, Ling-tiny
- `4` — gpt-oss-20b, GLM-4.7-Flash, gpt-oss-120b
- `10` — Qwen3-Coder-Next, Qwen3.8-Flash-Next, Qwen3.5-397B-A17B
- `6` — Nemotron-3.5-Lightning ← **独苗**

### `MoE.topk_group`

- `'—'` — AttnFusedShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B, gemma-4-12B, clef, Qwen3.8-27B, gemma-4-31B
- `0` — MoERoutingShaped, gpt-oss-20b, gemma-4-26B-A4B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `1` — GLM-text-tiny, GLM-4.7-Flash, Nemotron-3.5-Lightning, GLM-5.3-Flash
- `2` — Ling-tiny ← **独苗**
- `4` — Ling-3.0-tiny ← **独苗**

### `SSM.act`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'silu'` — SSM-shaped, Nemotron-3.5-Lightning

### `SSM.chunk_size`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `8` — SSM-shaped ← **独苗**
- `128` — Nemotron-3.5-Lightning ← **独苗**

### `SSM.conv_bias`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `True` — SSM-shaped, Nemotron-3.5-Lightning

### `SSM.conv_kernel`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `4` — SSM-shaped, Nemotron-3.5-Lightning

### `SSM.expand`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `2` — SSM-shaped, Nemotron-3.5-Lightning

### `SSM.head_dim`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `16` — SSM-shaped ← **独苗**
- `64` — Nemotron-3.5-Lightning ← **独苗**

### `SSM.heads`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `4` — SSM-shaped ← **独苗**
- `64` — Nemotron-3.5-Lightning ← **独苗**

### `SSM.n_groups`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `2` — SSM-shaped ← **独苗**
- `8` — Nemotron-3.5-Lightning ← **独苗**

### `SSM.norm_eps`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `1e-05` — SSM-shaped, Nemotron-3.5-Lightning

### `SSM.proj_bias`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `False` — SSM-shaped, Nemotron-3.5-Lightning

### `SSM.ssm_state`

- `'—'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, gpt2, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `32` — SSM-shaped ← **独苗**
- `128` — Nemotron-3.5-Lightning ← **独苗**

### `ctx`

- `262144` — Ling-tiny, Qwen3.5-0.8B, Qwen3-VL-2B-Thinking, Qwen3.5-9B, gemma-4-12B, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `1024` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, MLA-sparse-shaped, gpt2
- `131072` — DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, gpt-oss-20b, gpt-oss-120b
- `512` — AttnFusedShaped, MoERoutingShaped
- `1048576` — GLM-text-tiny, GLM-5.3-Flash
- `32768` — minimind-3 ← **独苗**
- `4096` — Instella-3B ← **独苗**
- `202752` — GLM-4.7-Flash ← **独苗**
- `65536` — Qwen3-Omni-30B-A3B-Thinking ← **独苗**

### `d_model`

- `2048` — Qwen3-VL-2B-Thinking, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next
- `128` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped
- `4096` — Qwen3.5-9B, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `256` — GLM-text-tiny, Ling-tiny, MLA-sparse-shaped
- `768` — minimind-3, gpt2
- `1536` — DeepSeek-R1-Distill-Qwen-1.5B, Ling-3.0-tiny
- `2560` — Instella-3B, Qwen3.8-Flash-Next
- `2880` — gpt-oss-20b, gpt-oss-120b
- `5120` — clef, Qwen3.8-27B
- `1024` — Qwen3.5-0.8B ← **独苗**
- `3584` — DeepSeek-R1-Distill-Qwen-7B ← **独苗**
- `3840` — gemma-4-12B ← **独苗**
- `2816` — gemma-4-26B-A4B ← **独苗**
- `5376` — gemma-4-31B ← **独苗**
- `2688` — Nemotron-3.5-Lightning ← **独苗**

### `n_layer`

- `4` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped
- `48` — gemma-4-12B, GLM-4.7-Flash, Qwen3-Coder-Next
- `2` — AttnFusedShaped, MoERoutingShaped
- `28` — DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B
- `36` — Instella-3B, gpt-oss-120b
- `24` — Ling-3.0-tiny, gpt-oss-20b
- `8` — minimind-3 ← **独苗**
- `12` — gpt2 ← **独苗**
- `37` — Qwen3.5-0.8B ← **独苗**
- `52` — Qwen3-VL-2B-Thinking ← **独苗**
- `60` — Qwen3.5-9B ← **独苗**
- `57` — gemma-4-26B-A4B ← **独苗**
- `91` — clef ← **独苗**
- `92` — Qwen3.8-27B ← **独苗**
- `87` — gemma-4-31B ← **独苗**
- `75` — Qwen3-VL-30B-A3B-Thinking ← **独苗**
- `107` — Qwen3-Omni-30B-A3B-Thinking ← **独苗**
- `40` — Laguna-XS-2_1 ← **独苗**
- `68` — Qwen3.6-35B-A3B ← **独苗**
- `54` — Nemotron-3.5-Lightning ← **独苗**
- `76` — Qwen3.8-Flash-Next ← **独苗**
- `45` — nerkyor/Step-3_7-Flash ← **独苗**
- `95` — Step-3.7-Flash ← **独苗**
- `121` — Qwen3-VL-235B-A22B-Thinking ← **独苗**
- `70` — GLM-5.3-Flash ← **独苗**
- `88` — Qwen3.5-397B-A17B ← **独苗**

### `norm_eps`

- `1e-06` — Ling-tiny, MLA-sparse-shaped, minimind-3, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Qwen3-Coder-Next, Qwen3.8-Flash-Next, Qwen3-VL-235B-A22B-Thinking, Qwen3.5-397B-A17B
- `1e-05` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, gpt2, Instella-3B, gpt-oss-20b, GLM-4.7-Flash, Nemotron-3.5-Lightning, gpt-oss-120b, nerkyor/Step-3_7-Flash, Step-3.7-Flash, GLM-5.3-Flash

### `norm_kind`

- `'rms'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'layer'` — gpt2 ← **独苗**

### `pos_kind`

- `'rope'` — AttnFusedShaped, MoERoutingShaped, KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped, minimind-3, Qwen3.5-0.8B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Instella-3B, DeepSeek-R1-Distill-Qwen-7B, Ling-3.0-tiny, Qwen3.5-9B, gemma-4-12B, gpt-oss-20b, gemma-4-26B-A4B, clef, Qwen3.8-27B, gemma-4-31B, GLM-4.7-Flash, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Thinking, Laguna-XS-2_1, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, Qwen3-Coder-Next, gpt-oss-120b, Qwen3.8-Flash-Next, nerkyor/Step-3_7-Flash, Step-3.7-Flash, Qwen3-VL-235B-A22B-Thinking, GLM-5.3-Flash, Qwen3.5-397B-A17B
- `'learned'` — gpt2 ← **独苗**

### `vocab`

- `512` — KDA-lowrank-shaped, KDA-GVA-shaped, SSM-shaped, KDA-shaped, GLM-text-tiny, Ling-tiny, MLA-sparse-shaped
- `248320` — Qwen3.5-0.8B, Qwen3.5-9B, clef, Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3.8-Flash-Next, Qwen3.5-397B-A17B
- `151936` — DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-VL-2B-Thinking, Qwen3-VL-30B-A3B-Thinking, Qwen3-Coder-Next, Qwen3-VL-235B-A22B-Thinking
- `262144` — gemma-4-12B, gemma-4-26B-A4B, gemma-4-31B
- `256` — AttnFusedShaped, MoERoutingShaped
- `152064` — DeepSeek-R1-Distill-Qwen-7B, Qwen3-Omni-30B-A3B-Thinking
- `201088` — gpt-oss-20b, gpt-oss-120b
- `154880` — GLM-4.7-Flash, GLM-5.3-Flash
- `128896` — nerkyor/Step-3_7-Flash, Step-3.7-Flash
- `6400` — minimind-3 ← **独苗**
- `50257` — gpt2 ← **独苗**
- `50304` — Instella-3B ← **独苗**
- `157184` — Ling-3.0-tiny ← **独苗**
- `100352` — Laguna-XS-2_1 ← **独苗**
- `131072` — Nemotron-3.5-Lightning ← **独苗**

### `参数`

- `361856` — AttnFusedShaped ← **独苗**
- `363152` — MoERoutingShaped ← **独苗**
- `686800` — KDA-lowrank-shaped ← **独苗**
- `693712` — KDA-GVA-shaped ← **独苗**
- `695472` — SSM-shaped ← **独苗**
- `727504` — KDA-shaped ← **独苗**
- `1067868` — GLM-text-tiny ← **独苗**
- `1397620` — Ling-tiny ← **独苗**
- `2535040` — MLA-sparse-shaped ← **独苗**
- `68827392` — minimind-3 ← **独苗**
- `163037184` — gpt2 ← **独苗**
- `982921280` — Qwen3.5-0.8B ← **独苗**
- `1777131008` — DeepSeek-R1-Distill-Qwen-1.5B ← **独苗**
- `2022834176` — Qwen3-VL-2B-Thinking ← **独苗**
- `3112497280` — Instella-3B ← **独苗**
- `7615716864` — DeepSeek-R1-Distill-Qwen-7B ← **独苗**
- `7893392800` — Ling-3.0-tiny ← **独苗**
- `8112820464` — Qwen3.5-9B ← **独苗**
- `12913614592` — gemma-4-12B ← **独苗**
- `20914757184` — gpt-oss-20b ← **独苗**
- `25108154544` — gemma-4-26B-A4B ← **独苗**
- `25336731376` — clef ← **独苗**
- `25708986608` — Qwen3.8-27B ← **独苗**
- `30436397744` — gemma-4-31B ← **独苗**
- `30578705344` — GLM-4.7-Flash ← **独苗**
- `30671256816` — Qwen3-VL-30B-A3B-Thinking ← **独苗**
- `31340076400` — Qwen3-Omni-30B-A3B-Thinking ← **独苗**
- `33442617088` — Laguna-XS-2_1 ← **独苗**
- `35463270768` — Qwen3.6-35B-A3B ← **独苗**
- `48464262848` — Nemotron-3.5-Lightning ← **独苗**
- `79674391296` — Qwen3-Coder-Next ← **独苗**
- `116829156672` — gpt-oss-120b ← **独苗**
- `127335531632` — Qwen3.8-Flash-Next ← **独苗**
- `180845807680` — nerkyor/Step-3_7-Flash ← **独苗**
- `198777931840` — Step-3.7-Flash ← **独苗**
- `234610436848` — Qwen3-VL-235B-A22B-Thinking ← **独苗**
- `320135571840` — GLM-5.3-Flash ← **独苗**
- `401857645040` — Qwen3.5-397B-A17B ← **独苗**

### `机制`

- `'FFN×4+KDA×4'` — KDA-lowrank-shaped, KDA-GVA-shaped, KDA-shaped
- `'Attention×28+FFN×28'` — DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B
- `'Attention×2+FFN×2'` — AttnFusedShaped ← **独苗**
- `'Attention×2+MoE×2'` — MoERoutingShaped ← **独苗**
- `'FFN×4+SSM×4'` — SSM-shaped ← **独苗**
- `'FFN×3+KDA×3+MLA×1+MoE×1'` — GLM-text-tiny ← **独苗**
- `'FFN×1+KDA×3+MLA×1+MoE×3'` — Ling-tiny ← **独苗**
- `'FFN×4+MLA×4'` — MLA-sparse-shaped ← **独苗**
- `'Attention×8+FFN×8'` — minimind-3 ← **独苗**
- `'Attention×12+FFN×12'` — gpt2 ← **独苗**
- `'Attention×19+FFN×37+Linear×18'` — Qwen3.5-0.8B ← **独苗**
- `'Attention×52+FFN×52'` — Qwen3-VL-2B-Thinking ← **独苗**
- `'Attention×36+FFN×36'` — Instella-3B ← **独苗**
- `'FFN×1+KDA×18+MLA×6+MoE×23'` — Ling-3.0-tiny ← **独苗**
- `'Attention×36+FFN×60+Linear×24'` — Qwen3.5-9B ← **独苗**
- `'Attention×48+FFN×48'` — gemma-4-12B ← **独苗**
- `'Attention×24+MoE×24'` — gpt-oss-20b ← **独苗**
- `'Attention×57+FFN×27+MoE×30'` — gemma-4-26B-A4B ← **独苗**
- `'Attention×43+FFN×91+Linear×48'` — clef ← **独苗**
- `'Attention×44+FFN×92+Linear×48'` — Qwen3.8-27B ← **独苗**
- `'Attention×87+FFN×87'` — gemma-4-31B ← **独苗**
- `'FFN×1+MLA×48+MoE×47'` — GLM-4.7-Flash ← **独苗**
- `'Attention×75+FFN×27+MoE×48'` — Qwen3-VL-30B-A3B-Thinking ← **独苗**
- `'Attention×107+FFN×59+MoE×48'` — Qwen3-Omni-30B-A3B-Thinking ← **独苗**
- `'Attention×40+FFN×1+MoE×39'` — Laguna-XS-2_1 ← **独苗**
- `'Attention×38+FFN×27+Linear×30+MoE×41'` — Qwen3.6-35B-A3B ← **独苗**
- `'Attention×7+MoE×24+SSM×23'` — Nemotron-3.5-Lightning ← **独苗**
- `'Attention×12+Linear×36+MoE×48'` — Qwen3-Coder-Next ← **独苗**
- `'Attention×36+MoE×36'` — gpt-oss-120b ← **独苗**
- `'Attention×40+FFN×27+Linear×36+MoE×49'` — Qwen3.8-Flash-Next ← **独苗**
- `'Attention×45+FFN×3+MoE×42'` — nerkyor/Step-3_7-Flash ← **独苗**
- `'Attention×95+FFN×53+MoE×42'` — Step-3.7-Flash ← **独苗**
- `'Attention×121+FFN×27+MoE×94'` — Qwen3-VL-235B-A22B-Thinking ← **独苗**
- `'Attention×24+FFN×27+KDA×34+MLA×12+MoE×43'` — GLM-5.3-Flash ← **独苗**
- `'Attention×43+FFN×27+Linear×45+MoE×61'` — Qwen3.5-397B-A17B ← **独苗**

