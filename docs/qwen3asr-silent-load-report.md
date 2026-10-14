# Qwen3-ASR-0.6B：transformers 会静默装载一个错误的模型

**对象**：`Qwen/Qwen3-ASR-0.6B`（checkpoint `transformers_version: 4.57.6`）
**对照产物**：`ggml-org/Qwen3-ASR-0.6B-GGUF` @ `928ab958557df9aa2ef1c93e0e83c7ad0933fae2`
**日期**：2026-10-14 · **环境**：Windows · CPU-only · transformers 5.15.1 与 4.57.6

---

## 结论

在这个 checkpoint 上，`transformers` 的 auto 类**不会报错，但会给你一个错误的模型**。

```python
m = AutoModel.from_pretrained("Qwen/Qwen3-ASR-0.6B")
```

返回的是**一个 2,039,626,880 参数、全部随机初始化**的模型，而不是发布方的预训练权重。
**没有异常、没有警告**，模型可以正常构造、正常前向。

## 它按什么建的 vs 它应该是什么

| | 模型类实际构造 | checkpoint / GGUF 的真实值 |
|---|---:|---:|
| audio tower 层数 | **24** | **18** |
| audio tower `d_model` | **1024** | **896** |
| audio tower FFN 维度 | **4096** | — |
| text `hidden_size` | **2048** | **1024** |
| text 层数 | 28 | 28 ✓ |

参数去向：

| 分量 | 参数量 | 占比 |
|---|---:|---:|
| `language_model` | 1,720,574,976 | 84.4 % |
| `audio_tower` | 314,328,704 | 15.4 % |
| `multi_modal_projector` | 4,723,200 | 0.2 % |
| **合计** | **2,039,626,880** | |

作为对照，GGUF 元数据里**真实的**参数量是：

| 产物 | 层数 | 维度 | 参数量 |
|---|---:|---:|---:|
| `Qwen3-ASR-0.6B-bf16.gguf`（text，`qwen3vl.*`） | 28 | 1024 | **751,632,384** |
| `mmproj-...-bf16.gguf`（audio，`clip.audio.*`） | 18 | 896 | **187,720,192** |

而 checkpoint 自身 612 个张量的元素总数合计为 **938,008,576**
（`thinker.model` 596,049,920 + `thinker.audio_tower` 186,376,192 +
`thinker.lm_head` 155,582,464）。**实际构造出的 2.04e9 是这个数的 2.17 倍。**

**装载报告**：`UNEXPECTED 43 组 / MISSING 42 组` —— 覆盖 `audio_tower.*`、
`language_model.*`、`multi_modal_projector.*` 全部权重，即
**checkpoint 的 612 个张量一个都没用上**。

## 两个独立的根因

### 根因 1：键前缀错位

checkpoint 的全部 612 个张量都带 `thinker.` 前缀：

```
thinker.audio_tower.conv2d1.weight   [480, 1, 3, 3]
thinker.model.layers.0.self_attn.q_proj.weight
```

而模型类的参数名没有这个前缀（`audio_tower.*` / `language_model.*` /
`multi_modal_projector.*`）。**612/612 全部落进 UNEXPECTED。**

层数在这一层是**对的**：checkpoint `thinker.audio_tower` = 18 层（与
`thinker_config.audio_config.num_hidden_layers = 18`、GGUF
`clip.audio.block_count = 18` 三方一致）；`thinker.model` = 28 层。

### 根因 2：配置读错了嵌套层级

`config.json` 把真实配置放在 `thinker_config` 下：

```
config.json
├── architectures, model_type, transformers_version   ← 顶层只有这些
└── thinker_config
    ├── audio_config   { num_hidden_layers: 18, d_model: 896 }
    └── text_config    { num_hidden_layers: 28, hidden_size: 1024 }
```

但类读的是**顶层**的 `audio_config` / `text_config`（顶层不存在），
于是**静默回退到默认值**：

| 字段 | 类读到的（默认） | 真实值 |
|---|---:|---:|
| `audio_config.num_hidden_layers` | **24** | **18** |
| `audio_config.d_model` | **1024** | **896** |
| `text_config.hidden_size` | **2048** | **1024** |

这解释了 `audio_tower.layers.{0...23}` 的 MISSING —— **多出来的 6 层是
按默认值新建的，checkpoint 里根本没有。**

两个根因**互相独立**：只修前缀，24 层的空壳依然接不上 18 层的权重；
只修配置，612 个张量依然因前缀全部落空。

## 更危险的是：升级会让它更难被发现

| transformers | 4.57.6 | 5.15.1 |
|---|---|---|
| `Qwen3ASR` 类 | **不存在** | 存在 |
| `AutoProcessor.from_pretrained()` | **静默返回 `Qwen2TokenizerFast`**（无 `feature_extractor`） | 正确返回 `Qwen3ASRProcessor` ✓ |
| `AutoModel.from_pretrained()` | — | 返回 `Qwen3ASRModel`，612 张量全未装载 |

在 4.57.6 下，`AutoProcessor` 那一步**至少会露出马脚**（拿到的不是处理器）。
在 5.15.1 下，处理器完全正确、类名完全正确、`AutoModel` 正常返回对象——
**没有任何一处提示你权重没装载**。升级把最容易发现异常的那一环修好了。

## 修正（已实测验证）

在 CPU 上完整跑通，`missing_keys = 0`、`unexpected_keys = 0`、
**612 个张量全部落位**。需要做三件事：

### 1. 显式键映射

checkpoint 名 → 模型参数名，规则三条：

```
thinker.<rest>                      →  <rest>            # 前缀
model.<rest>                        →  language_model.<rest>   # 注意外层是 model.
audio_tower.<rest>                  →  model.audio_tower.<rest>
audio_tower.proj1.<rest>            →  multi_modal_projector.linear_1.<rest>
audio_tower.proj2.<rest>            →  multi_modal_projector.linear_2.<rest>
```

注意 `Qwen3ASRForConditionalGeneration` 把两座塔包在 `model.` 下
（`model.language_model.*` / `model.audio_tower.*`），
而 `multi_modal_projector` 在 `model.` 内、`lm_head` 在外——层级不一致，
是第二处容易配错的地方。

### 2. 显式读嵌套配置

从 `thinker_config.audio_config` / `thinker_config.text_config` 取值。
不修则会静默用默认值，**6 个字段全部偏掉**：

| 字段 | 类读到的（默认） | 真实值 |
|---|---:|---:|
| audio `num_hidden_layers` | 24 | **18** |
| audio `d_model` | 1024 | **896** |
| audio `encoder_ffn_dim` | 4096 | **3584** |
| text `hidden_size` | 2048 | **1024** |
| text `intermediate_size` | 6144 | **3072** |
| text `rope_theta` | 10000.0 | **1000000** |

其中 `rope_theta` 尤其隐蔽：**10000 vs 1000000，差 100 倍。**
这一项不会导致形状不匹配、不会报错，**只会让位置编码全错**——
即使其它字段都修对了，漏掉它得到的仍是一个行为错误的模型。

### 3. 装载后校验

```python
info = model.load_state_dict(remapped, strict=False)
assert not info.missing_keys, info.missing_keys
assert not info.unexpected_keys, info.unexpected_keys
```

**验证结果**（`history/_hf_probe/fix_verify.py`）：

```
checkpoint tensors          : 612
no target in model          : 0
model params with no source : 0
SHAPE MISMATCHES            : 0
missing_keys                : 0
unexpected_keys             : 0
```

并逐位复核了 9 个抽样张量（两座塔的首层/末层/embed/norm/projector），
**全部 `identical=True`、`max|d| = 0`**——装进去的是 checkpoint 的原始权重。

参数总量为 **782,426,112**，与 checkpoint 张量求和 938,008,576 差
**155,582,464**，**恰好等于 `thinker.lm_head.weight`**：因
`tie_word_embeddings=True`，`lm_head` 与 `embed_tokens` 共享张量，
`.parameters()` 只计一次。**差值完全解释得通，无残留疑点。**

## 影响范围

任何用 `AutoModel` / `AutoProcessor` 直接加载该 checkpoint 做音频实验的流程，
**得到的都是随机权重下的结果**。用真实权重复现、评测或微调前，
应先确认 612 个张量全部落位。

## 复现

```bash
# 类是否存在、AutoProcessor / AutoModel 各自返回什么、装载报告
python history/_hf_probe/version_compare.py  model-cache/qwen3asr-hf

# 配置嵌套 vs 实际构造出的结构
python history/_hf_probe/config_mismatch.py  model-cache/qwen3asr-hf

# 装载失败后参数去了哪里
python history/_hf_probe/param_breakdown.py  model-cache/qwen3asr-hf

# checkpoint 键前缀与各栈层数
python history/_hf_probe/keys_dump.py

# checkpoint 自身的精确参数量（从 safetensors 头求和）
python history/_hf_probe/checkpoint_param_count.py

# 键映射后的装载校验（含逐位复核与参数差解释）
python history/_hf_probe/fix_verify.py  model-cache/qwen3asr-hf
python history/_hf_probe/fix_confirm.py model-cache/qwen3asr-hf

# GGUF 元数据里的层数 / 维度 / 参数量（无需 GPU）
python history/_hf_probe/gguf_param_count.py model-cache/audio/Qwen3-ASR-0.6B-bf16.gguf
python history/_hf_probe/gguf_param_count.py model-cache/audio/mmproj-Qwen3-ASR-0.6B-bf16.gguf
```

原始数据：`history/_hf_probe/summary.json`（本报告引用的全部数值）、
`history/_hf_probe/v5151_full.txt`（5.15.1 的完整装载报告）、
`history/_hf_probe/config_mismatch.txt`（配置嵌套实测）
—— 均为各次运行的直接输出，未手工改写。
