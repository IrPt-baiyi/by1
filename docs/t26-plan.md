# T26：第二架构毕业 —— mamba-130m-hf（选择性状态空间）

> 日期 **2026-10-16** · 路线图 T26 要求：选一个**机制、norm、bias 或状态合同明显不同**
> 的真实模型，"**不是只换 checkpoint 名称**"，并按 P0 那套闭环把它走完
> （声明 → 产物 → 独立对拍 → 标准产物 → 上游运行时）。
>
> 这一页写清三件事：**为什么选它、判卷人在哪、缺什么**。

## 一、选型：为什么是 mamba-130m-hf

| 候选 | 规模 | 判卷人 | 本机（CPU，15.7 GB）可跑 | 结论 |
|---|---|---|---|---|
| **mamba-130m-hf** | **130M**（24 层 · d768） | transformers `MambaForCausalLM`（v1）· llama.cpp `mamba` | ✅ | **选它** |
| clef | **27.36B**（64 层 · d5120） | transformers `Qwen3_5*` · llama.cpp `qwen35` | ❌ 真权重 fp32 约 110 GB | 判卷人有、体量不可行 |
| Qwen3.6-35B-A3B | 35B | Qwen3-MoE 系 | ❌ | 同上 |
| Qwen2.5-0.5B-Instruct | 0.5B | 有 | ✅ | 但**只是换 checkpoint 名称** —— T26 明确不要这个 |

**状态合同不同在哪**：mamba **没有 KV cache** —— 状态是**定长递归状态**
（config 实测：`state_size=16` · `conv_kernel=4` · `expand=2` · `use_conv_bias=true`）。
所以 P0 那条"增量 == 全量"的判据在这里要**换写法**（递归状态 vs 逐步重算），
这正是 T26 想压的东西。

## 二、判卷人现状（2026-10-16 实测）

| 侧 | 有吗 | 证据 |
|---|---|---|
| refs 产物 | ✅ | `refs/state-spaces__mamba-130m-hf.config.json` + `.tensors.json` |
| 契约（IR 路径） | ✅ 规格级 | `by1all` 档 3：`config mamba-130m-hf.by1 逐字段: 一致 32` + 张量名/形状 |
| transformers（独立实现） | ✅ **但没导出到顶层**：`CONFIG_MAPPING_NAMES['mamba'] = 'MambaConfig'`、`transformers.models.mamba` 里有 `MambaForCausalLM` —— 要用 `AutoModelForCausalLM`，`T.MambaForCausalLM` 取不到 | 本机 5.15.1 与克隆 5.20.0 **都有** |
| llama.cpp（独立运行时） | ✅ `LLM_ARCH_MAMBA`（`"mamba"`） | `llamacpp/llama.cpp/src/llama-arch.cpp:65` |
| by1 自己的执行（NumPy） | ❌ **明确拒绝**：`by1exec models/mamba-130m-hf.by1` → "第 N 层没有可用的 token 混合器" | 这正是"要么实现，要么拒绝"的**拒绝面**，不是静默近似 |
| by1 的 Transformers 后端家族 | ❌ `by1full.FAMILIES` 目前只有 `qwen3 / qwen2 / gpt2 / gpt_oss / qwen3_5` | `src/by1full.py` |

**一条要记住的旧账**：`src/modelcheck/by1ssm.py` 开头写着 ——
"**不能拿 `MambaForCausalLM` 当判卷人** —— 纯 Mamba 的 `in_proj` 是 … Nemotron-H
那一版的布局。判卷人换了就不是同一个判卷人。"
那句话的语境是 **Mamba2（Nemotron-H mixer）**；这次要判的恰恰是 **v1**，
所以判卷人正好要用 `MambaForCausalLM`。**两个方向都不能混。**

### 判卷人已实测可用（2026-10-16，本机 CPU）

```
权重 model-cache/mamba-130m-hf/model.safetensors  516.6 MB · F32
  头里张量 242 · refs 清单 242 · 名字集合相同 True · 形状不一致 0
判卷人 AutoModelForCausalLM.from_pretrained(..., local_files_only=True)
  → MambaForCausalLM · 前向 logits (1, 5, 50280) · max|.| = 65.19 · 参数 129.1 M
```

也就是说：**独立实现这一侧不是"大概有"，是已经装好跑过一次了。**
剩下要补的是 by1 那一侧（`by1full` 的 mamba 家族 + 声明 + 对拍）。

## 三、要做的四件事（判据沿用 P0 那一套，不新造尺子）

1. **`by1full` 加 `mamba` 家族**（`MambaConfig` / `MambaForCausalLM`）+ `SSM` 机制的字段表；
2. **写一份 Transformers 后端的 complete 声明**（命名按仓库规则：在
   `mamba-130m-hf` 前加 `complete-`，与现有 IR 声明并存），过 `by1complete`；
3. **真权重双 loader 对拍**（`by1shadow --smoke-refs --weights-dir`）——
   判据 2e-3，最好 0；同时给递归状态写一条"**增量 == 全量**"的等价判据（写法要新）；
4. **标准产物**：`by1llama build-gguf --arch mamba` → 上游 `llama-eval-callback`
   logits 逐位（短序列即可）。

### 进度（2026-10-16 当天）

| 步 | 状态 | 证据 |
|---|---|---|
| 1 | **✅** | `by1full`：`FAMILIES` 加 `mamba` · `MECH_FIELDS` 加 `SSM` · `_compile_mamba` 早退分支 · `_tclass`（**两个类没导出到顶层**，`getattr(T, …)` 会 AttributeError）· `_layers` 允许 **SSM 层没有 FFN**（Mamba1 的 mixer 就是整层） |
| 2 | **✅** | `models/complete-mamba-130m-hf.by1`：`by1check` **0 错 0 警** · `compile_config` 出 **32 个 config 字段** · `build_model(meta)` → `MambaForCausalLM` **129.1 M**（与真权重同）· `load_model` 真权重**零 missing/unexpected** · `by1complete` 的 `official_configs` 已把 mamba 接进逐字段判据（**PASS**） |
| 3 | **✅（前向那半）** | `by1shadow --smoke-refs --by1 complete-mamba-130m-hf.by1 --weights-dir model-cache/mamba-130m-hf` → **242 个真名 · 双 loader 零 missing/unexpected · 前向 max\|Δ\| = 0.000e+00** |
| 3' | 待做 | 递归状态的"**增量 == 全量**"等价判据（写法与 KV cache 那条不同：状态定长） |
| 4 | 待做 | `build-gguf --arch mamba` + 上游 `llama-eval-callback` logits 逐位 |
| — | 待做 | 把 mamba 接进 `by1complete` 的 **GALLERY**（那张表的用例是按 Attention/FFN 家族写的，mamba 入场要另一套夹具） |

## 四、完成定义

- [x] complete 声明过 `by1check` + `by1complete`（含 config 逐字段）；
- [x] 真权重双 loader：零 missing/unexpected + 前向 max|Δ| **0.000e+00**；
- [ ] 递归状态的"增量 == 全量"等价判据；
- [ ] 上游加载 + logits 逐位（`build-gguf --arch mamba`）；
- [ ] 报告进 `docs/`，并更新 [支持矩阵](support-matrix.md) 的"第二架构"一行。

## 五、不许做的

- 四项没齐之前，**不写"运行 ✅"**；
- 不因为"Mamba2 已经在别的机制上验过"就跳步 —— 判卷人换了就不是同一个判卷人。
