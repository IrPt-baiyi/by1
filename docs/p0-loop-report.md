# P0 闭环报告 · Qwen3-0.6B

> 日期 **2026-10-16** · 机器：本机 Windows · **CPU-only（无显卡）** ·
> 代表模型选择记录见 [p0-model-selection.md](p0-model-selection.md)。
>
> **环境（本报告所有结论都出自这一套）**：torch `2.13.0+cpu` ·
> transformers `5.15.1` · safetensors `0.8.0` · numpy `2.5.2`。
>
> ⚠ **版本口径分裂（已验，不影响本报告结论）**：`pyproject` 的 `[full]` 钉的是
> `transformers>=5.19,<6`，`gpu-results/` 里的 GPU 报告记录的也是 **5.19.0**；
> 而本报告的 CPU 结论跑在 **5.15.1** 上（与 `docs/which-gpu.md` 记的"和本机一致"
> 同一条）。**已补验**：在干净克隆里用 `transformers==5.19.0` 重跑关键那一步
> （真权重双 loader 对拍），结果与 5.15.1 下**逐位相同**（`max|Δ| = 0.000e+00`）
> —— 这条结论**跨这两个大版本稳定**，见第七节。
>
> 本报告只写**今天真正跑出来的那几段**。没跑的单独列在第四节 ——
> **不写成通过**，这是这个仓库对"证据"的最低要求。

## 结论（一句话）

`.by1 声明 → 真实产物对账 → 真权重独立数值对拍 → 标准 GGUF → 未修改的上游
llama.cpp 加载与生成` 五段，加上**增量状态**（T14）与 **tie 语义**两笔结案，
在 Qwen3-0.6B 上都有可重跑证据：**311/311 张量命中**、
**真权重前向 max|Δ| = 0.000e+00（CPU，两个独立 loader）**、
**上游 logits 逐位相同（106949 个值）**、**生成文本完全相同**、
**增量 == 全量（2.289e-05，判据 2e-3）**。

## 一、输入（每项可核对）

| 项 | 值 | 来源 / 命令 |
|---|---|---|
| 模型 | `Qwen/Qwen3-0.6B`（base，文本） | p0-model-selection.md |
| 许可 | Apache-2.0 | refs/SOURCES.tsv |
| 声明 | `models/complete-Qwen3-0.6B.by1` | `by1check` 0 错 / 0 警告 |
| config | `refs/Qwen__Qwen3-0.6B.config.json`（726 B） | `python src/data/by1fetch.py --only Qwen3-0.6B`（8.4 s） |
| 张量清单 | `refs/Qwen__Qwen3-0.6B.tensors.json`（**311** 张） | 同上 |
| 真权重 | `model-cache/qwen3-0.6b/model.safetensors`（**1433.7 MB**） | HF resolve（hf-mirror），含 tokenizer/config 全套 |
| 上游运行时 | `llamacpp/build-cli/bin/llama-cli.exe` · `llama-eval-callback.exe` | 本机构建 |

config 与声明的关键字段逐项吻合：`qwen3` · 28 层 · 16:8 头 · `head_dim` 128 ·
`tie_word_embeddings=true` · `rope_theta` 1e6 · `max_position_embeddings` 40960。

## 二、证据（逐段）

### ① 声明 ↔ 发布方产物（T12）

```bash
python src/modelcheck/by1shadow.py --smoke-refs --by1 complete-Qwen3-0.6B.by1
```

```
源二影子 …：311 个真名（2.80 GB）
源二 e2e OK：双 loader 零 missing/unexpected，前向 max|Δ| = 0.000e+00
```

权重文件本身也逐张量核过（对照同一份清单）：

```
文件大小 1433.7 MB · 头里张量 311 · refs 清单 311
名字集合相同: True · 形状不一致: 0 · 按头计算权重字节 1433.6 MB（自洽）
```

**没销的问题是 tie 语义**：清单里 `lm_head.weight` 与
`model.embed_tokens.weight` 同时存在，而 config 说 `tie_word_embeddings=true`；
Transformers 加载时给出警告"两者都在且值不同，故不绑"。
**存储布局 vs 共享语义要写清** —— 这正是"同名不同义"的高发区，见第四节。

### ② 标准产物（T16）

```bash
# 基准：上游自己的转换器（tokenizer KV 的来源，也是逐位基准）
python llamacpp/llama.cpp/convert_hf_to_gguf.py model-cache/qwen3-0.6b \
    --outfile build/qwen3-0.6b-ref-f32.gguf --outtype f32        # 3.01 GB

# by1 产物线（KV 表驱动；tokenizer KV 从 ref 整拷 —— S1 记档）
python src/checks/by1llama.py build-gguf model-cache/qwen3-0.6b \
    build/qwen3-0.6b-by1-f32.gguf --ref build/qwen3-0.6b-ref-f32.gguf \
    --arch qwen3 --n-blocks 28                                    # 311 张量 · 2.81 GB
```

> **本次顺带修掉的入口缺口**：`build_gguf` 原来是个**没有调用者的函数**
> （`main()` 的 7 个子命令里没有它，runbook 里那条命令跑不通），
> 而且 `arch` 被硬编成 `LLAMA`、`n_blocks` 默认 24 —— 那正是
> gap **S2**"只 qwen2 档实证"的字面来源。现在入口按参数走。

### ③ 未修改的上游运行时加载 + logits 逐位（T17）

```bash
CLI=llamacpp/build-cli/bin/llama-eval-callback.exe
python src/checks/by1llama.py logits-run --gguf build/qwen3-0.6b-ref-f32.gguf \
    -p "The capital of France is" -o cgen/qwen3-ref.logits --llama-cli $CLI
python src/checks/by1llama.py logits-run --gguf build/qwen3-0.6b-by1-f32.gguf \
    -p "The capital of France is" -o cgen/qwen3-by1.logits --llama-cli $CLI
python src/checks/by1llama.py logits-compare cgen/qwen3-ref.logits cgen/qwen3-by1.logits
```

```
报告写了 cgen/llama-logits-compare.json
[PASS] logits 逐位相同（106949 个值）
```

### ④ 短序列生成（T14 的一部分）

```bash
llamacpp/build-cli/bin/llama-cli.exe -m <gguf> -p "The capital of France is" \
    -n 16 --seed 42 --temp 0 -st
```

两份产物（上游 ref 与 by1 产物）输出**同一段文本**：

```
> The capital of France is
Okay, the user is asking about the capital of France. Let me
```

### ⑤ 真权重独立数值对拍（T13）—— **CPU 上就能跑**

```bash
python src/modelcheck/by1shadow.py --smoke-refs \
    --by1 complete-Qwen3-0.6B.by1 --weights-dir model-cache/qwen3-0.6b
```

```
真权重 model-cache/qwen3-0.6b：311 个真名（1.40 GB，未合成）
源二 e2e OK：双 loader 零 missing/unexpected，前向 max|Δ| = 0.000e+00
```

两个独立 loader（官方 Transformers 的 `Qwen3ForCausalLM` 与 by1 生成的模型）
装**同一份发布方权重**、喂同一输入，前向**逐位相同**。
判据一个字没改 —— 只是把权重来源从"按 refs 合成"换成"发布方原样"
（`--weights-dir`，2026-10-16 新增）。**无需显卡**：0.6B fp32 两份约 4.8 GB。

### ⑥ 增量状态（T14 的另一半）—— cache 与全量重算

```bash
python src/modelcheck/by1shadow.py --smoke-state --by1 complete-Qwen3-0.6B.by1 \
    --weights-dir model-cache/qwen3-0.6b
```

```
真权重 …：增量 vs 全量（长 16，prefill 8）
官方 loader：max|Δ| = 2.289e-05
by1 模型   ：max|Δ| = 2.289e-05
[PASS] 增量 == 全量（两个 loader 都在 2e-3 内）
```

判据 2e-3，实测余量约 87 倍（差值来自 fp32 累加序，不是接线）。
**为什么单列**：cache 那条路读 `position_ids` / `attention_mask` / rope 三样接线
—— 接错了**全量那一遍仍然是对的**，只有增量会歪。两个 loader 给出**同一个**
2.289e-05，说明这条路的接线两边一致。

### ⑦ tie 语义（原悬案，已结）

发布方的张量清单里 `lm_head.weight` 与 `model.embed_tokens.weight` **同时存在**，
而 config 说 `tie_word_embeddings=true` —— 到底是绑的还是不绑的？
直接比真权重文件里的值：

```
逐元素相同: True · max|Δ| = 0.0 · 非零元素占比 0.0000%
```

**共享关系被物化成两份逐位相同的存储。** 结论：
① by1 声明 `tie_word_embeddings=true` 与产物一致；
② 合成权重上那句"两者值不同故不绑"的警告在真 checkpoint 上不成立；
③ 因为两份值相同，**绑与不绑在数值上等价** —— 这也是两个 loader 逐位相同的一个前提。
（换成一个真·untied 模型时，这条要重验 —— 届时"绑不绑"就不再等价了。）

## 三、这次销掉的缺口（docs/gap-analysis.md 已同步）

- **S2**（build-gguf 族覆盖只 qwen2）：**qwen3 档已实证** —— 311 张量全映射、
  logits 逐位、生成同文。其余族仍不外推。
- **G2**（外部环境复现无记录）：**外部复现 #1** 已在
  `history/external-repro/2026-10-16-colab-hello.md`（Colab CPU 入门全对，
  并抓到 `head_dim=31` 缺口）。

## 四、没跑的（**不是通过**）

| 项 | 状态 | 原因 / 收口路径 |
|---|---|---|
| **T39 真权重训练步**（梯度 + AdamW 一步） | **未跑** | `by1gputrain.py` 硬要 CUDA；即便加 CPU 路径，本机可用内存 10.3 GB < 全参数 fp32 训练步需要的约 10–12 GB（params 2.4 + grads 2.4 + 两个优化器状态 4.8）。留租卡日，或换更小模型/梯度检查点 |
| **S1 tokenizer 组装线** | 仍未实现 | 本次 GGUF 的 tokenizer KV 从上游 ref 整拷 —— 按记档，非缺陷伪装 |
| **T15 干净克隆（剩余腿）** | **部分** | 安装 + README 入门 + 档 1 已在干净克隆里跑通；真权重对拍也在克隆里用 `transformers==5.19.0` 补验过（第七节）。仍缺：在克隆里**从零装** `.[full]`（torch 约 2 GB）重跑产物线 —— 现在那次是 `--system-site-packages` 借的系统 torch，环境干净度打了折 |
| 其余模型族 | 不外推 | Qwen3-0.6B 一个新档，不代表别的族 |

## 五、复现（按顺序，全程 CPU）

```bash
python src/data/by1fetch.py --only Qwen3-0.6B          # refs 语料
# 权重：HF Qwen/Qwen3-0.6B 的 model.safetensors + config.json + tokenizer 全套
python src/modelcheck/by1shadow.py --smoke-refs --by1 complete-Qwen3-0.6B.by1
python src/modelcheck/by1shadow.py --smoke-refs --by1 complete-Qwen3-0.6B.by1 \
    --weights-dir <真 checkpoint 目录>          # T13：真权重双 loader 逐位（CPU 可跑）
python src/modelcheck/by1shadow.py --smoke-state --by1 complete-Qwen3-0.6B.by1 \
    --weights-dir <真 checkpoint 目录>          # T14：增量 vs 全量（CPU 可跑）
python llamacpp/llama.cpp/convert_hf_to_gguf.py <hf-dir> --outfile ref-f32.gguf --outtype f32
python src/checks/by1llama.py build-gguf <hf-dir> by1-f32.gguf --ref ref-f32.gguf \
    --arch qwen3 --n-blocks 28
python src/checks/by1llama.py logits-run --gguf ref-f32.gguf -o ref.logits --llama-cli <eval-callback>
python src/checks/by1llama.py logits-run --gguf by1-f32.gguf -o by1.logits --llama-cli <eval-callback>
python src/checks/by1llama.py logits-compare ref.logits by1.logits
```

> 数字跟着机器走；本报告的值出自上面这台机器的那次运行。
> 上游版本与构建参数仍按 [llama.cpp 基线](llama.cpp-baseline.md) 钉。

## 六、判据归属：为什么这里**没有** `config 逐字段行`

`by1verify … --config --tensors` 对 P0 声明会打出两个 **[跳过]**：

```
[1] config 逐字段对拍   [跳过] by1 里没有 transformers.config 的 field 映射
[3] tensor names+shapes [跳过] by1 里没有 emit[torch.module] 规则
```

**这不是缺口，是后端分工**：`complete-*` 声明走 **Transformers 后端** ——
config 由 `by1full.compile_config` 直接生成、权重由 HF 自己的 loader 装载，
所以既不需要 `emit[torch.module]` 命名规则、也不需要 field 映射表
（那两样是**原生 IR 路径**的机制）。这两格的等价判据是：

| IR 路径的格子 | Transformers 后端的等价判据 | 本报告里的证据 |
|---|---|---|
| config 逐字段 | 生成 config → HF 装载**零 missing/unexpected** | ① / ⑤（311/311，零缺失） |
| emit + 张量名/形状 | 双 loader 装载同一份真权重 + 前向逐位 | ⑤（max\|Δ\| = 0.000e+00） |

换句话说：**同一件事有两种判据，别把"跳过"读成"没验"** ——
跳过的是不适用的那套机制。

## 七、干净克隆复现（T15 的第一腿）

```bash
git clone --depth 1 https://github.com/IrPt-baiyi/by1.git by1-clean
cd by1-clean
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[run]"     # 装出 by1-1.0.0rc1 + numpy 2.5.4
.venv/Scripts/by1 check models/hello.by1            # 0 错误 / 0 警告
.venv/Scripts/by1 run   models/hello.by1 --seq 8    # 跑通：输出 (1, 8, 256)，幅度 4.2576e-01
python src/checks/by1all.py --tier 1                # 14 项，0 项失败，13 项跳过
```

克隆规模 393 文件 / 41.4 MB。**README 的逐条命令全通**，而且装出来的正是
`1.0.0-rc1`（上一轮修的版本串当场被这条路径验证），`by1 run` 的数值
（4.2576e-01）与开发树**逐位相同**。

### 这条腿当场抓到两个真缺陷（都已修，commit `4f322b8`）

| 缺陷 | 表现 | 根因 | 修法 |
|---|---|---|---|
| `by1docs` 判据错 | 干净克隆里档 1 红：`refs/*.index.json` + `history/_hf_probe/*.py`(8) + `llamacpp/...convert_hf_to_gguf.py` 共 **10 处**判"不存在" | 这些路径都被 `.gitignore` 挡着（可再生语料 / 上游检出 / 门禁探针日志）—— **文档该引用它们**，判红等于要求"干净克隆里也得有本地产物" | 加 `ignored()`（`git check-ignore`）分类，进单独一桶「外部产物」，**既不算通过也不算失败**，并打印出来 |
| `build-gguf` 失败姿势 | 裸 `ModuleNotFoundError: No module named 'gguf'` traceback | `gguf-py` 来自 llama.cpp 检出，不在库里 | 改成明确的缺依赖提示（说清来源 + 给出取法），exit 1 |

### 这条腿**没有**覆盖的（不写成通过）

- **权重与 llama.cpp 二进制来自克隆之外**（`model-cache/`、`llamacpp/` 都被
  `.gitignore` 挡着）—— 这是设计，不是缺陷；但"从零开始"里"零"的边界要写清。
- **克隆的 venv 只装了 `.[run]`（numpy）**：torch 那几段没用克隆的 venv 装过
  （`.[full]` 约 2 GB 下载）。不过下面那条 5.19 补验用
  `--system-site-packages` 借了系统 torch —— **环境干净度打了折，如实记在这里**。

### 补验：换到依赖钉的那个版本（`transformers==5.19.0`）结论不变

```bash
# 在干净克隆里
python -m venv --system-site-packages .venv519
.venv519/Scripts/python -m pip install "transformers==5.19.0"
.venv519/Scripts/python src/modelcheck/by1shadow.py --smoke-refs \
    --by1 complete-Qwen3-0.6B.by1 --weights-dir <真 checkpoint 目录>
```

```
transformers 5.19.0 | torch 2.13.0+cpu | safetensors 0.8.0
源二 e2e OK：双 loader 零 missing/unexpected，前向 max|Δ| = 0.000e+00
```

`pyproject` 钉的版本（5.19.0，PyPI 上确实存在、12.6 MB 装得上）
与开发机的 5.15.1 **给出同一个结论**。所以"版本口径分裂"是记账问题，
**不是**结论的隐患 —— 但它仍然是记账问题，两处版本号该写明（本报告 + which-gpu）。
