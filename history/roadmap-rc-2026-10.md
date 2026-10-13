# 1.0.0 路线图（rc0 – rc9 → 1.0.0）

> 定于 2026-10-12。**尺子是量出来的**：仓库现有 33.7k 行
> （py 23.8k + .by1 9.8k + C 0.1k），整个东西 4 天写成 ——
> **1/4 仓库 ≈ 8.4k 行 = 一个版本 = 一天**。
>
> 发版铁律（每次动版本号都要满足）：
> ① 每版必须含 ≥1 项：新判卷人 / 新可算能力 / IR 变更 / 真实模型毕业 / 可交付产物
> ② 发版动作做全套：全档门禁绿 → `by1ver.py --write` → `history/` 记现量数字 → tag → `by1pack`
> ③ 版本史不许美化：不达标挪下一版，号不回收、不补写
> ④ 盖章跟版本走（by1all 先报版本的规矩已存在）

## 时刻表

| 版本 | 类 | 内容 | 发版门（判卷人） |
|---|---|---|---|
| **0.9.1**（补丁，不占号） | 修补 | 描述↔计算闭环：IR 形状/类型静态推导 · 契约与 shapes_of 同源 · 变异反例库进门禁 | 全护栏不退 + 每条规则反例必红 |
| **1.0.0-rc0** | 地基 | 原语图层（IR→规范化原语图，四后端机械化翻译，w.bin 布局独立）+ 数值规格与精度预算 + 导出命名反推 | 四后端对拍不退 + 逐算子神谕 + by1verify 覆盖不退 |
| **1.0.0-rc1** | 能力 | llama.cpp 全链：ggml 算子补齐 + 生成进本体 + decode/状态 + Instella 真 GGUF | llama-cli logits 逐位 |
| **1.0.0-rc2** | 能力 | 量化 × 多模态（**双倍规格**，内部 rc2.1/rc2.2 推进）：量化布局/数值/ggml 运行时 + by1-ir 1.1 + 视觉/音频塔 + 端到端 | 真 GGUF manifest 不退 + transformers/mtmd 逐位 |
| **1.0.0-rc3** | 能力 | GGUF 产物线：by1 + 权重 → 真 `.gguf`（含量化打包） | llama-cli 直接加载逐位 + convert 脚本结构 diff |
| **1.0.0-rc4** | 能力 | 采样与生成闭环：top-k/top-p/温度/惩罚 + 生成循环 | llama.cpp sampler 同种子逐 token |
| **1.0.0-rc5** | 地基 | by1fuzz 差分模糊测试：随机合法模型 + 多后端差分 + 已知 bug 种子回放必红 + 反例自动最小化 | 差分自证 + seed 回放 |
| **1.0.0-rc6** | 地基 | 后端插件协议 + ONNX 第五后端 | **onnxruntime 逐位**（第三独立生态） |
| **1.0.0-rc7** | 能力 | 扩散语言模型族（LLaDA）：masked diffusion + 迭代解码 + 双向掩码 | LLaDA 官方 modeling |
| **1.0.0-rc8** | 能力 | 线性注意力族外推（RWKV-7 / Griffin / DeltaNet 变体） | 官方 rwkv / flash-linear-attention |
| **1.0.0-rc9** | 能力 | 产物反推描述（auto-draft）：refs → 自动起草 .by1，机械 64% 全自动 | 起草结果对同一产物 by1verify 全绿 |
| **1.0.0** | 发布 | pack + tag + Instella 一页材料 | **干净 clone 复现**（by1rebuild）+ by1all 全档 |

### 量化能力的先后顺序（2026-10-13）

量化按依赖拆成三段，不把“能读出量化配置”写成“支持量化推理”：

| 顺序 | 能力 | 范围与验收 |
|---|---|---|
| **Q1** | 格式规格与校验 | 固定 FP8 块量化和 MXFP4 的权重存储、scale 语义、块形状、bias、融合规则与 dtype；实现元素/scale/块点积数值 oracle；真实 safetensors manifest 对拍名字、形状和可用 dtype；声明格式缺字段/冲突字段的反例必须被拒。 |
| **Q2** | 加载并运行量化 checkpoint | 直接保留量化权重执行，不先把双方都解量化成 BF16。先做现有 FP8 checkpoint，再做 MXFP4；覆盖权重共享、融合专家及边界维度，与发布方参考实现比较 logits、生成 token 和峰值显存，并在兼容 GPU 上复现。 |
| **Q3** | 生成量化产物 | 从已验证的浮点权重生成明确格式的量化权重和 GGUF，保持共享权重与融合布局语义；由 llama.cpp 加载，做结构 round-trip 与 logits 对拍。格式由 Q1 冻结，运行数学由 Q2 验证后再进入产物线。 |

当前边界：Q1 已把格式名、存储 dtype、bias、块形状与融合规则及数值 oracle 写入规格；`by1verify` 用真实 safetensors manifest 对拍量化张量名字、形状和 dtype。Q2 的 `by1full.load_model` 现可接收 `.by1` 量化声明，构造 Transformers 原生 FP8/MXFP4 配置，并在加载后拒绝静默解量化、检查权重共享；benchmark 可接收任意匹配的 `.by1` 量化契约与固定 checkpoint revision。这里使用 Transformers 原生算子，不代表 by1 自研 CUDA kernel。RTX 上尚未完成前向、生成、视觉塔、MTP 和显存对拍，因此状态仍是“量化加载路径已接通、GPU 执行待验”，不能记作 Q2 完成。现有仓库暂缺可直接跑的完整 FP8 checkpoint + `.by1` 后端组合；Qwen3.8 图像/MTP 路径需匹配的量化 checkpoint 与契约。Q3 尚未开始。

配比：**地基 3**（rc0/rc5/rc6）· **能力 7**（rc1–4、rc7–9）· **发布 1**。
版本移动：0.9.1 补丁 + rc0~rc9 十次大幅度 + 1.0.0 = 12 次 ✓

**备选车厢**（可换掉任意一节）：内存/性能实测化 · 训练并行声明（TP/PP/FSDP）·
多栈/跨栈投影（DeepSeek，等判卷人普查）· 安全边界 harness。

---

## 1.0.0-rc0 详情（执行中）

**地基合订，~8.4k 行。三个部件 + 一个前置补丁。**

### 前置 · 0.9.1 补丁（先发，rc0 的第 0 步）

| # | 任务 | 判卷人 |
|---|---|---|
| T0.1 | **IR 形状/类型静态推导**：每个 op 的输入输出宽度从 attrs + `in_dim` 静态推出，`by1ir.validate()` 查通 | 不合法 IR 必须被拒（宽度接不上的反例进库） |
| T0.2 | **契约与 shapes_of 同源**：`by1tens` 的机制→张量表与 `by1exec.shapes_of` 共用一份生成器，消灭两份手写事实 | shapes_of 与契约逐张量 diff 为空；by1verify 覆盖数字不退（gemma-4-31B 全局 1152 bug 就是双份事实漏出来的） |
| T0.3 | **变异反例库进门禁**：每条检查规则 ≥1 个「必须红」的反例 | 反例全红 + 正例全绿，进 by1all 档 2 |

### 部件 A · 原语图层（~4k 行，rc0 的主体）

| # | 任务 | 判卷人 |
|---|---|---|
| A1 | **图规范 by1-graph 1.0**：节点集（matmul/add/mul/norm/rope/softmax/get_rows/act 系/view…）+ 每节点唯一语义（可执行规格） | ir-spec 风格再生成；每节点语义有公式出处 |
| A2 | **lowering**：IR 13 kind → 图全覆盖；`by1emit` 的 GGML 映射表并入（它就是骨架） | **golden 图快照**（每模型 JSON）进仓库，逐字节稳定 |
| A3 | **四后端机械化**：by1exec（图求值）/ by1c / by1ggml 改为吃图；w.bin 布局（weight plan）从 `by1c.emit_c` 独立成唯一定义 | 四后端对拍全不退（by1opdiff 8 文件 + shaped 全套 + by1e2e）；C↔NumPy、ggml↔NumPy 相对差不劣化（改前改后数字都记） |
| A4 | **迁移纪律**：每挪一个算子跑一次对拍，不许大爆炸切换。rope 三缩放 / eps / qk_norm 三态 / gelu 三兄弟 / kv_tie 分叉点 —— 在图层里**只有一个家** | 覆盖不降（编译 N·import N·行尾 N）+ by1lint 静默写法 0 |

### 部件 B · 数值规格（~2k 行）

| # | 任务 | 判卷人 |
|---|---|---|
| B1 | 每算子**可执行规格**（公式 / 定义域 / 舍入约定）+ spec runner | spec runner 对四后端绿 |
| B2 | **精度预算表**：算子 × 相对差上界 × 出处（构造性 / 参照 / 实测） | 每个数字有出处链接；by1opdiff 常驻档按预算判定 |
| B3 | criteria-review 任务 A/B 重跑留产物（手算三模型 + 阈值复核） | 重算数字 vs `docs/INVENTORY.txt`【A】逐行 |

### 部件 C · 导出命名反推（~2.4k 行）

| # | 任务 | 判卷人 |
|---|---|---|
| C1 | 命名模板反推器：refs 产物清单 → scope/class/global_name 模板草案 | 反推结果 by1verify 全绿（gemma 530/530 · gpt-oss 687/687 一类不许退） |
| C2 | models.tsv / by1name 对照自动化；手写决策残余量化报表 | 现有真实模型命名段的机器生成率只升不降 |

**rc0 发版门**：by1all --tier 5 全绿 · 图 golden 快照齐 · 精度预算表成文 ·
by1verify 覆盖不退 → `by1ver.py --write`（TOOL_VERSION = 1.0.0-rc0）→ history 条目 → tag。

---

## 进度与下一步（2026-10-12 收尾，换机续作从这读）

### 已完成（都有判卷人，门禁绿）

| 项 | 状态 | 判卷人 |
|---|---|---|
| T0.1 IR 形状/类型静态推导（`by1ir.infer_widths`，换宽度进 gaps 不进 errs） | ✅ | `by1ir.py --selftest`：3 反例红 + 50 模型绿，挂 by1fast ③b |
| T0.2 第一步：形状表收敛 `by1ir.params_of` + 同步报表 `checks/by1sync.py` | ✅ | `by1sync --selftest`：1152 bug 注回去必响 |
| T0.3 变异反例库 `checks/by1mutate.py`（by1ir.validate 每规则一个必红反例） | ✅ | 20 反例红 + 基线绿，挂 by1all 档 2 |
| A1 图规范 `by1graph.py`：PRIMS 16 条（公式 + numpy ref + 旧家出处） | ✅ | `--selftest` 对拍手算 |
| A2 核心六类 lowering（Embed/Norm/Attention/FFN/Add/Head）+ golden | ✅ | `graphs-golden.json` **19 张图**逐字节稳定，**33 份**降不动在账上 |
| A2 补：**MoE（`softmax_topk` 那一刀）** | ✅ | mixtral-shaped e2e **4.269e-10**；可降 14 → **19 份** |
| A2 补：**Linear（GDN）** | ✅ | 隔离分段 11 段全一致 + 整步相对 1e-7（rep=1/2） |
| A2 补：**q_gate**（含 `one_plus` 属性修正） | ✅ | clef-tiny **5.984e-09** · qwen3-next-shaped **1.780e-07**；可降 19 → **29 份** |
| A2 补：学习式位置表 lowering（gpt2 那一支） | ✅ | e2e 对 by1exec **0.000e+00**（逐位）；golden 9 → 11 |
| A2 补：`qkv_fused` + `sink` 两条 lowering（两条新原语） | ✅ | e2e：attn-fused 5.081e-10；golden 11 → 13 |
| A2 补：**YaRN / llama3 / rope_scale 收编**（`rope_inv` 搬家） | ✅ | 27 个 (dim×base×yarn) 组合**逐位相同**；llama3-shaped 1.427e-09 |
| A2 补：**Attention.head_gate**（per_head，softplus / sigmoid） | ✅ | 缩维合成 IR 对 by1exec **5.400e-09**；Laguna 进 golden **30 张**（真模型 e2e 挂「租机跑全」） |
| A2 补：**KDA 收编**（低秩门 / GVA / gate_lower 三支） | ✅ | kda-shaped 7.396e-09 · kda-gva 6.431e-09 · kda-lowrank 6.848e-09；golden **33 张**；e2e 13 份一致 |
| A2 补：**MoE 选法 sigmoid_topk + topk_softmax** | ✅ | 单算子图对 op_moe：sigmoid_topk 4.917e-07 · softmax_topk 5.341e-07（不退）；Step×2 进 golden **35 张** |
| A2 补：**MoE 选法 sigmoid_group_topk（noaux_tc）** | ✅ | moe-routing-shaped **整模** e2e 3.783e-10；新原语 one_hot / masked_fill；golden **36 张** |
| A2 补：**MLA 收编**（非稀疏：低秩 q/kv + 单头 k_rot + head_gate） | ✅ | mla-shaped 2.267e-08 · ling-tiny（KDA+MLA+MoE 合体）2.564e-08；golden **40 张** |
| A2 补：**MoE expert_bias + gptoss 激活** | ✅ | gpt-oss-shaped 整模 e2e **1.269e-10**；新原语 clamp_max；golden **43 张** |
| A2 补：**SSM 收编**（Mamba2 递推 = `ssm_scan` 原语） | ✅ | ssm-shaped 整模 e2e **9.682e-09**；golden **44 张** |
| **修静默 bug：MoE 激活取值三后端一致化**（relu2/gelu 被算成 silu） | ✅ | 公式层 relu2 **逐位 0.000e+00**；图路径单算子 1.174e-07；Nemotron 进 golden **45 张** |
| **修静默 bug：qk_norm=full 的宽度与位置**（被当 per_head 算） | ✅ | by1opdiff 修前崩 `size 80 vs [2560]` → 修后全对；by1sync **布局差 1 → 0**；Instella 进 golden **46 张** |
| A2 补：**SparseMLA 非池路径**（DSA 索引器用原语拼） | ✅ | 单算子图 5.193e-09 · **C 机械表 7.863e-08**；glm-text-tiny 整模 1.111e-08；golden **48 张** |

### A2 的账：4 份降不动，按**第一个**拒绝分组（2026-10-12 普查 · SparseMLA 非池收编后）

    1 份  MLA.index_kpool > 1（mla-sparse-shaped 的 L2/L3 —— k-pool 路径）
    2 份  codegen 层就拒了（gate-probe 的假激活 / mamba-130m 的 SSM 缺属性）
    1 份  Raw（逃生舱，按设计不收编）

**注意**：这是"**第一个**拒绝"，不是"全部缺口"—— 例如 `gpt-oss-shaped`
先报 sink、修好 sink 后报 YaRN、修好 YaRN 后报 MoE。**每次只前进一步。**

### A2 补：Attention.head_gate 收编（2026-10-12）

`head_gate = per_head`（Laguna softplus / Ling sigmoid）——照
`by1exec.op_attention` 292-298 逐句降成 `matmul(g_proj) -> act -> view
[n,头,1] -> mul`，乘在注意力输出上、wo 之前。**非 per_head 明确拒绝**
（模型里还没有那一支）。

**判卷人两个**：
① 缩维合成 IR（llama-shaped 的骨架改小 + 每层加 head_gate，softplus /
   sigmoid 两支都走、GQA q=4/kv=2）对 by1exec：**相对 5.400e-09**；
② 真模型 `Laguna-XS-2_1` 降得动了，golden **29 → 30 张**（只增一条，
   既有条目一字未动）—— 但它 133.77 GB，e2e 数值证挂在「租机跑全」上，
   **不许当已验**。

普查跟着动：head_gate 那 3 份出列 —— Laguna 进 golden；Step-3.7-Flash
×2 前进到下一个拒绝（MoE.routing=sigmoid_topk）。顺带 e2e 小预算下
**5 份 → 10 份一致**（含 mixtral-shaped 4.269e-10 · qwen3-next-shaped
1.780e-07，与 A2 各刀的记录逐个吻合）。

### A2 补：KDA 收编（2026-10-12）

KDA（Kimi Delta Attention）—— 逐句照 `by1exec.op_kda` 降成原语链：
q/k/v 分开投影 + 各一条深度卷积（conv1d_causal + silu）、低秩门两层
（f_a/f_b，GLM）或一层（f_proj，Ling）、**衰减逐维** `[n, nv, dk]`、
输出门 **sigmoid**（不是 silu）、GVA 广播（q/k 头数 < v）。三个支路各有
代表模型：`kda-shaped`（gate_lower=-5）· `kda-gva-shaped`（GVA + 默认门）·
`kda-lowrank-shaped`（低秩门 + g_b_proj.bias）。

**配套两处**：`delta_rule` 的 decay 尾维从「只有 1」放宽到「1 或 dk」——
ref 本来就支持逐维（`g64[:, :, i][..., None]` 的广播），**by1c 的 C 内核
和发射器守卫跟上**（gk 参数）；`_ref_delta_rule` 的 docstring 也改了
（原来写死 `[b, 头, n, 1]`，KDA 是 `[b, 头, n, dk]`）。

**判卷人（e2e 图求值 vs by1exec，同权重同输入）**：

| 用例 | 相对差 | 支路 |
|---|---|---|
| `kda-shaped` | 7.396e-09 | gate_lower=-5 |
| `kda-gva-shaped` | 6.431e-09 | GVA（nk=2, nv=4）+ 默认门 |
| `kda-lowrank-shaped` | 6.848e-09 | 低秩门 + bias |

**一个坑**：beta 是 `sigmoid(b_proj @ x)`，第一版把**裸投影**喂给了递推
—— 形状全对、递推照跑，差 2e-03 起步。逐值定位（`_dbg_kda.py` 那套
中间量重放）第一个分叉点就是 `bh`。GDN 的 beta 有 sigmoid、KDA 的忘了：
**同一个公式在两个机制里出现，就得在两个地方都被盯住**。

普查跟着动：**KDA 7 份全部出列**（ling-tiny / Ling-3.0-tiny / GLM-5.3-Flash
/ glm-text-tiny 前进到 MLA 或 noaux_tc）。golden **30 → 33 张**（只增 3 条）；
e2e 小预算 **10 → 13 份一致**。C 图路径顺带吃下三份 KDA（见 A3-3 表）。

### A2 补：MoE 选法 sigmoid_topk + topk_softmax（2026-10-12）

四种路由**各是各的选法**（`ir.md 130` 那一课）。这一刀收编两支，照
`by1exec.op_moe` 605-623 逐句：

    sigmoid_topk（Step-3.7）：
        sc = score(logits)          # 打分吃**未加偏置**的 logits
        choice = sc + score_bias    # 只影响选择
        topi = top_k(choice)；tv = gather(sc, topi)
        topv = tv / (Σtv + 1e-20)   # 守卫是 **+**，不是 max

    topk_softmax（GPT-OSS）：
        topi = top_k(logits)        # 不看 score_bias、不经过打分函数
        topv = softmax(gather(logits, topi))

**新原语 `add_scalar`（a + c）**：`+ 1e-20` 和 softmax_topk 的
`max(·, 1e-9)` 是两个公式（Σ < c 时分叉），描述层不许合并成一条 ——
闭集加一条：公式 + ref() + 手算对照（含「和 clamp_min 分得开」的反例）
一起进来。

**判卷人（单算子图求值 vs by1exec.op_moe，真属性 + 随机权重）**：
`moe-routing-shaped` 的 sigmoid_topk **相对 4.917e-07**；mixtral 的
softmax_topk **5.341e-07**（回归对照，与 e2e 的 4.269e-10 同一条链）。
单算子图是 by1opdiff 的做法搬到图上 —— 整模型 e2e 被后面的
sigmoid_group_topk 层挡住时，**这一层的证不许跟着丢**。

普查跟着动：Step-3.7-Flash ×2 **整份降得动**（golden 33 → 35，798/723 GB，
数值证挂「租机跑全」）；gpt-oss 全系从 topk_softmax 前进到 **expert_bias**；
moe-routing-shaped / Ling 系停在 **sigmoid_group_topk**（要 noaux_tc 的
分组排名 + one-hot 那两步原语）。

### A2 补：MoE 选法 sigmoid_group_topk（noaux_tc，2026-10-12）

最后一支选法，照 `by1exec.op_moe` 585-604 逐句：**先打分** -> 组内**前 2
之和**排名选 `topk_group` 个组 -> 组外填 -inf -> top-k -> 权重用**未加
偏置**的 sc -> `tv / (Σtv + 1e-20)`。

**两个新原语**（都是通用的，公式 + ref() + 手算对照一起进闭集）：

- `one_hot`：下标行散成 0/1 行（`put_along_axis(gm, gi, 1.0)` 那一步）；
- `masked_fill`：`out = m > 0 ? x : v`（v 常是 -inf）。

**一处「拼出来就不加原语」**：`sort(gs)[..., -2:].sum(-1)` 没有加排序原语
—— `top_k(2) -> gather -> sum` 就是它（两个加数交换次序在 IEEE 里逐位
相同）。组掩码的展开也是现成的：`view -> repeat`（np.repeat 的整块重复，
GDN 的 GQA 广播同一个原语）。

**判卷人**：`moe-routing-shaped` 是**三层路由探针**（sigmoid_topk × 2：
sigmoid / sqrtsoftplus + noaux_tc），**整模** e2e **相对 3.783e-10** ——
一支判卷人盖三支选法、两种打分函数。C 图路径同模 **4.908e-07**
（融合核基线 4.370e-07），one_hot / masked_fill 的 C 内核一并验过。

普查跟着动：moe-routing-shaped 进 golden（**35 → 36 张**）；**Ling 系
（ling-tiny / Ling-3.0-tiny）的 MoE 通了，全部停在 MLA** —— MLA 成为
绝对大头（7 份），下一刀就是它。

### A2 补：MLA 收编（非稀疏，2026-10-12）

MLA（Multi-Latent Attention）—— 照 `by1exec.op_mla` 逐句：q / kv 各一条
**低秩分解**（a_proj -> layernorm -> b_proj），`kv_a` 出口切出**单头**的
k_rot 走 rope 再广播到 nh 头（`repeat` 原语，GDN 的 GQA 广播同一个），
拼回 `[q_pass | q_rot]` / `[k_pass | k_rot]` 走普通因果注意力
（scale = 1/√(nope+nr)），输出门 head_gate（per_head）+ dense。

**全链都是已收编原语** —— 一个新原语都没加（matmul / rms_norm / slice /
view / rope / repeat / concat / batch_matmul / soft_max / head_gate）。
**SparseMLA（`index_heads`）明确拒绝**：DSA 索引器还没收编（判卷人是
transformers 的 `GlmMoeDsaIndexer`）—— 要么实现，要么拒绝。

**判卷人**：`mla-shaped`（含 head_gate=sigmoid + interleaved rope）
**整模 e2e 2.267e-08**；`ling-tiny`（**KDA + MLA + noaux_tc MoE 合体**）
**2.564e-08** —— 合体模型把前几刀连起来又验了一遍。C 图路径：
mla-shaped **6.876e-07 与融合核逐位同一个数**；ling-tiny 6.369e-07
（融合核 7.598e-07）。

普查跟着动：golden **36 → 40 张**（mla-shaped / ling-tiny / Ling-3.0-tiny
/ GLM-4.7-Flash）；Ling 系全停在 **SparseMLA 或大模型**（Ling-3.0-tiny
31.57 GB / GLM-4.7-Flash 122.31 GB，数值证挂「租机跑全」）。
第一拒绝剩 **12 份**：index_heads 3 · expert_bias 3 · SSM 2 ·
codegen 2 · qk_norm 1 · Raw 1。

### A2 补：MoE expert_bias + gptoss 激活（2026-10-12）

gpt-oss 全系的两块缺口，照 `by1exec.op_moe` 647-655 逐句：

- **expert_bias**：b1/b3 加在激活前、b2 加在 w2 之后 —— 偏置也是**按
  下标整行取**（和权重同一个 `get_rows` + `add`，零新原语）；
- **gptoss 激活**：`g` 先 `min(g, lim)`、`u` 先 `clip(u, ±lim)`，
  `h = (u+1) · (g · sigmoid(α·g))` —— 全部用现有原语拼（scale /
  sigmoid / mul / add_scalar）。

**新原语 `clamp_max`（min(x, m)）**：`clamp_min` 的对称面，limit 钳制
要它（`clip` = clamp_min + clamp_max 两夹）。op_moe 对**别的 act 静默
走 silu** —— lowering 不跟这条，只收 silu / gptoss，其余拒绝。

**判卷人**：`gpt-oss-shaped`（topk_softmax + router_bias + expert_bias +
gptoss + limit=7）**整模 e2e 1.269e-10**；C 图路径 **2.965e-07**
（融合核基线 2.558e-07）。golden **40 → 43 张**（gpt-oss-shaped / 20b /
120b 全部降得动；后两份太大，数值证挂「租机跑全」）。

第一拒绝剩 **9 份** —— MoE 这一族**全部出列**：index_heads 3 · SSM 2 ·
codegen 2 · qk_norm 1 · Raw 1。

### A2 补：SSM 收编（Mamba2 递推，2026-10-12）

SSM（选择性状态空间，Nemotron-H 那一版）—— 照 `by1exec.op_ssm` 逐句：
融合投影切三段（gate / bc / dt）-> 因果深度卷积（+ conv.bias）+ silu ->
切 xs / B / C（B、C 从组**整块重复**广播到头）-> **递推** -> 门乘 ->
分组 gated norm（mean 版 = l2_norm(x, gs·ε)·√gs）-> out_proj。

**递推是新原语 `ssm_scan`**，理由和 delta_rule 一样：状态一步一步走、
没有并行公式，递推就是原语不是组合：

    st = exp(d·A)·st + d·(x ⊗ B)     # 先衰减，再**外积**写入
    y  = st·C + D·x                   # 读出是 h 右乘 C

公式 + ref()（float64 累，照 op_ssm）+ 手推两步的手算对照一起进闭集；
C 内核 `p_ssm_scan` 同步进机械表。

**判卷人**：`ssm-shaped` **整模 e2e 9.682e-09**；C 图路径 **4.349e-07**
（融合核基线 4.662e-07）。golden **43 → 44 张**（既有条目一字未动）。

**顺带撞见一个静默 bug**（单独立案）：Nemotron 的 MoE 前进到
`MoE.act=relu2` —— 而 `op_moe` 的 else 分支**对非 gptoss 一律算 silu**，
PyTorch 那边却走 `_swiglu(style)`（relu2 = relu(x)²）—— 两端不一致，
Nemotron 的 MoE 在两个后端会算出两个东西。见下一节。

### 修静默 bug：MoE 激活取值三后端一致化（2026-10-12）

**症状**：`act` 的 enum 里有 `relu2`（Nemotron 的 MoE 就是它），但
`op_moe` 的专家激活对非 gptoss **写死 silu**；PyTorch 的 `_swiglu`
**同样写死 silu** —— **两个后端一致，而两个都错**（`op_kda` 注释里
点过名的那一族病）。共享专家支更散：torch 走 `_swiglu(style,
limit_shared)`，numpy 写死 silu 且不夹取值。

**判卷人（公式层逐 style 对拍，两后端 helper 直接比）**：

| style | 修前相对差 | 修后 |
|---|---|---|
| silu / gptoss | 2.4e-08 / 8.2e-09 | 一致（不变） |
| **relu2** | **6.398e-01** | **0.000e+00（逐位）** |
| gelu / gelu_new | 1.221e-01 / 1.222e-01 | 9.5e-08 / 8.7e-08 |

**修了四处**（同一个 bug 在四处，四处一起改）：
① `by1exec.op_moe` 专家激活走 `_act_apply`（和 op_ffn 同一份），
   共享专家同 style + `limit_shared` 夹取值；② `by1codegen._swiglu`
   标准支走 `_act`（和无门控 FFN 同一份）；③ **by1c 融合核只有
   silu/gptoss 两支 —— 表外取值显式拒绝**（不许静默算成 silu），
   gptoss + 共享专家也拒（融合核的共享支没有 limit_shared）；
④ 图的 `_lower_moe` 激活分派（relu2 = `clamp_min` + 平方两节点拼），
   专家与共享两支同规则。

**回归**：golden 既有条目一字未动；by1opdiff 逐算子 [PASS]；图路径
relu2 + 共享专家单算子 vs op_moe **1.174e-07**。**Nemotron 整份降得动**
（golden **44 → 45 张**，30B 数值证挂「租机跑全」）。

### 修静默 bug：qk_norm=full 的宽度与位置（2026-10-12）

**症状**：`qk_norm` 的取值 `full`（Instella）是**拆头之前按整宽归一**
（`q_norm.weight` 实测 [2560]，`.by1` 里就写着这句），`per_head` 是
拆头之后按 head_dim（[80]）—— 名字一样、位置不同、宽度也不同。而
`op_attention` **不分 full**，一律按头算；`params_of` 也推 (80,)；
C 融合核的 `qk_norm()` 同样按头 —— **三方一致地错，唯独 torch 对**。

**判卷人两个**：
① `by1opdiff` 对 Instella 修前**直接崩**（`shape '[2560]' is invalid
   for input of size 80` —— 两后端参数形状对不上），修后**逐算子全对**
   （Attention 相对 4.656e-07）；
② `by1sync`（契约 ↔ shapes_of 同源报表）一直把这处列在「未编码的布局
   差」里 —— 修后 **1 份 → 0 份，那行归零**（它不是布局差，是语义差，
   分类错了一直没人喊）。

**修了四处**：`by1ir._qk_norm_shapes`（唯一定义，full → (q*hd)/(kv*hd)）·
`op_attention`（full 在拆头前归一，照 torch 的位置）· by1c 融合发射
**显式拒绝 full**（融合核按头，不许静默算成 per_head）· 图
`_lower_attn` 的 full 分支（flat 归一 + full+q_gate 拒绝）。

**新证**：图路径 full 单算子 vs by1exec **1.160e-09**；Instella 整份
降得动（golden **45 → 46 张**，3B 数值证挂「租机跑全」）。

### A2 补：SparseMLA 非池路径（DSA 索引器用原语拼，2026-10-12）

DSA 稀疏索引器（`by1exec._dsa_index` 的循环 + 散射）**没有加原语**，
用闭集拼了出来：`le` 外积做因果比较 -> `masked_fill` -> `top_k`（稳定，
并列取小）-> **`one_hot` 散成 keep 掩码**（循环版的两层
`keep[bi, ti, sel] = True` 就是这个散射）-> `masked_fill` 进注意力分数。
**两份实现独立**（循环版 vs 组合版），e2e 就是它们的对拍 —— 单算子
**5.193e-09**，**C 机械表 7.863e-08**（`le` / 掩码广播 / one_hot 的 C 侧
一并证了）。

**三个配套改动**：
① 新原语 `le`（`out = 1 若 a <= b`，掩码族第三员）；
② **`one_hot` 改为宽度参照**（第二输入的最后一维给宽度）—— DSA 的
   宽度是运行期的 n，静态 attr 表达不了；
③ `qk_rope = 0`（纯 nope）收口：空切片不 rope 不拼，和 op_mla 的
   `if nr:` 同构（glm-text-tiny 就是这个配置，空切片撞 reshape -1 会炸）。

**golden 的账（如实记）**：+2 新条目（GLM-5.3-Flash —— 它的索引器也是
非池、glm-text-tiny）；**5 条既有条目改了图格式**（GLM-4.7-Flash /
Ling-3.0-tiny / Nemotron / ling-tiny / moe-routing —— 全是 noaux_tc 模型，
`one_hot` 加宽度参照输入所致）。语义未变的证：可跑的两份**逐位同一个
数**（ling-tiny 2.564e-08 · moe-routing 3.783e-10），大模型那三份是同
一处机械变换、数值证挂「租机跑全」。

整模新证：**glm-text-tiny（KDA + SparseMLA + noaux_tc 合体）
1.111e-08**。第一拒绝剩 **4 份**：k-pool 1 · codegen 2 · Raw 1。

### A2 补：MoE 的 `softmax_topk` 那一刀（2026-10-12）

普查说 MoE 有四种选法：`softmax_topk` **11 份** · `sigmoid_group_topk` 6 ·
`sigmoid_topk` 3 · `topk_softmax` 3。**先切最大的那一刀**，且它最简单：

    sel  = logits + score_bias          # **只影响选择**
    topi = top_k(softmax(sel))          # 先全体 softmax 再选
    tv   = gather(softmax(base), topi)  # **权重用未加偏置的 logits 重算**
    topv = tv / max(sum(tv), 1e-9) · routed_scale
    out  = Σ_k topv[:,k] · Expert_k(x)  # **残差在外面**，不在这个算子里

**专家那一步没加新原语**：`get_rows(W1, ids)` 整行取到 `(n,K,H,d)`，
`view(x,[d,1,1,n])` 把它摆成 4 维（numpy 的 matmul 批维**从右对齐**，
3 维对不齐），再走现有 `matmul`。**能拼出来就不加原语。**

其余三种**明确拒绝**（要么实现、要么拒绝）：`sigmoid_topk` 要"先打分再
直接 top-k"，`sigmoid_group_topk` 要 noaux_tc 的分组排名，`topk_softmax`
要 gptoss 激活与 `expert_bias`。

**判据**：`mixtral-shaped`（8 专家 · top_k 2 · shared=0）e2e **4.269e-10**；
可降 **14 → 19 份**；golden **19 张图**逐字节稳定；
`by1all --tier 4` **113 项 · 0 失败**。

**两个坑记在 `ir.md 135`**（都不是公式错）：① `x @ w.T` 只在 2 维上正确
（>2 维 `.T` 反转**所有**轴）；② 残差在算子里还是在算子外 —— 差一次就是
**7.756e-01**，而形状全对。

### A2 补：YaRN 按「搬家，不照抄」收编（2026-10-12）

`rope_tables` 的 `inv` 公式原本有**三份**：`by1exec` 一份、`by1codegen`
一份 torch 版、图的 rope 原语又把普通那一支**内联**算了一遍 ——
正是 `ir.md 130` 记的「一条规则三处实现 = 迟早对不上」。

搬法：`inv` 的唯一家在 `by1graph.rope_inv(n_dims, base, yarn)`
（普通 / YaRN / llama3 三支逐字搬自 by1exec），`by1exec.rope_tables`
改为调它；图的 `rope` 原语也吃它（顺带收编 `rope_scale`）。
torch 那一份仍由 `by1rope.py` 盯着 —— **三份变两份**。

**判据（逐位，不是量级）**：搬之前把 (head_dim 64/96/128 × base 3 档 ×
{无, llama3, yarn}) = **27 个组合**的 cos/sin 表都取出来对过：
**逐位相同 27 · 不同 0**。

**顺带**：`e2e` 对"降不动"的模型原来是 `except SystemExit: continue`
—— **静默丢弃**（点名两个只验一个，还印 PASS）。改成印出来**并算失败**；
这次 `gpt-oss-shaped` 的 MoE 缺口就是被它明确报出来的。
| A3-0 图求值器 `by1graph.run_graph` + `--e2e` | ✅ | 小份全绿（见下「在跑」的账）；大份挂"租机跑全" |
| **A3-1 weight plan 独立**（`by1ir.weight_plan`） | ✅ | **C 文本 sha256 对照：52 份 0 差异**；`by1fast` PASS |
| **A3-2 by1ggml 吃图**（逐节点 PRIMS → ggml） | ✅ | `--selftest` **与 IR 路径逐位相同**（2.098e-07 / 1.317e-07 / 1.827e-07） |
| **A3-3 切片 1–3：by1c 吃图**（PRIMS → C 机械表：核心六类 + MoE + GDN 族） | ◐ 切片 3/4 | 10 份模型对拍全绿，改前改后数字见下（默认路径未切，融合核仍是判卷人） |

### A3-2 做了什么（2026-10-12）

**做法**：`emit_model_c_graph(g, ir, order, params)` —— 逐节点把 PRIMS 翻成
ggml 调用。**图里没有"机制"分支**：Attention / FFN 早被 lowering 拆成了
matmul / rope / soft_max 这样的原语，所以这里只剩一张机械表。

**共用的三件**（抽成小函数，两条路径同条件，对拍才有意义）：

    _head_lines(ir, order, params)   常量 + OFF_/LEN_（布局来自 weight_plan）
    _weight_lines(order, params)     权重张量
    _mask_lines(wins)                因果 + 滑窗掩码（host 侧填好）
    _budget(ir) / _assemble(...)     预算与骨架

**判据（差分对拍）**：同一个 `--selftest`，两条发射路径**数字逐位相同** ——

| 用例 | 图路径 | IR 路径 |
|---|---|---|
| `llama-shaped` | 2.098e-07 | 2.098e-07 |
| `gpt2-tiny` | 1.317e-07 | 1.317e-07 |
| `_gemma-shaped` | 1.827e-07 | 1.827e-07 |
| `gemma-4-31B` | 编译过（950 张量） | 同 |
| `qwen3-next-shaped` | 按承诺拒绝 ✓ | 同 |

**默认已切到图路径**；`--from-ir` 保留旧路径，**唯一用途是差分对照**
（A3-3 之后删掉 —— 别让两份发射器长期并存）。
`graphs-golden.json` **一个字没动**（lowering 没改）。

**门（环境补齐之后）**：`by1all --tier 5` = **129 项 · 0 失败 · 2 跳过 ·
1 已知缺口 [全过]**，其中 **ggml 后端 [PASS]**（跑的就是新的图路径）。

### A3-3 切片 1：by1c 吃图（2026-10-12）

**做法**：照 A3-2 的样子加第二条发射路径 —— `by1c.emit_c_graph` 逐节点把
PRIMS 翻成 `C_PRIMS` 里的 C 内核调用（**一张机械表，图里没有机制分支**）。
内核逐条对着 `by1graph.PRIMS` 的 ref()（可执行规格）写；`rope_inv` 的
逆频率表由发射器按唯一定义算好**烤进 C**（不在 C 里再算一遍）。
形状**从 PRIMS 的语义推**（numpy 右对齐广播那一套），不读 `g.vals` 的
声明 —— 声明里有推不出东西的（Head 的 `shapes=[(None, None)]`），
两份事实并存就是 `ir.md 130` 那个病。

`--from-graph` 显式走新路；**默认还是融合核**（迁移期间它仍是判卷人）。
覆盖不齐就互换不了 —— MLA / KDA / SSM 连 lowering 都还没有（A2 的账），
它们的图根本不存在，所以**默认的互换挂在 A2 补齐上**，不挂在本切片上。

**判卷人（C↔NumPy 相对差，改前改后都记）**：

| 用例 | 融合核（改前） | 图路径（改后） | 族 |
|---|---|---|---|
| `llama-shaped` | 5.266e-07 | 5.281e-07 | 核心六类 |
| `gpt2-tiny` | 3.977e-07 | 3.977e-07 | 核心六类 |
| `hello` | 4.660e-07 | 4.486e-07 | 核心六类 |
| `minimind-3` | 1.554e-06 | 1.554e-06 | 核心六类 |
| `attn-fused-shaped` | 4.704e-07 | 4.376e-07 | 核心六类 |
| `llama3-shaped` | 9.059e-07 | 1.117e-06 | partial rope |
| `_gemma-shaped` | 1.857e-07 | 1.960e-07 | partial rope |
| `mixtral-shaped` | 4.280e-07 | 4.017e-07 | MoE |
| `qwen3-next-shaped` | 1.017e-06 | 9.867e-07 | MoE + GDN |
| `clef-tiny` | 6.633e-07 | 7.310e-07 | GDN |
| `kda-shaped` | 4.426e-07 | 4.000e-07 | KDA |
| `kda-gva-shaped` | 3.322e-07 | 4.608e-07 | KDA |
| `kda-lowrank-shaped` | 4.554e-07 | 5.934e-07 | KDA |
| `moe-routing-shaped` | 4.370e-07 | 4.908e-07 | MoE 三支选法 |
| `mla-shaped` | 6.876e-07 | **6.876e-07** | MLA（**逐位同一个数**） |
| `ling-tiny` | 7.598e-07 | 6.369e-07 | KDA + MLA + MoE 合体 |
| `gpt-oss-shaped` | 2.558e-07 | 2.965e-07 | MoE gptoss + expert_bias |
| `ssm-shaped` | 4.662e-07 | 4.349e-07 | SSM（Mamba2 递推） |

全在 1e-4 判据内 [PASS]；gpt2-tiny / minimind-3 **逐位同一个数**。
`llama3-shaped` 1.117e-06 比融合核 9.059e-07 略高（无融合循环的累加序不同）
—— 如实记，别抹掉。

**三个坑（都不是"公式没写对"，是"公式写对了但另一处没跟上"）**：

1. **形状 ≠ 步长**：`p_matmul` 第一版把 dims 当 strides 用 —— 2 维时
   `(n, 128)` 恰好一路带病走到段错误。索引用后缀积，别偷懒。
2. **partial rope 的未旋转维度**：`_ref_rope` 是 `out = 复制 x` 再改
   旋转对；第一版直接写输出、其余留 calloc 的 0。全量 rope 看不出来，
   `partial` 一来就差 1e-2 ~ 1e-4 —— `llama3-shaped`（partial=0.5）和
   `_gemma-shaped`（partial=0.25）**两处同时露头**，修一处好两处。
3. **注释公式和 ref 实现对不上**：`_ref_conv1d_causal` 的注释写着
   `xpad[:, t+(kernel-1-i)]`，实现是 `mp[:, i:i+s] · w[:, i]` —— **核差一次
   翻转**。C 内核照注释写，clef-tiny / qwen3-next 差 3e-01。判卷人是
   `by1exec`（实现逐句对过），所以**改的是注释**（`by1graph` 已修）——
   但这条要记着：**注释也是规格，写错了照样咬人**。

**定位用的家伙（这轮用完删了，方法留下）**：让发射器把每个值缓冲
逐个 dump 出来，和 `by1graph.run_graph` 的 `V` 逐值比 —— 495 个值里
**第一个**分叉点就是病根（`L0O1_c3`），比看最终 logits 快一个量级。
B1 的 spec runner 就该长这样，只是常驻。

**下一刀**：A2 最后一块 = **k-pool 路径**（mla-sparse-shaped 的 L2/L3：
池打分 + 摊回 token + tail；要 arange 系的动态构造）—— 之后才谈
**默认互换 + 删 `by1ggml --from-ir`**。图路径 C 对拍已绿 **18 份**；
SHAPED 14 份里盖住 **12 份**。

### 本机环境（2026-10-12 补齐，**换机器照这条装**）

    python  3.13.14（venv: ~/.workbuddy-ai/binaries/python/envs/default）
    numpy   2.5.4（本来就有）
    torch   2.14.1+cpu      pip install --index-url https://download.pytorch.org/whl/cpu torch
    transformers 5.15.1     pip install transformers==5.15.1
    ggml    独立仓 llamacpp/ggml + build-ggml（见上面 A3-2 那段）

**版本要钉 5.15.1**：5.19.0 的 `GlmMoeDsaIndexer` 对 `attention_mask=None`
不再容忍，`by1sparse` 当场 `AttributeError`（不是我们的 bug）。
仓库文档里引用的权威语义本来就是 5.15.1（见 `ir.md 132`）。

**装之前档 4/5 的表现**（记下来，免得下次误判成代码坏了）：

    档 4  113 项 · 26 失败（全 torch）· 25 检查器坏了
    档 5  129 项 · 32 失败（全 torch）

**装之后**：

    档 4  113 项 · **0 失败** · 7 跳过 · 1 已知缺口   [全过]
    档 5  129 项 · **0 失败** · 2 跳过 · 1 已知缺口   [全过]（含 ggml 段）

**教训**：一台机器的"红"可能是**环境缺件**，不是代码坏了 ——
`by1all` 对缺 torch 的项**有的印跳过、有的印失败**（`by1irentry` 的 ② 号
步骤直接 import torch，于是整条 FAIL）。**先把环境补齐再读红绿。**

提交：`1c6020f`（T0.1/T0.2）· `5fee485`（T0.3）· `801392b`（A1/A2）。

### A3-1 做了什么（2026-10-12）

**问题**：`w.bin` 的布局表（参数名 + 顺序）长在 `by1c._emit_c_body` 里，
于是 `by1ggml` 为了拿**同一份**布局只能反向 import by1c ——
`decls, order, total = c_backend.emit_c(...)` 然后 `del decls`，
**`del decls` 就是"我只想要布局，不想要 C"的那个证据**。
布局是 IR 的事实，不该由某个后端代管。

**做法**：搬进 `by1ir.weight_plan(ir)`，返回 `[(宏名, 参数键)]`，
顺序就是 `w.bin` 的排布顺序。四处配套：

1. **不变量成了硬断言**：键集 == `params_of(ir)` 的键集且无重复。
   这条**当场抓到一处真遗漏** —— `params_of` 为 `head_gate` 声明了
   `g_proj`，而搬过来的表里没有（旧代码在 head_gate 上直接拒绝，
   所以从没暴露）。三份 head_gate 模型立刻报"两份事实漂移"。
2. **能力拒绝与布局分开**：`C_KINDS` 显式写出来（原来藏在 if 链最后
   那个 `else` 里），预扫**跑在布局之前** —— 否则 head_gate / Raw
   会以 `ValueError` 失败，报错长得像 IR 有问题。
3. **顺手删掉一处死代码**：旧循环里的 `body`（9 处 append，**0 处读**）。
4. `by1ggml` 改为直接取 `by1ir.weight_plan`，不再依赖 by1c。

**判据（零编译、零显存）**：把每份模型 `emit_c` 的产物
（C 文本 + order + 总长）取 sha256，重构前后对照：

```
52 份：相同 52 · 不同 0 · 新增 0
（46 份可发射逐字节相同；6 份跳过/报错的报文一字不差）
```

另：`by1fast` PASS（编译 77 · import 72 · 解析 53 · 判卷人 5 · 行尾 149）·
`by1ir --selftest` PASS（3 反例红 + 50 模型绿）。
**数值档没重跑**，理由是 C 文本逐字节相同 ⇒ 生成的二进制相同 ⇒ 数值不可能变；
ggml 侧同理（`order` 相同）。

### 在跑（未验完，第一件事就做它）

**A3-0 图求值器**：`by1graph.run_graph`（用 PRIMS 参考实现跑图）+
`by1graph.py --e2e`（图求值 vs by1exec，同权重同输入逐位）。代码已写完，
途中修了六个语义坑（权重叶子用真参数名 · 引用→产出名映射 ·
`batch_matmul` 从 `matmul` 拆出（一原语两公式正是要灭的病）·
soft_max 归一轴 · 掩码尺寸取轴 · **cont 带 shape = 物化+重排**）。

#### ⚠ e2e 的内存代价（2026-10-12 实测，**先看这段再跑**）

`e2e()` 对 golden 里每一份模型**按全尺寸铺 float32 随机权重**，于是
内存足迹 = 参数量 × 4 字节。golden 那 9 份里有真实大模型：

| 模型 | 张量 | 参数 | float32 足迹 |
|---|---|---|---|
| hello / llama-shaped | 39 | 1 M | ~0.00 GB |
| minimind-3 | 91 | 69 M | 0.28 GB |
| embeddinggemma-2 | 551 | 738 M | 2.95 GB |
| DeepSeek-R1-Distill-1.5B | 367 | 1.78 G | 7.11 GB |
| DeepSeek-R1-Distill-7B | 367 | 7.62 G | 30.46 GB |
| gemma-4-E4B | 749 | 5.64 G | 22.56 GB |
| gemma-4-12B | 523 | 12.9 G | 51.65 GB |
| **gemma-4-31B** | 950 | 32.7 G | **130.60 GB** |
| **合计** | | | **245.63 GB** |

**在 16 GB 的机器上跑裸 `--e2e` 会打爆页面文件、把系统冻住**
（2026-10-12 实测冻了两次机器 —— 不是报错，是整机无响应）。

修法（已进 `by1graph.py`）：`e2e()` 先算参数足迹，**超预算的模型显式跳过**
（印出"需 X.XX GB > 预算 Y GB"），默认预算 `E2E_MEM_GB = 4.0`，
`--mem-gb N` 覆盖；`--e2e 模型名...` 可只跑指定的几份。
**跳过是印出来的，不是静默的**（`ir.md 133` 记的正是静默跳过这一类）。

**换机后第一条命令**（按本机内存挑预算；16 GB 机器用 1）：

```
python src/by1graph.py --e2e --mem-gb 1   # 小份全绿才算机制通
python src/by1graph.py --golden --write   # lowering 若再动，重生成
```

**已知局限（A3 的账，别当成已完成）**：预算卡住 6 份，其中
`gemma-4-12B/31B/E4B` 带的正是**滑窗 / `layer_scalar` / K=V 无 `v_proj`**
这几处"语义住在四处"的属性 —— 它们不跑，e2e 就**没有**为那几处做证。

**决定（2026-10-12）**：不做缩比例孪生，**换租显卡机器后一次跑全**
（`--mem-gb 260`，合计 245.63 GB）。所以 A3 可以继续推进，
但这 6 份的证**挂在"租机跑全"这个未完成项上** —— 记在这里，不许当已验。

**当前实测**（2026-10-12，16 GB 机器，`--mem-gb 1`）：
`gpt2-tiny` 0.000e+00 · `gpt2` 0.000e+00（逐位）· `hello` 3.394e-10 ·
`llama-shaped` 4.462e-10 · `minimind-3` 6.468e-10
—— **5 份一致**，6 份超预算跳过（合计 245.34 GB）。

### A2 补：学习式位置表（2026-10-12）

`lower_model` 原来对 `pos_kind = learned` **直接拒绝**（"还没收编"）——
于是 `gpt2-tiny` / `gpt2` 降不了，而 **`gpt2-tiny` 正是 `by1ggml --selftest`
的 ② 号判据**（layernorm / 学习位置 / gelu_new / 无门控）。
**A3-2（by1ggml 吃图）就卡在这一条上。**

语义照 `by1exec`（`x = x + pos.weight[:n]`）：求值器给的 `pos` 就是
`arange(n)`，所以 `get_rows(pos.weight, pos)` 与 `pos.weight[:n]` 逐位相同 ——
实测 **0.000e+00**，不是"量级对"，是**每一位都一样**。

**这一步是 A3-2 的前置，不是旁支**：判据要的 5 个模型里，
① llama-shaped ✓ · ② gpt2-tiny（本次解锁）· ③ gemma 孪生 ✓ ·
④ gemma-4-31B ✓ · ⑤ qwen3-next-shaped 必须仍被拒 ✓。

e2e 绿了才算「唯一语义 = 现有语义」，**别跳过它直接做 A3 的后端重写**。

### 下一步（顺序照旧）

1. ~~**A3-1 weight plan 独立**~~ ✅ 2026-10-12（见上）。
2. **A3-2 by1ggml 吃图**：发射改为遍历图节点（PRIMS → ggml 调用的机械表），
   判据 = graphs-golden.json 不变 + `by1ggml --selftest` 数值不劣化
   （本机基线见下）。
   ✅ **ggml 本体已在位**（2026-10-12）：走 **gh-proxy.com 镜像** + **ggml 独立仓**
   （`llamacpp/ggml`，4.1 MB；llama.cpp 的 tarball 在 ~42 KB/s 下要 20–30 分钟，
   而 ggml 早就独立成仓）。`llamacpp/build-ggml/src/` 有 `ggml.a` / `ggml-base.a`
   / `ggml-cpu.a`（**不带 `lib` 前缀**，链接用显式路径 + `-fopenmp`）。
   `by1ggml.ggml_dirs` 加了兜底：`llamacpp/llama.cpp/ggml` 不在就用
   `llamacpp/ggml` —— 所以 `--selftest` 不用再带 `--ggml-src`。

   **本机基线（2026-10-12，gcc 16.1.0 / WinLibs，`-march=native`）**：

   | 用例 | 相对差 | 文档里的旧值 |
   |---|---|---|
   | `llama-shaped` | **2.098e-07** | 2.962e-07 |
   | `gpt2-tiny` | **1.317e-07** | 1.379e-07 |
   | `_gemma-shaped` | **1.827e-07** | 1.390e-07 |
   | `gemma-4-31B` | 编译过（5914 行 · 950 张量 · 图预算 3553 节点） | 同 |
   | `qwen3-next-shaped` | 按承诺拒绝 ✓ | 同 |

   **换机器数字会动**（编译器/`-march=native` 不同），所以判据是
   「**不劣化**」而不是「等于某个数」——A3-2 改完拿本机这五个数对照。
   M0 的判卷人 `tools/ggml-smoke.c` 已在本机 [PASS]（mul_mat/rms_norm/silu 对**手算**）。
3. **A3-3 by1c / by1exec 吃图**，A4 对拍矩阵全不退（by1opdiff 8 文件 +
   shaped 全套 + by1e2e）。**这条要 gcc**（本机有，但要占 CPU 一会儿）。
4. **B1-B3** 数值规格（by1graph.PRIMS 的 ref() 就是可执行规格的地基）、
   精度预算表、criteria-review 重跑。
5. **C1-C2** 导出命名反推。最后走 rc0 发版动作（全档 → `by1ver --write`
   → history 记数字 → tag）。

**A3-1 的改动还没提交** —— 门只跑了 `by1fast`（秒级）。
提交前要补 `by1all --tier 3`（98 项，含 gcc 编译）。

### 备忘（换机可能踩的）

- ggml 本体不进 git：`llamacpp/llama.cpp/`（codeload tarball 拉），
  构建 `llamacpp/build-ggml/`（MinGW + Ninja）；见 `docs/llamacpp-plan.md` M0。
  rc0 **不需要** ggml（by1ggml 的数值对拍要，跑 `src/by1ggml.py --selftest` 才用）。
- 环境：Windows / MinGW gcc 16.2 / python 3.12+；档 1-3 不要第三方依赖。
- 门禁基线：by1fast（编译 77 · import 72 · 解析 53）· by1all 档 2 = 36 项 0 失败 ·
  档 3 = 98 项 0 失败（1 项已知缺口在账）。

---

## 发版记录

### 0.9.1（补丁，不占号）—— 2026-10-14 发

**内容**（时刻表第 0 步，描述↔计算闭环三件套）：T0.1 IR 形状/类型静态
推导（换宽度进 gaps 不进 errs）· T0.2 张量形状表收敛 `by1ir.params_of`
单一真相源（1152 bug 注回去必响）· T0.3 变异反例库 `by1mutate` 进门禁
（20 反例全红 + 基线全绿）。

**现量数字**：仓库 41,289 行（py 31,279 + .by1 9,864 + C 146）。

**门禁**：`by1all --tier 5` 全绿 —— 131 项 0 失败 · 2 项跳过（档 6 显卡，
本机无卡）· 1 项已知缺口在账（Step-3_7 的 25 张量：2026-10-14 复查确认
上游仓库已下架/转私有，HF 401 / MS 404，对拍永久缺口）。golden 48 张图
逐字节稳定。`by1pack` 绿 —— 顺手补了 WANT 的三笔传递依赖账
（by1full / by1graph / by1quant，closure_missing() 抓的）。

**tag**：`v0.9.1`
