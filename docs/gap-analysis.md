# 项目缺口盘点（2026-10-15 · rc1 后）

> 目的：把**在案缺口**一次说清：真 bug、证据缺口、范围缺口、环境缺口、计划缺口五类分开。
> 规矩：每条带证据与收口路径；"跳过"永远不写成"通过"；不外推。

## ① 真 bug（验证矩阵红项，在案未修）

| # | 缺口 | 证据 | 影响 | 收口路径 |
|---|---|---|---|---|
| B1 | **已修 2026-10-15** · T26 采样矩阵 8/144 红：`top_k=0 + top_p=1.0 + min_p>0` 族，`chain_f32` 与 sampler-probe 语义不一致（我侧出错 token） | 根因 = min_p **两条路径**：数组未排序时按**当前存储序**过滤、不排序（`llama_sampler_min_p_apply` 未排序路），`chain_f32` 曾在 min_p 前擅自排序 —— 保留集合没变，但 dist 的 f32 求和序变了，选中 token 全变。修后 `--selftest` **144/144**；且已进 by1all 档 2 判卷人段（ok [PASS]） | 修前该参数族不可信；T25 的 3 配置全过不能外推到全网格 | **两项都已收口**：144/144 + 进门禁。顺带修了 dist 的 max_l 取法（未排序时须 max 全扫，不能取 x[0]） |

> 注：by1sample 自检**已进 by1all 门禁表**（档 2 判卷人段，2026-10-15）—— 它原来不在表里，所以 B1 红了 8 格没人知道。同类欠账还剩 S5（by1langspec 未进门禁表）。

## ② 证据缺口（缺口单未销账，docs/multimodal-gaps.md）

| # | 缺口 | 状态 | 收口路径 |
|---|---|---|---|
| E1 | 多模态缺口单 **7 未销 + 1 立案** | 未销 | 逐条：A 系（夹具/建模）B 系（by1 侧建模）C 系（端到端）D1（llama.cpp embd 注入路径语义） |
| E2 | **D1 · llama.cpp embd 注入路径**（token=-1 批）与 HF 占位+swap 同值不等价（cos 0.58–0.7），token 路径等价 | 立案（上游语义） | 上游 issue/复现包；或改用 mtmd_helper 正式通路复测 |
| E3 | **音频端到端逐 token** 未做生成级对照（边界三段已逐位：mel 4.7e-6 / 塔 1.4e-04 / token 路径 8.8e-02） | 未销 | 输入模板对齐后双侧贪心生成对照 |
| E4 | **视觉端到端生成**没跑（塔边界已验到同精度等价） | 未销 | 同 E3 形态 |

## ③ 范围缺口（明确不外推/待接线，各有记档）

| # | 缺口 | 记档位置 | 说明 |
|---|---|---|---|
| S1 | build-gguf 的 **tokenizer KV 拿上游产物**，分词器组装线未实现 | docs/convert-diff.md #2 | 产物链只负责模型侧；非缺陷伪装 |
| S2 | **已销 2026-10-16** · build-gguf 族覆盖原只 qwen2 档实证（TensorNameMap 全族但判据只跑过一档） | 根因 = 判据没跑过第二族，且 `build_gguf` 的 arch 被硬编成 LLAMA。Qwen3-0.6B（qwen3）走完 `build-gguf --arch qwen3 --n-blocks 28`：311 张量全映射、上游 llama-eval-callback **logits 逐位相同（106949 值）**、llama-cli 生成同文 —— 见 docs/p0-loop-report.md | **第二族已实证**；其余族仍不外推（docs/convert-diff.md #8 已更新） |
| S3 | 量化链只接 **Q8_0**；FP8/MXFP4 = 路线图阶段 C/D | by1quant/by1llama | Q3 未开 |
| S4 | **换宽度**类只进 gaps 不进 errs（宽度推导换宽没验，49 份原宽回归绿） | by1fast ③ 报文 | 换宽夹具进 gaps 是设计决定 |
| S5 | **已销 2026-10-17** · 判卷人未进门禁表的欠账 | 原记 by1langspec 一笔；当天扫尾时又发现 by1sync（`params_of` vs `tensors{}` 两处真相对账，自带注入 1152 bug 自检）同一形状 | **三笔全销**：by1sample（当年 min_p 8 格红没人知道）· by1rules · by1langspec · **by1sync** 都已进 by1all 判卷人表（by1langspec/by1sync 在档 2）。`by1cmp.py` 是**报告工具**（无判定行），按约定不入表。档 2 = 43 项 · 档 3 = **105 项**，0 失败 |
| S6 | 命名生成率基线 **41%**（refs→emit 反推），只升不降 | 1.0.0-rc0 HISTORY | 长线指标 |
| S7 | 产品边界**明确不做**：通用运行时/自研 CUDA 量化器/改 GGUF 语义/覆盖计数 | docs/product-definition.md | 是边界不是缺口 —— 列此防误记 |
| S8 | **Step-5 合同缺判卷人**：上游 transformers（5.15.1 / 5.20.0）与新检出 llama.cpp **都没有 step5 实现**（5.20 有 `Step3p7*`、llama.cpp 有 `step35`） | docs/step5-contract-plan.md ③（实测表） | **停在规格级**（调度有判卷人：config 的 `layer_types`；契约没有）—— watch：上游合入 `Step5*` / `step5` 架构 / 官方 modeling 公开，任一出现就重启 |

## ④ 环境缺口（跳过 ≠ 通过）

| # | 缺口 | 证据 | 收口路径 |
|---|---|---|---|
| G1 | **GPU 全线**（by1gpu/by1gpusmoke 档 6）：GPU 前向/量化真跑/显存吞吐 | 档 5 实跑 2 项跳过 | 路线图"GPU 日"（租卡一次排完）；量化 Q2 的 FP8 前向依赖此 |
| G2 | **已销 2026-10-16** · 外部环境复现原无记录 | 外部复现 #1：Colab 上按 README 跑通 CPU 入门（4 层 1,049,728 / 2 层 557,696 / (1,8,256) 全对），并抓到 `head_dim=31` 缺口（当天已修）—— 报告与全文见 history/external-repro/2026-10-16-colab-hello.md | **已有 1 次记录**；发布门前仍需第二人复现。**审计包已备**（2026-10-17）：`history/external-repro/audit/run-audit.py` 一条命令跑出**可核对的产物**（环境指纹 + 退出码 + 判定行 + stdout SHA-256 + 与 `expected.json` 的逐条比对）—— 复核者跑完把 json 发回，**一分钟核完**，不用信口头报告 |
| G3 | **已销 2026-10-17** · CI 的"完整模型"腿曾因 Linux 独有红被移出 | 诊断腿（`continue-on-error`，注解公开）给出数字：`torch.equal(存盘前, 读回后)` 差 **max\|Δ\|=7.451e-08、无 NaN** —— 1 ulp 级，不是结构问题 | **判据改两层**：权重逐位（state_dict，safetensors 往返无损）+ 前向在浮点舍入内一致（容差 1e-6，实测余量 13 倍，差值每次都报）。腿已回 CI（`complete` job） |

## ④b 迁移缺口（ggml 图路 vs IR 路 · 2026-10-17 晚）

| # | 缺口 | 证据 | 收口路径 |
|---|---|---|---|
| ~~A3-3~~ **已销 2026-10-17（第三轮）** | 旧 IR 路（`--from-ir` + `check_support` + `emit_model_c`）**已删**：`by1ggml` 2277 → 1490 行；那 6 份只走旧路的探针（hello / attn-fused / ssm / mla / llama3 / attn-qgate）在图路上逐个数值过（1.2e-07~4.5e-07） | 图路 = 唯一一条；矩阵单列 `ggml`；档 5 = 152 项 0 失败（第三轮重新跑） | **没有覆盖率损失**，少的是"两份发射器"这个长期隐患；**差分对照没有被牺牲**：同轮把 `by1c --from-graph`（同图 -> 手写 C 内核）放进 6b 门禁（现在 20 份逐份对拍，8.4e-08 ~ 2.9e-06），它比原来"IR 路 vs 图路"更独立 |
| ~~A3-4~~ **已销 2026-10-17（第四轮）** | `mla-sparse-shaped` 的 **k-pool** 收进图路（`index_kpool > 1` + `index_kpool_tail`）：池号/池起止全从 `view(pos,[P,None])` 的**推断维** + 槽切片拼出，不需要 arange/整除原语 | 图路数值 **1.914e-07 绝对 / 3.210e-07 相对**（判卷人 `by1exec._dsa_index_pool`）；矩阵 `ggml` 列 ✅，可补格 **1 -> 0**；图求值 vs by1exec **3.572e-08** | 两处写清楚的限制：① 图路要求 `seq % kpool == 0`（reshape 推维要整除，不满足时 ggml reshape 当场断言）；② 图->C 那条对照**明确拒绝**（A3-5）。C 融合核路不受 ① 限 |
| ~~A3-5~~ **已销 2026-10-17（第五轮）** | 图->C 对 k-pool 的**明确拒绝**拆掉了：补了 float 下标内核（`p_get_rows_f` / `p_one_hot_f` —— 下标是 `pos` 的视图，值是整数、类型是 float）与**推断维形状**（`_view_shp`：`None` 不再写死成 n，而是 `源元素数 / 静态积`，能约回 n 就约） | `mla-sparse-shaped` 图->C **4.022e-07 绝对 / 6.746e-07 相对**；`GRAPH_C_MODELS` 回到 **20/20**（= `GGML_GRAPH_MODELS` 全份），两个发射器的差分对照**没有缺口** | 第四轮那个 NaN的根因就是 `None -> n`：池 gather 的整块形状算错，读出去就 NaN —— 所以"拒掉"当时是对的（静默 NaN 比拒绝糟），补上之后才该放开 |

## ⑤ 计划级缺口（路线图"当前基线"的下一项）

| 领域 | 下一项缺口 |
|---|---|
| 描述合同 | 持续收敛重复规则（多检查器重复维护同一规则）—— **2026-10-17 第六轮推进一格**：属性默认值收归 `by1ir.ATTR_DEFAULTS`（唯一源头），`by1lint` 第⑪类盯两件事：① 活代码里的取值点不许再写数字；② **生成物模板里的字面量必须等于源头**（这一条当场抓到 SSM 门控 norm 的 1e-6 vs 1e-5）。两条都有反例自检（第七轮：谓词 `qk_on` 也收归了——它曾在五个模块各有一份、四份逐字相同，现在只能在 `by1ir` 定义、`by1lint` 第⑫类盯着；顺带把"规则数"从五处手写扁为从自检表派生）（第八轮：**取值强制转换**也收归了——`as_bool`/`as_num` 曾有三种写法（`bool()` 把 `"false"` 当真、`_bool` 白名单把 `per_head` 当假），现在只有 `by1ir` 一份；新检查器 `by1coerce.py`（档 2）四关：取值表 / 语言层八种写法 /IR 门口拒字符串 / 开关确实改变图和数）（第二十二轮：**档位归属变成自动判据** `by1tiers.py`：T1 存在、T2 依赖与档位一致（模块级 import，torch/transformers→≥4、numpy→≥3，对同目录 `by1*` 再跟一层）、T3 范围、T4 命令名、T5 modelcheck 全覆盖（豁免要写理由）；它第一次跑就抓到一处真漂：`by1coerce.py` 顶层 import numpy 却挂在档 2（已改 3））（第二十三轮：**耗时也是判据** `by1budget.py`：档 1/3/5 实测18.0/48.1/115.4 秒 x 3 为上限，B1 超限必红并指出最慢一段、B2 单段 >70%% 也红；`BY1_BUDGET_SCALE` 可放宽；时间表头直接印出上限）（第二十七轮：**T6 参照物来源** —— 每个参照对比类判卷人的 docstring 必须有一行 `参照物：...`，且它里的 CamelCase/.py 名字必须**真的出现在 docstring 之外的代码里**（防“文档写的参照物早就不存在了”）；实测这一条立刻查出 9 个判卷人没写参照物，已逐个补上）（第二十八轮：**T7 上游** —— 参照物行里点名的官方名字必须能在**本机装的 transformers** 里找到（没装就跳过；扫一次 63,380 个标识符、缓存在进程内、3 秒）；`by1instella` 的参照物在 AMD 官方仓库而不在 transformers，记在 UPSTREAM_EXEMPT）（第二十九轮：**闭集与出处** —— `by1attrs.py` 把 `by1codegen.ENUMS` 里每个取值钉到官方实现：A1 每个取值都要有出处、A2 没有陈旧条目、A3 出处里的家族名要能对上 `models/` 或 `refs/`；实测 20 个取值全部有出处，并列出**还没有模型用到**的 5 个（act=gelu、gate_act=softplus、qk_norm=1/on/yes）——它们不算失败，但看得见）（第三十轮：把那份“没人用的取值”清单变成动作 —— 新增 `models/laguna-tiny.by1`，Laguna 机械缩小、结构一个字没改；它同时写了 `qk_norm = true` 与 `gate_act = softplus` —— 两个之前没有任何模型走过的写法；已进图探针：ggml 2.259e-07 · 图->C 3.840e-07 · exec vs torch 4.216e-07；剩下没人用的只有 act=gelu 与 qk_norm=1/on/yes —— 均为别名写法）（第三十一轮：“没人用的取值”清单升级成 **A4 判据** —— 每个取值要么有模型走到（显式写出来或靠编译器默认值生效），要么在 RESERVED 里写保留理由；默认值从 by1codegen 的 attrs.get 静态取出来，实测隐式生效的有 act=silu 与 routing=softmax_topk；保留的只剩 act=gelu 与 qk_norm=1/on/yes）（第三十二轮：同一套“闭集 vs 语料”用到**图原语**上（A5）—— `by1graph.PRIMS` 里每个原语要么被 21 份图探针模型走到，要么在 PRIM_RESERVED 写理由；实测覆盖 **39/41**，没走到的只有 gelu 与 sqrt（已写理由））（第三十三轮：**A6 两个发射器都要有映射** —— 静态扫 by1ggml 与 by1c 的分派表（prim == / prim in (...) / prim in CONST，并用 AST 展开 _UN_SHAPE_OK = set(_UN_OP) | {...}），要求 PRIMS 里每个原语两边都有映射；实测 41/41。齿齿：从 _UN_OP 删掉 gelu -> A6 报 <gelu> 在 C 发射器没映射）（第三十四轮：新增 `models/gelu-shaped.by1`（最小探针，FFN 写 act = gelu），把图原语 gelu 真正走到；三个后端对拍 1.038e-07 / 1.327e-07 / 1.454e-07；图原语覆盖 39/41 -> **40/41**，保留表只剩 sqrt（没有任何 lowering 会产出它）；顺带修了检查器自己的一个 bug：正则读探针表只读到 19/22 个模型（注释里的 ] 把正则截断了），已改用 AST）（第三十五轮：**删掉死原语 `sqrt`** —— 没有任何 lowering 会产出它（rms_norm/layer_norm 里的 sqrt 是 numpy 参考，sqrtsoftplus 是独立原语）；按“要么实现、要么拒绝”把它从 PRIMS、两个发射器的分派、保留表里一并删掉；图原语 41 -> **40**，覆盖 **40/40**；顺带把 A5/A6 的异常处理收紧：**只有缺依赖/缺语料才算跳过**，其它异常（比如 by1graph 被改出缩进错）必须报错 —— 第三十五轮真遇到过这种“工具坏了却给了绿”）（第三十六轮：新判卷人 `by1gemma.py` —— **GeGLU**（门控 + tanh 近似）对官方 `GemmaMLP`：正例 2.980e-08；两条可证伪：`act=silu` 差 6.587e-02、精确 `gelu` 与 tanh 近似差 1.328e-04）（第三十七轮：`by1gemma.py` 再加两段 —— **relu2**（Nemotron 的 squared-relu）对官方 `NemotronMLP`：正例 **逐位一致 0.000e+00**，写成 silu 差 4.486e-01）（第三十八轮：`by1gemma.py` 再加两段把 `_swiglu` 的**第三个分支**钉上 —— **gptoss clamped swiglu** 对官方 `GptOssExperts._apply_gate`：正例 **逐位 0.000e+00**，去掉 clamp 差 2.034e+01；置信：输入 *4 保证越过 limit=7）（第三十九轮：新判卷人 `by1qknorm.py` —— **qk_norm 的三个取值是不是三件事**：① `per_head` 对官方 `Qwen3Attention` **逐位 0.000e+00**；② 旧写法 `true` 与 `per_head` **逐位等价**（by1langspec 的记档被验了）；③ `off` 差 2.571e-01；④ `full` 差 1.367e+00（同一组 qn 权重按头数重复，两边唯一差别就是归一轴））（第四十轮：`by1moe.py` 加一段 `routing_kinds` —— **四个 routing 取值的关系**：① `topk_softmax` 对官方 `GptOssTopKRouter` 逐位一致；② `softmax_topk` 与 `topk_softmax` **是同一个函数**（权重差 8.941e-08 = 1e-9 eps）；③ `sigmoid_topk` 与 `softmax_topk` 选中的专家**相同**（均单调）但输出差 1.207e-02；④ `sigmoid_group_topk` 比 `sigmoid_topk` 多 6/12 行不同）（第四十一轮：`by1laguna.py` 加 ⑦⑧ 两段 —— **滑窗**（前六段全部绕开了它，而 Laguna 真模型 3/4 层是滑窗）：W=5 对官方 `LagunaAttention`（`layer_types=["sliding_attention"]`）**逐位 0.000e+00**，把窗口关掉差 6.930e-01）（第四十二轮：**A7 产物里的证据** —— 跑 `by1c --from-graph` 真生成一份 `model.c`，对图里每个原语查**它的分支符号是否出现在产物里**（符号从 by1c 分支源码自动抽，22/32 个原语有命名内核）；实测 2 份产物、缺 0；齿齿：把 rope 的符号换成假名 -> 两份都报 A7）（第四十三轮：A7 的见证集按**贪心覆盖**从 2 份扩到 **5 份**（llama-shaped / gelu-shaped / ling-tiny / mla-sparse-shaped / ssm-shaped，合计 ±5 秒），产物证据从 8/22 原语扩到 **22/22**，缺 0）（第四十四轮：A7 扩到 **ggml 侧** —— 两侧都有产物证据：C 5 份见证/22 个有命名内核的原语 22/22；ggml 8 份见证/35 个原语 35/35；ggml 见证集同样贪心选出（gpt-oss-shaped + clamp_max/soft_max_sink · gpt2-tiny + gelu_tanh · qwen3-next-shaped + log1p）；齿齿：把 matmul 符号换成假名 -> 8 份都报 A7）——并记下一个环境区别：**by1ggml 找不到静态库时返回 30 + `[跳过]`**，A7 必须把它当“没验”而不是失败（CI 的档 3 腿就没有那个构建 —— 第一版正是这里红的）|| 参考执行 | 扩大独立参考的模型/上下文范围；合成图像检查 ≠ 视觉评测（第十一轮推进一格：`by1ssm` 判卷人从 **2 样形式** 扩到 **形状 × 上下文** 网格（9 格），含分块残块（seq 7/13/20）、n_groups 1/2/4/8、conv 2/4/7，逐格登记余量（26.7x~120.8x）；并记下配置约束 **n_groups 必须整除 num_heads**）（第十二轮：直连对拍接上了 —— 官方权重直接喂给 `by1exec.op_ssm`，实测 1.5e-07~2.1e-07；把 op_ssm 的 gate 顺序写错，式子那些行照样绿而直连行立刻红——这就是“两条链各自对上官方不等于彼此一致”的直接证据）（第十五轮：`by1rope` 从**只验 llama3 一支** 扩成 **default / yarn / llama3 三支 13 格网格**，每格同时比**生成物里的 rope_tables** 与 **`by1graph.rope_inv`**（唯一源头）两份实现，对手是 transformers 的官方频率函数；最差相对 5.96e-08（float32 精度）。可证伪三条（三种"用错类型"）全对不上；只改图那一份的 truncate -> 四格 tr1 红，且诊断行直接报`runtime 5.96e-08 / graph 4.07e-05`）（第十六轮：`by1mla` 把“两种配对任一对上就算过”改成**钉死 interleaved + half 必须对不上**，并加了官方权重 -> `by1exec.op_mla`的**直连对拍**（5.75e-07）；把 op_mla 的 scaling 改错 -> 生成物那行照样绿而直连行红（3.48e-02））（第十七轮：`by1moe` 加了**整块**与**专家路径**两条直连（均 <1e-6）；坑：官方模块独立构造时专家权重是 `torch.empty`、实测全零，于是“官方 experts 返回 0”且换 w1/w3 不变 —— 0 == 0 也“对得上”；判卷人现在自己初始化权重 + 逐 token 参考，换开 w1/w3 必红）（第十八轮：`by1kda` 从**永远跳过**变成**真的跑**——参照物改用 transformers 官方的 `recurrent_kimi_delta_attention` +`KimiLinearRMSNormGated`（纯 torch，不需 fla/triton）；三格三段1.2e-07~2.2e-07，8 条反例全红；GVA 那格官方参考不支持（nk≠nv），递推/尾巴如实标 `—` 不算过）（第十九轮：`by1sparse` 补上**直连后端**一角：官方 `GlmMoeDsaIndexer` 权重-> `by1exec._dsa_index`，rope=0 两组挑选集合逐个相同；发现**挑选对全局正缩放不变**（所以齿齿要改相对权重才能红）；验证行现在把直连也算进结论）（第二十轮：**新判卷人 `by1gptoss`** —— 注意力 sink 对着`GptOssAttention` 逐位 0.0；另钉两条：关掉 sink 必须变（8.6e-01）、**sink 清零 不等于没有 sink**（分母多一个 e^0，与官方也是 0.0）；判卷人已进档 5。另：官方要**显式因果掩码**（传 None 它不做因果，差 0.65））（第二十四轮：`by1gptoss` 从只验 sink 扩成 **sink / 滑窗 / YaRN 三段**，每段各自逐位 0.0 + 一条可证伪（关 sink 1.49 / 去 window 0.726 / 退回普通 rope 5.75e-02）；齿齿：滑窗的 `>` 改成 `>=`（边界差一）-> 只有②变红 0.584，①③ 仍 0.0）（第二十五轮：**新判卷人 `by1laguna`** —— **head gate** 的激活。它第一次跑就抓到一个真 bug：`Attention.__init__` 从来没设 `self.gate_act`（只有 `MLAttention` 设了），于是任何 `mech Attention` + `head_gate` 的模型会 AttributeError——已修；判据：per-head+softplus 逐位 0.0、sigmoid 必须不同（3.96e-01）、关掉必须变（5.27e-01）、`gating=False`（full 宽度门）也逐位 0.0；**已销（第二十六轮）**：那不是两边频率对不上，而是 **Laguna 的`rope_parameters` 按层型分开**（full 层 base=500000 + partial 0.5，滑动层 10000 + 1.0），我传的 `rope_theta` 被它覆盖了；现在 ⑤/⑥ 两段各自**逐位 0.0**（base 用错 9.94e-02、partial 当 1.0 差 2.86e-01）（第四十五轮：**A8 节点数 = 调用次数** —— 对实测 1:1 的 20 个原语（CALL_COUNT）数产物里的调用次数与图里的节点数，**5 份产物共 53 处全对得上**；齿齿：matmul 的符号换成 p_rope -> 五份都报 A8）（第四十六轮：A8 扩到 **ggml 侧** —— 同样先量后定表：8 份产物里共 14 组 1:1，**只取语义上真对应的 11 个**（softplus->ggml_sqrt、ggml_new_tensor_1d、ggml_map_custom1 这类巧合都剔掉）；实测 **C 53 处 + ggml 34 处 = 87 处**全对得上；齿齿：top_k 符号换成 ggml_soft_max -> 三份报 A8）（第四十七轮：**`qk_norm = full` 也钉到官方了** —— `by1qknorm.py` 加 ⑤⑥ 两段：full 对 `Olmo2Attention`（整宽 RMSNorm、作用在拆头之前）**逐位 0.000e+00**；同一批权重只把归一轴换成 per_head 差 5.469e-02；置信：本机 transformers 5.20.0 里只有 `olmo2`/`olmo3`/`minimax_m2`/`flex_olmo`/`olmo_hybrid` 把 q_norm 建成整宽 RMSNorm）（第四十八轮：**`act = silu` 补上正例** —— `by1gemma.py` 加 ⑧：by1 的门控 MLP（act=silu）对官方 `LlamaMLP` **逐位 0.000e+00**。此前 silu 只当过反例（“写成 silu 就不是 GeGLU/relu2”），从来没有一条判据说“它对”。五个取值（silu / gelu_new / gelu / relu2 / gptoss）现在都有正例或可证伪）（第四十九至五十一轮：**q_gate 待查 -> 已销**。根因不在 by1：Qwen3-Next 的 RMSNorm 用 one_plus 约定（weight 零初始化、乘 1+w），而 Qwen3 的是普通 w·x，同一份权重差一个 2 倍。by1 的 norm_one_plus 打开后 **q_gate 对官方 Qwen3NextAttention 逐位 0.000e+00**（新增 ⑦）；⑧ 可证伪：同一批权重把 norm_one_plus 关掉差 1.802e-01。顺带排除了 partial rotary：0.25 改 1.0 差异不变） |
| ggml 生成 | 只是验证工具，不代表可并入上游的完整架构 |
| llama.cpp 基线 | **干净 clone 构建（2026-10-17 第九轮已做）** · **llama-bench / llama-cli 数值实跑（未做）** —— tag `v0.6.0`（= `d812350`）clone 11 秒 / configure 2 秒 / build 207 秒，两个二进制版本串都是 `commit d812350`；`by1llama.py pin-check` 已进档 5 当判据（没构建时 `[跳过]`）。**剩下的一格是数据**：`llama-bench` 要完整模型 GGUF，仓库只有 tokenizer-only 的 vocab GGUF（实测报 `tensor token_embd.weight not found`），11.8 GB IQ3_S 的下载仍归 GPU 日（第二十一轮：`logits_compare` 原来**只有 CLI 一个调用者、没有任何门禁跑它**；现在它带七条自检（相同/一位不同/长度/空/NaN v NaN/NaN v 数/逐行文本）跑在档 5；顺手修了两处：`problems` 不进结论、NaN 只报“首个分歧”） |
| 量化 Q1/Q2 | 新格式必须真实 checkpoint + 独立判据；Q2 的 GPU 真跑未做（第十三轮：Q2 的 **CPU 一半** 立了判据 —— `audit_checkpoint_storage`原来只有 `by1gpubench` 一个调用者，没卡就从没执行过；现在`by1quant.selftest` 用合成 safetensors 分片跑 2 正 + 6 反，档 1 里跑；**GPU 那一半仍未验**）（第十四轮："无静默解量化"改成**直接点名**并配按真实 IR 形态的反例（FP8 -> bf16、MXFP4 丢 `_blocks`、scale 是 F16）；分片头审计另有2 正 6 反（第十三轮）。**GPU 那一半仍未验**） |
| 量化 Q3 | 未开 |

## 总账（一句话）

**零门禁失败、矩阵红项清零（B1 已修进门槛）、8 条证据未销、7 条范围记档、2 个环境跳过** ——
最重的剩下 E 系（多模态端到端）。

> **2026-10-16 更新**（由 [RC1 架构决策](rc1-decision.md) 触发）：
> - **P0 闭环主线已完成**（[p0-loop-report](p0-loop-report.md)：Qwen3-0.6B 311 张量、
>   上游 logits **逐位 106949 值**、增量 == 全量 2.289e-05）—— 旧版总账那句
>   "尚未开工"已过期；
> - 新增**三根结构轴**与**六个产物级模型**（见 [支持矩阵](support-matrix.md) A 段）；
> - S2、G2 已销（旧版正文里已标）；
> - 未变的：E 系多模态端到端、G1 GPU 全线、Q3 —— 它们在决策里被明确**延期**。

> **2026-10-17 晚（第二轮）**：ggml **图路**把 `ssm_scan` 也收了（逆 softplus 折算 +
> D 跳连外围补 + 布局对调），默认路公共子集 **18/23 -> 19/23**；
> 图路 19/20 个 shaped 探针数值过（只剩 `mla-sparse` 的 k-pool —— 第四轮销）。
> IR 路仍是 10/23 —— **A3-3（删 IR 路）现在只差把 `GGML_IR_MODELS` 那 6 份改走图路**，
> 这一笔记在 ④b。

> **2026-10-17 晚（第三轮）**：**A3-3 已销** —— 旧 IR 路删掉了（`by1ggml`
> 2277 -> 1490 行），矩阵回到单列 `ggml`（= 图路），四后端交集 **19/23**，
> 剩下一格"可补"= SparseMLA 的 k-pool（A3-4）。
> 同一轮补了 `ssm_scan`（第二轮）—— 两轮合起来，图路从 18/23 走到 19/23 且
> 不再有第二条路要维护。

> **2026-10-17 晚（第四轮）**：**A3-4 已销** —— SparseMLA 的 k-pool 收进图路，
> 四后端交集 **19/23 -> 20/23**，可补格 **1 -> 0**（三档只剩"设计如此" 26 格）。
> 图路 20/20 个 shaped 探针数值过；k-pool 那一份 **1.914e-07 / 3.210e-07**。
> 代价两条，都写进 ④b：图路要求 `seq % kpool == 0`；图->C 那条对照明确拒绝（A3-5）。

> **2026-10-17 晚（第五轮）**：**A3-5 已销** —— 图->C 也吃 k-pool 了（float 下标内核 +
> 推断维形状），`GRAPH_C_MODELS` **19 -> 20/20**，两个发射器的差分对照没有缺口。
> mla-sparse 图->C **4.022e-07 / 6.746e-07**。第四轮那个 NaN 的根因是 `None -> n`
> （池 gather 整块形状算错）—— 记在这里，免得下一次再把"能跑"当成"算对"。

> **2026-10-17 晚（第九轮）**：llama.cpp 基线的"干净 clone 构建"做了 ——
> tag `v0.6.0`（= `d812350`，GitHub API 核对过）clone 11 秒 / configure 2 秒 /
> build 207 秒（Windows + MinGW + CPU；要加 `-D_WIN32_WINNT=0x0A00` 才过
> `vendor/cpp-httplib`），`llama-cli` 与 `llama-bench` 的版本串都是 `commit d812350`。
> 旧的 `llamacpp/build-cli` 快照是 `9b0c827`（**上游查不到**）—— `pin-check` 对它
> 会红，这就是钉定值是真的证词。`llama-bench` 跑得起来但出不了数（要完整模型
> GGUF），记档为未验。