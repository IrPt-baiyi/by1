# by1-ir 1.2「多分支」计划（2026-10-19 · A 弧线第 1 轮）

> 解冻依据（rc1-decision F1–F6：动 IR 要先有新证据）：10 份 `complete-*` 降不进 IR 路，
> 四路数值判据（exec / torch codegen / C / ggml）对它们一律绕开；浏览器里挂着 partial 标记。
>
> **进度（2026-10-20）**：① 普查 ✅（〇 + 〇之二逐份表）· ③ 最小里程碑 ✅（二之二：
> 装配模块 + exec 后端**两条判据**都进档 5）· ② 形状设计 ✅（三节：字段/语义/兼容 +
> ir-spec 重生成 + by1bootir + **生产者与不拍平审计**）
> · ④ 浏览器 partial ⏳（四·5，做不完就记延期）。

## 〇、普查（70 份 `.by1`，`docs/ir-multibranch-survey.json`）

| 构造 | 份数 |
|---|---|
| 单栈、无分支、无状态 | **46** |
| 多栈（main + visual / mtp / audio 等 aux） | **21** |
| 有状态（states > 0，gpt-oss 两家 + selftest） | **3** |
| 声明了共享权重（`tie_word_embeddings`） | **11**（true 6 / false 5；其余 59 份没声明） |

普查每份记：`model / backend / stacks / aux / mtp / vision / states / tie / lowered / why`。
**lowered / why 两列跟着门走** —— G-A 放行后重生成过一次（探针口径：`by1check.check`
→ `by1codegen.compile_ir`）：**70 份里 59 降得动、11 降不动**（放行前是 58/12）。

**第一个反直觉**：21 份多栈里 **20 份 `compile_ir` 照样「能降」**（摘掉 backend 门的探针口径；
唯一例外 `Qwen3.8-27B-FP8`：hparams 缺 vocab）—— 因为它把各栈
**拍平成一条串行链**（Qwen3.8-27B 的 92 层 = main 64 + mtp 1 + visual 27；
Qwen3-VL-2B 的 52 层 = 28 + 24）。这条链**作为前向是错的**（视觉塔不是第 29 层，
它是喂进 merger 的旁路；MTP 是并列的头）。拍平的链只能自我对拍，不能对拍真模型。

### 〇之二、10 份 complete-* 逐份表（2026-10-19 · 普查收账）

| 模型 | 栈（层） | aux | mtp 共享 | vision | tie | states | 新门下 | 缺口 / 判据 |
|---|---|---|---|---|---|---|---|---|
| `complete-gpt2` | main 12 | – | – | – | true | 0 | **放行** | G-A 已销 |
| `complete-Qwen3-0.6B` | main 28 | – | – | – | true | 0 | **放行** | **③ 里程碑**：IR → 装配模块 vs transformers 官方前向 max\|Δ\| 2.682e-06，进档 5 |
| `complete-Qwen3-8B` | main 36 | – | – | – | false | 0 | **放行** | G-A 已销 |
| `complete-gpt-oss-20b` | main 24 | – | – | – | false | 0 | **放行** | **没有 states 声明**（G-D 的 2 个状态在非 complete 的 `gpt-oss-20b.by1`） |
| `complete-Qwen3.8-27B` | main 64 + mtp 1 + visual 27 | mtp, visual | share_embeddings, share_head | 10 键 | false | 0 | 拒（报文如实） | **真多分支**（G-C）：visual → merger → main、MTP 并列头 —— 等 1.2 |
| `complete-Falcon-H1-0.5B-Instruct` | main 36 | – | – | – | false | 0 | 拒 | G-B：层内并行 `Parallel` 未收编 |
| `complete-granite-4.0-h-1b` | main 40 | – | – | – | true | 0 | 拒 | G-B：混合栈 `SSM2` 未收编 |
| `complete-granite-4.0-h-350m` | main 32 | – | – | – | true | 0 | 拒 | G-B：混合栈 `SSM2` 未收编 |
| `complete-mamba-130m-hf` | main 24 | – | – | – | true | 0 | 拒 | G-B：SSM 缺 attrs（heads / head_dim / n_groups） |
| `complete-Qwen2.5-7B-Instruct` | main 28 | – | – | – | false | 0 | 拒 | G-B：Attention 的 `qkv_bias` 未收编 |

**逐份表的三个结论**：

1. **4 放行 / 5 拒于 G-B / 1 拒于真多分支** —— 一刀切门当年把 4 份「本来就单栈」的也拒了，
   报文还说「多分支降不成串行 IR」（G-A，已销）；
2. **10 份全部 states = 0** —— states 只在非 complete 的 `gpt-oss-20b/120b` 与 `selftest`（G-D），
   complete-* 的状态语义交给 transformers 后端自己管；
3. **共享权重两种形态**：`tie_word_embeddings`（10 份里 5 true / 5 false）与 MTP 的
   `share_embeddings / share_head`（只有 Qwen3.8-27B 有）—— 1.2 的分支图要把这两种
   都表达成**跨分支的参数绑定**。

## 一、「多分支」其实混着四个不同的缺口

| # | 缺口 | 证据 |
|---|---|---|
| **G-A** | **一刀切的门 + 报文失实**：`compile_ir` 见 `backend = transformers` 就拒，报文却说「不能把多分支模型降成串行 IR」—— 10 份 complete-* 里 **9 份是单栈**，根本不是多分支 | 摘掉 backend 后探针：4 份直接可降 |
| **G-B** | **机制/属性缺口**（正常拒绝，只是没实现）：`Parallel`（Falcon-H1 层内并行）· `SSM2`（granite 混合栈）· Attention 的 `qkv_bias`（Qwen2.5-7B）· SSM 缺 attrs（mamba-130m） | 探针逐份报文 |
| **G-C** | **多栈拍平**：分支接线（visual→merger→main、MTP 并列、audio）在 IR 里没有表达，全部压成一条 layers 列表 | 21 份多栈的层序数据（92 = 64+1+27 等） |
| **G-D** | **状态声明**（states）：gpt-oss 两家各 2 个、selftest 1 个 —— IR 的 forward 语义里没接 | 普查 st 列 |

## 二、快赢（下一轮）：G-A 放行 4 份

摘掉 backend 门后**直接可降**的：

    complete-gpt2.by1          12 层
    complete-Qwen3-0.6B.by1    28 层
    complete-Qwen3-8B.by1      36 层
    complete-gpt-oss-20b.by1   24 层（**无 states 声明**；G-D 的 2 个状态在非 complete 的 gpt-oss-20b.by1）

（`complete-Qwen3.8-27B` 虽然「能降」，但那是拍平产物，**不算**快赢。）

**放行条件（判据）**：这 4 份走 IR → exec 后端的前向，与 **transformers 官方前向**对拍
（1e-3 族）；过了才把 `backend = transformers` 的一刀切门改成「真多分支才拒」。
判据形式：`by1diff` 同款（同权重同输入），进档 5。

### 二之二、G-A 已销（2026-10-19 同日第二步）

> **档 5 复验（2026-10-19 收账时）**：`complete-Qwen3-0.6B` 前向最大绝对差 **2.921e-06**
> （1e-3 族内，种子不同所以和上面的 2.682e-06 略有出入），同段其余 4 份 shaped 全绿。
> 档 5 另有 **2 项既有红**（`by1sparse` 索引器对不上参考 · `by1gemma` 缺 import 引导
> 直接崩）—— **HEAD（37a0070）干净 worktree 对照复现一致**，与本弧线无关，另账。

**门改了**：`compile_ir` 不再见 `backend = transformers` 就拒 —— 按**真实构造**分：
真多分支（aux/mtp/vision/状态）才拒（报文如实），单栈放行。实测：

    complete-gpt2 / Qwen3-0.6B / Qwen3-8B / gpt-oss-20b   放行（12 / 28 / 36 / 24 层）
    complete-Qwen3.8-27B                                 拒：真多分支（等 1.2）
    其余 5 份                                            按具体缺口拒（G-B 清单不变）

**③ 最小里程碑已过**：`complete-Qwen3-0.6B` 走通 IR → 装配模块 → 与 transformers 官方
前向对拍 —— max|Δ| **2.682e-06**、相对 **8.577e-07**（< 1e-3 族），已进门禁档 5 的前向段。

**③ 的字面判据也过了（同日第三步）**：目标原文写的是「IR → **exec 后端**」——补了
`by1diff --exec`：IR 的权重按同一张 HF→IR 映射表（按算子下标定位）喂进 **by1exec 的
NumPy 运行时**，与 transformers 官方前向**同权重同输入**对拍 —— max|Δ| **4.301e-06**、
相对 1.375e-06（< 1e-3 族），进档 5 的「exec 前向」行（311 个张量全喂到）。
**它和上面那条是两条判据**：那条两边都是官方类（验契约装配），这条验 **by1 自己的
运行时**算得对不对 —— 两条都留，数字各记各的。

**顺带抓到一个真 bug（by1diff 的 tie）**：by1diff 的 ref 侧从 `ir.get("tie_word_embeddings")`
取 tie —— **ir 1.1 没这个键**（声明在 `hparams`）。于是 ref 不 tie、装配模块 tie，
搬权重时 `embed_tokens.weight` 与 `lm_head.weight` 两个不同的随机张量写进**同一个绑定参数**
（后写的赢），前向差 **4.78e0**（幅度 3.35）。读 `hparams` 后差降到 2.7e-06。

**另一个学名**：complete-* 的"生成模块"不是独立运行时 —— 渲染出来的是 33 行**装配器**
（`by1full.build_model(FAMILY, CONFIG, MTP)`，官方类 + 契约装配）。所以这条对拍验的是
**契约装配的正确性**（config 映射 / 权重映射 / tie / MTP 接线），不是独立实现的一致性 ——
两种结论都真，但要分清（T4 真权重那次同理）。

## 三、by1-ir 1.2 形状（2026-10-20 定稿，schema 已落地）

**「不拍平」判据是可回答，不是「不能有平表」** —— 1.2 的 IR 必须答得出两个问题：
① 这个节点在哪个分支上（`stacks` 的层段划分）；② 接线从哪来到哪去（`wiring`）。

**形状（加法变更 —— 1.1 文件一个字不改照样合法）：**

| 字段 | 版本 | 语义 |
|---|---|---|
| `stacks[]` | 1.1 定字段 · 1.2 升格 | 分支登记表：name / modality / width / **层段**（first_layer + n_layers，互不重叠）/ **role**（trunk / mtp / tower，1.2 新增） |
| `wiring[]` | **1.2 新增** | 显式接线（stacks 多于一条时必填）：kind（feed / concat / head）+ 端口文法 `input:<模态>` / `<分支>.in\|out\|hidden` / `logits[.<分支>]`；孤儿分支不合法 |
| `shared[]` | **1.2 新增** | 跨分支参数绑定：`tie`（一块权重两个名字）/ `share`（声明级共享）—— 普查〇之二结论 3 的两种形态 |

**三个决定，各带理由：**

1. **节点表照旧 `layers`（全局编号）** —— 不给分支各持一份层表：同一个节点两个家
   就是下一种漂移（实测 91/92 层的多栈模型从不撞名，params_of / 后端零改动）。
   分支归属由**层段**回答，不靠重排节点表。
2. **`role` 与 `modality` 不是一回事** —— mtp 是角色、vision 是模态
   （Qwen3.8-27B 的 mtp 分支 modality=text、role=mtp）。
3. **兼容 = 加法 + 多余字段拒绝** —— 1.1 文件零改动；1.1 的工具读 1.2 的 IR 会被
   未知字段检查**如实拒绝**（不静默忽略）—— 所以 IR_VERSION 动到 1.2、
   TOOL_VERSION 跟到 rc2（by1ver 的家规：IR 动、工具必须跟）。

**已落地（本轮）**：schema 三张表（STACKS 升格 + WIRING + SHARED）· validate 规则
（层段重叠 / 端口文法 / 孤儿分支 / 绑定合法性）· **9 条必红反例**（by1mutate T23，
30 → 39 全红）· `ir-spec.md` 由代码重新生成（by1docs 逐字符判据）· `by1bootir`
两条路对拍加进三件套（273 处一致、0 差异）。

**已落地（同日第四步 · 生产者 + 不拍平审计）**：compile_ir 现在给多栈模型填三件套
—— **19/21 份答得出两个问题**（另 2 份降不动、各有各的账：FP8 缺 vocab、
complete-Qwen3.8-27B 真多分支等后端）。**不拍平审计**（by1ir 自检 T24）进门禁，
档 1 起每档都跑：层段盖满 layers + wiring 非空 + validate 全绿，一条不满足逐份红。
Qwen3.8-27B 实例 = stacks 三条（trunk / mtp / tower，层段 0 / 64 / 65）+ 六条接线边；
`shared` 的 tie / share 三条合成验证（validate 0 问题）。**modality 49 处按名约定推**
（语言层没有一份显式声明）—— 审计逐份列出、不冒充声明；`stack_modality` 已暴露，
显式声明从此胜过约定。

**待做**：exec 后端按接线跑通（visual → merger → main / MTP 并列；后端顺序：
exec 先行，torch codegen / C / ggml 逐个跟）· ④ 浏览器 partial 换真结构。

## 四、顺序

1. **G-A 放行** ✅（二之二：门改了，4 份进 IR 路，③ 里程碑过）；
2. **G-C 分支表达**（1.2 核心，判据 = 4 路对拍 + 不拍平审计）；
3. **G-B 机制补齐**（逐个，每个自带四路判据；`qkv_bias` 最便宜先做）；
4. **G-D 状态**（最后，等分支语义定型）；
5. 浏览器 partial 标记换成真分支视图（能做就做，做不到记延期）。
