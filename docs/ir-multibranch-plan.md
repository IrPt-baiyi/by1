# by1-ir 1.2「多分支」计划（2026-10-19 · A 弧线第 1 轮）

> 解冻依据（rc1-decision F1–F6：动 IR 要先有新证据）：10 份 `complete-*` 降不进 IR 路，
> 四路数值判据（exec / torch codegen / C / ggml）对它们一律绕开；浏览器里挂着 partial 标记。

## 〇、普查（70 份 `.by1`，`docs/ir-multibranch-survey.json`）

| 构造 | 份数 |
|---|---|
| 单栈、无分支、无状态 | **46** |
| 多栈（main + visual / mtp / audio 等 aux） | **21** |
| 有状态（states > 0，gpt-oss 两家 + selftest） | **3** |

**第一个反直觉**：21 份多栈里 **20 份 `compile_ir` 照样「能降」** —— 因为它把各栈
**拍平成一条串行链**（Qwen3.8-27B 的 92 层 = main 64 + mtp 1 + visual 27；
Qwen3-VL-2B 的 52 层 = 28 + 24）。这条链**作为前向是错的**（视觉塔不是第 29 层，
它是喂进 merger 的旁路；MTP 是并列的头）。拍平的链只能自我对拍，不能对拍真模型。

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
    complete-gpt-oss-20b.by1   24 层（+2 states，先按 G-D 记录）

（`complete-Qwen3.8-27B` 虽然「能降」，但那是拍平产物，**不算**快赢。）

**放行条件（判据）**：这 4 份走 IR → exec 后端的前向，与 **transformers 官方前向**对拍
（1e-3 族）；过了才把 `backend = transformers` 的一刀切门改成「真多分支才拒」。
判据形式：`by1diff` 同款（同权重同输入），进档 5。

## 三、by1-ir 1.2 形状草案（待快赢定型后细化）

- IR 增加 **branches**：每个分支自己的 layers + 挂载；`layers` 保留（serial = 单分支特例，
  **1.1 文件不改一个字**）；
- **wiring 显式**：谁吃谁的输出（visual → merger → main 的拼接点；MTP 并列头的输入；
  audio/visual 的模态入口）；
- `ir-spec.md` 由代码重新生成（by1docs 的生成物判据把关）；`by1bootir` 两条路同 IR；
- 后端顺序：exec 先行（Python 好写），torch codegen / C / ggml 逐个跟；
- **不拍平是硬规则**：1.2 的图必须能回答「这个节点在哪个分支上、接线从哪来到哪去」。

## 四、顺序

1. **G-A 放行**（下一轮，判据见上）；
2. **G-C 分支表达**（1.2 核心，判据 = 4 路对拍 + 不拍平审计）；
3. **G-B 机制补齐**（逐个，每个自带四路判据；`qkv_bias` 最便宜先做）；
4. **G-D 状态**（最后，等分支语义定型）；
5. 浏览器 partial 标记换成真分支视图（能做就做，做不到记延期）。
