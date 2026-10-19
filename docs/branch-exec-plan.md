# B 弧线：多分支从「能表达」到「能执行」（2026-10-20 起）

> 依据：A 弧线的 by1-ir 1.2 三件套（stacks / wiring / shared）+ 不拍平审计（T24）已进门禁，
> 但后端仍按 `layers` 平表跑 —— **wiring 是结构事实、还不是执行语义**。
> 真多分支 `complete-Qwen3.8-27B` 的 IR 仍拒，报文如实写着「等后端按接线跑」。

## 〇、侦察（2026-10-20 已核实的三个事实）

1. **官方参照物存在**：`transformers.models.qwen3_5` ——
   `Qwen3_5Config / Qwen3_5ForConditionalGeneration`（含 `Qwen3_5TextConfig` /
   `Qwen3_5VisionConfig`）。tiny 实例实测可建（2.86M 参数 / 64 个键）；
   前向签名 = `input_ids + pixel_values + image_grid_thw + mm_token_type_ids`。
2. **官方视觉管线 = patch_embed → pos_embed → blocks → merger**。权重：
   `model.visual.patch_embed.proj.{weight,bias}` · `pos_embed.weight` ·
   `blocks.N.*`（qkv 融合带 bias，和语言里 V 机制的 `qkv_fused/bias` 对上）·
   `merger.norm.{weight,bias}` + `merger.linear_fc1/2.{weight,bias}`。
   `Qwen3_5VisionConfig` 的字段（depth/in_channels/patch_size/spatial_merge_size/
   temporal_patch_size）**就是语言里 `vision{}` 块的字段**。
3. **IR 缺口精确到三组**：blocks 有算子（layers 装得下），**patch_embed / pos_embed /
   merger 没有** —— `complete-Qwen3.8-27B` 的契约 1199 张量 vs `params_of` 1036，
   差的就是这三处。**这就是「能表达」到「能执行」的第一堵墙**：
   wiring 说清了「谁吃谁的输出」，但 merger 的**计算**没有家。

## 一、设计：视觉入口/出口进 IR（形状定稿见下一轮）

两个候选：

| 候选 | 做法 | 判 |
|---|---|---|
| **A：新增算子 kind** | `PatchEmbed` / `PosEmbed` / `Merger` 进 KIND_ATTRS + params_of + 各后端 + 反例库 | ✅ 采 |
| B：接线自带计算 | concat 边的 via 隐含 merger 的计算 | ❌ 弃 |

**选 A 的理由**：不拍平判据要求每个**节点**答得出「在哪个分支上」——
把计算藏进接线，节点表就答不出了（接线只该说「谁吃谁的输出」，不该暗藏算子）。
**MTP 头不需要新算子**：mtp 栈的层 + `shared` 绑定的 head 已经是全部语义
（vLLM qwen3_5_mtp 的口径：吃主干最终归一化后的 hidden）。

## 二、判据（① 的验收形式）

- **夹具** `models/multibranch-tiny.by1`：complete 式（backend transformers ·
  arch qwen3_5 · tiny）—— trunk + 视觉塔 + MTP 并列头，档 5 跑得动。
  结构 = complete-Qwen3.8-27B 的机械缩小（clef-tiny 同手法）。
- **同权重同输入**对拍（1e-3 族），两条各记各的数：
  1. **主 logits** vs `Qwen3_5ForConditionalGeneration` tiny 官方前向
     （visual → merger → main 全链）；
  2. **MTP logits** vs vLLM qwen3_5_mtp 语义（by1full 装配的那半）。
- 进档 5（前向段新行）。

## 三、顺序（每节做到哪记到哪）

1. **夹具 + 参照物可行**（本轮 ✅）：multibranch-tiny.by1 进 models/，
   by1check 过、by1full tiny 装配建得出官方类；
2. **IR 算子**：PatchEmbed / PosEmbed / Merger + params_of + validate +
   反例库（每条新规则配必红）+ ir-spec 重生成；
3. **exec 按接线前向** + 判卷人进档 5（① 收账）；
4. **真多分支放行**：complete-Qwen3.8-27B 降得出（②）—— 全尺寸前向装不下，
   **明记跳过、不假装验过**；
5. **by1sparse 查账**（③ 后一半）：DSA 索引器差异查清，修掉或如实记账；
6. **第二路后端**（④）：torch codegen 或 C 跟上接线语义，尽量。

## 四、口径备忘

- **放行 gate 的时机**：和 exec-by-wiring 同轮 —— 门早于执行语义打开的话，
  平表链会静默地当真前向跑（「拍平的链只能自我对拍」）；
- **报文如实**：装不下、没验、猜的 —— 三样都得说出来（A 弧线的老规矩）。
