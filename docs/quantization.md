# 量化格式与支持边界

这份文档区分三件事：**描述 checkpoint 如何存**、**直接加载并执行量化权重**、**生成新的量化产物**。完成第一件事不代表后两件事已经支持。

## `.by1` 声明

量化属于 checkpoint 的导出布局，不属于模型架构。它放在 `emit` 的 `quant` 块里，且必须写明 `format`：

```by1
emit torch.module {
  quant {
    format = fp8_e4m3_block
    block  = 128
    scale  = weight_scale_inv
    tensor = [SparseMLA.q_a_proj.weight, SparseMLA.q_b_proj.weight]
  }
}
```

`tensor` 是被量化的逻辑张量列表，不是对所有二维权重的推断。没有列出的张量保持原样。带机制前缀的名字用于区分不同机制中同名的投影。

`block` 是格式参数，不是硬件线程块大小。格式字段、必需字段和互斥规则集中定义在 `src/modelcheck/by1quant.py`；未知格式、缺字段和冲突声明会产生 E 级错误。

## 当前格式规格

| 格式 | 权重存储 | scale 存储 | bias / 融合 | 当前实证来源 |
|---|---|---|---|---|
| `mxfp4` | OCP MXFP4 元素为 E2M1；每 32 个值共享一个 8-bit E8M0 scale。checkpoint 把 4-bit 值打包成 U8，每 32 个值占 16 字节；`_blocks` 形状为 `[..., out, in/32, 16]`。 | 每块一个 U8 E8M0 scale；`_scales` 形状为 `[..., out, in/32]`。 | bias 保持 BF16，并以 `_bias` 存储。`fuse` 将多个逻辑权重沿输出维合成一个存储名；目前只给 MXFP4 开放。 | `gpt-oss-20b`、`gpt-oss-120b` 的 safetensors manifest；MX 数值格式见 [OCP MX specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) |
| `fp8_e4m3_block` | F8_E4M3 权重保留原名和原形状。 | 必须声明 scale 后缀；scale dtype 由 `scale_dtype` 明确记录（默认 F32；Qwen3.8 合同当前声明 BF16，并在 GPU benchmark 下载后从原始 safetensors 头核验）。二维块形状按各维度除以 block 向上取整。反量化值为解码后的 E4M3 值乘 `weight_scale_inv`。 | 当前规格不声明 bias 量化，也不支持 `fuse`。 | `GLM-5.3-Flash` 与 Qwen3.8 FP8 的量化权重布局；Qwen3.8 使用 128×128 block。 |

这些是项目针对已对拍 checkpoint 固定下来的存储合同。新的导出变体必须有真实 manifest 和对应判据后才能加入，不能只因格式名称相同就假定布局相同。

## 共享权重

量化声明选中张量，不改变模型的权重共享关系。若输入和输出 embedding、MTP 与主干或其他模块共享同一底层权重，运行时和导出器必须保留同一份量化存储及其别名关系；不能因为两个逻辑名字都匹配就重复量化、重复分配或静默拆成两份。当前 Q1 只校验单份 manifest 中声明的物理名字、形状与可用 dtype，共享存储的运行时身份校验属于 Q2，量化产物 round-trip 属于 Q3。

## 分阶段验收

1. **Q1：格式规格与校验。** `.by1` 明确格式；MXFP4/FP8 元素、scale 和块点积数值 oracle；正例和错误声明反例进入快速检查；真实 safetensors manifest 对拍名字、形状和 dtype。这里不宣称能加载或运行量化模型。
2. **Q2：直接加载并运行。** 量化权重在目标设备上保持量化存储并参与执行。先覆盖 FP8，再覆盖 MXFP4、专家融合和共享权重；与发布方实现对拍 logits、生成 token、峰值显存，并记录可复现的 GPU 环境。双方先解量化成同一浮点格式的测试只能作数值参考，不能作为“量化执行通过”的证据。
3. **Q3：生成量化产物。** 从浮点权重生成指定格式，输出包含配置、权重和共享关系的完整产物；重载后与源模型及 llama.cpp 对拍结构和 logits。

## 当前边界

`by1full` 的 Q2 加载入口已接收 `.by1` 量化规格：它会把已声明的格式与 checkpoint 原生量化配置核对，要求 `dequantize=False`，并在加载后审计 FP8 或 MXFP4 存储。审计结果按文本/视觉/MTP 模块分别计数；若声明 MTP / 共享词嵌入，还会检查这些别名关系。数值内核仍由 Transformers 原生量化适配器执行，by1 尚无自研 CUDA kernel。

**CPU 一半（2026-10-17 第十三轮）**：`safetensors` **头审计**
（`by1quant.audit_checkpoint_storage`）现在有**正反例判据**，跑在档 1（`by1fast` ->
`by1quant.selftest`）：合成分片（最小 safetensors：头 + 占位字节）的正例两条
（单分片 / 跨分片）+ 六条反例（scale dtype 不对、缺 scale、scale 形状不对、
权重不是 E4M3、分片间重名、数量对不上）。**这条审计原来只有
`by1gpubench` 一个调用者** —— 在没卡的机器上从没执行过。
为了让它在"只要 numpy"的那条腿里也跑得了，审计改用
`safe_open(..., framework="numpy")`（它只读 dtype/shape），CI 的 numpy 腿也因此
多装一个 `safetensors`。**这不是 Q2 通过**：运行期的数值内核仍
无自研实现，且 GPU forward / generation / 显存对拍仍未运行。
Q2 的 GPU forward / generation / 显存对拍仍未运行。现有 GPT-OSS checkpoint 可走原生 MXFP4 benchmark。FP8 首选现已固定为官方 `Qwen/Qwen3.8-27B-FP8`：它有独立量化描述 [Qwen3.8-27B-FP8.by1](../models/Qwen3.8-27B-FP8.by1)，基准使用完整 Qwen3.8 架构描述，覆盖文本、BF16 视觉塔、量化 MTP 和共享关系。原始 safetensors 头审计会核验合同声明的 BF16 scale、E4M3 权重、块形状与 `weight_scale_inv` 配对；运行时报告再记录实际 scale dtype。GPU 实测通过前不把 Q2 标为完成。Q3 仍未开始。

## 共享/附属分支检查清单（Q2 验收逐项，2026-10-14 固化）

每一项都有判据和反例 —— **反例不红就不算这项验过**：

| 分支 | 检查什么 | 判据 | 反例（必须红） |
|---|---|---|---|
| tied embeddings | 运行时共享身份（同一参数对象）+ 磁盘单份存储 | `by1quant.audit_shared_weights` 过；gpubench 的 tying 断言过 | 两个逻辑名各存一份 → 红 |
| 视觉塔 | 契约没点名的塔保持 BF16，审计**分类计数** | `checkpoint_quantized_storage` 分类与契约一致 | 视觉塔被静默量化 → 红 |
| MTP draft | `mtp.*` 按契约量化/不量化；embed/lm_head 与主干共享 | 同上 + `mtp.embed_tokens is get_input_embeddings()` 断言 | 共享断开或 mtp 被静默丢弃 → 红 |
| 权重别名 | 别名不重复量化、不重复分配、不静默拆份 | `audit_shared_weights` 的别名表 | 同一底层权重被量化两次 → 红 |

## Q3 前置评审（2026-10-14 记录，进 Q3 前先认这五条）

1. **格式冻结**：`by1quant.FORMATS` 表就是冻结面。新格式必须有真实
   manifest + 独立判据才能进表 —— 不因格式名相同就假定布局相同。
2. **生成用上游实现**：量化器用发布方/社区**在维护的**实现
   （GGUF 走 llama.cpp 的 convert；HF 格式走官方量化器），
   by1 只管合同、参数映射与验证 —— 不重造量化 kernel。
3. **round-trip 判据**：产物重载后逐项对 —— 权重名、dtype、块数、尺度、
   别名、分片、总字节；再由**未修改的上游**加载并 logits 对拍。
   少一项不算 round-trip 过。
4. **拒绝私有扩展是默认行为**：标准格式表达不了的语义，先证明
   "表达不了"（附格式对比证据）再议扩展；扩展未声明的输入一律拒绝，
   不静默按最接近的格式解读。
5. **依赖顺序不许跳**：格式（Q1 冻结）→ 运行数学（Q2）→ 产物（Q3）。
   Q2 没过就生成的产物没有判据可依 —— "能生成"不是"生成对了"。
