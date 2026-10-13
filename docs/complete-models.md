# 完整模型：参数共享、视觉与 MTP

这些声明生成 **Transformers 后端的完整模型**。架构参数来自 `.by1`，编译器生成
配置并检查所选家族能否表达这些参数；运行时使用 Transformers 的模型与算子。
Qwen3.8 的 MTP 由 by1 补充实现，因为 Transformers 会忽略原 checkpoint 的 `mtp.*`。
原生 NumPy/C/ggml 后端仍有各自的支持集；完整描述不会被降成错误的串行原语 IR。

## 已提供的完整声明

完整示例采用展开排版：参数逐项列出，模块分块，较长调用换行。
Qwen3.8 示例还标注了文本、视觉、MTP 分支及参数共享关系，便于阅读和修改。

| 文件 | 完整性 |
|---|---|
| [complete-Qwen3-0.6B.by1](../models/complete-Qwen3-0.6B.by1) | embedding/lm_head 为同一个参数，QK norm、RoPE、SwiGLU |
| [complete-Qwen3-8B.by1](../models/complete-Qwen3-8B.by1) | 独立输出头，36 层 GQA |
| [complete-Qwen2.5-7B-Instruct.by1](../models/complete-Qwen2.5-7B-Instruct.by1) | Q/K/V 有 bias，输出投影无 bias |
| [complete-gpt2.by1](../models/complete-gpt2.by1) | 共享输出头、学习位置、LayerNorm、GELU、训练时 dropout |
| [complete-gpt-oss-20b.by1](../models/complete-gpt-oss-20b.by1) | MoE、偏置、sink、交替滑窗、YaRN、gptoss 激活 |
| [complete-Qwen3.8-27B.by1](../models/complete-Qwen3.8-27B.by1) | 64 层文本、27 层视觉、1 层 MTP、投影器、位置策略和共享关系 |

这里的 Qwen3.8 是 **Qwen3.8-27B**。Flash-Next 的 mHC/PLE 等不同结构尚未加入这个完整后端。

## 生成、加载与执行

```bash
python -m pip install -e ".[full]"
by1 check models/complete-Qwen3.8-27B.by1
by1 build models/complete-Qwen3.8-27B.by1 -o hello_model.py
```

生成源码需要 by1；执行需要 torch、Transformers 和 safetensors。
本实现与验收使用 Transformers 5.19，依赖声明为 `>=5.19,<6`。

```python
from hello_model import build, load

# 按真实尺寸核对模块和参数，不分配 27B 权重内存。
architecture = build(device="meta")

# 实际执行需足够内存和完整 checkpoint。
# load 验证配置，再使用 HF 的加载/量化机制；缺权重或多余权重会报错。
model = load("path/to/checkpoint", local_files_only=True)
output = model(
    input_ids=input_ids,
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
    mm_token_type_ids=mm_token_type_ids,
    output_hidden_states=True,
    labels=labels,
)
output.loss.backward()
```

完整模型也可通过 `by1 train` 训练。`--batch-file` 是用 `torch.save` 保存的 tensor 字典，
包含 `input_ids`、`labels`，以及需要的图像/视频输入。包含 MTP 的模型还必须提供
`mtp_input_ids`、`mtp_position_ids`、`mtp_labels`；可提供 `mtp_embedding_indices` 和
`mtp_hidden_indices`，从主干输入 embedding/最终 hidden 中选取对齐位置。
这些索引使多模态特征保持在梯度图中。加载 batch 使用 `weights_only=True`。

```bash
by1 train models/complete-Qwen3.8-27B.by1 --weights path/to/checkpoint --batch-file batch.pt --steps 1 --save trained-checkpoint
```

随机初始化时默认限制为一亿参数；大模型需提供 checkpoint 或显式提高 `--max-params`。
小尺寸完整模型的图像/MTP batch 已实测训练一步，保存后 MTP 参数确实发生变化。

图像/视频预处理、tokenizer 和 chat template 使用 checkpoint 对应的 `AutoProcessor`。
图像先经过 patch embedding、位置编码、视觉 Transformer 和 merger，再按视觉 token
位置进入文本 embedding。视频使用对应的 `pixel_values_videos`、`video_grid_thw`。
标准 `generate()` 与 KV/递归状态管理由 Transformers 后端处理。

## MTP 的输入与训练

MTP 有独立的参数、decoder 和最终 norm，embedding 与输出头引用主干的同一对象。
它不串入主干，不增加文本主干的 `num_hidden_layers`。

```python
draft = model.forward_mtp(
    next_input_ids,
    output.hidden_states[-1],   # 主干最终 norm 后的 hidden
    next_position_ids,          # 下一 token 的绝对位置；支持 [B,T] / [3,B,T]
    labels=draft_target_ids,    # 与 draft 输出逐位置对齐；不是普通 next-token labels
)
draft["loss"].backward()
```

多模态 MTP 可传 `inputs_embeds=next_inputs_embeds`，使已合并图像/视频特征的 embedding
进入 MTP。训练时应按预测深度对齐下一 token embedding、主干 hidden 和目标；模块不猜
调用方的数据对齐方式。`step` 选择预测层。增量调用可传 cache 和完整 attention mask。
`forward_mtp` 提供草稿预测计算；`generate()` 当前仍使用标准自回归循环，不自动启用
MTP 的接受/拒绝加速循环。

MTP 连接：分别 RMSNorm embedding 与主干 hidden → 拼接 → `fc` → 全量 decoder → norm
→ 共享输出头。依据 [vLLM 的 Qwen3.5 MTP 实现](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py)。

## 验收与边界

```bash
python src/checks/by1complete.py
```

验收包括：

- 所有六份声明编译；真实尺寸的 Qwen3.8 **1199 个张量逐名逐形状相同**。
- 小尺寸保留完整组件的六种模型，对独立构造的官方配置进行前向、训练 dropout 和梯度对拍。
- 参数共享检查对象身份；Qwen2 检查 QKV/output bias 的不同关系。
- 图像和视频前向；图像变化影响文本输出；MTP 梯度经过视觉/文本主干。
- MTP 对独立的连接公式；改 MTP 权重影响草稿，主干输出保持原有值。
- 完整 checkpoint 分片保存/加载往返，缺 MTP 权重必须拒绝。
- `by1 train` 使用图像/MTP batch 更新参数并保存全部分支。

没有在本机加载并执行 27B 真权重。1199 项核验使用 meta 模型和发布方元数据，数值和
梯度验收使用小尺寸完整模型。量化 checkpoint 的实际执行还依赖 HF 对该格式、设备
及内核的支持；本次没有运行真实 MXFP4 checkpoint。MTP 的连接公式与参数契约已验，
没有在本机运行 vLLM 的独立完整 MTP 后端。
