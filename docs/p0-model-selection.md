# T1 · P0 代表模型选择记录（2026-10-14）

判据：选择记录包含模型、checkpoint revision、许可证、参考实现、上游运行时和硬件最低要求；明确哪些分支不纳入本轮。

## 选定模型

| 项 | 值 | 来源 |
|---|---|---|
| 模型 | **Qwen/Qwen3-0.6B**（base，文本） | 路线图"主线模型：默认 Qwen3-0.6B" |
| checkpoint revision | `c1899de289a04d12100db370d81485cdf75e47ca` | `HfApi.model_info` 实测 |
| 权重文件 | `model.safetensors`（单文件） | 仓库清单实测 |
| 许可证 | Apache-2.0 | HF tags 实测 |
| `.by1` 描述 | `models/complete-Qwen3-0.6B.by1`（已在库） | 仓库 |
| config 声明 | 28 层 / d_model 1024 / q 16 / kv 8 / head_dim 128（以 config 合同对拍为准，T5 逐项验） | 待 T5 |

## 参考与运行时

| 项 | 值 | 钉法 |
|---|---|---|
| 参考实现 | HuggingFace Transformers `Qwen3ForCausalLM`（发布方实现） | T2 锁版本 |
| 独立性 | 参考路径不经过 by1 编译结果（T6 判据） | — |
| 上游运行时 | **llama.cpp v0.6.0**（原版，路线图钉定） | T2 锁 commit + 构建选项 |
| Python | 3.12 | T2 锁 |
| 硬件最低 | CPU-only、16GB RAM（本机即验；GPU 项按计划跳过并明示 skip） | — |

## 本轮不纳入的分支（P0 边界）

照路线图"暂不把视觉、MTP、FP8 或 MXFP4 作为第一条闭环的必要条件"：

- 视觉塔 / 音频塔 / 多模态 —— **不纳入**（属 T21+ 的第二架构/阶段 E）
- MTP draft head —— **不纳入**
- FP8 / MXFP4 量化 —— **不纳入**（阶段 C/D）
- 训练步 —— **不纳入**（P0 只做推理：前向 + 短生成）

## 替代说明

无替代：Qwen3-0.6B 的 checkpoint、上游 llama.cpp 支持（Qwen3 架构）与本机条件（CPU/16GB）均满足；若后续条件变化，按判据在此记录替代物与原因。

## 判卷人

`HfApi.model_info`（revision/许可证实测）· `models/complete-Qwen3-0.6B.by1`（by1check）· 后续 T2–T10 的报告本体。
