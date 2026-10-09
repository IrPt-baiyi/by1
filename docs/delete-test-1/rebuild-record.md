# 重建记录：MANIFEST.txt 的 9 个路径

被测仓库不在这台机器上（在另一台），所以这里的目标只有一件事：
**把这 9 个路径的字节，恢复成和原件逐字节相同的内容。**

产物在 `build/`，按原相对路径摆好。复现脚本：`rebuild_manifest.py`。

## 结果

| 路径 | 字节(清单) | 字节(重建) | 与副本 sha256 |
|---|---|---|---|
| `mixtral-shaped.by1` | 1156 | 1156 | 一致 |
| `models.tsv` | 3580 | 3580 | 一致 |
| `refs/openai-community__gpt2.config.json` | 665 | 665 | 一致 |
| `refs/openai-community__gpt2.tensors.json` | 8478 | 8478 | 一致 |
| `refs/deepseek-ai__DeepSeek-R1-Distill-Qwen-7B.config.json` | 708 | 708 | 一致 |
| `refs/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.tensors.json` | 584839 | 584839 | 一致 |
| `refs/Qwen__Qwen3-VL-2B-Thinking.tensors.json` | 41281 | 41281 | 一致 |
| `refs/zai-org__GLM-4.7-Flash.tensors.json` | 551559 | 551559 | 一致 |
| `refs/poolside__Laguna-XS-2_1.gguf-tensors.json` | 42139 | 42139 | 一致 |
| **合计** | **1234405** | **1234405** | |

## 7 个 refs/ 产物的来源与导出规则

这些文件不是"手写"的，是**公开产物的再导出**。规则是逐字节试出来的，
每一条都用副本本身做了校验（不是猜的）：

| 上游（HF） | 取什么 | 导出规则 |
|---|---|---|
| `openai-community/gpt2` | `config.json` | **原样**，连最后的换行都没有 |
| `openai-community/gpt2` | `model.safetensors` 头 | 保留 header 自己的键序；每个张量只留 `dtype`+`shape`，丢掉 `data_offsets`；`json.dumps(separators=(",",":"))`，UTF-8，无尾换行 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `config.json` | 上游文本 **LF → CRLF**（28 行 → +28 字节，680→708） |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | 14 个分片头 | 分片顺序拼接，**键序按 header 原序**；紧凑 JSON；6513 个张量 |
| `Qwen/Qwen3-VL-2B-Thinking` | `model.safetensors` 头 | **键名字典序排序**；值只留 shape；`indent=0` + **CRLF**；625 个张量 |
| `zai-org/GLM-4.7-Flash` | 48 个分片头 | **每个分片块内部排序**后按分片顺序拼接；值全是 `null`；`indent=0` + **CRLF**；9703 个张量 |
| `poolside/Laguna-XS-2.1-GGUF` | `Laguna-XS-2.1-Q4_K_M.gguf` 头 | 读 GGUF 张量表；**维度反序**（ggml `ne[]` 顺序）；`ggml_type` 保留数字枚举；紧凑 JSON；678 个张量 |

要点：
- **两种 writer**：紧凑单行（gpt2 / Nemotron / Laguna）与 `indent=0`+CRLF（Qwen3-VL / GLM）。
  GLM 是"单行"的错觉——它其实是**每个键一行**、每行以 `\r\n` 结束，值写成 `null`。
- **排序不是全局的**：GLM 只在每个分片块内部排序。全局排序会把它排错（`model.norm.weight` 会跑到第一个）。
- **取哪一份**：Laguna 两个 GGUF 里，只有 `Q4_K_M` 的头能和副本对上（`BF16` 那份的 `ggml_type` 分布是 BF16/F32，对不上）。
- 全部读取都用 HTTP Range **只取文件头**：Nemotron 584 MB、GLM 551 MB 的正文一个字节都没下。

复现时的上游提交（可追溯）：

```
openai-community/gpt2                              (sha 见脚本输出)
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B            (sha 见脚本输出)
nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16  a9904d24bcc1d289a1950fa9d2b978c47cf903b9
Qwen/Qwen3-VL-2B-Thinking                          33e0ad94c327808ebc7d3bbc97e78e600f1eeecd
zai-org/GLM-4.7-Flash                              7dd20894a642a0aa287e9827cb1a1f7f91386b67
poolside/Laguna-XS-2.1-GGUF                        1a37c0a5fb8c7a18e6106decb6be6327d1b63fa6
```

## 2 个手写文件

`mixtral-shaped.by1` 和 `models.tsv` **在上游没有任何来源**——它们是手写的：

- 一个是 by1 的模型声明（MoE：路由 + top-k + 逐专家 SwiGLU）；
- 一个是模型清单（长名/短名/HF id/说明，短名的规矩写在文件头的注释里）。

这两份的**内容**只能用 by1 那套源码（`by1*.py` / `ir.md` / `plan.md`）来判对错，
而那套源码不在这里。**没有它，这两个文件无法从零写对**，所以这一轮：
按副本的字节恢复，并如实标出这一点——它们不是"推出来的"，是"抄回来的"。
剩下的 7 个不是。

## 怎么重跑

```
python rebuild_manifest.py --out build
```
