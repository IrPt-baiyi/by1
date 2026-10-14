# Qwen3-ASR-0.6B-GGUF 音频前处理边界实测

**对象**：`ggml-org/Qwen3-ASR-0.6B-GGUF` @ `928ab958557df9aa2ef1c93e0e83c7ad0933fae2`
**日期**：2026-10-14
**环境**：Windows · CPU-only（无 GPU）· transformers 4.57.6 · Python 3.12.10

---

## 1. 先确认测的是官方产物

本地文件与上述 commit 的 Git LFS 指针 SHA256 逐一对齐，四个文件全部一致：

| 文件 | SHA256 | 字节 |
|---|---|---:|
| `mmproj-Qwen3-ASR-0.6B-bf16.gguf` | `dae36c855f9a82a8916bea2238b24bda69a39d8da8b2f46dee7c103775656039` | 378,575,520 |
| `mmproj-Qwen3-ASR-0.6B-Q8_0.gguf` | `41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d` | 214,392,480 |
| `Qwen3-ASR-0.6B-Q8_0.gguf` | `bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971` | 804,749,248 |
| `Qwen3-ASR-0.6B-bf16.gguf` | `12b2894d9a7c98cd8f26670f5a47ab738f42bcc98df21e109be493870c71ba50` | 1,509,343,168 |

## 2. 两个 mmproj 是受控对照

`bf16` 与 `Q8_0` 两个 mmproj 的张量构成（302 个）：

| dtype | bf16 版 | Q8_0 版 |
|---|---:|---:|
| f32 | 188 | 188 |
| **权重** | **bf16 × 110** | **q8_0 × 110** |
| f16 | 4 | 4 |

**唯一差异是那 110 个权重张量。** 其余 192 个张量逐位相同，因此任何数值差异 100% 归因于量化，无其他变量。

## 3. 音频前处理边界（本报告实测）

输入：440 Hz 正弦，16 kHz，16000 样本（1 s）。
mtmd 侧 `clip_init` → `mtmd_audio_preprocessor_qwen3a`；参照侧 transformers 的
`WhisperFeatureExtractor`（据 `preprocessor_config.json`
`"feature_extractor_type": "WhisperFeatureExtractor"`，chunk 30 s / hop 160 / n_fft 400 / 128 bins）。

| 对比区域 | max \|Δ\| | mean \|Δ\| | 逐位相同的值 |
|---|---:|---:|---:|
| **前 99 帧（干净区）** | **1.204e-05** | **1.322e-07** | 307 / 12,672 |
| 前 199 帧（含 1 个边界帧） | 0.565127 | 0.180319 | 307 / 25,472 |
| 前 200 帧（含 2 个边界帧） | 0.565127 | 0.181238 | 307 / 25,600 |

逐帧看，突变位置是**明确的**：

```
frame   0–98 : max|Δ| ≈ 2.4e-07     ← 与 HF 逐位级一致
frame  99    : max|Δ| = 0.163       ← 突变
frame 100–199: 0.20 – 0.57          ← 持续
```

## 4. 成因：mtmd 按 chunk 算 mel，HF 整段算一次

- HF：30 s 填充后**一次**计算，全部帧共享同一个归一化基准 → 全帧"干净"
- mtmd：**按 chunk** 计算再拼接，第 1 个 chunk 在 **frame 99** 结束（100 帧/chunk），
  **边界帧的归一化基准与整段计算不同**

这解释了另一个现象：mtmd 的 mel 帧数随输入变化
（1 s → 200 帧 1 chunk；2 s → 300 帧 1 chunk；30 s → 3100 帧 4 chunks），
而非线性增长——与 30 s chunk 内的重采样/填充策略有关。

## 5. 两条操作结论

1. **做 mtmd ↔ HF 对照时，只能比干净区。**
   若把 mtmd 的全部 200 帧 mel 直接喂给 HF 音频塔，会把**跨 chunk 边界的不连续**
   （max 0.565）注入参照侧——而原本 mtmd 的自然输出里没有这个不连续。
   干净区（前 100 帧）才是方法学上可比的区域。

2. **量化代价（待 GPU 复现）**：在同一批干净帧下，bf16 与 Q8_0 的塔边界为
   `mean 1.42e-04 / cos 0.99987` vs `mean 6.36e-04 / cos 0.99906`——
   **局部约 4.5 倍劣化**。该数字来自此前的 GPU 会话，**本次 CPU 复现未做**，
   列在此处仅为标明干净的对照方法。

## 6. 一个容易踩的坑：transformers 版本

`config.json` 自报 `transformers_version: 4.57.6`，但：

| transformers | `qwen3_asr` 实现 |
|---|---|
| **4.57.6** | **不存在**（`models/` 下无 `qwen3_asr`） |
| 5.15.1 | 存在（`modeling_qwen3_asr.py` 等 6 个文件） |

在 4.57.6 下 `AutoProcessor.from_pretrained()` **不报错**，但静默返回
`Qwen2TokenizerFast`（没有 `feature_extractor`），程序照常运行。
按名字直接取 `WhisperFeatureExtractor` 可绕开。

## 7. 复现

```bash
# mtmd 侧（CPU，无需 GPU / Python）
tools/audio-mel-probe.exe model-cache/audio/mmproj-Qwen3-ASR-0.6B-bf16.gguf 16000 \
  > mtmd_mel.txt

# 参照侧（transformers 4.57.6）
python tools/qwen3asr-mel-ref2.py model-cache/qwen3asr-hf 16000 hf_mel.txt

# 对比（干净区）
python history/_hf_probe/cmp_mel.py mtmd_mel.txt hf_mel.txt 99
python history/_hf_probe/per_frame_diff.py mtmd_mel.txt hf_mel.txt 200
```

工具：`tools/audio-mel-probe.cpp`（mtmd 侧判卷人）·
`tools/qwen3asr-mel-ref2.py`（参照侧判卷人，直取 `WhisperFeatureExtractor`）·
`history/_hf_probe/cmp_mel.py` · `history/_hf_probe/per_frame_diff.py`
