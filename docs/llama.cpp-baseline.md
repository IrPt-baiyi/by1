# llama.cpp Baseline / llama.cpp 基线

This document freezes the first clean upstream comparison for the next project round. It separates llama.cpp's standard GGUF runtime baseline from the Transformers FP8-versus-by1 benchmark in [the GPU guide](../gpu/README.md).

本文固定下一轮第一份干净上游对照。llama.cpp 标准 GGUF 运行基线与 [GPU 指南](../gpu/README.md) 中 Transformers FP8 对 by1 的比较分开记录。

## Pinned inputs / 固定输入

| Item | Fixed value |
|---|---|
| llama.cpp | `ggml-org/llama.cpp` tag `v0.6.0`, release commit prefix `d812350` |
| Model repository | `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` |
| Model revision | `c7b6817b8affebf3b38c58313394b061345a4924` |
| Text GGUF | `Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`, 11.8 GB, SHA-256 `64b53b64c7aa39f20a7e54bd80582fe595b1d745624ee8a72e92508c0326d810` |
| Optional MTP GGUF | `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` |
| Vision projector | `mmproj-Qwen3.8-27B-BF16.gguf` |

The selected model files are published as ordinary GGUF and include the matching vision projector and an optional MTP build. The `-mtp` file carries the model's MTP head alongside the same base weights, so test it as a self-speculative model with `--spec-type draft-mtp`; it is not a separate draft-only file. The quantized GGUF is a community artifact, so its speed is a llama.cpp baseline for that fixed artifact; it is not numerically equivalent to the Qwen BF16 or FP8 safetensors checkpoint.

所选模型文件是标准 GGUF，发布方另提供匹配的视觉 projector 和可选 MTP 版本。`-mtp` 文件在相同主权重旁加入 MTP 头，因此以 `--spec-type draft-mtp` 作为自推测模型测试；它不是单独的 draft-only 文件。量化 GGUF 来自社区，因此它用于固定该文件的 llama.cpp 性能基线；它与 Qwen BF16 或 FP8 safetensors checkpoint 精度不同，不作逐 logits 等价比较。

The workspace contains a downloaded, Git-ignored llama.cpp source directory whose CMake file says `0.6.0`, but it has no upstream Git metadata. Treat it as an unverified snapshot. The commands below make a clean clone and check its resolved commit before building.

工作区已有一个被 Git 忽略的 llama.cpp 源码目录，CMake 文件写着 `0.6.0`，但没有上游 Git 元数据，不能视作已验证基线。以下命令会重新 clone，并在构建前核对解析出的 commit。

## Build / 构建

Run from the By1 repository root with CMake, Git, a C++ compiler, and a CUDA toolkit installed. Record `nvidia-smi`, `nvcc --version`, and the full `git rev-parse HEAD` output in the result report.

在 By1 仓库根目录执行；环境需安装 CMake、Git、C++ 编译器和 CUDA toolkit。将 `nvidia-smi`、`nvcc --version` 及完整的 `git rev-parse HEAD` 输出记入结果报告。

**Linux / Bash**

```bash
git clone --depth 1 --branch v0.6.0 https://github.com/ggml-org/llama.cpp llamacpp/upstream-v0.6.0
test "$(git -C llamacpp/upstream-v0.6.0 rev-parse --short HEAD)" = "d812350"
cmake -S llamacpp/upstream-v0.6.0 -B llamacpp/upstream-v0.6.0/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llamacpp/upstream-v0.6.0/build --parallel
```

**Windows PowerShell**

```powershell
git clone --depth 1 --branch v0.6.0 https://github.com/ggml-org/llama.cpp llamacpp/upstream-v0.6.0
$LLAMA_REV = git -C llamacpp/upstream-v0.6.0 rev-parse --short HEAD
if ($LLAMA_REV -ne "d812350") { throw "Unexpected llama.cpp revision: $LLAMA_REV" }
cmake -S llamacpp/upstream-v0.6.0 -B llamacpp/upstream-v0.6.0/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llamacpp/upstream-v0.6.0/build --config Release --parallel
```

## Download the fixed GGUF files / 下载固定 GGUF 文件

The Hugging Face revision below is immutable. The primary GGUF's SHA-256 is supplied by its file page; verify it after download. Keep all three files together.

以下 Hugging Face revision 不可变。主 GGUF 的 SHA-256 来自其文件页，下载后应核对。三个文件放在同一个目录。

**Linux / Bash**

```bash
hf download ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf mmproj-Qwen3.8-27B-BF16.gguf --revision c7b6817b8affebf3b38c58313394b061345a4924 --local-dir models/qwen38-gguf
sha256sum models/qwen38-gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf
```

**Windows PowerShell**

```powershell
hf download ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf mmproj-Qwen3.8-27B-BF16.gguf --revision c7b6817b8affebf3b38c58313394b061345a4924 --local-dir models/qwen38-gguf
Get-FileHash models/qwen38-gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf -Algorithm SHA256
```

## Run the baseline / 运行基线

`llama-bench` reports prompt-processing and generation speed. The fixed run uses prompt lengths 32 and 128, 16 generated tokens, five repetitions, and full GPU layer offload. Redirect stdout to preserve the JSON report and stderr to preserve loader/backend details.

`llama-bench` 会报告 prompt 处理和生成速度。固定设置为输入长度 32 和 128、生成 16 个 token、重复 5 次、尽可能将所有层卸载到 GPU。stdout 保存 JSON 报告，stderr 保存权重加载和后端信息。

**Linux / Bash**

```bash
IMAGE="path/to/test-image.png"
llamacpp/upstream-v0.6.0/build/bin/llama-bench -m models/qwen38-gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf -p 32,128 -n 16 -ngl 99 -r 5 -o json > cgen/llamacpp-qwen38-iq3s.json 2> cgen/llamacpp-qwen38-iq3s.log
llamacpp/upstream-v0.6.0/build/bin/llama-cli -m models/qwen38-gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf -ngl 99 -p "Name one benefit of a reproducible software benchmark." -n 32 --single-turn
llamacpp/upstream-v0.6.0/build/bin/llama-mtmd-cli -m models/qwen38-gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf --mmproj models/qwen38-gguf/mmproj-Qwen3.8-27B-BF16.gguf --image "$IMAGE" -ngl 99 -p "Describe this image briefly." -n 32 --single-turn
llamacpp/upstream-v0.6.0/build/bin/llama-cli -m models/qwen38-gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --spec-type draft-mtp -ngl 99 -p "Name one benefit of a reproducible software benchmark." -n 32 --single-turn
```

**Windows PowerShell**

```powershell
$LLAMA = "llamacpp/upstream-v0.6.0/build/bin/Release"
$MODEL = "models/qwen38-gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"
$MTP = "models/qwen38-gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
$MMPROJ = "models/qwen38-gguf/mmproj-Qwen3.8-27B-BF16.gguf"
$IMAGE = "path/to/test-image.png"
& "$LLAMA/llama-bench.exe" -m $MODEL -p 32,128 -n 16 -ngl 99 -r 5 -o json 1> cgen/llamacpp-qwen38-iq3s.json 2> cgen/llamacpp-qwen38-iq3s.log
& "$LLAMA/llama-cli.exe" -m $MODEL -ngl 99 -p "Name one benefit of a reproducible software benchmark." -n 32 --single-turn
& "$LLAMA/llama-mtmd-cli.exe" -m $MODEL --mmproj $MMPROJ --image $IMAGE -ngl 99 -p "Describe this image briefly." -n 32 --single-turn
& "$LLAMA/llama-cli.exe" -m $MTP --spec-type draft-mtp -ngl 99 -p "Name one benefit of a reproducible software benchmark." -n 32 --single-turn
```

Set `IMAGE` to a saved test image and record its SHA-256. Run text generation, image input, and embedded MTP as separate checks. `llama-bench` does not measure tokenization or sampling, and it does not report a reliable peak-VRAM value; capture GPU memory with an external sampler during the run. The run is pending a GPU instance. These commands are the frozen recipe, not measured results.

将 `IMAGE` 设为保存好的测试图片路径，并记录该图片的 SHA-256。文本生成、图像输入和内嵌 MTP 分别验收。`llama-bench` 不包含 tokenizer 和采样耗时，也不提供可靠的峰值显存；运行时需用外部工具采样显存。当前仍待 GPU 实例运行。以上命令是已固定的复现配方，不是实测结果。

## 本轮实测：干净 clone 构建（2026-10-17，Windows / MinGW / CPU）/ Measured clean build

这一节是**实测**，不是配方。命令在仓库根目录执行，输出逐条可复核。

| 项 | 实测值 |
|---|---|
| clone | `git clone --depth 1 --branch v0.6.0` -> **11 秒**，`tag = v0.6.0`，`sha = d81235049384534c167caea52b85a694f6103d14` |
| configure | `cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_CURL=OFF -DGGML_NATIVE=ON` -> **2 秒** |
| build | `cmake --build build --target llama-bench llama-cli -j 8` -> **207 秒** |
| 产物 | `llama-bench.exe` 13.06 MB · `llama-cli.exe` 17.44 MB |
| 版本串 | 两个都是 `version: 0.6.0-dev (build 1, commit d812350)` |

**Windows 上多一个必须的编译宏**：MinGW 的 `cpp-httplib` 会报
`::CreateFile2 has not been declared`（它要求 `_WIN32_WINNT >= 0x0602`）。
加 `-DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00 -DCMAKE_C_FLAGS=-D_WIN32_WINNT=0x0A00`
之后通过。第一次不加它的构建 9 秒就死在 `vendor/cpp-httplib` —— 那个错误信息里
没有任何线索指向 `_WIN32_WINNT`，所以记在这里。

**工作区里那个旧的 `llamacpp/build-cli` 快照不是钉定版本**：它的版本串是
`0.6.0-dev (build 345, commit 9b0c827)`，而 `9b0c827` 在上游**查不到**（GitHub API
返回 422）；`v0.6.0` 那个 tag 的提交是 `d812350`。这正是本文开头说的
"不能视作已验证基线" —— 现在它有了机器判据（见下）。

**`llama-bench` 能跑起来，但还不能出数**：它要一个**完整模型** GGUF；仓库里
只有 tokenizer-only 的 `ggml-vocab-*.gguf`，喂给它会在加载期报
`check_tensor_dims: tensor token_embd.weight not found`。真正的 tokens/s 基线
仍等 11.8 GB 的 IQ3_S 下载（GPU 日那一条，见上节）。这一项**记档为未验**，
不是通过。

**机器判据**：`python src/checks/by1llama.py pin-check` ——
有构建时核对 tag/commit/两个二进制的版本串，**没构建时 `[跳过]`（rc=2）**。
对旧的 `llamacpp/build-cli` 跑会红（`9b0c827` + 缺 `llama-bench`）——
这就是"钉定值是真的"的证词。
## 基线判卷人 / Baseline judges

配方本身由 `src/checks/by1llama.py` 当判卷人（每项判据可机器复核）：

- `record` —— 基线记录表：tag/commit、编译器、CUDA、GPU、OS 逐项
  采集，**每个值带来源命令**，且与本文的 Pinned inputs 对账（不符即红）。
- `gguf-check --dir models/qwen38-gguf` —— 三个固定 GGUF 对 sha256；
  缺/坏/没 pin 都点名。
- `logits-compare ref got` —— 两个 logits dump **逐位**比较（rc1 判据
  "llama-cli logits 逐位"落在这），报告定位首个分歧值。
- `logits-run --gguf …` —— 跑 llama.cpp 产 dump（要构建；GPU 日跑）。

以上工具的输出都是报告文件 —— **命令、值、判据三样同框**，复核不用问人。
