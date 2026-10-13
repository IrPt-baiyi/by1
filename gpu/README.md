# GPU Validation and Real-Checkpoint Benchmarks

GPU validation has two rounds, with separate reports. A GPU pass requires actual CUDA execution. When CUDA is unavailable, a check prints `[SKIP]` and exits with code 30; CPU self-checks do not count as GPU passes. See the [validation results](../docs/gpu-validation-results.md) for the measured summary and raw JSON reports.

The commands below work in Bash and Windows PowerShell. The PowerShell blocks use the project's virtual-environment interpreter. `gpu/run.sh` is a Bash script; on Windows, run the equivalent PowerShell commands shown here.

## 1. Full component checks at reduced size

Install a CUDA-enabled PyTorch build that matches the machine's driver first. Then install the project and the remaining GPU dependencies. `gpu/requirements.txt` deliberately does not include `torch`, so it will not replace the CUDA build.

**Bash:**

```bash
python -m venv .venv
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pip install -r gpu/requirements.txt accelerate
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
.venv/bin/python src/checks/by1gpusmoke.py --report cgen/gpu-smoke-fp32.json
.venv/bin/python src/checks/by1gpusmoke.py --dtype bfloat16 --report cgen/gpu-smoke-bf16.json
```

**Windows PowerShell:**

```powershell
python -m venv .venv
$PY = ".\.venv\Scripts\python.exe"
& $PY -m pip install -e . --no-deps
& $PY -m pip install -r gpu/requirements.txt accelerate
& $PY -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
& $PY src/checks/by1gpusmoke.py --report cgen/gpu-smoke-fp32.json
& $PY src/checks/by1gpusmoke.py --dtype bfloat16 --report cgen/gpu-smoke-bf16.json
```

The smoke checks cover `hello.by1` and six reduced-size complete architectures:

- Confirm parameters, buffers, inputs, and outputs are on CUDA; outputs and gradients contain no NaN or Inf.
- Compare the native model against independent NumPy operators and CPU PyTorch, and complete models against Transformers models built from independent configurations.
- Compare CPU/GPU gradients and confirm that the optimizer changes parameters.
- Run Qwen3.8 image and video forwards, check that image input affects text output, and compare MTP against an independent connection formula.
- Backpropagate through the MTP, vision, and text paths; confirm that the vision projection, MTP, and output head update.
- Save and reload a CUDA checkpoint and verify MTP parameter-object sharing.

FP32 checks use elementwise tolerances. For BF16 gradients compared with an FP32 CPU reference, the checks bound normalized L2 and peak error to avoid misleading relative errors near zero. Same-dtype GPU references still use elementwise checks. Criteria and measured errors are recorded in JSON; merely avoiding a crash does not count as a pass.

This round uses random weights and reduced model sizes. It checks that computation and connections execute on the GPU; it does not establish the capability of a full-size pretrained model.

## 2. Published checkpoints and a real training step

**Bash:**

```bash
.venv/bin/python src/checks/by1gpubench.py --report cgen/gpu-benchmark.json
.venv/bin/python src/checks/by1gputrain.py --report cgen/gpu-real-traincheck.json
```

**Windows PowerShell:**

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY src/checks/by1gpubench.py --report cgen/gpu-benchmark.json
& $PY src/checks/by1gputrain.py --report cgen/gpu-real-traincheck.json
```

By default, the benchmark tests GPT-2, Qwen3-0.6B, Qwen2.5-7B-Instruct, Qwen3-8B, gpt-oss-20b, and Qwen3.8-27B. To test one model first:

**Bash:**

```bash
.venv/bin/python src/checks/by1gpubench.py --model Qwen/Qwen3-0.6B --report cgen/qwen-small-benchmark.json
```

**Windows PowerShell:**

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY src/checks/by1gpubench.py --model Qwen/Qwen3-0.6B --report cgen/qwen-small-benchmark.json
```

Each model uses a pinned publisher revision and the same full checkpoint on both sides: the reference loads the publisher's configuration, while by1 compiles its configuration from the `.by1` file. The two large models load sequentially to avoid holding both in VRAM. The original Transformers Qwen3.8 class drops MTP parameters, so by1 strictly loads MTP and the report records this difference. The published gpt-oss MXFP4 checkpoint is dequantized to BF16 on both sides.

Defaults are BF16, batch size 1, text lengths 32 and 128, 2 warmups, 5 timed runs, and greedy generation of 16 new tokens. Qwen3.8 also receives image input processed by its original processor. Reports include maximum logit error, argmax agreement, generated-token equality, wall and CUDA time, prefill and generation throughput, and peak VRAM. Any output mismatch fails the benchmark; it is not just recorded alongside the speed.

**Both sides use the same Transformers operators.** This comparison checks configuration conversion, weight loading, branch wiring, numerical behavior, and runtime overhead. It does not show that a custom by1 CUDA kernel is faster than the official implementation, and it is not a task-quality evaluation such as MMLU. MTP is compared with an independent connection formula and timed, but this is not an end-to-end benchmark of the vLLM speculative-decoding backend.

The real-weight training check loads Qwen3-0.6B, runs BF16 forward and full-parameter backward passes, performs one AdamW update, then compares all gradients and updated parameter values between the official and by1 models. It validates one real-checkpoint training step, not multi-step large-model training.

If the network requires a Hugging Face mirror, set `HF_ENDPOINT` before starting the benchmark:

**Bash:**

```bash
export HF_ENDPOINT=https://your-hugging-face-mirror
```

**Windows PowerShell:**

```powershell
$env:HF_ENDPOINT = "https://your-hugging-face-mirror"
```

In an environment inherited from an older base image, a downloader error such as `process() takes no keyword arguments` may require upgrading Brotli in that environment:

**Bash:**

```bash
.venv/bin/python -m pip install --upgrade "brotli>=1.2"
```

**Windows PowerShell:**

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY -m pip install --upgrade "brotli>=1.2"
```

## 3. Hardware requirements and limits

The first round requires little VRAM. Real BF16 weights for the 27B model use about 55 GB; short-context forward comparisons are best run with 80 or 96 GB of VRAM, plus enough host memory and disk space. A 96 GB card does not guarantee that full-parameter AdamW training of a 27B model will fit.

RTX PRO 6000 Blackwell needs a compatible CUDA/PyTorch build. The test instance used PyTorch 2.8.0 and CUDA 12.8 with `sm_120` support, confirmed by actual CUDA matrix operations. For newer images, follow the vendor's compatibility matrix and avoid replacing an already working CUDA-enabled PyTorch build just to standardize versions.

`src/checks/by1gpu.py --plan 96` estimates VRAM candidates; it is not an execution report. The older random-weight FP32 forward check is not a substitute for the real BF16 checkpoint benchmark documented here.

---

# GPU 验证与真实权重 benchmark

GPU 验证分成两轮，结果分别保存。只有实际执行 CUDA 才算 GPU 通过。没有 CUDA 时会输出 `[跳过]` 并以退出码 30 结束；CPU 验收器自检不算 GPU 通过。实测摘要与原始 JSON 报告见[验证记录](../docs/gpu-validation-results.md)。

以下命令分别提供 Bash 和 Windows PowerShell 版本。PowerShell 命令使用项目虚拟环境中的 Python。`gpu/run.sh` 是 Bash 脚本；Windows 用户请直接运行这里对应的 PowerShell 命令。

## 1. 小尺寸完整组件验收

先安装与机器驱动匹配的 CUDA 版 PyTorch，再安装项目和其余 GPU 依赖。`gpu/requirements.txt` 特意不包含 `torch`，避免替换 CUDA 版安装。

**Bash：**

```bash
python -m venv .venv
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pip install -r gpu/requirements.txt accelerate
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
.venv/bin/python src/checks/by1gpusmoke.py --report cgen/gpu-smoke-fp32.json
.venv/bin/python src/checks/by1gpusmoke.py --dtype bfloat16 --report cgen/gpu-smoke-bf16.json
```

**Windows PowerShell：**

```powershell
python -m venv .venv
$PY = ".\.venv\Scripts\python.exe"
& $PY -m pip install -e . --no-deps
& $PY -m pip install -r gpu/requirements.txt accelerate
& $PY -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
& $PY src/checks/by1gpusmoke.py --report cgen/gpu-smoke-fp32.json
& $PY src/checks/by1gpusmoke.py --dtype bfloat16 --report cgen/gpu-smoke-bf16.json
```

验收覆盖 `hello.by1` 和六种缩小尺寸的完整架构：

- 确认参数、buffer、输入和输出都在 CUDA 上，输出和梯度没有 NaN/Inf。
- 原生模型对拍独立 NumPy 算子与 CPU PyTorch；完整模型对拍使用独立配置构建的 Transformers 模型。
- 对比 CPU/GPU 梯度，并确认优化器确实改变参数。
- 运行 Qwen3.8 图像和视频前向，检查图像输入是否影响文本输出，并用独立连接公式对拍 MTP。
- 反向传播经过 MTP、视觉和文本路径；确认视觉投影、MTP 与输出头都发生更新。
- 保存并重新加载 CUDA checkpoint，验证 MTP 参数对象共享。

FP32 检查逐元素容差。BF16 梯度对比 FP32 CPU 参考时，同时限制归一化 L2 与峰值误差，避免接近零的元素造成误导性的相对误差；同精度 GPU 参考仍逐元素检查。判据与实测误差都写入 JSON，不会把“没有崩溃”当作通过。

这一轮使用随机权重和缩小尺寸，证明计算与分支连接能在 GPU 上执行；它不证明完整预训练模型的能力。

## 2. 发布方 checkpoint 与真实训练步

**Bash：**

```bash
.venv/bin/python src/checks/by1gpubench.py --report cgen/gpu-benchmark.json
.venv/bin/python src/checks/by1gputrain.py --report cgen/gpu-real-traincheck.json
```

**Windows PowerShell：**

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY src/checks/by1gpubench.py --report cgen/gpu-benchmark.json
& $PY src/checks/by1gputrain.py --report cgen/gpu-real-traincheck.json
```

默认测试 GPT-2、Qwen3-0.6B、Qwen2.5-7B-Instruct、Qwen3-8B、gpt-oss-20b 和 Qwen3.8-27B。也可以先指定一个模型：

**Bash：**

```bash
.venv/bin/python src/checks/by1gpubench.py --model Qwen/Qwen3-0.6B --report cgen/qwen-small-benchmark.json
```

**Windows PowerShell：**

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY src/checks/by1gpubench.py --model Qwen/Qwen3-0.6B --report cgen/qwen-small-benchmark.json
```

每个模型都固定发布方 revision，两边使用同一份完整 checkpoint：对照组加载发布方配置，by1 从 `.by1` 文件编译配置。两份大模型顺序加载，以免同时占用显存。原始 Transformers Qwen3.8 类会丢弃 MTP 参数，因此 by1 严格加载 MTP，报告会单独记录这项区别。发布的 gpt-oss MXFP4 checkpoint 在两边都解量化到 BF16。

默认设置为 BF16、batch 1、文本长度 32 和 128、预热 2 次、计时 5 次、贪心生成 16 个新 token。Qwen3.8 还接收由原始 processor 处理的图像。报告记录 logits 最大误差、argmax 一致率、生成 token 是否相同、wall/CUDA 耗时、prefill/生成吞吐量和峰值显存。输出不一致会让 benchmark 失败，不会只附带记录速度。

**两边使用同一套 Transformers 算子。** 这项对比核验配置转换、权重加载、分支连接、数值行为和运行开销；不能证明 by1 自研 CUDA 内核比官方实现快，也不是 MMLU 等任务能力评测。MTP 会与独立连接公式对拍并计时，但这不是 vLLM speculative decoding 后端的端到端 benchmark。

真实权重训练检查加载 Qwen3-0.6B，执行 BF16 前向、全参数反向和一次 AdamW 更新，再比较官方模型与 by1 的所有梯度和更新后参数。这验证真实 checkpoint 的一步训练，不代表大模型多步训练。

如果网络环境需要 Hugging Face 镜像，请在启动 benchmark 前设置 `HF_ENDPOINT`：

**Bash：**

```bash
export HF_ENDPOINT=https://your-hugging-face-mirror
```

**Windows PowerShell：**

```powershell
$env:HF_ENDPOINT = "https://your-hugging-face-mirror"
```

继承较旧基础镜像的环境若遇到 `process() takes no keyword arguments` 一类下载器错误，可能需要在对应环境升级 Brotli：

**Bash：**

```bash
.venv/bin/python -m pip install --upgrade "brotli>=1.2"
```

**Windows PowerShell：**

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY -m pip install --upgrade "brotli>=1.2"
```

## 3. 机器需求与边界

第一轮显存需求较小。真实 27B BF16 权重大约占 55 GB；短上下文前向对比建议使用 80/96 GB 显存，并为主机内存和磁盘留出余量。96 GB 显存不保证 27B 全参数 AdamW 训练可以运行。

RTX PRO 6000 Blackwell 需要兼容的 CUDA/PyTorch 版本。本次实例的 PyTorch 2.8.0 与 CUDA 12.8 支持 `sm_120`，并通过了实际 CUDA 矩阵运算。较新的镜像请参考发布方兼容矩阵，不要为了统一版本盲目替换已经可用的 CUDA 版 PyTorch。

`src/checks/by1gpu.py --plan 96` 只估算显存候选，不是执行报告。旧的随机权重 FP32 前向检查也不能代替这里的真实 BF16 checkpoint benchmark。
