# GPU 验证与真实权重 benchmark

GPU 验证分成两轮，结果分别保存。只有实际 CUDA 执行才记为 GPU 通过。
没有 CUDA 时返回 `[跳过]` 和退出码 30；CPU 验收器自检不会记为 GPU 通过。
实测结果摘要与原始 JSON 报告见[验证记录](../docs/gpu-validation-results.md)。

## 1. 小尺寸完整组件验收

需要 CUDA 版 PyTorch、NumPy、Transformers `>=5.19,<6` 和 safetensors。
先在 GPU 机器上确认 `torch.cuda.is_available()` 为 `True`，再安装本项目的依赖：

```bash
python -m pip install -e ".[full]" accelerate
python src/checks/by1gpusmoke.py --report cgen/gpu-smoke-fp32.json
python src/checks/by1gpusmoke.py --dtype bfloat16 --report cgen/gpu-smoke-bf16.json
```

验收包含原生 `hello.by1` 和六种缩小尺寸的完整架构：

- 参数、buffer、输入和输出确实在 CUDA 上；输出和梯度没有 NaN/Inf。
- 原生模型对 NumPy 独立算子和 CPU PyTorch；完整模型对独立构造配置的 Transformers 模型。
- CPU/GPU 梯度对比，并检查优化器真的更新参数。
- Qwen3.8 图像和视频前向、图像影响文本输出、MTP 独立连接公式。
- MTP 反向经过视觉与文本主干，视觉投影、MTP 和输出头发生参数更新。
- CUDA checkpoint 保存/加载往返及 MTP 参数对象共享。

FP32 保留逐元素容差；BF16 对 FP32 CPU 梯度同时约束归一化 L2 与峰值误差，
避免接近零的元素造成不合理的相对误差。与同精度的 GPU 参考仍做逐元素检查。
所有判据和实际误差都在 JSON 中，不把“未崩溃”当作完整验收。

这轮使用随机权重和缩小尺寸，证明计算与连接能在 GPU 上执行，不证明真实大模型的能力。

## 2. 发布方真实权重对比

```bash
python src/checks/by1gpubench.py --report cgen/gpu-benchmark.json
python src/checks/by1gputrain.py --report cgen/gpu-real-traincheck.json
```

默认依次测试 GPT-2、Qwen3-0.6B、Qwen2.5-7B-Instruct、Qwen3-8B、gpt-oss-20b、
Qwen3.8-27B。也可以先指定一个模型：

```bash
python src/checks/by1gpubench.py --model Qwen/Qwen3-0.6B --report cgen/qwen-small-benchmark.json
```

每个模型固定发布仓库的提交版本，分别加载发布方原始配置与 by1 编译的配置，
使用同一份完整 checkpoint。两份大模型顺序加载，避免同时占用显存。
Qwen3.8 的原始 Transformers 类会丢弃 MTP 参数，by1 则严格加载 MTP；
报告单独记录这个区别。gpt-oss 的发布 MXFP4 checkpoint 在两侧都解量化到 BF16。

默认设置是 BF16、batch 1、文本长度 32/128、预热 2 次、测量 5 次，
贪心生成固定 16 个新 token。Qwen3.8 另有经原始 processor 处理的图像输入。
记录完整 logits 最大误差、argmax 一致率和生成 token 是否相同，以及 wall/CUDA
耗时、prefill/生成吞吐量与峰值显存。输出不一致时 benchmark 失败，不仅记录速度。

**两边使用同一套 Transformers 算子。** 此对比核验配置转换、权重加载、分支连接和
运行开销，不能证明 by1 自研算子快于官方实现，也不是 MMLU 等任务能力评测。
MTP 与独立连接公式对比并计时，但不宣称已经对 vLLM 的完整 MTP 后端跑过 benchmark。

另有一项真实权重训练步：Qwen3-0.6B 的官方与 By1 模型分别跑 BF16 前向、全参数反向
和一次 AdamW 更新，再比较所有梯度与更新后的参数。它补充验证真实 checkpoint 的训练路径，
不表示已经执行大模型多步训练。

网络环境需要镜像时，可在启动前指定 `HF_ENDPOINT`；报告会记录实际 endpoint 和 revision。
在继承较旧基础镜像的环境中，如果下载器报 `process() takes no keyword arguments`，
请在隔离环境中升级 Brotli（本次遇到的版本是 1.0.9）：

```bash
python -m pip install --upgrade "brotli>=1.2"
```

## 3. 机器需求与边界

第一轮显存需求较小。真实 27B 的 BF16 权重约 55GB，短上下文前向建议使用
80/96GB 显存，并为主机内存和磁盘留余量。96GB 不能保证 27B 全参数 AdamW 训练。

RTX PRO 6000 Blackwell 需要对应的 CUDA/PyTorch 支持。本次实例的 PyTorch 2.8.0
与 CUDA 12.8 包含 `sm_120`，已通过实际 CUDA 矩阵运算；较新版本的镜像需按发布方
支持矩阵选择，不要为统一版本盲目替换已经可用的 CUDA PyTorch。

现有 `src/checks/by1gpu.py --plan 96` 是显存候选估算，不是执行记录；
旧脚本的随机 FP32 前向也不等同于这里的真实 BF16 checkpoint benchmark。
