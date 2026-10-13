# AutoDL 文件存储操作单 —— 重租卡 0 下载

> **目标：** 权重只下一次。重租卡、换实例、甚至换卡型，都不再走一遍
> "网速不够"的痛苦 —— 下载是这个仓库最贵的操作（影子线就是为它开的）。
>
> **判据：** 按本单做完，新实例上 `by1cache verify` 全绿 + benchmark
> 从缓存直接加载，**外网流量为 0**。
>
> 路径/配额/挂载点以 AutoDL 控制台为准；本单写的是流程与命令，
> 控制台按钮名可能改版。

## 0. 原理（一句话）

AutoDL 的**文件存储**是同区域多实例共享、跨租期存活的盘；把它当
权重仓，`HF_HOME` 指过去，huggingface_hub 的缓存就天然复用。

## 1. 首次租卡：把权重下进文件存储

```bash
# 文件存储一般挂在 /root/autodl-fs（没看到就去控制台挂载）
ls /root/autodl-fs

# 权重缓存整个放文件存储（HF_HOME 是 huggingface_hub 的缓存根）
export HF_HOME=/root/autodl-fs/hf-cache
mkdir -p "$HF_HOME"

# 学术加速（下载 huggingface.co 时开，下完可关）
source /etc/network_turbo

# 下载目标模型（以 Qwen3-0.6B 为例；大数据集/模型同理）
huggingface-cli download Qwen/Qwen3-0.6B \
    --include '*.json' '*.safetensors' '*.txt' '*.model' '*.jinja' \
    --local-dir "$HF_HOME/models--Qwen--Qwen3-0.6B/snapshots/manual"
# （或者让 huggingface_hub 自己管缓存布局：不带 --local-dir 直接
#  snapshot_download，产物落在 $HF_HOME/hub/ 下，两种布局 by1cache 都认）

unset http_proxy https_proxy     # 关学术加速，别让后面的国内流量绕路
```

## 2. 下完立刻记清单（防"下次怎么知道没坏"）

```bash
python src/data/by1cache.py scan /root/autodl-fs/hf-cache \
    -o /root/autodl-fs/cache-manifest.json
```

清单和权重放同一个文件存储 —— **清单跟着权重走**，才不会出现
"权重在、清单在另一台机器上"的孤儿。

## 3. 重租卡（或换实例）之后

```bash
# ① 控制台确认同一区域的文件存储已挂载
ls /root/autodl-fs/hf-cache

# ② 校验没坏（改一个字节都报红 —— 这是 by1cache 的判据）
python src/data/by1cache.py verify /root/autodl-fs/cache-manifest.json

# ③ 指过去，直接跑
export HF_HOME=/root/autodl-fs/hf-cache
python src/checks/by1gpubench.py --report cgen/gpu-benchmark.json
```

`huggingface_hub` 看到缓存里有完整 snapshot 就**不会走网络** ——
`HF_HUB_OFFLINE=1` 可以把这件事变成硬保证：

```bash
export HF_HUB_OFFLINE=1   # 有缓存就用缓存；缺文件当场报错，不悄悄下
```

## 4. 边界与对策（说在前面）

- **文件存储有配额**（以控制台为准）。大模型（30B+ 的 BF16/FP8）
  可能放不下 → 只把**真权重轮次要用的**放进去；影子轮次（`--shadow`）
  本来就是 0 下载，不占配额。
- **同区域才共享。** 跨区域租卡要先从文件存储拉到本地，那一步还是
  内网速度（快），不是外网（慢）。
- **量化 checkpoint 与 BF16 分开放目录**，清单各记各的 —— 混在
  一个 snapshot 里，`from_pretrained` 会两份都看到。
- **坏了一份怎么办**：`verify` 点名的文件单独重下（`huggingface-cli
  download --include 那个文件`），重下后**重跑 scan** 更新清单 ——
  清单是"下载时的真值"，不是"应该是什么"的猜测。

## 5. 这份操作单的判据回执

做完第 3 步后，在新实例上跑：

```bash
python src/data/by1cache.py verify /root/autodl-fs/cache-manifest.json
# 期望：校验 N 个文件：一致 N · 问题 0
```

再看 benchmark 报告的 `download_endpoint` 与缓存命中：外网流量为 0
即达成"重租卡 0 下载"。
