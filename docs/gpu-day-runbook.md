# GPU 日 runbook —— 租卡那天照这张纸走

> 单入口：`python src/checks/by1gpuday.py`（三相按序，报告分文件落盘）。
> 本文是它的配套：**资源核对表**（跑之前）和**回收模板**（跑之后填）。
>
> 影子和真权重**永远分列**：影子 = 接线对拍（随机值），真权重 = 真值结论。
> 两者的报告文件、模板行、结论口径都不许混。

## 一、租卡之前：资源核对表

| 项 | 要求 | 不够会怎样 | 怎么查 |
|---|---|---|---|
| GPU 显存 | ≥ 80 GB（Qwen3.8-27B BF16 占 ~55 GB，短上下文前向） | 27B 相跳过 | `nvidia-smi --query-gpu=name,memory.total --format=csv` |
| CPU 内存 | ≥ 260 GB（A3-0 e2e 全量 245 GB float32 足迹） | **e2e 全量不跑**（硬跑会冻结整机，这个账记过） | `free -g` |
| 磁盘 | ≥ 60 GB（31 GB FP8 + 影子 + 临时输出） | 真权重相失败 | `df -h .` |
| 文件存储 | 同区域挂载 + `by1cache verify` 全绿 | 又得重下（网速不够） | 见文件存储操作单（已移出仓库，见 git 历史） |
| 权重缓存 | `HF_HOME` 指向文件存储，`cache-manifest.json` 在旁边 | 同上 | `python src/data/by1cache.py verify <清单>` |

`by1gpuday.py` 的预检会自动查前四项并**明说哪项不够** —— 跳过要出声，
不硬跑。

## 二、一条命令

```bash
export HF_HOME=/root/autodl-fs/hf-cache      # 权重走文件存储（0 外网）
export HF_HUB_OFFLINE=1                      # 缺文件当场报错，不悄悄下
python src/checks/by1gpuday.py               # 三相全跑
```

分相开关：`--shadow-only`（0 下载，验接线）· `--real-only`（真权重 Q2 链）
· `--plan`（只列步骤）· `--e2e-mem-gb 260`（按机器调预算）。

三相的内容：

| 相 | 跑什么 | 报告 |
|---|---|---|
| e2e | `by1graph --e2e --mem-gb 260`（rc0 挂账的"租机跑全"） | `cgen/gpuday/e2e-full.txt` |
| 影子 | 冒烟三家族 + bf16 边界 + refs 元对照 + 影子 bench/train（GPU） | `cgen/gpuday/smoke-*.txt`、`gpu-benchmark-shadow.json`、`gpu-traincheck-shadow.json` |
| 真权重 | bench/train 真权重（Q2 链；MXFP4/FP8 各模式见 gpu/README.md） | `cgen/gpu-benchmark-real.json`、`gpu-real-traincheck.json` |

## 三、跑完：回收模板（照抄进 docs/gpu-validation-results.md）

```markdown
## GPU 日 <YYYY-MM-DD>（机器：<GPU 型号 / 显存 / 内存>）

### synthetic（影子对拍 —— 接线结论，非真值）
- <模型>：前向 max|Δ|=<x> · 生成 token 一致 · 梯度/更新 <x>
  → 报告：cgen/gpuday/gpu-benchmark-shadow.json
- 没验（抄 shadow-manifest 的 not_verified，**不许只引一半**）：
  - 真权重数值语义
  - 量化存储格式
  - <图像塔/processor 等本次没跑的>

### 真权重（Q2 —— 真值结论）
- <模型>：存储审计 <x/y> · 前向 max|Δ|=<x> · 生成 <一致/不一致> · 显存 <x>
  → 报告：cgen/gpuday/gpu-benchmark-real.json
- 量化：<格式> 原生存储审计过 / 运行时无静默解量化 <是/否>

### A3-0 e2e 全量
- 一致 <N> 份 · 超预算跳过 <M> 份（合计 <x> GB）· 对拍无法进行 <K> 份（在账）
  → 报告：cgen/gpuday/e2e-full.txt

### 缺口（新发现的每一项都写，没有就写"无"）
- <缺口>：<原因/证据> → <归到哪张账：INVENTORY【F】/ shadow-manifest / 上游>
```

## 四、回来之后的三件事

1. 模板填进 `docs/gpu-validation-results.md`（synthetic 单列，不混排）。
2. 新缺口归账：INVENTORY【F】（没验的）· shadow-manifest 的 not_verified ·
   `by1all` 的 KNOWN（已知缺口）—— **归账才算完，发现了没归 = 藏起来了**。
3. `by1cache scan` 把用过的权重记进清单；重租卡后 `verify` 一下就能复用。
