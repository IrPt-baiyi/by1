#!/usr/bin/env python3
"""by1sample.py -- 采样与生成闭环的**可执行规格**（T25）。

判据：llama.cpp sampler 同种子逐 token。分两级，不混着报：

  · **greedy 一族**（temp<=0）：argmax —— 确定性，可与 llama-cli 逐 token
    对拍（T25-1，判卷人 = llama-cli --temp 0 的续写 ids）。
  · **随机族**（温度/top-k/top-p/min-p/惩罚）：规格与自检在这（手算
    对拍 + 反例必红）；与 llama.cpp 的 RNG **同种子逐 token** 需要对齐
    mt19937 + 实现定义的均匀映射（libstdc++ 的 generate_canonical）——
    单列 T25-2，不在这冒充。

公式逐条照 llama.cpp 的 sampler 语义：
  · repetition penalty（llama_sampler_penalties）：出现过的 token，
    正 logit 除以 penalty、负 logit 乘以 penalty（penalty>1 时压制复读）
  · presence / frequency：logit -= presence + freq * count
  · temperature：logit /= temp
  · top-k：只留 k 个最大，其余 -inf
  · top-p：softmax 降序累计，超过 p 的截掉（**保住第一个**）
  · min-p：概率 < p * max_prob 的截掉
"""
import math


def apply_penalties(logits, history, rep=1.0, presence=0.0, freq=0.0):
    """llama.cpp penalties：history 里出现过的 token 逐个处理。"""
    out = list(logits)
    counts = {}
    for t in history:
        counts[t] = counts.get(t, 0) + 1
    for t, c in counts.items():
        v = out[t]
        if rep != 1.0:
            v = v / rep if v > 0 else v * rep
        v = v - presence - freq * c
        out[t] = v
    return out


def apply_temperature(logits, temp):
    if temp <= 0:
        raise ValueError("temp<=0 走 greedy，不做除法")
    return [v / temp for v in logits]


def apply_top_k(logits, k):
    if k <= 0:
        return list(logits)
    idx = sorted(range(len(logits)), key=lambda i: -logits[i])
    cut = set(idx[k:])
    return [(-math.inf if i in cut else v) for i, v in enumerate(logits)]


def apply_top_p(logits, p):
    if p >= 1.0:
        return list(logits)
    order = sorted(range(len(logits)), key=lambda i: -logits[i])
    total, keep = 0.0, []
    # softmax 的累计（减 max 保稳定）
    mx = logits[order[0]]
    zs = [math.exp(logits[i] - mx) for i in order]
    s = sum(zs)
    for j, i in enumerate(order):
        keep.append(i)          # 第一个必保
        total += zs[j] / s
        if total >= p:
            break
    keepset = set(keep)
    return [(-math.inf if i not in keepset else v)
            for i, v in enumerate(logits)]


def apply_min_p(logits, p):
    if p <= 0:
        return list(logits)
    mx = max(logits)
    floor = math.log(p) + mx    # 概率 < p * max_prob 等价于 logit < log(p)+max
    return [(v if v >= floor else -math.inf) for v in logits]


def greedy(logits):
    return max(range(len(logits)), key=lambda i: logits[i])


def sample_mt(logits, seed):
    """T25-2 的占位：mt19937 + libstdc++ generate_canonical 的均匀映射。
    **现在不实现** —— 没对齐 llama.cpp 的实现定义映射之前，写出来
    就是假的"同种子"。返回 None = 未实现（不冒充）。"""
    return None


def selftest():
    """反例必红 + 手算对拍。"""
    bad = []
    # greedy
    if greedy([0.1, 0.9, 0.5]) != 1:
        bad.append("greedy argmax 错")
    # penalties：history=[1]，rep=2 -> logit[1] 减半方向（正数除）
    out = apply_penalties([2.0, 4.0, -4.0], [1, 2], rep=2.0, presence=1.0,
                          freq=0.5)
    # t=1: 4/2 -1 -0.5 = 0.5；t=2: -4*2 -1 -0.5 = -9.5
    if abs(out[1] - 0.5) > 1e-12 or abs(out[2] + 9.5) > 1e-12:
        bad.append("penalties 手算不符：%r" % out)
    # top-k=2
    out = apply_top_k([1.0, 3.0, 2.0], 2)
    if out[0] != -math.inf or out[1] != 3.0 or out[2] != 2.0:
        bad.append("top-k 截错：%r" % out)
    # top-p=0.8：softmax([4,3,1])=[0.705,0.259,0.036] -> 前两个累计 0.964
    out = apply_top_p([4.0, 3.0, 1.0], 0.8)
    if out[2] != -math.inf or out[0] == -math.inf or out[1] == -math.inf:
        bad.append("top-p 截错：%r" % out)
    # min-p=0.2：max=4 -> floor=log(0.2)+4=2.397；2.0 被截、3.0 留
    out = apply_min_p([4.0, 3.0, 2.0], 0.2)
    if out[2] != -math.inf or out[1] != 3.0:
        bad.append("min-p 截错：%r" % out)
    # 负例：temp=0 必须拒绝做除法（except 里不许只 pass —— by1lint ①）
    rejected = False
    try:
        apply_temperature([1.0], 0.0)
    except ValueError:
        rejected = True
    if not rejected:
        bad.append("temp=0 没被拒 —— 除零会出 inf")
    if bad:
        for b in bad:
            print("  [FAIL] " + b)
        return 1
    print("  [PASS] by1sample：greedy / penalties / top-k / top-p / min-p "
          "手算对拍 + 负例必红（随机族 RNG 对齐 = T25-2 未实现不冒充）")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
