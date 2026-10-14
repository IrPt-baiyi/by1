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

**T25-2 已对齐**（RNG 三条流 vs tools/rng-probe.cpp 同编译器 oracle）：
mt19937 原始流 · uniform01 = (g0+g1·2^32)/2^64 · discrete 索引。
**仍欠链级逐 token**：filters 的 float32 浮点序（llama 的 filter 全是
float 运算，本文件是 Python float）——要逐位就得 np.float32 按同操作
序复刻 top_k 排序序 / top_p 累计 / dist 内的 softmax。单列不冒充。
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


class MT19937:
    """libstdc++ 同款 mt19937 核（标准 624 状态 + tempering，标准即规格）。

    T25-2 的地基。上层两个映射也是 libstdc++ 的实现定义：
      · uniform01 = generate_canonical<double,53> = (g0 + g1·2^32)/2^64
      · discrete  = std::discrete_distribution：double 累计 + lower_bound
    判卷人：tools/rng-probe.cpp（同编译器 = 同 libstdc++ = oracle）。"""

    def __init__(self, seed):
        self.mt = [0] * 624
        self.mt[0] = seed & 0xFFFFFFFF
        for i in range(1, 624):
            x = self.mt[i - 1]
            self.mt[i] = (1812433253 * (x ^ (x >> 30)) + i) & 0xFFFFFFFF
        self.idx = 624

    def genrand(self):
        if self.idx >= 624:
            for i in range(624):
                y = (self.mt[i] & 0x80000000) | (self.mt[(i + 1) % 624] & 0x7FFFFFFF)
                self.mt[i] = self.mt[(i + 397) % 624] ^ (y >> 1)
                if y & 1:
                    self.mt[i] ^= 0x9908B0DF
            self.idx = 0
        y = self.mt[self.idx]
        self.idx += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF

    def uniform01(self):
        g0 = self.genrand()
        g1 = self.genrand()
        return (g0 + g1 * 4294967296.0) / 18446744073709551616.0

    def discrete(self, probs):
        cum, s = [], 0.0
        for p in probs:
            s += float(p)
            cum.append(s)
        if s <= 0:
            return 0
        t = self.uniform01() * s
        for i, c in enumerate(cum):
            if c >= t:
                return i
        return len(cum) - 1


def sample_mt(logits, seed, temp=1.0, top_k=0, top_p=1.0, min_p=0.0,
              history=(), rep=1.0, presence=0.0, freq=0.0):
    """T25-2：随机族采样。链 = penalties -> temperature -> top-k ->
    top-p -> min-p -> softmax -> discrete（照 common_sampler 的顺序）。
    RNG 与 libstdc++ 对齐；判卷人 tools/rng-probe.cpp。"""
    v = apply_penalties(logits, history, rep, presence, freq)
    if temp <= 0:
        raise ValueError("temp<=0 走 greedy")
    v = apply_temperature(v, temp)
    v = apply_top_k(v, top_k)
    v = apply_top_p(v, top_p)
    v = apply_min_p(v, min_p)
    mx = max(x for x in v if x != -math.inf)
    zs = [math.exp(x - mx) if x != -math.inf else 0.0 for x in v]
    s = sum(zs)
    probs = [z / s for z in zs]
    return MT19937(seed).discrete(probs)


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

    # T25-2：RNG 对齐对拍 tools/rng-probe.cpp（同编译器 = 同 libstdc++
    # = oracle）。探针不在就走 30 —— 没验的东西不报绿。
    import os
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    probe = os.path.join(root, "tools", "rng-probe.exe")
    if not os.path.exists(probe):
        print("  [PASS] by1sample 规格自检过；[跳过] rng-probe.exe 不在"
              " —— RNG 对齐没验（先 g++ tools/rng-probe.cpp）")
        return 30
    cases = [(42, 12, [0.5, 0.3, 0.2]),
             (7, 8, [0.1, 0.2, 0.3, 0.4]),
             (20261014, 10, [1.0, 1.0, 1.0, 1.0, 1.0])]
    for seed, n, probs in cases:
        p = subprocess.run(
            [probe, str(seed), str(n),
             ",".join(str(x) for x in probs)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace")
        if p.returncode != 0:
            bad.append("rng-probe 退出码 %d" % p.returncode)
            continue
        lines = {}
        for ln in p.stdout.splitlines():
            k, _, rest = ln.partition(":")
            lines[k.strip()] = rest.split()
        raw_exp = [int(x) for x in lines["raw-1"]]
        canon_exp = [float(x) for x in lines["canon"]]
        idx_exp = [int(x) for x in lines["idx"]]
        g = MT19937(seed)
        if [g.genrand() for _ in range(16)] != raw_exp:
            bad.append("mt19937 原始流对不上（seed=%d）" % seed)
        g2 = MT19937(seed)
        canon = [g2.uniform01() for _ in range(8)]
        if any(a != b for a, b in zip(canon, canon_exp)):
            bad.append("uniform01 映射对不上（seed=%d）" % seed)
        g3 = MT19937(seed)
        idx = [g3.discrete(probs) for _ in range(n)]
        if idx != idx_exp:
            bad.append("discrete 对不上（seed=%d）：%r vs %r"
                       % (seed, idx[:6], idx_exp[:6]))
    if bad:
        for b in bad:
            print("  [FAIL] " + b)
        return 1
    print("  [PASS] by1sample：规格手算 + 负例必红 + **RNG 对齐 3 组种子"
          "逐流对上**（mt19937 原始流 / uniform01 / discrete 索引）")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
