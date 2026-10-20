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


def chain_f32(logits, seed, temp=0.8, top_k=40, top_p=0.95, min_p=0.05):
    """T25-3：llama.cpp 链级逐 token 复刻 —— **float32 浮点序照抄
    各 apply() 真身**（llama-sampler.cpp 读出来的，不是猜的）：

      top_k  ：降序排 -> 截 k（等值序 = std::sort 实现定义，随机 logits
               无并列；并列场景单列）
      top_p  ：softmax_impl(false)：max 全扫、p=expf(x-max) f32、cum_sum
               **f32 按数组序**累加、p/=cum_sum f32；再降序排、f32 累计
               截到 cum>=p 且个数>=min_keep
      min_p  ：两条路径 —— 未排序：max 全扫、min_logit=f32(max+logf(p))、
               **按当前存储序**保 x>=min_logit（不排序！）；已排序或上路
               不适用：min_logit=f32(x[0]+logf(p))、保第一个、首个
               x[i]<min_logit 处截断（min_keep=1）
      temp   ：x = f32(x/temp)
      dist   ：max_l=data[0]（未排序则 max 全扫）；p=expf(x-max_l) f32；
               sum_cum **double** 累加；
               rnd=uniform01；sum_tgt=sum_cum*rnd；sum_run **double** 累加，
               `sum_run >= sum_tgt` 处选中；兜底最后一个

    判卷人 tools/sampler-probe.cpp（同编译器 oracle）。"""
    import numpy as np
    x = np.asarray(logits, dtype=np.float32).copy()
    ids = np.arange(len(x))
    f32 = np.float32

    sorted_flag = False   # llama cur_p->sorted 的忠实复刻

    if top_k > 0:
        k = min(int(top_k), len(x))
        order = np.argsort(-x, kind="stable")[:k]
        ids, x = ids[order], x[order]
        sorted_flag = True   # partial_sort_inplace 连 k < size 也置 sorted
    # top_k=0 = 空采样器（no-op），top_p=1.0 直接 return ——
    # 数组**保持原始存储序**；dist 的 fp 求和序敏感，不许擅自排序（B1）

    if top_p < 1.0:
        max_l = f32(x[0]) if sorted_flag else f32(np.max(x))
        p = np.exp(x - max_l).astype(f32)
        cum = f32(0)
        for i in range(len(p)):
            cum = f32(cum + p[i])
        p = (p / cum).astype(f32)
        if not sorted_flag:
            order = np.argsort(-x, kind="stable")
            x, p, ids = x[order], p[order], ids[order]
        sorted_flag = True
        c, keep = f32(0), len(x)
        for i in range(len(x)):
            c = f32(c + p[i])
            if c >= f32(top_p):
                keep = i + 1
                break
        x, ids = x[:keep], ids[:keep]

    if min_p > 0 and len(x):
        # min_p 两条路径（llama_sampler_min_p_apply）：
        # 未排序 -> max 全扫 + min_logit = max+logf(p)，按**当前存储序**保
        # x >= min_logit（不排序！dist 求和序敏感）；已排序/未排序路径
        # 不适用 -> 排序路径：min_logit = x[0]+logf(p)，保第一个，首个
        # x[i] < min_logit 处截断。min_keep = 1。
        applied = False
        if not sorted_flag:
            max_logit = f32(np.max(x))
            min_logit = f32(max_logit + f32(np.log(f32(min_p))))
            mask = x >= min_logit
            if int(np.count_nonzero(mask)) >= 1:
                x, ids = x[mask], ids[mask]
                applied = True
        if not applied:
            if not sorted_flag:
                order = np.argsort(-x, kind="stable")
                ids, x = ids[order], x[order]
                sorted_flag = True
            min_logit = f32(x[0] + f32(np.log(f32(min_p))))
            keep = len(x)
            for i in range(1, len(x)):
                if x[i] < min_logit:
                    keep = i
                    break
            x, ids = x[:keep], ids[:keep]

    x = (x / f32(temp)).astype(f32)

    max_l = f32(x[0]) if sorted_flag else f32(np.max(x))
    ps = np.exp(x - max_l).astype(f32)
    sum_cum = 0.0
    for i in range(len(ps)):
        sum_cum += float(ps[i])
    rnd = MT19937(seed).uniform01()
    tgt = sum_cum * rnd
    run, sel = 0.0, len(ps) - 1
    for i in range(len(ps)):
        run += float(ps[i])
        if run >= tgt:
            sel = i
            break
    return int(ids[sel])


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

    # T25-3：链级逐 token 对拍 tools/sampler-probe.cpp（多配置 x 8 种子）
    import random as _random
    probe2 = os.path.join(root, "tools", "sampler-probe.exe")
    if not os.path.exists(probe2):
        print("  [PASS] by1sample：规格 + RNG 三流对上；[跳过] "
              "sampler-probe.exe 不在 —— 链级逐 token 没验")
        return 30
    configs = [(0.8, 40, 0.95, 0.05), (1.2, 0, 1.0, 0.0), (0.5, 5, 0.5, 0.3)]
    lfile = os.path.join(root, "cgen", "by1sample-chain.txt")
    os.makedirs(os.path.dirname(lfile), exist_ok=True)
    for temp, tk, tp, mp in configs:
        _random.seed(20261014)
        logits = [_random.uniform(-30, 30) for _ in range(1024)]
        with open(lfile, "w", encoding="utf-8") as fh:
            fh.write("".join("%.9g\n" % v for v in logits))
        p2 = subprocess.run(
            [probe2, lfile, "0", "8", str(temp), str(tk), str(tp), str(mp)],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p2.returncode != 0:
            bad.append("sampler-probe 退出码 %d（cfg=%r）"
                       % (p2.returncode, (temp, tk, tp, mp)))
            continue
        exp = [int(ln.split()[1]) for ln in p2.stdout.splitlines() if ln.strip()]
        got = [chain_f32(logits, s, temp, tk, tp, mp) for s in range(8)]
        if got != exp:
            bad.append("链级对不上 cfg=%r：%r vs %r"
                       % ((temp, tk, tp, mp), got[:4], exp[:4]))
    if bad:
        for b in bad:
            print("  [FAIL] " + b)
        return 1
    # T26：sampler 对拍矩阵（参数组合全网格）—— 失败项单列
    mat_bad, cells = [], 0
    lfile2 = os.path.join(root, "cgen", "by1sample-matrix.txt")
    for temp in (0.5, 0.8, 1.0, 1.3):
        for tk in (0, 3, 40, 400):
            for tp in (0.5, 0.9, 1.0):
                for mp in (0.0, 0.05, 0.3):
                    cells += 1
                    _random.seed(20261014 + cells)
                    logits = [_random.uniform(-30, 30) for _ in range(1024)]
                    with open(lfile2, "w", encoding="utf-8") as fh:
                        fh.write("".join("%.9g\n" % v for v in logits))
                    p3 = subprocess.run(
                        [probe2, lfile2, "0", "8", str(temp), str(tk),
                         str(tp), str(mp)],
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace")
                    if p3.returncode != 0:
                        mat_bad.append(("rc=%d" % p3.returncode,
                                        (temp, tk, tp, mp)))
                        continue
                    exp = [int(ln.split()[1])
                           for ln in p3.stdout.splitlines() if ln.strip()]
                    got = [chain_f32(logits, s, temp, tk, tp, mp)
                           for s in range(8)]
                    if got != exp:
                        mat_bad.append(("mismatch",
                                        (temp, tk, tp, mp), got[:4], exp[:4]))
    if mat_bad:
        for b in mat_bad:
            print("  [FAIL-矩阵] %r" % (b,))
        print("  [FAIL] T26 矩阵：%d/%d 格过，失败 %d 项（单列如上）"
              % (cells - len(mat_bad), cells, len(mat_bad)))
        return 1
    print("  [PASS] T26 矩阵：%d/%d 格逐 token 全对（temp x top_k x top_p "
          "x min_p 全网格，每格 8 token）" % (cells, cells))

    print("  [PASS] by1sample：规格手算 + 负例必红 + RNG 三流对上 + "
          "**链级逐 token 3 配置 x 8 种子全对**（top_k/top_p/min_p/temp/"
          "dist 的 float32 浮点序复刻）")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
