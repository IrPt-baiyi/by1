#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1gpt2 -- GPT-2 前向的差分验证，**用真的 GPT-2**。

为什么是它：这是语言里"另一代"的第一个模型。
前面十三个全是 Llama 家族（或从它长出来的），所以语言里长出来的东西
也全是那一代的：RMSNorm、RoPE、门控 + SwiGLU。GPT-2 三样都不是。

判卷人是 **HF 的 GPT2LMHeadModel + 官方权重**（124M，这台机器跑得动），
维度全真。不是我又缩小维度的合成模型。

要盯的三处"同名不同物"：
    RMSNorm vs **LayerNorm**    后者要减均值、还有 bias
    RoPE    vs **学习式 wpe**    一个是旋转、在注意力里；一个是查表、加在输入上
    SwiGLU  vs **两层 + GELU**   没有门

用法:  python by1gpt2.py
"""
import os
import sys

import torch
# **副作用 import**：拉进 `by1io`，它把 stdout 钉成 UTF-8（见 by1gate 里同一句）。
import by1paths  # noqa: F401
import by1skip

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SEQ = 16


def main():
    print('=' * 78)
    print('  GPT-2 前向   by1 生成  vs  HF GPT2LMHeadModel（官方权重）')
    print('=' * 78)

    try:
        from transformers import GPT2LMHeadModel, GPT2Config
    except Exception as e:
        return by1skip.skip('没有 transformers: %s' % e)

    # 用官方 config —— 维度全真
    cfg = GPT2Config()
    cfg._attn_implementation = 'eager'
    torch.manual_seed(0)
    ref = GPT2LMHeadModel(cfg).eval()
    rp = dict(ref.named_parameters())
    print('\n  判卷人: %d 组参数，d=%d 层=%d 头=%d vocab=%d'
          % (len(rp), cfg.n_embd, cfg.n_layer, cfg.n_head, cfg.vocab_size))

    import importlib.util
    sp = importlib.util.spec_from_file_location('bcm', os.path.join(HERE, 'by1check.py'))
    bc = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(bc)
    import by1codegen as cg
    _r, info = bc.check('gpt2.by1')
    ns = {}
    exec(compile(cg.render(info, 'gpt2.by1'), '<by1-generated>', 'exec'), ns)
    mine = ns['build']().eval()

    # ── 权重搬运 ────────────────────────────────────────────────────
    # 三处**名字和布局都对不上**，全是照实处理的，没有假装它们一样：
    #
    #   ① 实现的内部名（wq / wk / wv / wo / w1 / w2）和契约的逻辑名
    #      （c_attn / c_proj / c_fc）不是一回事 —— 契约是给产物看的，
    #      内部名是给计算看的。
    #   ② GPT-2 存的是**融合的 c_attn**（一块 [768, 2304]），而实现用
    #      三个分开的矩阵。融合是 checkpoint 的事，拆开是映射的事。
    #   ③ GPT-2 的线性层是 **Conv1D 布局 [in, out]，转置的**。
    D = cfg.n_embd

    def src_of(k):
        """返回 (物理名, 切片 or None, 是否转置)。"""
        if k == 'embed.weight':
            return 'transformer.wte.weight', None, False
        if k == 'wpe.weight':
            return 'transformer.wpe.weight', None, False
        if k == 'final_norm.w':
            return 'transformer.ln_f.weight', None, False
        if k == 'final_norm.b':
            return 'transformer.ln_f.bias', None, False
        if k == 'head.weight':
            return 'transformer.wte.weight', None, False    # 权重共享

        import re
        m = re.match(r'layers\.(\d+)\.op(\d+)\.(.+)$', k)
        if not m:
            return None, None, False
        i, j, pn = int(m.group(1)), int(m.group(2)), m.group(3)
        kinds = [o['kind'] for o in cg.compile_ir(info)['layers'][i]['ops']]
        kd = kinds[j]
        if kd == 'Norm':
            # op0 = ln_1，op3 = ln_2（中间夹着 Attention 和 Add）
            base = 'transformer.h.%d.%s' % (i, 'ln_1' if j == 0 else 'ln_2')
            return base + ('.weight' if pn == 'w' else '.bias'), None, False
        if kd == 'Attention':
            if pn.startswith('wq.'):
                return ('transformer.h.%d.attn.c_attn.%s' % (i, pn[3:]),
                        slice(0, D), True)
            if pn.startswith('wk.'):
                return ('transformer.h.%d.attn.c_attn.%s' % (i, pn[3:]),
                        slice(D, 2 * D), True)
            if pn.startswith('wv.'):
                return ('transformer.h.%d.attn.c_attn.%s' % (i, pn[3:]),
                        slice(2 * D, 3 * D), True)
            if pn.startswith('wo.'):
                return ('transformer.h.%d.attn.c_proj.%s' % (i, pn[3:]), None, True)
            return None, None, False
        if kd == 'FFN':
            # gate = none，所以只有 w1 / w2（c_fc / c_proj）
            if pn.startswith('w1.'):
                return ('transformer.h.%d.mlp.c_fc.%s' % (i, pn[3:]), None, True)
            if pn.startswith('w2.'):
                return ('transformer.h.%d.mlp.c_proj.%s' % (i, pn[3:]), None, True)
            return None, None, False
        return None, None, False

    msd = dict(mine.named_parameters())
    n_ok, miss = 0, []
    with torch.no_grad():
        for k, v in msd.items():
            src, sl, tr = src_of(k)
            if src is None:
                miss.append('%s（没有映射规则）' % k)
                continue
            if src not in rp:
                miss.append('%s -> %s（参考里没有）' % (k, src))
                continue
            t = rp[src]
            # **顺序要紧：先转置、再切。**
            # c_attn.weight 是 Conv1D 的 [in, out] = [768, 2304]；
            # 先切第 0 轴会把 [2304, 768] 切成 [768, 768]，长度对不上。
            # （第一版就是这么错的：q 侥幸切出 (2304,768)，k/v 切出 0 列。）
            if tr and t.dim() == 2:
                t = t.t()
            if sl is not None:
                t = t[sl]
            if t.shape != v.shape:
                miss.append('%s -> %s 形状 %s vs %s'
                            % (k, src, tuple(v.shape), tuple(t.shape)))
                continue
            v.copy_(t)
            n_ok += 1

    print('  权重搬运: %d / %d' % (n_ok, len(msd)))
    for m in miss[:5]:
        print('     ' + m)
    if n_ok != len(msd):
        print('\n  [FAIL] 权重没搬全，比下去没有意义')
        return 1

    ids = torch.randint(0, cfg.vocab_size, (1, SEQ))
    # **必须显式给因果 mask。**
    # HF 的 eager 实现在 attention_mask=None 时**不做因果** ——
    # 它不再自带那个 [1,1,ctx,ctx] 的 buffer（transformers 5.x 删了）。
    # 这次是第二次撞上：MLA 那次也是同一个坑，当时差了 1.134。
    # 不传 mask 的话，比的是"我的因果注意力"和"HF 的全注意力"。
    cm = torch.tril(torch.ones(SEQ, SEQ)).view(1, 1, SEQ, SEQ)
    cm = (1.0 - cm) * torch.finfo(torch.float32).min
    with torch.no_grad():
        r = ref(ids, attention_mask=cm).logits.float()
        m = mine(ids)
        m = m.float() if not isinstance(m, tuple) else m[0].float()
    dd = (r - m).abs().max().item()
    amp = max(r.abs().max().item(), 1e-9)
    print('\n  最大绝对差 %.3e   相对 %.3e' % (dd, dd / amp))
    ok = dd / amp < 1e-4
    print('\n  [%s] GPT-2 的前向%s'
          % ('PASS' if ok else 'FAIL', '与官方实现一致' if ok else '对不上'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
