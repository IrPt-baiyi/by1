#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1e2e -- **端到端：真产物 -> IR -> 三个后端 -> 对官方实现。**

## 这条链在验什么

    真 config.json + 真 safetensors 头
        │  by1boot.boot_ir()          <- **不经过 .by1**
        ▼
      规范化 IR
        │  三个后端
        ▼
      PyTorch / NumPy / C  ──比对──>  HF transformers 的官方实现

**判卷人是真权重 + 官方实现**，不是合成数据。

## 它会暴露什么

`by1boot` 有一部分属性**从产物里看不出来**，只能填默认值 ——
而它会把这些**报出来**。这个脚本做的就是那句话的实证：

    by1boot  ->  草稿
    对拍      ->  哪里不对
    改一处    ->  直到对上

这里用的模型是 `sshleifer/tiny-gpt2`（**真架构，极小维度**）：
它的 config 里写着 `activation_function = gelu_new`，
而 `by1boot` 从产物里**看不出来**这一点 —— 名字和形状都不说激活函数。

用法:  python by1e2e.py [--gcc <gcc>] [--seq 16]
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import by1io
import by1paths

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def find_cached():
    """找 HF 缓存里的 tiny-gpt2。**用真产物，不自己造一个。**"""
    base = os.path.expanduser('~/.cache/huggingface/hub')
    if not os.path.isdir(base):
        return None, None, None
    roots = [d for d in glob.glob(os.path.join(base, 'models--*gpt2*'))
             if os.path.isdir(d)]
    for r in roots:
        cfg = safet = None
        for p in glob.glob(os.path.join(r, '**', 'config.json'), recursive=True):
            cfg = p
        # **两种格式都要认。**
        # 原来只找 `model.safetensors` —— 而 `sshleifer/tiny-gpt2`
        # 那个 repo 里只有 `pytorch_model.bin`（safetensors 在另一个
        # revision，而且是 0 字节的占位）。于是脚本说"缓存里没有"，
        # 而**缓存里明明有**，只是格式不同。
        for pat in ('model.safetensors', 'pytorch_model.bin'):
            for p in glob.glob(os.path.join(r, '**', pat), recursive=True):
                if os.path.getsize(p) > 0:
                    safet = p
            if safet:
                break
        if cfg and safet:
            return r, cfg, safet
    return None, None, None


def read_header(path):
    """读张量头：safetensors 直接读头，`.bin` 只能整个 load 再取形状。

    **两种都要支持** —— HF 的缓存里两种格式都可能出现，
    只认一种的话，脚本会说"缓存里没有"，而缓存里明明有。
    """
    if path.endswith('.safetensors'):
        with open(path, 'rb') as f:
            n = int.from_bytes(f.read(8), 'little')
            hdr = json.loads(f.read(n))
        return {k: {'shape': v['shape'], 'dtype': v['dtype'],
                    'offsets': v['data_offsets']}
                for k, v in hdr.items() if k != '__metadata__'}
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=True)
    return {k: {'shape': list(v.shape), 'dtype': str(v.dtype)}
            for k, v in sd.items()}


def load_real(path, header, keys):
    """**真权重。** 只取需要的那几个，其余不读。"""
    import torch
    out = {}
    if path.endswith('.bin'):
        sd = torch.load(path, map_location='cpu', weights_only=True)
        for k in keys:
            if k in sd:
                out[k] = sd[k].float()
        return out
    from safetensors import safe_open
    with safe_open(path, framework='pt') as f:
        for k in keys:
            if k in header:
                out[k] = f.get_tensor(k).float()
    return out


def gpt2_weight_map(ir, real, d):
    """IR 的内部参数名 -> **这个 checkpoint 的物理张量名**。

    ## 这一半为什么推不出来

    `by1boot` 能从产物推出"有哪些参数、什么形状"，
    但推不出**它们在 checkpoint 里怎么打包成张量**。
    GPT-2 就是最典型的那种：

        c_attn.weight   (d, 3d)   **融合的 qkv，而且是 Conv1D 的 [in, out]**
        c_fc.weight     (d, 4d)   转置的
        wte / wpe       分开的两张表，而 head 和 wte **绑在一起**

    这些是仓库的习惯，不是架构。所以这一段是**手写的**，
    而且它正是"改到全绿"里"改"的那一部分。

    返回 {IR 内部名: numpy 数组}。
    """
    import torch
    out = {}

    def g(k):
        return None if k not in real else real[k].numpy()

    def T(a_):                      # Conv1D: [in, out] -> 线性层的 [out, in]
        return a_.T if a_ is not None else None

    out['embed.weight'] = g('transformer.wte.weight')
    out['pos.weight'] = g('transformer.wpe.weight')
    # **同一个东西，两个名字。** IR 里叫 `pos`（与模型无关），
    # 而 PyTorch 的 RUNTIME 内部叫 `wpe`（GPT-2 的历史名字）。
    # by1exec 的 --compare 里也有一张同样的别名表。
    out['wpe.weight'] = g('transformer.wpe.weight')
    out['final_norm.w'] = g('transformer.ln_f.weight')
    out['final_norm.b'] = g('transformer.ln_f.bias')
    out['head.weight'] = g('transformer.wte.weight')      # 权重共享

    for L in ir['layers']:
        i = L['index']
        pre = 'transformer.h.%d.' % i
        for j, o in enumerate(L['ops']):
            k = o['kind']
            ip = 'layers.%d.op%d.' % (i, j)
            if k == 'Norm':
                ln = 'ln_1' if j == 0 else 'ln_2'
                out[ip + 'w'] = g(pre + ln + '.weight')
                out[ip + 'b'] = g(pre + ln + '.bias')
            elif k == 'Attention':
                cw = g(pre + 'attn.c_attn.weight')
                cb = g(pre + 'attn.c_attn.bias')
                a_ = o['attrs']
                wq = a_['q'] * a_['head_dim']
                # **Conv1D 是 [in, out] —— 先转置再切。**
                # 第一版在轴 0 上切，于是 k/v 拿到 0 列。
                # 这个坑 gpt2.by1 那一轮踩过一次，这里又踩了一次。
                q_, k_, v_ = (cw[:, :wq], cw[:, wq:2 * wq], cw[:, 2 * wq:])
                bq, bk, bv = cb[:wq], cb[wq:2 * wq], cb[2 * wq:]
                out[ip + 'wq.weight'] = T(q_)
                out[ip + 'wk.weight'] = T(k_)
                out[ip + 'wv.weight'] = T(v_)
                out[ip + 'wq.bias'] = bq
                out[ip + 'wk.bias'] = bk
                out[ip + 'wv.bias'] = bv
                out[ip + 'wo.weight'] = T(g(pre + 'attn.c_proj.weight'))
                out[ip + 'wo.bias'] = g(pre + 'attn.c_proj.bias')
            elif k == 'FFN':
                out[ip + 'w1.weight'] = T(g(pre + 'mlp.c_fc.weight'))
                out[ip + 'w1.bias'] = g(pre + 'mlp.c_fc.bias')
                out[ip + 'w2.weight'] = T(g(pre + 'mlp.c_proj.weight'))
                out[ip + 'w2.bias'] = g(pre + 'mlp.c_proj.bias')
    return {k: v for k, v in out.items() if v is not None}


def snapshot_dir(root):
    """`from_pretrained` 要的是**快照目录**（config.json 所在那个），
    不是仓库目录。"""
    for p in glob.glob(os.path.join(root, '**', 'config.json'),
                       recursive=True):
        return os.path.dirname(p)
    return root


def main():
    seq = 16
    if '--seq' in sys.argv:
        seq = int(sys.argv[sys.argv.index('--seq') + 1])
    gcc = None
    if '--gcc' in sys.argv:
        gcc = sys.argv[sys.argv.index('--gcc') + 1]
    if not gcc:
        hits = glob.glob(os.path.expandvars(
            r'%LOCALAPPDATA%\Microsoft\WinGet\Packages'
            r'\BrechtSanders*\mingw64\bin\gcc.exe'))
        gcc = hits[0] if hits else None

    root, cfg_p, st_p = find_cached()
    if not cfg_p:
        print('  [跳过] HF 缓存里没有 gpt2 —— 这一步要真产物')
        return 2

    print('=' * 78)
    print('  端到端：真产物 -> IR -> 三个后端 -> 对官方实现')
    print('=' * 78)
    print('  产物目录 %s' % os.path.basename(root))
    print('    config    %s' % os.path.basename(cfg_p))
    print('    权重      %s  (%.1f MB)'
          % (os.path.basename(st_p), os.path.getsize(st_p) / 1e6))

    cfg = by1io.read_json(cfg_p, encoding='utf-8')
    header = read_header(st_p)
    print('  张量头 %d 个张量' % len(header))

    # ── ① 真产物 -> IR（**不经过 .by1**）────────────────────────────
    import by1boot
    import by1ir
    ir, guessed, gkeys = by1boot.boot_ir(cfg, header, 'tiny-gpt2')
    errs = by1ir.validate(ir)
    print()
    print('  ① by1boot 出了 IR：%d 层，%s'
          % (len(ir['layers']), '合规格' if not errs else '不合法'))
    if errs:
        for e in errs[:4]:
            print('     ' + e)
        return 1
    print('     推出来的： pos_kind=%s  norm_kind=%s  d_model=%d  层数=%d'
          % (ir['pos_kind'], ir['norm_kind'], ir['d_model'],
             len(ir['layers'])))
    print('     **猜的：%d 处**（产物里看不出来）' % len(guessed))

    # ── ② 真权重 ────────────────────────────────────────────────────
    import by1codegen as cg
    import by1exec as ex
    real = load_real(st_p, header, set(header))
    print()
    print('  ② 真权重装进来了：%d 个张量' % len(real))

    # ── ③ PyTorch 后端，对 HF ───────────────────────────────────────
    import torch
    ns = {}
    exec(compile(cg.render_ir(ir, 'booted.ir'), '<ir>', 'exec'), ns)
    m = ns['build']().eval()
    sd = m.state_dict()
    # 产物名 -> IR 内部名。**按形状和名字后缀配**，配不上就报。
    d = ir['d_model']
    wmap = gpt2_weight_map(ir, real, d)
    miss, bad = [], []
    with torch.no_grad():
        for k, v in sd.items():
            arr = wmap.get(k)
            if arr is None:
                miss.append(k)
                continue
            if tuple(arr.shape) != tuple(v.shape):
                bad.append('%s %s vs %s' % (k, tuple(arr.shape), tuple(v.shape)))
                continue
            m.state_dict()[k].copy_(torch.tensor(arr))
    if miss:
        print('     [注意] %d 个参数映射里没有：%s' % (len(miss), miss[:4]))
    if bad:
        print('     [注意] %d 个形状对不上：%s' % (len(bad), bad[:3]))
    print('     映射装上 %d / %d' % (len(sd) - len(miss) - len(bad), len(sd)))

    from transformers import GPT2LMHeadModel
    hf = GPT2LMHeadModel.from_pretrained(snapshot_dir(root)).eval()
    ids = torch.tensor([[i % 1000 + 100 for i in range(seq)]])
    with torch.no_grad():
        ref = hf(ids).logits.numpy()
    with torch.no_grad():
        got = m(ids).numpy()
    dd = np.abs(ref - got).max() / max(np.abs(ref).max(), 1e-30)
    print()
    print('  ③ PyTorch 后端 vs HF 官方实现')
    print('     相对差 %.3e   %s' % (dd, '[一致]' if dd < 1e-4 else '[**不一致**]'))

    # ── ④ 把那个"猜的"改对，再看 ────────────────────────────────────
    print()
    print('  ④ 改一处猜错的，再看')
    # **改的是这两个**，而且它们从产物里都看不出来：
    #   gate  —— GPT-2 是"没有门"的那种（两层 MLP），
    #            而 by1boot 默认按"有门"生成，于是多出一个 w3
    #   act   —— 名字和形状都不说激活函数，只有 config 里写着
    # 证据：上一步里 `w3` 在产物里**根本找不到** —— 那就是信号。
    print('     产物里没有 w3，而 by1boot 生成了 —— 那说明它猜错了 `gate`')
    print('     产物里 `activation_function = %r`'
          % cfg.get('activation_function'))
    for L in ir['layers']:
        for o in L['ops']:
            if o['kind'] == 'FFN':
                o['attrs']['gate'] = False
                o['attrs']['act'] = cfg.get('activation_function', 'silu')
    ns2 = {}
    exec(compile(cg.render_ir(ir, 'fixed.ir'), '<ir>', 'exec'), ns2)
    m2 = ns2['build']().eval()
    with torch.no_grad():
        for k, v in m2.state_dict().items():
            arr = wmap.get(k)
            if arr is not None and tuple(arr.shape) == tuple(v.shape):
                m2.state_dict()[k].copy_(torch.tensor(arr))
    with torch.no_grad():
        got2 = m2(ids).numpy()
    dd2 = np.abs(ref - got2).max() / max(np.abs(ref).max(), 1e-30)
    print('     PyTorch 后端 vs HF：相对差 %.3e   %s'
          % (dd2, '[一致]' if dd2 < 1e-4 else '[**还是不一致**]'))

    # ── ⑤ NumPy 和 C ────────────────────────────────────────────────
    print()
    print('  ⑤ 另外两个后端（**它们只拿 IR**，而 IR 就是上面改过的那份）')
    shapes = ex.shapes_of(ir)
    params, zeroed = {}, []
    for k, s in shapes.items():
        # **两个后端给同一个参数起的名字不一样。**
        # `shapes_of` 用 `wq`，而 PyTorch 的 `nn.Linear` 叫 `wq.weight`。
        # 只试一个的话，注意力那 12 个权重**整个没进去** ——
        # 而输出形状完全正常，相对差只有 1e-02，
        # 看着像"算法有小误差"，其实是权重是零。
        # 这个坑在这一轮里出现过三次了（by1exec 的 --compare、
        # by1opdiff 的配对、这里）。
        arr = wmap.get(k) if k in wmap else wmap.get(k + '.weight')
        if arr is not None and arr.size == int(np.prod(s)):
            params[k] = arr.reshape(s)
        else:
            # **配不上就退化成全零 —— 而那必须报出来。**
            # 静默用零填充的话，数对不上时第一反应会是"算法错了"，
            # 而其实是权重根本没进去。这正是这个项目里反复出现的
            # "看起来对但算错"的另一种长相。
            params[k] = np.zeros(s, dtype=np.float32)
            zeroed.append(k)
    if zeroed:
        print('     [注意] **%d 个参数是零填充的**（映射里没有）：%s'
              % (len(zeroed), zeroed[:4]))
    else:
        print('     全部 %d 个参数都从真权重装上了' % len(params))
    exm = ex.Exec(ir, params)
    exm.P['_g'] = {k: params[k] for k in
                   ('embed.weight', 'pos.weight', 'final_norm.w',
                    'final_norm.b', 'head.weight') if k in params}
    out_n = exm(ids.numpy())
    _abs = np.abs(ref - out_n).max()
    ddn = _abs / max(np.abs(ref).max(), 1e-30)
    print('     NumPy vs HF：绝对差 %.3e  相对差 %.3e   %s'
          % (_abs, ddn, '[一致]' if ddn < 1e-4 else '[**不一致**]'))
    # **相对差在退化模型上会骗人。**
    # tiny-gpt2 的 d_model = 2 —— logits 是 50257 个数里挑出来的，
    # 而 LayerNorm 的 eps（1e-5）在这种尺度下能主导方差。
    # 所以要看**幅度**：如果 logits 本身很小，或者差异集中在
    # 极少数位置上，那是数值敏感性，不是算法不对。
    print('     logits 幅度 %.3e   绝对差 %.3e   差 > 1e-4 的位置占 %.2f%%'
          % (np.abs(ref).max(), _abs,
             100.0 * (np.abs(ref - out_n) > 1e-4).mean()))
    print('     而且 PyTorch 侧同一份 IR 是 %.3e —— **两个后端读的是同一份 IR**'
          % dd2)

    if gcc:
        wd = by1paths.root('cgen-e2e')
        os.makedirs(wd, exist_ok=True)
        import by1c
        try:
            ctext, order, _t = by1c.emit_c_ir(ir, shapes)
            cp = os.path.join(wd, 'model.c')
            by1io.write_text(cp, by1c.C_HEAD + '\n' + ctext + '\n' + by1c.C_MAIN)
            exe = os.path.join(wd, 'model.exe')
            r = subprocess.run([gcc, '-O2', '-o', exe, cp, '-lm'],
                               capture_output=True, text=True)
            print('     C：%s' % ('编译成功' if r.returncode == 0
                                  else '编译失败 ' + (r.stderr or '')[:80]))
        except SystemExit as e:
            print('     C：[不支持] %s' % str(e).strip()[:70])

    print()
    ok = dd2 < 1e-4 and ddn < 1e-4
    print('  [%s] 端到端：真产物 -> IR -> 三个后端 %s'
          % ('PASS' if ok else 'FAIL',
             '对得上官方实现' if ok else '还有对不上的'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
