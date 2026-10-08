#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1instella -- Instella-3B 前向的差分验证，**用真实维度**。

为什么单独做这一步：截至这篇文章，"六个真实模型全中"的准确版本是
**"config 与命名全中，数值行为只在合成维度上验过"**。
Instella-3B 是六个里唯一在这台机器上也跑得动的（稠密、3B），
所以它是第一个能把后半句去掉的。

**判卷人是官方仓库里的 `modeling_instella.py`** —— AMD 自己写的。

维度用**真实的**（d=2560 / 32 头 / head_dim=80 / FFN 6912），
只把层数降到 4（36 层 = 11GB fp32，这台机器装不下）。

顺带钉住一个"同名不同义"：
  Instella 的 QK-norm 是**整宽**的 `[2560]`，作用在拆头**之前**；
  Qwen3-Next 的也是 QK-norm，但按 `head_dim`、作用在拆头**之后**。
  名字一样，位置和宽度都不同。

用法:  python by1instella.py
"""
import importlib.util
import io
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'llamacpp'))

D, Q, KV, HD, FF = 2560, 32, 32, 80, 6912
LAYERS = 4          # 真实是 36；这里只降到层数，其余全真
SEQ = 16
VOCAB = 50304
EPS = 1e-5


def load_reference():
    """把官方 modeling_instella.py 当作判卷人拉起来。

    判卷人是**模型的官方代码**，不在 by1 的包里（`llamacpp/` 是 .gitignore 的，
    它是可重新下载的参照源码）。所以这里**缺了就自己下** ——
    否则在云端一跑就挂，白花租金。
    """
    path = os.path.join(HERE, 'llamacpp', 'modeling_instella.py')
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = ('https://hf-mirror.com/amd/Instella-3B/raw/main/'
               'modeling_instella.py')
        print('  本地没有判卷人，去下: %s' % url)
        try:
            import urllib.request
            # hf-mirror 会拒没有 User-Agent 的请求（403）
            req = urllib.request.Request(
                url, headers={'User-Agent': 'Mozilla/5.0 (by1)'})
            with urllib.request.urlopen(req, timeout=60) as r, \
                    open(path, 'wb') as f:
                f.write(r.read())
            print('  下好了（%d KB）' % (os.path.getsize(path) // 1024))
        except Exception as e:
            return None, '下载失败 %s: %s' % (type(e).__name__, str(e)[:160])
    try:
        spec = importlib.util.spec_from_file_location('mi', path)
        m = importlib.util.module_from_spec(spec)
        sys.modules['mi'] = m
        spec.loader.exec_module(m)
        return m, None
    except Exception as e:
        return None, '%s: %s' % (type(e).__name__, str(e)[:200])


def main():
    print('=' * 78)
    print('  Instella-3B 前向   by1 生成  vs  官方 modeling_instella.py')
    print('=' * 78)
    print('\n  维度: d=%d  q=%d  kv=%d  head_dim=%d  FFN=%d  层数=%d（真实 36）  seq=%d'
          % (D, Q, KV, HD, FF, LAYERS, SEQ))

    mi, err = load_reference()
    if mi is None:
        print('\n  [跳过] 拿不到判卷人: %s' % err)
        return 2

    cfg = mi.InstellaConfig(
        vocab_size=VOCAB, hidden_size=D, intermediate_size=FF,
        num_hidden_layers=LAYERS, num_attention_heads=Q,
        num_key_value_heads=KV, max_position_embeddings=4096,
        rms_norm_eps=EPS, hidden_act='silu', attention_bias=False,
        tie_word_embeddings=False, torch_dtype='float32')
    cfg._attn_implementation = 'eager'
    torch.manual_seed(0)
    ref = mi.InstellaForCausalLM(cfg).eval()

    print('  判卷人的张量（应当和 by1 的契约逐个同形）:')
    rp = dict(ref.named_parameters())
    for k in sorted(rp)[:6]:
        print('    %-52s %s' % (k, tuple(rp[k].shape)))
    print('    …共 %d 个' % len(rp))

    import by1check as bc
    import by1codegen as cg
    # **只把层数降下来，其余维度全真。**
    # 36 层 = 12GB fp32，加上参考模型这台机器装不下。
    # 做法是临时生成一份 LAYERS 层的 .by1 —— 不动工具，也不手改 IR。
    src = io.open(os.path.join(HERE, 'instella-3b.by1'), encoding='utf-8').read()
    src = src.replace('n_layer = 36', 'n_layer = %d' % LAYERS)
    src = re.sub(r'pattern\s*=\s*36\s*\*', 'pattern = %d *' % LAYERS, src)
    tmp = os.path.join(HERE, '_instella_shaped.by1')
    io.open(tmp, 'w', encoding='utf-8').write(src)
    _r, info = bc.check(tmp)
    ir = cg.compile_ir(info)
    print('  实际生成 %d 层' % len(ir['layers']))
    # by1diff 用的路子：渲染成源码、exec、再 build()
    try:
        ns = {}
        exec(compile(cg.render(dict(info, layers=None) if False else info,
                               'instella-3b.by1'),
                     '<by1-generated>', 'exec'), ns)
        mine = ns['build']()
    except Exception as e:
        import traceback
        print('  生成失败:', type(e).__name__, str(e)[:200])
        traceback.print_exc()
        return 1
    mine = mine.eval()

    # ── 权重搬运 ────────────────────────────────────────────────
    # by1 生成的模型用的是**内部名**（layers.0.op1.wq.weight），
    # 而参考用的是**物理名**（model.layers.0.self_attn.q_proj.weight）。
    # 桥是张量契约里的**逻辑名** —— 而且物理名由 .by1 的 emit 规则生成，
    # 不在这里硬编码 HF 的命名习惯。
    rule = info['emit']['torch.module']

    def phys(i, mech, logical):
        return bc.render_name(rule, i, 'main', mech, logical)

    # by1 内部参数名 -> (机制, 逻辑名)。按 op 序号区分，因为两个 Norm 的
    # 参数名都叫 w，只有位置不同。
    def pair(byname):
        m = re.match(r'^layers\.(\d+)\.op(\d+)\.(.+)$', byname)
        if not m:
            g = {'embed.weight': 'embed.weight',
                 'final_norm.w': 'final_norm.weight',
                 'head.weight': 'lm_head.weight'}
            return (None, 'global', g[byname]) if byname in g else None
        i, j, pn = int(m.group(1)), int(m.group(2)), m.group(3)
        kinds = [o['kind'] for o in ir['layers'][i]['ops']]
        if kinds[j] == 'Norm':
            logical = ('pre_attention_layernorm.weight' if j == 0
                       else 'pre_feedforward_layernorm.weight')
            return (i, 'layer', logical)
        if kinds[j] == 'Attention':
            tbl = {'wq.weight': 'q_proj.weight', 'wk.weight': 'k_proj.weight',
                   'wv.weight': 'v_proj.weight', 'wo.weight': 'o_proj.weight',
                   'qn.w': 'q_norm.weight', 'kn.w': 'k_norm.weight'}
            return (i, 'Attn', tbl[pn]) if pn in tbl else None
        if kinds[j] == 'FFN':
            tbl = {'w1.weight': 'gate_proj.weight',
                   'w3.weight': 'up_proj.weight',
                   'w2.weight': 'down_proj.weight'}
            return (i, 'FFN_', tbl[pn]) if pn in tbl else None
        return None

    msd = dict(mine.named_parameters())
    rsd = dict(ref.named_parameters())
    n_ok, miss = 0, []
    with torch.no_grad():
        for k, v in msd.items():
            pr = pair(k)
            if pr is None:
                miss.append(k)
                continue
            i, mech, logical = pr
            if i is None:
                # 全局张量走 **global_name** 模板（"{physical}"），
                # 用 name 模板会多出一个 model.layers.0. 前缀。
                pname = bc.render_name(dict(rule, name=rule['global_name']),
                                       0, '<g>', 'global', logical)
            else:
                pname = phys(i, mech, logical)
            if pname not in rsd:
                miss.append('%s -> %s（参考里没有）' % (k, pname))
                continue
            if rsd[pname].shape != v.shape:
                miss.append('%s -> %s 形状 %s vs %s'
                            % (k, pname, tuple(v.shape), tuple(rsd[pname].shape)))
                continue
            v.copy_(rsd[pname])
            n_ok += 1
    print('\n  权重搬运: %d / %d 成功' % (n_ok, len(msd)))
    if miss:
        print('  没搬成的前几个:')
        for m in miss[:5]:
            print('    ' + m)
    if n_ok != len(msd):
        print('  [FAIL] 权重没搬全，比下去没有意义')
        return 1

    ids = torch.randint(0, VOCAB, (1, SEQ))
    with torch.no_grad():
        # 判卷人是按 transformers 4.48 写的，而这台是 5.15：
        # DynamicCache 的 API 变了，走缓存那条路会 AttributeError。
        # 传 use_cache=False 绕开 —— **这是判卷人的版本问题，不是被测代码的**。
        r = ref(ids, use_cache=False).logits.float()
        m = mine(ids)
        m = m.float() if not isinstance(m, tuple) else m[0].float()
    dd = (r - m).abs().max().item()
    amp = max(r.abs().max().item(), 1e-9)
    print('\n  最大绝对差 %.3e   相对 %.3e' % (dd, dd / amp))
    ok = dd / amp < 1e-4
    print('\n  [%s] Instella-3B 的前向%s'
          % ('PASS' if ok else 'FAIL',
             '与官方实现一致' if ok else '对不上'))
    try:
        os.remove(tmp)          # 临时的那份 N 层 .by1
    except OSError:
        pass
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
