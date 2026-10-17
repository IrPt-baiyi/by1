#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1attrs -- **机制属性的闭集 与 官方出处**：闭集里不许躺着一个谁都说不清来历的取值。

## 为什么需要它

`by1codegen.ENUMS` 把 `act` / `qk_norm` / `routing` / `gate_act` / `pairing` 的合法取值钉死。
这是好事（取值不认识就拒绝，不许静默走默认分支）。但它只保证"取值在集合里"，**不保证
集合里每个取值都真的属于某个官方模型** —— 一个取值可能是当初为某个模型写的，那个模型
后来删了，取值却留在闭集里，从此没人说得清它对不对。

## 三条规则

    A1 每个取值都要有出处   ENUMS 里的每个 (属性, 取值) 都得在 PROVENANCE 里；
    A2 没有陈旧条目       PROVENANCE 里的键必须在 ENUMS 里（取值被删了，记录也要跟着删）；
    A3 出处要能对上模型   出处里的家族名要能在 `models/` 里有对应的 `.by1`
                     （或者是"旧写法"这类明确的别名）—— 否则出处只是一句话。

另外印一份 **清单**：`models/` 里**没有任何模型用到**的取值。它不算失败（有的取值
是为了接下来的模型准备的），但要让人看得见。
"""

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: (属性, 取值) -> 出处。**每条都要能对一个官方实现。**
PROVENANCE = {
    ('act', 'silu'): 'Llama/Qwen 的 SwiGLU（transformers ACT2FN silu）',
    ('act', 'gptoss'): 'GPT-OSS 的 clamped swiglu（limit 7.0 + alpha 1.702）',
    ('act', 'gelu'): 'GPT-2 的精确 GELU（ACT2FN gelu）',
    ('act', 'gelu_new'): 'GPT-2 的 tanh 近似（transformers 叫 gelu_new / gelu_pytorch_tanh）',
    ('act', 'relu2'): 'Nemotron 的 squared-relu（ACT2FN relu2）',
    ('qk_norm', 'off'): '无 qk_norm（GPT-2 / 多数老模型）',
    ('qk_norm', 'per_head'): 'Qwen3 的 per-head RMSNorm（transformers Qwen3RMSNorm）',
    ('qk_norm', 'full'): 'Instella 的整宽 RMSNorm（拆头之前）',
    ('qk_norm', 'true'): '旧写法，等价 per_head（by1langspec 记档）',
    ('qk_norm', 'on'): '旧写法，等价 per_head（by1langspec 记档）',
    ('qk_norm', 'yes'): '旧写法，等价 per_head（by1langspec 记档）',
    ('qk_norm', '1'): '旧写法，等价 per_head（by1langspec 记档）',
    ('routing', 'softmax_topk'): 'Mixtral 的 top-k 后重归一（Switch/Mixtral 那一类）',
    ('routing', 'topk_softmax'): 'GPT-OSS 的 top-k 后再 softmax（GptOssTopKRouter）',
    ('routing', 'sigmoid_topk'): 'Step-3.7 的 sigmoid 打分直接 top-k',
    ('routing', 'sigmoid_group_topk'): 'DeepSeek V3/V4 与 GLM 与 Ling 的 noaux_tc（先分组再 top-k）',
    ('gate_act', 'softplus'): 'Laguna 的 head gate（LagunaAttention 用 F.softplus）',
    ('gate_act', 'sigmoid'): 'Ling 的 head gate（同名不同义）',
    ('pairing', 'half'): 'Llama 等 HF 模型的 apply_rotary_pos_emb（rotate_half）',
    ('pairing', 'interleaved'): 'DeepSeek 与 GLM 的 interleaved RoPE（索引器与 MLA 用）',
}


def load_enums(codegen_path=None):
    """从 `by1codegen.py` **静态**取 ENUMS（不 import：那会把 torch 拖进来）。"""
    p = pathlib.Path(codegen_path or (ROOT / 'src/by1codegen.py'))
    tree = ast.parse(p.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == 'ENUMS':
            return {k: set(v) for k, v in ast.literal_eval(node.value).items()}
    return {}


def model_families(root=ROOT):
    """`models/` 与 `refs/` 里出现过的家族名（小写、拼接，含首段前缀）。

    出处可以指一份 `models/*.by1`，也可以指一个 refs 语料（真模型的 config）。
    """
    out = set()
    stems = []
    for f in (pathlib.Path(root) / 'models').rglob('*.by1'):
        stems.append(f.stem.lower())
    refs = pathlib.Path(root) / 'refs'
    if refs.is_dir():
        for f in refs.iterdir():
            if f.is_file():
                stems.append(f.name.split('.')[0].lower())
    for stem in stems:
        flat = re.sub(r'[^a-z0-9]', '', stem)
        out.add(flat)
        for part in re.split(r'[^a-z]+', stem):
            if len(part) >= 4:
                out.add(part)
        m = re.match(r'[a-z]+', stem)
        if m:
            out.add(m.group(0))
    return out

def used_values(root=ROOT):
    """`models/` 里真正出现过的 (属性, 取值)（正则扫文本）。"""
    seen = set()
    for f in (pathlib.Path(root) / 'models').rglob('*.by1'):
        text = f.read_text(encoding='utf-8', errors='replace')
        for key in ('act', 'qk_norm', 'routing', 'gate_act', 'pairing'):
            for m in re.finditer(r'\b' + key + r'\s*=\s*([A-Za-z_0-9]+)', text):
                seen.add((key, m.group(1).lower()))
    return seen


def check(root=ROOT, enums=None, provenance=None):
    root = pathlib.Path(root)
    enums = load_enums(root / 'src/by1codegen.py') if enums is None else enums
    prov = PROVENANCE if provenance is None else provenance
    fams = model_families(root)
    problems = []
    notes = []
    if not enums:
        return (['ENUMS 取不出来（by1codegen.py 改名了？）—— 这条检查会永远绿'], [])
    known = set()
    for attr, vals in enums.items():
        for v in vals:
            known.add((attr, str(v).lower()))
    for attr, v in sorted(known):
        if (attr, v) not in prov:
            problems.append('A1 %s = <%s> 没有出处（闭集里不许有说不清来历的取值）'
                            % (attr, v))
    for attr, v in sorted(prov):
        if (attr, str(v).lower()) not in known:
            problems.append('A2 出处里有一条 %s = <%s>，但闭集里没有它（陈旧条目）'
                            % (attr, v))
    for (attr, v), why in sorted(prov.items()):
        if (attr, str(v).lower()) not in known:
            continue
        if '旧写法' in why or '别名' in why:
            continue
        low = why.lower()
        flat = re.sub(r'[^a-z0-9]', '', low)
        # 家族前缀（llama / gpt / qwen …）：出处里提到一个就算对得上
        prefixes = set()
        for fam in fams:
            m = re.match(r'[a-z]+', fam)
            if m:
                prefixes.add(m.group(0))
        if not any(pre in flat for pre in prefixes):
            problems.append('A3 %s = <%s> 的出处 %r 对不上 models/ 里的任何模型'
                            % (attr, v, why[:40]))
    used = used_values(root)
    idle = sorted(known - used)
    if idle:
        notes.append('还没有任何模型用到的取值（%d 个）：%s'
                     % (len(idle), ' '.join('%s=%s' % x for x in idle)))
    return problems, notes


def _selftest():
    import shutil
    import tempfile
    d = pathlib.Path(tempfile.mkdtemp(prefix='by1attrs-'))
    bad = []
    try:
        (d / 'src').mkdir(parents=True)
        (d / 'models').mkdir(parents=True)
        (d / 'models/llama-shaped.by1').write_text('x\nact = silu\n',
                                                   encoding='utf-8')
        codegen = d / 'src/by1codegen.py'

        def enums_text(enums):
            body = ', '.join('%r: {%s}' % (k, ', '.join(repr(x) for x in v))
                             for k, v in enums.items())
            return 'ENUMS = {' + body + '}\n'

        def run(enums, prov):
            codegen.write_text(enums_text(enums), encoding='utf-8')
            return check(root=d, enums=enums, provenance=prov)

        probs, _n = run({'act': {'silu'}},
                        {('act', 'silu'): 'Llama 的 SwiGLU'})
        if probs:
            bad.append('正例被误判：%r' % (probs,))
        probs, _n = run({'act': {'silu', 'mystery'}},
                        {('act', 'silu'): 'Llama 的 SwiGLU'})
        if not [p for p in probs if 'A1' in p]:
            bad.append('A1 没抓到：%r' % (probs,))
        probs, _n = run({'act': {'silu'}},
                        {('act', 'silu'): 'Llama 的 SwiGLU',
                         ('act', 'gone'): 'Llama 的旧取值'})
        if not [p for p in probs if 'A2' in p]:
            bad.append('A2 没抓到：%r' % (probs,))
        probs, _n = run({'act': {'silu'}},
                        {('act', 'silu'): 'SomeVendor 的神秘实现'})
        if not [p for p in probs if 'A3' in p]:
            bad.append('A3 没抓到：%r' % (probs,))
        _p, notes = run({'act': {'silu', 'gelu'}},
                        {('act', 'silu'): 'Llama 的 SwiGLU',
                         ('act', 'gelu'): 'GPT-2 的 GELU'})
        if not notes:
            bad.append('没有模型用的取值没被列出来')
    finally:
        shutil.rmtree(d, ignore_errors=True)
    for b in bad:
        print('     ! ' + b)
    print('  [%s] 机制属性的出处：A1/A2/A3 各红一次，正例不误伤'
          % ('PASS' if not bad else 'FAIL'))
    return 0 if not bad else 1


def main(argv):
    if '--selftest' in argv:
        return _selftest()
    problems, notes = check(ROOT)
    for x in problems:
        print('     ! ' + x)
    for n in notes:
        print('  注：' + n)
    ok = not problems
    print('  [%s] 机制属性的出处 %s'
          % ('PASS' if ok else 'FAIL',
             '闭集里 %d 个取值都有官方出处' % len(PROVENANCE)
             if ok else '%d 处不成立' % len(problems)))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
