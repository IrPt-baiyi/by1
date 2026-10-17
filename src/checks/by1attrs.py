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


#: 闭集里**没有模型走到、但明确保留**的取值（A4 要它们）。
#: 每条都要写理由；一旦有模型用到它，这条就要删掉（否则 A4 报“残留”）。
RESERVED = {
    ('act', 'gelu'): '精确 GELU 实现留着；当前没有官方模型用它（GPT-2 用 tanh 近似），两条公式由 _gelu 的断言守着',
    ('qk_norm', '1'): '旧写法（等价 per_head），by1langspec 记档；不会有模型这么写',
    ('qk_norm', 'on'): '旧写法（等价 per_head），by1langspec 记档',
    ('qk_norm', 'yes'): '旧写法（等价 per_head），by1langspec 记档',
}

def codegen_defaults(codegen_path=None):
    """从 `by1codegen.py` 取每个机制类型的属性默认值（`attrs.get("x", "y")`）。"""
    p = pathlib.Path(codegen_path or (ROOT / "src/by1codegen.py"))
    text = p.read_text(encoding="utf-8")
    out, kind = {}, None
    for line in text.splitlines():
        m = re.match(r'\s*if kind == "(\w+)":', line)
        if m:
            kind = m.group(1)
            out.setdefault(kind, {})
            continue
        if kind is None:
            continue
        for a, v in re.findall(r'attrs\.get\("(\w+)", "([^"]+)"\)', line):
            out[kind].setdefault(a, v)
    return out


def implicit_values(root=ROOT):
    """模型里**没写**、但编译器会按默认值用到的 (属性, 取值)。"""
    defaults = codegen_defaults(pathlib.Path(root) / "src/by1codegen.py")
    seen = set()
    pat_mech = re.compile(r"mech\s+(\w+)\s*:\s*(\w+)\s*\{([^}]*)\}")
    for f in (pathlib.Path(root) / "models").rglob("*.by1"):
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in pat_mech.finditer(text):
            body, kind = m.group(3), m.group(2)
            for attr, val in (defaults.get(kind) or {}).items():
                if re.search(r"\b" + attr + r"\s*=", body):
                    continue      # 显式写了，不算隐式
                if attr == "gate_act":
                    hg = re.search(r"head_gate\s*=\s*([A-Za-z_0-9]+)", body)
                    if not hg or hg.group(1).lower() == "off":
                        continue  # 没有门就不会用到激活
                seen.add((attr, val.lower()))
    return seen

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


def check(root=ROOT, enums=None, provenance=None, reserved=None):
    root = pathlib.Path(root)
    enums = load_enums(root / 'src/by1codegen.py') if enums is None else enums
    prov = PROVENANCE if provenance is None else provenance
    RES = RESERVED if reserved is None else reserved
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
    # A4：每个取值要么**有模型走到**（显式写出来 或 靠默认值生效），
    # 要么在 RESERVED 里**写明保留理由** —— 不允许“谁也不知道为什么它在”。
    eff = set(used_values(root)) | set(implicit_values(root))
    for attr, v in sorted(known):
        if (attr, v) in eff:
            continue
        if (attr, v) not in RES:
            problems.append('A4 %s = <%s> 没有任何模型用到，也没在 RESERVED '
                            '里写保留理由（闭集里不允许谁也不知道为什么它在）'
                            % (attr, v))
    for attr, v in sorted(RES):
        if (attr, str(v).lower()) not in known:
            problems.append('A4 RESERVED 里有一条 %s = <%s>，但闭集里没有它'
                            % (attr, v))
        elif (attr, str(v).lower()) in eff:
            problems.append('A4 RESERVED 里 %s = <%s> 已经有模型用到了 —— '
                            '这条保留记录该删了' % (attr, v))
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

        def run(enums, prov, reserved=None):
            codegen.write_text(enums_text(enums), encoding='utf-8')
            if reserved is None:
                reserved = {} if enums != {'act': {'silu'}} else {}
            return check(root=d, enums=enums, provenance=prov,
                         reserved=reserved if reserved is not None else {})

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
        prov2 = {('act', 'silu'): 'Llama 的 SwiGLU',
                 ('act', 'gelu'): 'Llama 那边也有的 GELU'}
        # A4：没模型用、也没保留理由 -> 红
        probs, _n = run({'act': {'silu', 'gelu'}}, prov2, reserved={})
        if not [p for p in probs if 'A4' in p]:
            bad.append('A4 没抓到没人用的取值：%r' % (probs,))
        # A4：写了保留理由 -> 不红
        probs, _n = run({'act': {'silu', 'gelu'}}, prov2,
                        reserved={('act', 'gelu'): '保留着'})
        if probs:
            bad.append('A4 保留条目被误判：%r' % (probs,))
        # A4：保留了但已经有人用 -> 残留
        probs, _n = run({'act': {'silu'}}, {('act', 'silu'): 'Llama 的 SwiGLU'},
                        reserved={('act', 'silu'): '保留着'})
        if not [p for p in probs if 'A4' in p]:
            bad.append('A4 没抓到残留的保留条目：%r' % (probs,))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    for b in bad:
        print('     ! ' + b)
    print('  [%s] 机制属性的出处：A1/A2/A3/A4 各红一次，正例不误伤'
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
