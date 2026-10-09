#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1contract -- 从机制声明**推出**张量契约。

## 为什么要这个

量过：一份 `.by1` 里 **27% 是决定**（机制 / 调度 / 位置），
**64% 是机械**（张量契约 + 导出命名）。
而机械那部分里，**契约是完全由决定推出来的** ——
`in_proj_qkv` 的形状就是 `k_heads*k_dim*2 + v_heads*v_dim`，
这件事 `by1check` 每天在算，只是**只拿去比对、不拿去生成**。

## 但有一半推不出来，得说清楚

契约分两半，性质不同：

    机制作用域   Attn / FFN / MoE / Linear / MLA
                 由种类 + 属性定 —— **能推**

    层 / 全局     input_layernorm / pre_attention_layernorm / norm / attn_norm
                 **四个名字指同一件事**，纯粹是各仓库的习惯 —— **推不出来**

所以这里对层/全局给一套**默认**（Llama 那一套），并明确标出
"这几个是你的仓库习惯，要改自己改"。而那正是第 ② 件（从产物反推）
要解决的部分。

用法:  python by1contract.py <文件>.by1 [--only Mech]
"""
import importlib.util
import os
import re
import sys
import by1io
import by1paths

# **原来这里写的是 `HERE = '.'`** —— 它只是个字面量，从来不是目录，
# 这个脚本完全靠 cwd 恰好是仓库根。现在按老规矩来。
HERE = os.path.dirname(os.path.abspath(__file__))


# ── 机制作用域的模板 ────────────────────────────────────────────────
# 每条是 (逻辑名, 形状表达式, 条件)。条件是 mech 的属性字典 -> bool。
# 形状表达式里的符号（q / kv / head_dim / d_model / hidden …）会被
# by1check 求值 —— 写符号而不是数字，是为了让它跟着属性走。
def _attn(a):
    g = str(a.get("q_gate", "")).lower() in ("true", "1", "yes", "on", "per_head")
    kn = str(a.get("qk_norm", "")).lower() not in ("", "off", "none", "false")
    hg = str(a.get("head_gate", "off")).lower() not in ("", "off", "none", "false")
    tie = str(a.get("kv_tie", "")).lower() in ("true", "1", "yes", "on")
    b = str(a.get("bias", a.get("attn_bias", ""))).lower() in ("true", "1", "yes", "on")
    out = [("q_proj.weight", "(q * head_dim%s, d_model)" % (" * 2" if g else ""), True),
           ("k_proj.weight", "(kv * head_dim, d_model)", True)]
    if not tie:
        out.append(("v_proj.weight", "(kv * head_dim, d_model)", True))
    out.append(("o_proj.weight", "(d_model, q * head_dim)", True))
    if kn:
        out += [("q_norm.weight", "(head_dim,)", True),
                ("k_norm.weight", "(head_dim,)", True)]
    if hg:
        out.append(("g_proj.weight", "(q, d_model)", True))
    if b:
        out.append(("*bias*", None, True))     # 占位：所有投影各加一条 bias
    return out


def _ffn(a):
    # **不写 `gate` 是"有门"，不是"没门"。**
    # 这和 codegen 那边 `_flag(_gv, True)` 的默认值必须一致 ——
    # 第一版这里把"没写"判成无门，于是 clef 的 FFN 漏了 gate_proj。
    _g = a.get("gate")
    g = True if _g is None else str(_g).lower() not in (
        "none", "false", "no", "off", "0")
    return [("gate_proj.weight", "(hidden, d_model)", g),
            ("up_proj.weight", "(hidden, d_model)", True),
            ("down_proj.weight", "(d_model, hidden)", True)]


def _moe(a):
    sh = int(float(a.get("shared", 0) or 0))
    sg = str(a.get("shared_gate", "")).lower() in ("true", "1", "yes", "on")
    out = [("gate.weight", "(experts, d_model)", True),
           ("gate_proj.weight", "(experts, hidden, d_model)", True),
           ("up_proj.weight", "(experts, hidden, d_model)", True),
           ("down_proj.weight", "(experts, d_model, hidden)", True)]
    if sh:
        h = a.get("shared_hidden", "hidden")
        out += [("shared_experts.gate_proj.weight", "(%s, d_model)" % h, True),
                ("shared_experts.up_proj.weight", "(%s, d_model)" % h, True),
                ("shared_experts.down_proj.weight", "(d_model, %s)" % h, True)]
    if sg:
        out.append(("shared_expert_gate.weight", "(1, d_model)", True))
    return out


# 这些**是 checkpoint 的打包习惯，推不出来**，只能提醒：
#   per_expert   一个专家一个张量（`mlp.experts.{e}.gate_proj.weight`）
#                还是打包成一整块（`...gate_exps.weight [E,H,D]`）
#   fuse         gate/up 是融合存储（`experts.gate_up_proj`）还是分开
#   bias         每个专家有没有 bias
# 生成器给的是"最朴素的那种"，这几样要照产物改。
PACKING_HINTS = """\
      # ⚠️ 下面几样**是产物的打包习惯，推不出来，照 checkpoint 改**：
      #     per_expert   一个专家一个张量，还是打包成一整块
      #     fuse         gate/up 融合存储（experts.gate_up_proj）还是分开
      #     bias         投影有没有 bias
      #     bias         融合存储时，名字是 c_attn 还是分开的 q/k/v
      #     --           权重共享时为"声明为不该存在"（lm_head.weight : --）"""


def _linear(a):
    return [("q_proj.weight", "(k_heads * k_dim, d_model)", True),
            ("k_proj.weight", "(k_heads * k_dim, d_model)", True),
            ("v_proj.weight", "(v_heads * v_dim, d_model)", True),
            ("q_conv1d.weight", "(k_heads * k_dim, 1, conv_kernel)", True),
            ("k_conv1d.weight", "(k_heads * k_dim, 1, conv_kernel)", True),
            ("v_conv1d.weight", "(v_heads * v_dim, 1, conv_kernel)", True),
            ("A_log", "(v_heads,)", True),
            ("dt_bias", "(v_heads,)", True),
            ("o_norm.weight", "(v_dim,)", True),
            ("o_proj.weight", "(d_model, v_heads * v_dim)", True)]


def _mla(a):
    return [("q_a_proj.weight", "(q_lora, d_model)", True),
            ("q_a_layernorm.weight", "(q_lora,)", True),
            ("q_b_proj.weight", "(q * qk_nope, q_lora)", True),
            ("kv_a_proj_with_mqa.weight", "(kv_lora + qk_rope, d_model)", True),
            ("kv_a_layernorm.weight", "(kv_lora,)", True),
            ("kv_b_proj.weight", "(q * (qk_nope + v_dim), kv_lora)", True),
            ("o_proj.weight", "(d_model, q * v_dim)", True)]


TEMPLATES = {"Attention": _attn, "FFN": _ffn, "MoE": _moe,
             "Linear": _linear, "MLA": _mla}

# 层 / 全局给一套默认。**这四个名字是各仓库的习惯，不是推出来的** ——
# 生成器只能给一套，改要你自己改（或者用 by1boot 从产物反推）。
LAYER_DEFAULT = [("input_layernorm.weight", "(d_model,)"),
                 ("post_attention_layernorm.weight", "(d_model,)")]
GLOBAL_DEFAULT = [("embed.weight", "(vocab, d_model)"),
                  ("final_norm.weight", "(d_model,)"),
                  ("lm_head.weight", "(vocab, d_model)")]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    f = sys.argv[1]
    only = None
    if '--only' in sys.argv:
        only = sys.argv[sys.argv.index('--only') + 1]

    sp = importlib.util.spec_from_file_location('bcm', os.path.join(HERE, 'by1check.py'))
    bc = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(bc)
    _r, info = bc.check(f)
    mechs = info.get('mechs') or {}

    # 每个 mech 用它自己的属性；同名机制被改了属性的话，两个都出
    src = by1io.read_text(by1paths.model(f), encoding='utf-8')
    seen = set()
    print('    tensors {')
    for m in re.finditer(r'mech\s+(\w+)\s*:\s*(\w+)\s*\{(.*?)\n  \}', src, re.S):
        name, kind, body = m.group(1), m.group(2), m.group(3)
        if only and name != only:
            continue
        attrs = dict(re.findall(r'^\s+(\w+)\s*=\s*([^#\n]+)', body, re.M))
        attrs = {k: v.strip().rstrip(',') for k, v in attrs.items()}
        fn = TEMPLATES.get(kind)
        if fn is None:
            print('      # %s : %s —— **这种类没有模板**（codegen 也没实现它）'
                  % (name, kind))
            continue
        key = (kind, tuple(sorted(attrs.items())))
        if key in seen:
            continue
        seen.add(key)
        rows = [(n, s) for (n, s, c) in fn(attrs) if c]
        print('      %s {' % name)
        for n, s in rows:
            if n == '*bias*':
                for bn in [x for x, _ in rows if x.endswith('.weight')]:
                    print('        %-34s (out_dim,)   # ← 名字要对上产物，自己改'
                          % (bn[:-7] + '.bias'))
                continue
            print('        %-34s %s' % (n + ' :', s))
        print('      }')
    print()
    print(PACKING_HINTS)
    print()
    print('      # ── 下面两个作用域**推不出来** ──────────────────────')
    print('      # input_layernorm / pre_attention_layernorm / norm / attn_norm')
    print('      # 是四个名字指同一件事 —— 各仓库的习惯。这里给的是 Llama 那一套。')
    print('      layer {')
    for n, s in LAYER_DEFAULT:
        print('        %-34s %s' % (n + ' :', s))
    print('      }')
    print('      global {')
    for n, s in GLOBAL_DEFAULT:
        print('        %-34s %s' % (n + ' :', s))
    print('      }')
    print('    }')
    return 0


if __name__ == '__main__':
    sys.exit(main())
