#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1gate -- 取值门的**可证伪对照**。

规则是：「声明了一个 codegen 没实现的取值，编译必须被拒」。

一条只会通过的规则不是规则 —— 所以这里拿**已知该被拒**的和
**已知该通过**的各试一遍。

这个脚本存在的理由和 `selftest.by1` 一样：**门本身也要有判卷人。**

    2026-10 的一次真实教训：GPT-2 原来在这个列表里是"必须被拒"
    （act = gelu_new 没实现）。后来把 gelu 实现了，它变成"必须通过" ——
    **而这个脚本立刻红了**，因为预期没跟着改。那是它该干的事。
"""

import os as _os
import sys as _sys
# **引导：把自己上面那一层（`src/`）放上 sys.path。**
# 加了它，`import by1paths` 才找得到；而 `by1paths` 在 import 时
# 会把 `src/` 和每个子目录都放上 sys.path —— 于是 `import by1check`
# 这种裸名 import 照旧能用。**这两行是生成的，别手改**
# （判据在 `by1paths.check_boot()`）。
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401
import importlib.util
import sys

sp = importlib.util.spec_from_file_location(
        'bcm', by1paths.tool('by1check.py'))
bc = importlib.util.module_from_spec(sp)
sp.loader.exec_module(bc)
import by1codegen as cg
# **这个 import 是为副作用**：`by1paths` 拉进 `by1io`，而 `by1io` 在
# import 时把 stdout/stderr 钉成 UTF-8。这个脚本用 `sys.exit(main())` 出口，
# 在一个非 UTF-8 的控制台上，判定行会打不出来（见 by1io 的 `force_utf8_stdio`）。
# 所以它不能删 —— `# noqa` 是给"看起来没用、其实有用"留的写法。
import by1paths  # noqa: F401
import os
import by1io

# (文件, 是否应当接受, 拒绝理由里应当出现的字样, 为什么)
CASE = [
    # ── 该被拒的 ──────────────────────────────────────────────────
    ('gate-probe.by1', False, 'nosuchactivation',
     '故意的反例：act 是个不存在的取值'),
    # **Ling 从"该被拒"搬到了"该通过"。**
    #
    # 它原来在"该被拒"那一组里，理由从 `KDA` 换成过一次 `MLA`（KDA 那一半
    # 实现了之后）。挡住它的最后一道是 **`qk_head`** —— MLA 的属性表没收它。
    #
    # 而 `qk_head` **不是一个新维度**：它是官方 config 里 `qk_head_dim` 的照抄，
    # 值必须等于 `qk_nope + qk_rope`。所以做的是"收它、并验它"，
    # 不是"实现一个新算子"。和 Nemotron 那次是同一个动作：
    # **期望过期了就改方向，不删。**
    ('Ling-3.0-tiny.by1', True, '',
     'qk_head 收进 MLA 的属性表了 —— 而且校验它 == qk_nope + qk_rope'),
    # **GLM-5.3-Flash 也从"该被拒"搬到了这里。** 挡住它的是稀疏索引器的
    # `index_heads` / `index_dim` / `index_topk` —— 那三个是**真机制**，
    # 不是别名：判卷人 `src/modelcheck/by1sparse.py`，对着 transformers 的
    # `GlmMoeDsaIndexer` 逐个 query 比"挑中了哪些 key"。
    #
    # ⚠ **这不等于 GLM-5.3-Flash 的前向做完了。** 它的 checkpoint 上还有
    #   `index_kpool_compress_gate` / `index_kpool_compress_ape` 两个张量，
    #   而**整套依赖里没有官方实现可比**（实测 0 处命中）——
    #   那一块没实现、也没验。见 1.md 的「已知缺口」。
    ('GLM-5.3-Flash.by1', True, '',
     '稀疏索引器的三个形状收了 —— 判卷人 src/modelcheck/by1sparse.py'),
    # ── 该通过的 ──────────────────────────────────────────────────
    # **Nemotron 从"该被拒"搬到了这里。**
    #
    # 它原来在这一行：`(..., False, 'SSM', 'Mamba 整族没实现')`。
    # 而 `SSM` 在 `by1codegen` 里实现之后，那条期望就**过时**了 ——
    # 这个护栏编码的是**旧的真相** ✅，而真相变了。
    #
    # 它不是被删掉，是**换了方向** ✓：现在要求它**编得出来**。
    # 少一边就等于把护栏拆了 —— 而那正是"一个不可能红的检查"。
    ('NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.by1', True, '',
     'SSM（选择性状态空间）实现了 —— 判卷人 src/modelcheck/by1ssm.py'),
    ('clef-tiny.by1', True, '', '取值全都实现过'),
    ('gpt2.by1', True, '',
     'gelu / LayerNorm / 学习式位置 / 无门控 MLP 现在都实现了'),
]

# ── 反例：**改坏一处，必须被拒** ───────────────────────────────────
#
# CASE 那张表只证明"这几份描述现在的走向对"。它证明不了**规则本身会红** ——
# 一个谁都不触发的检查和没有检查一样。所以这里逐条拿一份**好的**描述，
# 改坏一处，要求它被拒、而且拒绝理由里点得出那一处。
#
# **不另写 fixture**：反例就写成"改哪一行 -> 改成什么"，读的人一眼看得见
# 错在哪，也不会多出一份谁都不跑的 .by1。
#
#   (源文件, 把这串换成, 换成这个, 拒绝理由里该出现的, 为什么)
PROBE = [
    ('Ling-3.0-tiny.by1', '    qk_head    = 192', '    qk_head    = 193', 'qk_head',
     'qk_head 是别名 —— 不等于 qk_nope(128) + qk_rope(64) 就必须报，'
     '不许静默过（静默过 = "写了不报错、什么也没做"）'),
    ('Ling-3.0-tiny.by1', '    qk_head    = 192', '    qk_head    = 0', 'qk_head',
     '同上，取 0 也一样要报 —— 非 None 就参与校验'),
    ('GLM-5.3-Flash.by1', '    index_topk  = 2048', '    index_topk  = 0',
     'index_topk',
     'index_topk 是"每层挑多少个 key"，0 不是稀疏、是什么都不要'),
    ('GLM-5.3-Flash.by1', '    index_heads = 32\n', '', 'index_heads',
     '稀疏索引器要 index_heads / index_dim / index_topk 三个一起给 —— '
     '少一个就是只说了一半'),
]


def _rejects(path, needle):
    """返回 (被拒?, 理由)。"""
    try:
        _r, info = bc.check(path)
    except Exception as e:
        return True, '检查器异常: %s' % str(e)[:60]
    try:
        cg.compile_ir(info)
        return False, ''
    except cg.CodegenError as e:
        return True, str(e)


def probe(bad):
    """改坏一处的反例。写在临时目录里，跑完就删。"""
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix='by1gate-')
    try:
        for src, old, new, needle, why in PROBE:
            # 裸名走 `by1paths.find_model` —— 和 bc.check 同一个解析，
            # 否则 cwd 一换就 FileNotFoundError。
            text = by1io.read_text(by1paths.find_model(src), encoding='utf-8')
            if old not in text:
                print('  !! %-22s **反例的原句找不到** —— %r' % (src, old.strip()))
                bad.append(src + '(反例)')
                continue
            if old == new:
                print('  !! %-22s 反例没改动任何东西' % src)
                bad.append(src + '(反例)')
                continue
            p = os.path.join(tmp, src)
            by1io.write_text(p, text.replace(old, new, 1), encoding='utf-8')
            rejected, msg = _rejects(p, needle)
            if rejected and (not needle or needle in msg):
                print('  ok %-22s 反例被拒：%s' % (src, msg.strip().splitlines()[0][:46]))
            elif not rejected:
                print('  !! %-22s **反例被接受了** —— %s' % (src, why))
                bad.append(src + '(反例)')
            else:
                print('  !! %-22s **拒了，但理由没点到 %s** —— %s'
                      % (src, needle, msg.strip()[:40]))
                bad.append(src + '(反例理由)')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def short_probe(bad):
    """短描述与完整声明同 IR；挂载参数、拒绝和命令入口都有反例。"""
    import pathlib
    import subprocess
    import tempfile

    header = "model Tiny {\n hparams { d_model = 16, vocab = 32 }\n"
    short = header + """ stack main {
  pattern = 2 * Attention(q = 4, kv = 2, head_dim = 4)
  main[:] >> FFN(hidden = 32)
 }
}
"""
    full = header + """ mech Attention : Attention { heads = { q = 4, kv = 2, head_dim = 4 } }
 mech FFN : FFN { hidden = 32 }
 stack main { pattern = 2 * Attention, main[:] >> FFN }
}
"""
    with tempfile.TemporaryDirectory(prefix='by1-short-') as tmp:
        root = pathlib.Path(tmp).resolve()
        if root.parent != pathlib.Path(tempfile.gettempdir()).resolve():
            raise RuntimeError('测试目录不在临时目录内')
        path = root / 'tiny.by1'

        def require(condition, message='检查不通过'):
            if not condition:
                raise AssertionError(message)

        def read(text):
            path.write_text(text, encoding='utf-8')
            return bc.check(str(path))

        def compile_text(text):
            report, info = read(text)
            require(not report.count(bc.E), report.items)
            return cg.compile_ir(info), info

        def ffn_widths(ir):
            return [o['attrs']['hidden'] for l in ir['layers']
                    for o in l['ops'] if o['kind'] == 'FFN']

        def case(name, fn):
            try:
                fn()
                print('  ok 短描述：' + name)
            except Exception as ex:
                bad.append('短描述/' + name)
                print('  !! 短描述：%s — %s' % (name, ex))

        def equivalent():
            a, _ = compile_text(short)
            b, _ = compile_text(full)
            require(a == b, '简写与完整声明的计算语义不一致')
        case('完整声明与简写同 IR', equivalent)

        def selected():
            text = short.replace('main[:] >> FFN(hidden = 32)',
                'main[0] >> FFN(hidden = 24, gate = none, act = gelu_new)\n'
                '  main[1] >> FFN(hidden = 40)')
            ir, _ = compile_text(text)
            require(ffn_widths(ir) == [24, 40])
            op = next(o for o in ir['layers'][0]['ops'] if o['kind'] == 'FFN')
            require(op['attrs']['gate'] is False and op['attrs']['act'] == 'gelu_new')
        case('选择器与调用参数进入计算', selected)

        def declared():
            ir, _ = compile_text(full.replace('main[:] >> FFN', 'main[:] >> FFN(hidden = 48)'))
            require(ffn_widths(ir) == [48, 48])
        case('显式机制支持调用覆盖', declared)

        def multistack():
            text = short.replace('2 * Attention', '1 * Attention').replace(
                '\n}\n', '\n stack aux {\n aux = true\n'
                ' pattern = 1 * Attention(q = 4, kv = 2, head_dim = 4)\n'
                ' aux[:] >> FFN(hidden = 48)\n }\n'
                ' emit {\n train -> transformers.config {\n field {\n'
                ' widths = per_layer(attach.hidden)\n'
                ' widths_default = per_layer(attach.hidden, 0)\n'
                ' widths_map = by_layer(attach.hidden)\n'
                ' }\n }\n }\n}\n')
            ir, info = compile_text(text)
            require(ffn_widths(ir) == [32, 48])
            require(info['hparams']['n_layer'] == '1')
            require(info['config']['widths'] == [32, 48], info['config'])
            require(info['config']['widths_default'] == [32, 48], info['config'])
            require(info['config']['widths_map'] == {'0': 32, '1': 48}, info['config'])
        case('不同栈的挂载参数互不串层', multistack)

        def override():
            text = short.replace('main[:] >> FFN(hidden = 32)',
                'main[:] >> FFN(hidden = 32)\n  main[1] : FFN.hidden = [48]')
            ir, _ = compile_text(text)
            require(ffn_widths(ir) == [32, 48])
        case('逐层覆盖优先于调用参数', override)

        for name, text in [
            ('机制拼写错误', short.replace('Attention(', 'Attentoin(')),
            ('选择器未命中', short.replace('main[:]', 'main[9]')),
            ('挂载目标不是单个调用', short.replace('FFN(hidden = 32)', 'FFN(hidden = 32) + FFN(hidden = 32)')),
            ('重叠挂载参数冲突', short.replace('main[:] >> FFN(hidden = 32)',
                'main[:] >> FFN(hidden = 32)\n  main[0] >> FFN(hidden = 48)')),
        ]:
            def rejected(text=text):
                report, _ = read(text)
                require(report.count(bc.E), '错误描述被接受')
            case(name + '被拒绝', rejected)

        for name, text in [
            ('缺少必要维度', short.replace(', head_dim = 4', '')),
            ('未知激活函数', short.replace('hidden = 32)', 'hidden = 32, act = bogus)')),
        ]:
            def unsupported(text=text):
                report, info = read(text)
                try:
                    cg.compile_ir(info)
                except cg.CodegenError:
                    return
                raise AssertionError('生成器静默放行')
            case(name + '被拒绝', unsupported)

        def cli():
            script = by1paths.tool('by1.py')
            output = root / 'model.py'
            read(short)
            result = subprocess.run([sys.executable, script, 'build', str(path), '-o', str(output)],
                                    capture_output=True, encoding='utf-8')
            require(result.returncode == 0 and output.exists(), result.stdout + result.stderr)
            read(short.replace('Attention(', 'Attentoin('))
            output.write_text('sentinel', encoding='utf-8')
            result = subprocess.run([sys.executable, script, 'build', str(path), '-o', str(output)],
                                    capture_output=True, encoding='utf-8')
            require(result.returncode == 1 and output.read_text() == 'sentinel')
        case('build 可生成源码且拒绝错误输入', cli)


def main():
    bad = []
    for f, want_ok, needle, why in CASE:
        try:
            _r, info = bc.check(f)
        except Exception as e:
            bad.append('%s: 检查器就没过 %s' % (f, str(e)[:40]))
            continue
        try:
            cg.compile_ir(info)
            got, msg = True, ''
        except cg.CodegenError as e:
            got, msg = False, str(e)

        if got == want_ok:
            detail = '接受' if got else '拒绝：' + msg.strip().replace('\n', ' ')[:52]
            print('  ok %-22s %s' % (f, detail))
        else:
            print('  !! %-22s **预期%s，实际%s** —— %s'
                  % (f, '接受' if want_ok else '拒绝',
                     '接受' if got else '拒绝', why))
            bad.append(f)

        if not got and needle and needle not in msg:
            print('         拒绝理由里没提到 %s' % needle)
            bad.append(f + '(理由)')

    print()
    print('  反例（**改坏一处，必须被拒**）:')
    probe(bad)
    print('\n  几行定义架构：')
    short_probe(bad)

    print()
    print('  [%s] 取值门 %s'
          % ('PASS' if not bad else 'FAIL',
             '工作正常' if not bad else '坏了：%s' % ', '.join(bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
