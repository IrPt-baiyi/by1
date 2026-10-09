#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1triage -- **把一批新 config 过一遍 by1，看谁需要新原语。**

这是"加模型"唯一靠谱的起点：**先让它拒，再决定做什么。**
不预判哪个模型"应该有"什么机制 —— 让工具说。

## 判据

    ✓ 已覆盖    boot 能出 IR，而且全是现成机制
    ! 猜了      boot 出得来，但有 guessed 的字段（**产物里看不出来的**）
    ✗ 拒了      缺机制 / 缺类型 / 解析不了 —— **这才是要干活的地方**
    ✗ 文件错    config 本身有问题（比如 gated repo 下下来的是错误页）

用法:  python by1triage.py [refs/xxx.config.json ...]
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
import glob
import importlib.util
import os
import sys
import by1io
import by1paths

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load(name):
    s = importlib.util.spec_from_file_location(
        name, by1paths.tool(name + '.py'))
    m = importlib.module_from_spec(s) if hasattr(importlib, 'module_from_spec') \
        else importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    import importlib.util as iu

    def ld(name):
        s = iu.spec_from_file_location(name, by1paths.tool(name + '.py'))
        m = iu.module_from_spec(s)
        s.loader.exec_module(m)
        return m

    boot = ld('by1boot')

    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        # 只要**新抓的**：时间上最新的那些，或者全都要
        files = sorted(glob.glob(by1paths.ref('*.config.json')))

    print()
    print('=' * 92)
    print('  新 config 过 by1 —— **谁需要新原语**')
    print('=' * 92)
    print()

    rows = []
    rows2 = []   # (名字, 判定, 备注, [认不出的张量名])
    for f in files:
        name = os.path.basename(f).replace('.config.json', '')
        try:
            cfg = by1io.read_json(f, encoding='utf-8')
        except Exception as e:
            rows.append((name, '文件错', str(e)[:40])); rows2.append((name, '文件错', '', []))
            continue
        if not isinstance(cfg, dict) or 'model_type' not in cfg:
            rows.append((name, '文件错', '没有 model_type（可能是错误页）')); rows2.append((name, '文件错', '', []))
            continue
        # **张量名才是判据。** classify() 从名字认机制；
        # 只给 config 的话什么机制都认不出来（见下面"空壳拒了"那段）。
        #
        # 这里原来有一张 `TENSOR_ALIAS` 表，把模型名映射到早年随手起的
        # 张量清单名（`stepfun-ai__Step-3.7-Flash` -> `step37-official`）。
        #
        # **那是"这个模型 -> 那个文件"的硬编码，不该有。** 用户指出得对。
        #
        # 正确的做法是让 `refs/` 遵循同一条规则：
        #     refs/<owner>__<Repo>.config.json
        #     refs/<owner>__<Repo>.tensors.json     <- 同名，只差后缀
        #
        # 文件名统一之后表就是多余的 —— 已经删了。
        # **规则能解决的事，不要用表；表只能解决规则解决不了的事。**
        real = {}
        _p = by1paths.ref(
                          os.path.basename(f)[:-len('.config.json')]
                          + '.tensors.json')
        if os.path.exists(_p):
            try:
                real = by1io.read_json(_p, encoding='utf-8')
            except Exception:
                real = {}
            # 这里原来还有一段：文件名对照表删掉之后的残留，缩在
            # `except` 里面、**永远跑不到**，还引用了一个不存在的 `p`。
            # 搬家的这一轮顺手删掉 —— 留着比没有更坏：它看起来像
            # "还有一条兜底的路"，而实际上那条路一步都走不了。
        try:
            out = boot.boot_ir(cfg, real, name)
            ir, guessed, gkeys = out if isinstance(out, tuple) else (out, [], [])
            kinds = set()
            for L in (ir.get('layers') or []):
                for o in (L.get('ops') or []):
                    kinds.add(o.get('kind'))
            # **"没崩"不等于"覆盖了"。**
            #
            # `classify()` 是从**张量名**认机制的。只给 config 的话，
            # 它认不出任何机制，退化成 `Add+Norm+Raw` —— 而第一版
            # triage 把这个当成了"✓ 已覆盖"，于是 37/39 全绿，
            # 连 gpt-oss-120b 都是 `Add+Norm+Raw`。
            #
            # **一个什么机制都没认出来的 IR，和"不支持"没区别，
            # 但报的是"✓"。** 那正是这个项目一直在防的那种谎。
            mechs = {k for k in kinds
                     if k not in ('Add', 'Norm', 'Raw', 'Embed', 'Head')}
            note = '+'.join(sorted(k for k in kinds if k))[:40]

            # **把认不出来的张量名捞出来 —— 那才是待办清单。**
            # boot_ir 在 Raw 那条里记了 attrs["tensors"]，
            # 以及 attrs["detected"]（classify 认出来的机制，可能为 None）。
            #
            # **两者要分开统计**：detected 有值 = 认出来了但 boot 不会展开
            # （是 boot 的活）；detected 为 None = 可能真是新机制。
            raw = []
            for L in (ir.get('layers') or []):
                for o in (L.get('ops') or []):
                    if o.get('kind') == 'Raw':
                        a = o.get('attrs') or {}
                        det = a.get('detected')
                        for t in (a.get('tensors') or []):
                            raw.append((t, det))
            rows2.append((name, '', note, raw))

            if not mechs:
                rows.append((name, '空壳拒了',
                             '只有 %s —— **没给张量名，认不出机制**'
                             % (note or '无')))
            elif gkeys:
                rows.append((name, '猜了%d' % len(gkeys), note))
            else:
                rows.append((name, '✓', note))
        except SystemExit as e:
            rows.append((name, '拒了', str(e)[:56]))
            rows2.append((name, '拒了', '', []))
        except Exception as e:
            rows.append((name, '拒了', '%s: %s' % (type(e).__name__, str(e)[:44])))
            rows2.append((name, '拒了', '', []))

    order = {'✓': 0, '猜了': 1}
    rows.sort(key=lambda r: (0 if r[1] == '✓' else
                             1 if r[1].startswith('猜') else 2, r[0]))
    n_ok = sum(1 for r in rows if r[1] == '✓')
    for name, verdict, note in rows:
        mark = {'✓': ' ✓ ', '文件错': ' ✗ '}.get(verdict, ' ! ')
        print('  %s %-46s %-8s %s' % (mark, name[:46], verdict[:8], note))

    # ── **待办清单：认不出来的张量** ────────────────────────────────
    #
    # 这才是整个流程唯一的产出。
    # "这个模型需要新机制"是感觉；"这三个张量名认不出来"才是账。
    print()
    print('=' * 92)
    print('  要加的机制 —— **按张量名分组，跨模型合并**')
    print('=' * 92)
    print()
    # 归一化：把层号替成 N，好把"每层都有的同一个模式"合成一条
    import re as _re

    def canon(t):
        # model.layers.0.xxx / layers.3.xxx / h.5.xxx -> N
        t = _re.sub(r'\.(\d+)\.', '.N.', t)
        t = _re.sub(r'\.layers\.\d+', '.layers.N', t)
        return t

    groups = {}
    unknown_mech = {}
    for name, verdict, note, raw in rows2:
        for t, det in raw:
            groups.setdefault(canon(t), set()).add(name)
            if not det:
                unknown_mech.setdefault(canon(t), set()).add(name)

    if groups:
        print('  **① 认出来了，但 boot 还不会展开**')
        print('     （`classify()` 返回了机制名 —— 是 **boot_ir 的活**，')
        print('      不是 by1 缺机制。加一个分支就行。）')
        print()
        n1 = 0
        for pat, models in sorted(groups.items(),
                                  key=lambda kv: (-len(kv[1]), kv[0])):
            if pat in unknown_mech:
                continue
            n1 += 1
            print('  %-54s %2d 个模型' % (pat[:54], len(models)))
        if not n1:
            print('  （没有）')
        print()
        print('  **② `classify()` 也认不出**（才可能是真新机制）')
        print()
        if unknown_mech:
            for pat, models in sorted(unknown_mech.items(),
                                      key=lambda kv: (-len(kv[1]), kv[0])):
                print('  %-54s %2d 个模型' % (pat[:54], len(models)))
        else:
            print('  **（没有 —— 一个真新机制都没有）**')
        print()
        print('  **① 是 boot_ir 没写完，② 才是 by1 可能要加的。**')
        print('  第一版把两者混成一句"444 种认不出来的张量"，')
        print('  读起来像 by1 缺一大堆机制 —— 而真相是 ① 占绝大多数。')
    else:
        print('  （没有认不出来的张量）')
    print()
    print('  ① %d 种模式  ·  ② %d 种模式'
          % (len(groups) - len(unknown_mech), len(unknown_mech)))
    print()
    print('  **"✓" 不等于"能算"** —— 那只是 boot 能出 IR。')
    print('  要证明能算，还得走 codegen / exec / C 三后端。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
