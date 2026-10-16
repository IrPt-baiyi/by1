"""by1backends -- 后端 × 机制矩阵：**承诺的公共子集到底有多大**。

## 它回答的是哪一问

[RC1 决策](../docs/rc1-decision.md) 把这一条列成"**未定**"：

> **承诺的公共子集有多大**（NumPy/C/ggml 三个后端对 SSM2、混合栈、`Parallel`
> 一律拒绝）—— 不许把它写成"多后端已支持"。

"未定"是因为没人**量过**。这个判卷人就是去量：拿"一机制一份"的探针声明，
挨个问五个后端要一个明确答复。

## 三态（和别处一样，跳过 ≠ 通过）

    ✅ 接受      后端给出产物
    ✗ 拒绝      后端明确报错 —— **而且必须带理由**（见下）
    ? 没验      缺依赖（numpy / transformers），或探针本身没过检查

## 两条判据

1. **拒绝必须出声。** 每个拒绝都要有非空、且点了机制名的消息 ——
   沉默的拒绝（返回空、或抛一个没信息的异常）等于"我以为我拒绝了"。
2. **文档与实测一致。** `docs/backend-matrix.md` 是这个脚本的**生成物**；
   实测变了而没重新生成 -> 红。文档因此不会漂。

## 产出

屏幕上的矩阵 + `docs/backend-matrix.md`（`--write` 时写）。
**公共子集**（所有后端都接受的探针）单独一行 —— 那一行就是"承诺"。
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import by1check  # noqa: E402
import by1paths  # noqa: E402

DOC = 'docs/backend-matrix.md'

#: 探针：一份声明 = 一组机制。挑的都是"一机制一份"的 shaped 件，
#: 这样"哪个后端支持哪种机制"能从矩阵里直接读出来。
PROBES = [
    ('hello.by1', 'Attention + FFN（最小完整）'),
    ('attn-fused-shaped.by1', 'Attention（融合 qkv + 门）'),
    ('mla-shaped.by1', 'MLA'),
    ('mla-sparse-shaped.by1', 'MLA + 稀疏索引（DSA）'),
    ('moe-routing-shaped.by1', 'MoE（路由）'),
    ('kda-shaped.by1', 'KDA'),
    ('kda-gva-shaped.by1', 'KDA（GVA）'),
    ('kda-lowrank-shaped.by1', 'KDA（低秩）'),
    ('ssm-shaped.by1', 'SSM（Mamba2/Nemotron-H 参数化）'),
    ('raw-escape.by1', 'Raw（逃生舱）'),
    ('complete-granite-4.0-h-350m.by1', 'SSM2 + Attention（逐层混排）'),
    ('complete-Falcon-H1-0.5B-Instruct.by1', 'Parallel（层内并行：SSM2 + Attention）'),
    # ── 第二批（2026-10-17 扩探针）：SHAPED 剩下的九个 ──────────────
    # 把"缺口清单"做细的前提是**量得够全** —— 有几个机制（Linear 线性注意力、
    # LLaMA3 的 RoPE 缩放、learned 位置）原来只在三后端对拍里出现过，
    # 后端矩阵从来没量过它们。
    ('llama-shaped.by1', 'Attention + FFN（RoPE）'),
    ('llama3-shaped.by1', 'Attention + FFN（LLaMA3 RoPE 缩放）'),
    ('mixtral-shaped.by1', 'Attention + MoE'),
    ('gpt-oss-shaped.by1', 'Attention + MoE（sigmoid 路由）'),
    ('qwen3-next-shaped.by1', 'Linear + MoE + Attention（线性注意力）'),
    ('ling-tiny.by1', 'KDA + MLA + MoE（Ling 结构的机械缩小版）'),
    ('glm-text-tiny.by1', 'KDA（低秩门）+ SparseMLA + MoE'),
    ('clef-tiny.by1', 'Linear + FFN + Attention（视觉塔布局）'),
    ('gpt2-tiny.by1', 'Attention + FFN（learned 位置）'),
]

BACKEND_NAMES = ('codegen', 'c', 'ggml', 'exec', 'full')

#: **两扇门要分开算。** `codegen / c / ggml / exec` 吃的是**串行 IR**
#: （一层一个混合器的机制组合）；`full` 吃的是**官方家族**
#: （`arch = qwen3 / granitemoehybrid / …`，多分支、多模态）。
#: 一把梭地算"五后端交集"会被门型混淆 —— 得到的 0 说明不了"机制支持不支持"。
#: 所以两个数都报：
#:   · **IR 路公共子集**（四后端交集）—— 这是"机制层面的承诺"；
#:   · **五后端交集** —— 顺带记下"没有任何声明同时走两扇门"这个事实。
IR_PATH = ('codegen', 'c', 'ggml', 'exec')

#: **拒绝的两条通道都要接。** 这个仓库里 `by1c` 的拒绝是 `raise SystemExit`、
#: `by1exec` 也是（`[不支持] 这个后端还没有实现 X`）—— `SystemExit` 不是
#: `Exception` 的子类，只接 `except Exception` 就会让**整个判卷人当场退掉**
#: （实测：`raw-escape.by1` 在 `C 后端还没有 Raw` 那句之后直接 exit 1）。
#: "进程活下来了吗"本身就是判据的一部分。
_REFUSAL = (Exception, SystemExit)


def _accept():
    return True, ''


def _probe_codegen(info, ir):
    """PyTorch 生成器：`compile_ir` 不支持的 kind 会抛 `UnsupportedError`。"""
    try:
        import by1codegen
        by1codegen.compile_ir(info)
    except _REFUSAL as ex:
        return False, str(ex) or type(ex).__name__
    return _accept()


def _probe_c(info, ir):
    """生成 C：不支持的算子/属性在 `emit_c_ir` 里明确报错。"""
    if ir is None:
        # **不是本后端的拒绝，是"根本没走到"。** 冒充成拒绝会污染
        # "每个拒绝都带理由"那条判据（上游拒了，本后端没什么可说的）。
        return None, '上游 codegen 先拒（本后端未到）'
    try:
        import by1c
        import by1exec
        by1c.emit_c_ir(ir, by1exec.shapes_of(ir))
    except _REFUSAL as ex:
        return False, str(ex) or type(ex).__name__
    return _accept()


def _probe_ggml(info, ir):
    """ggml 生成器（**第四后端 = IR → ggml C**，见 by1all 的 6b 段）。

    不只问 `check_support` —— **真把 C 发出来**（发射器里的断言/拒绝也算拒绝）。
    图路径（`emit_model_c_graph`）是并行的迁移线，有自己的缺口清单，
    不在这一格里判 —— 矩阵每格写的是"IR 路"这句承诺。
    """
    if ir is None:
        return None, '上游 codegen 先拒（本后端未到）'
    try:
        import by1c
        import by1exec
        import by1ggml
        problems = by1ggml.check_support(ir)
        if problems:
            return False, '；'.join(str(p) for p in problems)
        order = by1c.emit_c(ir, info, by1exec.shapes_of(ir))[1]
        by1ggml.emit_model_c(ir, order, by1exec.shapes_of(ir))
    except _REFUSAL as ex:
        return False, str(ex) or type(ex).__name__
    return _accept()


def _probe_exec(info, ir):
    """NumPy 执行器：支持集就是 `OPS` 表（前向里 `OPS.get(kind)` 那一步）。

    所以这里**静态**读同一张表 —— 走的不是另一条判据，而是一模一样的判据。
    """
    if ir is None:
        return None, '上游 codegen 先拒（本后端未到）'
    import by1exec
    missing = []
    for layer in ir['layers']:
        for op in layer['ops']:
            if op['kind'] not in by1exec.OPS and op['kind'] not in missing:
                missing.append(op['kind'])
    if missing:
        return False, '这个后端还没有实现 %s' % '、'.join(missing)
    return _accept()


def _probe_full(info, ir):
    """完整模型后端（Transformers）：把声明编成官方 config。缺 transformers = 没验。"""
    try:
        import transformers  # noqa: F401
    except ImportError:
        return None, '缺 transformers'
    try:
        import by1full
        by1full.compile_config(info)
    except _REFUSAL as ex:
        return False, str(ex) or type(ex).__name__
    return _accept()


PROBES_FN = {
    'codegen': _probe_codegen, 'c': _probe_c, 'ggml': _probe_ggml,
    'exec': _probe_exec, 'full': _probe_full,
}


def measure(verbose=True):
    """跑遍探针 × 后端。返回 (行, 问题)。

    行：`(探针, 机制说明, {后端: (True/False/None, 消息)})`
    问题：协议级的缺陷（比如"拒绝了但没说为什么"）
    """
    rows, problems = [], []
    for name, label in PROBES:
        # 探针先过声明检查 —— 否则量的是"声明写错了"，不是"后端支持吗"。
        report, info = by1check.check(name)
        if report.count(by1check.E):
            problems.append('%s 声明检查就不过（%d 个错误），矩阵里这一行没有意义'
                            % (name, report.count(by1check.E)))
            rows.append((name, label, {b: (None, '声明检查不过') for b in BACKEND_NAMES}))
            continue
        ir = None
        try:
            import by1codegen
            ir = by1codegen.compile_ir(info)
        except _REFUSAL:                           # noqa: BLE001
            ir = None                              # 各后端自己会说"上游拒了"
        result = {}
        for backend in BACKEND_NAMES:
            ok, msg = PROBES_FN[backend](info, ir)
            # 两条协议判据（判的是**真的缺陷**，不是风格）：
            #   ① 拒绝必须出声 —— 空消息 = "我以为我拒绝了"；
            #   ② 消息不能只有异常类型名（`str(ex)` 为空时我兜底成类型名，
            #      那种行等于没说）。
            # "上游先拒"（ok=None）不算本后端的拒绝，所以不进这两条。
            if ok is False and not str(msg).strip():
                problems.append('%s × %s：**拒绝了但没说为什么**' % (name, backend))
            if ok is False and str(msg).strip() and len(str(msg).strip()) < 8:
                problems.append('%s × %s：拒绝消息太短，等于没说（%r）'
                                % (name, backend, str(msg)[:40]))
            # **拒绝必须归档**（三档之一）：可补 / 卡上游 / 设计如此。
            # 归不了档 = 缺口没想清 —— 红出来补分类，矩阵才不会
            # 退化成"红了一片没人管"。
            if ok is False:
                bucket, _why = classify(name, backend, str(msg))
                if bucket is None:
                    problems.append('%s × %s：拒绝**没分类**（可补/卡上游/设计如此）：'
                                    '%r' % (name, backend, str(msg)[:60]))
            result[backend] = (ok, str(msg).strip())
        rows.append((name, label, result))
        if verbose:
            def _mark(b):
                ok, msg = result[b]
                return ('✅' if ok is True else '✗' if ok is False
                        else '—' if str(msg).startswith('上游') else '?')
            print('  %-36s %s' % (name, ' '.join(_mark(b) for b in BACKEND_NAMES)))
    return rows, problems


def render(rows, problems):
    """生成文档（`docs/backend-matrix.md`）。"""
    import datetime
    common = [r for r in rows
              if all(r[2].get(b, (None, ''))[0] is True for b in BACKEND_NAMES)]
    out = []
    out.append('# 后端 × 机制矩阵（生成物，别手改）')
    out.append('')
    out.append('> 这个文件是 [`src/checks/by1backends.py`](../src/checks/by1backends.py) '
               '的产物：')
    out.append('> 跑 `python src/checks/by1backends.py --write` 重新生成。')
    out.append('> 实测与它不一致时，`by1all` 判卷人段会红 —— **所以它不会漂**。')
    out.append('')
    out.append('## 一、矩阵')
    out.append('')
    out.append('| 探针 | 机制 | ' + ' | '.join('`%s`' % b for b in BACKEND_NAMES) + ' |')
    out.append('|---|---|' + '---|' * len(BACKEND_NAMES))
    for name, label, result in rows:
        cells = []
        for b in BACKEND_NAMES:
            ok, msg = result.get(b, (None, ''))
            if ok is True:
                cells.append('✅')
            elif ok is False:
                cells.append('✗')
            elif str(msg).startswith('上游'):
                cells.append('—')      # 根本没走到（上游先拒），不是"没验"
            else:
                cells.append('? 没验')  # 缺依赖
        out.append('| `%s` | %s | %s |' % (name, label, ' | '.join(cells)))
    out.append('')
    out.append('**读法**：✅ = 后端接受并给出产物；✗ = 明确拒绝（原文见第二节）；'
               '— = 根本没走到（上游 `codegen` 先拒，本后端无从表态）；'
               '? = 没验（缺依赖，**不算通过**）。')
    out.append('')
    out.append('## 二、每一个拒绝的**原文**')
    out.append('')
    out.append('> 这张表是"要么实现，要么拒绝"的落点：越界不是沉默，是一句'
               '能贴进 issue 的话。')
    out.append('')
    out.append('| 探针 | 后端 | 拒绝原文 |')
    out.append('|---|---|---|')
    for name, _label, result in rows:
        for b in BACKEND_NAMES:
            ok, msg = result.get(b, (None, ''))
            if ok is False:
                out.append('| `%s` | `%s` | %s |' % (name, b, msg.replace('|', '\\|')[:300]))
    out.append('')
    out.append('## 三、**公共子集**（两扇门分开算）')
    out.append('')
    ir_common = [r for r in rows
                 if all(r[2].get(b, (None, ''))[0] is True for b in IR_PATH)]
    out.append('### 3.1 IR 路（`codegen` / `c` / `ggml` / `exec` 四后端交集）')
    out.append('')
    out.append('这是**机制层面的承诺**：这些声明在四个后端上都出得了产物。')
    out.append('')
    if ir_common:
        for name, label, _r in ir_common:
            out.append('- `%s` —— %s' % (name, label))
    else:
        out.append('**一个都没有。**')
    out.append('')
    out.append('### 3.2 五后端交集（含 `full`）')
    out.append('')
    if common:
        for name, label, _r in common:
            out.append('- `%s` —— %s' % (name, label))
    else:
        out.append('**一个都没有。** 这不奇怪：`full` 是**另一扇门**'
                   '（吃 `arch = qwen3 / granitemoehybrid / …` 的官方家族），'
                   '而上面那些 shaped 探针走的是串行 IR 门。'
                   '**但对外措辞要照实**：不存在"一份声明五后端通吃"，'
                   '"多后端已支持"这句话现在不成立。')
    out.append('')
    out.append('## 四、剩余缺口的**三档分类**')
    out.append('')
    out.append('> 每个 ✗ 格都要归档（判据会红出未分类的拒绝）：')
    out.append('> **可补** = 我们的工程活；**卡上游** = 缺 ggml/上游原语；')
    out.append('> **设计如此** = 边界而非缺口。')
    out.append('')
    groups = {b: [] for b in BUCKETS}
    for name, _label, result in rows:
        for b in BACKEND_NAMES:
            ok, msg = result.get(b, (None, ''))
            if ok is False:
                bucket, why = classify(name, b, str(msg))
                if bucket:
                    groups[bucket].append((name, b, why))
    for bucket in BUCKETS:
        items = groups[bucket]
        out.append('### %s（%d 格）' % (bucket, len(items)))
        out.append('')
        if not items:
            out.append('**没有。**')
        for name, b, why in items:
            out.append('- `%s` × `%s` —— %s' % (name, b, why))
        out.append('')
    out.append('生成时间：%s（仅作记录，判据不看它）' % datetime.date.today().isoformat())
    out.append('')
    return '\n'.join(out)


#: 每个 ✗ 格必须归到**三档**之一 —— 不许有未分类的拒绝
#: （新增缺口时判据会要求补分类，矩阵才不会退化成"红了一片没人管"）：
#:   **可补**     我们的工程活：判据现成、只差实现；
#:   **卡上游**   缺 ggml/上游原语（top-k / 比较 / delta_rule 递推）；
#:   **设计如此** 边界而非缺口（两扇门 / 语义没定义就不猜）。
BUCKETS = ("可补", "卡上游", "设计如此")


def classify(probe, backend, msg):
    """给一个拒绝格归档。返回 (档, 一句理由)；归不了返回 (None, '')。"""
    if ("完整后端的 arch 必须是" in msg
            or "完整模型使用 transformers 后端" in msg
            or "完整后端不能忽略这些块" in msg):
        return "设计如此", ("两扇门：`full` 吃官方家族的 arch；`tensors{}` / "
                          "`emit{}` 是 IR 路的契约块 —— 不是缺口")
    if ("top-k" in msg or "top_k" in msg or "比较算子" in msg
            or "算子 MoE 还没有 ggml 映射" in msg
            or "index_heads" in msg):
        # **2026-10-17 实测打脸一条摸底结论**：by1ggml 里写过"这版 ggml
        # 没有 top_k 和比较算子"（2026-10-14 摸底）—— 逐个数 ggml.h 的
        # `enum ggml_op`（101 个）之后发现 **`GGML_OP_TOP_K`、
        # `GGML_OP_ARGSORT`、`GGML_OP_LIGHTNING_INDEXER`（DSA 索引器）
        # 全在**。缺的是映射实现，不是上游能力 —— 归"可补"。
        return "可补", ("ggml 有 `TOP_K`/`ARGSORT`/`LIGHTNING_INDEXER`"
                       "（2026-10-17 逐个数过 enum ggml_op，101 个）—— "
                       "缺的是路由语义到这些原语的映射实现")
    if ("delta_rule" in msg or "算子 KDA 还没有 ggml 映射" in msg
            or "算子 Linear 还没有 ggml 映射" in msg
            or ("递推" in msg and backend == "ggml")):
        # 同上：`GGML_OP_GATED_DELTA_NET` 与 `GGML_OP_GATED_LINEAR_ATTN`
        # 就是 delta-rule / GDN 族的原语（`RWKV_WKV7` 也在）。与 by1 的
        # KDA/Linear 逐项对表是**实现工作**（可能要折算，像 SSM 那次）。
        return "可补", ("ggml 有 `GATED_DELTA_NET`/`GATED_LINEAR_ATTN`"
                       "（delta-rule/GDN 族原语）—— 与 by1 的 KDA/Linear "
                       "对表是实现工作（可能要折算）")
    if ("yarn" in msg or "llama3" in msg.lower()) and backend == "ggml":
        # **顺序要紧**：这条排在"缺算子"后面 —— 一个格子同时有 yarn 和 MoE
        # 时（gpt-oss 就是），硬阻塞（卡上游）才算数，yarn 是"顺手能补"。
        # 真·可补：LLaMA3 的 RoPE 缩放是 yarn 风格（beta_fast/beta_slow/
        # factor），`ggml_rope_ext` 的参数位就摆在那里 —— 缺的是折算实现。
        return "可补", ("LLaMA3 的 RoPE 缩放（yarn 风格 beta_fast/beta_slow/factor）"
                       "—— `ggml_rope_ext` 的参数位就有，缺的是折算实现")
    if "Raw" in msg and backend in ("c", "exec"):
        # **两层逃生舱的分工**（2026-10-17 核对过代码）：
        #   第一层 `Raw` = `raw.py` 的工厂，**契约就是 PyTorch**（返回 nn.Module）；
        #   第二层 `External` = .so + C —— **exec/c/codegen 三后端都在**
        #   （`by1extdemo` 刚验过："PyTorch · NumPy · C 39 行 11 个张量"）。
        # 所以 c/exec 的 Raw 拒绝不是缺口：要那两个后端的逃生舱，走第二层。
        return "设计如此", ("两层分工：`Raw`（第一层）的契约是 PyTorch 工厂；"
                          "c/exec 的逃生舱走**第二层 `External`（.so）** —— "
                          "已由 by1extdemo 三后端验过，不是缺口")
    if "Raw" in msg and backend == "ggml":
        return "设计如此", ("ggml 是**图发射**，装不下任意外部调用"
                          "（进图要 custom op）—— 两层逃生舱在 ggml 上"
                          "没有对应的门，这是结构边界")
    return None, ""


def _doc_cells(text):
    """把生成文档里的**矩阵表**读回 `{(探针, 后端): 格子}`。

    只比矩阵格子、不比拒绝原文 —— 消息会随 torch/transformers 版本细微变化，
    那不是漂移；**格子变了才是漂移**（某后端从"接受"变成"拒绝"，或反过来）。
    """
    cells = {}
    for line in text.splitlines():
        if not line.startswith('| `'):
            continue
        parts = [p.strip() for p in line.strip('|').split('|')]
        if len(parts) != 2 + len(BACKEND_NAMES):
            continue                            # 拒绝原文表是 3 列，筛掉
        name = parts[0].strip('`')
        for b, cell in zip(BACKEND_NAMES, parts[2:]):
            cells[(name, b)] = cell
    return cells


def _compare_with_doc(rows, text):
    """**环境无关**的比对：只比本次量得出的格子。

    为什么不能整篇比：文档记录的是"依赖齐全时的实测"；而 CI 只装 numpy，
    `full` 列会是 `? 没验` —— 整篇比的话**CI 永远红**，那是判据自己造的假红。
    （实测：2026-10-16 第一版就是这么红的。）
    """
    want = _doc_cells(text)
    bad = []
    for name, _label, result in rows:
        for b in BACKEND_NAMES:
            ok, msg = result.get(b, (None, ''))
            if ok is None and str(msg).startswith('缺'):
                continue                        # 本机缺依赖：这一格本次没量
            measured = '✅' if ok is True else '✗' if ok is False else '—'
            got = want.get((name, b))
            if got is None:
                bad.append('%s × %s：文档里没有这一格' % (name, b))
            elif got != measured:
                bad.append('%s × %s：文档写 %s，实测 %s' % (name, b, got, measured))
    return bad


def main(argv=None):
    argv = list(_sys.argv[1:] if argv is None else argv)
    write = '--write' in argv
    print()
    print('=' * 78)
    print('  后端 × 机制矩阵 —— 承诺的公共子集有多大')
    print('=' * 78)
    print()
    print('  %-36s %s' % ('探针', ' '.join(BACKEND_NAMES)))
    rows, problems = measure()
    text = render(rows, problems)
    doc = _os.path.join(by1paths.ROOT, DOC)
    ir_common = sum(1 for r in rows
                    if all(r[2].get(b, (None, ''))[0] is True for b in IR_PATH))
    common = sum(1 for r in rows
                 if all(r[2].get(b, (None, ''))[0] is True for b in BACKEND_NAMES))
    print()
    print('  IR 路公共子集（codegen/c/ggml/exec）：%d / %d 个探针四后端全通'
          % (ir_common, len(rows)))
    print('  五后端交集（含 full）：%d / %d' % (common, len(rows)))
    print()
    if write:
        with open(doc, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(text)
        print('  写了 %s' % DOC)
    elif _os.path.exists(doc):
        with open(doc, encoding='utf-8') as fh:
            old = fh.read()
        # **比格子，不比整篇**（环境无关的判据，见 `_compare_with_doc`）。
        drift = _compare_with_doc(rows, old)
        if drift:
            print('  [FAIL] %s 与实测不一致 %d 格 —— 跑 `--write` 重新生成'
                  % (DOC, len(drift)))
            for d in drift[:8]:
                print('     · %s' % d)
            return 1
    if problems:
        print('  [FAIL] 协议问题 %d 条：' % len(problems))
        for p in problems:
            print('     · %s' % p)
        return 1
    print('  [PASS] 矩阵：%d 个探针 × %d 个后端，拒绝全部带理由，文档与实测一致'
          % (len(rows), len(BACKEND_NAMES)))
    return 0


if __name__ == '__main__':
    _sys.exit(main())
