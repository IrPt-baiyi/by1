#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1all -- 一次跑完全部验证，失败就非零退出。

存在的理由：在这之前，"护栏"是我一条条敲命令、肉眼看输出。
**没有 CI、没有测试运行器 —— 我停下来，就没人知道这个项目是不是还是好的。**

用法:  python by1all.py [--quick]
        --quick   跳过 C 后端（要调 gcc，慢）

**三种结论，不是两种**（协议见 `by1skip.py`）：

    ok     验过了
    skip   **这台机器上没验**（没有 gcc / 没有 CUDA / 缓存里没有那个模型）
           —— 不算失败，但也不是通过；报告里**单独列出来**
    fail   验了，不对

退出码: 0 = 没有失败   1 = 有失败
"""
import os
import re
import subprocess
import sys
import contextlib
# **副作用 import，别删**：它在 import 时把 stdout/stderr 钉成 UTF-8。
# 这个脚本是全量护栏的汇总口 —— 它自己那几行中文（`！！`、`跳过`）
# 在一个 cp936 的控制台上打不出来，而"打不出来"会被读成"没跑"。
import by1io          # noqa: F401
import by1paths
import by1refs as _refs
import by1skip

PY = sys.executable
# **找 gcc 的规则不在这里了** —— 它只有一个地方：`by1paths.find_gcc`。
# 原来这里有一份 `GCC_GLOBS`，by1extdemo 抄了第二份、by1e2e 抄了第三份，
# 而第三份抄漏了（只剩 WinGet 一条），症状是"整段 C 后端静默跳过"。
# （连那个空壳 `GCC_GLOBS = []` 也删了：一个"兼容旧引用"的变量，
#  在没有旧引用之后，只会让下一个人以为还有第二条路。）

#: 真实模型：(by1, **张量清单覆盖**, backend, 额外参数)
#
# **`config` 那一列删掉了。** 它从来没被读过一次 —— `main()` 里
# `for f, _cfg, _t, _b, _x in REAL` 那个下划线就是它。
# 而它**自己会烂**：15 条里 3 条指的是不存在的文件
# （`refs/poolside_Laguna-XS-2_1.config.json` 之类的旧名），
# 只是没人读它，所以没人发现。**一个没人读的字段不是"备用",
# 是一个迟早会说谎的字段。**
#
# **张量清单那一列留着，但只是"覆盖"。** 默认从 `.by1` 头部的
# `by1-repo` 推（`by1refs.paths`，那是路径规则的唯一真相源），
# 只有**推不出来**的才写在这里：
#
#     gpt-oss-120b 的两套命名（HF 侧 + GGUF 侧）
#     gemma-4-31B 只有 GGUF 侧
#
# 其余 8 条原来写的是 `refs/clef.tensors.json` / `refs/step37.tensors.json`
# 这种**早已不存在**的短名 —— 靠 `os.path.exists` 兜底才没炸。
# 删掉它们，`tensor_list()` 会把"兜底"换成"报错"。
REAL = [
    # gpt-oss 有两个侧：HF 的 `torch.module` 和 GGUF 的那一套命名。
    # 同一个 `.by1`，两份清单，两个后端 —— 这正是"能描述 ≠ 能算"的另一面：
    # **命名是发布决策，不是架构**，所以同一份描述对得上两套名字。
    ('gpt-oss-120b.by1', None, 'torch.module', []),
    ('gpt-oss-120b.by1', 'refs/openai__gpt-oss-120b.gguf-tensors.json',
     'ggml', []),
    ('gemma-4-31B.by1', 'refs/google__gemma-4-31B.gguf-tensors.json',
     'ggml', []),
    ('Laguna-XS-2_1.by1', None, 'torch.module', []),
    ('Instella-3B.by1', None, 'torch.module', []),
    # 剪枝版。和官方**同名不同源**，所以短名带 owner（见 models.tsv）。
    ('Step-3_7-Flash-180B-LynnStyle-GLM52-SFT-GPT55-RL.by1',
     None, 'torch.module', []),
    ('Ling-3.0-tiny.by1', None, 'torch.module', []),
    # **收敛的判卷人**：最普通的那种模型（标准 Qwen3 形状）。
    # 前面几个都是特意挑来压东西的，如果连这一个都要新属性，就是没收敛。
    ('minimind-3.by1', None, 'torch.module', []),
    # 第二个收敛判卷人，比 minimind 严格得多：48 层线性注意力 + 16 层全量，
    # 而这一整套 Qwen3-Next 时代就有了（GDN + 3+1 混合 + q_gate + qk_norm）。
    ('clef.by1', None, 'torch.module', []),
    # clef + MTP。MTP 是这一族里唯一的新东西，它逼出了三个语言改动：
    # `aux = true`（辅助栈不算解码层）、`name_<栈名>`（各栈物理前缀不同）、
    # 以及**栈内序号**（传全局层号会拼出 mtp.layers.64. 这种名字）。
    ('Qwen3.8-27B.by1', None, 'torch.module', []),
    # 同一个 Qwen3.5 形状换成 MoE。**零个新属性** —— MoE、共享专家、
    # 共享专家门控、MTP、线性注意力、3+1 混合，全是现成的。
    ('Qwen3.6-35B-A3B.by1', None, 'torch.module', []),
    # 官方 Step-3.7（未剪枝）—— 和剪枝版的差别就是被删掉的那几行。
    ('Step-3.7-Flash.by1', None, 'torch.module', []),
    # **语言的边界**：GPT-2 —— LayerNorm / 学习式位置编码 / 无门控 MLP，
    # 和前面十一个 Llama 家族是**两代人**。nanoGPT 是同一个架构。
    ('gpt2.by1', None, 'torch.module', []),
    # **唯一真正的新机制族：Mamba（选择性状态空间）。**
    # 顺带逼出两个改动：显式的逐层序列、按栈的专家名字模板。
    ('NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.by1', None,
     'torch.module', []),
    # **第 14 个真实模型 —— 而它原来是漏在护栏外面的那一个。**
    #
    # 一份 `.by1` 写在 `models/` 里、`models.tsv` 里也有、`1.md` 的表里
    # 还写了数字（"36289（97%）"），**但 by1all 从来没跑过它**。
    # 手动一跑：契约 37534 个张量里**缺 1245 个** —— 名字模板少了
    # `language_model.` 这一节，而专家张量走另一条模板（那条是对的），
    # 于是 36288 个专家张量全中、报告写着"97%"，**差的是别的一切**。
    #
    # 修好模板之后：37534/37534 全中、0 缺失。
    # 教训进 KNOWN 那条路是错的 —— 那是**描述写错了**，不是产物缺东西。
    ('GLM-5.3-Flash.by1', None, 'torch.module', []),
]

# 六个合成模型：三后端
SHAPED = ['llama-shaped.by1', 'mixtral-shaped.by1', 'gpt-oss-shaped.by1',
          'qwen3-next-shaped.by1', 'mla-shaped.by1', 'llama3-shaped.by1',
          # **由 clef.by1 机械缩小维度得来，结构一个字没改。**
          # 它证明"对着真 checkpoint 验过的那份描述"**同时是能跑的** ——
          # 不是两套东西。
          'clef-tiny.by1',
          # **无门控 FFN + LayerNorm + 学习式位置** —— 上一代的东西。
          # 别的 SHAPED 全是有门控 + silu，所以这一类从来没被三个后端
          # 一起对过 —— 而 `by1exec` 的 `op_ffn` 里就藏着写死的 `silu`
          # （还不认 bias）。它是怎么被发现的：改 C 后端让它支持无门控，
          # 然后拿 gpt2 本体对拍，差 3.9e-01。
          # 这个文件是让那条检查**留下来**：162M 参数的进不了 --quick。
          'gpt2-tiny.by1',
          # ── **最小例子也要回归。** ────────────────────────────────
          #
          # `hello.by1` 是语言的第一课（README 里就指着它）——
          # 而它**从来没被任何测试跑到过**。
          #
          # 用户提的"兼容性：你好"就是这个：**最小例子必须一直能跑。**
          # 一个语言最先坏的地方，永远是它最简单的那个例子 ——
          # 因为所有新功能都拿它试手，而没人回头跑它。
          'hello.by1']

# 三个判卷人脚本
JUDGES = ['by1mla.py', 'by1moe.py', 'by1rope.py',
          # **refs 语料**：每个文件都能解析、都有出处（by1 / models.tsv / SOURCES）。
          # 在这之前，"一个 refs 文件从哪来"从来没有被问过 ——
          # 于是里面躺着一个 29 字节的 `Invalid username or password.`。
          'by1refs.py',
          # **文档里写的路径还在吗。** 它原来没有判定行、也没有 `__main__`
          # 守卫（import 它就印一屏），所以既不能被判、也没人跑它。
          'by1docs.py',
          # **FILES.md 和目录树还一致吗。**
          # 那份文档是生成的，所以它不会过期 —— 但**有人忘了重新生成**
          # 就会。没有这一步的话，"生成了"只是个愿望。
          'by1files.py',
          # 前向，**真实维度**（14 个真实模型里唯一在这台机器上跑得动的）
          'by1instella.py',
          # 真实维度、官方 config 的前向 —— **另一代**
          # （LayerNorm / 学习式位置 / 无门控 GELU）
          'by1gpt2.py',
          # 逃生舱：三条断言（能建 / 缺 raw.py 必须拒 / 缺 impl 必须拒）
          'by1raw.py',
          # **三个后端只从 IR 跑** —— IR 是接口，那就得有一条路
          # 不经过 .by1 也能跑。它还带 JSON 往返。
          'by1irentry.py',
          # **两条路的 IR 对拍**：.by1 出来的 vs 从产物反推出来的
          'by1bootir.py',
          # 逃生舱第二层：IR 引用外部符号（.so + ABI）。
          # 它证的正是「加一个新机制不用改编译器」。
          'by1extdemo.py',
          # **逐算子**比 NumPy 和 PyTorch —— 整模型差的时候用它定位
          'by1opdiff.py',
          # **端到端：真产物 -> IR -> 三个后端 -> 对官方实现**
          # 没下载过 gpt2 的话它自己会跳过（返回 2）。
          'by1e2e.py',
          # 设备无关性：前向期间不带 device 的张量创建。
          # 这台机器没有显卡，所以验的是「显卡上能跑」的必要条件。
          'by1dev.py',
          # **显卡那半没验，要说出来。** 它在这台机器上走 by1skip
          # （没有 CUDA 设备）—— 于是每一轮报告里都印着
          # 「-- by1gpu.py [跳过] 没有可用的 CUDA 设备」，
          # 而不是像以前那样：这件事只存在于 README 的叙述里。
          'by1gpu.py']

fails, rows = [], []

# **子进程必须有个上限。** 一个挂住的生成程序（或一个等 stdin 的编译器）
# 会让整轮验证**永远停在那里** —— 而"永远停在那里"在日志里看起来
# 和"还在跑"一模一样。上限可以用 BY1_TIMEOUT 调。
TIMEOUT = int(os.environ.get('BY1_TIMEOUT', '600'))


def run(args, tag=None):
    """跑一个子进程。**不在这里记失败** —— 调用方才知道「非零退出」算不算失败
    （selftest.by1 就是必须有错的）。第一版两边都记，于是失败列表里出现重复。"""
    # **裸脚本名 -> src/ 下的真实路径。**
    # 这个文件里十几处调用写的都是裸名（`by1check.py` / `by1exec.py` …），
    # 让它们各自去拼 `src/` 就是把同一个意思写十几遍。
    if args and args[0].endswith('.py'):
        args = [by1paths.tool(args[0])] + list(args[1:])
    # **子进程的 cwd 固定成仓库根。** 生成物（`cgen-<模型>/`）落在这里，
    # 和搬家前一致；`.gitignore` 的 `cgen-*/` 指的也正是这里。
    #
    # **顺手把子进程的输出也钉成 UTF-8。** 判据是在**这里**解出来的
    # （`'逐字段' in out`），所以"子进程用什么编码"不能听凭系统代码页 ——
    # 中文 Windows 的管道默认是 cp936，于是子进程写的中文判定行到了这边
    # 变成 U+FFFD：一个退出码 0、判定正确的脚本被记成"没有判定行"。
    # 只动这一个变量，不碰 PATH、不碰别的环境。
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    try:
        r = subprocess.run([PY] + args, cwd=by1paths.ROOT, capture_output=True,
                           text=True, encoding='utf-8', errors='replace',
                           env=env, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        # **超时是失败，不是跳过。** 跳过是"这台机器上验不了"；
        # 挂住是"它坏了" —— 两者混起来，一个死循环会变成一条安静的缺口。
        return 'fail', ('（%d 秒没跑完 —— **超时**。这不是"没验"，'
                        '是它挂住了。）' % TIMEOUT)
    # **三态，不是布尔。** "退出码 ≠ 0" 里混着两类完全不同的东西：
    # 真的算错了，和"这台机器上验不了"。by1skip 用两条通道
    # （退出码 2 + `[跳过]` 标记）把它们分开 —— 两条通道不一致时取更保守的。
    out = (r.stdout or '') + (r.stderr or '')
    return by1skip.verdict(r.returncode, out), out


def _st(v):
    """行状态归一：布尔（老写法）和字符串都收。"""
    if v is True:
        return 'ok'
    if v is False:
        return 'fail'
    return v


def confirmed(st, out, line):
    """**输出通道能否决一个 0。**（判卷人的判据不能只有一条通道。）

    两种"绿得可疑"：

        · 退出码说通过，输出里却有一行 `[FAIL]`     -> 那是失败
        · 退出码说通过，输出里连一行判定都没有       -> 那是最坏的一种绿
          （它可能什么都没跑）

    这条规矩原来只写在 NumPy 那一段里（那次事故：一个脚本打印了 FAIL
    却 `return 0`，于是被当成通过）。现在它对**每一段**都成立。
    """
    if st != 'ok':
        return st
    if any('[FAIL]' in l for l in (out or '').splitlines()):
        return 'fail'
    return 'fail' if not line else 'ok'


def note_of(st, out, line, empty='（没有判定行）'):
    """行的备注：判定行优先；跳过就用它自己那行 `[跳过] …`。"""
    if line:
        return line[-1]
    if st == 'skip':
        for ln in (out or '').splitlines():
            if by1skip.MARK in ln:
                return ln.strip()
        return '（跳过，但没说为什么）'
    return empty


# 已知缺口：(那一项的名字前缀, 为什么)。**仍然打印出来**，只是不算失败 ——
# 藏起来的缺口和没发现过的缺口一样糟。
KNOWN = [
    ('tensors torch.module Step-3_7-Flash-180B-LynnStyle-GLM52-SFT-GPT55-RL.by1',
     '那 25 个张量在 HF 的 model-00009 分片里，而那个分片的头是全零 —— '
     '镜像的问题，不是 by1 的'),
    # **原来这里有一条 `C gpt2-tiny.by1`**（"C 还没有 LayerNorm 和学习式
    # 位置表"）。现在实现了，所以撤掉。
    #
    # 留着它有个好处：证明这个清单**会缩短**，不是只增不减的垃圾桶。
]


def tensor_list(f, override):
    """这个真实模型的张量清单在哪。**推不出来就报错，不静默跳过。**

    ## 为什么这不只是"少写几个字符串"

    这里原来是 `want = root(_t) if _t else ''` 然后
    `ten = want if os.path.exists(want) else _refs.paths(f, 'tensors')`
    —— 一份手写的路径，配一个静默兜底。于是：

      · 有人把 `refs/clef.tensors.json` 改名成
        `refs/Cloudflare__clef.tensors.json`，**手写的那 8 条全部过期**
      · `os.path.exists` 把过期变成了"换一条路走"，
        **没有任何输出说这件事发生过**
      · 而 `by1refs` 的 docstring 写着"refs 的路径规则，只此一处"

    现在：默认走 `by1refs`（唯一真相源），只有**推不出来**的才由
    `REAL` 覆盖；两条都拿不到就是**失败**，不是跳过。
    """
    if override:
        p = by1paths.root(override)
        if not os.path.exists(p):
            raise SystemExit(
                '  **%s 的张量清单覆盖写错了**：\n'
                '    REAL 里写的是 %s，而它不存在。\n'
                '    要么它是错的，要么文件被改名了 —— 两种都要人看一眼，'
                '不能悄悄换一条路走。' % (f, override))
        return p
    return _refs.paths(f, 'tensors')


def find_gcc():
    """**找 gcc 的规则只有一个地方**：`by1paths.find_gcc`。

    这里原来自己维护一份 `GCC_GLOBS` + `shutil.which` 的逻辑，
    `by1extdemo` 抄了第二份，`by1e2e` 抄了第三份（而它抄漏了，
    见 `by1paths.find_gcc` 的注释）。现在三处都调这一个。
    """
    return by1paths.find_gcc()


def run_many(pairs, jobs=None):
    """**并行跑一批独立命令。**

    量过：`--quick` 123 秒里，绝大部分是 **55 条子进程各自
    `import torch + transformers`**（4.7 秒/条）。
    计算本身不慢 —— 慢的是启动。

    判据一个字没改：还是**退出码 + 输出**（现在是三态：`by1skip` 认
    退出码 2 加输出里的 `[跳过]` 标记）。并行不放松任何一条规则，
    只是不再排队等 import。
    """
    import os as _os
    from concurrent.futures import ThreadPoolExecutor
    if jobs is None:
        jobs = int(_os.environ.get('BY1_JOBS', '0')) or min(6, (_os.cpu_count() or 2))
    if jobs <= 1 or len(pairs) <= 1:
        return [run(c, tag) for c, tag in pairs]
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(lambda pt: run(pt[0], pt[1]), pairs))


def main():
    # **先报版本。** 一份失败的输出要能追回是哪一版跑的。
    # 版本行是**锦上添花**：拿不到就少一行，不影响这次跑得对不对。
    # 用 `suppress` 而不是 `try/except/pass` —— 后者读起来像"没想好"。
    with contextlib.suppress(ImportError):
        from by1ver import version_line as _vl
        print('\n  ' + _vl())
    quick = '--quick' in sys.argv

    # ---- 1. 检查器：**逐个跑**才能归属到文件 ----
    # 一次跑全部的话输出里只有「摘要」行，分不清是哪个文件的 —— 第一版就是这么错的。
    # selftest.by1 是故意装错的反例（9 个错误），它不是"失败"。
    # **模型在 `models/`。** 取的是**裸名** —— 传给子进程和打印都用它，
    # 于是每一行的输出和搬家前逐字一致（判据就是靠这个能对上的）。
    specs = by1paths.names()
    bad = []
    # 覆盖缺口（"还没实现"，不是"算错了"）—— 走 KNOWN 那条路。
    gaps = []
    # **数出来的，不是减出来的。** 这里原来写 `len(specs) - 1`
    # （只减了 selftest），而循环里**跳过了两个**（selftest 和 gate-probe）
    # —— 于是报告印"检查器 25 个"，实际查了 24 个。
    # 一个"少算一个"的计数不会让谁崩，它只是**让这一行不能信**。
    n_checked = 0
    for f in specs:
        if f in ('selftest.by1', 'gate-probe.by1'):
            continue          # 两个故意的反例
        n_checked += 1
        _, out = run(['by1check.py', f], 'check ' + f)
        m = re.search(r'摘要:\s*(\d+)\s*错误\s*/\s*(\d+)\s*警告', out)
        e, w = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
        if e != 0:
            bad.append('%s(%d 错 %d 警)' % (f, e, w))
    rows.append(('检查器 %d 个 .by1' % n_checked,
                 'ok' if not bad else 'fail',
                 '除 selftest 外全 0 错' if not bad else '; '.join(bad)))
    if bad:
        fails.append('by1check')
    # selftest 单独确认：它**必须**报错，否则说明检查器坏了
    _, out = run(['by1check.py', 'selftest.by1'], 'selftest')
    m = re.search(r'摘要:\s*(\d+)\s*错误', out)
    n = int(m.group(1)) if m else 0
    rows.append(('selftest（反例，必须有错）', 'ok' if n > 0 else 'fail',
                 '报了 %d 个错误' % n))
    if n == 0:
        fails.append('selftest')

    # ---- 1.4 模式分类：**想象 vs 数据** ----
    #
    # `by1boot` 的识别词分两张表：数据里有的、和一次都没出现的。
    # 这个检查确认那张分类是对的 —— **分类一旦错了，
    # 把想象的放进已验证，它就永远不会被发现**（死模式不出声）。
    st, out = run(['by1pat.py'], 'patterns')
    line = [l.strip() for l in out.splitlines() if '分类' in l]
    st = confirmed(st, out, line)
    rows.append(('模式分类 by1pat.py', st, note_of(st, out, line)))
    if st == 'fail':
        fails.append('patterns')

    # ---- 1.5 神谕：**不依赖 by1 的期望值** ----
    #
    # 放在这里而不是最后：三后端互拍只能证明**自洽** ——
    # 而 pply_rope 的 docstring 里记着一次真实的事故：
    #
    #   > 曾经这里写成 np.stack（交错出）而 PyTorch 那边也写成 stack ——
    #   > **两边"一致地错"**，所以四个模型的 NumPy<->PyTorch 对拍全是绿的。
    #
    # 神谕验的是**对**。它慢一点，但它是唯一能抓到"一致地错"的东西。
    st, out = run(['by1oracles.py'], 'oracles')
    line = [l.strip() for l in out.splitlines() if '/ 10' in l]
    st = confirmed(st, out, line)
    rows.append(('神谕 by1oracles.py', st, note_of(st, out, line)))
    if st == 'fail':
        fails.append('oracles')

    # ---- 1.6 十类"不出声"的写法 + 它自己的自检 ----
    #
    # **`by1lint` 原来是个永远绿的检查。** 它 `return 0`，一条判定线都没有，
    # 而且**没有任何东西在跑它** —— 一个不存在的检查长什么样，
    # 它就是什么样。（它自己的 docstring 说"查十类"，而第⑧类
    # 根本没实现、规则⑨ 把 `capture_output=True` 当成"看过返回码"。）
    #
    # 现在它有一条线（十类必须 0 处）、有一个自检（**每条规则都要在
    # 反例上真的红一次**，外加两个不许误报的正例），而这两样都进护栏。
    st, out = run(['by1lint.py'], 'lint')
    line = [l.strip() for l in out.splitlines() if '[PASS]' in l
            or '[FAIL]' in l]
    st = confirmed(st, out, line)
    rows.append(('静态检查 by1lint.py', st, note_of(st, out, line)))
    if st == 'fail':
        fails.append('lint')

    st, out = run(['by1lint.py', '--selftest'], 'lint-selftest')
    line = [l.strip() for l in out.splitlines() if '[PASS]' in l
            or '[FAIL]' in l]
    st = confirmed(st, out, line)
    rows.append(('by1lint 自检（十条规则都要会红）', st, note_of(st, out, line)))
    if st == 'fail':
        fails.append('lint-selftest')

    # ---- 2. config 逐字段 ----
    #
    # **路径从 `.by1` 推，不用 REAL 清单里那两个字符串。**
    #
    # 那两个字符串会**各自过期**：我把 refs 统一命名之后，
    # 改名脚本知道，**但这里 27 处字符串不知道**。
    # 于是 11 个检查被 `if not os.path.exists(...)` **静默跳过** ——
    # `--quick` 从 55 项掉到 45 项，而报的是「45 项，0 项失败」。
    #
    # **跳过和通过，在输出里长得一样。那是最坏的一种绿。**
    #
    # 所以：路径由规则推（by1refs），**推不出来就报错**，不跳过。
    # （`by1refs` 现在在文件头 import —— `tensor_list()` 也要用它，
    #  而它是个模块级函数，看不见 main() 里的局部 import。）

    seen = set()
    derived_fail = []
    for f, _t, _b, _x in REAL:
        if f in seen:
            continue
        seen.add(f)
        cfg = _refs.paths(f, 'config')
        if cfg is None:
            if _refs.repo_of(f):
                derived_fail.append(f)
            continue                     # 合成模型没有 config，正常
        st, out = run(['by1verify.py', f, cfg, '--config'], 'config ' + f)
        line = [l.strip() for l in out.splitlines() if '逐字段' in l]
        st = confirmed(st, out, line)
        rows.append(('config ' + f, st,
                     note_of(st, out, line, empty='（没有 field 映射）')))
        if st == 'fail':
            fails.append('config ' + f)
    if derived_fail:
        # **不静默。** 有 by1-repo 却推不出 config，是引用坏了。
        print('  !! **%d 个模型的 refs 引用推不出来**（不是合成模型）：'
              % len(derived_fail))
        for f in sorted(set(derived_fail)):
            print('       %s' % f)
        fails.extend('refs ' + f for f in sorted(set(derived_fail)))

    # ---- 3. 张量名与形状 ----
    ten_fail = []
    for f, _t, backend, extra in REAL:
        cfg = _refs.paths(f, 'config')
        # **清单从 `by1refs` 推，不再手写。** ggml 那一路的文件名不同
        # （`.gguf-tensors.json`），那种由 REAL 的第二列**覆盖** ——
        # 覆盖也只写"文件名不同"这一种，因为它推不出来。
        # 覆盖写错、或者推不出来，都是**失败**，不是静默跳过（见 `tensor_list`）。
        ten = tensor_list(f, _t)
        if cfg is None or ten is None or not os.path.exists(ten):
            if _refs.repo_of(f):
                ten_fail.append((f, backend))
            continue
        st, out = run(['by1verify.py', f, cfg, '--tensors', ten,
                       '--backend', backend] + extra, 'tensors ' + f)
        line = [l.strip() for l in out.splitlines() if '契约声明存在' in l]
        st = confirmed(st, out, line)
        rows.append(('%s %s' % (backend, f), st,
                     note_of(st, out, line, empty='').replace('   ', ' ')))
        if st == 'fail':
            fails.append('tensors %s %s' % (backend, f))
    if ten_fail:
        print('  !! **%d 个张量检查找不到文件**（不是合成模型）：' % len(ten_fail))
        for f, b in ten_fail:
            print('       %s  [%s]' % (f, b))
        fails.extend('tensors-refs %s %s' % (f, b) for f, b in ten_fail)

    # ---- 4. 前向：参考实现 ----
    # **先收集，再并行，最后按顺序解析。**
    # 判据没变 —— 顺序只影响打印，不影响判定。
    #
    # **`hello.by1` 不在这里。** 这一步是拿**参考实现**（按 config 建的
    # 那份 HF 模型）对拍的 —— 而 `hello.by1` 是一个教学用的最小模型，
    # **没有 config，也没有参考实现**。
    #
    # 我一开始把它塞进 SHAPED，于是它掉进这一步，报
    # "前向 hello.by1 最大绝对差 1.756e-02" ——
    # **看起来像最小例子算错了**，其实是**拿它跟一个不存在的东西比**。
    #
    # 它该验的是"三个后端自洽"（第 5 步 `by1exec --compare`
    # 和后面的 C 后端），那不依赖任何外部参考。
    _todo = [f for f in SHAPED
             if f not in ('mla-shaped.by1', 'llama3-shaped.by1',
                          'clef-tiny.by1', 'gpt2-tiny.by1', 'hello.by1')]
    for f, (st, out) in zip(_todo, run_many(
            [(['by1diff.py', f], 'diff ' + f) for f in _todo])):
        line = [l.strip() for l in out.splitlines() if '最大绝对差' in l]
        st = confirmed(st, out, line)
        rows.append(('前向 ' + f, st,
                     note_of(st, out, line, empty='').replace('   ', ' ')))
        if st == 'fail':
            fails.append('diff ' + f)

    # ---- 5. 三后端 ----
    for f, (st, out) in zip(SHAPED, run_many(
            [(['by1exec.py', f, '--compare'], 'exec ' + f)
             for f in SHAPED])):
        line = [l.strip() for l in out.splitlines() if '最大绝对差' in l]
        # **"输出否决退出码"这条规矩现在在 `confirmed()` 里**，对每一段
        # 都成立，不再是这一段的特例。这里只负责把那次的备注写得更好看：
        if st == 'ok' and any('[FAIL]' in l for l in out.splitlines()):
            line = ['（退出码说 ok，但输出里是 FAIL —— 两个通道不一致）']
        st = confirmed(st, out, line)
        rows.append(('NumPy ' + f, st,
                     note_of(st, out, line, empty='').replace('   ', ' ')))
        if st == 'fail':
            fails.append('exec ' + f)

    # ---- 6. C 后端 ----
    gcc = find_gcc()
    if quick:
        # **这里原来写的是 True（通过）。** 整个 C 后端在 `--quick` 下是
        # "没验"，不是"验过了" —— 而 `gpu/README.md` 正是叫人**在租卡前**
        # 跑 `--quick`，于是那一步的失败会显示成一片绿。
        rows.append(('C 后端', 'skip', '--quick 跳过（要 gcc，慢）'))
    elif not gcc:
        # 缺 gcc = **这台机器上验不了**，不是"C 后端错了"。
        # 跳过会在报告里单列出来 —— 它没有被藏起来，只是不再冒充失败。
        rows.append(('C 后端', 'skip', '找不到 gcc —— C 后端这一整段没验'))
    else:
        for f, (st, out) in zip(SHAPED, run_many(
                [(['by1c.py', f, '--gcc', gcc, '--seq', '16',
                   # **每个模型一个工作目录。**
                   # 这一条是并行化带出来的 bug：8 个 by1c 同时跑，
                   # 共用默认的 `cgen/` —— 互相覆盖 model.c / w.bin /
                   # ids.bin / model.exe。
                   # Windows 上恰好没撞上（I/O 慢），Linux 上一撞就全错，
                   # 而症状是"C 后端算错了 6 个模型"——**看起来像 C 的 bug**。
                   # 单独跑一个模型永远是对的，这最误导。
                   '--workdir', 'cgen-' + f.replace('.by1', '')],
                  'C ' + f) for f in SHAPED])):

            line = [l.strip() for l in out.splitlines() if '最大绝对差' in l]
            # **分清「没实现」和「算错了」。**
            # 前者是覆盖率缺口（已知、可数），后者是 bug。
            # 混在一起的话，一个是"还没做"、一个是"做错了"，
            # 却长得一样 —— 而真问题会被覆盖率噪音淹掉。
            # （by1irentry 里是同一条规矩。）
            if st == 'fail' and '[不支持]' in out:
                miss = [l.strip() for l in out.splitlines()
                        if '[不支持]' in l]
                # 走 KNOWN 那条路（按前缀匹配），不算失败。
                gaps.append('C %s' % f)
                continue
            st = confirmed(st, out, line)
            rows.append(('C ' + f, st,
                         note_of(st, out, line, empty='').replace('   ', ' ')))
            if st == 'fail':
                fails.append('C ' + f)

    # ---- 7. 取值门 —— **可证伪对照** ----
    # 「声明了一个 codegen 没实现的取值，必须被拒」这条规则本身要被验。
    # `gate-probe.by1` 是故意的反例（act 是个不存在的取值）；
    # nemotron / ling 是整族没实现。三个都必须被拒，三个已实现的必须通过。
    # 门坏了比没有门更糟 —— 它给人虚假的安心。
    #
    # （判据看它自己的判定行，不看某个具体字样：2026-10 把 gelu 实现之后，
    #   GPT-2 从"必须被拒"变成"必须通过"，而这个脚本立刻红了 ——
    #   那是它该干的事，但这里的判据不该绑死在某个模型的某个取值上。）
    st, out = run(['by1gate.py'], 'gate')
    line = [l.strip() for l in out.splitlines() if '[PASS]' in l
            or '[FAIL]' in l]
    st = confirmed(st, out, line)
    note = line[-1] if line else (out.strip().splitlines()[-1][:70]
                                  if out.strip() else '（没输出）')
    rows.append(('取值门（三个反例 + 三个正例）', st, note))
    if st == 'fail':
        fails.append('gate')

    # ---- 8. 判卷人脚本 ----
    # **15 条判卷人，各跑各的，互不依赖，所以并行。**
    # （这里原来写"13 条，74 秒" —— 而 `JUDGES` 是 15 条。
    #  一个数写进注释就没人再数它了；所以要写就写能一眼数出来的。）
    for s, (st, out) in zip(JUDGES, run_many([([s], s) for s in JUDGES])):
        line = [l.strip() for l in out.splitlines() if '[PASS]' in l
                or '[FAIL]' in l]
        st = confirmed(st, out, line)
        rows.append((s, st, note_of(st, out, line)))
        if st == 'fail':
            fails.append(s)

    # ---- 输出 ----
    print('=' * 78)
    print('  by1 全量验证')
    print('=' * 78)
    for name, st, note in rows:
        print('%s%-30s %s' % (by1skip.mark(_st(st)), name, note[:80]))
    print('-' * 78)
    known, real = [], []
    for f in fails + gaps:
        hit = next((why for pre, why in KNOWN if f.startswith(pre)), None)
        (known if hit else real).append((f, hit))
    # **跳过单独列。** 它不算失败 —— 但它也不是通过。
    # 藏起来的话，读者会把"总数 − 失败数"读成"验过的数"。
    skipped = [(n, note) for n, st, note in rows if _st(st) == 'skip']
    if skipped:
        print('  跳过（**没验** —— 不算失败，但也不是通过）:')
        for n, note in skipped:
            print('    - %-30s %s' % (n, note[:70]))
    if known:
        print('  已知缺口（不算失败，但仍然存在）:')
        for f, why in known:
            print('    - %s\n      %s' % (f, why))
    print('  %d 项，%d 项失败%s%s'
          % (len(rows), len(real),
             ('，%d 项跳过' % len(skipped)) if skipped else '',
             ('，另有 %d 项已知缺口' % len(known)) if known else ''))
    if real:
        print('  失败: %s' % ', '.join(f for f, _ in real))
    if real:
        verdict = '有失败'
    elif skipped:
        # **带着 3 项"没验"还说"全过"，那就是骗人。** 跳过不拦退出码，
        # 但它必须出现在这最后一行里。
        verdict = '全过（%d 项没验）' % len(skipped)
    else:
        verdict = '全过'
    print('  [%s]' % verdict)
    return 0 if not real else 1


if __name__ == '__main__':
    sys.exit(main())
