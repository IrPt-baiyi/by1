#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1pack -- 打出要上传到云端的那一包。

**只装"描述 + 判卷人 + 工具"，不装权重和可重新下载的东西。**

    不装：模型权重（在 GPU 机器上从 HF 下）· llamacpp/ 里的参考源码
          drafts/（没有判卷人的草稿）· cgen/ __pycache__

用法:  python src/by1pack.py [输出目录]
       默认输出 ../by1-upload/

## 搬家之后

包里的目录结构和仓库**一样**（`src/` `models/` `refs/` + 根下的文档）。
原来是把所有东西摊在包的根目录，而 gpu/ 那一层又是平铺进来的 ——
于是打包完印出来的那句 `bash gpu/run.sh` 指的路径**根本不存在**。
现在包里长什么样、仓库里就长什么样。
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
import os
import shutil
import sys
import tarfile
import time

import by1io
import by1paths

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = by1paths.ROOT
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(ROOT), 'by1-upload')

# 要装的东西。每条是 (来源, 说明) —— **来源写裸名**，"它在哪个目录"
# 由 `_where()` 一处决定。原来这里把 30 个脚本名和 20 个文档名写在一起、
# 还各自带目录前缀，搬一次家就要改 50 行。
WANT = [
    ('by1.py', '入口'),
    ('by1check.py', '不变式检查器（解析 + 契约 + 等价类）'),
    ('by1codegen.py', '后端一：PyTorch nn.Module'),
    ('by1exec.py', '后端二：纯 NumPy'),
    ('by1c.py', '后端三：生成 C，编译，运行'),
    ('by1verify.py', '产物对拍：config / 张量名与形状'),
    ('by1diff.py', '对 transformers 的前向与反向'),
    ('by1mem.py', '内存计划'),
    ('by1emit.py', 'ggml 建图与 GGUF manifest'),
    ('by1oracle.py', '状态尺寸神谕'),
    ('by1train.py', '训练脚手架'),
    ('by1mla.py', '判卷人：MLA'),
    ('by1moe.py', '判卷人：noaux_tc 分组路由'),
    ('by1rope.py', '判卷人：llama3 RoPE'),
    ('by1all.py', '一次跑完全部验证'),
    ('by1instella.py', '真实模型前向：Instella-3B，**真实维度**'),
    ('by1pack.py', '打这一包'),
    ('by1contract.py', '从机制声明推出张量契约'),
    ('by1boot.py', '从产物反推一份 .by1 草稿'),
    ('by1gate.py', '取值门的可证伪对照'),
    ('by1raw.py', '逃生舱的判卷人'),
    ('by1ir.py', 'IR 规格 + 校验 + JSON 往返（**规格的唯一真相源**）'),
    ('by1irentry.py', '三个后端只从 IR 跑'),
    ('by1bootir.py', '两条路的 IR 对拍（.by1 vs 产物）'),
    ('by1ext.py', '逃生舱第二层：外部符号的加载与 ABI'),
    ('by1extdemo.py', '逃生舱第二层的判卷人'),
    ('by1opdiff.py', '逐算子比 NumPy 和 PyTorch'),
    ('by1e2e.py', '**端到端**：真产物 -> IR -> 三个后端 -> 对官方实现'),
    ('by1dev.py', '设备无关性（显卡上能跑的必要条件）'),
    # **`by1lint.py` 是 `by1all` 的判卷人之一（两条：本体 + 自检）。**
    # 它进来的时候我漏了这一行 —— 而 `closure_missing()` **看不见**它：
    # 那个检查算的是 import 闭包，而 `by1lint` 是被**当脚本调用**的，
    # 没人 import 它。**一个判卷人不在包里 = 那一项在云端变成"文件不存在"**，
    # 而症状长得像"这一包本身有问题"（gpu/run.sh 第一行就会停）。
    ('by1lint.py', '"不出声"的写法 + 自检（by1all 的判卷人）'),
    # **这四个也是 `by1all` 当脚本调的，而清单里一直没有。**
    # 加 `judges_missing()` 那一条检查时它一次报出四个 ——
    # 也就是说"上一包"在云端跑到这几项时会报"文件不存在"，
    # 而那个症状是"这一包本身有问题"（gpu/run.sh 第一行就停）。
    ('by1pat.py', '模式分类：说已验证的都有命中，说未验证的都是 0'),
    ('by1oracles.py', '十个神谕：不依赖 by1 的期望值'),
    ('by1coerce.py', '取值写法：as_bool/as_num 的四关判据'),
    ('by1docs.py', '文档里的路径**和数字**'),
    ('by1gpt2.py', '真实维度前向：GPT-2（另一代）'),
    # **2026-10-16：`judges_missing()` 修好之后，它一次报出这 9 个。**
    # 那个守卫原来硬拼 `src/by1all.py`（而那文件早就搬进 `src/checks/`），
    # 路径不存在就**静默返回 []** —— 于是"守卫在看着"是错觉，
    # 包里少了这 9 个判卷人，而打包脚本一声不吭。症状要到云端跑门禁才现形。
    ('by1fast.py', '快速检查：编译/import/解析/行尾（档 1 的判卷人）'),
    ('by1files.py', 'FILES.md 与目录树一致（生成的文档也会过期）'),
    ('by1mutate.py', '变异反例库：`by1ir.validate` 每条规则一个必红反例'),
    ('by1sample.py', '采样链逐 token 判卷人（对 sampler-probe）'),
    ('by1complete.py', '完整模型：前向/反向 + config 逐字段 + 负例'),
    ('by1ssm.py', '判卷人：Mamba2 / Nemotron-H mixer 的递推'),
    ('by1kda.py', '判卷人：KDA（Kimi Delta Attention）'),
    ('by1sparse.py', '判卷人：DSA 稀疏索引器（对 `GlmMoeDsaIndexer`）'),
    ('by1gptoss.py', '判卷人：注意力 sink（对 `GptOssAttention`）'),
    ('by1ggml.py', '后端四：生成 ggml C（并按 manifest 对拍）'),
    ('by1llama.py', '产物线：GGUF 构建 / 上游 logits 逐位对拍 / 转换期改写'),
    ('by1quant.py', '量化格式声明（MXFP4 / FP8）的数值 oracle 与运行时规格'),
    ('by1q2schema.py', 'Q2 报告的 schema：指标与容差写在一处（分片量化的口径）'),
    ('by1rules.py', '转换期改写表的判卷人（每条规则的上游出处 + 探针）'),
    ('by1backends.py', '后端 × 机制矩阵（承诺的公共子集有多大，文档是生成物）'),
    # **下面三个是"文档叫你跑，而包里没有"的那一类。**
    # `by1docs.py` 在包里自己把它们抓出来了 —— 它扫 README 里
    # `python xxx.py` 的调用，而包里没有这些文件。
    # 也就是说：**新读者照 README 敲的第一条命令就会 FileNotFoundError**，
    # 而那个报错长得像"这一包坏了"。
    #
    # 这三条是"打包清单"和"文档"之间的缝 —— 两边都没有引用的地方，
    # 所以谁也没发现。判据：文档里提到的脚本，包里有；反之亦然的那些
    # 由 `judges_missing()` 守。
    ('by1fetch.py', 'refs/ 抓取（README 的"数据从哪来"第一条命令）'),
    ('by1gpu.py', '显卡/显存规划（GPU runbook §3 的第一条命令）'),
    ('by1real.py', '真权重真维度前向（GPU runbook §3 的最后一条命令）'),
    ('by1refs.py', 'refs/ 的路径规则'),
    ('by1io.py', '文件读写的每一种意图'),
    # **`by1skip.py` 原来不在这个清单里，而 8 个已打包的模块在模块级
    # `import by1skip`** —— 于是包打出来了、`gpu/run.sh` 第一行
    # （`src/by1all.py --quick`）就 `ModuleNotFoundError: No module named
    # 'by1skip'`。**这个包从来没能跑起来过**，而打包脚本一声不吭。
    # 下面的 `closure_missing()` 就是这条的判卷人：以后少一个就红。
    ('by1skip.py', '三种结论（ok / 跳过 / 失败）的协议'),
    ('by1paths.py', '仓库布局的唯一真相源'),
    ('by1ver.py', '版本号的唯一真相源'),
    ('by1name.py', '改名：models.tsv <-> 磁盘'),
    # **语言层：`by1check` 在模块级 import 它们，而清单里一直没有。**
    # 这 8 个是 `closure_missing()` 修好路径 bug 之后一次报出来的 ——
    # 也就是说"上一包"（以及之前每一包）在云端跑到 by1check 时
    # 就是 ModuleNotFoundError。modules 拆进 `src/lang/` 子目录时
    # 漏了它们；WANT 写裸名、`_where()` 认子目录，所以补清单即可。
    ('by1blocks.py', '块语句（stack / pattern）的展开'),
    ('by1export.py', '.by1 -> 各框架 config 的导出（判卷人之一）'),
    ('by1hp.py', 'hparams 的读取与别名'),
    ('by1lower.py', '机制声明 -> 层算子序列'),
    ('by1sched.py', '层的调度（layer_types）'),
    ('by1stacks.py', '栈与逐层覆盖'),
    ('by1state.py', 'state 块（KV/SSM 状态声明）'),
    ('by1tens.py', '张量名与形状的推导'),
    # **closure_missing() 的三笔账**（2026-10-14）：`by1lower` 现在模块级
    # import `by1quant`（Q1 的量化校验）、`by1exec` 要 `by1graph.rope_inv`、
    # 完整后端链要 `by1full` —— 传递边长出来了，清单没跟上，包打出来
    # 就是 ModuleNotFoundError。少一个就红，靠的就是 closure_missing()。
    ('by1full.py', '完整架构 -> Transformers 模型（视觉 / MTP / 量化装载）'),
    ('by1graph.py', '原语图：IR -> 规范化原语（语义的唯一定义 + 图求值）'),
    ('by1quant.py', '量化 checkpoint 的存储契约与 manifest 审计'),
    ('gpt2-tiny.by1', '上一代的缩小版（无门控 FFN + LayerNorm + 查表位置）'),
    ('ext-demo.c', '示例外部机制：两个符号，证「不用改编译器」'),
    ('raw.py', '逃生舱的示例实现（配 raw-escape.by1）'),
    ('gate-probe.by1', '取值门的反例（故意写错，别修）'),
    ('ir-spec.md', 'IR 规格（从 src/by1ir.py 生成）'),
    ('README.md', '门面'),
    ('pyproject.toml', '装得上'),
    ('VERSION', '版本的投影（打包读它，不导入模块）'),
    # **`models.tsv` 必须进包。** 它是"长名 / 短名 / HF repo"的唯一真相源，
    # 也是 `by1refs` 判 **refs 出处**的来源之一（`.by1` 头部没有
    # `# by1-repo:` 的那些，就靠它）。
    # 实测（2026-10-16）：漏了它 -> 包里 `by1refs` 报 "models.tsv 0"，
    # 于是 granite-350m/1b 与 Falcon-H1 的 6 个 refs 文件"没有出处" ->
    # **档 1 在包里红** —— 而 CI 的发布包 job 正是"在包里跑档 1"。
    ('models.tsv', '模型清单：长名/短名/HF repo 的唯一真相源 + refs 出处来源'),
    ('1.md', '项目说明（宣言）'),
    ('history/ir.md', '开发日志'),
    ('history/plan.md', '2026-10-08 的计划'),
    ('docs/INVENTORY.txt', '验证清单：每个模型验到了什么'),
    ('.gitignore', ''),
    # **`.gitattributes` 也要进包。** 它是那份"文本一律 LF"的规则 ——
    # 拿到包的人在**那棵树里**继续改动时，规则要跟着走；
    # 不然他本地一提交，行尾又乱回去（原来就没这个文件，见它的注释）。
    ('.gitattributes', '行尾规范：文本一律 LF'),
    # **2026-10-16：包里少的东西不是脚本，是文档和许可证。**
    # 症状：照 README 敲第一条命令就 FileNotFoundError ——
    # 而 README 自己就在包里，指着 13 个不存在的文件。
    # 这些是 `selfcheck()`（打完包在产物里跑 `by1docs`）一次报出来的。
    ('TESTING.md', '怎么自己跑一遍（README 指向它）'),
    ('FILES.md', '每个文件是干什么的（生成的）'),
    ('LICENSE', 'GPL-3.0'),
    ('NOTICE', '第三方来源与许可'),
    ('graphs-golden.json', '原语图 golden 快照（A3 的护栏，档 1 判卷人要它）'),
    ('models.md', '模型清单：机制与出处的对照表（`docs/by1-language-spec.md` 指向它）'),
    ('by1langspec.py', '语言规格：写不出名字的机制必须被拒绝（文档引用它）'),
    ('by1sync.py', '把各处的数字与清单对齐（GPU 手册引用它）'),
    # **`docs/` 整层由下面的循环装**（见 `main()` 里那一段）——
    # 不再一个一个往这里写。2026-10-16 实测：手写清单漏了 13 份文档，
    # 而新增一份指南又会引出新的"链接指向不在包里的文档"。
    # 判据是 `selfcheck()`：包里的文档链接得到的东西必须在包里。
    ('by1cache.py', 'HuggingFace 缓存/存储的搬迁与审计（AutoDL 相关）'),
    ('by1triage.py', '分诊：一次门禁里哪些是真红、哪些是没验'),
    ('by1gpubench.py', 'CUDA 前向/反向基准（gpu/README 的第一条命令）'),
    ('by1gpuday.py', 'GPU 日的一键脚本'),
    ('by1gpusmoke.py', '显卡冒烟：装好了吗'),
    ('by1gputrain.py', 'GPU 训练步（含 CPU 回退）'),
    ('by1shadow.py', '影子对拍：同一份声明走两条 loader'),
]

# refs/ 全要（官方产物，判卷人的原材料）
# models/*.by1 全要（不含 drafts/）


def _where(name):
    """清单里的**裸名** -> 仓库根相对的来源路径。

    **规则只写这一处。** `.by1`（还有 `raw.py`）在 `models/`，
    代码在 `src/` **或其子目录**（子目录清单在 `by1paths.SUBDIRS`，
    也只此一处），其余按写的来。

    原来这里只会拼 `src/ + name` —— 而 41 个模块里 29 个在子目录里
    （`checks/` `modelcheck/` `lang/` `data/` `escape/`），于是打包自检
    永远报"清单里写了、磁盘上没有"，`by1pack` 必 exit 1。
    """
    if name.endswith('.by1') or name == 'raw.py':
        return 'models/' + name
    if name.endswith(('.py', '.c')):
        for sub in ('',) + tuple(by1paths.SUBDIRS):
            rel = 'src/' + (sub + '/' if sub else '') + name
            if os.path.exists(norm(rel)):
                return rel
        return 'src/' + name      # 兜底：让 absent 报出来，而不是静默错位
    return name


def disk_modules():
    """`src/` 下所有模块的裸名（**含子目录** —— 41 个模块里 29 个在那）。"""
    import glob as _g
    out = set()
    for pat in ('src/*.py',) + tuple(
            'src/%s/*.py' % s for s in by1paths.SUBDIRS):
        out.update(os.path.basename(p) for p in _g.glob(norm(pat)))
    return out


def closure_missing():
    """**WANT 的传递依赖是不是都装进去了。**

    这是"包打出来却起不来"那一类 bug 的判卷人。

    在这之前，`WANT` 是一张手写清单，少写一个模块的后果是：
    包照常打出来、体积照常报、`gpu/run.sh` 第一行就 `ModuleNotFoundError`
    —— 而**打包脚本什么都不说**（`if not os.path.exists(s): continue`）。
    一个不会失败的检查不是检查；一个不会报错的清单不是清单。

    判据：把 WANT 里每个 `.py` 的**模块级 import**（AST 取，不用正则）
    连成闭包；闭包里凡是在 `src/` 下、却没进 WANT 的，都是缺口。
    """
    import ast
    disk = disk_modules()
    want = {w[0] for w in WANT if w[0].endswith('.py')} & disk
    graph = {}
    for f in sorted(want):
        try:
            # 路径走 `_where()` —— 模块散在 `src/` 和 5 个子目录里，
            # 硬拼 `src/ + f` 会在第一个子目录模块上 FileNotFoundError。
            tree = ast.parse(by1io.read_text(norm(_where(f)), encoding='utf-8'))
        except (SyntaxError, OSError):
            graph[f] = set()
            continue
        deps = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                mods = [n.module or '']
            else:
                continue
            for m in mods:
                base = m.split('.')[0] + '.py'
                if base in disk:
                    deps.add(base)
        graph[f] = deps
    need, frontier = set(want), set(want)
    while frontier:
        nxt = set()
        for f in frontier:
            for d in graph.get(f, ()):
                if d not in need:
                    need.add(d)
                    nxt.add(d)
        frontier = nxt
    return sorted(need - want)


def judges_missing():
    """`by1all` **当脚本调用**的那些 `.py`，有没有漏出这个包。

    ## 为什么 import 闭包不够

    `closure_missing()` 算的是"谁 import 谁" —— 而护栏里有一半的东西
    是**被 subprocess 调用**的（`by1all.run(['by1lint.py'])`），
    没有任何模块 import 它们。于是：

        `by1lint.py` 成了 `by1all` 的判卷人，而它没进 `WANT`
        -> `closure_missing()` **一声不吭**
        -> 到了云端那一项变成"文件不存在"，
           而症状长得像"这一包本身有问题"

    这就是"两个清单各说各话"的老毛病。所以这里**从 `by1all` 自己读**：
    `JUDGES` 加那几个 `run(['xxx.py'])` 的调用点。
    它是 `by1all` 的一部分，改了这边就跟着改。
    """
    import re as _re
    # **路径要用 `_where()` 解，不能硬拼 `src/`。**
    # 实测（2026-10-16）：这一行原来写 `norm('src/by1all.py')`，而
    # `by1all.py` 早就搬到了 `src/checks/` —— 于是 `os.path.exists` 为假、
    # 函数**静默返回 []**：守卫看起来在看着，其实一直是死的。
    # 后果：包里少了 9 个 `by1all` 会调用的判卷人（`by1fast`/`by1files`/
    # `by1mutate`/`by1sample`/`by1complete`/`by1ssm`/`by1kda`/`by1sparse`/
    # `by1ggml`），而打包脚本一声不吭 —— 症状要到云端跑门禁才现形
    # （"文件不存在"，长得像"这一包本身有问题"）。
    p = norm(_where('by1all.py'))
    if not os.path.exists(p):
        # **找不到就大声说。** 静默返回 [] 正是上面那个坑。
        raise SystemExit('  [FAIL] 判卷人清单解析不了：找不到 %s' % p)
    src = by1io.read_text(p, encoding='utf-8')
    names = set()
    # `JUDGES = [...]` 里的裸名
    m = _re.search(r'^JUDGES\s*=\s*\[(.*?)\]', src, _re.S | _re.M)
    if m:
        names |= set(_re.findall(r"'([\w.]+\.py)'", m.group(1)))
    # `run(['xxx.py', ...])` 这类调用点
    names |= set(_re.findall(r"run\(\[\s*'([\w.]+\.py)'", src))
    want = {w[0] for w in WANT}
    return sorted(n for n in names if n not in want and n in disk_modules())


def norm(p):
    return os.path.join(ROOT, p)


def selfcheck(staged):
    """在**刚打出来的产物**里跑**档 1**：包自己得站得住。

    绿 -> 返回 None；红 -> 返回截断后的报告（调用方负责打印并失败）。

    ## 为什么是"跑档 1"，而不是只跑一个检查器

    原来这里只跑 `by1docs`（查"文档指向的东西在不在"）。**那不够**：实测
    （2026-10-16）包里漏了 `models.tsv` —— 文档一处都没引用它，`by1docs` 全绿，
    可 `by1refs` 因此判 6 个 refs 文件"没有出处"，于是**包里的档 1 红**。
    而 CI 的发布包 job 干的就是"在包里跑档 1"。

    所以判据直接取**那一件事本身**：在产物里跑 `by1all --tier 1`。
    它也是"包能不能用"的最短定义 —— 而且只要标准库，打包脚本本来就只依赖标准库。
    """
    import subprocess
    # **必须绝对化**：下面 `cwd=staged` 一换，相对路径就会再拼一层
    # （实测报的是 `build/by1-pack/build/by1-pack/src/checks/by1docs.py`）。
    staged = os.path.abspath(staged)
    script = os.path.join(staged, 'src', 'checks', 'by1all.py')
    if not os.path.exists(script):
        return '  产物里没有 src/checks/by1all.py —— 自检根本跑不了（这本身就是缺口）'
    try:
        r = subprocess.run([sys.executable, script, '--tier', '1'], cwd=staged,
                           capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=1800)
    except Exception as exc:                     # noqa: BLE001 - 打包要报出来，不要吞
        return '  自检跑不起来：%s' % exc
    if r.returncode == 0 and '[全过' in (r.stdout or ''):
        return None
    # 两个通道都看：退出码（进程活下来了吗）与判定行（检查器自己怎么说）。
    tail = ((r.stdout or '') + (r.stderr or '')).strip().splitlines()
    return '  自检退出码 %d（在产物里跑 `by1all --tier 1`）\n%s' % (
        r.returncode, '\n'.join(tail[-25:]))


def main():
    # **先验清单，再动磁盘。** 校验失败时不该留下一个半成品目录
    # （调用方看到的应该是"没产出"，不是"产出了一份坏的"）。
    absent = [_where(name) for name, _why in WANT
              if not os.path.exists(norm(_where(name)))]
    gaps = closure_missing()
    judges = judges_missing()
    if absent or gaps or judges:
        print('=' * 70)
        print('  **打包失败 —— 这一包跑不起来**')
        print('=' * 70)
        if absent:
            print('  清单里写了、磁盘上没有（%d 个）：' % len(absent))
            for r in absent:
                print('    · %s' % r)
        if gaps:
            print('  清单漏了传递依赖（%d 个）—— 装进包里才能 import：'
                  % len(gaps))
            for g in gaps:
                print('    · %s' % g)
        if judges:
            print('  清单漏了 `by1all` 会调用的判卷人（%d 个）——'
                  ' 它们不是被 import 的，所以闭包查不到：' % len(judges))
            for j in judges:
                print('    · %s' % j)
        print()
        print('  清单在 %s 的 `WANT` 里。'
              % os.path.relpath(__file__, ROOT))
        return 1

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    n, total = 0, 0

    def put(src_abs, rel):
        """拷进包里，**保持仓库里的相对路径**。"""
        nonlocal n, total
        d = os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(src_abs, d)
        n += 1
        total += os.path.getsize(d)

    # 工具与文档（上面已经验过"清单里的东西都在"，所以这里不会静默跳过）
    for name, _why in WANT:
        rel = _where(name)
        put(norm(rel), rel)

    # **`src/` 下的模块一律补齐。**
    #
    # `WANT` 是手写的"有说明的核心清单"，而**文档可能提到任何一个模块** ——
    # "包缺脚本"这一类在这个仓库里反复发生（2026-10-16 的判卷人那一批一次
    # 报出 9 个，随后文档引用又逐个报出 `by1langspec` / `by1sync` / `by1cmp` …）。
    # 每次补一个名字，就是再等下一次漏。所以这里收口：
    # **凡是 `src/` 下存在的模块，包里都得有** —— 它们加起来才几百 KB。
    # `WANT` 仍然有用：它是"每个文件是干什么的"那份**说明**，
    # 也是 `closure_missing()` / `judges_missing()` 的判据来源。
    staged = {_where(name) for name, _why in WANT}
    for bare in sorted(disk_modules()):
        rel = _where(bare)
        if rel not in staged:
            put(norm(rel), rel)

    # models/ 的 .by1 描述
    for s in sorted(glob.glob(norm('models/*.by1'))):
        put(s, 'models/' + os.path.basename(s))

    # refs/（官方产物）
    for s in sorted(glob.glob(norm('refs/*'))):
        if os.path.isfile(s):
            put(s, 'refs/' + os.path.basename(s))

    # gpu/（云端专属：runbook + 依赖 + 需要 CUDA 的检查）
    gsrc = norm('gpu')
    if os.path.isdir(gsrc):
        for f in sorted(os.listdir(gsrc)):
            s = os.path.join(gsrc, f)
            if os.path.isfile(s):
                put(s, 'gpu/' + f)

    # drafts/（**草稿也要装**）：`docs/` 里的工作单引用它们
    # （例如 `docs/step5-contract-plan.md` 指向 `drafts/step5-schedule-probe.by1`）
    # —— 不装的话，拿到包的人照文档点链接就是 404。
    # 它们不是"没有判卷人的草稿就不发"，而是"文档引用得到的东西必须在包里"。
    for s in sorted(glob.glob(norm('drafts/*'))):
        if os.path.isfile(s):
            put(s, 'drafts/' + os.path.basename(s))

    # history/external-repro/（第三方复现报告；README 里指向这个目录）
    for s in sorted(glob.glob(norm('history/external-repro/*'))):
        if os.path.isfile(s):
            put(s, 'history/external-repro/' + os.path.basename(s))

    # docs/ 整层（含 .md 与 .txt）：**文档之间的互链必须能落地**。
    # 手写清单在这件事上漏过 13 份，而"新写一份指南"又会引出新的缺口 ——
    # 所以这里整层装，另外由 `selfcheck()` 兜底（包里的文档链接得到的东西
    # 缺一个就打包失败）。
    for s in sorted(glob.glob(norm('docs/*'))):
        if os.path.isfile(s):
            put(s, 'docs/' + os.path.basename(s))

    # docs/gpu-results/（GPU 轮次的原始结果；`docs/gpu-validation-results.md`
    # 逐个链接它们 —— 不装就是 12 处 404）
    for s in sorted(glob.glob(norm('docs/gpu-results/*'))):
        if os.path.isfile(s):
            put(s, 'docs/gpu-results/' + os.path.basename(s))

    # ── 把"外部产物"名单写进包里 ────────────────────────────────
    #
    # 包里不是 git 检出，所以它没法自己回答"这个缺失的路径是不是**故意不在库**"。
    # 在这里（仓库里、git 可用）问一次 `by1docs`，把那桶路径写进包根的
    # `.by1-ignored`；包里的 `by1docs` 读它来分类，而不是把
    # `history/_hf_probe/*.py` 这种被 `.gitignore` 挡着的引用判成"缺文件"。
    import json as _json
    import subprocess
    try:
        r = subprocess.run([sys.executable, norm('src/checks/by1docs.py'),
                            '--external-json'], cwd=ROOT, capture_output=True,
                           text=True, encoding='utf-8', errors='replace', timeout=900)
        lines = [x for x in (r.stdout or '').splitlines() if x.strip()]
        external = _json.loads(lines[-1]) if r.returncode == 0 and lines else []
    except Exception:                            # noqa: BLE001 - 拿不到就当没有
        external = []
    with open(os.path.join(OUT, '.by1-ignored'), 'w', encoding='utf-8') as f:
        f.write('# 由 by1pack 生成：仓库里被 .gitignore 挡着、而文档引用了的路径。\n')
        f.write('# 包里的 by1docs 读它来分类（包里没有 git 可问）。\n')
        for k in external:
            f.write('%s\n' % k)

    # ── 自检：**在自己刚打出来的产物里跑一遍文档检查** ──────────────
    #
    # 这是"包不完整"这一类事故的判卷人。以前没有它，于是可以打出
    # 一个少 17 个判卷人、少 13 份文档、少一个 golden 快照的包，
    # 而打包脚本照样印"打包完成" —— 症状要到云端（或收包的人手里）
    # 才现形，长得还像"这一包本身有问题"。
    #
    # 判据：`by1docs` 在包里必须是绿的。它查的正是"文档里写的路径还在吗"，
    # 而包最容易被诟病的就是这个。
    problem = selfcheck(OUT)
    if problem:
        print('=' * 70)
        print('  **打包失败 —— 这一包自己都不自洽**')
        print('=' * 70)
        print(problem)
        return 1

    # 打成一个 tar.gz（保留目录结构）
    stamp = time.strftime('%Y%m%d')
    tgz = os.path.join(os.path.dirname(OUT), 'by1-upload-%s.tar.gz' % stamp)
    with tarfile.open(tgz, 'w:gz') as tf:
        for base, _dirs, files in os.walk(OUT):
            for f in files:
                full = os.path.join(base, f)
                tf.add(full, os.path.relpath(full, os.path.dirname(OUT)))
    tsize = os.path.getsize(tgz)

    print('=' * 70)
    print('  打包完成')
    print('=' * 70)
    print('  目录   %s' % OUT)
    print('  压缩包 %s  (%.2f MB)' % (tgz, tsize / 1048576))
    print('  %d 个文件，原始 %.2f MB' % (n, total / 1048576))
    print()
    print('  **没有装**（在云端重新拿）：')
    print('    · 模型权重 —— 从 HF 下，下多少取决于跑哪个模型')
    print('    · llamacpp/ 里的参考源码 —— 可重新下载')
    print('    · refs/ 的子目录（只装顶层那些官方产物文件）')
    print('    · cgen/ build/ __pycache__ —— 生成物')
    print()
    print('  包里的结构和仓库一样（src/ models/ refs/ + 根下的文档）。')
    print('  上传后：  tar xzf %s && cd by1-upload && bash gpu/run.sh'
          % os.path.basename(tgz))
    return 0


if __name__ == '__main__':
    sys.exit(main())
