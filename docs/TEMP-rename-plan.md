# 命名重命名与大规模迁移计划（2026-10-20）

> **TEMP —— 临时文件。** 迁移完成（或决定不做）后删除；历史留在 git 里。

> **一句话**：项目叫 `by1` 就够了 —— 不必让**每个文件**都再喊一遍。
> 本文只做计划（1. 重命名 2. 大规模迁移），**不动代码**；每阶段落地时单独提交、单独过门禁。

## 〇、现状与为什么现在做

| 事实 | 数字 |
|---|---|
| `src/` 下 `by1*.py` | **97**（checks 38 · modelcheck 20 · src 16 · lang 13 · data 8 · escape 2） |
| `tools/` 下 `by1*.py` | **3**（distill · ggufrecon · site） |
| 这些文件总行数 | **46,425** |
| 代码里的 `import by1*` / `from by1* import` | **382 处** |
| `by1paths.tool(...)` 调用（按文件名定位） | **51 处** |
| 文档里提到 `by1*.py` | **402 处** |

三条理由，按重要性排：

1. **读噪声**：`ls src/checks` 出来 38 行，每行都以 `by1` 开头 —— 前缀不携带信息，把真正有信息的部分（`all` / `docs` / `mutate`）挤到了第 4 个字符。
2. **新文件成本**：每加一个模块都要先写四遍 `by1`，且**无法靠语义判断该不该带**（带是默认，不带是例外）。
3. **安装会撞名（pyproject 自己记着）**：`checks` / `lang` / `data` / `modelcheck` / `escape` 这些**顶层包名**装进 site-packages 会和别人的同名包撞。`pyproject.toml` 的注释原话：“要改就把目录改名（`by1_lang` 之类）并把 `by1paths.SUBDIRS` 跟着改 —— **一处**。” —— 这条和文件前缀是同一个问题的两面：**命名空间没有被收拢**。

## 一、三个方案

| 方案 | 做法 | 代价 | 结论 |
|---|---|---|---|
| **A · 真包化（推荐）** | `src/by1/` 成为真正的包；模块名变成 `by1.ir` / `by1.codegen` / `by1.checks.docs` | 一次性机械迁移（382 import / 51 tool / 402 文档 / pyproject / 门禁自身） | ✅ **前缀只写在包名里一次**；同时解决安装撞名 |
| B · 只改目录名 | `checks/` → `by1_checks/`，文件名不动 | 小 | ❌ 只解决撞名，文件前缀还在 |
| C · 不改 + 写规范 | 约定新文件别带前缀 | 零 | ❌ 一半带一半不带，比现在更乱 |

**推荐 A。** 净收益不是审美：新增模块从“先想四遍前缀”变成“`by1/<名字>.py` 一眼归位”，安装撞名一起消失。

## 二、目标布局

```text
src/by1/__init__.py
src/by1/paths.py  ir.py  codegen.py  check.py  boot.py  c.py  exec_.py
src/by1/ext.py  full.py  ggml.py  graph.py  io.py  refs.py  sample.py  skip.py  ver.py
src/by1/checks/*.py        # 38 个判卷人，去掉 by1 前缀
src/by1/lang/*.py          # 12 个语言层模块 + cli.py
src/by1/modelcheck/*.py    # 20 个逐模型判卷人
src/by1/data/*.py          # 8 个数据取回/比对
src/by1/escape/*.py        # 2 个逃生舱
tools/distill.py  ggufrecon.py  site.py
```

**三条例外（必须记住）**：

- `by1exec.py` → `exec_.py`：`exec` 是**关键字**，`import by1.exec` / `from by1 import exec` 都是语法错误；
- `lang/by1.py` → `lang/cli.py`：它就是 `by1` 命令的入口（`pyproject` 的入口点指向它），模块名不能为空；
- `by1paths.py` → `paths.py`：**引导模块**，每个脚本开头都 import 它 —— 放**最后一个阶段**单独迁，避免所有脚本同时坏掉。

**不动的**：项目名 `by1`、CLI 名 `by1`、`models/` `refs/` `site/` 的数据文件。

## 三、逐文件映射

规则：`by1<X>.py` → `<X>.py`（迁进 `src/by1/`）。

| 现路径 | 新路径 |
|---|---|
| `src/by1boot.py` | `src/by1/boot.py` |
| `src/by1c.py` | `src/by1/c.py` |
| `src/by1check.py` | `src/by1/check.py` |
| `src/by1codegen.py` | `src/by1/codegen.py` |
| `src/by1exec.py` | `src/by1/exec_.py` |
| `src/by1ext.py` | `src/by1/ext.py` |
| `src/by1full.py` | `src/by1/full.py` |
| `src/by1ggml.py` | `src/by1/ggml.py` |
| `src/by1graph.py` | `src/by1/graph.py` |
| `src/by1io.py` | `src/by1/io.py` |
| `src/by1ir.py` | `src/by1/ir.py` |
| `src/by1paths.py` | `src/by1/paths.py` |
| `src/by1refs.py` | `src/by1/refs.py` |
| `src/by1sample.py` | `src/by1/sample.py` |
| `src/by1skip.py` | `src/by1/skip.py` |
| `src/by1ver.py` | `src/by1/ver.py` |
| `src/checks/by1all.py` | `src/by1/checks/all.py` |
| `src/checks/by1attrs.py` | `src/by1/checks/attrs.py` |
| `src/checks/by1backends.py` | `src/by1/checks/backends.py` |
| `src/checks/by1blind.py` | `src/by1/checks/blind.py` |
| `src/checks/by1bootir.py` | `src/by1/checks/bootir.py` |
| `src/checks/by1budget.py` | `src/by1/checks/budget.py` |
| `src/checks/by1coerce.py` | `src/by1/checks/coerce.py` |
| `src/checks/by1complete.py` | `src/by1/checks/complete.py` |
| `src/checks/by1debt.py` | `src/by1/checks/debt.py` |
| `src/checks/by1diff.py` | `src/by1/checks/diff.py` |
| `src/checks/by1docs.py` | `src/by1/checks/docs.py` |
| `src/checks/by1e2e.py` | `src/by1/checks/e2e.py` |
| `src/checks/by1fast.py` | `src/by1/checks/fast.py` |
| `src/checks/by1files.py` | `src/by1/checks/files.py` |
| `src/checks/by1gate.py` | `src/by1/checks/gate.py` |
| `src/checks/by1gpu.py` | `src/by1/checks/gpu.py` |
| `src/checks/by1gpubench.py` | `src/by1/checks/gpubench.py` |
| `src/checks/by1gpuday.py` | `src/by1/checks/gpuday.py` |
| `src/checks/by1gpusmoke.py` | `src/by1/checks/gpusmoke.py` |
| `src/checks/by1gputrain.py` | `src/by1/checks/gputrain.py` |
| `src/checks/by1irentry.py` | `src/by1/checks/irentry.py` |
| `src/checks/by1langspec.py` | `src/by1/checks/langspec.py` |
| `src/checks/by1lint.py` | `src/by1/checks/lint.py` |
| `src/checks/by1llama.py` | `src/by1/checks/llama.py` |
| `src/checks/by1mutate.py` | `src/by1/checks/mutate.py` |
| `src/checks/by1opdiff.py` | `src/by1/checks/opdiff.py` |
| `src/checks/by1oracle.py` | `src/by1/checks/oracle.py` |
| `src/checks/by1oracles.py` | `src/by1/checks/oracles.py` |
| `src/checks/by1pack.py` | `src/by1/checks/pack.py` |
| `src/checks/by1pat.py` | `src/by1/checks/pat.py` |
| `src/checks/by1rebuild.py` | `src/by1/checks/rebuild.py` |
| `src/checks/by1rules.py` | `src/by1/checks/rules.py` |
| `src/checks/by1run.py` | `src/by1/checks/run.py` |
| `src/checks/by1sitedata.py` | `src/by1/checks/sitedata.py` |
| `src/checks/by1smoke.py` | `src/by1/checks/smoke.py` |
| `src/checks/by1sync.py` | `src/by1/checks/sync.py` |
| `src/checks/by1tiers.py` | `src/by1/checks/tiers.py` |
| `src/checks/by1verify.py` | `src/by1/checks/verify.py` |
| `src/data/by1cache.py` | `src/by1/data/cache.py` |
| `src/data/by1cloud.py` | `src/by1/data/cloud.py` |
| `src/data/by1cmp.py` | `src/by1/data/cmp.py` |
| `src/data/by1fetch.py` | `src/by1/data/fetch.py` |
| `src/data/by1index.py` | `src/by1/data/index.py` |
| `src/data/by1revname.py` | `src/by1/data/revname.py` |
| `src/data/by1shapes.py` | `src/by1/data/shapes.py` |
| `src/data/by1triage.py` | `src/by1/data/triage.py` |
| `src/escape/by1extdemo.py` | `src/by1/escape/extdemo.py` |
| `src/escape/by1raw.py` | `src/by1/escape/raw.py` |
| `src/lang/by1.py` | `?` |
| `src/lang/by1blocks.py` | `src/by1/lang/blocks.py` |
| `src/lang/by1contract.py` | `src/by1/lang/contract.py` |
| `src/lang/by1emit.py` | `src/by1/lang/emit.py` |
| `src/lang/by1export.py` | `src/by1/lang/export.py` |
| `src/lang/by1hp.py` | `src/by1/lang/hp.py` |
| `src/lang/by1lower.py` | `src/by1/lang/lower.py` |
| `src/lang/by1name.py` | `src/by1/lang/name.py` |
| `src/lang/by1sched.py` | `src/by1/lang/sched.py` |
| `src/lang/by1stacks.py` | `src/by1/lang/stacks.py` |
| `src/lang/by1state.py` | `src/by1/lang/state.py` |
| `src/lang/by1tens.py` | `src/by1/lang/tens.py` |
| `src/lang/by1vocab.py` | `src/by1/lang/vocab.py` |
| `src/modelcheck/by1dev.py` | `src/by1/modelcheck/dev.py` |
| `src/modelcheck/by1gemma.py` | `src/by1/modelcheck/gemma.py` |
| `src/modelcheck/by1gpt2.py` | `src/by1/modelcheck/gpt2.py` |
| `src/modelcheck/by1gptoss.py` | `src/by1/modelcheck/gptoss.py` |
| `src/modelcheck/by1instella.py` | `src/by1/modelcheck/instella.py` |
| `src/modelcheck/by1kda.py` | `src/by1/modelcheck/kda.py` |
| `src/modelcheck/by1laguna.py` | `src/by1/modelcheck/laguna.py` |
| `src/modelcheck/by1load.py` | `src/by1/modelcheck/load.py` |
| `src/modelcheck/by1mem.py` | `src/by1/modelcheck/mem.py` |
| `src/modelcheck/by1mla.py` | `src/by1/modelcheck/mla.py` |
| `src/modelcheck/by1moe.py` | `src/by1/modelcheck/moe.py` |
| `src/modelcheck/by1q2schema.py` | `src/by1/modelcheck/q2schema.py` |
| `src/modelcheck/by1qknorm.py` | `src/by1/modelcheck/qknorm.py` |
| `src/modelcheck/by1quant.py` | `src/by1/modelcheck/quant.py` |
| `src/modelcheck/by1real.py` | `src/by1/modelcheck/real.py` |
| `src/modelcheck/by1rope.py` | `src/by1/modelcheck/rope.py` |
| `src/modelcheck/by1shadow.py` | `src/by1/modelcheck/shadow.py` |
| `src/modelcheck/by1sparse.py` | `src/by1/modelcheck/sparse.py` |
| `src/modelcheck/by1ssm.py` | `src/by1/modelcheck/ssm.py` |
| `src/modelcheck/by1train.py` | `src/by1/modelcheck/train.py` |
| `tools/by1distill.py` | `tools/distill.py` |
| `tools/by1ggufrecon.py` | `tools/ggufrecon.py` |
| `tools/by1site.py` | `tools/site.py` |

## 四、迁移阶段（每阶段：门禁绿 + CI 11/11 + 单独提交）

| 阶段 | 内容 | 判据 |
|---|---|---|
| **0 冻结** | `src/` 下**不得再新增** `by1*.py`（by1fast 加一条白名单检查，白名单随阶段缩减到空） | 新规则配必红反例（by1mutate 风格） |
| **1 骨架+叶子** | 建 `src/by1/__init__.py`；先迁无内部依赖的 `io` `skip` `ver`；旧名留 shim（可执行转发） | 档 1 + pack 腿 |
| **2 核心链路** | `ir` `codegen` `check` `boot` `full` `graph` `c` `ggml` `exec_` `sample` `ext` `refs` | 档 1/3/5 + 四路对拍 |
| **3 判卷人（最重）** | `checks/*` 38 个；同步改 `by1all` / `by1tiers` / `by1fast` 的清单、CI 的脚本路径 | 档 1/3/5 + pack + sitedata |
| **4 其余模块** | `lang`（13）· `modelcheck`（20）· `data`（8）· `escape`（2） | 档 1/3/5 + 四路 |
| **5 引导收尾** | `by1paths` → `by1/paths.py`；44 份脚本开头两行改 `from by1 import paths`；`check_boot` 判据同步 | 档 1 + 全脚本可执行 |
| **6 拆 shim + 清尾** | 删旧名；`pyproject` 改单包/入口点；`FILES.md` 重生成；**402 处文档引用一次性替换** | 全门禁 + CI 11/11 |

**为什么分 7 个阶段**：每一步都能**单独回滚**（一个提交一个阶段），且任何一个阶段绿了，仓库都处于可发布状态。不要一次搬完 —— 那是唯一会让你分不清“是搬错了还是本来就坏”的做法。

## 五、兼容层（迁移期专用）

- **shim 文件**：旧路径保留同名文件，内容 = 转发到新模块；判卷人是**当脚本执行**的（`python src/checks/by1all.py`），所以 shim 必须带 `if __name__ == "__main__": raise SystemExit(main())`，不能只写 `from ... import *`；
- **`by1paths.tool()` 双解析**：迁移期先找新名、再找旧名（路径只在这一处推，51 个调用点零改动）；
- **兼容期上限 2 个阶段**：shim 不是长久的家 —— 到期必拆，否则下次搬家时没人知道哪个文件是真的；
- **文档引用先不动**，阶段 6 一次性替换（by1docs 会逐条验存在性，分批改会出现“一半旧一半新”的红）。

## 六、风险与应对

| 风险 | 应对 |
|---|---|
| `import by1.exec` 语法错误 | 已定：`exec_.py`（三条例外的第一条） |
| 动态加载按文件名找脚本（`spec_from_file_location('bcm', by1paths.tool('by1check.py'))`） | `tool()` 双解析兜住；调用点不改 |
| “单独拷一个文件出去”的兜底（`try/except ImportError`） | 包化后**主动放弃**，在文档里明说；它现在只在少数模块里有 |
| by1fast 的「编译每个 `.py` / import 每个模块」 | 改成遍历包内模块（`import by1.xxx`）+ 文件编译，规则不变 |
| `pyproject` 的 14 个 py-modules + 5 个 packages + 2 个入口点 | 阶段 6 收成单包 `by1`；入口点 `by1 = "by1.lang.cli:main"`、`by1-spec = "by1.ir:_cli"` |
| 402 处文档引用 | 阶段 6 机械替换 + by1docs 验证；**不要分批混改** |
| `__pycache__` 里旧名的 .pyc 残留 | 迁移后清一次（`by1fast` 的编译检查会重建） |

## 七、验收（最终态）

- `src/` 递归找 `by1*.py` 只剩 `src/by1/` 目录本身（文件为 0）；
- 入口全部可用：`by1`、`by1-spec`、`python -m by1.checks.all --tier 1`、`--tier 5`；
- 门禁：档 1 / 档 3 / 档 5 + pack 腿 + **CI 11/11**；
- 文档：by1docs 绿（402 处引用全部指向新名）；
- 规则落地：第 0 阶段那条“新文件不得带 by1 前缀”的检查，白名单为空后仍然绿。

## 八、执行记录（就地更新）

### 阶段 1 —— 完成（2026-10-20）

- `src/by1/` 包骨架 + `__init__.py`（**故意不 import 任何子模块**——避免循环与启动开销）。
- 搬了三个叶子：`by1io` → `by1/io.py` · `by1skip` → `by1/skip.py` · `by1ver` → `by1/ver.py`。
- 旧路径留**可执行 shim**：`globals().update`（连私有名一起转）+ `runpy.run_path`（转发 `__main__`，不用 `run_module`——模块已被 import，会报 RuntimeWarning）。
- `by1fast`：递归收 `src/by1/` 编译，包内模块按 `by1.<name>` import；**没有**把 `by1` 加进 `by1paths.SUBDIRS`（会把 `src/by1/` 放上 sys.path，遮蔽标准库 `io`）。
- `by1files`：递归收包内文件 + 新分组（`io/skip/ver`），`lang/by1.py → cli.py` 同步进 GROUPS。
- **先说代价**：`lang/by1.py → cli.py` 本来排在阶段 4，但**必须提前** —— 它是模块 `by1`，和包 `by1` 撞名，`import by1` 会解析到它（`'by1' is not a package`）。`pyproject` 的入口点已改成 `lang.cli:main`。
- 判据：by1fast 编译 106 / import 101 / 解析 71 全绿 · by1files 绿 · by1docs 绿。

### 阶段 2 —— 完成（2026-10-20）

- 12 个核心模块进 `src/by1/`（`exec` 是关键字 → `exec_.py`）。
- shim 升级成**双模式**：`globals().update`（按路径装载拿到 shim 对象时也要有属性）+ `sys.modules` 原地替换（按名字 import 直接拿真身，循环 import 按原生语义）。
- 判据：编译 118 / import 113 · `by1ir --selftest` · 43 变异反例 · by1files/by1docs 绿。

### 阶段 3 —— 完成（2026-10-20）

- 38 个判卷人进 `src/by1/checks/`。
- `by1paths.tool('by1<X>.py')` **先解析包里真身** —— 读源码 / 按路径装载的调用拿到 shim 会读到空表。
- 包内模块的引导行改成「往上找到 `by1paths.py` 那一层」，并确保 `import by1paths`（否则 `SUBDIRS` 不在 `sys.path`，裸名 import 失败）。
- 用 `__file__` 数层数算 `ROOT` 的判卷人（attrs/tiers/sitedata/langspec）改问 `by1paths.ROOT`。
- `by1attrs` 的源码读取认「迁移期 shim」标记；自测夹具缺失 → `FileNotFoundError` → A5 按「缺语料」跳过（不拿真仓库的表去判假树）。
- 判据：**档 3 120 项 0 失败** · 编译 157 / import 152。

### 阶段 3.5 —— CI 三条腿修复（`ec6c71d`）

- **打包**：`WANT` 的 `by1.py` → `cli.py`、删 `history/plan.md`；`_where()`/`disk_modules()` 递归收 `src/by1/**`；`closure_missing` 跳过标准库名（包里有 `by1/io.py` 后 `import io` 会误报）。
- **完整模型腿**：`by1complete` 里漏改的一处 `tool('by1.py')` → `tool('cli.py')`。
- 本地复现并通过：打包 → 解包 → **包内档 1 全过**（15 项 0 失败）· by1complete/by1rules exit 0。

### 阶段 4 —— 完成（2026-10-20）

- `lang` 13（含 `cli`）· `modelcheck` 20 · `data` 8 · `escape` 2 → `src/by1/{lang,modelcheck,data,escape}/`，四个子包各加 `__init__.py`。
- 判据：**档 3 120 项 0 失败** · 编译 204 / import 199 · by1files/by1docs 绿。

### 阶段 5 —— 未开始

下一步：`by1paths` → `by1/paths.py`；44 份脚本开头的引导统一；`check_boot` 判据同步。然后阶段 6：拆 shim、`pyproject` 单包、文档引用、`src/` 下 `by1*.py` 归零。

## 九、不做的事

- 不改项目名、不改 CLI 名、不改 `models/` `refs/` 的数据命名；
- 不借迁移顺手改任何逻辑（**搬家不装修** —— 否则出问题时分不清是搬的还是改的）；
- 不为了“顺便统一”改判卷人的输出文案。
