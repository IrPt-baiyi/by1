# paper/ —— 论文稿（中英双语全文）

一份正经系统论文的 LaTeX 源码，内容围绕 **by1：声明式大模型架构描述 + 机器可验证合同**。
两篇正文同构（摘要 / 引言 / 问题与边界 / 语言设计 / IR / 验证方法学 / 实现 /
评测 / 教训与负结果 / 相关工作 / 结论），共享一份参考文献。

| 文件 | 说明 |
|---|---|
| [by1-en.tex](by1-en.tex) | 英文全文（article，pdfLaTeX / XeLaTeX 均可） |
| [by1-zh.tex](by1-zh.tex) | 中文全文（ctexart，**需 XeLaTeX**） |
| [by1-refs.bib](by1-refs.bib) | 共享参考文献（27 条，两篇都引用） |
| [build_docx.py](build_docx.py) | LaTeX → Word 转换脚本（生成两份 .docx） |
| by1-en.docx / by1-zh.docx | Word 版全文（由脚本从 .tex 生成，内容一致） |

## 编译

本机没有安装 TeX 发行版，推荐 Overleaf：

1. 新建项目，上传 by1-en.tex / by1-zh.tex / by1-refs.bib 三个文件；
2. 英文稿：Menu → Compiler 选 **pdfLaTeX**；中文稿：选 **XeLaTeX**；
3. Overleaf 会自动跑 bibtex，参考文献即出。

本地命令（装了 TeX Live / MiKTeX 时）：

    latexmk -pdf by1-en.tex
    latexmk -xelatex by1-zh.tex

## 数据来源（每个数字都可回溯）

- 六模型前向对拍、长上下文、训练步、显存、MTP/视觉：[docs/gpu-validation-results.md](../docs/gpu-validation-results.md)
  及 docs/gpu-results/ 下的原始 JSON（2026-10-13，RTX PRO 6000 Blackwell 96GB）；
- 四个支持状态与覆盖数（规格 37 / 运行 8 / 产物 1）：[docs/support-matrix.md](../docs/support-matrix.md)；
- 语言与 IR 细节、错误模型、负例基线 9 个错误：[docs/by1-language-spec.md](../docs/by1-language-spec.md)、[ir-spec.md](../ir-spec.md)；
- 「可描述 ≠ 可计算」事故（gpt2.by1 与 llama-shaped.by1 出同一份 IR）：[README.md](../README.md)；
- 目标用户与非目标：[docs/product-definition.md](../docs/product-definition.md)；
- 判据审计任务单：[docs/criteria-review.md](../docs/criteria-review.md)。

## 交稿前需要你补的

1. **作者、单位、致谢**（两篇的 author 字段现在是占位）；
2. **2026 年模型卡引用**（bib 里 qwen382026 等 @misc 项）——按你目标投稿的引用格式
   核对一下出处；经典文献（Vaswani、TVM、MLIR、vLLM 等）已按正式条目写好；
3. 如果目标是会议投稿，摘要长度、章节编号、匿名化（去掉 provenance 段落里的
   项目身份）按征稿要求再调。

## 一句话定位（写给审稿人也写给你自己）

论文的主张刻意收窄：**六个真实 checkpoint 上，从中立 IR 编译、真权重加载、
成熟运行时执行的描述，精确复现参考数值**（含 27B 视觉 + MTP、MXFP4 MoE），
另有一次逐梯度一致的真实训练步；因为复用参考算子，**不主张任何内核正确性或
速度**。方法论主张更宽：支持按证据状态报告、判卷人在项目外、未实现的组合要
拒绝、规格尽量由程序生成。这个口径就是 docs/ 里那套「不许夸大」的纪律，
照原样搬进了论文。
