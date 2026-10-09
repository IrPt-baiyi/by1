#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1io -- **文件读写的每一种意图，各有一个名字。**

## 为什么

仓库里原来有 57 处这样写：

    src = io.open(p, encoding='utf-8').read()

**在 CPython 上它没问题** —— 引用计数会立刻关掉句柄。
所以这不是 bug 修复，是**把一件事从 57 份实现收成 1 份**：

    src = by1io.read_text(p)          # 比原来还短一格

**这正是这个项目自己的原则**：

    同一个意思不要两处实现。

而且它顺手解决了一个真问题：57 处里每一处都可能忘了
`encoding='utf-8'` —— 在 Windows 上就是 gbk 乱码。
这一轮已经吃过：`by1debt` 里 `subprocess` 没给 encoding，
中文一进来就 `UnicodeDecodeError`。

## 意图，不是函数

    read_text(p)          整个读进来
    head_text(p)          只读头部标签区
    iter_lines(p)         逐行（大文件不进内存）
    read_json(p) / write_json(p, obj)
    read_bytes(p) / write_bytes(p, data)

**三种意图三个名字，不用去数 `read(2000)` 里的那个 2000。**

## 只依赖标准库

这个模块会被几乎所有 `by1*.py` import，所以它**不能**引入
numpy / torch / transformers —— 否则每个小脚本的启动都变贵。
"""
import contextlib
import io
import json
import os
import sys

# 整个仓库的编码约定。**只此一处。**
ENCODING = 'utf-8'


def _read_enc(encoding):
    """读的时候用哪种编码。见 `read_json` 里那段"BOM 要能读"。

    `utf-8` -> `utf-8-sig`：**只影响带 BOM 的文件**。
    别的编码（谁真传了 `gbk` 之类）原样返回，不做猜测。
    """
    return 'utf-8-sig' if encoding == 'utf-8' else encoding


def force_utf8_stdio(streams=('stdout', 'stderr')):
    """把 stdout / stderr 钉成 UTF-8。**整个仓库只此一处。**

    ## 为什么非做不可

    管道里的 Python 用的是**系统 ANSI 代码页**，中文 Windows 上就是 cp936。
    而 `by1all` 是用 `encoding='utf-8'` 去读子进程输出的 —— 于是子进程的
    中文判定行在父进程手里变成一堆 U+FFFD：

        !! 模式分类 by1pat.py         （没有判定行）      <- 其实打印的是"分类全对"
        UnicodeEncodeError: 'gbk' codec can't encode character '\ufffd'
                                     ^ 连汇总行都打不出来，by1all 自己崩掉

    **一个退出码 0、判定正确的脚本被记成失败** —— 正是这个仓库最反对的那类
    "跳过和通过长得一样"，只不过方向反了。

    ## errors='replace' 是判据的一部分

    宁可把编不出来的字符换成 `?`，也不能让**报告本身**抛异常。
    一份打不出来的报告等于没有报告。

    ## 老 Python 没有 `reconfigure`

    没有它也能跑，只是中文可能乱码 —— 所以这里是 `suppress`，不是崩溃。
    """
    for name in streams:
        s = getattr(sys, name, None)
        if s is None:
            continue
        # **用 `suppress` 而不是 `try/except/pass`。**
        # 后者读起来像"没想好"（by1lint 的规则① 也是这么判的），
        # 而这里的意思很明确：老解释器没有 reconfigure，那就随它去。
        # 同一个意图，两种写法 —— 挑那个一眼能看出是"故意忽略"的。
        with contextlib.suppress(Exception):
            s.reconfigure(encoding=ENCODING, errors='replace')


# **import 即生效。** 这个模块几乎所有 by1*.py 都会 import，
# 所以钉一次就够 —— 不需要每个脚本各写一遍（那正是"同一个意思两处实现"）。
force_utf8_stdio()


def _ensure_dir(path):
    """写之前先把目录建出来。

    原来 57 处里有一半没做这一步 —— 于是往一个还不存在的子目录写就炸，
    而报的是 `FileNotFoundError`，**看起来像"路径写错了"**，
    不像"我忘了建目录"。
    """
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def read_text(path, encoding=ENCODING, errors='strict'):
    """读文本。**句柄一定关。**

    比 `io.open(p, encoding='utf-8').read()` 短，而且不会忘掉编码。

    读的时候容忍 BOM（见 `read_json` 那段）—— 一个被别人用记事本
    改过的 `.by1`，它的第一行会变成 `\\ufeff# by1-repo: ...`，
    于是 `^#\\s*by1-repo:` 那条正则**匹配不上**，
    而症状是"这个模型没有 by1-repo 头"，不像"文件带了个 BOM"。
    """
    with io.open(path, encoding=_read_enc(encoding), errors=errors) as f:
        return f.read()


def head_text(path, n=2000, encoding=ENCODING):
    """读**前 n 个字符**。句柄一定关。

    ## 为什么单列一个

    仓库里有几处写的是 `io.open(f, encoding='utf-8').read(2000)` ——
    而那个 2000 得**数一遍才知道是干嘛的**：
    它是"头部标签区"，不是"前 2000 字节的数据"。

    有了名字之后，调用处读起来就是意图本身：

        head = by1io.head_text(f)          # 看头部标签
        for line in by1io.iter_lines(f):   # 逐行
        src = by1io.read_text(f)           # 整个读进来
    """
    with io.open(path, encoding=_read_enc(encoding)) as f:
        return f.read(n)


def iter_lines(path, encoding=ENCODING):
    """逐行读。**生成器** —— 大文件不会整个进内存。

    很多地方写的是 `for line in io.open(f):` —— 那个句柄在循环
    结束前一直开着，而且循环里 `break` 的话**就漏了**。
    """
    with io.open(path, encoding=_read_enc(encoding)) as f:
        for line in f:
            yield line


def read_json(path, encoding=ENCODING):
    """读 JSON。**句柄一定关**，而且不用先读成字符串再 parse。

    ## BOM 要能读

    `encoding='utf-8'` 碰上带 BOM 的文件会**直接抛**：

        json.decoder.JSONDecodeError: Unexpected UTF-8 BOM
        (decode using utf-8-sig)

    而这是一个 **Windows 优先**的仓库（README 的装机步骤、`%LOCALAPPDATA%`
    的 gcc 路径都是 Windows），而 Windows 上"记事本另存为 UTF-8"、
    `Set-Content -Encoding utf8`、以及一些编辑器**默认都加 BOM**。
    一个手写进来（或者被别人拿记事本改过）的 `refs/*.json` /
    `models.tsv` 就是这样变成"文件看着好好的，解析却报 JSON 错"。

    `utf-8-sig` 对**没有** BOM 的文件行为和 `utf-8` 完全一样，
    只在有 BOM 时把它吃掉 —— 所以读的默认值统一用它。
    写的默认值仍然是 `utf-8`（**不写 BOM**），这样产物字节稳定。
    """
    with io.open(path, encoding=_read_enc(encoding)) as f:
        return json.load(f)


def read_bytes(path):
    """读二进制。"""
    with open(path, 'rb') as f:
        return f.read()


def write_text(path, text, encoding=ENCODING):
    """写文本。**先建目录**，句柄一定关，**而且行尾恒为 LF**。

    ## `newline=''` 是干什么的

    不加它，Python 在 Windows 上会把 `\\n` 翻译成 `\\r\\n` ——
    于是**每个由本仓库生成的文件在 Windows 上都是 CRLF**：
    `FILES.md`、`models.md`、`ir-spec.md`、`VERSION`。

    而 `.gitattributes` 规定文本一律 LF。后果正是那份文件里预言的：

    > 在另一台机器上检出，每个文件都显示成改动过 ——
    > 于是"这次改了什么"这个问题失去意义。

    只不过这一次是**生成方**造成的：跑一遍 `--write`，git 就说
    "FILES.md 改了"，而 diff 里全是行尾。

    `newline=''` 关掉那个翻译：写进去什么字节就是什么字节。
    仓库约定 LF，所以生成方一律写 LF，不随平台变。
    """
    _ensure_dir(path)
    with io.open(path, 'w', encoding=encoding, newline='') as f:
        f.write(text)


def append_text(path, text, encoding=ENCODING):
    """追加文本。行尾同样恒为 LF（理由见 `write_text`）。"""
    _ensure_dir(path)
    with io.open(path, 'a', encoding=encoding, newline='') as f:
        f.write(text)


def write_json(path, obj, encoding=ENCODING, **kw):
    """写 JSON。`indent=0` 是这个仓库 `refs/` 的既有格式。

    行尾恒为 LF —— 理由见 `write_text`。这一条对 `refs/` 尤其要紧：
    那些文件的字节被拿来当"重建的判据"（见 `docs/delete-test-1/`），
    **平台不同就字节不同的话，那个判据就不成立了。**
    """
    kw.setdefault('ensure_ascii', False)
    kw.setdefault('indent', 0)
    _ensure_dir(path)
    with io.open(path, 'w', encoding=encoding, newline='') as f:
        json.dump(obj, f, **kw)


def write_bytes(path, data):
    """写二进制。**先建目录**，然后句柄一定关。

    `by1c.py` 往 `--workdir` 写 `w.bin` / `ids.bin`。
    """
    _ensure_dir(path)
    with open(path, 'wb') as f:
        f.write(data)


def exists(path):
    """就是 `os.path.exists` —— 写在这里是为了让调用处读起来一致。"""
    return os.path.exists(path)
