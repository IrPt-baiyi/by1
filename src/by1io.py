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
import io
import json
import os
import sys

# 整个仓库的编码约定。**只此一处。**
ENCODING = 'utf-8'


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
        try:
            s.reconfigure(encoding=ENCODING, errors='replace')
        except Exception:
            pass


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
    """
    with io.open(path, encoding=encoding, errors=errors) as f:
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
    with io.open(path, encoding=encoding) as f:
        return f.read(n)


def iter_lines(path, encoding=ENCODING):
    """逐行读。**生成器** —— 大文件不会整个进内存。

    很多地方写的是 `for line in io.open(f):` —— 那个句柄在循环
    结束前一直开着，而且循环里 `break` 的话**就漏了**。
    """
    with io.open(path, encoding=encoding) as f:
        for line in f:
            yield line


def read_json(path, encoding=ENCODING):
    """读 JSON。**句柄一定关**，而且不用先读成字符串再 parse。"""
    with io.open(path, encoding=encoding) as f:
        return json.load(f)


def read_bytes(path):
    """读二进制。"""
    with open(path, 'rb') as f:
        return f.read()


def write_text(path, text, encoding=ENCODING):
    """写文本。**先建目录**，然后句柄一定关。"""
    _ensure_dir(path)
    with io.open(path, 'w', encoding=encoding) as f:
        f.write(text)


def append_text(path, text, encoding=ENCODING):
    """追加文本。"""
    _ensure_dir(path)
    with io.open(path, 'a', encoding=encoding) as f:
        f.write(text)


def write_json(path, obj, encoding=ENCODING, **kw):
    """写 JSON。`indent=0` 是这个仓库 `refs/` 的既有格式。"""
    kw.setdefault('ensure_ascii', False)
    kw.setdefault('indent', 0)
    _ensure_dir(path)
    with io.open(path, 'w', encoding=encoding) as f:
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
