#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1io -- **"读文件要关句柄"这件事只说一次。**

## 为什么

仓库里有 57 处这样写：

    src = io.open(p, encoding='utf-8').read()

**在 CPython 上它没问题** —— 引用计数会立刻关掉句柄。
所以这不是 bug 修复，是**把一件事从 57 份实现收成 1 份**。

而它同时让调用处**更短**：

    src = by1io.read_text(p)          # 比原来还短一格

**这正是这个项目自己的原则**：

    > 同一个意思不要两处实现。

57 处 `io.open(...).read()` 就是 57 处实现 —— 而它们每一处都可能
忘了 `encoding='utf-8'`（在 Windows 上就是 gbk 乱码，
这一轮已经吃过一次：`by1debt` 里 subprocess 没给 encoding，
中文一进来就 UnicodeDecodeError）。

**收成一处之后，"读文本要 utf-8"就不可能忘。**

## 只依赖标准库

这个模块会被几乎所有 `by1*.py` import，所以它**不能**引入
numpy / torch / transformers —— 否则每个小脚本的启动都变贵。
"""
import io
import json
import os

# 整个仓库的编码约定。**只此一处。**
ENCODING = 'utf-8'


def read_text(path, encoding=ENCODING, errors='strict'):
    """读文本。**句柄一定关。**

    比 `io.open(p, encoding='utf-8').read()` 短，而且不会忘掉编码。
    """
    with io.open(path, encoding=encoding, errors=errors) as f:
        return f.read()


def head_text(path, n=2000, encoding=ENCODING):
    """读**前 n 个字符**。句柄一定关。

    ## 为什么单列一个

    仓库里有三处写的是 `io.open(f, encoding='utf-8').read(2000)` ——
    而 `read(2000)` 这个 2000 得**数一遍才知道是干嘛的**：
    它是"头部标签区"，不是"前 2000 字节的数据"。

    给它一个名字之后，调用处读起来就是意图本身：

        head = by1io.head_text(f)          # 看头部标签
        for line in by1io.iter_lines(f):   # 逐行
        src = by1io.read_text(f)           # 整个读进来

    三种意图，三个名字，**不用去数数字**。
    """
    with io.open(path, encoding=encoding) as f:
        return f.read(n)


def read_json(path, encoding=ENCODING):
    """读 JSON。**句柄一定关**，而且不用先读成字符串再 parse。"""
    with io.open(path, encoding=encoding) as f:
        return json.load(f)


def read_bytes(path):
    """读二进制。"""
    with open(path, 'rb') as f:
        return f.read()


def write_text(path, text, encoding=ENCODING):
    """写文本。**先建目录**，然后句柄一定关。

    `os.makedirs(os.path.dirname(p), exist_ok=True)` 这一步在原来
    57 处里有一半忘了写 —— 于是往一个还不存在的子目录写就炸，
    而报的是 FileNotFoundError，看起来像"路径写错了"。
    """
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with io.open(path, 'w', encoding=encoding) as f:
        f.write(text)


def write_json(path, obj, encoding=ENCODING, **kw):
    """写 JSON。`indent=0` 是仓库里 refs/ 的既有格式。"""
    kw.setdefault('ensure_ascii', False)
    kw.setdefault('indent', 0)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with io.open(path, 'w', encoding=encoding) as f:
        json.dump(obj, f, **kw)


def iter_lines(path, encoding=ENCODING):
    """逐行读。**生成器** —— 大文件不会整个进内存。

    很多地方写的是 `for line in io.open(f):` —— 那个句柄在循环
    结束前一直开着，而且循环里 `break` 的话**就漏了**。
    """
    with io.open(path, encoding=encoding) as f:
        for line in f:
            yield line


def exists(path):
    """就是 `os.path.exists` —— 写在这里是为了让调用处读起来一致。"""
    return os.path.exists(path)
