#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1skip -- **"这一次没验"是一个独立的结论。**

## 为什么要有这个文件

在这个文件之前，护栏只有两格：

    退出码 0     通过
    退出码 ≠ 0   失败

而"这台机器上验不了"（没有 gcc / 没有 CUDA / HF 缓存里没有那个模型）
只能往这两格里挤。**挤进哪一格都是错的：**

    挤成 0   一个没验过的东西看起来像验过了   <- 最坏的一种绿
    挤成 1   一个没坏的东西看起来像坏了       <- 然后有人去修它

第二种在这个仓库里真的发生过：`by1e2e`（缺 HF 缓存）和 `by1extdemo`
（缺 gcc）长期躺在 `by1all` 的失败清单里，而它们退出码 2 —— **它们的
作者本来就想说"跳过"**，只是没人读这一格。

## 协议：两条通道，缺一不可

    ① 退出码 == 2         ② 输出里有 `[跳过]` 这一行

**为什么要两条。** 退出码 2 在这个仓库里是双义的 ——
`by1ir.py` 参数不全、`by1gpu.py` 没显卡、`by1check.py` 用法不对，
都是 2。只认退出码的话，**一个参数写错的调用会被记成"跳过"**，
那就是把错误变成了沉默。

判据不能只有一条通道 —— 这条规矩本来就写在 `by1all` 里
（"退出码说 ok，但输出里是 FAIL —— 两个通道不一致"），
这里只是把它用到第三种结论上。

## 产出方必须让"失败"压过"跳过"

一个调用里如果**既**跳过了某项**又**验出了错，退出码必须是失败的那个：

    if 有失败:  return 1
    if 有跳过:  return by1skip.CODE
    return 0

反过来（跳过压过失败）会把真失败藏起来。`by1verify` 就是这么写的。

## 用法

    # 产出方：打印和退出码**一次写成**，不可能只改一半
    return by1skip.skip('找不到 gcc —— 这一步要编译那个 .so')

    # 判卷人：
    st = by1skip.verdict(r.returncode, out)     # 'ok' / 'skip' / 'fail'

## 它是标准库级的

和 `by1io` / `by1paths` 一样**只依赖标准库** —— 它会被最底层的脚本 import。
"""
import sys

#: 跳过的退出码。**这是协议的一半**，另一半是 MARK。
CODE = 2

#: 跳过的标记行。判卷人靠它把"跳过"和"用法错误"分开。
MARK = '[跳过]'


def skip(msg, code=CODE):
    """打印一行跳过说明，**返回该用的退出码**。

    调用处写成 `return by1skip.skip(...)`：打印和退出码变成同一句话，
    于是不存在"改了一行忘了改另一行"的可能。
    """
    print('\n  %s %s' % (MARK, msg))
    return code


def is_skip(returncode, output):
    """两条通道都对，才算跳过。"""
    return returncode == CODE and MARK in (output or '')


def verdict(returncode, output):
    """三态判决：`'ok'` / `'skip'` / `'fail'`。

    **输出通道可以推翻一个 0。** 说了跳过就不是通过 —— 哪怕退出码是 0。
    这跟 `by1all` 里那条反向规矩是同一个道理：判卷人的判据不能只有一条
    通道，而两条通道不一致时，**取更保守的那一个**（这里就是"没验"）。
    """
    out = output or ''
    if returncode == 0:
        return 'skip' if MARK in out else 'ok'
    return 'skip' if is_skip(returncode, out) else 'fail'


def mark(status):
    """给人看的三个记号。**三种结论必须长得不一样。**"""
    return {'ok': '  ok ', 'skip': '  -- ', 'fail': '  !! '}.get(status, '  ?? ')


if __name__ == '__main__':
    # 自检：三种结论各走一遍，外加"退出码 2 但没有标记"这个陷阱。
    CASES = [
        (0, '一切正常', 'ok', '普通的通过'),
        (1, '[FAIL] 算错了', 'fail', '普通的失败'),
        (CODE, '  %s 没有 gcc' % MARK, 'skip', '两条通道都说是跳过'),
        (CODE, '  用法: by1ir.py <file>', 'fail',
         '**陷阱**：退出码 2 但没说跳过 —— 那是用法错误，不是跳过'),
        (0, '  %s 缓存里没有它' % MARK, 'skip',
         '**陷阱**：退出码说通过、输出说跳过 —— 取更保守的那个'),
    ]
    bad = 0
    for rc, out, want, why in CASES:
        got = verdict(rc, out)
        ok = got == want
        bad += 0 if ok else 1
        print('  %s rc=%-2d -> %-5s  %s' % ('ok ' if ok else '!! ', rc, got, why))
    print()
    print('  [%s] 跳过协议 %s'
          % ('PASS' if not bad else 'FAIL',
             '三条结论互不串台' if not bad else '%d 个反例串台了' % bad))
    sys.exit(0 if not bad else 1)
