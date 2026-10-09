#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1skip -- **一次检查有五种结论，不是两种。**

## 退出码：首位是 HTTP 的类

    0     通过                  2xx  "成了"
    1     被测对象错了            5xx  "服务端"的错
    30    这台机器上没验          3xx  "不在我这判"
    40    调用方用错了            4xx  "你给的东西不对"
    51    检查器自己坏了          5xx  但"服务端"是**工具**

**为什么成功是 0 而不是 20。** 不是惯例问题，是**会坏**：
pre-push 钩子写的是 `exec "$PY" by1all.py --tier 1`，而 git 把非零
当失败 —— 成功码一旦不是 0，**钩子会永远拦住推送**。`&&`、
`set -e`、Makefile 同理。

**为什么失败是 1 而不是 50。** 那一条是 79 处调用点的机械替换，
而 1 是那个"全世界的 shell、CI、`&&` 都认"的码。为了一致的首位
去动 79 个地方，收益不抵风险。**其余每一个码都两位、首位是类。**

## 为什么需要五种

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

**后来发现 2 还是双义的**：`by1ir` 参数不全、`by1gpu` 没显卡、
`by1check` 用法不对，都是 2。所以拆成 **30（没验）** 和
**40（调用方错）** —— 这两件事的修法完全不同：一个是换台机器，
一个是改调用。

**再后来发现还缺一格**：检查器自己崩了（一个 `AttributeError`）
和"验了不对"长得一模一样，报告里都印「失败」。而这是最会误导人的
一种 —— **一个坏掉的检查器看起来像"这个模型有问题"**，
然后有人去改模型。所以有了 **51**。

## 两条通道

    ① 退出码 == 30        ② 输出里有 `[跳过]` 这一行

**为什么要两条。** 判据不能只有一条通道 —— 这条规矩本来就写在
`by1all` 里（"退出码说 ok，但输出里是 FAIL —— 两个通道不一致"）。

## 产出方必须让"更坏"的压过"没那么坏"

    失败(1) > 检查器坏(51) > 调用方错(40) > 跳过(30) > 通过(0)

一个调用里如果**既**跳过了某项**又**验出了错，退出码必须是更坏的那个。
反过来会把真失败藏起来。

## 它是标准库级的

和 `by1io` / `by1paths` 一样**只依赖标准库** —— 它会被最底层的脚本 import。
"""
import sys

#: **没验**。这台机器上做不了（缺 gcc / 缺显卡 / 缺缓存）。
#: **这是协议的一半**，另一半是 `MARK`。
CODE = 30

#: **调用方用错了**。参数不全、路径不存在、命令名不认识。
#: 它以前和 `CODE` 挤在同一个 2 里，而两者的修法完全不同。
CALLER = 40

#: **被测对象错了**。验了，算得不对。
FAIL = 1

#: **检查器自己坏了**。崩了（traceback）、或者环境缺了它自己要的东西。
#: 它和 `FAIL` 的区别是**该去修哪个东西** —— 修模型，还是修工具。
BROKEN = 51

#: 跳过的标记行。判卷人靠它把"跳过"和"用法错误"分开。
MARK = '[跳过]'

#: 五个码 -> 五个名字。给人看的时候用这一张，不要各处自己写 if。
NAMES = {0: 'ok', FAIL: 'fail', CODE: 'skip', CALLER: 'caller',
         BROKEN: 'broken'}


def skip(msg, code=CODE):
    """打印一行跳过说明，**返回该用的退出码**。

    调用处写成 `return by1skip.skip(...)`：打印和退出码变成同一句话，
    于是不存在"改了一行忘了改另一行"的可能。
    """
    print('\n  %s %s' % (MARK, msg))
    return code


def caller(msg, code=CALLER):
    """调用方用错了。**和 `skip` 分开** —— 修法完全不同。"""
    print('\n  %s %s' % ('[用法]', msg))
    return code


def broken(msg, code=BROKEN):
    """检查器自己坏了（或者环境缺了它自己要的东西）。"""
    print('\n  %s %s' % ('[坏了]', msg))
    return code


def is_skip(returncode, output):
    """两条通道都对，才算跳过。"""
    return returncode == CODE and MARK in (output or '')


def verdict(returncode, output):
    """**五态判决**：`'ok'` / `'fail'` / `'skip'` / `'caller'` / `'broken'`。

    顺序（从"最确定"到"最不确定"）：

        0              -> ok（除非输出里有 [跳过] —— 那取更保守的）
        CODE + MARK    -> skip
        CALLER         -> caller
        BROKEN         -> broken
        其它非零        -> fail

    **输出通道可以推翻一个 0。** 说了跳过就不是通过 —— 哪怕退出码是 0。
    两条通道不一致时**取更保守的那一个**。
    """
    out = output or ''
    if returncode == 0:
        return 'skip' if MARK in out else 'ok'
    if is_skip(returncode, out):
        return 'skip'
    if returncode == CALLER:
        return 'caller'
    if returncode == BROKEN:
        return 'broken'
    # **崩了也是"检查器坏了"。** 一个没被接住的异常，退出码是 1 ——
    # 和"验了不对"撞在一起。所以这里看输出：有 traceback 就是工具的问题。
    # （这一条让 50 个脚本一行都不用改。）
    if 'Traceback (most recent call last)' in out:
        return 'broken'
    return 'fail'


def mark(status):
    """给人看的记号。**五种结论必须长得不一样。**"""
    return {'ok': '  ok ', 'skip': '  -- ', 'fail': '  !! ',
            'caller': '  ?? ', 'broken': '  XX '}.get(status, '  ?? ')


if __name__ == '__main__':
    # 自检：五种结论各走一遍，外加三个陷阱。
    CASES = [
        (0, '一切正常', 'ok', '普通的通过'),
        (FAIL, '[FAIL] 算错了', 'fail', '普通的失败'),
        (CODE, '  %s 没有 gcc' % MARK, 'skip', '两条通道都说是跳过'),
        (CALLER, '  用法: by1ir.py <file>', 'caller',
         '**新**：调用方用错了 —— 以前它和"跳过"挤在同一个 2 里'),
        (BROKEN, '  [坏了] 环境里没有 transformers', 'broken',
         '**新**：检查器自己坏了'),
        (1, 'Traceback (most recent call last):\n  ...\nAttributeError',
         'broken',
         '**陷阱**：崩了也是 1，但输出里有 traceback —— 那是工具的问题，'
         '不是模型的问题'),
        (CODE, '  用法: by1ir.py <file>', 'fail',
         '**陷阱**：退出码是跳过的码、却没说跳过 —— 那是用法错误'),
        (0, '  %s 缓存里没有它' % MARK, 'skip',
         '**陷阱**：退出码说通过、输出说跳过 —— 取更保守的那个'),
    ]
    bad = 0
    for rc, out, want, why in CASES:
        got = verdict(rc, out)
        ok = got == want
        bad += 0 if ok else 1
        print('  %s rc=%-3d -> %-7s %s' % ('ok ' if ok else '!! ', rc, got, why))
    print()
    print('  [%s] 五态协议 %s'
          % ('PASS' if not bad else 'FAIL',
             '五种结论互不串台' if not bad else '%d 个反例串台了' % bad))
    sys.exit(0 if not bad else 1)
