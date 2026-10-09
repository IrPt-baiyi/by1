#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1ext -- 逃生舱第二层：**引用一个外部符号**。

## 和第一层（`Raw`）的差别

    Raw       impl = "名字"        -> 找 raw.py 里的工厂函数
              **要改编译器** —— 加一个机制就得动 by1codegen.py

    External  lib + symbol + 签名  -> 动态加载一个 .so
              **不用改编译器** —— 编译器不认识这个机制，只认识调用约定

## ABI（固定，不自由发挥）

    void <symbol>(const float *x, float *y,
                  int B, int T, int D,
                  const float *const *w, int nw);

    x    输入，行主序 [B, T, D]
    y    输出，行主序 [B, T, D]
    w    权重指针数组，**按名字字典序排列**
    nw   w 的个数

**为什么权重是指针数组而不是拼成一块**：C 那边权重本来就是分开的缓冲区。
拼成一块要么多一次拷贝，要么要求两边布局一致 —— 而那是**约定**不是**语义**。
指针数组 + 排序规则，两边都能自己算出来，不需要额外通道。

## 契约没有放松

外部算子**照样要声明自己的张量**，照样被 `by1verify` 逐个对产物查。
松掉的只有一样：这段计算不在这门语言能表达的范围内。
"""
import ctypes
import os

_CACHE = {}


def load_lib(path, base_dir=None):
    """加载动态库。**找不到就报错，不返回一个空壳。**"""
    if base_dir and not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    path = os.path.abspath(path)
    if path in _CACHE:
        return _CACHE[path]
    if not os.path.exists(path):
        raise RuntimeError(
            "外部算子要加载 %s，但它不存在。"
            "**相对路径按 IR 文件所在目录解析** —— 换个目录就跑不了，"
            "所以这里把它拼成了绝对路径：%s" % (path, path))
    try:
        lib = ctypes.CDLL(path)
    except OSError as e:
        raise RuntimeError("加载 %s 失败：%s" % (path, e))
    _CACHE[path] = lib
    return lib


def bind(lib, symbol):
    """按 ABI 绑定符号。**签名在这里定死，调用处不自由发挥。**"""
    try:
        fn = getattr(lib, symbol)
    except AttributeError:
        raise RuntimeError(
            "库里没有符号 `%s`。**不静默给个恒等** —— 那会生成一个"
            "看起来对但什么都没算的模型。" % symbol)
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int,
                   ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    return fn


def call(fn, x, y, B, T, D, weight_arrays):
    """按 ABI 调一次。

    `weight_arrays` 必须**已经按名字字典序排好** —— 排序规则是 ABI 的一部分，
    两边都得自己算出来，不靠额外通道传。
    """
    nw = len(weight_arrays)
    arr = (ctypes.c_void_p * max(nw, 1))()
    for i, w in enumerate(weight_arrays):
        arr[i] = ctypes.c_void_p(w.ctypes.data)
    fn(ctypes.c_void_p(x.ctypes.data), ctypes.c_void_p(y.ctypes.data),
       B, T, D, arr, nw)


def sorted_weight_names(names):
    """**权重顺序 = 名字字典序。** 这是 ABI 的一部分，不是实现细节。"""
    return sorted(names)
