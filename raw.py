"""逃生舱的实现 —— 给 raw-escape.by1 用。

一个 by1 描述里的 `mech X : Raw { impl = "scale_mix" }` 会去找这里的
`raw_scale_mix(d, attrs)`，它返回一个 `nn.Module`。

## 这个文件存在的意义

`by1` 只覆盖了它见过的机制。**遇到没见过的，以前只有两条路**：
把机制硬塞进某个已有种类（生成一个"看起来对但算错"的模型），
或者等语言支持它。

逃生舱是第三条：**在这里实现，而契约和验证照旧生效。**

## 和"绕过检查"的区别

约束没有松：

    ✓ 张量契约仍然要声明 —— 名字和形状仍然逐个对产物对拍
    ✓ 三个后端仍然要对拍 —— 对不齐就是错
    ✓ impl 找不到就拒绝 —— 不静默给个恒等

松掉的只有一样：这段计算不用 by1 能表达的写法来写。

## 一条约定（**不是可选的**）

**参数名要和契约里的逻辑名对得上。**

契约写 `scale.weight : (d_model,)`，这里就得让建出来的模块里出现
`…scale.weight` 这个路径。做法是套一层子模块挂在 `self.scale` 上：

    class _Scale(nn.Module):        # 里面那个 Parameter 叫 weight
        def __init__(self, d):
            self.weight = nn.Parameter(torch.ones(d))

    class ScaleMix(nn.Module):
        def __init__(self, d):
            self.scale = _Scale(d)   # -> …scale.weight  ✓

**不这么写会怎样**：产物那一侧照样验得过（`by1verify` 对的是 checkpoint），
但实现这一侧的名字和契约对不上 —— 而**没有任何东西会告诉你**。
这正是这个项目里反复出现的那一类问题：名字看着对，指的是别的东西。

（第一版这里写的是 `self.scale = nn.Parameter(...)`，建出来是
`…op1.inner.scale` —— 少了一段。契约说 `scale.weight`，实现给 `scale`。）
"""
import torch
import torch.nn as nn


class _Scale(nn.Module):
    """挂成 `self.scale`，里面的参数叫 `weight` —— 合起来就是契约里的
    `scale.weight`。"""
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))


def raw_scale_mix(d, attrs):
    """一个语言里没有的机制：逐维缩放。

    契约里声明的是 `scale.weight : (d_model,)` —— 所以这里必须建出
    那个路径。**这就是"约束没松"的意思。**
    """
    class ScaleMix(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = _Scale(d)

        def forward(self, x):
            return x * self.scale.weight

    return ScaleMix()
