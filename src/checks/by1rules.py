"""by1rules -- 转换期改写表的判卷人。

## 为什么它是个独立脚本

`by1llama` 的 CLI 要子命令（`build-gguf` / `logits-*` …），而 `by1all` 的判卷人
是按"**无参数跑一个脚本**"调用的。所以这里包一层 —— 而不是把 `by1llama` 的
默认行为改掉。

## 它判什么

`by1llama.CONVERT_RULES` 那张表：**每条规则的出口处 + 一个探针**。

- 出处：`conversion/<arch>.py:行号` 或 `:符号` —— 行号核行、符号核"那个名字还在文件里"
  （上游检出不在时算**没验**，不算通过）；
- 探针：拿**真实管线**（`_gguf_name` / `_tensors_for`）跑一个小 fixture，
  期望值是**独立写一遍**的上游公式，不是把实现抄一遍。

## 为什么必须有它

这个仓库栽过同一类跟头：**没进门禁的检查会烂**（`by1langspec` 那条欠账，
以及 `by1sample` 不在表里时 min_p 的 8 格红没人知道）。
改写表是产物线的核心知识 —— 它不能只活在注释里。

判据三态：通过 / 失败 / **没验**（缺 numpy、torch、gguf-py，或上游检出不在）。
"没验"会写出来，不混进"通过"。
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import by1llama  # noqa: E402


def main():
    print()
    print('=' * 78)
    print('  转换期改写表 —— 每条规则自带上游出处与探针')
    print('=' * 78)
    print()
    code = by1llama.selftest_convert_rules()
    print()
    return code


if __name__ == '__main__':
    _sys.exit(main())
