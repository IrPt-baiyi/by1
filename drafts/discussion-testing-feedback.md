# 想请人帮忙跑一下 5 分钟（零依赖）

项目主张是「描述即架构，判据能复核」。所以比起夸奖，我更想知道它在你机器上能不能跑起来。

```bash
git clone https://github.com/IrPt-baiyi/by1.git
cd by1
python3 src/checks/by1all.py --tier 1
```

应该看到最后两行：

```
  15 项，0 项失败，13 项跳过
  [全过（13 项没验）]
```

"13 项跳过"不是失败，是这一档没验。

如果对不上，麻烦告诉我：系统/Python 版本、完整命令、最后 20 行输出。
另外三类反馈我也很想要：

- 文档写的和实际输出不一致的地方
- 哪个判据你觉得永远不会红（假绿）
- 想让哪个模型被描述出来（带 HF repo 名）

已知的坑：Windows 行尾设 `core.autocrlf=false`；transformers 5.19+ 需要 torch ≥ 2.6；
没有 nvcc 时 llama.cpp 相关项会跳过。

如果你没时间跑，也完全没关系。
