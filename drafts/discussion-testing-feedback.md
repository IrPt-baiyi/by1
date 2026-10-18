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

---

# Could you spare 5 minutes to run this? (no dependencies)

The claim is "description is architecture; a judge you can re-check".
So rather than praise, I'd like to know whether it runs on your machine.

```bash
git clone https://github.com/IrPt-baiyi/by1.git
cd by1
python3 src/checks/by1all.py --tier 1
```

You should see, on the last two lines (the output is printed in Chinese):

```
  15 项，0 项失败，13 项跳过
  [全过（13 项没验）]
```

"13 skipped" is not a failure — it means that tier was not verified.

If it doesn't match, please tell me: OS / Python version, the full command, and the last 20 lines of output.
Three other kinds of feedback are also very welcome:

- places where the docs disagree with the actual output
- any check you think can never fail (a false green)
- a model you'd like to see described (with its HF repo name)

Known pitfalls: on Windows set `core.autocrlf=false`; transformers 5.19+ needs torch ≥ 2.6;
llama.cpp checks are skipped when nvcc is absent.

If you don't have time to run it, that's completely fine too.
