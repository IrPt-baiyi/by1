# 想请教一下：下一步先做哪个方向？

我在做一个挺小众的项目：用声明式语法描述模型架构，再用判据验证描述对不对。
现在门禁在干净 clone 上全绿，T4 上 0.6B 的真权重前向/反向能对上官方实现。
但下一步该往哪走，我自己拿不准，想听听意见。

粗粗列了几个候选（欢迎拍砖）：

- 把真权重对拍从 0.6B 扩到 8B（需要更大显存）
- 拿 llama.cpp 的 GGUF 和 by1 产物逐 logits 对拍（手头就能做）
- 让上游 llama.cpp 支持 Zamba2 / Step-5（工作量大，维护面也大）
- 继续做蒸馏的形状搜索（已在 tiny 上跑通过）
- 架构浏览器补证据
- 量化 Q2/Q3

我个人偏向第二个：不用新硬件、不用新权重，只差一次当面核对。
但也可能这个方向其实没那么重要，或者你看到更值得做的，都请说。

如果只回一句，请告诉我：你觉得哪个最值得先做，为什么。

---

# A small project — which direction should I take next?

I'm working on a fairly niche project: a declarative language for describing model architectures,
plus machine checks that verify whether the description is right.
The gates are all green on a clean clone, and on a T4 the real-weight forward/backward of a 0.6B model
matches the official implementation. But I'm not sure what to do next, so I'd like to hear your thoughts.

A rough list of candidates (pushback welcome):

- extend the real-weight comparison from 0.6B to 8B (needs more VRAM)
- compare llama.cpp's GGUF against by1's output logits, step by step (doable right now)
- add Zamba2 / Step-5 support to upstream llama.cpp (large effort, large maintenance surface)
- keep going on shape search for distillation (already works on tiny models)
- add evidence to the architecture browser
- quantization Q2/Q3

I lean toward the second one: no new hardware, no new weights, just one face-to-face check that hasn't been done.
But maybe that direction isn't as important as I think, or you see something more worthwhile — please say so.

If you only reply one line, tell me: which one do you think is worth doing first, and why.
