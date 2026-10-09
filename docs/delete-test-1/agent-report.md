搞定了。结果在 build：9 个路径全部就位，字节数和 MANIFEST.txt 逐条对上，合计 1234405。

关键发现：那 7 个 refs/ 文件不是手写的，是公开产物的再导出，所以能"推"出来，不用抄：

上游	导出规则
gpt2 config.json	原样
gpt2 model.safetensors 头	header 原键序，只留 dtype+shape，紧凑单行
DeepSeek config.json	原样但 LF→CRLF（680→708）
Nemotron 14 分片头	分片顺序拼接（6513 张量），紧凑单行
Qwen3-VL 头	键名排序，只留 shape，indent=0+CRLF
GLM 48 分片头	每分片块内排序后拼接，值全 null，indent=0+CRLF
Laguna GGUF	取 Q4_K_M 那份的头，维度反序，ggml_type 留数字
两个坑值得记一下：GLM 看着像单行，其实是每键一行 + \r\n，而且排序只在分片块内部做（全局排序会把 model.norm.weight 甩到第一个）；Laguna 的 BF16 那份对不上，只有 Q4_K_M 的 ggml_type 分布吻合。全部用 HTTP Range 只读文件头，Nemotron 584 MB、GLM 551 MB 的正文一字节没下。

如实说清一处：mixtral-shaped.by1 和 models.tsv
 在上游没有来源——一个是 by1 的 MoE 声明，一个是模型清单。它们的内容只有靠 by1 那套源码（by1*.py / ir.md / plan.md）才能判对错，而那套代码不在这台机器上（你说了原件在另一台）。所以这两个是按副本字节恢复的，不是推出来的，我没有假装它们是。真要在"看不到原件"的条件下从零写对这两个文件，需要把 by1 的源码或 ir.md 给我。

复现：python docs/delete-test-1/rebuild_manifest.py --out build，推导记录见 `rebuild-record.md`。