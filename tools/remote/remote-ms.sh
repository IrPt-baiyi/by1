#!/bin/bash
# 换源：不走 HF/Xet，走 ModelScope（阿里的，Qwen 是阿里的模型）。
# **全程限时。**
set -u
PY=/root/miniconda3/bin/python
pkill -f 'dl[0-9].py' 2>/dev/null; sleep 1

echo "=== ① ModelScope 上有 Qwen3.8 / Qwen3.6 吗（限时 40 秒）==="
timeout 40 curl -s --max-time 15 \
  'https://www.modelscope.cn/api/v1/models/Qwen/Qwen3.8-27B' 2>&1 |
  head -c 400
echo
timeout 40 curl -s --max-time 15 \
  'https://www.modelscope.cn/api/v1/models/Qwen/Qwen3.6-35B-A3B' 2>&1 |
  head -c 400
echo

echo
echo "=== ② modelscope 装了吗 ==="
$PY -c "import modelscope;print('  装了', modelscope.__version__)" 2>&1 | tail -1

echo
echo "=== ③ ModelScope 的速度（限时 30 秒）==="
timeout 30 curl -sL -o /dev/null --max-time 25 -r 0-20000000 \
  -w "  速度 %{speed_download} B/s  下了 %{size_download} 字节  HTTP %{http_code}\n" \
  'https://www.modelscope.cn/models/Qwen/Qwen3.8-27B/resolve/master/config.json' \
  2>&1 | tail -2
