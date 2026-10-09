#!/bin/bash
set -u
pkill -f dl1.py 2>/dev/null
pkill -f clef-download 2>/dev/null
sleep 1
echo "=== 停掉了 ==="
pgrep -af 'dl1|clef-download' | head -2 || echo "  没有残留进程"

echo
echo "=== hf-mirror 到底多快（直连测一个 100 MB 的文件）==="
timeout 40 curl -s -o /dev/null -w "  hf-mirror: %{speed_download} B/s, 下了 %{size_download} 字节\n" \
  --max-time 35 \
  https://hf-mirror.com/Cloudflare/clef/resolve/main/joint_head.safetensors 2>&1 | tail -2

echo
echo "=== ModelScope 可达吗（国内源，通常快得多）==="
timeout 15 curl -sI https://www.modelscope.cn 2>&1 | head -1
echo "  modelscope 装了吗: $(/root/miniconda3/bin/python -c 'import modelscope;print(modelscope.__version__)' 2>&1 | tail -1)"

echo
echo "=== 换个思路：从 HF 直连（不走镜像）==="
timeout 30 curl -s -o /dev/null -w "  huggingface.co: %{speed_download} B/s\n" \
  --max-time 25 https://huggingface.co 2>&1 | tail -1
