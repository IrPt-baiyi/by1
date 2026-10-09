#!/bin/bash
# **只量，不启动。** 30 秒。
set -u
D=/dev/shm/qwen38
A=$(du -sm "$D" 2>/dev/null | cut -f1); A=${A:-0}
sleep 30
B=$(du -sm "$D" 2>/dev/null | cut -f1); B=${B:-0}
D2=$(( B - A ))
echo "  30 秒: ${A} MB -> ${B} MB"
echo "  总速度: ${D2} MB/min = $(( D2 / 30 )) MB/s"
if [ "$D2" -gt 0 ]; then
  echo "  **55.6 GB 还要约 $(( (56900 - B) / D2 )) 分钟**"
else
  echo "  **没动**"
fi
echo "  分片: $(ls $D/*.safetensors 2>/dev/null | wc -l) 个"
pgrep -f do-dl >/dev/null && echo "  在跑" || echo "  **停了**"
tail -c 400 /dev/shm/dl.log | tr '\r' '\n' | grep -oE '[0-9.]+[MG]B/s' | tail -4 | tr '\n' ' '
echo
