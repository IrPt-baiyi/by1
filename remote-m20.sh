#!/bin/bash
set -u
A=$(du -sm /dev/shm/qwen38 2>/dev/null | cut -f1); A=${A:-0}
sleep 20
B=$(du -sm /dev/shm/qwen38 2>/dev/null | cut -f1); B=${B:-0}
D=$(( B - A ))
echo "  20 秒: ${A} -> ${B} MB"
echo "  速度: $(( D / 20 )) MB/s"
if [ "$D" -gt 0 ]; then
  echo "  剩余: $(( (56900 - B) / (D / 20 + 1) / 60 )) 分钟"
fi
pgrep -f do-dl >/dev/null && echo "  在跑" || echo "  **停了**"
