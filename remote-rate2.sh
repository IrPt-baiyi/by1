#!/bin/bash
# 量**总**速度 —— 单个分片 14.7 MB/s，但那是六条流里的一条。
set -u
echo "=== 进程在吗 ==="
pgrep -af 'do-dl' | head -1 || echo "  **没在跑**"

echo
echo "=== 60 秒内的增长 ==="
A=$(du -sm /dev/shm/qwen38 2>/dev/null | cut -f1)
sleep 60
B=$(du -sm /dev/shm/qwen38 2>/dev/null | cut -f1)
D=$(( B - A ))
echo "  ${A} MB -> ${B} MB，增量 ${D} MB/min = $(( D * 1024 / 60 / 1024 )) MB/s"
if [ "$D" -gt 0 ]; then
  echo "  55.6 GB 还要约 $(( (56900 - B) / D )) 分钟"
else
  echo "  **没动**"
fi

echo
echo "=== 日志里的瞬时速度（取最后几个）==="
tail -c 3000 /dev/shm/dl.log 2>/dev/null | tr '\r' '\n' |
  grep -oE '[0-9.]+[MG]B/s' | tail -8 | tr '\n' ' '
echo
echo "=== 已经下了几个分片 ==="
ls /dev/shm/qwen38/*.safetensors 2>/dev/null | wc -l | xargs echo "  分片数:"
df -h /dev/shm | tail -1
