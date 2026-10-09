#!/bin/bash
set -u
echo "=== 量 30 秒的下载速度 ==="
A=$(du -sm /dev/shm/clef 2>/dev/null | cut -f1)
sleep 30
B=$(du -sm /dev/shm/clef 2>/dev/null | cut -f1)
echo "  30 秒前 ${A} MB -> 现在 ${B} MB"
echo "  速度 $(( (B - A) / 30 )) MB/s"
if [ "$B" -gt "$A" ]; then
  R=$(( (55296 - B) / ((B - A) / 30 + 1) / 60 ))
  echo "  按这个速度，剩下的还要约 ${R} 分钟"
else
  echo "  **没在增长** —— 可能卡住了"
fi
echo
echo "=== 大文件下到了吗 ==="
ls -la /dev/shm/clef/*.safetensors 2>/dev/null | head -4
echo "  safetensors 文件数: $(ls /dev/shm/clef/*.safetensors 2>/dev/null | wc -l) / 13"
echo
echo "=== 日志尾部（去掉进度条刷屏）==="
tail -c 2000 /dev/shm/clef-dl.log 2>/dev/null | tr '\r' '\n' | grep -viE '^\s*$|it/s\]$' | tail -6
