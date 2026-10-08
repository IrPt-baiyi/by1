#!/bin/bash
set -u
A=$(du -sm /dev/shm/clef 2>/dev/null | cut -f1)
sleep 60
B=$(du -sm /dev/shm/clef 2>/dev/null | cut -f1)
D=$(( B - A ))
echo "  60 秒: ${A} MB -> ${B} MB，即 ${D} MB/min"
if [ "$D" -gt 0 ]; then
  echo "  55 GB 还要 $(( (55296 - B) / D )) 分钟 = $(( (55296 - B) / D / 60 )) 小时"
else
  echo "  **没动**"
fi
echo
echo "  当前总量: ${B} MB / 55296 MB"
ls /dev/shm/clef/*.safetensors 2>/dev/null | wc -l | xargs echo "  safetensors 文件数:"
