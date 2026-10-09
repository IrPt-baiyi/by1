#!/bin/bash
set -u
echo "=== clef 下载进度 ==="
du -sh /dev/shm/clef 2>/dev/null
df -h /dev/shm | tail -1
echo "--- 日志尾部 ---"
tail -c 800 /dev/shm/clef-dl.log 2>/dev/null | tr '\r' '\n' | grep -v '^$' | tail -4
echo "--- 进程还在吗 ---"
pgrep -f by1-clef-download >/dev/null && echo "  在跑" || echo "  **停了**"
echo "--- shm 里有什么 ---"
ls -la /dev/shm/clef 2>/dev/null | head -8
