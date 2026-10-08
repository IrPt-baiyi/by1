#!/bin/bash
# 用 -u 重启：非 TTY 时 Python 会缓冲 stdout，日志看起来"卡住了"，
# 其实在跑。加 -u 就能看见进度。
set -u
cd /root/by1
PY=/root/miniconda3/bin/python
LOG=/dev/shm/real.log

pkill -f by1real 2>/dev/null; sleep 2
: > "$LOG"
setsid nohup $PY -u by1real.py --by1 qwen38.by1 --dir /dev/shm/qwen38 --seq 4 \
       > "$LOG" 2>&1 < /dev/null &
sleep 5
echo "  重启 pid=$(pgrep -f by1real | head -1)"
sleep 25
echo "  --- 30 秒后 ---"
tail -c 900 "$LOG" | tr '\r' '\n' | tail -10
