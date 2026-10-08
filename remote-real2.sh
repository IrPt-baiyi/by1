#!/bin/bash
# 真权重 27B 前向。**后台跑** —— 装 55 GB 要几分钟，不能占着连接。
set -u
cd /root/by1
PY=/root/miniconda3/bin/python
LOG=/dev/shm/real.log

pkill -f by1real 2>/dev/null; sleep 1
setsid nohup $PY by1real.py --by1 qwen38.by1 --dir /dev/shm/qwen38 --seq 4 \
       > "$LOG" 2>&1 < /dev/null &
sleep 3
echo "  启动 pid=$(pgrep -f by1real | head -1)"
echo "  日志 $LOG"
