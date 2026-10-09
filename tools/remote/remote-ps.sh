#!/bin/bash
set -u
PID=$(pgrep -f by1real | head -1)
echo "  pid=$PID"
if [ -n "$PID" ]; then
  ps -o pid,stat,pcpu,pmem,rss,etime,wchan:20 -p "$PID"
  echo
  echo "  --- 它在哪个系统调用上 ---"
  cat /proc/$PID/wchan 2>/dev/null; echo
  echo "  --- 内核栈（最后几行）---"
  cat /proc/$PID/stack 2>/dev/null | head -5 || echo "  （读不到）"
  echo
  echo "  --- 打开了什么大文件 ---"
  ls -l /proc/$PID/fd 2>/dev/null | grep -c safetensors | xargs echo "  打开的 safetensors 数:"
fi
echo
echo "  --- 日志（全部）---"
wc -l /dev/shm/real.log
tail -6 /dev/shm/real.log
