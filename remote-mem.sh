#!/bin/bash
set -u
echo "=== 内存 cgroup 上限（这才是真正管用的那个）==="
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  [ -f "$f" ] && echo "  $f = $(cat $f)"
done
echo "  当前用量:"
for f in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory/memory.usage_in_bytes; do
  [ -f "$f" ] && echo "    $f = $(( $(cat $f) / 1073741824 )) GB"
done
echo
echo "=== 内核有没有报 OOM ==="
dmesg 2>/dev/null | grep -i 'killed process' | tail -3 || echo "  （读不到 dmesg）"
echo
echo "=== /dev/shm 现在多少 ==="
du -sh /dev/shm/qwen38 2>/dev/null
df -h /dev/shm | tail -1
echo
echo "=== free 认可多少 ==="
free -g | head -2
