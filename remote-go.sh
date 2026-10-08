#!/bin/bash
# 启动 + 当场量**总**速度。单个分片 14.7 MB/s 只是六条流里的一条。
set -u
PY=/root/miniconda3/bin/python
DEST=/dev/shm/qwen38
LOG=/dev/shm/dl.log

# 清干净
pkill -f 'do-dl' 2>/dev/null
pkill -f snapshot_download 2>/dev/null
sleep 1
rm -rf /dev/shm/ms /dev/shm/clef /dev/shm/qwen38
mkdir -p "$DEST"

cat > /dev/shm/do-dl.py <<'EOF'
import os, time
from modelscope import snapshot_download
t0 = time.time()
p = snapshot_download(
    'Qwen/Qwen3.8-27B',
    local_dir='/dev/shm/qwen38',
    allow_patterns=['*.safetensors', '*.json', '*.py', '*.txt'],
    max_workers=8,
)
tot = sum(os.path.getsize(os.path.join(dp, f))
          for dp, _, fs in os.walk(p) for f in fs)
dt = time.time() - t0
print('DONE %.1f GB / %.0f 分钟 / 平均 %.1f MB/s'
      % (tot / 1e9, dt / 60, tot / 1e6 / max(dt, 1)))
EOF

nohup $PY /dev/shm/do-dl.py > "$LOG" 2>&1 &
PID=$!
echo "  启动 pid=$PID"

echo
echo "=== 量 90 秒的总速度 ==="
sleep 30
A=$(du -sm "$DEST" 2>/dev/null | cut -f1)
sleep 60
B=$(du -sm "$DEST" 2>/dev/null | cut -f1)
D=$(( B - A ))
echo "  ${A} MB -> ${B} MB，即 ${D} MB/min = $(( D * 1024 / 60 )) KB/s"
if [ "$D" -gt 0 ]; then
  echo "  55.6 GB 还要约 $(( (56900 - B) / D )) 分钟"
else
  echo "  **没动** —— 看日志"
fi
echo
echo "  分片数: $(ls $DEST/*.safetensors 2>/dev/null | wc -l)"
pgrep -f do-dl >/dev/null && echo "  进程在跑" || echo "  **进程没了**"
tail -c 600 "$LOG" | tr '\r' '\n' | grep -viE '^\s*$' | tail -3
