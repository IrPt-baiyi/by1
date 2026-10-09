#!/bin/bash
# **只启动，不量。** 量是下一个脚本的事 —— 合在一起会阻塞。
set -u
PY=/root/miniconda3/bin/python
DEST=/dev/shm/qwen38
LOG=/dev/shm/dl.log

pkill -f 'do-dl' 2>/dev/null; pkill -f snapshot_download 2>/dev/null
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

setsid nohup $PY /dev/shm/do-dl.py > "$LOG" 2>&1 < /dev/null &
sleep 3
echo "  已启动，pid=$(pgrep -f do-dl | head -1)"
echo "  日志 $LOG"
