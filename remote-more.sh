#!/bin/bash
# 加流：8 -> 32。已下完的分片会复用，不重下。
set -u
PY=/root/miniconda3/bin/python
DEST=/dev/shm/qwen38
LOG=/dev/shm/dl.log

BEFORE=$(du -sm "$DEST" 2>/dev/null | cut -f1); BEFORE=${BEFORE:-0}
pkill -f 'do-dl' 2>/dev/null; pkill -f snapshot_download 2>/dev/null
sleep 2

cat > /dev/shm/do-dl.py <<'EOF'
import os, time
from modelscope import snapshot_download
t0 = time.time()
p = snapshot_download(
    'Qwen/Qwen3.8-27B',
    local_dir='/dev/shm/qwen38',
    allow_patterns=['*.safetensors', '*.json', '*.py', '*.txt'],
    max_workers=32,
)
tot = sum(os.path.getsize(os.path.join(dp, f))
          for dp, _, fs in os.walk(p) for f in fs)
dt = time.time() - t0
print('DONE %.1f GB / %.0f 分钟 / 平均 %.1f MB/s'
      % (tot / 1e9, dt / 60, tot / 1e6 / max(dt, 1)))
EOF

setsid nohup $PY /dev/shm/do-dl.py > "$LOG" 2>&1 < /dev/null &
sleep 3
echo "  重启完成 pid=$(pgrep -f do-dl | head -1)  （之前已有 ${BEFORE} MB）"
