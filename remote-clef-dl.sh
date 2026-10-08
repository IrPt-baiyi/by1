#!/bin/bash
# 把 clef 的 55 GB 下到 /dev/shm（它是 RAM 盘，60 GB）。
#
# **后台跑**，因为要几分钟到几十分钟，不能占着 ssh 连接。
# 进度写到 /dev/shm/clef-dl.log，另开一个连接看。
set -u
export HF_ENDPOINT=https://hf-mirror.com
DEST=/dev/shm/clef
LOG=/dev/shm/clef-dl.log
mkdir -p "$DEST"

# 已经在跑就不重复启动
if pgrep -f "by1-clef-download" >/dev/null 2>&1; then
  echo "  下载已经在跑了"
  tail -3 "$LOG" 2>/dev/null
  exit 0
fi

cat > /dev/shm/by1-clef-download.py <<'EOF'
import os, sys, time
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download(
    'Cloudflare/clef',
    local_dir='/dev/shm/clef',
    allow_patterns=['*.safetensors', '*.json', '*.py', '*.txt'],
    max_workers=8,
)
dt = time.time() - t0
tot = sum(os.path.getsize(os.path.join(dp, f))
          for dp, _, fs in os.walk(p) for f in fs)
print('DONE %.1f GB in %.0f s (%.0f MB/s)' % (tot / 1e9, dt, tot / 1e6 / max(dt, 1)))
EOF

nohup /root/miniconda3/bin/python /dev/shm/by1-clef-download.py \
      > "$LOG" 2>&1 &
echo "  已在后台启动，pid=$!"
echo "  日志 $LOG"
sleep 20
echo "  --- 20 秒后 ---"
du -sh "$DEST" 2>/dev/null
tail -2 "$LOG" 2>/dev/null | tr '\r' '\n' | tail -2
