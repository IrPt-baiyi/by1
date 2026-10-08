#!/bin/bash
# 从 ModelScope 下 Qwen3.8-27B 的 55.6 GB 到 /dev/shm。
# **后台跑**（约 1 小时），进度写日志。
#
# 为什么换 ModelScope：hf-mirror 上这个仓库的权重走 Xet
# （cas-bridge.xethub.hf.co），实测 26 KB/s —— 55 GB 要 8 天。
# ModelScope 是阿里的源，Qwen 也是阿里的，实测 14.7 MB/s。
set -u
PY=/root/miniconda3/bin/python
DEST=/dev/shm/qwen38
LOG=/dev/shm/dl.log

pkill -f 'snapshot_download' 2>/dev/null
rm -rf /dev/shm/ms /dev/shm/clef          # 清掉之前下的半截
mkdir -p "$DEST"

cat > /dev/shm/do-dl.py <<'EOF'
import os, time
from modelscope import snapshot_download
t0 = time.time()
p = snapshot_download(
    'Qwen/Qwen3.8-27B',
    local_dir='/dev/shm/qwen38',
    allow_patterns=['*.safetensors', '*.json', '*.py', '*.txt'],
    max_workers=6,
)
tot = sum(os.path.getsize(os.path.join(dp, f))
          for dp, _, fs in os.walk(p) for f in fs)
dt = time.time() - t0
print('DONE %.1f GB / %.0f 分钟 / 平均 %.1f MB/s'
      % (tot / 1e9, dt / 60, tot / 1e6 / max(dt, 1)))
EOF

nohup $PY /dev/shm/do-dl.py > "$LOG" 2>&1 &
echo "  后台启动 pid=$!  日志 $LOG"
sleep 25
echo "  --- 25 秒后 ---"
du -sh "$DEST" 2>/dev/null
tail -c 400 "$LOG" | tr '\r' '\n' | grep -oE '[0-9.]+[MG]/s' | tail -1 | xargs echo "  速度:"
