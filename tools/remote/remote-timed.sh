#!/bin/bash
# **全程限时。** 卸载 + 测速，5 分钟封顶。到点就停，不磨。
#
# Xet（cas-bridge.xethub.hf.co）是 HF 的新存储后端，hf-mirror 不代理它，
# 于是下载卡在 "read operation timed out" 然后无限重试。
# 设 HF_HUB_DISABLE_XET=1 不够 —— 装了 hf_xet 就得卸掉。
set -u
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
PY=/root/miniconda3/bin/python

pkill -f dl1.py 2>/dev/null; pkill -f dl2.py 2>/dev/null
pkill -f clef-download 2>/dev/null; sleep 1

echo "=== 卸 hf_xet（限时 120 秒）==="
timeout 120 $PY -m pip uninstall -y hf_xet 2>&1 | tail -1
$PY -c "import hf_xet" 2>&1 | tail -1

echo
echo "=== 下一个分片测速（**限时 240 秒**）==="
rm -rf /dev/shm/clef && mkdir -p /dev/shm/clef
cat > /dev/shm/dl2.py <<'EOF'
import os, sys, time, threading
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import hf_hub_download

def report():
    import glob
    while True:
        time.sleep(20)
        tot = sum(os.path.getsize(f) for f in
                  glob.glob('/dev/shm/clef/**/*', recursive=True)
                  if os.path.isfile(f))
        print('    [%3ds] %.0f MB' % (time.time() - T0, tot / 1e6), flush=True)
threading.Thread(target=report, daemon=True).start()

T0 = time.time()
try:
    p = hf_hub_download('Cloudflare/clef', 'model-00002-of-00012.safetensors',
                        local_dir='/dev/shm/clef')
    dt = time.time() - T0
    sz = os.path.getsize(p)
    print('  %.0f MB in %.0f s = %.1f MB/s' % (sz / 1e6, dt, sz / 1e6 / dt))
    print('  -> 55 GB 要约 %.0f 分钟' % (55296 / (sz / 1e6 / dt) / 60))
except Exception as e:
    print('  失败：', str(e)[:100])
EOF
timeout 240 $PY /dev/shm/dl2.py 2>&1 | grep -viE 'it/s\]|B/s\]|^\s*$' | tail -10
echo "  （240 秒到点就杀了）"
du -sm /dev/shm/clef 2>/dev/null | xargs echo "  实际下了:"
