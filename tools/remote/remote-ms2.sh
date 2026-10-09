#!/bin/bash
# 装 modelscope + 用真权重分片测速。**全程限时。**
set -u
PY=/root/miniconda3/bin/python

echo "=== ① 装 modelscope（限时 180 秒）==="
timeout 180 $PY -m pip install -q modelscope 2>&1 | tail -2
$PY -c "import modelscope;print('  版本', modelscope.__version__)" 2>&1 | tail -1

echo
echo "=== ② 列出 Qwen3.8-27B 的文件（限时 90 秒）==="
timeout 90 $PY - <<'EOF' 2>&1 | tail -14
from modelscope.hub.api import HubApi
try:
    api = HubApi()
    fs = api.get_model_files('Qwen/Qwen3.8-27B', recursive=True)
    tot = 0
    rows = []
    for f in fs:
        sz = f.get('Size') or 0
        tot += sz
        rows.append((sz, f.get('Path')))
    rows.sort(reverse=True)
    for sz, p in rows[:8]:
        print('  %-50s %8.1f MB' % (str(p)[:50], sz / 1e6))
    print('  ---')
    print('  合计 %.1f GB，%d 个文件' % (tot / 1e9, len(rows)))
except Exception as e:
    print('  失败：', str(e)[:200])
EOF

echo
echo "=== ③ 下一个权重分片，限量测速（**限时 60 秒**）==="
timeout 60 $PY - <<'EOF' 2>&1 | tail -4
import time, os
from modelscope import snapshot_download
t0 = time.time()
try:
    p = snapshot_download('Qwen/Qwen3.8-27B',
                          allow_patterns=['model-00001-of-*.safetensors'],
                          cache_dir='/dev/shm/ms', max_workers=4)
    dt = time.time() - t0
    tot = sum(os.path.getsize(os.path.join(dp, f))
              for dp, _, fs in os.walk(p) for f in fs)
    print('  %.0f MB in %.0f s = %.1f MB/s' % (tot / 1e6, dt, tot / 1e6 / max(dt, 1)))
except Exception as e:
    print('  失败：', str(e)[:160])
EOF
du -sm /dev/shm/ms 2>/dev/null | xargs echo "  实际下了:"
