#!/bin/bash
# 看清 clef 那个仓库有多大、什么格式，再决定下不下。
set -u
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python

echo "=== /dev/shm 和磁盘 ==="
df -h /dev/shm / | grep -v Filesystem

echo
echo "=== Cloudflare/clef 的文件清单 ==="
$PY - <<'EOF' 2>&1 | tail -30
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import HfApi
api = HfApi(endpoint='https://hf-mirror.com')
try:
    info = api.repo_info('Cloudflare/clef', files_metadata=True)
    tot = 0
    rows = []
    for s in info.siblings:
        sz = s.size or 0
        tot += sz
        rows.append((sz, s.rfilename))
    rows.sort(reverse=True)
    for sz, name in rows[:14]:
        print('  %-52s %9.1f MB' % (name[:52], sz / 1e6))
    print('  ---')
    print('  合计 %.1f GB，%d 个文件' % (tot / 1e9, len(rows)))
except Exception as e:
    print('  查不到：', str(e)[:200])
EOF
