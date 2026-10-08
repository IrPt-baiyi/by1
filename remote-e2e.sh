#!/bin/bash
set -u
cd /root/by1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python

echo "=== 重新下 tiny-gpt2（这次要带上权重文件）==="
$PY - <<'EOF' 2>&1 | tail -6
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
p = snapshot_download('sshleifer/tiny-gpt2',
                      allow_patterns=['*.json', '*.txt', '*.bin',
                                      '*.safetensors'])
print('  dir =', p)
import glob
for f in sorted(glob.glob(os.path.join(p, '*'))):
    print('    %-28s %8d' % (os.path.basename(f), os.path.getsize(f)))
EOF

echo
echo "=== 现在 find_cached 找得到吗 ==="
$PY - <<'EOF'
import glob, os
base = os.path.expanduser('~/.cache/huggingface/hub')
for r in glob.glob(os.path.join(base, 'models--*gpt2*')):
    cfg = safet = None
    for p in glob.glob(os.path.join(r, '**', 'config.json'), recursive=True):
        cfg = p
    for p in glob.glob(os.path.join(r, '**', 'model.safetensors'),
                       recursive=True):
        if os.path.getsize(p) > 0:
            safet = p
    print('  cfg  :', bool(cfg), ' safet:', bool(safet))
    if cfg and safet:
        print('  **可以跑了**')
EOF

echo
echo "=== 跑 by1e2e ==="
$PY by1e2e.py 2>&1 | tail -16
