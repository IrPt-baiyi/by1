#!/bin/bash
set -u
cd /root/by1
/root/miniconda3/bin/python - <<'EOF'
import importlib.util, glob, os, sys
sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location('e2e', 'by1e2e.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)          # 会跑 main() 吗？看有没有 __main__ 保护
EOF
echo "--- 直接复刻 find_cached 的逻辑 ---"
/root/miniconda3/bin/python - <<'EOF'
import glob, os
base = os.path.expanduser('~/.cache/huggingface/hub')
roots = [d for d in glob.glob(os.path.join(base, 'models--*gpt2*')) if os.path.isdir(d)]
print('roots:', roots)
for r in roots:
    cfg = safet = None
    for p in glob.glob(os.path.join(r, '**', 'config.json'), recursive=True):
        cfg = p
    for p in glob.glob(os.path.join(r, '**', 'model.safetensors'), recursive=True):
        if os.path.getsize(p) > 0:
            safet = p
    print('  cfg  :', cfg)
    print('  safet:', safet)
    print('  返回 :', bool(cfg and safet))
EOF
