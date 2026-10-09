#!/bin/bash
# HOME 设了吗？缓存找得到吗？
set -u
echo "HOME=[${HOME:-未设}]"
echo "PWD=$(pwd)"
echo "---"
ls -d /root/.cache/huggingface/hub/models--*gpt2* 2>&1 | head -2
echo "---"
/root/miniconda3/bin/python - <<'EOF'
import os
p = os.path.expanduser('~/.cache/huggingface/hub')
print('expanduser ->', p)
print('isdir      ->', os.path.isdir(p))
import glob
print('glob       ->', glob.glob(os.path.join(p, 'models--*gpt2*'))[:2])
EOF
