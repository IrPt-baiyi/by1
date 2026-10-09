#!/bin/bash
# 限时 90 秒：到底走没走 hf-mirror？
set -u
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
PY=/root/miniconda3/bin/python
pkill -f 'dl[0-9].py' 2>/dev/null; sleep 1

echo "=== env 里有什么 ==="
echo "  HF_ENDPOINT=$HF_ENDPOINT"
echo "  HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
echo "  hf_xet 装了吗: $($PY -c 'import hf_xet;print("装了")' 2>&1 | tail -1)"

echo
echo "=== huggingface_hub 认为 endpoint 是哪个 ==="
timeout 60 $PY - <<'EOF' 2>&1 | tail -6
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'
import huggingface_hub as H
from huggingface_hub import constants
print('  hub 版本 :', H.__version__)
print('  ENDPOINT :', constants.ENDPOINT)
print('  有 hf_xet:', getattr(constants, 'HF_HUB_DISABLE_XET', '?'),
      '/', getattr(constants, 'is_xet_available', lambda: '?')())
EOF

echo
echo "=== 直接问：这个分片的下载 URL 指向哪（限时 60 秒）==="
timeout 60 $PY - <<'EOF' 2>&1 | tail -4
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import hf_hub_url
u = hf_hub_url('Cloudflare/clef', 'model-00002-of-00012.safetensors')
print('  URL:', u[:120])
print('  指向镜像吗:', 'hf-mirror' in u)
EOF
