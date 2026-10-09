#!/bin/bash
# 在 A800 上跑真模型的前向。**这是第一次能做这件事。**
#
# 为什么现在才能：本地跑 instella-3b 要 12.4 GB fp32，勉强；
# A800 上 6.1 GB bf16，轻松。而它是**唯一一个有真实维度、
# 对官方实现验过前向的模型**（0.000e+00）—— 却从来没在显卡上跑过。
set -u
cd /root/by1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python

echo "=== 磁盘够不够 ==="
df -h / | tail -1

echo
echo "=== 下 Instella-3B 的 config + 权重头（不下权重）==="
$PY - <<'EOF' 2>&1 | tail -8
import os, json
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download
try:
    p = hf_hub_download('amd/Instella-3B', 'config.json')
    print('  config  下到', p)
    print('  ', json.dumps(json.load(open(p)))[:200])
except Exception as e:
    print('  失败：', str(e)[:160])
EOF

echo
echo "=== by1instella.py（真维度、对官方实现）==="
$PY by1instella.py 2>&1 | tail -14
