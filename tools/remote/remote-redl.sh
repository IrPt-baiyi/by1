#!/bin/bash
# 关掉 Xet 重下。
#
# **401 那段是关键**：HF 现在默认走 Xet（cas-server.xethub.hf.co），
# 而 hf-mirror 不代理它 —— 于是主要分片全部失败，
# 只有 joint_head 那个（走老路）下来了。
#
# HF_HUB_DISABLE_XET=1 强制走经典 HTTP 下载，镜像会代理。
set -u
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

echo "=== 停掉旧的 ==="
pkill -f by1-clef-download 2>/dev/null && echo "  停了" || echo "  （没有在跑）"
rm -f /dev/shm/clef-dl.log

echo
echo "=== 先只下一个小分片，量真实速度 ==="
cat > /dev/shm/dl1.py <<'EOF'
import os, time
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import hf_hub_download
t0 = time.time()
p = hf_hub_download('Cloudflare/clef', 'model-00002-of-00012.safetensors',
                    local_dir='/dev/shm/clef')
dt = time.time() - t0
sz = os.path.getsize(p)
print('  %.0f MB in %.0f s = %.1f MB/s' % (sz / 1e6, dt, sz / 1e6 / max(dt, 1)))
print('  -> 55 GB 要约 %.0f 分钟' % (55296 / (sz / 1e6 / max(dt, 1)) / 60))
EOF
/root/miniconda3/bin/python /dev/shm/dl1.py 2>&1 | grep -viE 'it/s\]|B/s\]' | tail -5

echo
echo "=== 现在开始下全部（后台）==="
cat > /dev/shm/by1-clef-download.py <<'EOF'
import os, time
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download('Cloudflare/clef', local_dir='/dev/shm/clef',
                      allow_patterns=['*.safetensors', '*.json', '*.py'],
                      max_workers=4)
tot = sum(os.path.getsize(os.path.join(dp, f))
          for dp, _, fs in os.walk(p) for f in fs)
print('DONE %.1f GB in %.0f s' % (tot / 1e9, time.time() - t0))
EOF
nohup /root/miniconda3/bin/python /dev/shm/by1-clef-download.py \
      > /dev/shm/clef-dl.log 2>&1 &
echo "  pid=$!"
