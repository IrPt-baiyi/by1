#!/bin/bash
# 限时 60 秒：镜像那个 URL 到底给字节，还是 302 到别处？
set -u
U=https://hf-mirror.com/Cloudflare/clef/resolve/main/joint_head.safetensors

echo "=== ① 看响应头（限时 25 秒）==="
timeout 25 curl -sI -L --max-time 20 "$U" 2>&1 | grep -iE '^HTTP|^location|^content-length|^x-' | head -8

echo
echo "=== ② 前 2 MB 的实测速度（限时 30 秒）==="
timeout 30 curl -s -o /dev/null --max-time 25 -r 0-2000000 \
  -w "  速度 %{speed_download} B/s   下了 %{size_download} 字节   HTTP %{http_code}\n" \
  "$U" 2>&1 | tail -2

echo
echo "=== ③ ModelScope 上有没有这个模型（限时 20 秒）==="
timeout 20 curl -s --max-time 15 \
  'https://www.modelscope.cn/api/v1/models?PageSize=5&Name=clef' 2>&1 |
  head -c 300
echo
