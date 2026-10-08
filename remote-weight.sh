#!/bin/bash
# 限时 100 秒：直接 curl 几个**权重**文件的前几 MB，看真实速度。
# config.json 太小，测不出 Xet 的问题 —— 权重才走那条路。
set -u
M=https://hf-mirror.com
test_one() {
  repo=$1; file=$2; label=$3
  out=$(timeout 20 curl -sL -o /dev/null --max-time 16 -r 0-5000000 \
        -w "%{speed_download} %{size_download} %{http_code}" \
        "$M/$repo/resolve/main/$file" 2>/dev/null)
  spd=$(echo "$out" | cut -d' ' -f1 | cut -d. -f1)
  sz=$(echo "$out" | cut -d' ' -f2)
  code=$(echo "$out" | cut -d' ' -f3)
  if [ "${sz:-0}" -gt 1000000 ] 2>/dev/null; then
    echo "  ✓ $label  $((spd/1024)) KB/s（下了 $((sz/1000000)) MB）"
  else
    echo "  ✗ $label  只下了 ${sz:-0} 字节，HTTP $code —— **基本不通**"
  fi
}

echo "=== 权重文件的真实速度（各限时 16 秒）==="
test_one Cloudflare/clef      model-00002-of-00012.safetensors  "clef        26.9B"
test_one Qwen/Qwen3.8-27B     model-00001-of-00014.safetensors  "qwen38      27.3B"
test_one amd/Instella-3B      model-00001-of-00002.safetensors  "instella-3b  3.1B"
test_one jingyaogong/minimind-3 model.safetensors               "minimind-3  68.8M"
