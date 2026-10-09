#!/bin/bash
# 限时 90 秒：这 14 个模型，哪些不在 Xet 上（不在的才下得动）
set -u
M=https://hf-mirror.com
check() {
  repo=$1; label=$2
  loc=$(timeout 12 curl -sI --max-time 10 \
        "$M/$repo/resolve/main/config.json" 2>/dev/null |
        grep -i '^location' | head -1)
  if echo "$loc" | grep -q xethub; then
    echo "  ✗ $label   （Xet，下不动）"
  elif [ -n "$loc" ]; then
    echo "  ? $label   （跳到别处：$(echo $loc | cut -c1-60)）"
  else
    echo "  ✓ $label   （不走 Xet，可以下）"
  fi
}

echo "=== 哪些不在 Xet 上 ==="
check Cloudflare/clef                      "clef            26.9B"
check Qwen/Qwen3.8-27B                     "qwen38          27.3B"
check Qwen/Qwen3.6-35B-A3B                 "qwen36          35.5B"
check amd/Instella-3B                      "instella-3b      3.1B"
check jingyaogong/minimind-3               "minimind-3      68.8M"
check openai-community/gpt2                "gpt2             163M"
check sshleifer/tiny-gpt2                  "tiny-gpt2       0.1M"
check nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 "nemotron  30B"
check zai-org/GLM-5.3-Flash                "glm53            ?"
check inclusionAI/Ling-3.0-tiny            "ling-3.0-tiny    ?"
check stepfun-ai/Step-3.7-Flash            "step37-official  ?"
check poolside/Laguna-XS-2.1               "laguna          33.4B"
check google/gemma-4-31B                   "gemma-4-31b     32.1B"
check openai/gpt-oss-120b                  "gpt-oss-120b    117B"
