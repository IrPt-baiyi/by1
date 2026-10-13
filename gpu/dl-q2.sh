#!/usr/bin/env bash
# Q2 权重断点续传下载 —— 大文件只下一次，中断了重跑就续。
#
# 判据（T13）：命令与 revision 全固化（和 by1gpubench 的 PINNED_REVISIONS
# 一致）；重跑同一命令 = 断点续传，不是重下；下完 by1cache 记账。
#
# 用法：HF_HOME=/root/autodl-fs/hf-cache bash gpu/dl-q2.sh [fp8|mxfp4|all]
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
WHAT=${1:-all}
# 学术加速（下载 huggingface.co 时开）；下完脚本会自己关
source /etc/network_turbo 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

# ── 固定 revision（和 src/checks/by1gpubench.py 的 PINNED_REVISIONS 一致）──
REV_GPTOSS=6cee5e81ee83917806bbde320786a8fb61efebee   # openai/gpt-oss-20b（MXFP4）
REV_Q38FP8=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a    # Qwen/Qwen3.8-27B-FP8
REV_Q38=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0       # Qwen/Qwen3.8-27B（BF16 参照）

dl() {  # dl <repo> <revision>
  echo "==== $1 @ $2 ===="
  # huggingface-cli download 的缓存是内容寻址的：**重跑即续传**
  huggingface-cli download "$1" --revision "$2" \
      --include '*.json' '*.safetensors' '*.txt' '*.model' '*.jinja' \
      || echo "  !! $1 下载中断 —— 重跑本脚本即续传（不是重下）"
}

case "$WHAT" in
  fp8)   dl Qwen/Qwen3.8-27B-FP8 "$REV_Q38FP8" ;;
  mxfp4) dl openai/gpt-oss-20b "$REV_GPTOSS" ;;
  all)   dl Qwen/Qwen3.8-27B-FP8 "$REV_Q38FP8"
         dl openai/gpt-oss-20b "$REV_GPTOSS" ;;
  *) echo "用法: bash gpu/dl-q2.sh [fp8|mxfp4|all]"; exit 40 ;;
esac

unset http_proxy https_proxy 2>/dev/null || true

# 记账：清单跟权重走（文件存储里），重租卡 verify 一下就能复用
if [ -n "${HF_HOME:-}" ]; then
  "$PY" src/data/by1cache.py scan "$HF_HOME" -o "$HF_HOME/cache-manifest.json"
else
  echo "  （HF_HOME 没设 —— 记账跳过；建议 HF_HOME=/root/autodl-fs/hf-cache）"
fi
