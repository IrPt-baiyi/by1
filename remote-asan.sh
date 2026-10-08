#!/bin/bash
# 用 sanitizer 找 C 后端在 Linux 上算错的原因。
#
# **判据**：Windows 上 6 个模型全过，Linux 上全错 —— 同一份代码、
# 同一份 IR、同一组权重。那不是精度问题（那是 1e-7 量级），
# 相对误差 1.0~1.5 说明**读到了垃圾**。
#
# 这种东西本地跑一万遍也发现不了 —— Windows 上"恰好对"。
# 而 Linux 有 ASan/UBSan，一次就能指到行。
set -u
cd /root/by1 || exit 1
PY=/root/miniconda3/bin/python
F=${1:-llama-shaped.by1}

echo "=== 生成 C（by1c 自己会编译一次，那个我们不用）==="
rm -rf /tmp/asan && mkdir -p /tmp/asan
$PY by1c.py "$F" --gcc /usr/bin/gcc --seq 8 --workdir /tmp/asan 2>&1 | tail -3

echo
echo "=== 用 ASan + UBSan 重编 ==="
cd /tmp/asan
gcc -O1 -g -fno-omit-frame-pointer \
    -fsanitize=address,undefined -fno-sanitize-recover=all \
    -o model_asan model.c -lm 2>&1 | head -20
if [ ! -x model_asan ]; then
  echo "  **编译失败** —— sanitizer 自己就报错了，那就是线索"
  exit 1
fi
echo "  编译成功"

echo
echo "=== 跑（ASan 会指出越界/未初始化）==="
ASAN_OPTIONS=detect_leaks=0:halt_on_error=0 UBSAN_OPTIONS=print_stacktrace=1 \
  ./model_asan 8 2>&1 | head -60

echo
echo "=== 顺便：把 -O0 和 -O2 各跑一次，看差多少 ==="
for opt in O0 O2; do
  gcc -$opt -o m_$opt model.c -lm 2>/dev/null
  if [ -x m_$opt ]; then
    r=$(./m_$opt 8 2>&1 | head -1)
    echo "  -$opt : $r"
  fi
done
