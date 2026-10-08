#!/bin/bash
# 在显卡机器上跑一遍，只输出要紧的行。
# **存在的理由**：从 Windows 那边一条条 ssh 发命令，引号会被吃好几层，
# 而且网关限制连接频率。写成一个脚本，一次跑完。
set -u
cd /root/by1 || exit 1
PY=/root/miniconda3/bin/python

echo "=== 找到了什么 ==="
$PY -c "import by1all; print('  gcc:', by1all.find_gcc())"

echo
echo "=== 全量 ==="
$PY by1all.py 2>&1 | grep -E "^  !!|^  ok C |^  ok by1|项，|失败:|全过|有失败|已知缺口" || true
