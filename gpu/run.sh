#!/usr/bin/env bash
# Run the real CUDA validation and published-checkpoint comparison.
# Set PY to select a project virtualenv, for example: PY=.venv/bin/python bash gpu/run.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}

"$PY" src/checks/by1all.py --quick
"$PY" src/checks/by1gpusmoke.py --dtype float32 --report cgen/gpu-smoke-fp32.json
"$PY" src/checks/by1gpusmoke.py --dtype bfloat16 --report cgen/gpu-smoke-bf16.json
"$PY" src/checks/by1complete.py
"$PY" src/checks/by1gpubench.py --report cgen/gpu-benchmark.json
"$PY" src/checks/by1gputrain.py --report cgen/gpu-real-traincheck.json
