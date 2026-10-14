#!/usr/bin/env python3
"""instella-ref-logits.py -- reference-side logits dump (T18 cross-check).

Loads AMD's own modeling (llamacpp/modeling_instella.py) + the real weights,
runs one forward on the EXACT token ids (read from the llama-side dump's
stderr echo), prints one float per line (same format as llama-eval-logits).

Weight fidelity note: source weights are bf16; the F16 GGUF holds them
losslessly (bf16 mantissa 8 bits <= f16 mantissa 10 bits, all values in
range). torch bf16 matmul accumulates in fp32, llama.cpp computes fp32 --
so the two sides differ only by op ordering / accumulation order.

Usage:
  python tools/instella-ref-logits.py model-cache/Instella-3B <tokens-file> <out>
where <tokens-file> holds the "tokens(N): id id ..." line (llama-side stderr).
"""
import glob
import importlib.util
import json
import sys
from pathlib import Path


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, tok_file, out = Path(argv[0]), Path(argv[1]), Path(argv[2])

    ids = []
    for line in Path(tok_file).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("tokens("):
            ids = [int(x) for x in line.split(":", 1)[1].split()]
            break
    if not ids:
        print("  [FAIL] tokens 回显没找到 —— 跑 llamacpp/llama-eval-logits.exe 先")
        return 2
    print(f"  tokens({len(ids)}): {ids}")

    spec = importlib.util.spec_from_file_location(
        "mi", str(Path(__file__).resolve().parent.parent / "llamacpp" / "modeling_instella.py"))
    mi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mi)

    import torch
    cfg = mi.InstellaConfig(**json.loads(
        (src / "config.json").read_text(encoding="utf-8")))
    model = mi.InstellaForCausalLM(cfg)
    sd = {}
    from safetensors.torch import load_file
    for f in sorted(glob.glob(str(src / "model-*.safetensors"))):
        sd.update(load_file(f))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [FAIL] state_dict 对不上: missing={missing[:4]} "
              f"unexpected={unexpected[:4]}")
        return 1
    model = model.to(torch.bfloat16).eval()
    print(f"  权重装载 {len(sd)} 个张量（bf16）")

    with torch.no_grad():
        # use_cache=False：transformers 5.x 的 DynamicCache 没有
        # to_legacy_cache（参照代码是 4.x 时代的）—— 只要前向 logits，
        # 不走 cache 路径就没有这个兼容坑
        out_t = model(torch.tensor([ids], dtype=torch.long),
                      use_cache=False).logits[0, -1]
    vals = out_t.to(torch.float32)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(f"{v:.9g}" for v in vals.tolist()) + "\n",
                        encoding="utf-8")
    print(f"  参照 dump 写了 {out_path}（{vals.numel()} 个值）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
