#!/usr/bin/env python3
"""instella-greedy.py -- by1 side of the T25-1 greedy judge.

Reference forward (AMD modeling + real weights) + by1sample.greedy loop;
compare token-for-token against the llama side (llama-eval-logits --greedy,
which runs llama.cpp's own argmax continuation). Greedy is deterministic,
so "same seed" degenerates to exact token equality -- that IS the judge.

Usage:
  python tools/instella-greedy.py model-cache/Instella-3B <tokens-file> <n_gen>
"""
import glob
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, tok_file, n_gen = Path(argv[0]), Path(argv[1]), int(argv[2])

    ids = []
    for line in Path(tok_file).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("tokens("):
            ids = [int(x) for x in line.split(":", 1)[1].split()]
            break
    if not ids:
        print("  [FAIL] tokens 回显没找到")
        return 2

    sys.path.insert(0, str(ROOT / "src"))
    import by1sample

    spec = importlib.util.spec_from_file_location(
        "mi", str(ROOT / "llamacpp" / "modeling_instella.py"))
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
        print(f"  [FAIL] state_dict 对不上: {missing[:3]} {unexpected[:3]}")
        return 1
    model = model.to(torch.bfloat16).eval()

    seq = list(ids)
    gen = []
    for _ in range(n_gen):
        with torch.no_grad():
            logits = model(torch.tensor([seq], dtype=torch.long),
                           use_cache=False).logits[0, -1]
        t = by1sample.greedy(logits.to(torch.float32).tolist())
        gen.append(t)
        seq.append(t)
    print("gen-ids:" + "".join(f" {t}" for t in gen))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
