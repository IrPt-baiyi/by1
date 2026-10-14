#!/usr/bin/env python3
"""qwen3asr-embd-ref.py -- T21 judge: audio tower boundary (transformers side).

Same synthetic sine as tools/mtmd-embd-probe.cpp --sine (440 Hz, N samples
at 16 kHz). Hooks the audio projector output (the embeddings fed to the LM)
and dumps "embd n d" + one float per line.

Usage:
  python tools/qwen3asr-embd-ref.py <hf-dir> <n_samples> <out.txt>
"""
import sys
from pathlib import Path


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    hf_dir, n_s, out_path = argv[0], int(argv[1]), argv[2]
    import numpy as np
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3ASRForConditionalGeneration as M
    except ImportError:                      # 5.x 某些版本只在子模块里
        from transformers.models.qwen3_asr.modeling_qwen3_asr import (
            Qwen3ASRForConditionalGeneration as M)

    proc = AutoProcessor.from_pretrained(hf_dir)
    model = M.from_pretrained(
        hf_dir, dtype=torch.float32, trust_remote_code=True).eval()

    t = np.arange(n_s, dtype=np.float32) / 16000.0
    sine = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    fe = getattr(proc, "feature_extractor", proc)
    # 手造 input_ids：占位检查数的是 config.audio_token_id（151676）——
    # 给字符串会被处理器改写，id 直给才稳
    n_ph = 13 if n_s >= 16000 else max(1, n_s * 13 // 16000)
    fe = getattr(proc, "feature_extractor", proc)
    bf = fe(raw_speech=sine, sampling_rate=16000, padding="longest",
            return_tensors="pt")
    inputs = {
        "input_ids": torch.tensor([[int(model.config.audio_token_id)] * n_ph]),
        "input_features": bf["input_features"],
        "input_features_mask": torch.ones(
            bf["input_features"].shape[0], bf["input_features"].shape[-1],
            dtype=torch.long),
    }

    cap = {}

    def hook(name):
        def fn(mod, inp, out):
            v = out
            if not torch.is_tensor(v):
                v = getattr(out, "last_hidden_state", None)
                if v is None and isinstance(out, tuple):
                    v = out[0]
            if v is not None and torch.is_tensor(v):
                cap[name] = v.detach().float().cpu()
        return fn

    handles = []
    for name, mod in model.named_modules():
        low = name.lower()
        if any(k in low for k in ("projector", "audio_tower", "encoder")) \
                and hasattr(mod, "forward"):
            handles.append(mod.register_forward_hook(hook(name)))

    with torch.no_grad():
        model(**{k: v for k, v in inputs.items()
                 if k in ("input_ids", "input_features",
                          "input_features_mask", "attention_mask")})
    for h in handles:
        h.remove()

    print("captured modules:", list(cap))
    # pick the projector-ish output (widest last-dim == LM hidden)
    best = max(cap.items(), key=lambda kv: kv[1].shape[-1])
    name, v = best
    e = v.reshape(-1, v.shape[-1])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("embd %d %d\n" % (e.shape[0], e.shape[1]))
        fh.write("".join("%.9g\n" % x for x in e.reshape(-1).tolist()))
    print("  参照嵌入 %s from %s -> %s" % (tuple(e.shape), name, out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
