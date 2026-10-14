#!/usr/bin/env python3
"""qwen3asr-mel-ref.py -- T21 judge: audio mel boundary (transformers side).

Same synthetic sine as tools/audio-mel-probe.cpp (440 Hz, N samples at the
processor's sample rate), fed through the HF processor's feature
extractor. Prints "mel n_mel n_len" + one float per line (mel-major).

Usage:
  python tools/qwen3asr-mel-ref.py <hf-dir> <n_samples> <out.txt>
"""
import sys
from pathlib import Path


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    hf_dir, n_s, out_path = argv[0], int(argv[1]), argv[2]
    import numpy as np
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(hf_dir)
    fe = getattr(proc, "feature_extractor", None) or proc
    sample_rate = int(getattr(fe, "sampling_rate", 16000) or 16000)
    t = np.arange(n_s, dtype=np.float32) / sample_rate
    sine = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    feats = fe(raw_speech=sine, sampling_rate=sample_rate,
               padding="longest", return_tensors="np")
    key = "input_features" if "input_features" in feats else "input_mel"
    mel = np.asarray(feats[key], dtype=np.float64)
    # HF conventions differ per family: find the (128, T) plane
    mel2 = mel[0] if mel.ndim == 3 else mel
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("mel %d %d\n" % (mel2.shape[0], mel2.shape[-1]))
        fh.write("".join("%.9g\n" % v for v in mel2.reshape(-1).tolist()))
    print("  参照 mel %s 写了 %s" % (mel2.shape, out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
