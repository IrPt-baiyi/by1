#!/usr/bin/env python3
"""qwen3asr-mel-ref2.py -- mel boundary, transformers side, taken directly.

Why this exists: `AutoProcessor.from_pretrained` on this checkpoint returns a
Qwen2TokenizerFast in transformers 4.57.6 (Qwen3ASRProcessor does not exist
there), so the original qwen3asr-mel-ref.py fell back to the tokenizer and
raised. The audio preprocessing is a WhisperFeatureExtractor per
preprocessor_config.json ("feature_extractor_type": "WhisperFeatureExtractor"),
so load that class by name.

Usage:
  python tools/qwen3asr-mel-ref2.py <hf-dir> <n_samples> <out.txt>
"""
import sys


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    hf_dir, n_s, out_path = argv[0], int(argv[1]), argv[2]
    import numpy as np
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor.from_pretrained(hf_dir)
    sr = int(getattr(fe, "sampling_rate", 16000) or 16000)
    t = np.arange(n_s, dtype=np.float64) / sr
    sine = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    feats = fe(sine, sampling_rate=sr, return_tensors="np")
    mel = np.asarray(feats["input_features"], dtype=np.float64)
    mel2 = mel[0] if mel.ndim == 3 else mel
    print("  hf mel shape", mel2.shape, "sr", sr)

    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("mel %d %d\n" % (mel2.shape[0], mel2.shape[-1]))
        fh.write("".join("%.9g\n" % v for v in mel2.reshape(-1).tolist()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
