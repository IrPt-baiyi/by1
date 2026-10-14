#!/usr/bin/env python3
"""norm_hypothesis.py -- is the mel drift a normalization-scope difference?

Compares the same frames under different total lengths. If the per-frame values
change when the total input length changes, the transform is global (30 s
padding applied before normalization), not per-chunk.

Usage: python norm_hypothesis.py <hf-dir>
"""
import sys
import numpy as np
from transformers import WhisperFeatureExtractor


def mel_for(fe, sr, n_s):
    t = np.arange(n_s, dtype=np.float64) / sr
    sine = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    feats = fe(sine, sampling_rate=sr, return_tensors="np")
    m = np.asarray(feats["input_features"], dtype=np.float64)
    return m[0] if m.ndim == 3 else m


def main(argv):
    hf_dir = argv[0]
    fe = WhisperFeatureExtractor.from_pretrained(hf_dir)
    sr = int(getattr(fe, "sampling_rate", 16000) or 16000)

    print("extractor: chunk_length=%s n_samples=%s hop=%s n_fft=%s nb_max_frames=%s"
          % (getattr(fe, "chunk_length", None), getattr(fe, "n_samples", None),
             getattr(fe, "hop_length", None), getattr(fe, "n_fft", None),
             getattr(fe, "nb_max_frames", None)))

    short = mel_for(fe, sr, 16000)        # 1 s
    long_ = mel_for(fe, sr, 480000)       # 30 s
    print("1 s  ->", short.shape, "  30 s ->", long_.shape)

    n = 200
    a = short[:, :n]
    b = long_[:, :n]
    d = np.abs(a - b)
    print("first %d frames, 1 s vs 30 s input:" % n)
    print("  max |d|  = %.6g" % d.max())
    print("  mean |d| = %.6g" % d.mean())
    print("  max relative = %.6g" % (d / (np.abs(b) + 1e-12)).max())

    # per-frame mean magnitude, to see whether the offset is multiplicative
    print("  frame-mean over mel bins: 1 s first=%.6g  |  30 s first=%.6g"
          % (a.mean(axis=0)[0], b.mean(axis=0)[0]))
    print("  ratio of column means (1 s / 30 s), first 5:",
          np.round(a.mean(axis=0)[:5] / (b.mean(axis=0)[:5] + 1e-12), 6))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
