#!/usr/bin/env python3
"""per_frame_diff.py -- where exactly does the mtmd/HF mel drift start?

Walks frame by frame and reports the max |d| over mel bins for each frame, so a
boundary effect (drift starting at a particular frame index) is visible.

Usage: python per_frame_diff.py <a.txt> <b.txt> [n_frames]
"""
import sys


def load(p):
    with open(p, encoding="utf-8") as fh:
        head = fh.readline().split()
        n_mel, n_len = int(head[1]), int(head[2])
        vals = [float(x) for x in fh.read().split()]
    return n_mel, n_len, vals


def main(argv):
    a_path, b_path = argv[0], argv[1]
    n_mel, n_len_a, a = load(a_path)
    _, n_len_b, b = load(b_path)
    n = int(argv[2]) if len(argv) > 2 else min(n_len_a, n_len_b)
    n = min(n, n_len_a, n_len_b)

    print("A mel %d x %d   B mel %d x %d   comparing %d frames"
          % (n_mel, n_len_a, n_mel, n_len_b, n))
    print("frame  max|d|      mean|d|     note")
    first_bad = None
    for t in range(n):
        d = [abs(a[m * n_len_a + t] - b[m * n_len_b + t]) for m in range(n_mel)]
        mx = max(d)
        mn = sum(d) / len(d)
        if mx > 1e-3 and first_bad is None:
            first_bad = t
        if t < 8 or t % 25 == 0 or (first_bad is not None and t < first_bad + 4):
            print("%5d  %-10.4g %-10.4g %s" % (t, mx, mn, "<-- first >1e-3" if t == first_bad else ""))
    print("first frame with max|d| > 1e-3 : %s" % first_bad)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
