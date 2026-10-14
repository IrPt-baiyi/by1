#!/usr/bin/env python3
"""cmp_mel.py -- numeric comparison of the two mel boundaries.

Reads "mel n_mel n_len" + one float per line (mel-major) from each file,
aligns on the first `n` frames, and reports the difference statistics.

Usage: python cmp_mel.py <a.txt> <b.txt> [n_frames]
"""
import sys


def load(p):
    with open(p, encoding="utf-8") as fh:
        head = fh.readline().split()
        n_mel, n_len = int(head[1]), int(head[2])
        vals = [float(x) for x in fh.read().split()]
    assert len(vals) == n_mel * n_len, (p, len(vals), n_mel * n_len)
    return n_mel, n_len, vals


def main(argv):
    a_path, b_path = argv[0], argv[1]
    n = int(argv[2]) if len(argv) > 2 else None
    n_mel_a, n_len_a, a = load(a_path)
    n_mel_b, n_len_b, b = load(b_path)
    print("A %s: mel %d x %d" % (a_path, n_mel_a, n_len_a))
    print("B %s: mel %d x %d" % (b_path, n_mel_b, n_len_b))
    if n_mel_a != n_mel_b:
        print("mel bin mismatch -> cannot compare"); return 1
    n = min(n_len_a, n_len_b) if n is None else n
    if n > n_len_a or n > n_len_b:
        print("requested %d frames exceeds available (%d/%d)" % (n, n_len_a, n_len_b)); return 1

    # slice first n frames, mel-major
    def slice_first(a, n_len, n):
        out = []
        for m in range(n_mel_a):
            out.extend(a[m * n_len: m * n_len + n])
        return out

    sa, sb = slice_first(a, n_len_a, n), slice_first(b, n_len_b, n)
    diffs = [abs(x - y) for x, y in zip(sa, sb)]
    mx, mean = max(diffs), sum(diffs) / len(diffs)
    col0 = [abs(a[m * n_len_a + 0] - b[m * n_len_b + 0]) for m in range(n_mel_a)]
    exact = sum(1 for d in diffs if d == 0.0)
    print("compared frames        : %d (mel %d x %d)" % (n, n_mel_a, n))
    print("max |d|                : %.6g" % mx)
    print("mean |d|               : %.6g" % mean)
    print("bit-identical values   : %d / %d" % (exact, len(diffs)))
    print("column 0 max |d|       : %.6g" % max(col0))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
