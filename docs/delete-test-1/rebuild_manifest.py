#!/usr/bin/env python3
"""rebuild_manifest.py - reconstruct the 9 paths taken out of the by1 repo.

Every path in MANIFEST.txt is rebuilt from its public upstream source, using
the same derivation the repo's `refs/` snapshots use.  Nothing is invented.

    python rebuild_manifest.py --out build

Sources (HuggingFace):
    openai-community/gpt2                          -> refs/openai-community__gpt2.config.json
                                                      refs/openai-community__gpt2.tensors.json
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B        -> refs/deepseek-ai__DeepSeek-R1-Distill-Qwen-7B.config.json
    nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
                                                   -> refs/nvidia__NVIDIA-Nemotron-...tensors.json
    Qwen/Qwen3-VL-2B-Thinking                      -> refs/Qwen__Qwen3-VL-2B-Thinking.tensors.json
    zai-org/GLM-4.7-Flash                          -> refs/zai-org__GLM-4.7-Flash.tensors.json
    poolside/Laguna-XS-2.1-GGUF                    -> refs/poolside__Laguna-XS-2_1.gguf-tensors.json

Two paths are hand-written, not fetched, and carry no upstream to derive from:
    models/mixtral-shaped.by1, models.tsv
They must be authored against the by1 sources, which are not present here, so
this script takes them verbatim from `handwritten/`; see README-rebuild.md.

Derivation rules (all verified byte-for-byte against the repo's own snapshots):
  config.json           upstream bytes verbatim; the deepseek one is the
                        upstream text with LF rewritten to CRLF.
  *.tensors.json        safetensors headers, `data_offsets` dropped, keeping
                        dtype+shape.  Two writers:
                          compact : json.dumps(..., separators=(",", ":"))
                          pretty  : json.dumps(..., indent=0) with CRLF
                        gpt2 keeps the header's own key order; Nemotron keeps
                        the shard concatenation order; GLM and Qwen3-VL sort
                        each shard's key block before concatenating.
  *.gguf-tensors.json   the GGUF tensor table: shape reversed (ggml ne[] order),
                        ggml_type kept as the numeric enum, one compact line.
"""

import argparse
import json
import os
import struct
import sys
import urllib.request

HF = "https://huggingface.co"
UA = {"User-Agent": "by1-rebuild/1.0"}


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------

def fetch(url, byte_range=None, timeout=300, attempts=3):
    last = None
    for _ in range(attempts):
        try:
            headers = dict(UA)
            if byte_range:
                headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:          # noqa: BLE001 - retried below
            last = exc
    raise RuntimeError(f"GET {url} failed: {last}")


def resolve(repo, path):
    return f"{HF}/{repo}/resolve/main/{path}"


def repo_sha(repo):
    """Current commit of a repo, recorded so a rebuild is traceable."""
    try:
        meta = json.loads(fetch(f"{HF}/api/models/{repo}", timeout=60).decode("utf-8"))
        return meta.get("sha")
    except Exception:                     # noqa: BLE001 - metadata is optional
        return None


# --------------------------------------------------------------------------
# safetensors
# --------------------------------------------------------------------------

def safetensors_header(url):
    """(tensor table, header byte length) from a safetensors file's front."""
    (n,) = struct.unpack("<Q", fetch(url, (0, 7)))
    header = fetch(url, (8, 8 + n - 1))
    table = json.loads(header.decode("utf-8"))
    table.pop("__metadata__", None)
    return table, n


def shard_names(repo, count, prefix="model"):
    return [f"{prefix}-{i:05d}-of-{count:05d}.safetensors" for i in range(1, count + 1)]


def collect_shards(repo, count, sort_each):
    """Concatenate shard tensor tables in shard order."""
    shards = []
    for i, name in enumerate(shard_names(repo, count), 1):
        table, n = safetensors_header(resolve(repo, name))
        keys = sorted(table) if sort_each else list(table)
        print(f"    [{i:>2}/{count}] {name}: {len(table):>5} tensors, header {n} B",
              file=sys.stderr)
        shards.append((keys, table))
    return shards


def compact(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def pretty(obj):
    return json.dumps(obj, indent=0, ensure_ascii=False).replace("\n", "\r\n").encode("utf-8")


def dtype_shape(table):
    return {k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in table.items()}


# --------------------------------------------------------------------------
# gguf
# --------------------------------------------------------------------------

def _read_val(cur, t):
    if t == 0:
        return cur.u8()
    if t == 1:
        return cur.i8()
    if t == 2:
        return cur.u16()
    if t == 3:
        return cur.i16()
    if t == 4:
        return cur.u32()
    if t == 5:
        return cur.i32()
    if t == 6:
        return cur.f32()
    if t == 7:
        return bool(cur.u8())
    if t == 8:
        return cur.string()
    if t == 9:
        et = cur.u32()
        n = cur.u64()
        return [_read_val(cur, et) for _ in range(n)]
    if t == 10:
        return cur.u64()
    if t == 11:
        return cur.i64()
    if t == 12:
        return cur.f64()
    raise ValueError(f"unknown GGUF value type {t}")


class _Cur:
    def __init__(self, buf):
        self.buf, self.pos = buf, 0

    def _take(self, n):
        out = self.buf[self.pos:self.pos + n]
        if len(out) != n:
            raise EOFError(f"GGUF header truncated at {self.pos}")
        self.pos += n
        return out

    def _unpack(self, fmt, n):
        return struct.unpack("<" + fmt, self._take(n))[0]

    def u8(self):
        return self._unpack("B", 1)

    def i8(self):
        return self._unpack("b", 1)

    def u16(self):
        return self._unpack("H", 2)

    def i16(self):
        return self._unpack("h", 2)

    def u32(self):
        return self._unpack("I", 4)

    def i32(self):
        return self._unpack("i", 4)

    def f32(self):
        return self._unpack("f", 4)

    def u64(self):
        return self._unpack("Q", 8)

    def i64(self):
        return self._unpack("q", 8)

    def f64(self):
        return self._unpack("d", 8)

    def string(self):
        return self._take(self.u64()).decode("utf-8", "replace")


def gguf_tensor_table(url, probe_bytes=24 * 1024 * 1024):
    """Tensor table of a GGUF file, read from its front only."""
    buf = fetch(url, (0, probe_bytes - 1))
    cur = _Cur(buf)
    magic = cur._take(4)
    if magic != b"GGUF":
        raise ValueError(f"{url}: not a GGUF file ({magic!r})")
    cur.u32()                              # version
    n_tensors = cur.u64()
    n_kv = cur.u64()
    for _ in range(n_kv):
        cur.string()
        _read_val(cur, cur.u32())
    table = {}
    for _ in range(n_tensors):
        name = cur.string()
        dims = [cur.u64() for _ in range(cur.u32())]
        ggml_type = cur.u32()
        cur.u64()                          # offset; position is what matters
        table[name] = {"shape": dims[::-1], "ggml_type": ggml_type}
    return table


# --------------------------------------------------------------------------
# the nine paths
# --------------------------------------------------------------------------

def build(out_dir):
    results = {}

    def emit(rel, data, provenance):
        dst = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(dst) or out_dir, exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(data)
        results[rel] = (len(data), provenance)
        print(f"  {rel:<64} {len(data):>8} B   {provenance}", file=sys.stderr)

    # ---- 1. openai-community/gpt2 -----------------------------------------
    repo = "openai-community/gpt2"
    print(f"[{repo}] {repo_sha(repo)}", file=sys.stderr)
    emit("refs/openai-community__gpt2.config.json",
         fetch(resolve(repo, "config.json")),
         "upstream config.json verbatim")
    table, _ = safetensors_header(resolve(repo, "model.safetensors"))
    emit("refs/openai-community__gpt2.tensors.json",
         compact(dtype_shape(table)),
         "safetensors header, data_offsets dropped, compact, header key order")

    # ---- 2. deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -----------------------
    repo = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    print(f"[{repo}] {repo_sha(repo)}", file=sys.stderr)
    cfg = fetch(resolve(repo, "config.json"))
    emit("refs/deepseek-ai__DeepSeek-R1-Distill-Qwen-7B.config.json",
         cfg.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"),
         "upstream config.json, LF -> CRLF")

    # ---- 3. nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 ------------
    repo = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    print(f"[{repo}] {repo_sha(repo)}", file=sys.stderr)
    shards = collect_shards(repo, 14, sort_each=False)
    table = {}
    for keys, t in shards:
        for k in keys:
            table[k] = t[k]
    emit("refs/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.tensors.json",
         compact(dtype_shape(table)),
         "14 safetensors headers concatenated, compact, dropped offsets")

    # ---- 4. Qwen/Qwen3-VL-2B-Thinking ------------------------------------
    repo = "Qwen/Qwen3-VL-2B-Thinking"
    print(f"[{repo}] {repo_sha(repo)}", file=sys.stderr)
    table, _ = safetensors_header(resolve(repo, "model.safetensors"))
    shapes = {k: table[k]["shape"] for k in sorted(table)}
    emit("refs/Qwen__Qwen3-VL-2B-Thinking.tensors.json",
         pretty(shapes),
         "safetensors header, names sorted, shapes only, indent=0 + CRLF")

    # ---- 5. zai-org/GLM-4.7-Flash ----------------------------------------
    repo = "zai-org/GLM-4.7-Flash"
    print(f"[{repo}] {repo_sha(repo)}", file=sys.stderr)
    shards = collect_shards(repo, 48, sort_each=True)
    keys = [k for ks, _ in shards for k in ks]
    emit("refs/zai-org__GLM-4.7-Flash.tensors.json",
         pretty({k: None for k in keys}),
         "48 safetensors headers, per-shard sorted, null values, indent=0 + CRLF")

    # ---- 6. poolside/Laguna-XS-2_1 (GGUF) --------------------------------
    repo = "poolside/Laguna-XS-2.1-GGUF"
    print(f"[{repo}] {repo_sha(repo)}", file=sys.stderr)
    table = gguf_tensor_table(resolve(repo, "Laguna-XS-2.1-Q4_K_M.gguf"))
    emit("refs/poolside__Laguna-XS-2_1.gguf-tensors.json",
         compact(table),
         "Q4_K_M GGUF tensor table, dims reversed, ggml_type numeric, compact")

    # ---- 7-8. hand-written, no upstream ----------------------------------
    for rel in ("mixtral-shaped.by1", "models.tsv"):
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "handwritten", rel)
        if os.path.isfile(src):
            with open(src, "rb") as fh:
                emit(rel, fh.read(), "hand-written (see README-rebuild.md)")
        else:
            print(f"  {rel}: NOT REBUILT - hand-written, needs the by1 sources",
                  file=sys.stderr)
            results[rel] = (None, "hand-written - missing")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build", help="output directory")
    args = ap.parse_args()
    results = build(args.out)
    print(json.dumps({k: v[0] for k, v in results.items()}, indent=2))
    missing = [k for k, v in results.items() if v[0] is None]
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
