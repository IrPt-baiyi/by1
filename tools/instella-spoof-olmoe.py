#!/usr/bin/env python3
"""instella-spoof-olmoe.py —— Instella-3B -> **OlmoE 单专家**（数学等价改写）。

为什么等价：olmoe.cpp 与 Instella 同型 ——
  · 前置 norm（olmoe.cpp 75/131 vs modeling_instella.py 698-721）
  · q/k norm **拆头前全宽**（olmoe.cpp 91/95 在 reshape 99-101 之前；
    modeling_instella.py 401-402 同序）
  · rope / silu / 1/sqrt(head_dim) 相同
唯一差别是 dense vs MoE。把 dense 的 gate/up/down 塞进 **1 个专家**、
router 补一行全零权重（[1, hidden]）：softmax 单专家 = 恒 1.0，
x·1.0 位级不变 —— **MoE 壳装 dense 芯，数学逐位同 dense**。

对照组：`instella-spoof-olmo2.py`（AMD 社区配方）已判**结构不忠实**
（前置 norm 塞进夹心槽）。本配方的忠实性不靠嘴 —— 判卷人是
logits 对拍（llama-cli vs by1 的 HF 参照前向）。

只读原目录、只写副本目录。用法：
  python tools/instella-spoof-olmoe.py model-cache/Instella-3B model-cache/Instella-3B-olmoe
"""
import json
import re
import shutil
import sys
from pathlib import Path


def rename_key(k):
    """dense mlp -> 单专家；norm 里 pre_feedforward 换同位置的 llama 式名
    （都是「FFN 前那个 norm」，换名不换位置）；其余原名
    （q_norm/k_norm/pre_attention_layernorm 在 OLMOE 的张量集里有家）。"""
    if "pre_feedforward_layernorm" in k:
        return k.replace("pre_feedforward_layernorm", "post_attention_layernorm")
    m = re.match(r"(model\.layers\.\d+\.)mlp\.(gate_proj|up_proj|down_proj)"
                 r"(\.weight)$", k)
    if m:
        return f"{m.group(1)}mlp.experts.0.{m.group(2)}{m.group(3)}"
    return k


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    src, dst = Path(argv[0]), Path(argv[1])
    if not src.is_dir():
        print(f"  [FAIL] 源目录不存在：{src}")
        return 2
    dst.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    old_arch = cfg.get("architectures")
    cfg["architectures"] = ["OlmoeForCausalLM"]
    cfg["model_type"] = "olmoe"
    # 单专家：total=1、top-k=1；无共享专家（shared_expert_every_n_layers=0）
    cfg["num_local_experts"] = 1
    cfg["num_experts"] = 1
    cfg["num_experts_per_tok"] = 1
    cfg["shared_expert_every_n_layers"] = 0
    cfg.pop("torch_dtype", None)
    (dst / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  config: {old_arch} -> ['OlmoeForCausalLM'] + 单专家参数")

    for f in sorted(src.iterdir()):
        if f.suffix in (".json", ".txt", ".model") and f.name != "config.json" \
                and f.name != "model.safetensors.index.json":
            shutil.copy2(f, dst / f.name)
            print(f"  拷贝 {f.name}")

    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch
    hidden = int(cfg["hidden_size"])
    total, routers = 0, 0
    new_index = {"weight_map": {}}
    for f in sorted(src.glob("*.safetensors")):
        out = {}
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                t = sf.get_tensor(k)
                nk = rename_key(k)
                out[nk] = t
                new_index["weight_map"][nk] = f.name
                m = re.match(r"model\.layers\.\d+\.mlp\.experts\.0\.down_proj"
                             r"\.weight$", nk)
                if m:
                    bid = int(re.search(r"layers\.(\d+)\.", nk).group(1))
                    # router：全零 [n_expert=1, hidden] —— logits 恒 0 ->
                    # softmax([0]) = 1.0 恒成立
                    rname = f"model.layers.{bid}.mlp.gate.weight"
                    out[rname] = torch.zeros((1, hidden), dtype=t.dtype)
                    new_index["weight_map"][rname] = f.name
                    routers += 1
        save_file(out, str(dst / f.name))
        total += len(out)
        print(f"  {f.name}: {len(out)} 个张量落盘（含合成 router）")

    (dst / "model.safetensors.index.json").write_text(
        json.dumps(new_index, indent=2) + "\n", encoding="utf-8")

    names = []
    for f in sorted(dst.glob("*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            names.extend(sf.keys())
    n_exp = sum(1 for k in names if ".mlp.experts.0." in k)
    n_dense_left = sum(1 for k in names
                       if re.search(r"\.mlp\.(gate_proj|up_proj|down_proj)\.weight$", k))
    n_ffn_norm = sum(1 for k in names if "post_attention_layernorm" in k)
    print("  ---- 核验 ----")
    print(f"  张量总数 {total}（399 + 36 router = 435）")
    print(f"  专家张量 {n_exp}（36 层 x 3 = 108）· router {routers}（36）· "
          f"残留 dense mlp 名 {n_dense_left}（必须 0）· "
          f"FFN 前 norm {n_ffn_norm}（36，改名后）")
    if total != 399 + 36 or n_exp != 108 or routers != 36 or n_dense_left \
            or n_ffn_norm != 36:
        print("  [FAIL] 核验不过")
        return 1
    print("  [PASS] 单专家改写落盘（数学等价性归 logits 对拍判）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
