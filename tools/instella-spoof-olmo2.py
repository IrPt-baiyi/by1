#!/usr/bin/env python3
"""instella-spoof-olmo2.py —— Instella-3B 的「Olmo2 改名欺骗」配方，
**已判不忠实（2026-10-14 核验）——留档当反例，产物不许当 T18 的链**。

来源：AMD dev 社区「Breaking the Silicon Barrier: Hacking AMD's Instella-3B
into llama.cpp for HFT」（Wilderman_Duarte）——作者报 399 张量过
converter、q8_0 出 3.31GB GGUF，只测了速度没测数值。

**核验结论**：结构对不上。Instella 是**前置 norm**
（llamacpp/modeling_instella.py 698-721：`pre_attention_layernorm(x)` ->
attn -> +residual；夹心那两行在 AMD 自己的参照里是**注释掉的**），
而 llama.cpp 的 olmo2 是**夹心 norm**（olmo2.cpp 157-179：norm 打在
支路上再加残差）。把前置 norm 的权重塞进夹心槽 = 另一个模型 ——
能跑、能出字，**不是 Instella 的函数**。这正是"社区补丁只当线索、
数值有独立对拍才当结论"要抓的那一类。

**正路**（见 docs/llamacpp-plan.md T18 地形判定）：找/写结构忠实的宿主。
当前最像的是 **olmoe**（olmoe.cpp 75-149：前置 norm + 全宽 q/k norm，
结构与 Instella 同型），差别只剩 dense vs MoE。

用法（复现证据用）：
  python tools/instella-spoof-olmo2.py model-cache/Instella-3B model-cache/Instella-3B-olmo2
"""
import json
import shutil
import sys
from pathlib import Path


def rename_key(k):
    """帖子里的两规则，都对**原名**k 判断（顺序安全）。"""
    if "input_layernorm" in k:
        return k.replace("input_layernorm", "post_attention_layernorm")
    if "post_attention_layernorm" in k:
        return k.replace("post_attention_layernorm",
                         "post_feedforward_layernorm")
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

    # config：只改 architectures（照帖），其余一字不动
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    old_arch = cfg.get("architectures")
    cfg["architectures"] = ["Olmo2ForCausalLM"]
    (dst / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  config.architectures: {old_arch} -> ['Olmo2ForCausalLM']")

    # tokenizer 等小文件原样拷
    for f in sorted(src.iterdir()):
        if f.suffix in (".json", ".txt", ".model") and f.name != "config.json":
            shutil.copy2(f, dst / f.name)
            print(f"  拷贝 {f.name}")

    # safetensors：改名写副本
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch
    total = 0
    for f in sorted(src.glob("*.safetensors")):
        out = {}
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                out[rename_key(k)] = sf.get_tensor(k)
        if len(out) != len(set(out)):
            print(f"  [FAIL] {f.name} 改名后有重名 —— 配方在这一档不成立")
            return 1
        save_file(out, str(dst / f.name))
        total += len(out)
        print(f"  {f.name}: {len(out)} 个张量改名落盘")

    # 核验：399 是 refs/amd__Instella-3B.tensors.json 的实测数
    names = []
    for f in sorted(dst.glob("*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            names.extend(sf.keys())
    n_norm = sum(1 for k in names if "layernorm" in k)
    print(f"  ---- 核验 ----")
    print(f"  张量总数 {total}（refs 契约 399）")
    print(f"  改名后的 norm 张量 {n_norm}（36 层 x 2 = 72）")
    sample = [k for k in names if "layernorm" in k][:2]
    print(f"  样例：{sample}")
    if total != 399:
        print("  [FAIL] 总数对不上 refs 的 399")
        return 1
    if n_norm != 72:
        print("  [FAIL] norm 改名数不对")
        return 1
    print("  [PASS] 改名配方落盘 + 契约数核验过（数值忠实性归 logits 对拍判）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
