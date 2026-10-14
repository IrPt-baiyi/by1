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
    import glob
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from transformers import AutoProcessor
    # **版本钉住**（T21-6）：checkpoint 配 transformers==4.57.6；那边
    # 这个架构叫 qwen3_omni_moe 的 Thinker（thinker.* 布局、audio_tower
    # 含 proj1/proj2、18 层）。5.x 的 Qwen3ASR 类是另一代架构，不许混。
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerForConditionalGeneration as T)
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerConfig)

    proc = AutoProcessor.from_pretrained(hf_dir)
    import json as _json
    _raw = _json.loads(
        (Path(hf_dir) / "config.json").read_text(encoding="utf-8"))
    # **from_dict 才解 thinker_config**（from_pretrained 走的是外层默认：
    # hidden 2048/ffn 18944/vocab 3584 全错）—— 实测 from_dict 出
    # hidden 1024/3072/28 层 = checkpoint 真值
    cfg = Qwen3OmniMoeThinkerConfig.from_dict(_raw["thinker_config"])
    # **强制 dense**：checkpoint 的 text_config 没有 moe 键（dense MLP），
    # 而类默认 num_experts=128 -> 每层 3 亿参 -> 34GB -> 装载即 OOM。
    # 不改这两行，前面所有数字都是被这个默认值污染的。
    cfg.text_config.num_experts = 0
    cfg.text_config.moe_intermediate_size = 0
    # from_dict 的 rope_scaling 解析会把 mrope_section 洗掉（rope_type=default
    # 路径丢未知键）—— raw 原值直接塞回，RotaryEmbedding 就要它
    cfg.text_config.rope_scaling = _raw["thinker_config"]["text_config"].get(
        "rope_scaling") or {"rope_type": "default"}
    model = T(cfg).to(torch.float32).eval()
    sd = {}
    for f in sorted(glob.glob(str(Path(hf_dir) / "model*.safetensors"))):
        for k, v in load_file(f).items():
            if k.startswith("thinker."):
                sd[k[len("thinker."):]] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # **visual.* 是未用支路**（ASR 无视觉）：只有非 visual 的缺失才算映射坏
    real_missing = [k for k in missing if not k.startswith("visual")]
    print("  state_dict: %d in | missing %d（visual 未用支路 %d）"
          "| unexpected %d"
          % (len(sd), len(missing), len(missing) - len(real_missing),
             len(unexpected)))
    if real_missing or unexpected:
        print("  missing[:4]:", real_missing[:4])
        print("  unexpected[:4]:", unexpected[:4])
        print("  [FAIL] 映射不干净 —— 后面的数字不许引用")
        return 1

    t = np.arange(n_s, dtype=np.float32) / 16000.0
    sine = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained(hf_dir)
    # 手造 input_ids：占位检查数的是 config.audio_token_id（151676）——
    # 给字符串会被处理器改写，id 直给才稳
    n_ph = 13 if n_s >= 16000 else max(1, n_s * 13 // 16000)
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained(hf_dir)
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
                          "input_features_mask", "attention_mask")},
              feature_attention_mask=inputs.get("input_features_mask"))
    for h in handles:
        h.remove()

    print("captured modules:", list(cap))
    # 边界 = audio_tower.proj2（合并头的输出，喂 LM 的 1024）——
    # 按名选，别用"最宽维"（会抓到 fc1 的 3584 ffn 维）
    best = None
    for key in ("audio_tower.proj2", "multi_modal_projector"):
        for name, v in cap.items():
            if name.endswith(key):
                best = (name, v)
    if best is None:
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
