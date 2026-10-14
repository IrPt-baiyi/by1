import json, sys
from transformers import AutoProcessor

d = "model-cache/qwen3asr-hf"
proc = AutoProcessor.from_pretrained(d)
print("processor class:", type(proc).__name__)
print("mro:", [c.__name__ for c in type(proc).__mro__][:6])
for attr in ["feature_extractor", "image_processor", "tokenizer", "audio_processor"]:
    v = getattr(proc, attr, "<absent>")
    print("  %-18s %s" % (attr, type(v).__name__ if v is not None and v != "<absent>" else v))

cfg = json.load(open(d + "/config.json", encoding="utf-8"))
print("model_type:", cfg.get("model_type"))
print("architectures:", cfg.get("architectures"))
print("transformers_version:", cfg.get("transformers_version"))
print("top-level keys:", sorted(cfg.keys()))
ac = cfg.get("audio_config") or (cfg.get("thinker_config") or {}).get("audio_config")
print("audio_config present:", ac is not None)
if ac:
    print("  audio keys:", sorted(ac.keys())[:20])
