#!/usr/bin/env python3
"""tiny-vision-ref.py -- T20 judge: transformers side of the vision boundary.

Gemma3ForConditionalGeneration.get_image_features() on the same image --
the post-projector image embeddings, the same boundary mtmd-embd-probe
dumps. Prints "embd n d" header + one float per line (same format).

Usage:
  python tools/tiny-vision-ref.py <hf-dir> <image.png> <out.txt>
"""
import sys
from pathlib import Path


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    hf_dir, img_path, out_path = argv

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    proc = AutoProcessor.from_pretrained(hf_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        hf_dir, torch_dtype=torch.float32).eval()

    im = Image.open(img_path).convert("RGB")
    inputs = proc(images=im, return_tensors="pt")
    with torch.no_grad():
        feat = model.get_image_features(
            pixel_values=inputs["pixel_values"].to(torch.float32))
    # transformers 5.x 返回 BaseModelOutputWithPooling —— 取
    # last_hidden_state 再过 multi_modal_projector（=喂给 LM 的边界）
    if hasattr(feat, "last_hidden_state"):
        h = feat.last_hidden_state
        inner = getattr(model, "model", model)
        # Gemma3 全链：soft_emb_norm（RMSNorm）-> multi_modal_projector
        # 少了 norm 会得到"方向对、逐元漂移"（余弦 0.98 那一族）
        soft = getattr(inner, "mm_soft_emb_norm", None) or \
            getattr(model, "mm_soft_emb_norm", None)
        if soft is not None:
            h = soft(h)
        proj = getattr(model, "multi_modal_projector", None) or \
            getattr(inner, "multi_modal_projector", None)
        if proj is not None:
            h = proj(h)
        e = h[0]
    else:
        e = feat[0] if feat.dim() == 3 else feat
    vals = e.reshape(-1).to(torch.float32).tolist()
    n, d = (e.shape[0], e.shape[1]) if e.dim() == 2 else (1, e.shape[0])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("embd %d %d\n" % (n, d))
        fh.write("".join("%.9g\n" % v for v in vals))
    print("  参照嵌入 %d x %d 写了 %s" % (n, d, out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
