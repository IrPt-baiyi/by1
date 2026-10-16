# charts.py —— 2026Q3 开源模型大战配图（PIL 程序绘制，数据 = 实测 config/tensors）
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
BG = "#FFFFFF"
INK = "#1A1A2E"
MUT = "#666677"
ACC = ["#2F6FB2", "#D95F3B", "#2E9E6B", "#8E5BB8", "#C9A227"]

def font(size, bold=False):
    p = "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"
    try:
        return ImageFont.truetype(p, size)
    except OSError:
        return ImageFont.load_default()

def canvas(w, h):
    im = Image.new("RGB", (w, h), BG)
    return im, ImageDraw.Draw(im)

def text(d, xy, s, size=22, bold=False, fill=INK, anchor="la"):
    d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)

# ── 图 0：Q3 战场全景（24 款阵容）────────────────────────────────
def chart_roster():
    im, d = canvas(1620, 980)
    text(d, (60, 40), "2026 Q3 开源模型大战 · 24 款阵容全景", 36, True)
    text(d, (60, 96), "★ 张量级实测（safetensors 逐张量）   ● 配置实测（官方 config）   ○ 未验（公开信息，规格不猜）", 20, fill=MUT)
    groups = [
        ("DeepSeek", ["V4 Pro 0813  ●", "V4 Flash 0731  ●", "V4.1 Flash  ★"]),
        ("Qwen", ["3.8 2.4T A95B  ●", "3.8-Flash-Next  ★", "3.8 27B  ★"]),
        ("GLM", ["GLM-5.3  ●", "GLM-5.3-Flash  ★"]),
        ("Kimi", ["Kimi K3  ●", "K2 Horizon 375B-A23B  ○"]),
        ("Step", ["Step 5 Preview  ★", "（Step 3.7 基线  ★）"]),
        ("Ling", ["3.0-flash-VL  ●", "3.0-flash-Fin  ●", "3.1 Flash  ●"]),
        ("腾讯 Hy", ["Hy3  ●", "Hy4-Preview  ○"]),
        ("小米 MiMo", ["V2.6-Pro  ○", "V2.6-Flash  ○"]),
        ("NVIDIA", ["Nemotron 3.5 Super VL  ○", "Nemotron 3.5 Bolt  ○"]),
        ("Muse Spark", ["Muse Spark Open  ○"]),
        ("Motif / Inkling", ["Motif 3  ○", "Inkling  ○", "Inkling Small  ○"]),
    ]
    cols, gap = 4, 22
    cw, ch = (1620 - 120 - (cols - 1) * gap) // cols, 240
    for i, (vendor, models) in enumerate(groups):
        x = 60 + (i % cols) * (cw + gap)
        y = 150 + (i // cols) * (ch + gap)
        d.rounded_rectangle([x, y, x + cw, y + ch], radius=12, fill="#F7F7FA", outline="#CCC", width=2)
        d.rounded_rectangle([x, y, x + cw, y + 50], radius=12, fill="#33384D")
        d.rectangle([x, y + 30, x + cw, y + 50], fill="#33384D")
        text(d, (x + 18, y + 25), vendor, 22, True, fill="#FFFFFF", anchor="lm")
        ty = y + 76
        for m in models:
            col = INK
            if "★" in m:
                col = "#1E7A46"
            elif "●" in m:
                col = "#2F6FB2"
            else:
                col = "#999999"
            text(d, (x + 18, ty), m, 19, fill=col)
            ty += 40
    im.save(OUT / "fig0-roster.png")

# ── 图 1：张量世界观 ───────────────────────────────────────────
def chart_tensors():
    im, d = canvas(1280, 720)
    text(d, (60, 48), "张量级深挖样本的「张量世界观」（safetensors 实测）", 34, True)
    text(d, (60, 100), "横轴为对数刻度 · 数据来源：各模型官方张量清单 · 2026-10-15", 18, fill=MUT)
    rows = [
        ("Step-3.7-Flash", 1471, ACC[0]),
        ("Qwen3.8-Flash-Next", 1658, ACC[1]),
        ("Step-5-Preview", 2449, ACC[2]),
        ("GLM-5.3-Flash", 76108, ACC[3]),
        ("DeepSeek-V4.1-Flash", 96085, ACC[4]),
    ]
    x0, x1 = 340, 1180
    lo, hi = math.log10(1000), math.log10(120000)
    for tick in (1000, 10000, 100000):
        x = x0 + (math.log10(tick) - lo) / (hi - lo) * (x1 - x0)
        d.line([(x, 150), (x, 640)], fill="#DDDDDD")
        text(d, (x, 650), "{:,}".format(tick), 16, fill=MUT, anchor="ma")
    y = 180
    for name, n, col in rows:
        w = (math.log10(n) - lo) / (hi - lo) * (x1 - x0)
        d.rounded_rectangle([x0, y, x0 + w, y + 54], radius=8, fill=col)
        text(d, (320, y + 27), name, 22, anchor="ra")
        text(d, (x0 + w + 14, y + 27), "{:,}".format(n), 22, True, anchor="lm")
        y += 92
    text(d, (60, 680), "DSV4.1 的 96,085 个张量里 94,821 个是逐专家量化张量（w1/w2/w3+scale × 15360）", 18, fill=MUT)
    im.save(OUT / "fig1-tensors.png")

# ── 图 2：dtype 存储分布 ───────────────────────────────────────
def chart_dtype():
    im, d = canvas(1280, 720)
    text(d, (60, 48), "权重存储的「三派」：谁裸奔 BF16，谁把专家切碎", 34, True)
    text(d, (60, 100), "按张量个数归一化 · 实测 dtype 分布", 18, fill=MUT)
    colors = {"BF16": "#2F6FB2", "F32": "#D95F3B", "F8_E4M3": "#2E9E6B",
              "F8_E8M0": "#8E5BB8", "I8": "#C9A227", "I64": "#999999"}
    rows = [
        ("Step-5-Preview", {"BF16": 1856, "F32": 593}),
        ("Step-3.7-Flash", {"BF16": 1429, "F32": 42}),
        ("Qwen3.8-Flash-Next", {"BF16": 1655, "I64": 3}),
        ("GLM-5.3-Flash", {"F8_E4M3": 37338, "F32": 37629, "BF16": 1141}),
        ("DeepSeek-V4.1-Flash", {"F8_E8M0": 47589, "I8": 47232, "BF16": 520, "F32": 387, "F8_E4M3": 357}),
    ]
    x0, x1 = 340, 1180
    y = 170
    for name, dist in rows:
        total = sum(dist.values())
        x = x0
        for k, v in sorted(dist.items(), key=lambda kv: -kv[1]):
            w = (x1 - x0) * v / total
            d.rectangle([x, y, x + w, y + 58], fill=colors.get(k, "#888"))
            if w > 90:
                pct = 100.0 * v / total
                lab = "%s %.0f%%" % (k, pct) if pct >= 5 else "%.0f%%" % pct
                text(d, (x + w / 2, y + 29), lab, 17, True, fill="#FFFFFF", anchor="mm")
            x += w
        text(d, (320, y + 29), name, 22, anchor="ra")
        y += 96
    lx = 340
    for k in ("BF16", "F32", "F8_E4M3", "F8_E8M0", "I8"):
        d.rectangle([lx, 645, lx + 26, 671], fill=colors[k])
        text(d, (lx + 34, 658), k, 18, anchor="lm")
        lx += 150
    im.save(OUT / "fig2-dtype.png")

# ── 图 3：五大流派对比网格（含 Kimi-K3）────────────────────────
def chart_grid():
    im, d = canvas(1680, 930)
    text(d, (60, 40), "2026 Q3 五大流派：同一个形状，五种哲学", 34, True)
    cols = [("Step-5-Preview", ACC[2]), ("Kimi-K3", ACC[0]), ("DSV4.1-Flash", ACC[4]),
            ("Qwen3.8-Flash-Next", ACC[1]), ("GLM-5.3-Flash", ACC[3])]
    rows = [
        ("主注意力", ["GQA 64q×192\nkv 4 组", "Gated MLA\nq/o LoRA 1536/512\n+线性配置",
                      "MLA 系\nq 64×512, kv 1\nq/o LoRA", "GDN 线性\nv48 / qk16 头",
                      "MLA-NoPE\nqk256(v256)\nkv_lora 512"]),
        ("层型 / 周期", ["3×滑窗(512)\n+1×full", "AttnRes 3:1\n93 层=7×12+9 ※",
                         "滑窗 128\nBounded Replay", "每 4 层 1 full\n其余线性",
                         "每 4 层 1 full\n其余线性"]),
        ("稀疏 / 检索", ["16 头×256\ntop-512 块(8)\nssmax+CSA", "AttnRes 检索单元\n※ 待核",
                         "32 头×128\ntop-512\nKV 跨层共享", "4 头×128\n预算 2048",
                         "DSA 32 头×128\ntop-2048\nkpool 4"]),
        ("MoE", ["352 专家 top-8\nsigmoid×3.0", "896 专家 ※\nnoaux_tc",
                 "384 专家 top-6\nsqrtsoftplus", "512 专家 top-10\nsigmoid",
                 "288 专家 top-8\nsigmoid×2.5"]),
    ]
    lx, y0 = 60, 150
    cw, ch, gap = 268, 150, 14
    for j, (name, col) in enumerate(cols):
        x = lx + 170 + j * (cw + gap)
        d.rounded_rectangle([x, y0, x + cw, y0 + 52], radius=8, fill=col)
        text(d, (x + cw / 2, y0 + 26), name, 20, True, fill="#FFFFFF", anchor="mm")
    for i, (rname, cells) in enumerate(rows):
        y = y0 + 68 + i * (ch + gap)
        text(d, (lx + 150, y + ch / 2), rname, 22, True, anchor="ra")
        for j, cell in enumerate(cells):
            x = lx + 170 + j * (cw + gap)
            d.rounded_rectangle([x, y, x + cw, y + ch], radius=10,
                                outline="#CCCCCC", width=2, fill="#F7F7FA")
            ty = y + ch / 2 - 15 * (cell.count("\n") + 1)
            for line in cell.split("\n"):
                text(d, (x + cw / 2, ty), line, 18, anchor="ma")
                ty += 30
    text(d, (60, 880), "※ = 推断/待核（字段名对照或公开资料，确切计算图以官方 modeling 为准）", 18, fill=MUT)
    im.save(OUT / "fig3-schools.png")

# ── 图 4：收敛形状示意图 ───────────────────────────────────────
def chart_shape():
    im, d = canvas(1520, 700)
    text(d, (60, 40), "收敛出来的「公共形状」：Q3 多数新模型长这样", 34, True)
    y = 130
    x = 60
    bw, bh = 118, 92
    labels = ["S 滑窗", "S 滑窗", "S 滑窗", "F 全量\n+indexer"]
    for rep in range(2):
        for lab in labels:
            col = "#DCE9F7" if lab.startswith("S") else "#FBE3D3"
            d.rounded_rectangle([x, y, x + bw, y + bh], radius=10, fill=col, outline="#999")
            ty = y + bh / 2 - 14 * (lab.count("\n") + 1)
            for line in lab.split("\n"):
                text(d, (x + bw / 2, ty), line, 19, anchor="ma")
                ty += 28
            x += bw + 10
        text(d, (x + 6, y + bh / 2), "× N", 22, True, fill=MUT, anchor="lm")
        x += 70
    d.rounded_rectangle([x, y, x + bw + 40, y + bh], radius=10, fill="#E7F3E7", outline="#999")
    text(d, (x + (bw + 40) / 2, y + bh / 2), "MTP ×0-3", 19, anchor="mm")
    text(d, (60, 250), "每层内：", 22, True)
    inner = [("Attention / Linear", "#DCE9F7"), ("MoE（sigmoid 路由）", "#F6E7C9"),
             ("mHC / AttnRes 超连接", "#EDE3F5"), ("Norm（RMS，多 F32）", "#EEEEEE")]
    ix = 200
    for lab, col in inner:
        w = 260
        d.rounded_rectangle([ix, 238, ix + w, 292], radius=8, fill=col, outline="#999")
        text(d, (ix + w / 2, 265), lab, 18, anchor="mm")
        if ix + w + 44 < 1400:
            text(d, (ix + w + 8, 265), "→", 22, True, fill=MUT, anchor="lm")
        ix += w + 44
    text(d, (60, 340), "稀疏/检索注意力（full 层）：", 22, True)
    idx = [("q/k proxy 投影", "#FBE3D3"), ("块打分 top-k", "#FBE3D3"),
           ("块选择（512~2048）", "#FBE3D3"), ("ssmax / kpool 压缩", "#FBE3D3")]
    ix = 420
    for lab, col in idx:
        d.rounded_rectangle([ix, 328, ix + 190, 382], radius=8, fill=col, outline="#999")
        text(d, (ix + 95, 355), lab, 18, anchor="mm")
        if ix + 240 < 1180:
            text(d, (ix + 196, 355), "→", 22, True, fill=MUT, anchor="lm")
        ix += 214
    d.rounded_rectangle([60, 430, 1460, 540], radius=12, fill="#F7F7FA", outline="#CCC")
    text(d, (90, 455), "共同点：混合层型（每 N 层一个 full）· top-k 块稀疏/检索 · sigmoid 路由 MoE · MTP · 1M 上下文", 22, True)
    text(d, (90, 500), "分歧点：主注意力路线（GQA / MLA / 线性）· 专家布局（3D / 1D+scale）· 量化存储（BF16 / I8+E8M0 / F8 block）", 20, fill=MUT)
    im.save(OUT / "fig4-shape.png")

for fn in (chart_roster, chart_tensors, chart_dtype, chart_grid, chart_shape):
    fn()
    print("wrote", fn.__name__)
