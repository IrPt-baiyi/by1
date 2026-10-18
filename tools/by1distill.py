#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1distill -- 用 by1 描述出**学生形状**，再从 by1 描述的教师蒸馏（A 层 MVP）。

判据（写进 docs/distill-plan.md，这里逐条实现）：

1. **对照组**：学生 A 从零训练（只 CE）、学生 B 蒸馏（CE + T^2 * KL），
   同数据、同超参、同初始化（A/B 的初始权重逐位相同）、**同一批 batch 下标**；
2. **报告成对**：每 N 步 loss + 最后 20 步均值 + 学生-教师 KL 的起止值；
3. **可归因**：教师/学生的参数量、温度、alpha、步数全部打印；
4. 不放进主门禁：它是实验工具，跑一次几分钟，判据是"数字与结论"，不是门禁行。

    python tools/by1distill.py --teacher models/distill-teacher.by1 \\
        --student models/distill-student.by1 --text 1.md --steps 300
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import by1paths  # noqa: E402,F401


def build(by1_path):
    """按描述搭模型（与 by1train 同一条路：render -> exec -> build）。"""
    import by1check as bc
    import by1codegen as cg

    name = os.path.basename(by1_path)
    _rep, info = bc.check(by1_path)
    ir = cg.compile_ir(info)
    src = cg.render(info, name)
    ns = {}
    exec(compile(src, "<by1-distill>", "exec"), ns)
    model = ns["build"]()
    return model, ir


def weights_digest(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().numpy().tobytes())
    return h.hexdigest()[:16]


def read_corpus(paths):
    buf = []
    for p in paths:
        with open(p, "rb") as f:
            buf.append(f.read())
    return b"\n".join(buf)


def train_teacher(model, data, steps, seq, batch, lr, gen, log_every):
    import torch
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        ix = torch.randint(0, len(data) - seq - 1, (batch,), generator=gen)
        x = torch.stack([data[i:i + seq] for i in ix])
        y = torch.stack([data[i + 1:i + seq + 1] for i in ix])
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % log_every == 0 or step == 1:
            print("    教师 步 %-5d loss %.3f  %.0fs"
                  % (step, sum(losses[-20:]) / len(losses[-20:]), time.time() - t0))
    return losses


def train_student(model, data, steps, seq, batch, lr, batches, teacher=None,
                  temp=2.0, alpha=0.5, log_every=50, tag="A"):
    """batches 是**预先固定**的 (x, y) 列表 —— 两组学生必须吃同一批。"""
    import torch
    import torch.nn.functional as F

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses, kls, ces = [], [], []
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = batches[(step - 1) % len(batches)]
        logits = model(x)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        loss = ce
        if teacher is not None:
            with torch.no_grad():
                tl = teacher(x)
            kl = F.kl_div(
                F.log_softmax(logits / temp, dim=-1),
                F.softmax(tl / temp, dim=-1),
                reduction="batchmean") * (temp ** 2)
            loss = (1.0 - alpha) * ce + alpha * kl
            kls.append(float(kl.detach()))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        ces.append(float(ce.detach()))
        if step % log_every == 0 or step == 1:
            print("    学生%s 步 %-5d loss %.3f%s  %.0fs"
                  % (tag, step, sum(losses[-20:]) / len(losses[-20:]),
                     ("  KL %.3f" % (sum(kls[-20:]) / len(kls[-20:]))) if kls else "",
                     time.time() - t0))
    return losses, kls, ces


def evaluate(model, batches, teacher=None, temp=2.0):
    """**同一把尺子**：验证集 CE（与训练目标无关）+ 与教师的 KL。"""
    import torch
    import torch.nn.functional as F

    model.eval()
    ce_sum, kl_sum = 0.0, 0.0
    with torch.no_grad():
        for x, y in batches:
            logits = model(x)
            ce_sum += float(F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.reshape(-1)))
            if teacher is not None:
                tl = teacher(x)
                kl_sum += float(F.kl_div(
                    F.log_softmax(logits / temp, dim=-1),
                    F.softmax(tl / temp, dim=-1),
                    reduction="batchmean") * (temp ** 2))
    n = max(1, len(batches))
    return ce_sum / n, (kl_sum / n if teacher is not None else None)




def main(argv=None):
    ap = argparse.ArgumentParser(description="by1 形状 + 蒸馏 MVP")
    ap.add_argument("--teacher", default="models/distill-teacher.by1")
    ap.add_argument("--student", default="models/distill-student.by1")
    ap.add_argument("--text", action="append", default=None)
    ap.add_argument("--steps", type=int, default=300, help="每个学生的训练步数")
    ap.add_argument("--teacher-steps", type=int, default=400)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher-ckpt", default="build/distill-teacher.pt")
    ap.add_argument("--json", default=None, help="把曲线与结论存成 JSON")
    args = ap.parse_args(argv)

    import torch
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    texts = args.text or [os.path.join(ROOT, "1.md")]
    raw = read_corpus(texts)
    data = torch.tensor(list(raw), dtype=torch.long)
    print("\n  语料 %s · %d 字节" % (", ".join(os.path.basename(t) for t in texts),
                                   len(data)))
    if len(data) < args.seq + 2:
        print("  [语料太短] 至少要 %d 字节" % (args.seq + 2))
        return 1

    teacher, t_ir = build(os.path.join(ROOT, args.teacher))
    t_par = sum(p.numel() for p in teacher.parameters())
    print("  教师 %d 层 · d_model %d · %s 参数"
          % (len(t_ir["layers"]), t_ir["d_model"], format(t_par, ",")))

    ck = os.path.join(ROOT, args.teacher_ckpt)
    if os.path.exists(ck):
        state = torch.load(ck, map_location="cpu")
        teacher.load_state_dict(state["state"])
        print("  教师权重 %s（跳过训练）" % args.teacher_ckpt)
        t_losses = state.get("losses") or []
    else:
        print("  训练教师（%d 步）" % args.teacher_steps)
        t_losses = train_teacher(teacher, data, args.teacher_steps, args.seq,
                                 args.batch, args.lr, gen, max(1, args.teacher_steps // 4))
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        torch.save({"state": teacher.state_dict(), "losses": t_losses}, ck)
        print("  教师权重已存 %s" % args.teacher_ckpt)
    teacher.eval()

    # 验证集：**尾部 1/8 不参与训练**，两个学生在同一批上评价
    val_len = max(args.seq * 4, len(data) // 8)
    train_n = len(data) - val_len
    val_gen = torch.Generator().manual_seed(args.seed + 1)
    val_batches = []
    for _ in range(20):
        ix = torch.randint(0, val_len - args.seq - 1, (args.batch,), generator=val_gen)
        x = torch.stack([data[train_n + t:train_n + t + args.seq] for t in ix])
        y = torch.stack([data[train_n + t + 1:train_n + t + args.seq + 1] for t in ix])
        val_batches.append((x, y))

    # ── 同一批 batch 下标（两组学生必须吃同一批）──────────────────────
    n_batches = max(1, args.steps)
    batches = []
    for _ in range(n_batches):
        ix = torch.randint(0, train_n - args.seq - 1, (args.batch,), generator=gen)
        x = torch.stack([data[i:i + args.seq] for i in ix])
        y = torch.stack([data[i + 1:i + args.seq + 1] for i in ix])
        batches.append((x, y))

    # ── 学生：搭一次、深拷贝两份 -> 初始权重逐位相同 ──────────────────
    student, s_ir = build(os.path.join(ROOT, args.student))
    s_par = sum(p.numel() for p in student.parameters())
    a = copy.deepcopy(student)
    b = copy.deepcopy(student)
    da, db = weights_digest(a), weights_digest(b)
    if da != db:
        print("  [FAIL] 两个学生的初始权重不同（%s != %s）—— 对照不成立" % (da, db))
        return 1
    print("  学生 %d 层 · d_model %d · %s 参数（教师的 %.0f%%）"
          % (len(s_ir["layers"]), s_ir["d_model"], format(s_par, ","),
             100.0 * s_par / t_par))
    print("  初始权重 %s（A/B 逐位相同）" % da)

    print("\n  学生 A：从零训练（只 CE）")
    la, ka, ca = train_student(a, data, args.steps, args.seq, args.batch, args.lr,
                           batches, teacher=None, tag="A",
                           log_every=max(1, args.steps // 6))
    print("\n  学生 B：蒸馏（CE + %.2f * KL，T=%.1f）" % (args.alpha, args.temp))
    lb, kb, cb = train_student(b, data, args.steps, args.seq, args.batch, args.lr,
                           batches, teacher=teacher, temp=args.temp,
                           alpha=args.alpha, tag="B",
                           log_every=max(1, args.steps // 6))

    def tail(xs, n=20):
        return sum(xs[-n:]) / len(xs[-n:]) if xs else float("nan")

    a_ce, a_kl = evaluate(a, val_batches, teacher=teacher, temp=args.temp)
    b_ce, b_kl = evaluate(b, val_batches, teacher=teacher, temp=args.temp)
    kl0 = sum(kb[:20]) / 20 if kb else float("nan")
    kl1 = tail(kb) if kb else float("nan")
    print("\n  ── 结论（**同一把尺子**：验证集 %d 个 batch）──" % len(val_batches))
    print("    A 从零   验证 CE %.3f   与教师 KL %.3f" % (a_ce, a_kl))
    print("    B 蒸馏   验证 CE %.3f   与教师 KL %.3f   （CE 差 %+.3f）"
          % (b_ce, b_kl, b_ce - a_ce))
    print("    训练期 KL（B）：开始 %.2f -> 结束 %.2f" % (kl0, kl1))
    print("    参数量：教师 %s / 学生 %s（%.1f%%）"
          % (format(t_par, ","), format(s_par, ","), 100.0 * s_par / t_par))
    verdict = "蒸馏更好" if b_ce < a_ce else "从零更好"
    print("    判读：%s（同一批 %d 个训练 batch、同 %.2e lr、同 %d seed、同初始权重）"
          % (verdict, len(batches), args.lr, args.seed))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({
                "teacher": {"by1": args.teacher, "params": t_par,
                            "steps": args.teacher_steps, "loss_tail": tail(t_losses)},
                "student": {"by1": args.student, "params": s_par,
                            "ratio": s_par / t_par},
                "init_digest": da,
                "plain": {"train_ce": la, "val_ce": a_ce, "val_kl": a_kl},
                "kd": {"train_ce": lb, "kl": kb, "val_ce": b_ce, "val_kl": b_kl,
                       "kl_start": kl0, "kl_end": kl1},
                "verdict": verdict,
                "temp": args.temp, "alpha": args.alpha, "steps": args.steps,
                "seq": args.seq, "batch": args.batch, "lr": args.lr,
                "seed": args.seed, "val_batches": len(val_batches),
            }, f, ensure_ascii=False, indent=1)
        print("    曲线 %s" % args.json)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
