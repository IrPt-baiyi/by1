#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
by1 diff -- 把 by1 生成的前向 与 transformers 的参考实现**逐位对拍**。

    python by1diff.py llama-shaped.by1      # 稠密：对 LlamaForCausalLM
    python by1diff.py mixtral-shaped.by1    # MoE：对 MixtralForCausalLM

参考实现是**别人写的**。同一份权重分别装进两边，同一条输入，比 logits。
不一致的地方，就是 by1 在**语义**上的错——不是命名，不是形状，是算错了。

Block 0 之后，权重是按 **IR 里的算子下标** 搬运的：
  逻辑名 model.layers.N.self_attn.q_proj.weight  →  layers.N.op<j>.wq.weight
其中 j 由 IR 自己算出来，不是写死的。
"""

import os as _os
import sys as _sys
# **引导：把自己上面那一层（`src/`）放上 sys.path。**
# 加了它，`import by1paths` 才找得到；而 `by1paths` 在 import 时
# 会把 `src/` 和每个子目录都放上 sys.path —— 于是 `import by1check`
# 这种裸名 import 照旧能用。**这两行是生成的，别手改**
# （判据在 `by1paths.check_boot()`）。
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401

import argparse
import importlib.util
import os
import sys

import by1io      # noqa: F401  —— import 即把 stdout 钉成 UTF-8
import by1skip

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, by1paths.tool(name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def op_index(layer, kinds, nth=0):
    """找第 nth 个 kind 属于 kinds 的算子的下标 —— 不靠写死的顺序。"""
    if isinstance(kinds, str):
        kinds = (kinds,)
    c = 0
    for j, op in enumerate(layer["ops"]):
        if op["kind"] in kinds:
            if c == nth:
                return j
            c += 1
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="by1 前向对拍")
    ap.add_argument("by1")
    ap.add_argument("--seq", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backward", action="store_true",
                    help="同时对比梯度（1.md 说前向与反向都要一致）")
    ap.add_argument("--bisect", action="store_true",
                    help="逐个子模块比（投影 / 输出投影 / 两个 norm），"
                         "找第一个不一致的 —— 整模型差 1e-02 时用它定位")
    ap.add_argument("--probe", action="store_true",
                    help="给第 0 层的注意力挂 hook，比 by1 和 HF 的**注意力输出**"
                         "（逐层说'第 1 层之后偏了'，这一步说'偏在注意力还是前馈'）")
    ap.add_argument("--layers", action="store_true",
                    help="逐层对比：整模型的 logits 差 2e-02 时看不出差在哪，"
                         "这一步说得出是**哪一层**先偏离参考实现")
    args = ap.parse_args(argv)

    import torch
    import transformers as T

    torch.manual_seed(args.seed)

    bc, cg = load("by1check"), load("by1codegen")
    name = os.path.basename(args.by1)
    print(f"\n  读取 {name}")
    _rep, info = bc.check(args.by1)
    try:
        ir = cg.compile_ir(info)
    except cg.UnsupportedError as ex:
        # **"生成器还不认这个机制"不是"验了不对"，是"这台机器上没验"。**
        # 三态是 0 验过了 / 30 这台机器上没验 / 1 验了不对，而
        # KDA / SSM / 稀疏索引器 / mHC "只有契约，算不了"（README 原话）
        # 属于第二种。**假红和假绿一样会让人忽略一个检查。**
        #
        # 子类要先捕 —— CodegenError 是它的父类。
        print(f"\n  [不支持]\n{ex}\n")
        return by1skip.skip("生成器还不认这套机制：%s"
                            % str(ex).strip().replace("\n", "；")[:120])
    except cg.CodegenError as ex:
        # **文件写坏了 —— 那是"验了不对"。**
        # 属性缺失、取值非法、机制没有类型都属于这一种，而不是"没验"。
        print(f"\n  [不支持]\n{ex}\n")
        return 1

    # 逐层的算子下标 —— 混合栈（线性层 + 全量层）每层的布局本来就不同
    ATT = ("Attention", "Linear")
    layouts = [(op_index(L, ATT), op_index(L, "Norm", 0),
                op_index(L, "Norm", 1), op_index(L, ("FFN", "MoE")))
               for L in ir["layers"]]
    if any(None in t for t in layouts):
        print("  有层的算子形状目前不支持对拍")
        # 有层类型的多布局暂不支持 -> by1skip.CODE
        return by1skip.CODE
    has_linear = any(op_index(L, "Linear") is not None for L in ir["layers"])
    attn_layer = next((i for i, L in enumerate(ir["layers"])
                       if op_index(L, "Attention") is not None), None)
    if attn_layer is None:
        return by1skip.skip('没有全量注意力层，本脚本的对拍路径还不支持')
    L0 = ir["layers"][attn_layer]
    j_attn, j_norm0, j_norm1, j_ffn = layouts[attn_layer]
    attn = L0["ops"][j_attn]["attrs"]
    ffn_op = L0["ops"][j_ffn]

    is_moe = ffn_op["kind"] == "MoE"
    is_gptoss = bool(is_moe and ffn_op["attrs"].get("routing") == "topk_softmax"
                     and ffn_op["attrs"].get("expert_bias"))
    common = dict(vocab_size=ir["vocab"], hidden_size=ir["d_model"],
                  num_hidden_layers=len(ir["layers"]),
                  num_attention_heads=attn["q"], num_key_value_heads=attn["kv"],
                  head_dim=attn["head_dim"],
                  max_position_embeddings=max(ir["ctx"], args.seq + 8),
                  # **这两个从前是写死的，而写死的那两个都是错的。**
                  #
                  # `rms_norm_eps=1e-5` 写死 -> 而 `minimind-3` 声明的是
                  # **1e-6**（官方 config 的 `rms_norm_eps`）。同一个输入、
                  # 同一份权重、不同的 eps，输出差 **1.09e-02** ——
                  # 而报出来是"这个模型算错了"。**又一个假红，超参写死造成的。**
                  # （定位它的过程：`--layers` 说第一层之后偏了、`--probe`
                  #   说偏在注意力、二分到第一个算子 `op0`，然后量出
                  #   输入差 0.000e+00、权重差 0.000e+00、**eps 差 10 倍**。）
                  #
                  # `tie_word_embeddings=False` 写死同样是错的：
                  # `minimind-3` 官方是真 tied 的（产物里只有
                  # `model.embed_tokens.weight`，没有 `lm_head.weight`）。
                  #
                  # **用 `or` 不用 `.get(k, default)`** —— 后者的默认值只在
                  # **键缺失**时生效；键存在而值是 `None` 时拿到 `None`，
                  # eps 就变成 None、参考实现全红。（第一版就是那样写的：
                  # 修好了 minimind-3，把四个 shaped 全弄崩了。）
                  rms_norm_eps=ir.get("norm_eps") or 1e-5,
                  tie_word_embeddings=bool(ir.get("tie_word_embeddings")),
                  attention_bias=attn["bias"],
                  # 基值必须从 .by1 传过来：各有各的默认（Llama 1e4 / Mixtral 1e6），
                  # 不传就是两边在转不同角度 —— 这个坑 Oracle 第一次就抓到了
                  rope_theta=float(attn["rope_base"]))
    # **`has_linear` 只是必要条件之一。**
    #
    # Qwen3Next 是"Linear 层 + MoE"—— 而 `clef` 有 Linear 层的写法、
    # FFN 却是稠密的。原来这里只看 `has_linear`，于是去取
    # `ma["experts"]` 时 `KeyError: 'experts'`。
    #
    # **"这个家族认不认得"要按机制集合判，不能按"没有别的可能"判。**
    # ── 家族守卫：**表示不了就不要比** ─────────────────────────────
    #
    # 26 份跑下来是 `ok 4 · skip 5 · timeout 3 · fail 14`，而 14 个 fail
    # 里一大半的消息是"参考实现里少了 N 个张量" —— **那是参考实现建错了，
    # 不是模型错了**。gpt2 是 LayerNorm + 学习式位置 + Conv1D，而 Llama
    # 是 RMSNorm + RoPE + Linear：拿两个不同的东西对拍，比的不是数学。
    #
    # **假绿的反面是假红，而假红同样会让人忽略这个检查。**
    #
    # 判据从 IR 来 —— 它自己声明了这些。
    # **建参考实现要按真实维度分配内存。** 35B / 120B 在这台机器上
    # 建不出来 —— 那不是"算错了"，是"验不了"。
    #
    # （`by1gpu.py` 在没显卡的机器上就是这么做的：走 `by1skip`，
    #   而报告里一直印着「-- by1gpu.py [跳过]」。**先说没验，再说别的。**）
    #
    # 只认内存那一种 `RuntimeError`（按消息判），别的原样抛 ——
    # 把 `RuntimeError` 全归成跳过会把真错误一起吞掉。
    try:
        _hp = info.get("hparams") or {}
        _why = []
        _act = str(_hp.get("act") or "").lower()
        if _act and _act not in ("silu", "swiglu", "silu_and_mul"):
            _why.append("FFN 激活是 %s（这几个参考实现都用 silu）" % _act)
        _nk = str(_hp.get("norm_kind") or "").lower()
        if _nk and _nk not in ("rmsnorm", "rms"):
            _why.append("归一化是 %s（参考实现用 RMSNorm）" % _nk)
        elif "ln_eps" in _hp and "rms_eps" not in _hp:
            _why.append("声明的是 ln_eps（LayerNorm）而不是 rms_eps")
        if len(info.get("position") or {}) > 1:
            _why.append("声明了不止一种位置（%s）—— 说明有逐层不同的排法"
                        % ", ".join(sorted(info.get("position") or {})))
        if _why:
            return by1skip.skip("这份声明的东西，本脚本的参考实现表示不了：%s"
                                % "；".join(_why))

        def _pairing_unsupported():
            """**只有 Llama / Mixtral 的 HF 类用 half 配对。**

            `rope_pairing` 的 interleaved / half 就是这个仓库历史上那个
            `apply_rope` 交错 bug 的那一对概念。HF 的 Llama 参考实现用 half，
            所以声明 interleaved 的模型**和它算的不是同一个东西** ——
            不是一个"错"，是两种不同的位置编码。

            实测：`hello.by1` 没声明 position，于是拿到默认的 `interleaved`，
            和 Llama 参考差 **1.756e-02**；而 `llama-shaped.by1` 声明了
            `half`，差 4.470e-07。

            （`by1all` 里原来记的理由是"拿它跟一个不存在的东西比" ——
             **结论对，理由错**。不是"不存在"，是"用了不同的配对"。
             而理由错了就会修错地方。）
            """
            return str(attn.get("rope_pairing") or "").lower() not in ("", "half")

        def _qk_norm_style():
            """这份声明带 qk_norm 吗 —— 带了就走 Qwen3 那一族。

            和 `_qk_unsupported()` 是**同一件事的两面**：那个问
            "Llama / Mixtral 表示得了吗"，这个问"那谁表示得了"。
            答案只有 Qwen3（`q_norm` / `k_norm` 是它的内建层）。
            """
            _qn = str(attn.get("qk_norm") or "").lower()
            return _qn not in ("", "off", "false", "0", "none")

        def _qk_unsupported():
            """**只有 Llama / Mixtral 的 HF 类没有 qk_norm 这一层。**

            Qwen3Next（`T.Qwen3NextConfig`）和 gpt-oss 自己的类里都有 ——
            所以这一条不能放进全局守卫：拦了它就会把 `qwen3-next-shaped`
            从 ok 变成 skip，而那是**误伤真绿**（上一版就是这么错的）。
            """
            _qn = str(attn.get("qk_norm") or "").lower()
            return _qn not in ("", "off", "false", "0", "none")

        if has_linear and is_moe:
            lin = next(o["attrs"] for L in ir["layers"] for o in L["ops"]
                       if o["kind"] == "Linear")
            ma = ffn_op["attrs"]
            lt = ["linear_attention" if op_index(L, "Linear") is not None
                  else "full_attention" for L in ir["layers"]]
            common.pop("rope_theta", None)
            common.pop("attention_bias", None)
            cfg = T.Qwen3NextConfig(
                intermediate_size=ma["hidden"],
                num_experts=ma["experts"],
                num_experts_per_tok=ma["top_k"],
                moe_intermediate_size=ma["hidden"],
                shared_expert_intermediate_size=ma["shared_hidden"] or ma["hidden"],
                linear_num_key_heads=lin["k_heads"],
                linear_num_value_heads=lin["v_heads"],
                linear_key_head_dim=lin["k_dim"],
                linear_value_head_dim=lin["v_dim"],
                linear_conv_kernel_dim=lin["conv_kernel"],
                layer_types=lt,
                partial_rotary_factor=float(attn["rope_partial"]),
                **common)
            ref = T.Qwen3NextForCausalLM(cfg).eval()
            fam = "Qwen3NextForCausalLM"
        elif is_gptoss:
            ma = ffn_op["attrs"]
            y = attn.get("yarn") or {}
            rs = {"rope_type": "yarn", "rope_theta": float(attn["rope_base"]),
                  "factor": float(y.get("factor", 1)),
                  "original_max_position_embeddings": int(y.get("original", 4096)),
                  "beta_fast": float(y.get("beta_fast", 32)),
                  "beta_slow": float(y.get("beta_slow", 1)),
                  "truncate": bool(y.get("truncate", True)),
                  "attention_factor": float(attn.get("rope_scale", 1.0))}
            win = attn["window"] or ir["ctx"]
            common.pop("rope_theta")
            cfg = T.GptOssConfig(intermediate_size=ma["hidden"],
                                 num_experts=ma["experts"],
                                 num_experts_per_tok=ma["top_k"],
                                 layer_types=["sliding_attention"] * len(ir["layers"]),
                                 sliding_window=win, rope_scaling=rs, **common)
            ref = T.GptOssForCausalLM(cfg).eval()
            fam = "GptOssForCausalLM"
        elif is_moe:
            if _pairing_unsupported():
                return by1skip.skip(
                    "这份声明的东西，本脚本的参考实现表示不了："
                    "rope_pairing = %s，而 Mixtral 的 HF 类用 half 配对"
                    % attn.get("rope_pairing"))
            if _qk_unsupported():
                return by1skip.skip(
                    "这份声明的东西，本脚本的参考实现表示不了："
                    "注意力带 qk_norm=%s，而 Mixtral 的 HF 类里没有这一层"
                    % attn.get("qk_norm"))
            ma = ffn_op["attrs"]
            cfg = T.MixtralConfig(intermediate_size=ma["hidden"],
                                  num_experts=ma["experts"],
                                  num_experts_per_tok=ma["top_k"], **common)
            ref = T.MixtralForCausalLM(cfg).eval()
            fam = "MixtralForCausalLM"
        elif not has_linear and not is_moe and not is_gptoss:
            # **只有"整体就是 Llama 形状"的才走这一支。**
            #
            # 原来这里是无条件的 `else` —— 凡是没被前面几支认出来的模型
            # 都被拿去和 Llama 比。**那不是一个判据，那是"有一个参考实现"
            # 的错觉**：比出来的差会被当成模型的错，而真相是"没有参考"。
            if _pairing_unsupported():
                return by1skip.skip(
                    "这份声明的东西，本脚本的参考实现表示不了："
                    "rope_pairing = %s，而 Llama 的 HF 类用 half 配对"
                    % attn.get("rope_pairing"))
            # **这里原来有一句 `_qk_unsupported()` 就跳过 —— 现在不用了。**
            # 带 qk_norm 的稠密模型交给下面的 Qwen3 那一支（它的
            # `q_norm` / `k_norm` 是内建层）。留在这里会把
            # `minimind-3.by1` 挡在门外，而它正是唯一一个"加一族就能判"的。
            fa = ffn_op["attrs"]
            if _qk_norm_style():
                # **带 qk_norm 的走 Qwen3 那一族。**
                #
                # 那是唯一一个"加一族就能多判一个真模型"的情况：
                # `minimind-3.by1` 的单一位置、全 full 层、rms_eps 都对得上
                # Llama 形状，**只差 `qk_norm = per_head`** ——
                # 而 `T.Qwen3Config` 里 q_norm / k_norm 是内建的。
                #
                # （"多种位置"那一类不能这么办：那是**不同层用不同的位置
                # 编码**，HF 里没有任何一个类表示得了 —— 见上面那段。）
                cfg = T.Qwen3Config(intermediate_size=fa["hidden"], **common)
                ref = T.Qwen3ForCausalLM(cfg).eval()
                fam = "Qwen3ForCausalLM"
            else:
                cfg = T.LlamaConfig(intermediate_size=fa["hidden"], **common)
                ref = T.LlamaForCausalLM(cfg).eval()
                fam = "LlamaForCausalLM"
        else:
            # 到这个分支说明：机制组合我认不出该建哪个 HF 参考实现。
            # **按三态协议说"这台机器上没验"，不要说"验了不对"。**
            kinds = sorted({o["kind"] for L in ir["layers"] for o in L["ops"]})
            return by1skip.skip(
                # **说这个工具的局限，不说世界的事实。**
                # `clef` 有真的 HF 对应物（Cloudflare/clef）—— 说"没有对应的
                # 参考实现"是把"我不认得怎么建"讲成了"那个东西不存在"。
                '这台机器上没法给这套机制组合建参考实现'
                '（本脚本只认 Llama / Mixtral / gpt-oss / Qwen3Next 四种；'
                '这份的算子是 %s）' % ', '.join(kinds))

    except (RuntimeError, MemoryError) as ex:
        _m = str(ex).lower()
        if "memory" in _m or "alloc" in _m:
            return by1skip.skip(
                "这台机器装不下这个参考实现（按真实维度要 %.1f GB 量级）"
                % (ir["d_model"] * ir["d_model"] * len(ir["layers"])
                   * 12 / 1e9))
        raise
    ns = {}
    exec(compile(cg.render(info, name), "<by1-generated>", "exec"), ns)
    mine = ns["build"]().eval()

    print(f"  参考实现 transformers.{fam}")
    print(f"    {len(ir['layers'])} 层 · hidden {ir['d_model']} · "
          f"q {attn['q']} / kv {attn['kv']} / head_dim {attn['head_dim']} · "
          f"FFN {'MoE ' + str(ffn_op['attrs']) if is_moe else ffn_op['attrs']['hidden']}")

    # ── 权重搬运：全部按 IR 的算子下标定位 ──────────────────────────
    ref_sd = ref.state_dict()

    class Rec(dict):
        """记录 (参考名 -> by1 名)。用 is 比对，所以那些 dst[...] = ref_sd[...]
        的赋值点一行都不用改 —— 反向对拍需要这个对应关系。"""

        def __init__(self, ref, miss):
            super().__init__()
            self.pairs = []
            self._ref, self._miss = ref, miss

        def __setitem__(self, k, v):
            super().__setitem__(k, v)
            for rn, rv in self._ref.items():
                if rv is v:
                    self.pairs.append((rn, k))
                    break

    dst, missing = Rec(ref_sd, []), []
    for src, d in (("model.embed_tokens.weight", "embed.weight"),
                   ("model.norm.weight", "final_norm.w"),
                   ("lm_head.weight", "head.weight")):
        (dst.__setitem__(d, ref_sd[src]) if src in ref_sd else missing.append(src))

    LIN_NAMES = [("in_proj_qkvz.weight", "in_proj_qkvz.weight"),
                 ("in_proj_ba.weight", "in_proj_ba.weight"),
                 ("conv1d.weight", "conv.weight"),
                 ("dt_bias", "dt_bias"), ("A_log", "A_log"),
                 ("norm.weight", "norm.w"),
                 ("out_proj.weight", "out_proj.weight")]
    for i in range(len(ir["layers"])):
        p = f"model.layers.{i}."
        ja, jn0, jn1, jf = layouts[i]
        is_lin = op_index(ir["layers"][i], "Linear") is not None
        pairs = [
            (p + "input_layernorm.weight", f"layers.{i}.op{jn0}.w"),
            (p + "post_attention_layernorm.weight", f"layers.{i}.op{jn1}.w"),
        ]
        if is_lin:
            for s_, d_ in LIN_NAMES:
                pairs.append((p + "linear_attn." + s_,
                              f"layers.{i}.op{ja}." + d_))
        else:
            pairs += [
                (p + "self_attn.q_proj.weight", f"layers.{i}.op{ja}.wq.weight"),
                (p + "self_attn.k_proj.weight", f"layers.{i}.op{ja}.wk.weight"),
                (p + "self_attn.v_proj.weight", f"layers.{i}.op{ja}.wv.weight"),
                (p + "self_attn.o_proj.weight", f"layers.{i}.op{ja}.wo.weight"),
            ]
            if attn.get("qk_norm"):
                pairs += [
                    (p + "self_attn.q_norm.weight", f"layers.{i}.op{ja}.qn.w"),
                    (p + "self_attn.k_norm.weight", f"layers.{i}.op{ja}.kn.w"),
                ]
        for src, d in pairs:
            (dst.__setitem__(d, ref_sd[src]) if src in ref_sd else missing.append(src))
        if is_gptoss:
            p2 = f"layers.{i}.op{jf}."
            for a_, b_ in ((p + "self_attn.q_proj.bias", f"layers.{i}.op{ja}.wq.bias"),
                           (p + "self_attn.k_proj.bias", f"layers.{i}.op{ja}.wk.bias"),
                           (p + "self_attn.v_proj.bias", f"layers.{i}.op{ja}.wv.bias"),
                           (p + "self_attn.o_proj.bias", f"layers.{i}.op{ja}.wo.bias"),
                           (p + "self_attn.sinks", f"layers.{i}.op{ja}.sink"),
                           (p + "mlp.router.weight", p2 + "router.weight"),
                           (p + "mlp.router.bias", p2 + "router.bias")):
                (dst.__setitem__(b_, ref_sd[a_]) if a_ in ref_sd else missing.append(a_))
            gu = ref_sd.get(p + "mlp.experts.gate_up_proj")     # [E, d, 2H] 交错
            gb = ref_sd.get(p + "mlp.experts.gate_up_proj_bias")  # [E, 2H]
            dn = ref_sd.get(p + "mlp.experts.down_proj")        # [E, H, d]
            db = ref_sd.get(p + "mlp.experts.down_proj_bias")   # [E, d]
            if gu is not None:
                dst[p2 + "w1"] = gu[:, :, 0::2].transpose(1, 2)
                dst[p2 + "w3"] = gu[:, :, 1::2].transpose(1, 2)
            if gb is not None:
                dst[p2 + "b1"] = gb[:, 0::2]
                dst[p2 + "b3"] = gb[:, 1::2]
            if dn is not None:
                dst[p2 + "w2"] = dn.transpose(1, 2)
            if db is not None:
                dst[p2 + "b2"] = db
        elif is_moe:
            # transformers 这一版把专家融合了：
            #   mlp.gate.weight            [E, d]
            #   mlp.experts.gate_up_proj   [E, 2H, d]   ← gate 与 up 各占一半
            #   mlp.experts.down_proj      [E, d, H]
            # 逐专家存储的那个版本也有，两种都认。
            gw = p + "mlp.gate.weight"
            gu = p + "mlp.experts.gate_up_proj"
            dn = p + "mlp.experts.down_proj"
            if gw in ref_sd:
                dst[f"layers.{i}.op{jf}.router.weight"] = ref_sd[gw]
            else:
                missing.append(gw)
            n_e = ffn_op["attrs"]["experts"]
            if gu in ref_sd and dn in ref_sd:
                half = ref_sd[gu].shape[1] // 2
                dst[f"layers.{i}.op{jf}.w1"] = ref_sd[gu][:, :half]
                dst[f"layers.{i}.op{jf}.w3"] = ref_sd[gu][:, half:]
                dst[f"layers.{i}.op{jf}.w2"] = ref_sd[dn]
            else:
                import torch as _t
                for w in ("w1", "w2", "w3"):
                    keys = [f"{p}mlp.experts.{e}.{w}.weight" for e in range(n_e)]
                    if all(k in ref_sd for k in keys):
                        dst[f"layers.{i}.op{jf}.{w}"] = _t.stack(
                            [ref_sd[k] for k in keys])
                    else:
                        missing.extend(keys[:2])
            # 共享专家 —— 之前漏了这几个。模型照样建出来，只是它们停在随机初始化上，
            # 而"只检查搬过去的那些对不对"完全看不出来。
            for s_, d_ in (("shared_expert.gate_proj.weight", "sw1.weight"),
                           ("shared_expert.up_proj.weight", "sw3.weight"),
                           ("shared_expert.down_proj.weight", "sw2.weight"),
                           ("shared_expert_gate.weight", "shared_gate.weight")):
                src, tgt = p + "mlp." + s_, f"layers.{i}.op{jf}.{d_}"
                if src in ref_sd:
                    dst[tgt] = ref_sd[src]
        else:
            for src, d in ((p + "mlp.gate_proj.weight", f"layers.{i}.op{jf}.w1.weight"),
                           (p + "mlp.up_proj.weight", f"layers.{i}.op{jf}.w3.weight"),
                           (p + "mlp.down_proj.weight", f"layers.{i}.op{jf}.w2.weight")):
                (dst.__setitem__(d, ref_sd[src]) if src in ref_sd else missing.append(src))
    if missing:
        print(f"  [注意] 参考实现里少了 {len(missing)} 个张量，例如 {missing[:3]}")

    own = mine.state_dict()
    # 先查覆盖面：by1 侧有参数没被搬到，说明映射漏了 —— 它会**静默地**停在随机初始化上，
    # 而只检查"搬过去的那些对不对"是看不出来的。这条护栏今晚才补上。
    miss_keys = sorted(k for k in own if k not in dst)
    if miss_keys:
        print(f"  [FAIL] by1 侧有 {len(miss_keys)} 个参数没搬到（映射漏了）：")
        for k in miss_keys[:8]:
            print(f"      {k}  {tuple(own[k].shape)}")
        return 1
    bad = []
    for k, v in dst.items():
        if k not in own:
            bad.append(f"{k}: by1 侧没有")
        elif tuple(own[k].shape) != tuple(v.shape):
            bad.append(f"{k}: 形状 {tuple(own[k].shape)} vs {tuple(v.shape)}")
    if bad:
        print("  [FAIL] 权重对不上：")
        for b in bad[:10]:
            print("      " + b)
        return 1
    print(f"  权重搬运 {len(dst)} 个张量，逐一同名同形 ✓")
    mine.load_state_dict(dst, strict=False)

    # ── 前向 ────────────────────────────────────────────────────────
    ids = torch.randint(0, ir["vocab"], (1, args.seq))
    with torch.no_grad():
        a = ref(ids).logits.float()
        b = mine(ids).float()

    # ── 逐层（`--layers`）：哪一层的输出**先**偏离参考实现 ──────────
    #
    # 为什么要有这一步：整模型的 logits 差 2e-02 时，**看不出差在哪**。
    # 逐算子全过（`by1opdiff`）说明不是某个算子算错 —— 那剩下的可能
    # 就只有"组合方式"：norm 的位置、残差的接法、rope 挂在哪一步。
    #
    # 两边都取得到逐层：
    #   HF   `output_hidden_states=True` -> (嵌入, 每层输出…)  共 N+1 个
    #   by1  生成模型有 `embed` / `layers`(ModuleList) -> 手动逐层前向
    #
    # **HF 的 `hidden_states[-1]` 是最后一层输出、在 final_norm 之前** ——
    # 所以 by1 这边也**不加** `final_norm`，两边才对得齐。
    if args.layers:
        print("\n  逐层对比（第一个明显偏离的层就是线索）")
        try:
            with torch.no_grad():
                ra = ref(ids, output_hidden_states=True).hidden_states
                x = mine.embed(ids)
                rb = [x]
                for L in mine.layers:
                    x = L(x)
                    rb.append(x)
            print("    %-8s %-14s %-14s %s" % ('层', '最大绝对差', '相对', '参考幅度'))
            print("    " + '-' * 62)
            first_bad = None
            for i in range(min(len(ra), len(rb))):
                A, B = ra[i].float(), rb[i].float()
                if A.shape != B.shape:
                    print("    %-8d **形状不同** %s vs %s"
                          % (i, tuple(A.shape), tuple(B.shape)))
                    continue
                dd = (A - B).abs().max().item()
                rel = dd / max(A.abs().max().item(), 1e-12)
                mark = ''
                if rel > 1e-4 and first_bad is None:
                    first_bad = i
                    mark = '   <- **从这里开始**'
                print("    %-8d %-14.3e %-14.3e %.3e%s"
                      % (i, dd, rel, A.abs().max().item(), mark))
            if first_bad is None:
                print("    每一层都在 1e-4 以内 —— 差异出在层之外"
                      "（嵌入 / final_norm / head）")
            else:
                print("    第 %d 个（0 是嵌入、1 是第一层之后）先偏离" % first_bad)
        except (AttributeError, TypeError, RuntimeError) as ex:
            # 生成模型的层签名可能不是 `(x)` —— 说清楚是哪一步不行，
            # 而不是崩掉。**"量不了"和"量出来不对"是两件事。**
            print("    量不了：%s: %s" % (type(ex).__name__, str(ex)[:90]))
            print("    （多半是生成模型的层签名不是 `(x)` 单参）")

    # ── 二分（`--bisect`）：逐个子模块比，找第一个不一致的 ────────
    #
    # 整模型差 1e-02 时，只比 logits 看不出差在哪。而一层里能比的
    # 中间结果只有形状相同的那些：
    #
    #     op0  <-> input_layernorm      [b,n,d]        ✓
    #     wq/wk/wv <-> q_proj/k_proj/v_proj  Linear(d,…) ✓
    #     wo   <-> o_proj                Linear(q*hd,d) ✓
    #
    # **`qn` / `kn` 形状对不上**（HF 把 `[b,n,q*hd]` 摊平了再 norm、
    # by1 是 `[b,q,n,hd]`），直接比会得到假结论，所以不比。
    #
    # **每一对都要先取得到** —— 没有 `qk_norm` 的模型根本没有 `qn`，
    # 直接访问会 AttributeError（临时版就是这么把四个 shaped 弄崩的）。
    #
    # 这个工具在 `minimind-3` 上把范围从"注意力那 55 行"缩到了
    # "第一个算子 `op0`"，然后量出 **eps 差 10 倍**。
    # **诊断绝不能改变判定。** 认不出的家族上量不了，就说"量不了"
    # 然后退场 —— 而不是把 ok 变成 fail。
    #
    # （`--probe` / `--layers` 本来就是这个写法：外面包一层
    #   `except (AttributeError, TypeError, RuntimeError)`。）
    try:
        if args.bisect:
            print("\n  二分：第 0 层里形状对得上的子模块")
            got = {}

            def _mkb(side, nm):
                def _h(mod, inp, out):
                    o = out[0] if isinstance(out, (tuple, list)) else out
                    got[(side, nm)] = o.detach().float()
                return _h

            _lay = mine.layers[0]
            _att = getattr(_lay, "op1", None)
            _hlay = ref.model.layers[0]
            # **HF 那侧也要 `getattr(..., None)`。**
            #
            # Qwen3Next 那一族的层不叫 `self_attn` —— 直接访问会
            # `AttributeError`（第一版就是这样把 `qwen3-next-shaped` 弄崩的）。
            # 一侧做了、另一侧忘了，和前面那些"修好一个弄坏四个"是同一类。
            _h_att = getattr(_hlay, "self_attn", None)
            _h_ln = getattr(_hlay, "input_layernorm", None)
            ROWS = [("op0", getattr(_lay, "op0", None),
                     "input_layernorm", _h_ln),
                    ("wq", getattr(_att, "wq", None), "q_proj",
                     getattr(_h_att, "q_proj", None)),
                    ("wk", getattr(_att, "wk", None), "k_proj",
                     getattr(_h_att, "k_proj", None)),
                    ("wv", getattr(_att, "wv", None), "v_proj",
                     getattr(_h_att, "v_proj", None)),
                    ("wo", getattr(_att, "wo", None), "o_proj",
                     getattr(_h_att, "o_proj", None))]
            hs = []
            for bn, bmod, hn, hmod in ROWS:
                # **判据是"这一行能不能挂 hook"**，不是"是不是 None"。
                # 拿 `object()` 占位只会换一个 AttributeError。
                if bmod is None or hmod is None \
                        or not hasattr(bmod, "register_forward_hook") \
                        or not hasattr(hmod, "register_forward_hook"):
                    print("    %-6s %-16s 有一边挂不上，跳过" % (bn, hn))
                    continue
                hs.append(bmod.register_forward_hook(_mkb("by1", bn)))
                hs.append(hmod.register_forward_hook(_mkb("ref", hn)))
            try:
                with torch.no_grad():
                    ref(ids, output_hidden_states=True)
                    mine(ids)
            finally:
                for h in hs:
                    h.remove()
            print("    %-6s %-16s %-12s %-12s %s"
                  % ("by1", "hf", "绝对差", "相对", "判定"))
            print("    " + "-" * 64)
            first = None
            for bn, _bm, hn, _hm in ROWS:
                # **别用 `a` / `b`** —— 外层用它们存 logits，
                # 这里一覆盖，后面 `d = (a - b).abs()` 就拿到两个 None ✗。
                # （症状：`TypeError: unsupported operand type(s) for -:
                #   'NoneType' and 'NoneType'`，而报的位置在外面。）
                ra_, rb_ = got.get(("ref", hn)), got.get(("by1", bn))
                if not hasattr(ra_, "shape") or not hasattr(rb_, "shape"):
                    print("    %-6s %-16s （有一边没取到）" % (bn, hn))
                    continue
                if ra_.shape != rb_.shape:
                    print("    %-6s %-16s 形状不同 %s vs %s"
                          % (bn, hn, tuple(ra_.shape), tuple(rb_.shape)))
                    continue
                dd = (ra_ - rb_).abs().max().item()
                rel = dd / max(ra_.abs().max().item(), 1e-12)
                ok = rel < 1e-5
                if not ok and first is None:
                    first = "%s <-> %s" % (bn, hn)
                print("    %-6s %-16s %-12.3e %-12.3e %s"
                      % (bn, hn, dd, rel, '一致' if ok else '**不一致**'))
            print()
            if first is None:
                print("    这些都一致 -> 差在它们**之间**的接线")
            else:
                print("    第一个不一致的是 `%s`" % first)
                print("    下一步：量它的**输入**和**权重**，以及两边用的超参")

    except (AttributeError, TypeError, RuntimeError, IndexError,
            KeyError) as _ex:
        print("    量不了：%s: %s"
              % (type(_ex).__name__, str(_ex)[:90]))
        print("    （这个家族的结构和二分假设的不一样 —— 不影响判定）")
    # ── 探针（`--probe`）：第 0 层的注意力，两边各挂一个 hook ──────
    #
    # 逐层说"第 1 层之后偏了"。而一层里有注意力和前馈两块 ——
    # 这一步把它们分开：比**注意力子模块的输出**。
    #
    #   HF    `ref.model.layers[0].self_attn`
    #   by1   `mine.layers[0].op1`   （参数名 `layers.0.op1.wq.weight`
    #                                  就是这么来的）
    if args.probe:
        print("\n  探针：第 0 层的注意力输出")
        cap = {}

        def _mk(tag):
            def _h(mod, inp, out):
                # **HF 的注意力返回元组** `(attn_output, attn_weights, past)`
                # —— 而 by1 的返回张量。取第一个张量就好。
                o = out[0] if isinstance(out, (tuple, list)) else out
                i = inp[0] if inp else None
                cap[tag] = (i.detach().float() if i is not None else None,
                            o.detach().float())
            return _h

        # **两边的模块都要先取得到。** Qwen3Next 那一族的层不叫
        # `self_attn` —— 直接访问会 AttributeError，把一个 ok 变成 fail ✗。
        # （这一条是"加完对 26 份全量验"抓到的：`--probe` 上一轮只拿
        #   `minimind-3` 试过，没验过别的家族。）
        #
        # **诊断绝不能改变判定** —— 挂不上就说"看不到"然后退场。
        _hmod = getattr(ref.model.layers[0], "self_attn", None)
        _bmod = getattr(mine.layers[0], "op1", None)
        if _hmod is None or _bmod is None \
                or not hasattr(_hmod, "register_forward_hook") \
                or not hasattr(_bmod, "register_forward_hook"):
            print("    （这个家族的层挂不上注意力 hook，探针看不到）")
            hs = []
        else:
            hs = [_hmod.register_forward_hook(_mk("ref")),
                  _bmod.register_forward_hook(_mk("by1"))]
        try:
            with torch.no_grad():
                ref(ids)
                mine(ids)
        finally:
            for h in hs:
                h.remove()
        for tag in ("ref", "by1"):
            if tag in cap:
                xi, xo = cap[tag]
                si = ("%s 幅度 %.3e" % (tuple(xi.shape), xi.abs().max().item())
                      if xi is not None else "（取不到）")
                print("    %-4s 输入 %-28s  输出 %s 幅度 %.3e"
                      % (tag, si, tuple(xo.shape), xo.abs().max().item()))
        if "ref" in cap and "by1" in cap:
            xi_a, xo_a = cap["ref"]
            xi_b, xo_b = cap["by1"]
            # **输出总是能比的。** 输入那边可能取不到（HF 用关键字参数调
            # `self_attn(hidden_states=..., position_embeddings=...)`，
            # 于是 hook 的 `inp` 是空的）—— 但那不该挡住输出对比：
            # 上一版就是卡在这里，白跑一次。
            do = (xo_a - xo_b).abs().max().item()
            print("    **注意力输出差 %.3e**   相对 %.3e"
                  % (do, do / max(xo_a.abs().max().item(), 1e-12)))
            if xi_a is not None and xi_b is not None:
                di = (xi_a - xi_b).abs().max().item()
                print("    注意力输入差 %.3e" % di)
                print("    " + ("输入一致而输出不一致 —— **差在注意力内部**"
                                if di < 1e-5 < do
                                else "输入就已经不一样 —— 差在这一层之前"
                                if di >= 1e-5 else "注意力这一步是一致的"))
            else:
                print("    " + ("注意力的输出就是不一样的 —— **差在它内部或之前**"
                                if do > 1e-5 else "注意力这一步是一致的"))

    d = (a - b).abs()
    print(f"\n  前向对比（seq={args.seq}）")
    print(f"    logits 形状        {tuple(a.shape)}  vs  {tuple(b.shape)}")
    print(f"    我的 logits 幅度   {b.abs().max().item():.3e}")
    print(f"    最大绝对差         {d.max().item():.3e}")
    print(f"    平均绝对差         {d.mean().item():.3e}")
    print(f"    参考 logits 幅度   {a.abs().max().item():.3e}")
    print(f"    相对最大差         {(d.max()/a.abs().max()).item():.3e}")
    fwd_ok = d.max().item() < 1e-3
    print("\n  [PASS] 两边前向一致" if fwd_ok
          else "\n  [FAIL] 两边前向不一致 —— 差异处就是 by1 语义上的错")

    if not args.backward:
        return 0 if fwd_ok else 1

    # ── 反向 ────────────────────────────────────────────────────────
    print("\n  反向（梯度）对拍")
    for m in (mine, ref):
        m.zero_grad(set_to_none=True)
    ra, ma = ref(ids).logits.float(), mine(ids).float()
    ra.pow(2).sum().backward()
    ma.pow(2).sum().backward()
    rp = dict(ref.named_parameters())
    mp = dict(mine.named_parameters())
    rows, ng = [], 0
    for rn, bn in dst.pairs:
        gr = rp[rn].grad
        gm = mp[bn].grad
        if gr is None and gm is None:
            continue
        if gr is None or gm is None:
            rows.append((rn, float("inf"), "一边没有梯度"))
            continue
        dd = (gr.float() - gm.float()).abs().max().item()
        amp = gr.float().abs().max().item()
        # 相对差只在梯度真的有量级时才有意义 —— 接近 0 的梯度比出来必然是 1e0
        rel = dd / amp if amp > 1e-7 else 0.0
        rows.append((rn, rel, dd, amp, "" if amp > 1e-7 else "梯度极小，比值无意义"))
        ng += 1
    rows.sort(key=lambda r: -r[1])
    # 判据：**绝对差**和全局梯度幅度比。
    # 拿趋零的梯度去比"相对差"没有意义 —— 层 1 的 A_log 幅度只有 1e-6（衰减太强，
    # 梯度基本消失），两边的噪声当然会长得不一样。层 0/2 幅度正常的地方相对差是 1e-5。
    gmax = max((r[3] for r in rows), default=1.0)
    thr = 1e-4 * gmax
    nsig = sum(1 for r in rows if r[2] <= thr)
    print(f"    比了 {ng} 个参数的梯度；全局幅度 {gmax:.3e}，绝对差阈值 {thr:.3e}")
    print(f"    {'参考参数':<48} {'相对差':>11} {'绝对差':>11} {'参考幅度':>11}")
    for rn, rel, dd, amp, note in rows[:6]:
        print(f"    {rn:<48} {rel:>11.3e} {dd:>11.3e} {amp:>11.3e}  {note}")
    ok_b = max((r[2] for r in rows), default=0.0) <= thr
    print(f"\n  [{'PASS' if ok_b else 'FAIL'}] {nsig}/{ng} 个参数的梯度在阈值内"
          + ("   （前向与反向都一致）" if ok_b and fwd_ok else ""))
    return 0 if (ok_b and fwd_ok) else 1
    print("\n  [FAIL] 两边前向不一致 —— 差异处就是 by1 语义上的错")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
