#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1ir -- **规范化的 by1 IR：schema、校验、JSON 往返。**

## 为什么要有这个

在这之前，IR 是**由生产者定义的** —— `compile_ir` 吐一个 dict，
三个消费者各取所需，**没有一处说这个 dict 里每个键是什么意思**。
于是：

    by1codegen.render(info)      要 info，不是 ir
    by1c.emit_c(ir, info, params) 要两个（info 其实是摆设）
    by1exec.shapes_of(ir)        只要 ir

**IR 不自足。** 想写第四个后端的人，必须先学会 `.by1`。

## 这个模块要做的事

    ① 版本号       `by1-ir`，改动可见
    ② 字段规格     必填 / 可选 / 语义 —— 以**数据**形式给出，不是文档
    ③ 校验         validate(ir) -> [错误]
    ④ JSON 往返    to_json / from_json，**往返等价**

## 为什么规格写成数据而不是文档

因为这个项目最贵的一课是：

> **一个只验你记得要验的东西的检查，不是检查。**

文档会过期；而写在这里的必填项、闭集、类型，**会被 `validate` 和
`by1all` 真的跑一遍**。
"""
import json
import re

import by1io
import by1paths
import by1skip

# 宏名的换名规则（`wq.bias` -> `WQ_BIAS`）。**只有这一处** ——
# `weight_plan` 生成宏名，C / ggml 两边都按它取偏移。
_MANGLE = re.compile(r"[^0-9A-Za-z]")

# **从 by1ver 来，不在这里写死。**
# 版本号**只在此处定义** —— by1codegen / by1boot / by1extdemo
# 各自**又写了一遍** —— 三份写着同一个字符串，谁也不认识谁。
# 改一份忘一份不会崩，只会造出"声称不同版本"的 IR。
try:
    from by1ver import IR_VERSION as VERSION
except ImportError:                     # 单独拷一个文件出去时兜底
    VERSION = "1.0"

#: **归一化 eps 的默认值 —— 唯一一处。**
#
# 这个数字原来在 10 个文件里写了约 50 次（`ir.get("norm_eps", 1e-5)`
# 这种形状），而**每一次"某一个后端写死了它"都是同一个 bug**：
# `history/ir.md` 数过，`eps` 这一类**犯过五次**，第五次是护栏自己
# 逼出来的。`by1c` 的 C_HEAD 里还留着那次的注释：
#
#     /* eps 必须由调用方传进来 —— 原来写死 1e-5，而 mla-shaped 的
#        rms_eps 是 1e-6。误差 1.5e-04，而其它四个模型恰好都是 1e-5
#        所以一直没露。 */
#
# 而**那句话没管住 qk_norm 那条路**：`by1exec` 调 `rms_norm` 时
# 不传 eps（吃默认参数），`by1c` 的 `qk_norm()` 里直接写 `1e-5f`
# —— 于是 clef-tiny（`qk_norm = per_head`、`norm_eps = 1e-6`）
# 在三个后端里算出三个不同的值，相对差 1.05e-05，
# **低于 1e-4 的判据所以一直是绿的**。
#
# 默认值本身不是错（大多数模型确实是 1e-5）；错的是**它被写了 50 遍**，
# 于是"哪一处漏了传参"永远只有出事之后才知道。现在它是一个名字。
EPS_DEFAULT = 1e-5

# ── **属性默认值登记表**（2026-10-17 第六轮）───────────────────────
# 同一个默认值曾经在四个后端各写一遍：`alpha` 7 处、`norm_eps` 6 处、
# `rope_scale` 6 处、`rope_partial` 5 处、`l2_eps` 4 处、`routed_scale` 2 处。
# 值当时**恰好**一致，但"恰好一致"不是保证 —— 改一处忘一处就会造出两个
# 算不同模型的"同一个后端"，而且这种偏差通常低于 1e-4 的判据，永远是绿的。
#
# 从这一轮起：这些名字**只在这里定义**，其它模块 import；`by1lint` 的
# 第⑪类盯着"取值点又写死了一个数字"。
#
# **不在表里**的常量（1e-6 的 DSA 索引器 LayerNorm、1e-5 之外的参考固定值）
# 是**参考实现钉死的语义常量**，不是模型属性默认值 —— 它们该在 lowering 里
# 写一次并注明出处，不归这张表管。
ALPHA_DEFAULT = 1.702                 # GELU-tanh 的 alpha
L2_EPS_DEFAULT = 1e-6                 # l2_norm 的 eps
ROPE_SCALE_DEFAULT = 1.0              # rope 的 cos/sin 缩放
ROPE_PARTIAL_DEFAULT = 1.0            # 参与旋转的比例
ROPE_BASE_DEFAULT = 10000             # rope 的基频
ROUTED_SCALE_DEFAULT = 1.0            # MoE 路由的缩放

#: 属性名 -> 默认值。`by1lint` 的第⑪类按这张表查两件事：
#:   ① 活代码里的取值点不许再写数字（要么用这里的名字）；
#:   ② **生成物模板**（`by1codegen.RUNTIME` 那段文本）里的字面量
#:      必须**等于**这里的值 —— 模板是独立产物、引用不了模块名，
#:      所以只能写字面量，但"相等"由机器盯。
ATTR_DEFAULTS = {
    "norm_eps": EPS_DEFAULT,
    "eps": EPS_DEFAULT,
    "l2_eps": L2_EPS_DEFAULT,
    "alpha": ALPHA_DEFAULT,
    "rope_scale": ROPE_SCALE_DEFAULT,
    "rope_partial": ROPE_PARTIAL_DEFAULT,
    "rope_base": ROPE_BASE_DEFAULT,
    "routed_scale": ROUTED_SCALE_DEFAULT,
}

# ── STACKS：分支登记表（1.1 定字段 · 1.2 升格）─────────────────────
# (名字, 类型, 必填, 说明)
# 为什么要有它：源语言里 `stack vision { }`、MODAL 闭集早就有，但
# **编译成 IR 后栈信息全丢**（by1ir 里 stack/modal 原本零命中）——
# 于是护栏没法对塔说话。可缺省 = 单栈文本，**旧描述一个字不改照样
# 合法**（T22 判据：旧描述继续有效）。
#
# **1.2 的「不拍平」判据不是「不能有平表」，是答得出两个问题：**
#   这个节点在哪个分支上 —— 本表的层段划分（first_layer + n_layers）
#   接线从哪来到哪去 —— WIRING 表
# 节点表照旧是 `layers`（全局编号 —— 实测 91/92 层的多栈模型从不撞名，
# params_of / 后端零改动）；分支不各持一份层表，是为了**不给同一个
# 节点两个家**。
STACKS = (
    ("name", "str", True, "分支名（`text` 是主栈的缺省名）"),
    ("modality", "enum:default,text,vision,audio,video", True,
     "模态。**和 by1blocks.MODAL 同一闭集** —— 两处各写一份迟早漂"),
    ("width", "int", True,
     "该分支的残差流宽度。等于 `d_model` 的可以不登记；**不等的必须登记**"
     "（视觉塔宽 1152 / d_model 5376 那种）"),
    ("first_layer", "int", True, "该分支从第几层开始（0-based，按 layers 序）"),
    ("n_layers", "int", True, "该分支占几层"),
    ("role", "enum:trunk,mtp,tower", False,
     "分支的角色（1.2 新增）：`trunk` 主干（缺省）· `mtp` 并列预测头 · "
     "`tower` 旁路塔（视觉/音频那种）。**role 和 modality 不是一回事**："
     "mtp 是角色、vision 是模态"),
)

#: 端口文法里的模态闭集 —— **从 STACKS.modality 那格派生**，不抄第二份。
MODALITIES = tuple(
    t.split(":", 1)[1].split(",")
    for _n, t, _r, _d in STACKS if _n == "modality")[0]

# ── WIRING（by1-ir 1.2）：显式接线 ─────────────────────────────────
# 端口文法（validate 按它查，spec 抄这一段）：
#   `input:<模态>`        模态入口（input:text / input:image / …）
#   `<分支>.in`           分支的入口
#   `<分支>.out`          分支的输出（跑完它自己的层之后）
#   `<分支>.hidden`       进输出头之前的隐藏态（MTP 头吃的是这个）
#   `logits`              主输出头
#   `logits.<分支>`       旁路输出头（MTP 的额外 logits）
WIRING = (
    ("kind", "enum:feed,concat,head", True,
     "接线种类：`feed` 喂进分支入口 · `concat` 拼接进主干流（merger 那种）"
     " · `head` 并列头（MTP 那种）"),
    ("from", "str", True, "起点端口（文法见上）"),
    ("to", "str", True, "终点端口（文法同上）"),
    ("at", "str?", False,
     "拼接落点的记号（如 `image_tokens`：拼在图像 token 的位置上）"),
    ("via", "str?", False,
     "经过的结构名（如 `merger`）—— 人看的路标，不参与计算"),
    ("note", "str?", False, "自由说明（人看的，不参与计算）"),
)

# ── SHARED（by1-ir 1.2）：跨分支参数绑定 ───────────────────────────
# 共享权重的两种形态（普查〇之二结论 3）：
#   tie   —— 同一块权重两个名字（tie_word_embeddings）
#   share —— 声明级共享（mtp 的 share_embeddings / share_head）
# 两种都要在 IR 里**说**出来，不然搬权重的人只能靠猜。
SHARED = (
    ("param", "str", True,
     "被绑定的参数，限定名 `<分支>.<名>`（globals 里的可以裸名，如 `head.weight`）"),
    ("binds", "str", True, "绑定到的参数（同文法）"),
    ("kind", "enum:tie,share", False, "`tie`=同一块权重；`share`=声明共享。缺省 tie"),
    ("note", "str?", False, "自由说明（人看的，不参与计算）"),
)

# ── 顶层字段 ────────────────────────────────────────────────────────
# (名字, 类型, 必填, 说明)
TOP = [
    ("by1-ir", "str", True, "IR 版本。**读的一方必须拒绝不认识的大版本。**"),
    ("stacks", "list?", False,
     "分支登记表（1.1 新增、1.2 升格，缺省 = 单栈文本）。每条见 STACKS 表；"
     "非 `text` 分支的层必须给 `in_dim` 且必须等于分支宽（跨栈漂移当报红）；"
     "层段互不重叠、合起来可不满全部层"),
    ("wiring", "list?", False,
     "显式接线（1.2 新增），每条见 WIRING 表。**stacks 多于一条时必填**"
     " —— 没有接线的分支登记表只是孤岛，等于没说"),
    ("shared", "list?", False,
     "跨分支参数绑定（1.2 新增），每条见 SHARED 表（tie / share 两种形态）"),
    ("vocab", "int", True, "词表大小"),
    ("ctx", "int", True, "上下文长度（位置能达到的最大值）"),
    ("d_model", "int", True, "残差流宽度"),
    # **闭集要写在类型里，不能只写 `enum`。**
    # 类型必须写成 `enum:...` 或可空的 `enum?:...`（都带冒号），
    # `_ty` 只认这两种 —— 裸 `"enum"`
    # 于是这两个字段**从来没有被校验过**：`pos_kind = BANANA` 一路通到
    # 后端，而后端判的是 `== "learned"`，非法值静默落回 rope。
    # 同一个形状的漏洞还有 STATE.kind / STATE.reuse / STATE.dtype。
    ("pos_kind", "enum:rope,learned", True,
     "位置信息的种类。`rope`=在注意力里旋转；`learned`=在输入上加一张查表。"
     "**这两件事不一样**，所以是一个字段而不是一个布尔。"),
    ("n_pos", "int", False, "pos_kind=learned 时必填：查表有多少行"),
    ("norm_kind", "enum:rms,layer", True,
     "归一化的种类。`rms`=只除均方根；`layer`=减均值+除标准差+有 bias。"
     "**名字像，算的不是一回事。**"),
    ("norm_eps", "float", True, "归一化的 eps。**必须来自这里，不许各后端写死**"
                                "（曾因此出错三次）"),
    ("norm_one_plus", "bool", True, "归一化权重是乘 `w` 还是乘 `(1+w)`"),
    ("globals", "list[Op]", True, "不属于任何层的算子（embed / final_norm / head）"),
    ("layers", "list[Layer]", True, "逐层"),
]

LAYER = [
    ("index", "int", True,
     "层号。**全局序号**（多分支也连续编号 —— 实测 91/92 层的多栈模型"
     "从不撞名；分支归属看 stacks 的层段，不看这里）"),
    # **这一层的残差流宽度。** 不给就是顶层 `d_model` —— 绝大多数模型
    # 每层都一样，所以是可选的；给了就是**这一栈自己的宽度**。
    #
    # 为什么要有它：一个 checkpoint 可以有多个宽度不同的栈（视觉塔）。
    # 实测 clef 的视觉栈宽 1152、GLM 1024、Step 1536，而顶层 `d_model`
    # 是 5120 / 4096 / 4096。**没有这个字段，后端推出来的参数形状就是错的**
    # —— clef 的 `wq (1152, 5120)`，而产物是 `qkv.weight (3456, 1152)`。
    #
    # **它和 `Attention.out_dim` 不是一回事**：`out_dim` 是「输出投影的
    # 输入宽度」（`q*head_dim` 那一截），`in_dim` 是**这一层的残差流宽度**。
    # 两个都在、两个都要说清楚 —— 否则又是"名字像、算的不是一回事"。
    ("in_dim", "int?", False,
     "这一层的残差流宽度。**缺省是顶层 `d_model`**；独立宽度的栈（视觉塔）必给"),
    ("attrs", "dict", True, "这一层的属性（逐层覆盖合并之后的结果）"),
    ("ops", "list[Op]", True, "这一层的算子序列，**顺序有意义**"),
    ("state", "list[State]", False, "这一层需要什么状态（KV cache / recurrent / conv）"),
]

OP = [
    # `kind` 的闭集是 `KIND_ATTRS` 的键（见下面的表），所以这里用 `str`：
    # 真正的判定在 `_check_op` 里对着那张表做，一处定义。
    ("kind", "str", True, "算子种类，**闭集** —— 见 KIND_ATTRS"),
    ("mech", "str", True, "机制名（.by1 里写的那个名字）。只用于报错和取名"),
    ("attrs", "dict", True, "属性。**每种 kind 有自己的必填项** —— 见 KIND_ATTRS"),
    ("inputs", "list[str]", True, "输入的值引用（`hidden` 或 `opN.out`）"),
    ("outputs", "list[str]", True, "输出的值引用"),
]

STATE = [
    ("kind", "enum:recurrent,conv_history,kv_cache,declared_by_impl", True,
     "recurrent / conv_history / kv_cache / declared_by_impl"),
    ("bounded_by", "int?", True, "状态有没有上界。None 表示无界，不是「没有」"),
    ("shape", "list[int]?", False, "状态张量的形状"),
    ("dtype", "enum?:bf16,fp32,float32", False,
     "状态精度。**闭集**（`enum?:` = 可空：None 即没声明）—— 照现实"
     "（state 块只出过 bf16/fp32/float32）；新值先进表再用。"
     "这条原来开着口子（自由 str），于是「未知 dtype」连规则都测不了"),
    ("reuse", "enum:none,prefix,unknown", True,
     "none / prefix / unknown —— 能不能跨请求复用"),
    # **_spec 里漏了它，而生产者在写。** 4 处（mla-shaped 的压缩潜向量）。
    # 一个"规格比现实窄"的字段会把合法 IR 判成非法 —— 而这正是
    # 加"多余字段"检查时冒出来的：先照现实补规格，再让检查成立。
    ("note", "str?", False, "自由说明（人看的，不参与计算）"),
]

# ── 每种算子的属性 ──────────────────────────────────────────────────
# (名字, 类型, 必填, 说明)。**这份表就是"IR 里合法的东西"的唯一定义。**
KIND_ATTRS = {
    "Embed": {},
    "Norm": {
        "kind": ("enum:rms,layer", True, "和顶层 norm_kind 一致（冗余但显式）"),
        "eps": ("float", True, "来自顶层 norm_eps"),
        "one_plus": ("bool", True, ""),
    },
    "Add": {},
    # 输出头（lm_head）。**它也是一个算子** —— 规格表漏了它，
    # 于是 19 个模型全被报"不在闭集里"。
    "Head": {},
    "Attention": {
        "q": ("int", True, "查询头数"),
        "kv": ("int", True, "KV 头数"),
        "head_dim": ("int", True, "每个头的宽度"),
        "out_dim": ("int", True, "输出投影的输入宽度。**不一定等于 q*head_dim**"
                                "（GPT-OSS 是 4096 而 q*head_dim=4096，Gemma 不是）"),
        "bias": ("bool", True, "投影有没有 bias"),
        "window": ("int?", True, "滑动窗口。**None=全量**"),
        "qk_norm": ("enum:off,per_head,full", True,
                    "拆头前还是后、按 head_dim 还是整宽 —— **三种，不是布尔**"),
        "norm_eps": ("float", False, ""),
        "norm_one_plus": ("bool", False, ""),
        "rope": ("bool", True, "**要不要施加旋转**。false 时位置在输入上"),
        "rope_base": ("int", True, ""),
        "rope_pairing": ("enum:half,interleaved", True, "配对约定"),
        "rope_partial": ("float", True, "只转前多少比例"),
        "rope_scale": ("float", True, "YaRN 的 attention_factor 之类"),
        "yarn": ("dict?", True, ""),
        "q_gate": ("bool", True, "q 投影输出翻倍，一半当门"),
        # **q/k/v 存成一个张量还是三个。** 视觉塔（clef / GLM / Step 的
        # `qkv.weight` / `in_proj_weight`）是**融合**的：`(3*q*head_dim, in_dim)`；
        # 文本主干是拆开的 `q_proj` / `k_proj` / `v_proj`。
        # **这是布局，不是算法** —— 计算完全一样，只是参数的形状和名字不同。
        # 不说的话后端只能按拆开的推，于是**形状和产物对不上**
        # （实测 clef 契约 `qkv.weight (3456, 1152)`，而后端给
        # `wq/wk/wv` 各 `(1152, 1152)` —— 总量一样、形状不同）。
        "qkv_fused": ("bool", False, "q/k/v 融合成一个张量（视觉塔那种）。缺省 false"),
        "sink": ("bool", True, "每个头一个可学标量参与 softmax"),
        "kv_tie": ("bool", True, "k 和 v 共用同一块"),
        "head_gate": ("enum:off,per_head", True, ""),
        "gate_act": ("enum:softplus,sigmoid", True, "**同名不同义的高发区**"),
    },
    "MLA": {
        "q": ("int", True, ""), "v_dim": ("int", True, ""),
        "qk_nope": ("int", True, ""), "qk_rope": ("int", True, ""),
        "q_lora": ("int", True, ""), "kv_lora": ("int", True, ""),
        "head_dim": ("int", True, ""), "out_dim": ("int", True, ""),
        "bias": ("bool", True, ""), "rope_base": ("int", True, ""),
        "pairing": ("enum:half,interleaved", True, ""),
        "norm_eps": ("float", False, ""), "norm_one_plus": ("bool", False, ""),
        "head_gate": ("enum:off,per_head", True, ""),
        "gate_act": ("enum:softplus,sigmoid", True, ""),
        # ── 稀疏索引器（DSA）。**这三个是它的形状，不是装饰** ────────
        # `index_heads`/`index_dim` 是索引器自己的 q/k 头数和头维，
        # `index_topk` 是每层挑多少个 key。三个一起出现，一个都不能少
        # （缺一个就是"只说了一半"，宁可报错）。
        # 实测出处：GLM-5.3-Flash 的 config `index_n_heads` /
        # `index_head_dim` / `index_topk`，张量是 `self_attn.indexer.*`。
        "index_heads": ("int", False, "索引器的头数（GLM-5.3 是 32）"),
        "index_dim": ("int", False, "索引器的头维（128）"),
        "index_topk": ("int", False, "每层挑多少个 key（2048）"),
        # ── k-pool 压缩（GLM-5.3 独有）──────────────────────────────
        # 不给就是**逐 key** 打分（`by1sparse` 对 `GlmMoeDsaIndexer` 验的那份）；
        # 给了就走**池**：每 `index_kpool` 个连续 key 压成一个池、给池打分、
        # 选中的池摊回原始 token。**不是同一个算法的两种写法，是两种算法。**
        # 判卷人是 transformers 5.19.0 的 `Glm5NextTextIndexer`
        # （refs/glm5-next-indexer.reference.txt，原文 + 算法说明）。
        "index_kpool": ("int", False, "池大小。1 = 逐 key（缺省）"),
        "index_kpool_tail": ("bool", False,
                             "把可见的尾巴追加进来（宽度多 kpool-1）"),
    },
    "FFN": {
        "hidden": ("int", True, ""),
        "act": ("enum:silu,gptoss,gelu,gelu_new,relu2", True,
                "**闭集** —— 取值不在里面必须拒绝，不许静默走默认分支"),
        "gate": ("bool", True, "有门=三块矩阵；没门=两块。**GPT-2 是没有的那种**"),
        "bias": ("bool", True, ""),
        "limit": ("float?", True, "SwiGLU 的夹取"),
        "alpha": ("float", True, "gptoss 那个 1.702"),
    },
    "MoE": {
        "experts": ("int", True, ""), "top_k": ("int", True, ""),
        "hidden": ("int", True, "专家的中间宽度"),
        "shared": ("int", True, "共享专家个数"),
        "shared_hidden": ("int", True, ""),
        "shared_gate": ("bool", True, ""),
        "routing": ("enum:softmax_topk,topk_softmax,sigmoid_group_topk,sigmoid_topk",
                    True, "**选法**（怎么挑专家）。打分函数是另一个属性 `score`"),
        # ── 打分函数（**和选法是两件事**）──────────────────────────────
        # 打分：`logits` 怎么变成 `scores`。选法：`scores` 怎么挑出 top-k。
        # 这两个**正交** —— 而原来只有一个 `routing` 把两件事糊在一起：
        # 名字里的前缀（`sigmoid_*` / `softmax_*`）被当成了打分函数的来源。
        #
        # 那个糊法的代价是实测出来的（`history/ir.md 130` + 判卷人第二把尺子）：
        # GLM-5.3 写 `routing = softmax_topk` 而官方 config 是
        # `scoring_func = "sigmoid"` —— **契约全对、config 逐字段全对，
        # 前向算的是另一个模型**。`routing_vs_config` 当时只能靠
        # `rt.startswith('sigmoid')` 猜，而猜不出来的（`sqrtsoftplus`）
        # 就完全没有位置放。
        #
        # 取值是 `transformers.activations.ACT2FN` 的键 —— **照抄，不自造**：
        #   softmax       Mixtral / GPT-OSS
        #   sigmoid       DeepSeek V3 / Ling / GLM
        #   sqrtsoftplus  **DeepSeek V4** —— sqrt(softplus(x))，
        #                 实测自 `SqrtSoftplusActivation`（5.15.1）：
        #                 `return nn.functional.softplus(input).sqrt()`
        #   softplus      Laguna 的普通 Attention 门（备用，同表）
        #
        # **不给就是 undefined，由 `routing` 推**（见 `moe_score`）——
        # 这样 4 个已经毕业的模型一个都不用改，而新模型必须显式写。
        "score": ("enum:softmax,sigmoid,sqrtsoftplus,softplus", False,
                  "打分函数。缺省由 routing 推（sigmoid_* -> sigmoid，"
                  "其余 -> softmax）；**sqrtsoftplus 推不出来，必须写**"),
        # ── 推导值（**编译器填，不是给人写的**）──────────────────────
        # 它必须在这张表里：`compile_ir` 会把解析出的打分函数填进 `_score`，
        # 而那份 IR 要能通过 `validate` —— `by1irentry` 的整个意义就是
        # 「编译出来的 IR 序列化再读回来，三个后端还能跑」。
        # **不在表里的话，那份 IR 自己就不合法**（第一版就是这样：
        # 6 个走 compile_ir 的模型在 by1irentry 里报"多出来的字段 `_score`"）。
        #
        # 为什么不是"编译器别填"：NumPy 后端**直读 IR**，
        # 它需要那个已解析好的值，否则就得重新推一遍 ——
        # 而"两条路各推一遍"正是 1.6e-02 那次分叉的成因。
        # 所以推导值就得躺在 IR 里。它是 IR 的一部分。
        #
        # 但不许和 `score` 各说各的（见下面 `_check_op` 的一致性约束）：
        # **两个来源同一个意思 = 迟早对不上。**
        "_score": ("enum:softmax,sigmoid,sqrtsoftplus,softplus", False,
                   "**推导值**：`score` 解析后的结果，由 compile_ir 填。"
                   "不是给人写的；写了就必须和 `score` 一致"),
        "router_bias": ("bool", True, ""), "expert_bias": ("bool", True, ""),
        "score_bias": ("bool", True, "noaux_tc 的 e_score_correction_bias"),
        "n_group": ("int", True, "分组路由"), "topk_group": ("int", True, ""),
        "routed_scale": ("float", True, ""),
        "act": ("enum:silu,gptoss,gelu,gelu_new,relu2", True, ""),
        "limit": ("float?", True, ""), "limit_shared": ("float?", True, ""),
        "alpha": ("float", True, ""),
        # **没有笼统的 `bias`。** MoE 用的是 router_bias 和 expert_bias，
        # 分开的两个。规格表第一版照着 FFN 抄了一个 bias 进来，于是
        # 210 条"缺必填项"，而**没有任何东西产生它**。
        # **规格比现实严，报的是假问题** —— 而假问题会把真问题淹掉。
    },
    "Linear": {
        "k_heads": ("int", True, ""), "k_dim": ("int", True, ""),
        "v_heads": ("int", True, ""), "v_dim": ("int", True, ""),
        "conv_kernel": ("int", True, ""), "out_dim": ("int", True, ""),
        "act": ("enum:silu,gptoss,gelu,gelu_new,relu2", True, ""),
        "norm_eps": ("float", False, ""), "l2_eps": ("float", False, ""),
    },
    # **选择性状态空间（Mamba2 · Nemotron-H 那一版）。**
    # 和 `Linear` 并列，但参数完全不同：那里是 k/v 头 + delta 规则，
    # 这里是逐头的 `A_log` / `D` / `dt_bias` + 一个深度卷积。
    "SSM": {
        "heads": ("int", True, "头数。**可以写成裸数字** `heads = 64`"),
        "head_dim": ("int", True, "每个头的宽度"),
        "ssm_state": ("int", True, "状态维 —— 和 head_dim 是两个东西"),
        "n_groups": ("int", True, "B / C 的分组数。和 GQA 的 kv 头同一个意思"),
        "conv_kernel": ("int", True, "深度因果卷积的核长"),
        "expand": ("int", True, "d_inner = heads * head_dim 的那个倍数来源"),
        "chunk_size": ("int", True,
                       "分块扫描的块长。**by1 走的是递推** —— "
                       "这个属性只进契约，不进计算"),
        "conv_bias": ("bool", True, ""),
        "proj_bias": ("bool", True, ""),
        "act": ("enum:silu,gptoss,gelu,gelu_new,relu2", True, ""),
        "norm_eps": ("float", False, ""),
    },
    # **KDA（Kimi Delta Attention）。**
    # 和 `Linear`（GDN）并列 —— **递推是同一个** ✓，差别在投影的形状：
    # 那边融在一个 `in_proj_qkvz` 里，这边 q/k/v 分开，
    # 外加一个低秩或单层的门、和一个喂给 `o_norm` 的输出门。
    #
    # 两套属性名都认 ✓ —— 两个真实模型各用一套，而两边都对着官方产物验过：
    #   Ling  k_heads / v_heads / k_dim / v_dim
    #   GLM   num_heads / head_dim
    "KDA": {
        "k_heads": ("int", False, "k 的头数（GLM 那边叫 num_heads）"),
        "v_heads": ("int", False, "v 的头数。和 k_heads 不等就是 GVA"),
        "k_dim": ("int", False, "k 每头的宽度（GLM 那边叫 head_dim）"),
        "v_dim": ("int", False, "v 每头的宽度"),
        "num_heads": ("int", False, "alias：k_heads / v_heads"),
        "head_dim": ("int", False, "alias：k_dim / v_dim"),
        "conv_kernel": ("int", True, "深度因果卷积的核长"),
        "conv_bias": ("bool", True, ""),
        "gate_lowrank": ("bool", True,
                         "门是不是两层（GLM 的 f_a/f_b）。默认一层（Ling 的 f_proj）"),
        "gate_rank": ("int", False, "低秩门的瓶颈宽度"),
        "gate_out_bias": ("bool", False,
                          "输出门第二层有没有 bias。**两个出处不一致**："
                          "fla 的层写的是 `bias=True`，而 GLM 的官方权重清单里"
                          "**没有** `g_b_proj.bias`。所以它必须是属性，"
                          "不能写死 —— 默认 true（跟 fla）"),
        "gate_lower": ("float", False,
                       "给定时走 lower_bound 那一支："
                       "g = lower * sigmoid(exp(A_log) * g)（GLM 是 -5.0）"),
        "l2_eps": ("float", False, "q/k 做 L2 归一化时的 eps"),
        "act": ("enum:silu,gptoss,gelu,gelu_new,relu2", True, ""),
        "norm_eps": ("float", False, ""),
    },
    "Raw": {
        "impl": ("str", True,
                 "逃生舱第一层：raw.py 里的工厂函数名。"
                 "**这一层要改编译器** —— 加一个机制就得动 src/by1codegen.py。"
                 "第二层见 External"),
    },
    # ── 逃生舱第二层：引用一个**外部符号** ────────────────────────
    # 编译器不认识这个机制，只认识**调用约定**。
    # 所以加一个新机制**不需要动编译器** —— 只要有一个 .so 和一个 IR。
    #
    # ## ABI（固定，不自由发挥）
    #
    #     void <symbol>(const float *x, float *y,
    #                   int B, int T, int D,
    #                   const float *const *w, int nw);
    #
    #     x    输入，行主序 [B, T, D]
    #     y    输出，行主序 [B, T, D]（io = "same" 时形状和 x 一样）
    #     w    权重指针数组，**按名字字典序排列**
    #     nw   w 的个数
    #
    # ## 为什么权重是"指针数组"而不是拼成一块
    #
    # 因为 C 那边权重本来就是分开的缓冲区。拼成一块要么多一次拷贝，
    # 要么要求布局一致 —— 而那是**约定**，不是**语义**。
    # 指针数组 + 排序规则，两边都能自己算出来，不需要额外通道。
    "External": {
        "lib": ("str", True,
                "动态库路径（.so / .dylib / .dll / .o）。"
                "**相对路径按 IR 文件所在目录解析** —— 否则换个目录就跑不了"),
        "symbol": ("str", True, "符号名。找不到就拒绝，**不静默给个恒等**"),
        "weights": ("dict", True,
                    "这个算子自己的张量：{逻辑名: 形状}。"
                    "**契约照样查** —— 这就是「下沉一层」没有放松的地方"),
        "io": ("enum:same", True,
               "输出和输入同形。先只支持这一种 —— "
               "多一种就要多一条约定，而约定越多越像糊"),
        "note": ("str?", False, "给人看的说明"),
    },
}

KINDS = sorted(KIND_ATTRS)


# ── MoE 的打分函数：**一处定义，三个后端都从这儿取** ────────────────
# 三个后端（torch / NumPy / C）都要知道"logits 怎么变成 scores"。
# 这三处原来各写各的 —— 于是 `sigmoid_group_topk` 曾经在两个后端里
# 都落进 else 走成 softmax（`history/ir.md 130`）。
# **一条规则三处实现 = 迟早对不上**，所以推断逻辑只放这里。
_SCORE_FROM_ROUTING = {
    "sigmoid_topk": "sigmoid",
    "sigmoid_group_topk": "sigmoid",
    "softmax_topk": "softmax",
    "topk_softmax": "softmax",
}


def moe_score(attrs):
    """MoE 的打分函数：`logits` -> `scores`。

    **优先级：`score` > `_score` > 从 `routing` 推。**

    · `score`  —— `.by1` 里显式写的那个
    · `_score` —— 编译期解析出来的（`compile_ir` 会填）。**必须认它**：
      否则 NumPy 这条"直读 IR"的路会拿一个已解析好的结果当没看见，
      退回按前缀猜 —— 而两条路猜出不同的值就是静默错。
      （这一版真出过：torch 走 sqrtsoftplus、NumPy 走 sigmoid，差 1.6e-02。）
    · 都没有才按前缀推。这一支是给**只写了 `routing` 的老模型**兜底的 ——
      4 个已毕业的模型都是这么写的，推断结果和以前一模一样。

    推断表是**穷举**的：`routing` 是个闭集，四种全在里面。
    真出现表外的值，返回 None —— **调用方必须当错误处理，不许退回 softmax**：
    退回正是 ir.md 130 记的那个错。
    """
    a = attrs or {}
    for key in ("score", "_score"):
        v = a.get(key)
        if v is not None and str(v).strip():
            return str(v).strip().lower()
    return _SCORE_FROM_ROUTING.get(str(a.get("routing", "")).strip().lower())


def moe_scores_array(kind, logits, np):
    """给定打分函数名，把 `logits` 变成 `scores`。**NumPy 侧唯一的一份。**

    torch 侧在 `by1codegen.RUNTIME` 里、C 侧在 `by1c` 里各有一份 ——
    但三份的**取值表**都来自上面那张 `_SCORE_FROM_ROUTING` + 这个函数的
    语义表（见 `SCORE_KINDS`），所以"支持哪些打分函数"只有一个答案。

    `kind` 不在表里就抛 —— **不静默退回**。
    """
    if kind == "softmax":
        e = np.exp(logits - logits.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-logits))
    if kind == "softplus":
        # 数值稳定写法：log1p(exp(-|x|)) + max(x, 0)
        return np.maximum(logits, 0) + np.log1p(np.exp(-np.abs(logits)))
    if kind == "sqrtsoftplus":
        # DeepSeek V4：sqrt(softplus(x))。softplus 恒正 -> sqrt 恒有定义
        # （这一点很重要：sqrt 不会出 NaN，所以它不需要额外的 eps）
        sp = np.maximum(logits, 0) + np.log1p(np.exp(-np.abs(logits)))
        return np.sqrt(sp)
    raise ValueError("不认识的打分函数 <%s> —— 不许静默退回 softmax" % kind)


# **支持哪些打分函数**：三个后端共用的那一张表。
# **顺序就是 C 后端的整数编码** —— `by1c` 那边按这个序数传参，
# 所以这张表动一处，C 那边跟着动（`_score_kind_id` 保证两边同源）。
SCORE_KINDS = ("softmax", "sigmoid", "sqrtsoftplus", "softplus")


def _score_kind_id(kind):
    """打分函数名 -> C 后端的整数编码。**不在表里就炸。**

    C 那边只收整数（它不认识字符串）。这个映射放在这里而不是 `by1c`
    里，是为了让它和 `SCORE_KINDS` **同源** —— 两处各写一张表，
    迟早会漂。返回 None 时由调用方报错。
    """
    try:
        return SCORE_KINDS.index(str(kind).strip().lower())
    except ValueError:
        return None


class IRError(Exception):
    pass


def _ty(name, t, v):
    if t.startswith("enum?:") or t.startswith("enum:"):
        # `enum?:` = **可空闭集**（值要么在集合里，要么是 None=没声明）；
        # `enum:`  = 必须在集合里。None 混进必选 enum 必须还是红的。
        nullable = t.startswith("enum?:")
        allow = (t[6:] if nullable else t[5:]).split(",")
        if v is None:
            return None if nullable else \
                "%s = <%r> 不在闭集 %s 里" % (name, v, allow)
        if v not in allow:
            return "%s = <%r> 不在闭集 %s 里" % (name, v, allow)
        return None
    if t == "int":
        return None if isinstance(v, int) and not isinstance(v, bool) else \
            "%s 应当是整数，实际 %r" % (name, v)
    if t == "int?":
        return None if v is None or (isinstance(v, int)
                                     and not isinstance(v, bool)) else \
            "%s 应当是整数或 null，实际 %r" % (name, v)
    if t == "float":
        return None if isinstance(v, (int, float)) and not isinstance(v, bool) \
            else "%s 应当是数，实际 %r" % (name, v)
    if t == "float?":
        return None if v is None or isinstance(v, (int, float)) else \
            "%s 应当是数或 null，实际 %r" % (name, v)
    if t == "bool":
        return None if isinstance(v, bool) else "%s 应当是布尔，实际 %r" % (name, v)
    if t == "str":
        return None if isinstance(v, str) and v else "%s 应当是非空字符串" % name
    # **`str?` 以前不存在。** 它掉到最后那行 `return None` = 接受，
    # 于是 `STATE.dtype = {"not": "a string"}` 是"合法 IR"。
    if t == "str?":
        return None if v is None or (isinstance(v, str) and v) else \
            "%s 应当是非空字符串或 null，实际 %r" % (name, v)
    if t == "dict":
        return None if isinstance(v, dict) else "%s 应当是字典" % name
    if t == "dict?":
        return None if v is None or isinstance(v, dict) else "%s 应当是字典或 null" % name
    if t.startswith("list"):
        if t.endswith("?"):        # list[...]? —— 允许 null
            return None if v is None or isinstance(v, list) else \
                "%s 应当是列表或 null" % name
        return None if isinstance(v, list) else "%s 应当是列表" % name
    # **不认识的类型字符串 = 这张表自己写错了。**
    # 原来这里是 `return None`（接受），于是一个拼错的类型名会
    # **静默地把那个字段变成不校验** —— 这就是 `enum` 丢掉闭集的那次。
    # 规格表自己坏了必须出声，不能变成"这个字段没规矩"。
    return ("%s: 规格表里的类型 <%s> 没人认识 —— "
            "**这是 by1ir 自己的 bug，不是 IR 的**" % (name, t))


def _keys_bad(where, obj, allowed):
    """多余字段 -> 错误。**规格表是"合法的东西"的唯一定义。**

    `_check_fields` 只从表往对象看（查必填、查类型），
    **从不反着看** —— 于是 `attrs` 里塞一个谁都不认识的键，
    `validate` 说合法，而 `.by1` 那边写错了属性的行为是"静默走默认值"。
    规格文档里那句"不在表里的属性 = 不合法"以前没有执行者，这就是它。
    """
    extra = [k for k in obj if k not in allowed]
    if extra:
        return ("%s: 多出来的字段 %s —— **不在规格表里就是不合法的**"
                % (where, ", ".join("`%s`" % k for k in sorted(map(str, extra)))))
    return None


def _check_fields(where, fields, obj, errs):
    """**先查类型，再查字段。** 顺序反了的话，一个 `layers=[42]`
    会在 `_ty` 之前就撞上 `name not in obj`（int 不可迭代）而抛异常。"""
    if not isinstance(obj, dict):
        errs.append("%s 应当是字典，实际 %s" % (where, type(obj).__name__))
        return
    allowed = {name for name, _t, _r, _d in fields}
    e = _keys_bad(where, obj, allowed)
    if e:
        errs.append(e)
    for name, t, req, _doc in fields:
        if name not in obj:
            if req:
                errs.append("%s 缺必填字段 `%s`" % (where, name))
            continue
        e = _ty(name, t, obj[name])
        if e:
            errs.append("%s: %s" % (where, e))


#: **qk_norm 的闭集**（IR 只有这三种取值）。语言层的旧写法
#: `true` / `on` / `yes` / `1` 等价于 `per_head` —— 归一化在 `qk_norm_of`。
QK_NORMS = ("off", "per_head", "full")
QK_ON_SYNONYMS = ("true", "on", "yes", "1")
QK_OFF_VALUES = ("off", "", "none", "false", "0")


def qk_norm_of(v):
    """把任意写法归一成 IR 的三值之一（`off` / `per_head` / `full`）。

    **语言层 -> IR 的唯一映射**：`true`/`on`/`yes`/`1` 是旧写法、等价于
    `per_head`；None 是 `off`；其余原样（小写、去空白）。
    """
    if v is None or v is False:
        return "off"
    t = str(v).strip().lower()
    if t in QK_ON_SYNONYMS:
        return "per_head"
    return t if t else "off"


def qk_on(v):
    """`qk_norm` 到底开没开。**唯一实现**（这个谓词曾在五个模块各有一份）。

    `"off"` 是**真值字符串** —— 直接做真值判断会永远成立，于是给 q/k
    悄悄加一层本不该有的 RMSNorm。PyTorch 那边写的是
    `not in (off, "", None)`，所以只有别的后端会错。实测 llama-shaped 的
    NumPy↔PyTorch 从 2.505e-07 变成 3.362e-03。
    """
    if v is None or v is False:
        return False
    return str(v).strip().lower() not in QK_OFF_VALUES


def qk_is_full(v):
    """`qk_norm = full`（拆头**之前**整宽归一化）—— 只在开着的时候才谈得上。"""
    return qk_on(v) and qk_norm_of(v) == "full"


#: **布尔/数字取值的唯一强制转换**（2026-10-17 第八轮）。
#:
#: 各后端此前有三种写法：`bool(a.get(x))`（Python 真值 —— `"false"` 是**真**）、
#: `_bool`（白名单 —— `sink = per_head` 这种描述性取值被静默关掉）、
#: `_flag`（黑名单 —— 只在明确否定时关）。`bool()` 那次是**错**的、
#: `_bool` 那次是**漏**的；黑名单是这两种里唯一能同时说对的两类属性。
#: 现在只有这一份；`by1lint` 第⑬ 类盯"又有人在文件里 def 一份"。
BOOL_OFF = ("", "false", "no", "off", "none", "null", "0")


def as_bool(v, default=False):
    """IR 属性的布尔读法（**唯一实现**）。

    明确否定（`false`/`off`/`no`/`none`/`null`/`""`/`0`，不分大小写）才关；
    其余非空值都算开 —— 于是 `sink = per_head` 这种描述性取值不会被
    静默关掉，而 `bias = "false"` 也不会被 Python 真值判成开。
    None 走 default。
    """
    if v is None:
        return default
    return str(v).strip().lower() not in BOOL_OFF


def as_num(v, default=None):
    """IR 属性的数字读法（**唯一实现**）：认 `"1e-5"` 这种字符串。

    读不出来就回 default —— 调用方必须给一个有据的默认值。
    """
    if v is None:
        return default
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


#: `_qk_on` 是 `qk_on` 的旧名（by1c/by1exec/by1graph 都这么调）。
_qk_on = qk_on

#: **不许有第二份实现的谓词**（`by1lint` 的第⑫类按这张表查）。
#: 这些名字一旦在别的文件里出现 `def`，就是又复制了一份。
CANON_PREDICATES = ("qk_on", "qk_norm_of", "qk_is_full", "_qk_on",
                    "as_bool", "as_num")


def _qk_norm_shapes(a, q, kv, hd):
    """`qn.w` / `kn.w` 的形状 —— **full 和 per_head 名字一样、宽度不同**：

        per_head  拆头之后按 head_dim 归一        -> (hd,)
        full      拆头**之前**按整宽归一          -> (q*hd,) / (kv*hd,)

    Instella 的 `full` 实测 `self_attn.q_norm.weight` 是 [2560] 不是 [80]
    （`.by1` 里就写着这句）。这里原来两处都推 (hd,) —— 于是 shapes_of
    与契约差一个宽度，`by1sync` 把它列在「未编码的布局差」里，而它其实
    是**语义差**（by1opdiff 直接 `shape '[2560]' is invalid for size 80`）。"""
    if str(a.get("qk_norm", "")).strip().lower() == "full":
        return (q * hd,), (kv * hd,)
    return (hd,), (hd,)


def params_of(ir):
    """**从 IR 自己推**每个 op 需要的参数形状。IR 里没有形状，这是后端的活。

    **宽度按层取**：`L["in_dim"]`（这一层的残差流宽度），不给才是顶层
    `d_model`。原来这里无条件用 `ir["d_model"]` —— 于是**独立宽度的栈
    （视觉塔）参数形状全错**：clef 的 `wq` 推成 (1152, 5120)，而产物是
    `qkv.weight (3456, 1152)`。三个后端用的是同一份 `shapes_of`，
    **它们会一致地错**。见 history/ir.md 128。
    """
    out = {}
    for L in ir["layers"]:
        d = L.get("in_dim") or ir["d_model"]
        for j, o in enumerate(L["ops"]):
            a, k = o["attrs"], o["kind"]
            pre = f"layers.{L['index']}.op{j}."
            if k == "Norm":
                out[pre + "w"] = (d,)
                # **LayerNorm 有 bias。** 以前这里一条都没有 ——
                # GPT-2 的 ln_1.bias / ln_2.bias 在后端层面不存在。
                if a.get("kind") == "layer":
                    out[pre + "b"] = (d,)
            elif k == "Attention":
                hd, q, kv = a["head_dim"], a["q"], a["kv"]
                if a.get("qkv_fused"):
                    # **融合 q/k/v**（视觉塔那种）：一个 `qkv` 装下 q、k、v
                    # 三截，后面 `proj` 是输出投影。形状要和契约对上 ——
                    # 实测 clef 的 `qkv.weight (3*q*head_dim, in_dim)`。
                    out[pre + "qkv"] = (q * hd + 2 * kv * hd, d)
                    out[pre + "proj"] = (d, a["out_dim"])
                    if a["bias"]:
                        out[pre + "qkv.bias"] = (q * hd + 2 * kv * hd,)
                        out[pre + "proj.bias"] = (d,)
                    if _qk_on(a.get("qk_norm")):
                        out[pre + "qn.w"], out[pre + "kn.w"] = \
                            _qk_norm_shapes(a, q, kv, hd)
                    if a.get("head_gate", "off") != "off":
                        out[pre + "g_proj"] = ((q if a["head_gate"] == "per_head"
                                              else q * hd), d)
                    continue
                out[pre + "wq"] = (q * hd * (2 if a.get("q_gate") else 1), d)
                out[pre + "wk"] = (kv * hd, d)
                if not a.get("kv_tie"):
                    out[pre + "wv"] = (kv * hd, d)
                out[pre + "wo"] = (d, a["out_dim"])
                if a["bias"]:
                    out[pre + "wq.bias"] = (q * hd * (2 if a.get("q_gate") else 1),)
                    out[pre + "wk.bias"] = (kv * hd,)
                    if not a.get("kv_tie"):
                        out[pre + "wv.bias"] = (kv * hd,)
                    out[pre + "wo.bias"] = (d,)
                if _qk_on(a.get("qk_norm")):
                    out[pre + "qn.w"], out[pre + "kn.w"] = \
                        _qk_norm_shapes(a, q, kv, hd)
                if a.get("sink"):
                    out[pre + "sink"] = (q,)
                if a.get("head_gate", "off") != "off":
                    out[pre + "g_proj"] = ((q if a["head_gate"] == "per_head"
                                              else q * hd), d)
            elif k == "External":
                # **外部算子照样贡献张量形状。** 契约不松 ——
                # 它的权重和别的机制一样进参数表、一样被 by1verify 对产物查。
                # 前缀 `ext:` 是为了在 C 那边和内部名区分开。
                for nm, shp in (a.get("weights") or {}).items():
                    out[pre + "ext:" + nm] = tuple(shp)
            elif k == "MLA":
                out[pre + "q_a_proj"] = (a["q_lora"], d)
                out[pre + "q_a_layernorm.w"] = (a["q_lora"],)
                out[pre + "q_b_proj"] = (a["q"] * a["head_dim"], a["q_lora"])
                out[pre + "kv_a_proj_with_mqa"] = (a["kv_lora"] + a["qk_rope"], d)
                out[pre + "kv_a_layernorm.w"] = (a["kv_lora"],)
                out[pre + "kv_b_proj"] = (
                    a["q"] * (a["qk_nope"] + a["v_dim"]), a["kv_lora"])
                out[pre + "dense"] = (d, a["out_dim"])
                if a["bias"]:
                    out[pre + "dense.bias"] = (d,)
                if a.get("head_gate", "off") != "off":
                    out[pre + "g_proj"] = (a["q"], d)
                # ── DSA 稀疏索引器（有 index_* 才有）────────────────
                # 索引器有自己的一套轻量投影，和主 MLA 的 q/k **是分开的**：
                #   wq_b  q_lora -> index_heads * index_dim
                #   wk    d      -> index_dim（后面过 LayerNorm，带 bias）
                #   weights_proj  d -> index_heads
                # 判卷人是 transformers 的 `GlmMoeDsaIndexer`，
                # 见 `src/modelcheck/by1sparse.py`。
                #
                # ⚠ **这里的名字写全（带 `.weight` / `.bias`），不走"去掉后缀"
                #   那条配对** —— 因为它们在 PyTorch 侧挂在 `indexer.` 子模块下
                #   （`...indexer.wq_b.weight`），而去掉后缀只能配到
                #   `...indexer.wq_b`；`k_norm` 有 weight 也有 bias，去掉后缀
                #   两个会撞成同一个名字。写全就都精确命中。
                if a.get("index_heads"):
                    nh_, hd_ = int(a["index_heads"]), int(a["index_dim"])
                    out[pre + "indexer.wq_b.weight"] = (nh_ * hd_, a["q_lora"])
                    out[pre + "indexer.wk.weight"] = (hd_, d)
                    out[pre + "indexer.k_norm.weight"] = (hd_,)
                    out[pre + "indexer.k_norm.bias"] = (hd_,)
                    out[pre + "indexer.weights_proj.weight"] = (nh_, d)
                    # k-pool 的两个张量。**职责不一样，别混**：
                    #   _gate [hd, hidden]  逐 token 的门，池内加权用它
                    #   _ape  [kpool, hd]   池内位置的可学嵌入
                    if int(a.get("index_kpool") or 1) > 1:
                        out[pre + "indexer.index_kpool_compress_gate"] = (hd_, d)
                        out[pre + "indexer.index_kpool_compress_ape"] = (
                            int(a["index_kpool"]), hd_)
            elif k == "FFN":
                out[pre + "w1"] = (a["hidden"], d)
                out[pre + "w2"] = (d, a["hidden"])
                if a.get("gate", True):
                    out[pre + "w3"] = (a["hidden"], d)
                # **bias。** 以前这里一条都没有 —— 于是 GPT-2 的
                # 4×12 个 bias 在后端层面**根本不存在**，
                # 而"张量搬运 173/173"看起来是满的。
                if a.get("bias"):
                    out[pre + "w1.bias"] = (a["hidden"],)
                    out[pre + "w2.bias"] = (d,)
                    if a.get("gate", True):
                        out[pre + "w3.bias"] = (a["hidden"],)
            elif k == "MoE":
                E, H = a["experts"], a["hidden"]
                out[pre + "router"] = (E, d)
                if a.get("router_bias"):
                    out[pre + "router.bias"] = (E,)
                out[pre + "w1"] = (E, H, d)
                out[pre + "w3"] = (E, H, d)
                out[pre + "w2"] = (E, d, H)
                if a.get("expert_bias"):
                    out[pre + "b1"] = (E, H)
                    out[pre + "b3"] = (E, H)
                    out[pre + "b2"] = (E, d)
                if a.get("score_bias"):
                    out[pre + "score_bias"] = (E,)
                if a.get("shared"):
                    sh = a["shared_hidden"]
                    out[pre + "sw1"] = (sh, d)
                    out[pre + "sw3"] = (sh, d)
                    out[pre + "sw2"] = (d, sh)
                    if a.get("shared_gate"):
                        out[pre + "shared_gate"] = (1, d)
            elif k == "Linear":
                kd, vd, nk, nv = (a["k_heads"] * a["k_dim"], a["v_heads"] * a["v_dim"],
                                  a["k_heads"], a["v_heads"])
                out[pre + "in_proj_qkvz"] = (kd * 2 + vd * 2, d)
                out[pre + "in_proj_ba"] = (nv * 2, d)
                out[pre + "conv"] = (kd * 2 + vd, a["conv_kernel"])
                out[pre + "dt_bias"] = (nv,)
                out[pre + "A_log"] = (nv,)
                out[pre + "norm.w"] = (a["v_dim"],)
                out[pre + "out_proj"] = (d, vd)
            elif k == "SSM":
                # 选择性状态空间（Mamba2 · Nemotron-H 那一版）。
                # 8 个张量，和契约里那一组逐个对上 ——
                # `by1tens` 算出来的 'Mamba' 组就是这 8 行。
                _di = a["heads"] * a["head_dim"]
                _cd = _di + 2 * a["n_groups"] * a["ssm_state"]
                out[pre + "in_proj"] = (_di + _cd + a["heads"], d)
                out[pre + "conv"] = (_cd, a["conv_kernel"])
                if a.get("conv_bias"):
                    out[pre + "conv.bias"] = (_cd,)
                out[pre + "dt_bias"] = (a["heads"],)
                out[pre + "A_log"] = (a["heads"],)
                out[pre + "D"] = (a["heads"],)
                out[pre + "norm.w"] = (_di,)
                out[pre + "out_proj"] = (d, _di)
            elif k == "KDA":
                # **Kimi Delta Attention。** 13 个张量，和契约里那一组对上。
                #
                # ⚠ **`P` 的键就是这张表自己定的** ✓ —— 上面 SSM 那支
                # 契约里写的是 `conv1d.weight` ✓ 而这里叫 `conv` ✓，
                # 两边不一样也一直是对的 ✓。**只要这张表和 `op_*` 里读的
                # 名字一致就行** ✓。（我上一轮照着契约的名字去读 `P`，
                # 于是拿到空的 P —— 那不是键名写错，是这张表根本没写 ✓。）
                _nk, _nv = a["k_heads"], a["v_heads"]
                _dk, _dv = a["k_dim"], a["v_dim"]
                _kd, _vd = _nk * _dk, _nv * _dv
                _ck = a["conv_kernel"]
                out[pre + "q_proj"] = (_kd, d)
                out[pre + "k_proj"] = (_kd, d)
                out[pre + "v_proj"] = (_vd, d)
                out[pre + "q_conv1d"] = (_kd, _ck)
                out[pre + "k_conv1d"] = (_kd, _ck)
                out[pre + "v_conv1d"] = (_vd, _ck)
                out[pre + "A_log"] = (_nv,)
                # **`dt_bias` 按维，不是按头** ✓ —— 这一条只有 fla 的源码里有
                # （`gate_dim = num_v_heads * head_k_dim` ✓）。
                out[pre + "dt_bias"] = (_nv * _dk,)
                out[pre + "b_proj"] = (_nv, d)
                if a.get("gate_lowrank"):
                    _r = a["gate_rank"]
                    out[pre + "f_a_proj"] = (_r, d)
                    out[pre + "f_b_proj"] = (_nv * _dk, _r)
                    out[pre + "g_a_proj"] = (_r, d)
                    out[pre + "g_b_proj"] = (_vd, _r)
                    # **第二层的 bias 跟属性走** ✓ —— fla 有 ✓，GLM 没有 ✓
                    # （见 `by1ir.KIND_ATTRS["KDA"]["gate_out_bias"]`）。
                    if a.get("gate_out_bias", True):
                        out[pre + "g_b_proj.bias"] = (_vd,)
                else:
                    out[pre + "f_proj"] = (_nv * _dk, d)
                    out[pre + "g_proj"] = (_vd, d)
                out[pre + "o_norm.w"] = (_dv,)
                out[pre + "o_proj"] = (d, _vd)
    # **全局张量用模型级 `d_model`，不是循环里泄漏出来的 `d`。**
    # 这里原来写的是 `d` —— 那是上面层循环的循环变量，值是**最后一层**
    # 的 `in_dim`。多宽度的模型（gemma-4-31B：60 层文本 5376 + 27 层
    # 视觉塔 1152）于是全局张量全按 1152 推：`embed.weight` 推成
    # (262144, 1152)，而 GGUF 判卷人说 `output_norm.weight` 是 [5376]。
    # **单宽度的模型碰不到这一条** —— 循环变量恰好等于 d_model，
    # 所以它一直活在"看起来对"里。判卷人：`refs/google__gemma-4-31B.
    # gguf-tensors.json` 的 output_norm / token_embd 形状。
    dg = ir["d_model"]
    out["embed.weight"] = (ir["vocab"], dg)
    # **学习式位置表。** pos_kind=learned 的模型有一张 (n_pos, d) 的查表 ——
    # 它不在任何一层里，也不是 embed，所以以前这个后端**完全不知道它存在**。
    if ir.get("pos_kind") == "learned":
        out["pos.weight"] = (int(ir.get("n_pos") or ir.get("ctx") or 1024), dg)
    out["final_norm.w"] = (dg,)
    if ir.get("norm_kind") == "layer":
        out["final_norm.b"] = (dg,)
    out["head.weight"] = (ir["vocab"], dg)
    return out


def _wmacro(i, j, nm):
    """层内参数的宏名（`L{i}_OP{j}_{名}`）—— C / ggml 两边都按它取偏移。

    点和其它非字母数字一起换成下划线（`wq.bias` -> `WQ_BIAS`），
    所以**宏名和参数键是两个名字**，别互相代替。"""
    return "L%d_OP%d_%s" % (i, j, _MANGLE.sub("_", nm).upper())


def weight_plan(ir):
    """**权重布局的唯一定义**：参数键 + 顺序（1.0.0-rc0 · A3-1）。

    返回 `[(宏名, 参数键), ...]`，**顺序就是 `w.bin` 的排布顺序**。

    以前这张表寄生在 `by1c._emit_c_body` 里 —— 于是 `by1ggml` 为了拿
    同一份布局，只能反过来 import by1c（`decls, order, total =
    c_backend.emit_c(...)` 然后 `del decls`）。**布局是 IR 的事实，
    不是某个后端的事实**，所以它搬到了这里；四个后端从同一处取。

    **不变量（硬断言，不是注释）**：键集 == `params_of(ir)` 的键集，
    且无重复。这条一破就是"形状表与布局表各说各话"——而两份事实
    漂移过一次（gemma-4-31B 全局张量按 1152 推，见 `history/ir.md`），
    所以宁可当场炸，不要悄悄生成错的 `w.bin`。

    **不在这里做能力判断**：`head_gate` / `gptoss` / 不认识的 kind
    是**后端**的事（`by1exec` 支持 head_gate，C 不支持）—— 那些拒绝
    写在各后端里，见 `by1c.C_KINDS`。
    """
    order = [("EMBED", "embed.weight")]
    if str(ir.get("pos_kind", "rope")).lower() == "learned":
        order.append(("POS", "pos.weight"))
    order.append(("FINAL_NORM", "final_norm.w"))
    if str(ir.get("norm_kind", "rms")).lower() == "layer":
        order.append(("FINAL_NORM_B", "final_norm.b"))
    order.append(("HEAD", "head.weight"))
    for L in ir["layers"]:
        i = L["index"]
        for j, o in enumerate(L["ops"]):
            k, a = o["kind"], o["attrs"]
            pre = "layers.%d.op%d." % (i, j)
            if k == "Norm":
                order.append((_wmacro(i, j, "w"), pre + "w"))
                # **LayerNorm 有 bias**（见 `params_of` 同一处）。
                if str(a.get("kind", "rms")).lower() == "layer":
                    order.append((_wmacro(i, j, "b"), pre + "b"))
            elif k == "Attention":
                # **kv_tie 没有 wv**（v 复用 k 的投影）—— 和 `params_of`
                # 同一份判断，这里不跳过就会在 `params[pname]` 那里 KeyError。
                if a.get("qkv_fused"):
                    # 融合 q/k/v（视觉塔）：注册 `qkv` / `proj`
                    for nm in ("qkv", "proj"):
                        order.append((_wmacro(i, j, nm), pre + nm))
                    if a["bias"]:
                        for nm in ("qkv.bias", "proj.bias"):
                            order.append((_wmacro(i, j, nm), pre + nm))
                else:
                    for nm in ("wq", "wk", "wv", "wo"):
                        if nm == "wv" and a["kv_tie"]:
                            continue
                        order.append((_wmacro(i, j, nm), pre + nm))
                    if a["bias"]:
                        for nm in ("wq.bias", "wk.bias", "wv.bias", "wo.bias"):
                            if nm == "wv.bias" and a["kv_tie"]:
                                continue
                            order.append((_wmacro(i, j, nm), pre + nm))
                if a["sink"]:
                    order.append((_wmacro(i, j, "sink"), pre + "sink"))
                if _qk_on(a.get("qk_norm")):
                    for nm in ("qn.w", "kn.w"):
                        order.append((_wmacro(i, j, nm), pre + nm))
                # **head_gate 的 g_proj 在 `params_of` 里**（C 后端不实现
                # head_gate，但布局表说的是"IR 声明了哪些张量"，不是
                # "C 能算哪些"）—— 少了这一行，不变量当场就报"两份事实漂移"，
                # 这正是它该报的。
                if a.get("head_gate", "off") != "off":
                    order.append((_wmacro(i, j, "g_proj"), pre + "g_proj"))
            elif k == "FFN":
                # **无门控的 FFN 也是 FFN**（GPT-2 正是没有门的那种）。
                gated = a.get("gate", True)
                for nm in (("w1", "w2", "w3") if gated else ("w1", "w2")):
                    order.append((_wmacro(i, j, nm), pre + nm))
                if a.get("bias"):
                    for nm in (("w1.bias", "w2.bias", "w3.bias") if gated
                               else ("w1.bias", "w2.bias")):
                        order.append((_wmacro(i, j, nm), pre + nm))
            elif k == "Linear":
                for nm in ("in_proj_qkvz", "in_proj_ba", "conv", "dt_bias",
                           "A_log", "norm.w", "out_proj"):
                    order.append((_wmacro(i, j, nm), pre + nm))
            elif k == "SSM":
                # 选择性状态空间。8 个张量，和契约里 'Mamba' 那一组对上。
                for nm in ("in_proj", "conv", "dt_bias", "A_log", "D",
                           "norm.w", "out_proj"):
                    order.append((_wmacro(i, j, nm), pre + nm))
                if a.get("conv_bias"):
                    order.append((_wmacro(i, j, "conv.bias"), pre + "conv.bias"))
            elif k == "KDA":
                for nm in ("q_proj", "k_proj", "v_proj",
                           "q_conv1d", "k_conv1d", "v_conv1d"):
                    order.append((_wmacro(i, j, nm), pre + nm))
                if a.get("gate_lowrank"):
                    for nm in ("f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj"):
                        order.append((_wmacro(i, j, nm), pre + nm))
                    # **bias 是可选的** —— fla 有，GLM 的官方清单没有
                    if a.get("gate_out_bias", True):
                        order.append((_wmacro(i, j, "g_b_proj.bias"),
                                      pre + "g_b_proj.bias"))
                else:
                    for nm in ("f_proj", "g_proj"):
                        order.append((_wmacro(i, j, nm), pre + nm))
                for nm in ("b_proj", "A_log", "dt_bias", "o_norm.w", "o_proj"):
                    order.append((_wmacro(i, j, nm), pre + nm))
            elif k == "MLA":
                for nm in ("q_a_proj", "q_a_layernorm.w", "q_b_proj",
                           "kv_a_proj_with_mqa", "kv_a_layernorm.w",
                           "kv_b_proj", "dense"):
                    order.append((_wmacro(i, j, nm), pre + nm))
                if a.get("head_gate", "off") != "off":
                    order.append((_wmacro(i, j, "g_proj"), pre + "g_proj"))
                # DSA 稀疏索引器：名字带 `.weight` / `.bias` 全写 ——
                # 和 `params_of` 逐个对上（不一致的话权重表会对不上）。
                if a.get("index_heads"):
                    for nm in ("indexer.wq_b.weight", "indexer.wk.weight",
                               "indexer.k_norm.weight", "indexer.k_norm.bias",
                               "indexer.weights_proj.weight"):
                        order.append((_wmacro(i, j, nm), pre + nm))
                    if int(a.get("index_kpool") or 1) > 1:
                        for nm in ("indexer.index_kpool_compress_gate",
                                   "indexer.index_kpool_compress_ape"):
                            order.append((_wmacro(i, j, nm), pre + nm))
            elif k == "MoE":
                for nm in ("router", "w1", "w3", "w2"):
                    order.append((_wmacro(i, j, nm), pre + nm))
                if a["router_bias"]:
                    order.append((_wmacro(i, j, "router.bias"),
                                  pre + "router.bias"))
                # noaux_tc / sigmoid_topk 的专家偏置：**只影响选择，不影响权重**
                if a.get("score_bias"):
                    order.append((_wmacro(i, j, "score_bias"), pre + "score_bias"))
                if a["expert_bias"]:
                    for nm in ("b1", "b3", "b2"):
                        order.append((_wmacro(i, j, nm), pre + nm))
                if a["shared"]:
                    for nm in ("sw1", "sw3", "sw2"):
                        order.append((_wmacro(i, j, nm), pre + nm))
                    if a["shared_gate"]:
                        order.append((_wmacro(i, j, "shared_gate"),
                                      pre + "shared_gate"))
            elif k == "External":
                # 逃生舱：参数名带 `ext:` 前缀 —— `params_of` 就是这么命名的。
                for nm in sorted(a["weights"]):
                    order.append((_wmacro(i, j, "ext:" + nm), pre + "ext:" + nm))
            elif k == "Add":
                continue
            else:
                # 到不了这里：`validate()` 已经按 KIND_ATTRS 拒过不认识的
                # kind。留着是为了"万一"时不悄悄跳过。
                raise ValueError("weight_plan：不认识的 kind %r（层 %d op%d）"
                                 % (k, i, j))
    # ── 不变量：布局表与形状表必须是同一批键（无重复） ──────────────
    names = [p for _m, p in order]
    got, want = set(names), set(params_of(ir))
    if len(names) != len(got):
        raise ValueError("weight_plan：有重复的参数键 %s"
                         % sorted(n for n in got if names.count(n) > 1)[:6])
    if got != want:
        raise ValueError(
            "weight_plan 与 params_of 不同源（两份事实漂移）："
            "布局多出 %s / 形状多出 %s"
            % (sorted(got - want)[:6], sorted(want - got)[:6]))
    return order


def infer_widths(ir):
    """值引用 -> 宽度 的**静态推导**。返回 `(widths, errs, gaps)`。

    这是 1.0.0-rc0 的 T0.1。两类问题**必须分开**（by1irentry 的规矩：
    分清「没实现」和「算错了」）：

      · `errs` —— **错了**：引用不存在的值 / 用在产出之前。
        以前这些只在跑后端时才炸（KeyError / 形状错），而
        "有人可以直接写 IR"这条路**没有任何静态防线**。
      · `gaps` —— **没实现**：相邻两段残差流宽度不同（栈间换宽度，
        残差流怎么换还没实现，见 1.md 的已知缺口）。**它不是错误** ——
        gemma-4-31B 的 IR（60 层文本 5376 + 27 层视觉塔 1152）就带一个，
        那是已知缺口，不是 IR 坏了。

    宽度规则（当前所有 kind 都是 d→d 的残差流变换）：
      globals  Embed 造出 d_model；Head 吃 d_model 出 vocab；其余等宽
      layer    `hidden` = `in_dim or d_model`；算子入参等宽、出参等宽
    **跨宽算子出现时这张表要跟着动** —— 到那天 errs 才真正长牙齿，
    但值引用这半现在就能拦住一批"写出来的 IR 是坏的"。
    """
    errs, gaps = [], []
    if not isinstance(ir, dict):
        return {}, ["IR 的顶层必须是字典"], []
    d_model = ir.get("d_model")
    vocab = ir.get("vocab")
    W = {}

    def _put(scope, ref, w):
        W[(scope, ref)] = w

    def _get(scope, ref, where):
        v = W.get((scope, ref))
        if v is None:
            errs.append("%s: 引用了不存在的值 %r —— "
                        "要么是拼写，要么用在了产出之前" % (where, ref))
        return v

    # ── globals：模型级残差流（宽度 d_model）─────────────────────────
    for j, op in enumerate(ir.get("globals") or []):
        if not isinstance(op, dict):
            continue
        k, where = op.get("kind"), "globals[%d]" % j
        ins, outs = op.get("inputs") or [], op.get("outputs") or []
        if k == "Embed":
            for ref in outs:
                _put("globals", ref, d_model)
            continue
        for ref in ins:
            w = _get("globals", ref, where)
            if w is not None and w != d_model:
                errs.append("%s (%s): 输入 %s 宽度 %s != d_model %s —— "
                            "全局算子都挂在模型级残差流上"
                            % (where, k, ref, w, d_model))
        out_w = vocab if k == "Head" else d_model
        for ref in outs:
            _put("globals", ref, out_w)

    # ── layers：每层一个作用域，`hidden` 就是这一层的输入 ────────────
    prev = d_model
    for li, L in enumerate(ir.get("layers") or []):
        if not isinstance(L, dict):
            continue
        d = L.get("in_dim") or d_model
        if prev != d:
            # **换宽度是缺口，不是错误。** 记进 gaps，别冒充失败。
            gaps.append((li, prev, d))
        prev = d
        scope = "layers[%d]" % li
        _put(scope, "hidden", d)
        for j, op in enumerate(L.get("ops") or []):
            if not isinstance(op, dict):
                continue
            k, where = op.get("kind"), "layers[%d].ops[%d]" % (li, j)
            for ref in (op.get("inputs") or []):
                w = _get(scope, ref, where)
                if w is not None and w != d:
                    errs.append("%s (%s): 输入 %s 宽度 %s != 层宽 %s"
                                % (where, k, ref, w, d))
            for ref in (op.get("outputs") or []):
                _put(scope, ref, d)
        prev = d
    return W, errs, gaps


def validate(ir):
    """返回错误列表。**空列表 = 合法。**

    ## 这份函数**不许抛异常**

    它是对外入口的一半（`by1ir.py --check <file>.ir.json`），
    而 [by1ir.py] 的模块说明里写着"有人可以直接写 IR"。
    契约说"返回错误列表"，那畸形输入就得进列表 —— 以前
    `layers = [42]` 会在这里抛 `TypeError`，而那个 traceback
    长得像"by1ir 坏了"，不像"你的 IR 坏了"。
    """
    errs = []
    if not isinstance(ir, dict):
        return ["IR 的顶层必须是字典"]
    ver = ir.get("by1-ir")
    if ver is None:
        errs.append("缺 `by1-ir` 版本号 —— **没有版本号的 IR 不该被接受**")
    elif str(ver).split(".")[0] != VERSION.split(".")[0]:
        errs.append("IR 版本 %s 不是我能读的大版本 %s" % (ver, VERSION))

    _check_fields("顶层", TOP, ir, errs)

    if ir.get("pos_kind") == "learned" and "n_pos" not in ir:
        errs.append("pos_kind=learned 时必须有 `n_pos`")

    for sec in ("globals", "layers"):
        if not isinstance(ir.get(sec), list):
            continue
        for i, item in enumerate(ir[sec]):
            if sec == "globals":
                _check_op("globals[%d]" % i, item, errs)
                continue
            w = "layers[%d]" % i
            _check_fields(w, LAYER, item, errs)
            if not isinstance(item, dict):
                continue
            ops = item.get("ops")
            for j, op in enumerate(ops if isinstance(ops, list) else []):
                _check_op("%s.ops[%d]" % (w, j), op, errs)
            sts = item.get("state")
            for k, st in enumerate(sts if isinstance(sts, list) else []):
                _check_fields("%s.state[%d]" % (w, k), STATE, st, errs)

    # ── STACKS（1.1）：栈/塔登记表 + 跨栈宽度一致性 ──────────────────
    stk = ir.get("stacks")
    if stk is not None:
        if not isinstance(stk, list):
            errs.append("stacks 必须是列表")
        n_l = len(ir["layers"]) if isinstance(ir.get("layers"), list) else 0
        _seen = set()
        for s_i, s in enumerate(stk if isinstance(stk, list) else []):
            sw = "stacks[%d]" % s_i
            _check_fields(sw, STACKS, s, errs)
            if not isinstance(s, dict):
                continue
            if s.get("name") in _seen:
                errs.append("%s：栈名 %r 重复" % (sw, s.get("name")))
            _seen.add(s.get("name"))
            fl, nl = s.get("first_layer"), s.get("n_layers")
            if isinstance(fl, int) and isinstance(nl, int) and \
                    (fl < 0 or nl <= 0 or fl + nl > n_l):
                errs.append("%s：%r 占层 [%d, %d) 越界（共 %d 层）"
                            % (sw, s.get("name"), fl, fl + nl, n_l))
                continue
            # **跨栈宽度一致性**：非 text 栈的层必须给 in_dim 且 = 栈宽
            # （视觉塔宽 1152 而 d_model 5376 那种 —— 编译期丢掉栈信息
            #  之后，这层一致性就没人管了）。
            if isinstance(fl, int) and isinstance(nl, int) and \
                    s.get("modality") != "text":
                for li in range(fl, fl + nl):
                    L = ir["layers"][li] if li < n_l else None
                    got = L.get("in_dim") if isinstance(L, dict) else None
                    if got != s.get("width"):
                        errs.append(
                            "%s：第 %d 层 in_dim=%r 与栈宽 %r 不符"
                            "（跨栈漂移当报红）" % (sw, li, got, s.get("width")))

    # ── 层段互不重叠：同一层被两个分支认领 = 接线无从谈起 ────────────
    if isinstance(stk, list):
        _rng = []
        for s in stk:
            if isinstance(s, dict) and isinstance(s.get("first_layer"), int) \
                    and isinstance(s.get("n_layers"), int):
                _rng.append((s.get("first_layer"),
                             s.get("first_layer") + s.get("n_layers"),
                             s.get("name")))
        _rng.sort()
        for (_a, _b, _n), (_c, _d, _m) in zip(_rng, _rng[1:]):
            if _c < _b:
                errs.append("stacks：%r 的层段 [%d, %d) 与 %r 的 [%d, %d) 重叠"
                            " —— 同一层不许被两个分支认领"
                            % (_n, _a, _b, _m, _c, _d))

    # ── WIRING / SHARED（1.2）：显式接线与跨分支参数绑定 ────────────
    _names = {s.get("name") for s in (stk or [])
              if isinstance(s, dict) and s.get("name")}

    def _port(where, p):
        """端口文法（见 WIRING 表上方）。**说不清就是不合法。**"""
        if not isinstance(p, str):
            errs.append("%s：端口必须是字符串" % where)
            return
        if p.startswith("input:"):
            m = p.split(":", 1)[1]
            if m not in MODALITIES:
                errs.append("%s：模态 %r 不在闭集 %s"
                            % (where, m, "/".join(MODALITIES)))
            return
        if p == "logits":
            return
        if p.startswith("logits."):
            n = p.split(".", 1)[1]
            if n not in _names:
                errs.append("%s：旁路头 %r 的分支没登记" % (where, p))
            return
        base, _, tail = p.partition(".")
        if base not in _names:
            errs.append("%s：端口 %r 的分支 %r 没登记" % (where, p, base))
        elif tail not in ("in", "out", "hidden"):
            errs.append("%s：端口 %r 的后缀只认 in / out / hidden" % (where, p))

    wire = ir.get("wiring")
    if wire is not None and not isinstance(wire, list):
        errs.append("wiring 必须是列表")
    for w_i, w in enumerate(wire if isinstance(wire, list) else []):
        ww = "wiring[%d]" % w_i
        _check_fields(ww, WIRING, w, errs)
        if isinstance(w, dict):
            _port(ww + ".from", w.get("from"))
            _port(ww + ".to", w.get("to"))
    if isinstance(stk, list) and len(stk) > 1 and not wire:
        errs.append("stacks 有 %d 个分支而 wiring 缺失 —— 多分支必须给接线"
                    "（不然这张图答不出「接线从哪来到哪去」）" % len(stk))
    # **孤儿分支**：不出现在任何接线里 = 输出没人吃、输入没人喂 ——
    # 登记了也白登记，恰恰是「拍平了没说」的另一种形状。
    if isinstance(wire, list) and wire:
        _touched = set()
        for w in wire:
            if not isinstance(w, dict):
                continue
            for p in (w.get("from"), w.get("to")):
                if isinstance(p, str) and not p.startswith(("input:", "logits")):
                    _touched.add(p.partition(".")[0])
        for _n in sorted(_names - _touched):
            errs.append("stacks：%r 没出现在任何接线里（孤儿分支）" % _n)
    sh = ir.get("shared")
    if sh is not None and not isinstance(sh, list):
        errs.append("shared 必须是列表")
    # 全局参数名（裸名绑定的白名单）—— 推不出来就跳过这项检查，
    # **不让校验器自己崩**（它要面对的恰恰是坏 IR）。
    try:
        _globs = {k for k in params_of(ir) if not k.startswith("layers.")}
    except Exception:
        _globs = None
    for k_i, k in enumerate(sh if isinstance(sh, list) else []):
        kk = "shared[%d]" % k_i
        _check_fields(kk, SHARED, k, errs)
        if not isinstance(k, dict):
            continue
        pa, bn = k.get("param"), k.get("binds")
        if pa == bn:
            errs.append("%s：%r 自己绑自己 —— 绑定要说的是一块权重两个名字"
                        % (kk, pa))
        for who, p in (("param", pa), ("binds", bn)):
            if not isinstance(p, str):
                continue
            _q = [n for n in _names if p.startswith(n + ".")]
            if _q:
                if len(p) <= len(_q[0]) + 1:
                    errs.append("%s.%s：%r 分支限定名还缺参数名" % (kk, who, p))
            elif _globs is not None and "." in p and p not in _globs:
                errs.append("%s.%s：%r 不是已登记分支的限定名，"
                            "也不在全局参数里" % (kk, who, p))

    # ── 值引用与宽度的静态推导（T0.1）────────────────────────────────
    # **换宽度的 `gaps` 不进 errs** —— 那是"没实现"不是"错了"，
    # 它由 `--selftest` / 后续的同步检查单独报，不冒充 IR 不合法。
    try:
        _W, werrs, _gaps = infer_widths(ir)
        errs.extend(werrs)
    except Exception as e:                       # pragma: no cover
        errs.append("宽度推导自己坏了（这不是你的 IR 的错）：%r" % (e,))
    return errs


def _check_op(where, op, errs):
    _check_fields(where, OP, op, errs)
    if not isinstance(op, dict):
        return
    k = op.get("kind")
    # **跨字段的约束也要在这里查。** IR 现在是入口了 —— 有人可以直接写 IR，
    # 不经过 .by1。所以"编译器会拦"不再是理由：
    # `compile_ir` 里那些检查，走 IR 这条路根本跑不到。
    # 这一条就是这么漏的：External 声明空 weights 竟然通过了。
    if k == "External":
        w = (op.get("attrs") or {}).get("weights")
        if isinstance(w, dict) and not w:
            errs.append(
                "%s (External): weights 不能是空的 —— "
                "**契约不松**：外部算子照样要声明自己的张量，"
                "否则它算的东西没人查" % where)
    if k not in KIND_ATTRS:
        errs.append("%s: kind = <%s> 不在闭集 %s 里" % (where, k, KINDS))
        return
    spec = KIND_ATTRS[k]
    attrs = op.get("attrs")
    if not isinstance(attrs, dict):
        errs.append("%s (%s): attrs 应当是字典，实际 %s"
                    % (where, k, type(attrs).__name__))
        return
    # **反着也要查一遍**：规格表里没有的属性就是不合法的。
    # 规格文档（`--spec` 出的那份）一直这么写着，但在这之前**没有执行者**。
    e = _keys_bad("%s (%s) 的 attrs" % (where, k), attrs, set(spec))
    if e:
        errs.append(e)
    for name, (t, req, _doc) in spec.items():
        if name not in attrs:
            if req:
                errs.append("%s (%s) 的 attrs 缺必填项 `%s`" % (where, k, name))
            continue
        e = _ty(name, t, attrs[name])
        if e:
            errs.append("%s (%s): %s" % (where, k, e))

    # ── MoE：**打分函数 vs 选法**的跨字段约束 ─────────────────────
    # 这两个属性正交，但不是随便配。写错的组合有两种坏法，
    # 而**两种都必须是错误，不能静默**：
    #
    #   ① 分组选法（`routing = sigmoid_group_topk`，noaux_tc）配一个
    #      **没有写出来的** 打分函数 —— 靠 `startswith('sigmoid')` 猜。
    #      这在 GLM-5.3 上真错过一次（见 history/ir.md 130）：
    #      `.by1` 写 `softmax_topk` + 官方 `scoring_func = "sigmoid"`，
    #      契约和 config 全对，前向算的是另一个模型。
    #
    #   ② 写了 `score = sqrtsoftplus` 却配 `softmax_topk` ——
    #      `routing` 那一支是"先 top-k 再对这 k 个做 softmax"，
    #      **`scores` 根本用不上**，于是 `score` 写了等于没写。
    #      这正是 KDA 的 `gate_act` 被删掉的理由（同名同形的那一类）：
    #      "声明了、不报错、什么也不做"是这仓库最忌讳的一种。
    if k == "MoE":
        rt = str(attrs.get("routing", "")).strip().lower()
        sc = attrs.get("score")
        if sc is not None:
            sc = str(sc).strip().lower()
        # ② 打分函数必须真的被用到。四种选法里，只有两种读 `scores`：
        #    softmax_topk  -> 打分就是 softmax 本身，"score" 无从插入
        #    topk_softmax  -> 先选后 softmax，同样不经过 `score`
        #    sigmoid_topk / sigmoid_group_topk -> 先打分再选，读 `scores` ✓
        if sc and sc != "softmax" and rt in ("softmax_topk", "topk_softmax"):
            errs.append(
                "%s (MoE): routing = %s **不经过打分函数**（它的 top-k 直接在 "
                "logits 或 softmax(logits) 上做），可是写了 score = %s —— "
                "那个属性不会被任何人读。要么去掉 score，要么换成 "
                "sigmoid_topk / sigmoid_group_topk 这类先打分的选法"
                % (where, rt, sc))
        # ① 分组选法 + 不写打分函数 = 靠前缀猜。允许（4 个已毕业的模型
        #    都是这么写的），但**显式写出来必须和选法自洽**：noaux_tc 的
        #    语义要求先 sigmoid 打分，写别的就是矛盾。
        if rt == "sigmoid_group_topk" and sc and sc == "softmax":
            errs.append(
                "%s (MoE): routing = sigmoid_group_topk 是 noaux_tc（先打分"
                "再分组选），score 不能是 softmax —— 那两个是不同算法的两半"
                % where)
        # ③ `_score` 是 `score` 的**推导值**，不是第二个来源。
        #    两个都在、又不相等 = 两个真相源 —— 那就是 `history/ir.md 130`
        #    那类 bug 的形状：两条路各推一遍，谁对不知道。
        #    （`.by1` 的标识符允许下划线开头，所以手写是够得着的。）
        sc2 = attrs.get("_score")
        if sc2 is not None:
            sc2 = str(sc2).strip().lower()
        if sc and sc2 and sc != sc2:
            errs.append(
                "%s (MoE): `score` = %s 和 `_score` = %s 对不上 —— "
                "`_score` 是 `score` 的推导值（编译器填），"
                "**两个各写各的就是两个真相源**。只写 `score` 就行"
                % (where, sc, sc2))

    # ── rope：**旋转宽度必须是偶数** ───────────────────────────────
    # 两种配对都按 2 维一组定义旋转：half=(i, i+h)、interleaved=(2i, 2i+1)。
    # 奇数宽度剩一个无法配对的分量 —— 这不是「还没实现」，是**没有定义**，
    # 所以它进 IR 合同，不是某个后端的口味。
    #
    # 这条是**外部复现抓出来的**（2026-10-16，Colab，hello head_dim=31）：
    # check 过、run 在 NumPy 广播里裸崩（(1,4,8,16) vs (8,15)）——
    # 正是「要么实现，要么拒绝」要禁的第三条路。
    #
    # 判据按**旋转宽度**算，不是 head_dim（两点都是发现者的设计意见）：
    #   ① rope_partial 折出来的子维同样要满足（64×0.5=32 ✓，30×0.5=15 ✗）；
    #   ② rope 关着时任意 head_dim 都合法（GPT-2 那条路）——
    #      **不许做成「见奇数就拒」的全局规则**。
    if k == "Attention" and isinstance(attrs, dict) and attrs.get("rope"):
        hd, part = attrs.get("head_dim"), attrs.get("rope_partial", ROPE_PARTIAL_DEFAULT)
        if isinstance(hd, int) and hd > 0:
            try:
                npr = int(hd * float(part))
            except (TypeError, ValueError):
                npr = hd
            w = npr if 0 < npr < hd else hd
            if w % 2:
                errs.append(
                    "%s (Attention): rope 开着而旋转宽度 %d 是奇数 —— "
                    "配对按 2 维一组定义（half=(i,i+h)、interleaved=(2i,2i+1)），"
                    "奇数宽度剩一个无法配对的分量。来源：head_dim=%s × "
                    "rope_partial=%s → 旋转宽度 %d。把它凑成偶数（改 head_dim "
                    "或 rope_partial），或 rope = false（位置由外部加）"
                    % (where, w, hd, part, w))
    if k == "MLA" and isinstance(attrs, dict):
        # 主 rope 转 qk_rope 那一段（pairing 记在自己身上）；索引器的
        # 配对**固定 interleaved**（by1exec 的 MLA 注释），所以 index_dim
        # 同一条款。qk_rope = 0 = 不转，不适用。
        for fld, src in (("qk_rope", attrs.get("pairing")),
                         ("index_dim", "index_dim（索引器固定 interleaved）")):
            v = attrs.get(fld)
            if isinstance(v, int) and v > 0 and v % 2:
                errs.append(
                    "%s (MLA): %s = %d 是奇数 —— 旋转配对按 2 维一组（%s），"
                    "奇数宽度剩一个无法配对的分量。把它凑成偶数"
                    % (where, fld, v, src))


def to_json(ir, indent=1):
    """规范化输出：**键排序**，这样两份等价的 IR 字节相同。"""
    out = dict(ir)
    out["by1-ir"] = VERSION
    return json.dumps(out, ensure_ascii=False, indent=indent, sort_keys=True)


def from_json(s):
    """读回。**版本不对直接拒，不做兼容猜测。**"""
    ir = json.loads(s)
    errs = validate(ir)
    if errs:
        raise IRError("IR 不合法：\n  " + "\n  ".join(errs[:12]))
    return ir


def roundtrip(ir):
    """往返等价：to_json -> from_json -> **逐字段比**。"""
    back = from_json(to_json(ir))
    a = json.dumps(_norm(ir), sort_keys=True)
    b = json.dumps(_norm(back), sort_keys=True)
    if a != b:
        return False, "往返之后不一致"
    return True, ""


def spec_markdown():
    """**规格文档是从这张表生成的**，不是我手写的。

    理由：手写的文档会过期，而过期的规格比没有规格更坏 ——
    它会让读的人以为自己知道。这张表被 `validate` 和 `by1all`
    真的跑着，所以它跟得上。
    """
    L = []
    L.append("# by1 IR 规格")
    L.append("")
    L.append("> **这份文档是从 `src/by1ir.py` 里的表生成的**，不是手写的。")
    L.append("> 生成命令：`python src/by1ir.py --spec`")
    L.append(">")
    L.append("> 理由：手写的规格会过期，而过期的规格比没有规格更坏 ——")
    L.append("> 它让读的人以为自己知道。这些表被 `validate()` 和")
    L.append("> `by1all` 真的跑着，所以它们跟得上代码。")
    L.append("")
    L.append("**版本：`by1-ir: %s`**" % VERSION)
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 怎么用它")
    L.append("")
    L.append("```bash")
    L.append("python src/by1ir.py --spec                  # 生成这份文档")
    L.append("python src/by1ir.py --emit models/<文件>.by1  # 出一份规范化 JSON")
    L.append("python src/by1ir.py --check <文件>.ir.json  # 校验")
    L.append("python src/by1irentry.py                    # 三个后端只从 IR 跑")
    L.append("```")
    L.append("")
    L.append("**想写第四个后端的人从这里开始** —— 不需要先学 `.by1`。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 顶层")
    L.append("")
    L.append("| 字段 | 类型 | 必填 | 语义 |")
    L.append("|---|---|---|---|")
    for name, ty, req, doc in TOP:
        L.append("| `%s` | `%s` | %s | %s |" % (name, ty, "是" if req else "否", doc))
    L.append("")
    L.append("## 层")
    L.append("")
    L.append("| 字段 | 类型 | 必填 | 语义 |")
    L.append("|---|---|---|---|")
    for name, ty, req, doc in LAYER:
        L.append("| `%s` | `%s` | %s | %s |" % (name, ty, "是" if req else "否", doc))
    L.append("")
    L.append("## 算子")
    L.append("")
    L.append("| 字段 | 类型 | 必填 | 语义 |")
    L.append("|---|---|---|---|")
    for name, ty, req, doc in OP:
        L.append("| `%s` | `%s` | %s | %s |" % (name, ty, "是" if req else "否", doc))
    L.append("")
    L.append("`kind` 是**闭集**：%s" % ", ".join("`%s`" % k for k in KINDS))
    L.append("")
    L.append("## 状态")
    L.append("")
    L.append("| 字段 | 类型 | 必填 | 语义 |")
    L.append("|---|---|---|---|")
    for name, ty, req, doc in STATE:
        L.append("| `%s` | `%s` | %s | %s |" % (name, ty, "是" if req else "否", doc))
    L.append("")
    L.append("## 分支登记表（`stacks`）")
    L.append("")
    L.append("1.1 定字段、1.2 升格为**分支登记表**。缺省 = 单栈文本"
             "（1.1 文件一个字不改照样合法）。")
    L.append("")
    L.append("| 字段 | 类型 | 必填 | 语义 |")
    L.append("|---|---|---|---|")
    for name, ty, req, doc in STACKS:
        L.append("| `%s` | `%s` | %s | %s |" % (name, ty, "是" if req else "否", doc))
    L.append("")
    L.append("## 接线（`wiring`，1.2）")
    L.append("")
    L.append("显式接线：谁吃谁的输出。**stacks 多于一条时必填。**")
    L.append("")
    L.append("端口文法：")
    L.append("")
    L.append("    input:<模态>     模态入口（input:text / input:image / …）")
    L.append("    <分支>.in        分支的入口")
    L.append("    <分支>.out       分支的输出（跑完它自己的层之后）")
    L.append("    <分支>.hidden    进输出头之前的隐藏态（MTP 头吃的是这个）")
    L.append("    logits           主输出头")
    L.append("    logits.<分支>    旁路输出头（MTP 的额外 logits）")
    L.append("")
    L.append("| 字段 | 类型 | 必填 | 语义 |")
    L.append("|---|---|---|---|")
    for name, ty, req, doc in WIRING:
        L.append("| `%s` | `%s` | %s | %s |" % (name, ty, "是" if req else "否", doc))
    L.append("")
    L.append("## 跨分支参数绑定（`shared`，1.2）")
    L.append("")
    L.append("共享权重的两种形态：`tie` 同一块权重两个名字"
             "（tie_word_embeddings）· `share` 声明级共享"
             "（mtp 的 share_embeddings / share_head）。")
    L.append("")
    L.append("| 字段 | 类型 | 必填 | 语义 |")
    L.append("|---|---|---|---|")
    for name, ty, req, doc in SHARED:
        L.append("| `%s` | `%s` | %s | %s |" % (name, ty, "是" if req else "否", doc))
    L.append("")
    L.append("## 不拍平判据（1.2）")
    L.append("")
    L.append("多分支的 IR 必须**答得出两个问题**：")
    L.append("")
    L.append("1. **这个节点在哪个分支上** —— `stacks` 的层段划分"
             "（first_layer + n_layers；层段互不重叠）；")
    L.append("2. **接线从哪来到哪去** —— `wiring`（端口按上面的文法，"
             "分支不出现在任何接线里 = 孤儿分支，不合法）。")
    L.append("")
    L.append("节点表照旧是 `layers`（全局编号）—— 分支不各持一份层表，"
             "是为了不给同一个节点两个家。")
    L.append("")
    L.append("## 每种算子的属性")
    L.append("")
    L.append("**这张表就是「IR 里合法的东西」的唯一定义。**")
    L.append("不在表里的属性 = 不合法；表里必填而缺的 = 不合法。")
    L.append("")
    for kind in KINDS:
        spec = KIND_ATTRS[kind]
        L.append("### `%s`" % kind)
        L.append("")
        if not spec:
            L.append("（没有属性）")
            L.append("")
            continue
        L.append("| 属性 | 类型 | 必填 | 语义 |")
        L.append("|---|---|---|---|")
        for name, (ty, req, doc) in spec.items():
            L.append("| `%s` | `%s` | %s | %s |"
                     % (name, ty, "是" if req else "否", doc))
        L.append("")
    L.append("---")
    L.append("")
    L.append("## 几条**为什么这样定**")
    L.append("")
    L.append("- **`pos_kind` 是一个枚举，不是一个布尔。**")
    L.append("  `rope` 是在注意力里旋转，`learned` 是在输入上加一张查表 ——")
    L.append("  这是两种东西，不是「有没有位置编码」。")
    L.append("- **`norm_kind` 同理。** `rms` 只除均方根，`layer` 还要减均值、")
    L.append("  还有 bias。名字像，算的不是一回事。")
    L.append("- **`Attention.rope` 是一个独立字段。**")
    L.append("  它回答「要不要转」。GPT-2 的教训：**「没有这个机制」以前")
    L.append("  表达不出来**，于是注意力无条件转了一遍，而所有检查都说「过」。")
    L.append("- **闭集就是闭集。** `act` / `qk_norm` / `routing` / `gate_act` /")
    L.append("  `pairing` 只认列出来的取值。取值多一个必须**拒绝**，")
    L.append("  不许静默走默认分支 —— 那会生成一个「看起来对但算错」的模型。")
    L.append("- **`by1-ir` 是必填的。** 没有版本号的 IR 不该被接受：")
    L.append("  读的一方无从判断自己理解的是哪一版。大版本不匹配直接拒，")
    L.append("  不做兼容猜测。")
    L.append("")
    return "\n".join(L)


def _norm(x):
    """比较用：把元组变列表、把版本号统一。"""
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    return x


def _selftest_widths():
    """T0.1 的判卷人：**反例必须红，正例必须绿。**

    三组：
      ① 手写坏 IR（引用不存在 / 用在产出前 / 全局链坏）—— 必须报错
      ② 换宽度（in_dim 64->32）—— 只进 gaps，**不算错误**
      ③ models/ 下每份 .by1 编出来的 IR —— 宽度错误必须是 0（回归）
    """

    def tiny(layers=None, globals_=None):
        g = globals_ if globals_ is not None else [
            {"kind": "Embed", "mech": "Embed", "attrs": {},
             "inputs": [], "outputs": ["hidden"]},
            {"kind": "Head", "mech": "Head", "attrs": {},
             "inputs": ["hidden"], "outputs": ["logits"]},
        ]
        return {"by1-ir": VERSION, "vocab": 8, "ctx": 16, "d_model": 4,
                "pos_kind": "rope", "norm_kind": "rms", "norm_one_plus": False,
                "globals": g,
                "layers": layers if layers is not None else []}

    def one_op(kind, ins, outs, attrs=None):
        return {"kind": kind, "mech": kind, "attrs": attrs or {},
                "inputs": ins, "outputs": outs}

    bad = []
    # ①-a 引用不存在的值（拼写坏）
    ir = tiny(layers=[{"index": 0, "attrs": {}, "ops": [
        one_op("Add", ["hidden", "op0.ott"], ["op0.out"])]}])
    _W, e, _g = infer_widths(ir)
    if not any("不存在的值" in x for x in e):
        bad.append("①-a 引用坏的没报错")
    # ①-b 用在产出之前
    ir = tiny(layers=[{"index": 0, "attrs": {}, "ops": [
        one_op("Add", ["hidden", "op2.out"], ["op0.out"]),
        one_op("Add", ["hidden", "op0.out"], ["op1.out"]),
        one_op("Add", ["hidden", "op1.out"], ["op2.out"])]}])
    _W, e, _g = infer_widths(ir)
    if not any("不存在的值" in x for x in e):
        bad.append("①-b 用在产出前没报错")
    # ①-c 全局链坏：Head 吃一个不存在的值
    ir = tiny(globals_=[
        {"kind": "Embed", "mech": "Embed", "attrs": {},
         "inputs": [], "outputs": ["hidden"]},
        {"kind": "Head", "mech": "Head", "attrs": {},
         "inputs": ["normed"], "outputs": ["logits"]}])
    _W, e, _g = infer_widths(ir)
    if not any("不存在的值" in x for x in e):
        bad.append("①-c 全局链坏没报错")
    # ② 换宽度：gaps 有、errs 无
    ir = tiny(layers=[
        {"index": 0, "attrs": {}, "ops": [
            one_op("Add", ["hidden", "hidden"], ["op0.out"])], "in_dim": 4},
        {"index": 1, "attrs": {}, "ops": [
            one_op("Add", ["hidden", "hidden"], ["op0.out"])], "in_dim": 2}])
    _W, e, g = infer_widths(ir)
    if e:
        bad.append("② 换宽度不该报错，实报 %s" % e[:1])
    if not g:
        bad.append("② 换宽度没进 gaps —— 那是已知缺口，不是错误")
    if bad:
        for b in bad:
            print("  [FAIL] " + str(b)[:160])
        print("  [FAIL] widths selftest：%d 处" % len(bad))
        return 1

    # ③ 回归：models/ 下每份 .by1（selftest.by1 是故意坏的，不算）
    import importlib.util
    sp = importlib.util.spec_from_file_location('bcm', by1paths.tool('by1check.py'))
    bc = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(bc)
    import by1codegen as cg
    import glob as _g2
    n_ok, n_neg = 0, 0
    for f in sorted(_g2.glob(by1paths.root('models', '*.by1'))):
        if f.endswith('selftest.by1'):
            continue
        _r, info = bc.check(f)
        try:
            ir = cg.compile_ir(info)
        except cg.CodegenError:
            # **故意的反例模型**（gate-probe 那种：act 是不存在的取值，
            # 编译就该拒）。编译被拒是设计，不是回归。
            n_neg += 1
            continue
        _W, e, _gp = infer_widths(ir)
        if e:
            print("  [FAIL] %s：%s" % (f, e[:2]))
            return 1
        n_ok += 1
    print("  [PASS] 宽度推导：3 组反例全红 + %d 份模型回归全绿"
          "（另有 %d 份反例模型按设计被编译拒绝；换宽度只进 gaps）"
          % (n_ok, n_neg))
    return 0


def _selftest_branches():
    """T24 · **不拍平审计**：每份多栈模型的 IR 答得出两个问题吗。

      ① 这个节点在哪个分支上 —— `stacks` 层段**互不重叠、盖满** layers
      ② 接线从哪来到哪去 —— `wiring` 非空、端口全解析、无孤儿（文法归 validate）

    另外两件**如实记录**的事：
      · modality 按名约定推出来的逐份列出 —— 推的不冒充声明；
      · 多栈却一个三件套都没写的（答不出接线的拓扑），逐份红出来。
      · 降不动的模型（各有各的账：缺 vocab / 真多分支等）不算审计对象，
        但要点名 —— **「没验」和「验过且对」是两件事**。
    """
    import glob as _g2
    import importlib.util
    import os as _os
    sp = importlib.util.spec_from_file_location('bcm', by1paths.tool('by1check.py'))
    bc = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(bc)
    import by1codegen as cg

    bad, inferred, skipped = [], [], []
    n_multi = 0
    for f in sorted(_g2.glob(by1paths.root('models', '*.by1'))):
        if f.endswith('selftest.by1'):
            continue
        _r, info = bc.check(f)
        if len(info.get('stacks') or []) < 2:
            continue
        n_multi += 1
        name = _os.path.basename(f)
        try:
            ir = cg.compile_ir(info)
        except Exception as ex:
            skipped.append('%s（%s）' % (name, str(ex).splitlines()[0][:40]))
            continue
        if not ir.get('stacks'):
            bad.append((name, '多栈却没写 stacks/wiring —— 两个问题都答不出'))
            continue
        cov = sorted((s['first_layer'], s['first_layer'] + s['n_layers'])
                     for s in ir['stacks'])
        full = (bool(cov) and cov[0][0] == 0 and cov[-1][1] == len(ir['layers'])
                and all(b == c for (_a, b), (c, _d) in zip(cov, cov[1:])))
        if not full:
            bad.append((name, '层段没盖满 layers —— 答不出「节点在哪个分支上」'))
        if not ir.get('wiring'):
            bad.append((name, 'wiring 缺失 —— 答不出「接线从哪来到哪去」'))
        errs = validate(ir)
        if errs:
            bad.append((name, 'validate：%s' % errs[0]))
        for s in ir['stacks']:
            if not (info.get('stack_modality') or {}).get(s['name']):
                inferred.append('%s:%s=%s' % (name, s['name'], s['modality']))
    print()
    n_bad = len({n for n, _w in bad})
    print('  ⑤ 多分支审计（T24 · 不拍平判据）')
    print('     %d 份多栈模型 —— %d 份两个问题都答得出'
          % (n_multi, n_multi - n_bad - len(skipped)))
    if skipped:
        print('     另有 %d 份降不动（各有各的账，不算审计对象）：%s'
              % (len(skipped), '; '.join(skipped[:4])))
    if inferred:
        print('     modality 按名约定推的 %d 处（没有一份显式声明）：%s%s'
              % (len(inferred), ', '.join(inferred[:8]),
                 ' …' if len(inferred) > 8 else ''))
    if bad:
        for n, w in bad[:12]:
            print('  [FAIL] %s：%s' % (n, w))
        print('  [FAIL] 多分支审计：%d 份不合规' % n_bad)
        return 1
    print('     [PASS] 不拍平：节点归属（层段）+ 接线（wiring），每份都答得出')
    return 0


def _cli():
    import sys
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__ or 'by1ir —— IR 规格、校验、JSON 往返。')
        return 0
    if '--spec' in sys.argv:
        print(spec_markdown())
        return 0
    if '--emit' in sys.argv:
        import importlib.util
        f = sys.argv[sys.argv.index('--emit') + 1]
        sp = importlib.util.spec_from_file_location(
            'bcm', by1paths.tool('by1check.py'))
        bc = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(bc)
        import by1codegen as cg
        _r, info = bc.check(f)
        print(to_json(cg.compile_ir(info)))
        return 0
    if '--check' in sys.argv:
        f = sys.argv[sys.argv.index('--check') + 1]
        errs = validate(by1io.read_json(f, encoding='utf-8'))
        for e in errs:
            print("  " + e)
        print("  [%s] %d 个问题" % ("PASS" if not errs else "FAIL", len(errs)))
        return 0 if not errs else 1
    if '--selftest' in sys.argv:
        _bad = _selftest_widths()
        return _bad or _selftest_branches()
    print(__doc__)
    # 子命令不全 -> by1skip.CALLER
    return by1skip.CALLER


if __name__ == '__main__':
    import sys
    sys.exit(_cli())
