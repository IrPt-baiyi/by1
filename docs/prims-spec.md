# by1 原语数值规格（生成物）

> **这份文档是生成的**：`python src/by1graph.py --spec`（`--write` 落盘）。
> 改措辞改 `src/by1graph.py` 的 `PRIMS` / `_SPEC`；键集对账是 import 时的
> 硬断言，缺一个条目整个库都跑不起来 —— 所以它不可能比原语表少。
>
> 每条的**公式**与**状态更新**在 `f`；`ref()` 是可执行规格
> （`by1graph.py --selftest` 对手算逐条盯着）。

| 原语 | 入 | 公式（f） | 形状 | dtype | 边界 / 不变量 | 出处（搬家前的旧家） |
|---|---|---|---|---|---|---|
| add | 2 | a + b（广播） | out = broadcast(ins0, ins1) | 浮点逐元，两入同 dtype | 广播规则按 numpy；不接受 dtype 提升 | by1exec.op_add / by1c LAYER_BODY |
| mul | 2 | a * b（广播） | out = broadcast(ins0, ins1) | 浮点逐元，两入同 dtype | 同 add | by1exec / by1c 的门控乘法 |
| scale | 1 | a * s | out = ins0 | 浮点逐元 | s 为标量属性 | gelu_tanh 公式的 0.5 与系数 |
| matmul | 2 | out[...,i,j] = Σ_k x[...,i,k]·w[j,k] = x @ w.T | x[..,i,k] · w[j,k]ᵀ -> out[..,i,j]；w 恒为 2-D [out,in] | 浮点；乘加语义按 ref 的 float32 累加 | k 维必须相等（不等是形状错，不是数值差） | by1c.linear / ggml_mul_mat / torch F.linear |
| batch_matmul | 2 | out[β,j,i] = Σ_k a[β//r,i,k]·b[β,j,k]，r = B/b_a（GQA 广播） | a[β//r,i,k]·b[β,j,k] -> out[β,i,j]，r = B/b_a | 浮点；同 matmul | r 必须整除；r>1 即 GQA 广播 | by1exec.op_attention 的 q@kᵀ 与 att@v |
| rms_norm | 2 | x · rsqrt(mean(x²)+eps) · (one_plus?1+w:w) | w = [d]；out = ins0 | 浮点逐元 | eps > 0；mean 沿最后一轴；one_plus 二选一见 f | by1c.rmsnorm / ggml_rms_norm / by1exec.rms_norm |
| layer_norm | 3 | (x-μ)/sqrt(var+eps) · (one_plus?1+w:w) + b | w = b = [d]；out = ins0 | 浮点逐元 | eps > 0；μ/var 沿最后一轴 | by1c.layernorm / ggml_norm / by1exec |
| silu | 1 | x · σ(x) | out = ins0 | 浮点逐元 | 无前提 | 四后端各一份 —— 收敛于此 |
| gelu_tanh | 1 | 0.5·x·(1+tanh(√(2/π)·(x+0.044715x³))) | out = ins0 | 浮点逐元 | 近似式系数是规格的一部分（0.044715, √(2/π)） | by1c.gelu_tanh_ / by1exec（ggml 那边要拼原语） |
| gelu | 1 | 0.5·x·(1+erf(x/√2)) | out = ins0 | 浮点逐元 | erf 精确式 | by1c.gelu_erf_ / ggml_gelu_erf |
| sigmoid | 1 | σ(x)（= `moe_scores_array('sigmoid')`） | out = ins0 | 浮点逐元 | 无前提 | by1ir.moe_scores_array / ACT2FN |
| softmax | 1 | exp(x-max)/Σ，沿最后轴 | out = ins0（沿最后轴归一） | 浮点逐元 | 减 max 保稳；Σ 规范化 | by1ir.moe_scores_array / ACT2FN |
| softplus | 1 | log(1+eˣ)（数值稳定：max(x,0)+log1p(e^-\|x\|)） | out = ins0 | 浮点逐元 | 数值稳定式（max(x,0)+log1p(e^-\|x\|)）是规格的一部分 | by1ir.moe_scores_array / ACT2FN |
| sqrtsoftplus | 1 | √softplus(x)（DeepSeek V4 的打分函数） | out = ins0 | 浮点逐元 | √(softplus(x))；softplus 稳定式同上 | by1ir.moe_scores_array / ACT2FN |
| top_k | 1 | 每行前 k 大的**下标**（并列取小） | out = ins0[.., :k] 的**下标** | 出 int 下标 | 1 ≤ k ≤ 尾维；**并列取小下标**（三后端同一句） | by1exec.op_moe._topk / ggml_top_k |
| gather | 2 | 沿最后轴按 idx 取值（take_along_axis） | 沿最后轴对齐 idx 形状 | 出同 ins0 的值 | idx ∈ [0, 尾维) | by1exec 的 np.take_along_axis |
| sum | 1 | 沿 ggml 轴 axis 求和（keepdims 可关） | 沿 ggml 轴 axis 归约；keepdims 决定保不保轴 | 浮点；累加语义按 ref | axis 合法；keepdims 默认开 | ggml_sum_rows / numpy .sum |
| clamp_min | 1 | max(x, m) | out = ins0 | 浮点逐元 | m 为标量属性 | by1exec 的 np.maximum(分母, eps) |
| clamp_max | 1 | min(x, m) | out = ins0 | 浮点逐元 | m 为标量属性 | by1exec.op_moe 的 np.minimum(g, limit) |
| add_scalar | 1 | a + c | out = ins0 | 浮点逐元 | c 为标量属性 | by1exec.op_moe 的归一守卫 `sum + 1e-20` |
| one_hot | 2 | out[...,j] = 1 若 j ∈ ids 否则 0（宽度 = 参照的最后一维） | 宽度 = ins1 的最后一维；ids 形状在前 | 出 float（1.0 / 0.0） | ids ∈ [0, 宽度) | by1exec.op_moe 的 put_along_axis(gm, gi, 1.0) |
| masked_fill | 2 | out = m > 0 ? x : v | out = ins0（m 广播对齐） | 值 v 的 dtype | m > 0 取 x，否则取 v —— **边界是 0 不是 0.5** | by1exec.op_moe 的 np.where(mask, choice, -inf) |
| le | 2 | out = 1 若 a <= b 否则 0（广播二元） | out = broadcast(ins0, ins1) | 出 float（1.0 / 0.0） | <= 是含等号的（因果比较靠它） | by1exec._dsa_index 的因果比较 ar[None,:] <= ar[:,None] |
| div | 2 | a / b（广播） | out = broadcast(ins0, ins1) | 浮点逐元 | b ≠ 0；用途里配 clamp_min 护栏 | ggml_div / numpy |
| concat | 2 | 沿 ggml 轴 axis 接两条 | 沿 ggml 轴 axis 接；其余轴必须相等 | 浮点；两入同 dtype | 轴不等是形状错 | torch.cat / ggml_concat / by1exec 的 np.concatenate |
| conv1d_causal | 2 | 深度可分离因果卷积（逐通道一份核，左边补 kernel-1）；**激活不在这里** | 核 [c,k]；out[..,t,c]，左补 k-1 后取有效段 | 浮点；逐通道一份核 | **激活不在这里**（下游原语的事） | by1exec.op_linear 的短卷积 / causal_conv1d |
| delta_rule | 5 | 逐 token：st*=exp(decay)；δ=(v−(st·k))·beta；st+=k⊗δ；out=(st·q) | q/k/v [t,c]；beta/decay [t,c] 或 [t,1]；out [t,c] | 浮点；状态 st 在 float 语义下逐 token 更新 | 状态更新见 f 的逐 token 式；零初始 | by1exec.op_linear / op_kda 的 for |
| ssm_scan | 6 | 逐 token：st=exp(d·A)·st + d·(x⊗B)；y=st·C + D·x | d [t,c]；A [c]；x [t,c]；B/C [t,c]；D [c]；out [t,c] | 浮点；状态 st 逐 token 更新 | 状态更新见 f 的递推式；零初始 | by1exec.op_ssm 的递推 for（Mamba2） |
| exp | 1 | eˣ | out = ins0 | 浮点逐元 | 无前提（溢出即真溢出） | by1exec 的 np.exp（衰减门要用） |
| log1p | 1 | log(1+x) | out = ins0 | 浮点逐元 | x > -1 | by1exec.op_linear 的 np.log1p（衰减门要用） |
| sqrt | 1 | √x | out = ins0 | 浮点逐元 | x ≥ 0 | by1exec 的 1/sqrt(·)（L2 归一 / 输出归一） |
| l2_norm | 1 | x / √(Σx²+eps)，沿最后轴（**sum 不是 mean**） | out = ins0（沿最后轴） | 浮点逐元 | eps > 0；**sum 不是 mean**（均值那支是现拼的） | by1exec.op_linear 的 q/k 归一 / torch l2norm |
| repeat | 1 | 沿 ggml 轴 axis 整块重复 reps 次 | 沿 ggml 轴 axis 整块重复 reps 次 | 同 ins0 | reps ≥ 1；axis 合法 | np.repeat / by1exec.op_linear 的 GQA 广播 |
| rope | 2 | 每对维度转 θ_t = t·inv[i]（inv 见 rope_inv：普通 / YaRN / llama3）；half 配对 (i,i+h)，interleaved 配对 (2i,2i+1)且**拼接输出** | x [n, nh, hd]（exec 视角转置进出）；inv 长度 hd/2 | 浮点逐元 | n_dims ≤ hd；**n_dims 为偶数**（配对按 2 维一组，奇数宽度没有定义 —— 2026-10-16 外部复现的 head_dim=31）；pairing ∈ {half, interleaved}；half=(i,i+h)、interleaved=(2i,2i+1) 且拼接输出 | by1c.rope / by1exec.apply_rope / ggml_rope_ext |
| soft_max | 2 | softmax(a·scale + mask)，沿 key 轴 | a [头,q,k] -> out 同形；mask [q,k] | 浮点逐元 | scale > 0；掩码规则 j<=i 且 j>i-window（window 取自 mask 名） | by1c.attention / ggml_soft_max_ext |
| soft_max_sink | 3 | softmax(a·scale + mask)，但尾巴上多一个 sink 槽 | 同 soft_max；sink 为逐头标量（多一槽） | 浮点逐元 | 同 soft_max；sink 槽参与归一 | by1exec.attention 的 concat+softmax / by1c 的 sink 分支 |
| get_rows | 2 | out[i] = table[ids[i]] | out = ids 形状 + table 行宽 | 出同 table | ids ∈ [0, 行数) | ggml_get_rows / torch nn.Embedding |
| view | 1 | 重解释形状（元素序不变） | 总数不变、元素序不变 | 同 ins0 | 总数必须相等（不等是形状错） | reshape / ggml_reshape |
| permute | 1 | 轴置换（带步长的视图） | 轴置换（带步长的视图） | 同 ins0 | axes 必须是 0..ndim-1 的排列 | transpose / ggml_permute |
| cont | 1 | 按当前视图物化为连续内存 | 物化为连续；带 shape 时同时重解释 | 同 ins0 | 无前提 | ggml_cont / numpy .copy() |
| slice | 1 | 沿 ggml 轴 axis 取 [start, stop) | 沿 ggml 轴 axis 取 [start, stop) | 同 ins0 | 0 ≤ start < stop ≤ 轴长 | ggml_view_* 的 offset / numpy 切片 |

共 41 个原语（闭集）。
布局原语（view/permute/cont）的语义是**数据重解释**，不是数学。
还没收编的机制见 `UNLOWERED`（Raw / External 逃生舱按设计不收编）。
