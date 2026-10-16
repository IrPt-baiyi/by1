#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1shadow -- **影子 checkpoint：真格式、随机值，数值对拍不需要下载权重。**

## 它解决什么

GPU 对拍（`checks/by1gpubench`）要 `snapshot_download` 整个 checkpoint ——
几十 GB。租来的卡（AutoDL 之类）出口带宽就那么点，提速手段（学术加速、
ModelScope、hf_transfer、aria2、文件存储）全用上也还是慢，**因为慢的
是带宽本身，不是下载方式**。于是只剩一条路：把要下的字节数变成零。

## 为什么随机值就够了

对拍真正要比的是两个 loader 的**接线**：

    张量映射完整性 · tying / 共享 · config 声明与真品一致 ·
    前向接线（reshape/transpose/RoPE 拼接）· 贪心生成 token 一致

这些**和权重值无关**。拿随机值权重，接线错了一样对不上。所以影子
checkpoint 的配方是：

    张量名/形状/dtype  = 真的（模型类 state_dict，主干命名即官方命名）
    config.json        = 真的（refs/*.config.json，或调用方给的真品 config）
    权重值             = 按张量名哈希确定性生成的随机数

**config.json 必须用真品，不能用声明合成。** `by1full.load_model` 里的
`_check_config`（声明 vs checkpoint）是对拍的半个主体；config 从声明生成
就变成自己比自己，**声明写错也全绿**。真 config + 随机值：声明错一个
字段，`_check_config` 或 logits 对拍当场抓到。

## 它验什么、不验什么（说清楚，别让"绿"超过"验过的"）

验：张量映射、tying、声明↔config 一致、前向接线、贪心生成 token 一致。
**不验：**

- 真权重的数值语义（这本来就不是对拍的目标）
- **量化存储**（MXFP4/FP8 的打包格式没法"随机"——随机字节不是合法量化
  块。量化模型走真权重，影子模式显式跳过而不是假装验过）
- checkpoint 命名与官方发布物的一致性 —— 名字取自 by1 的完整模型类，
  主干是官方类自带的命名，`mtp.*` 的命名靠"真权重对拍今天能过"兜着。
  这条缺口真权重对拍补。

**"藏起来的缺口和没发现过的缺口一样糟"** —— 所以上面这段不许删。

## 怎么生成

在 meta device 上实例化完整模型类（不占内存），拿到全部张量名/形状/dtype；
共享（tied）参数按 `id()` 归组、组内每个名字都写、值相同 —— 和
`save_pretrained` 的去重语义兼容，两个 loader 都不会报 missing/unexpected。
值按名字哈希播种，**重跑逐字节可复现**；初始化缩放 `initializer_range /
sqrt(2*n_layer)`（GPT-2 的 scaled init），保证深模型 bf16 前向数值有限。
分片流式写，峰值内存 ≈ 一个分片。

用法:
    python src/modelcheck/by1shadow.py --by1 models/complete-gpt2.by1 \\
        --config refs/openai-community__gpt2.config.json --out shadow-gpt2
    python src/modelcheck/by1shadow.py --smoke     # CPU 端到端自检（gpt2）
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
import glob
import hashlib
import itertools
import json
import math
import os
import pathlib

import by1skip

#: 默认种子。固定值是为了"重跑逐字节复现"；换种子 = 换一套随机权重。
SEED = 20261013

#: 单分片上限（字节）。和 HF 发布物同量级，让 index/shard 布局也像真的。
MAX_SHARD_BYTES = 8 * 1024 ** 3

#: 影子产物里的旁注文件名。**不是**给 from_pretrained 看的（它只认
#: config.json / model.safetensors*），是给人看的"这堆权重是什么"。
MANIFEST = 'shadow-manifest.json'

#: 这份产物**验过什么**。写进 manifest，也印在报告里。
VERIFIED = [
    '张量映射完整（missing/unexpected = 0）',
    'tying / 共享参数（含 mtp.embed_tokens / mtp.lm_head）',
    '声明 ↔ checkpoint config 逐字段一致（config 为真品时）',
    '前向接线：logits 数值一致',
    '贪心生成 token 一致',
]

#: 这份产物**没验什么**。同一份清单，永远一起出现。
NOT_VERIFIED = [
    '真权重的数值语义（对拍本来就不比这个）',
    '量化存储格式（MXFP4/FP8 打包块无法随机合成，量化模型走真权重）',
    'checkpoint 命名与官方发布物的一致性（名字取自模型类；mtp.* 由真权重对拍兜底）',
]


class ShadowSkip(Exception):
    """影子模式下做不了，但不是失败 —— 调用方转成 by1skip 的"跳过"。"""


def _seed_for(seed, name):
    """张量名 -> 生成种子。**名字哈希，和遍历顺序无关**，所以可复现。"""
    digest = hashlib.sha256(('%d:%s' % (seed, name)).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % (2 ** 63 - 1)


def _target_dtype(name, dtype):
    """生成值的落地 dtype。整型/布尔 buffer 不是"值"，保持原样。"""
    return dtype


def sample(group, seed, std, torch):
    """一个共享组的随机值。判据：确定性、数值温和、语义像初始化。"""
    canonical = min(group['names'])
    shape = group['shape']
    dtype = group['dtype']
    if not dtype.is_floating_point:
        # 位置 id 之类的整型 buffer：零是无害值，也谈不上"随机"。
        return torch.zeros(shape, dtype=dtype)
    generator = torch.Generator().manual_seed(_seed_for(seed, canonical))
    if canonical.endswith('.bias'):
        values = torch.zeros(shape)
    elif len(shape) == 1 and 'norm' in canonical:
        # 归一化权重取 1 —— 随机值会让激活幅度乱飘，没必要。
        values = torch.ones(shape)
    else:
        values = torch.randn(shape, generator=generator) * std
    return values.to(dtype)


def tensor_groups(model):
    """模型的全部持久张量，按共享（tied）归组。

    `named_parameters(remove_duplicate=False)` 对共享参数**逐次返回同一个
    对象**，所以按 `id()` 归组就是 tying 图本身。`state_dict()` 不行：
    它内部 `param.detach()` 每个名字造一个新对象，`id()` 对不上 ——
    那会把 tied 对当成两个独立张量，**tying 对拍就废了**。
    """
    state_keys = set(model.state_dict().keys())
    groups, order = {}, []
    pairs = itertools.chain(model.named_parameters(remove_duplicate=False),
                            model.named_buffers(remove_duplicate=False))
    for name, tensor in pairs:
        if name not in state_keys:
            continue
        group = groups.get(id(tensor))
        if group is None:
            group = {'shape': tuple(tensor.shape), 'dtype': tensor.dtype,
                     'names': []}
            groups[id(tensor)] = group
            order.append(group)
        group['names'].append(name)
    covered = {n for g in order for n in g['names']}
    if covered != state_keys:
        missing, extra = sorted(state_keys - covered), sorted(covered - state_keys)
        raise RuntimeError('影子张量清单对不齐：缺 %s 多 %s' % (missing[:5], extra[:5]))
    return order


def _shards(groups, max_shard_bytes):
    """按字节上限装分片。一组的全部名字进同一分片（值是一份）。"""
    out, current, used = [], [], 0
    for group in groups:
        size = math.prod(group['shape']) * group['dtype'].itemsize
        if current and used + size > max_shard_bytes:
            out.append(current)
            current, used = [], 0
        current.append(group)
        used += size
    if current:
        out.append(current)
    return out


def build(config_text, family, mtp_spec, out_dir, seed=SEED, dtype='bfloat16',
          max_shard_bytes=MAX_SHARD_BYTES, verbose=True):
    """造一个影子 checkpoint 目录，返回 manifest。

    `config_text` 必须是**真品 config.json 的原文**（见模块头：声明合成的
    config 会让声明对拍空转）。目录里写的就是这份原文，一字不改。
    """
    import torch
    import transformers as T
    import by1full
    from safetensors.torch import save_file

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(config_text, encoding='utf-8')
    config = T.AutoConfig.from_pretrained(str(out), local_files_only=True)
    if config.model_type != family:
        raise ValueError('config 的模型家族 %s 与 by1 声明的 %s 不一致'
                         % (config.model_type, family))
    torch_dtype = getattr(torch, dtype)

    def say(text):
        if verbose:
            print('  ' + text)

    # meta device：实例化只拿形状/名字，不占内存 —— 27B 也一样轻。
    with torch.device('meta'):
        model = by1full._model_class(family, mtp_spec)(config)
    groups = tensor_groups(model)
    n_layer = (getattr(config, 'num_hidden_layers', None)
               or getattr(config, 'n_layer', None) or 1)
    std = (getattr(config, 'initializer_range', None) or 0.02) / math.sqrt(2 * n_layer)
    say('影子张量 %d 个共享组（tied 并组），初始化 std=%.5f' % (len(groups), std))

    shard_plan = _shards(groups, max_shard_bytes)
    weight_map, total, shard_names = {}, 0, []
    for index, plan in enumerate(shard_plan):
        tensors = {}
        for group in plan:
            values = sample(group, seed, std, torch)
            if values.is_floating_point():
                values = values.to(torch_dtype).contiguous()
            for name in group['names']:
                tensors[name] = values.clone() if len(group['names']) > 1 else values
                weight_map[name] = None  # 稍后填文件名
                total += values.numel() * values.element_size()
        if len(shard_plan) == 1:
            filename = 'model.safetensors'
        else:
            filename = 'model-%05d-of-%05d.safetensors' % (index + 1, len(shard_plan))
        save_file(tensors, str(out / filename))
        for name in tensors:
            weight_map[name] = filename
        shard_names.append(filename)
        say('写 %s（%d 个张量）' % (filename, len(tensors)))
    if len(shard_plan) > 1:
        (out / 'model.safetensors.index.json').write_text(json.dumps({
            'metadata': {'total_size': total},
            'weight_map': weight_map}, indent=2) + '\n', encoding='utf-8')

    manifest = {
        'schema': 1,
        'kind': 'by1-shadow',
        'synthetic_values': True,
        'note': '随机值、真格式。验接线，不验真值。',
        'family': family,
        'mtp': mtp_spec or None,
        'seed': seed,
        'dtype': dtype,
        'init_std': std,
        'config_sha256': hashlib.sha256(config_text.encode('utf-8')).hexdigest(),
        'tensor_names': len(weight_map),
        'shared_groups': len(groups),
        'bytes': total,
        'shards': shard_names,
        'verified': list(VERIFIED),
        'not_verified': list(NOT_VERIFIED),
    }
    (out / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                encoding='utf-8')
    say('影子 checkpoint 就绪：%s（%.2f GB，%d 分片）'
        % (out, total / 1024 ** 3, len(shard_names)))
    return manifest


def reusable(out_dir, family, mtp_spec, seed, dtype, config_text):
    """已有的影子目录能直接用吗？判据全对上才复用，否则重建。"""
    path = pathlib.Path(out_dir) / MANIFEST
    try:
        old = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return False
    return (old.get('kind') == 'by1-shadow'
            and old.get('family') == family
            and old.get('mtp') == (mtp_spec or None)
            and old.get('seed') == seed
            and old.get('dtype') == dtype
            and old.get('config_sha256') == hashlib.sha256(config_text.encode('utf-8')).hexdigest()
            and (pathlib.Path(out_dir) / 'config.json').exists())


# ---------------------------------------------------------------------------
# --smoke：CPU 端到端自检。默认 gpt2 + 真 config（refs/），全程离线；
# --by1/--config 换家族（qwen3 等），--smoke-deep 扫 bf16 深模型边界。
# ---------------------------------------------------------------------------

def smoke(dtype='float32', by1_name='complete-gpt2.by1', config_file=None):
    """影子 checkpoint 的完整流程 rehearsal：造 -> 双 loader -> 对拍。

    config 必须是**真品**（声明对拍的一半在这里）；换家族时给
    `--config`，或用 `fetch_config()` 只拉 config 一个文件。"""
    import torch
    import transformers as T
    import by1check
    import by1full

    model_file = by1paths.model(by1_name)
    config_file = config_file or by1paths.ref('openai-community__gpt2.config.json')
    report, info = by1check.check(model_file)
    if report.count(by1check.E):
        raise RuntimeError('%s 没过检查：%s' % (by1_name, report.items[:3]))
    family, declared, mtp = by1full.compile_config(info)
    config_text = pathlib.Path(config_file).read_text(encoding='utf-8')

    out = by1paths.root('build', 'shadow-smoke-' + by1_name.replace('.by1', ''))
    manifest = build(config_text, family, mtp, out, dtype=dtype)
    print('  两个 loader 从影子目录加载……')

    # **两边同一套注意力算子**（照 by1gpubench 的对拍语义）——
    # reference 不给这个参数会默认走 SDPA，而 by1 侧 load_model 强制
    # eager：两个内核的归约序不同，qwen3 会差出 4.4e-03（实测），看起来
    # 像接线错，其实是内核差。接线对拍不比内核。
    reference, loading = T.AutoModelForCausalLM.from_pretrained(
        str(out), output_loading_info=True, local_files_only=True,
        attn_implementation='eager')
    problems = {k: loading.get(k) for k in ('missing_keys', 'unexpected_keys',
                                            'mismatched_keys') if loading.get(k)}
    if problems:
        raise RuntimeError('官方 loader 对影子 checkpoint 有意见：%s' % problems)
    by1_model = by1full.load_model(family, declared, mtp, str(out),
                                   local_files_only=True)

    generator = torch.Generator().manual_seed(manifest['seed'])
    vocab = reference.config.vocab_size
    ids = torch.randint(0, min(vocab, 10000), (1, 16), generator=generator)
    with torch.no_grad():
        a = reference(ids).logits
        b = by1_model(ids).logits
    if a.shape != b.shape:
        raise RuntimeError('logits 形状不一致：%s vs %s' % (tuple(a.shape), tuple(b.shape)))
    difference = (a.float() - b.float()).abs().max().item()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError('logits 有非有限值 —— 初始化缩放要修')
    if not torch.allclose(a.float(), b.float(), atol=2e-3, rtol=2e-3):
        raise RuntimeError('logits 对不上：max|Δ|=%.3e' % difference)
    with torch.no_grad():
        ga = reference.generate(ids, do_sample=False, min_new_tokens=8, max_new_tokens=8)
        gb = by1_model.generate(ids, do_sample=False, min_new_tokens=8, max_new_tokens=8)
    if not torch.equal(ga, gb):
        raise RuntimeError('贪心生成 token 不一致')
    print('  正向 OK：logits max|Δ| = %.3e，贪心 token 一致' % difference)

    # ── 反向半段：篡改必须当场红 ──────────────────────────────────
    # 「全绿」只有在「弄坏必红」成立时才算数。两个方向都试：
    # ① checkpoint 侧的 config 被篡改 ② 声明侧被篡改。
    def must_fail(what, fn):
        try:
            fn()
        except Exception as exc:
            print('  反向 OK：%s 被抓住 -> %s' % (what, str(exc)[:90]))
            return
        raise RuntimeError('反向失效：%s 篡改后竟然过了 —— 全绿是空转的' % what)

    def tamper(text):
        data = json.loads(text)
        for key in ('layer_norm_epsilon', 'rms_norm_eps', 'rms_epsilon'):
            if key in data:
                return key, json.dumps({**data, key: float(data[key]) * 7 + 0.01})
        raise RuntimeError('找不到可篡改的 eps 字段，反向半段没法做')

    print('  反向半段：篡改 config / 声明，都必须当场红……')
    key, bad_text = tamper(config_text)
    bad_out = by1paths.root('build', 'shadow-smoke-bad')
    build(bad_text, family, mtp, bad_out, dtype=dtype, verbose=False)
    must_fail('篡改 checkpoint config（%s）' % key,
              lambda: by1full.load_model(family, declared, mtp, str(bad_out),
                                         local_files_only=True))
    bad_declared = json.loads(json.dumps(declared))
    for probe in ('layer_norm_epsilon', 'rms_norm_eps', 'rms_epsilon'):
        if probe in bad_declared:
            bad_declared[probe] = float(bad_declared[probe]) * 7 + 0.01
            break
    else:
        raise RuntimeError('声明里找不到可篡改的 eps 字段，反向半段没法做')
    must_fail('篡改声明（与 checkpoint 不一致）',
              lambda: by1full.load_model(family, bad_declared, mtp, str(out),
                                         local_files_only=True))
    print('  验过：%s' % '；'.join(manifest['verified']))
    print('  没验：%s' % '；'.join(manifest['not_verified']))
    return manifest


def fetch_config(repo, out_dir=None):
    """只拉 `config.json` 一个文件（几 KB）—— 影子模式允许的唯一网络动作，
    慢网也下得动。hf-mirror 优先、HF 直连兜底；落在 build/ 里可复用。"""
    import urllib.request
    out_dir = pathlib.Path(out_dir or by1paths.root('build', 'shadow-configs'))
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / (repo.replace('/', '__') + '.config.json')
    if dest.exists():
        return str(dest)
    last = None
    for tpl in ('https://hf-mirror.com/%s/resolve/main/config.json',
                'https://huggingface.co/%s/resolve/main/config.json'):
        try:
            req = urllib.request.Request(tpl % repo, headers={'User-Agent': 'by1'})
            with urllib.request.urlopen(req, timeout=30) as r:
                dest.write_bytes(r.read())
            return str(dest)
        except Exception as ex:
            last = ex
    raise RuntimeError('config.json 两个源都拉不到（%s）：%s' % (repo, last))


def deep_scan(dtype='bfloat16', layers=64, hidden=1024, ff=3072):
    """bf16 **深模型**边界扫描：scaled init 在深度下数值必须有限。

    标准 init（std=initializer_range）在 64 层深、残差流逐层累积会把
    幅度推爆；`build()` 的 `initializer_range/sqrt(2*n_layer)` 就是压它的。
    判据：影子前向 logits **全有限**、幅度 O(1~1e3)。越界先修 `sample()`
    的缩放 —— **不许**放宽 isfinite，那不是修，是把爆掉藏起来。"""
    import torch
    import transformers as T

    cfg = T.Qwen3Config(
        vocab_size=4096, hidden_size=hidden, intermediate_size=ff,
        num_hidden_layers=layers, num_attention_heads=8, num_key_value_heads=4,
        head_dim=hidden // 8, rms_norm_eps=1e-6, tie_word_embeddings=False,
        max_position_embeddings=256)
    out = by1paths.root('build', 'shadow-smoke-deep')
    manifest = build(cfg.to_json_string(), 'qwen3', None, out, dtype=dtype,
                     verbose=False)
    model = T.AutoModelForCausalLM.from_pretrained(str(out), local_files_only=True)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits = model(ids).logits
    peak = float(logits.abs().max())
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError('深模型 bf16 前向有非有限值 —— scaled init 要修')
    print('  bf16 边界 OK：%d 层 x hidden %d（%.2f GB 影子），'
          'logits 全有限，max|logits| = %.3e'
          % (layers, hidden, manifest['bytes'] / 1024 ** 3, peak))
    return {'layers': layers, 'hidden': hidden, 'dtype': dtype,
            'max_abs_logit': peak, 'bytes': manifest['bytes']}


def smoke_qwen35(dtype='float32'):
    """qwen3_5 家族冒烟（视觉 + MTP）：**缩小配置**模式。

    真 config 的 27B 在 CPU 上建不动，所以配置用 `by1complete.config_of`
    的独立 tiny 配置（同一个意思只写一遍）；**mtp 规格取真 .by1**
    （complete-Qwen3.8-27B.by1），不自己编。范围：

        双 loader 完整装载 · 文本/图像前向 · 贪心生成 ·
        MTP 对独立公式（by1complete.mtp_formula）· tying 断言 ·
        mtp.* 命名对 refs 真品 · mtp 深度不符必红

    **不在本模式**：声明↔config 对拍（缩小配置没有对应 .by1 声明）。
    """
    import torch
    import transformers as T
    import by1check
    import by1full
    import by1complete

    report, info = by1check.check(by1paths.model('complete-Qwen3.8-27B.by1'))
    if report.count(by1check.E):
        raise RuntimeError('complete-Qwen3.8-27B.by1 没过检查：%s' % report.items[:3])
    family, _declared, mtp = by1full.compile_config(info)
    if family != 'qwen3_5' or not mtp:
        raise RuntimeError('本冒烟要的是 qwen3_5 + mtp 规格，拿到 %s / %s'
                           % (family, mtp))
    cfg = by1complete.config_of('qwen3_5', False, T)
    out = by1paths.root('build', 'shadow-smoke-qwen35')
    manifest = build(cfg.to_json_string(), family, mtp, out, dtype=dtype)

    def load(cls, tag):
        model, loading = cls.from_pretrained(
            str(out), config=T.AutoConfig.from_pretrained(str(out), local_files_only=True),
            output_loading_info=True, local_files_only=True,
            attn_implementation='eager')
        extra = [k for k in (loading.get('unexpected_keys') or [])
                 if not k.startswith('mtp.')]
        bad = {k: v for k, v in (('missing', loading.get('missing_keys')),
                                ('extra', extra),
                                ('mismatched', loading.get('mismatched_keys')))
               if v}
        if bad:
            raise RuntimeError('%s 对影子有意见：%s' % (tag, bad))
        return model

    reference = load(T.AutoModelForImageTextToText, '官方 loader')
    by1_model = load(by1full._model_class(family, mtp), 'by1 loader（含 mtp）')

    generator = torch.Generator().manual_seed(manifest['seed'])
    ids = torch.randint(0, cfg.text_config.vocab_size, (1, 16), generator=generator)
    with torch.no_grad():
        a = reference(input_ids=ids, use_cache=False).logits
        b = by1_model(input_ids=ids, use_cache=False).logits
    difference = (a.float() - b.float()).abs().max().item()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError('文本前向有非有限值')
    if not torch.allclose(a.float(), b.float(), atol=2e-3, rtol=2e-3):
        raise RuntimeError('文本前向对不上：max|Δ|=%.3e' % difference)
    print('  文本前向 OK：max|Δ| = %.3e' % difference)

    batch = by1complete.inputs('qwen3_5', torch)   # 自带 output_hidden_states
    with torch.no_grad():
        ia = reference(**batch).logits
        base = by1_model(**batch)
        ib = base.logits
    idiff = (ia.float() - ib.float()).abs().max().item()
    if not torch.isfinite(ia).all() or not torch.isfinite(ib).all():
        raise RuntimeError('图像前向有非有限值')
    if not torch.allclose(ia.float(), ib.float(), atol=2e-3, rtol=2e-3):
        raise RuntimeError('图像前向对不上：max|Δ|=%.3e' % idiff)
    print('  图像前向 OK：max|Δ| = %.3e（视觉塔接线）' % idiff)

    # ── MTP：对独立公式（by1complete.mtp_formula 不走 forward_mtp）──
    if not (by1_model.mtp.embed_tokens is by1_model.get_input_embeddings()
            and by1_model.mtp.lm_head is by1_model.get_output_embeddings()):
        raise RuntimeError('MTP 参数共享断了（embed/lm_head 不是同一对象）')
    mtp_ids = batch['input_ids'].roll(-1, 1)
    positions = torch.arange(mtp_ids.shape[1], device=ids.device)[None] + 1
    hidden = base.hidden_states[-1]
    with torch.no_grad():
        draft = by1_model.forward_mtp(mtp_ids, hidden, positions)['logits']
        formula = by1complete.mtp_formula(by1_model, mtp_ids, hidden, positions, torch)
    mdiff = (draft.float() - formula.float()).abs().max().item()
    if not torch.allclose(draft.float(), formula.float(), atol=2e-3, rtol=2e-3):
        raise RuntimeError('MTP 对独立公式不一致：max|Δ|=%.3e' % mdiff)
    print('  MTP OK：对独立公式 max|Δ| = %.3e；tying 断言过' % mdiff)

    # ── mtp.* 命名对 refs 真品（真权重对拍今天能过，所以必须相等）──
    refs_path = pathlib.Path(by1paths.ref('Qwen__Qwen3.8-27B.tensors.json'))
    if refs_path.exists():
        real = json.loads(refs_path.read_text(encoding='utf-8'))
        real_mtp = {k for k in real if k.startswith('mtp.')}
        mine_mtp = {k for k in by1_model.state_dict() if k.startswith('mtp.')}
        if real_mtp and mine_mtp != real_mtp:
            raise RuntimeError('mtp.* 命名与 refs 真品不一致：%s'
                               % sorted(mine_mtp ^ real_mtp)[:6])
        print('  mtp.* 命名与 refs 真品逐个一致（%d 个）' % len(mine_mtp))

    # ── 反向：mtp 深度不符必须被缺键抓住 ──────────────────────────
    bad = dict(mtp, depth=mtp['depth'] + 1)
    _m, lbad = by1full._model_class(family, bad).from_pretrained(
        str(out), config=T.AutoConfig.from_pretrained(str(out), local_files_only=True),
        output_loading_info=True, local_files_only=True, attn_implementation='eager')
    missing = lbad.get('missing_keys') or []
    if not any(k.startswith('mtp.layers.%d.' % mtp['depth']) for k in missing):
        raise RuntimeError('反向失效：mtp 深度不符竟然没报缺键 —— 空转的绿')
    print('  反向 OK：mtp 深度 +1 被缺键抓住（%d 个 missing_keys）' % len(missing))
    return manifest


# ---------------------------------------------------------------------------
# 影子源二：吃 refs/*.tensors.json（**真名真形状**）—— 命名一致性的正面关账。
# 源一（模型类）的命名是 by1 自己说的；源二的命名是发布产物说的。
# 两边都装载干净，"checkpoint 命名与官方一致"才算验过。
# ---------------------------------------------------------------------------

#: refs dtype -> torch。量化存储（U8 / F8_*）**不能**随机合成 ——
#: 随机字节不是合法量化块，合成出来的是假货（manifest 的 not_verified
#: 第二条）。碰上就显式跳过。
_DTYPE_MAP = {'BF16': 'bfloat16', 'F16': 'float16', 'F32': 'float32',
              'F64': 'float64', 'U8': None, 'F8_E4M3': None, 'F8_E5M2': None}


def build_from_refs(config_text, tensors_json_text, out_dir,
                    seed=SEED, dtype=None, max_shard_bytes=MAX_SHARD_BYTES):
    """refs 的 (真名, 真形状, 真 dtype) + 随机值 -> 影子 checkpoint。

    和 `build()` 的唯一区别：**名字清单来自 refs 而不是模型类**。
    值照旧按名字哈希播种（可复现）；dtype 跟 refs 走。"""
    import torch
    from safetensors.torch import save_file

    real = json.loads(tensors_json_text)
    if not real or any(v is None for v in real.values()):
        raise ValueError('refs 是 F3（只有名字没形状）—— 无法合成，先跑 by1shapes')
    specs = []
    for name in sorted(real):
        meta = real[name]
        src_dtype = str(meta.get('dtype', 'F32')).upper()
        if _DTYPE_MAP.get(src_dtype) is None:
            raise ValueError('量化存储 dtype %s 不能随机合成（%s）—— 走真权重'
                             % (src_dtype, name))
        shape = tuple(int(x) for x in meta.get('shape') or ())
        specs.append((name, shape, _DTYPE_MAP.get(src_dtype, 'float32')))

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(config_text, encoding='utf-8')
    import transformers as T
    config = T.AutoConfig.from_pretrained(str(out), local_files_only=True)
    std = (getattr(config, 'initializer_range', None) or 0.02) / math.sqrt(
        2 * (getattr(config, 'num_hidden_layers', None)
             or getattr(config, 'n_layer', None) or 1))

    groups = [{'names': [n], 'shape': s, 'dtype': getattr(torch, d)}
              for n, s, d in specs]
    shard_plan = _shards(groups, max_shard_bytes)
    weight_map, total, shard_names = {}, 0, []
    for index, plan in enumerate(shard_plan):
        tensors = {}
        for group in plan:
            name = group['names'][0]
            values = sample(group, seed, std, torch)
            if values.is_floating_point() and dtype:
                values = values.to(getattr(torch, dtype)).contiguous()
            tensors[name] = values
            weight_map[name] = None
            total += values.numel() * values.element_size()
        filename = ('model.safetensors' if len(shard_plan) == 1 else
                    'model-%05d-of-%05d.safetensors' % (index + 1, len(shard_plan)))
        save_file(tensors, str(out / filename))
        for name in tensors:
            weight_map[name] = filename
        shard_names.append(filename)
    if len(shard_plan) > 1:
        (out / 'model.safetensors.index.json').write_text(json.dumps({
            'metadata': {'total_size': total},
            'weight_map': weight_map}, indent=2) + '\n', encoding='utf-8')
    manifest = {
        'schema': 1, 'kind': 'by1-shadow-refs',
        'synthetic_values': True, 'real_names': True,
        'note': '名字/形状/dtype 取自 refs 真产物；值是随机的。验命名一致性与接线。',
        'config_sha256': hashlib.sha256(config_text.encode('utf-8')).hexdigest(),
        'seed': seed, 'tensor_names': len(weight_map), 'bytes': total,
        'shards': shard_names, 'verified': VERIFIED + [
            'checkpoint 命名与官方发布物一致（名字即 refs 真名）'],
        'not_verified': list(NOT_VERIFIED[:1]) + list(NOT_VERIFIED[2:]),
    }
    (out / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2)
                                + '\n', encoding='utf-8')
    return manifest


def refs_check(only=''):
    """**元对照**：模型类的装载键（state_dict）vs refs 真名 —— 全部有
    complete 声明 + refs 元数据的模型，逐个比名字和形状，零字节。

    判据（T8）：by1 loader 的键集 == refs 真名键集（tying 容忍项除外），
    零 missing / zero unexpected —— 这就是"命名缺口"的正面关账。"""
    import glob
    import torch
    import transformers as T
    import by1check
    import by1full

    ok_n = bad_n = 0
    for f in sorted(glob.glob(os.path.join(by1paths.MODELS, 'complete-*.by1'))):
        base = os.path.basename(f)[len('complete-'):-len('.by1')]
        if only and only not in base:
            continue
        cands = glob.glob(by1paths.ref('*__%s.tensors.json' % base)) \
            or glob.glob(by1paths.ref('%s.tensors.json' % base))
        ccands = glob.glob(by1paths.ref('*__%s.config.json' % base)) \
            or glob.glob(by1paths.ref('%s.config.json' % base))
        if not cands or not ccands:
            continue
        real = {k: v for k, v in
                json.loads(pathlib.Path(cands[0]).read_text(encoding='utf-8')).items()
                if v}
        if not real:
            print('  -- %-40s refs 是 F3（无形状），名字可比形状跳过' % base)
        # 量化存储展开（_blocks/_scales/_bias）不适用本对照 ——
        # 模型类的键是解量化视角的基名，真文件是打包块。**走真权重**。
        if any(k.endswith(('_blocks', '_scales')) for k in real):
            print('  -- %-40s 量化存储展开（blocks/scales），不适用本对照，'
                  '走真权重' % base)
            continue
        report, info = by1check.check(f)
        if report.count(by1check.E):
            print('  !! %-40s .by1 没过检查' % base)
            bad_n += 1
            continue
        family, _declared, mtp = by1full.compile_config(info)
        config_text = pathlib.Path(ccands[0]).read_text(encoding='utf-8')
        # 直接从 config 原文建配置对象（不落盘；多余字段 transformers 会收下）
        cfg = getattr(T, by1full.FAMILIES[family][0])(**json.loads(config_text))
        with torch.device('meta'):
            model = by1full._model_class(family, mtp)(cfg)
        keys = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        tied = bool(getattr(cfg, 'tie_word_embeddings', False))
        out_name = getattr(model, 'get_output_embeddings', lambda: None)()
        tied_keys = set()
        if tied and out_name is not None:
            # remove_duplicate=False —— 默认去重会把 tied 的第二个名字
            # 从 named_parameters 里吞掉，tied 容忍就永远查不出来。
            # 容忍集是**该参数的全部名字**（wte/lm_head 同一对象）。
            for k, v in model.named_parameters(remove_duplicate=False):
                if v is out_name.weight:
                    tied_keys.add(k)
        # **跟加载器同一套容错**：`_keys_to_ignore_on_load_*` 是 transformers
        # 显式声明的"这两边都别管"（gpt2 真文件里的 `h.N.attn.bias` 掩码
        # buffer 就靠它），对照不认它就会把加载器管的事误报成命名缺口。
        import re as _re
        def _ignores(attr):
            return [_re.compile(p) for p in
                    (getattr(type(model), attr, None) or [])]
        ign_unexp = _ignores('_keys_to_ignore_on_load_unexpected')
        ign_miss = _ignores('_keys_to_ignore_on_load_missing')
        # **base_model_prefix 别名**：加载器给检查点键自动补前缀
        # （gpt2 的真文件是 `h.0.…`，模型键是 `transformer.h.0.…`）。
        # 对照两侧都收这个别名，否则会把加载器管的事误报成命名缺口。
        bp = (getattr(model, 'base_model_prefix', '') or '').strip('.')
        def ref_key(k):
            if k in real:
                return k
            if bp and k.startswith(bp + '.') and k[len(bp) + 1:] in real:
                return k[len(bp) + 1:]
            return None
        missing, shape_bad = [], []
        for k, shape in keys.items():
            rk = ref_key(k)
            if rk is None:
                if k not in tied_keys and not any(p.search(k) for p in ign_miss):
                    missing.append(k)
                continue
            if real[rk].get('shape') and tuple(int(x) for x in real[rk]['shape']) != shape:
                shape_bad.append(k)
        extra = [k for k in real if k not in keys
                 and not (bp and (bp + '.' + k) in keys)
                 and not any(p.search(k) for p in ign_unexp)]
        if missing or extra or shape_bad:
            bad_n += 1
            print('  !! %-40s missing %d extra %d 形状不符 %d'
                  % (base, len(missing), len(extra), len(shape_bad)))
            for tag, lst in (('缺', missing), ('多', extra), ('形', shape_bad)):
                for k in lst[:3]:
                    print('       %s %s' % (tag, k))
        else:
            ok_n += 1
            print('  ok %-40s %d 个键与 refs 真名逐个一致%s'
                  % (base, len(keys),
                     '（tied %s 容忍）' % '/'.join(sorted(tied_keys))
                     if tied_keys else ''))
    print()
    print('  [T8 refs 元对照] %d 个模型：一致 %d · 不一致 %d' % (ok_n + bad_n, ok_n, bad_n))
    return 0 if bad_n == 0 else 1


def smoke_refs(by1_name='complete-gpt2.by1', weights_dir=None, seed=None):
    """源二 e2e：refs 真名真形状合成 -> 双 loader 零 missing/unexpected +
    前向逐位。**这是"命名与官方一致"从假设变成验证的那一步。**

    `weights_dir` 给了就用**真 checkpoint**（T13：真权重独立数值对拍）——
    判据一个字不改（同两个 loader、同一输入、同一容差），
    只把权重来源从"按 refs 合成"换成"发布方原样"。
    **CPU 也能跑**：0.6B fp32 两份约 4.8 GB，不需要显卡。
    """
    import torch
    import transformers as T
    import by1check
    import by1full

    base = (by1_name[len('complete-'):-len('.by1')]
            if by1_name.startswith('complete-') else by1_name[:-len('.by1')])
    tcands = glob.glob(by1paths.ref('*__%s.tensors.json' % base)) \
        or glob.glob(by1paths.ref('%s.tensors.json' % base))
    ccands = glob.glob(by1paths.ref('*__%s.config.json' % base)) \
        or glob.glob(by1paths.ref('%s.config.json' % base))
    if not tcands or not ccands:
        raise RuntimeError('refs 里缺 %s 的 tensors/config —— 源二要有真产物元数据' % base)
    report, info = by1check.check(by1paths.model(by1_name))
    if report.count(by1check.E):
        raise RuntimeError('%s 没过检查：%s' % (by1_name, report.items[:3]))
    family, declared, mtp = by1full.compile_config(info)
    config_text = pathlib.Path(ccands[0]).read_text(encoding='utf-8')
    tensors_text = pathlib.Path(tcands[0]).read_text(encoding='utf-8')

    out = by1paths.root('build', 'shadow-smoke-refs-' + base)
    real = bool(weights_dir)
    if real:
        # **真权重分支**：不合成、不下载 —— 目录由调用方给（本地已有）。
        out = str(weights_dir)
        n_names = len(json.loads(tensors_text))
        total = sum(f.stat().st_size for f in
                    pathlib.Path(out).glob('*.safetensors'))
        manifest = {'tensor_names': n_names, 'bytes': total,
                    'seed': seed if seed is not None else 20261016}
        print('  真权重 %s：%d 个真名（%.2f GB，未合成）'
              % (out, n_names, total / 1024 ** 3))
    else:
        manifest = build_from_refs(config_text, tensors_text, out, dtype='float32')
        print('  源二影子 %s：%d 个真名（%.2f GB）' % (out, manifest['tensor_names'],
                                                  manifest['bytes'] / 1024 ** 3))

    load_kwargs = {} if not real else {'dtype': torch.float32}
    reference, loading = T.AutoModelForCausalLM.from_pretrained(
        str(out), output_loading_info=True, local_files_only=True,
        attn_implementation='eager', **load_kwargs)
    extra = [k for k in (loading.get('unexpected_keys') or [])
             if not k.startswith('mtp.')]
    bad = {k: v for k, v in (('missing', loading.get('missing_keys')),
                            ('extra', extra),
                            ('mismatched', loading.get('mismatched_keys'))) if v}
    if bad:
        raise RuntimeError('官方 loader 对源二影子有意见：%s' % bad)
    by1_model = by1full.load_model(family, declared, mtp, str(out),
                                   local_files_only=True, **load_kwargs)

    generator = torch.Generator().manual_seed(manifest['seed'])
    ids = torch.randint(0, min(reference.config.vocab_size, 10000),
                        (1, 16), generator=generator)
    with torch.no_grad():
        a = reference(ids).logits
        b = by1_model(ids).logits
    difference = (a.float() - b.float()).abs().max().item()
    if not torch.allclose(a.float(), b.float(), atol=2e-3, rtol=2e-3):
        raise RuntimeError('源二前向对不上：max|Δ|=%.3e' % difference)
    print('  源二 e2e OK：双 loader 零 missing/unexpected，前向 max|Δ| = %.3e'
          % difference)
    return manifest


def smoke_state(by1_name='complete-Qwen3-0.6B.by1', weights_dir=None,
                length=16, prefill=8, seed=20261016):
    """T14：**增量（带 cache）与全量重算的逐位对照**（真权重，CPU 可跑）。

    两个 loader 各跑两遍：一遍 `use_cache=False` 全量、一遍
    prefill + 逐 token 增量（带 `past_key_values`）。判据 = 同一模型两条路
    logits 一致（2e-3，与仓库其他对拍的尺子同一把）。

    **为什么值得单列**：cache 那条路读的是 `position_ids` / `attention_mask` /
    rope 三样的接线 —— 接线错了**全量那一遍仍然是对的**，只有增量会歪。
    这也是路线图 T14「增量状态」那一格的判据。

    需要 `--weights-dir`（真 checkpoint 目录）：合成权重上验 cache 语义没有意义。
    """
    import gc
    import torch
    import transformers as T
    import by1check
    import by1full

    if not weights_dir:
        raise RuntimeError('--smoke-state 需要 --weights-dir（真 checkpoint 目录）')
    report, info = by1check.check(by1paths.model(by1_name))
    if report.count(by1check.E):
        raise RuntimeError('%s 没过检查：%s' % (by1_name, report.items[:3]))
    family, declared, mtp = by1full.compile_config(info)

    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 10000, (1, length), generator=generator)
    mask = torch.ones_like(ids)

    def run(model):
        with torch.no_grad():
            full = model(input_ids=ids, attention_mask=mask,
                         use_cache=False).logits
            pre = model(input_ids=ids[:, :prefill],
                        attention_mask=mask[:, :prefill], use_cache=True)
            cache = pre.past_key_values
            parts = [pre.logits]
            for t in range(prefill, length):
                step = model(input_ids=ids[:, t:t + 1],
                             attention_mask=mask[:, :t + 1],
                             past_key_values=cache, use_cache=True)
                parts.append(step.logits)
        inc = torch.cat(parts, dim=1)
        if inc.shape != full.shape:
            raise RuntimeError('增量与全量的形状不一致：%s vs %s'
                               % (tuple(inc.shape), tuple(full.shape)))
        return (inc.float() - full.float()).abs().max().item()

    kwargs = {'dtype': torch.float32, 'attn_implementation': 'eager',
              'local_files_only': True}
    out = str(weights_dir)
    print('  真权重 %s：增量 vs 全量（长 %d，prefill %d）'
          % (out, length, prefill))
    reference = T.AutoModelForCausalLM.from_pretrained(out, **kwargs).eval()
    d_ref = run(reference)
    del reference
    gc.collect()
    candidate = by1full.load_model(family, declared, mtp, out, **kwargs).eval()
    d_by1 = run(candidate)
    del candidate
    gc.collect()
    print('  官方 loader：max|Δ| = %.3e' % d_ref)
    print('  by1 模型   ：max|Δ| = %.3e' % d_by1)
    if not (d_ref <= 2e-3 and d_by1 <= 2e-3):
        raise RuntimeError('增量与全量对不上（判据 2e-3）：官方 %.3e · by1 %.3e'
                           % (d_ref, d_by1))
    print('  [PASS] 增量 == 全量（两个 loader 都在 2e-3 内）')
    return {'official_max_abs': d_ref, 'by1_max_abs': d_by1,
            'length': length, 'prefill': prefill}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--by1', help='.by1 完整架构声明（取家族与 mtp 规格）')
    parser.add_argument('--config', help='真品 config.json 的路径（refs/ 里有现成的）')
    parser.add_argument('--out', help='影子 checkpoint 输出目录')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--dtype', default=None,
                        choices=['bfloat16', 'float16', 'float32'],
                        help='不给：构建默认 bfloat16，冒烟默认 float32')
    parser.add_argument('--max-shard-gb', type=float, default=8.0)
    parser.add_argument('--smoke', action='store_true',
                        help='CPU 端到端自检（默认 gpt2 离线；--by1/--config 换家族）')
    parser.add_argument('--smoke-deep', action='store_true',
                        help='bf16 深模型边界扫描（64 层合成配置）')
    parser.add_argument('--smoke-q35', action='store_true',
                        help='qwen3_5 家族冒烟（视觉+MTP，缩小配置模式）')
    parser.add_argument('--refs-check', nargs='?', const='', metavar='ONLY',
                        help='元对照：模型类装载键 vs refs 真名（T8 判据，零字节）')
    parser.add_argument('--smoke-refs', action='store_true',
                        help='源二 e2e：refs 真名合成 -> 双 loader + 前向（默认 gpt2）')
    parser.add_argument('--weights-dir',
                        help='配 --smoke-refs：用**真 checkpoint 目录**代替合成影子 '
                             '（T13 真权重对拍；CPU 可跑，0.6B fp32 约 4.8 GB）')
    parser.add_argument('--smoke-state', action='store_true',
                        help='T14：真权重下「增量（cache）vs 全量重算」逐位对照 '
                             '（要 --weights-dir）')
    parser.add_argument('--fetch-config', metavar='REPO',
                        help='冒烟前只拉该仓库的 config.json（几 KB）')
    args = parser.parse_args(argv)
    if args.refs_check is not None:
        return refs_check(args.refs_check)
    if args.smoke_state:
        try:
            smoke_state(by1_name=args.by1 or 'complete-Qwen3-0.6B.by1',
                        weights_dir=args.weights_dir)
        except Exception as exc:
            return by1skip.broken('T14 增量对拍失败：%s' % exc)
        return 0
    if args.smoke_refs:
        try:
            smoke_refs(by1_name=args.by1 or 'complete-gpt2.by1',
                       weights_dir=args.weights_dir)
        except Exception as exc:
            return by1skip.broken('源二 e2e 失败：%s' % exc)
        return 0
    if args.smoke_deep:
        try:
            deep_scan(dtype=args.dtype or 'bfloat16')
        except Exception as exc:
            return by1skip.broken('bf16 边界扫描失败：%s' % exc)
        if not args.smoke:
            return 0
    if args.smoke_q35:
        try:
            smoke_qwen35(dtype=args.dtype or 'float32')
        except Exception as exc:
            return by1skip.broken('qwen3_5 家族冒烟失败：%s' % exc)
        if not args.smoke:
            return 0
    if args.smoke:
        config_file = args.config
        try:
            if args.fetch_config:
                config_file = fetch_config(args.fetch_config)
            smoke(dtype=args.dtype or 'float32',
                  by1_name=args.by1 or 'complete-gpt2.by1',
                  config_file=config_file)
        except Exception as exc:
            return by1skip.broken('影子自检失败：%s' % exc)
        return 0
    if not (args.by1 and args.config and args.out):
        return by1skip.caller('--by1 / --config / --out 三样都要给（或用 --smoke）')
    try:
        import by1check
        import by1full
        report, info = by1check.check(by1paths.model(args.by1))
        if report.count(by1check.E):
            return by1skip.caller('%s 没过检查：%s' % (args.by1, report.items[:3]))
        family, _declared, mtp = by1full.compile_config(info)
        config_text = pathlib.Path(args.config).read_text(encoding='utf-8')
        manifest = build(config_text, family, mtp, args.out,
                         seed=args.seed, dtype=args.dtype or 'bfloat16',
                         max_shard_bytes=int(args.max_shard_gb * 1024 ** 3))
    except Exception as exc:
        return by1skip.broken('影子构建失败：%s' % exc)
    print('  验过：%s' % '；'.join(manifest['verified']))
    print('  没验：%s' % '；'.join(manifest['not_verified']))
    return 0


if __name__ == '__main__':
    _sys.exit(main())
