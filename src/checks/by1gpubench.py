#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1gpubench -- 真实权重与发布方 Transformers 模型的顺序加载、数值与性能对比。"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401

import argparse
import datetime
import gc
import hashlib
import json
import os
import pathlib
import statistics
import time

import by1check
import by1full
import by1shadow
import by1skip

MODELS = [
    ('complete-gpt2.by1', 'openai-community/gpt2'),
    ('complete-Qwen3-0.6B.by1', 'Qwen/Qwen3-0.6B'),
    ('complete-Qwen2.5-7B-Instruct.by1', 'Qwen/Qwen2.5-7B-Instruct'),
    ('complete-Qwen3-8B.by1', 'Qwen/Qwen3-8B'),
    ('complete-gpt-oss-20b.by1', 'openai/gpt-oss-20b'),
    ('complete-Qwen3.8-27B.by1', 'Qwen/Qwen3.8-27B'),
]
PINNED_REVISIONS = {
    'openai/gpt-oss-20b': '6cee5e81ee83917806bbde320786a8fb61efebee',
    'Qwen/Qwen3.8-27B': '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
    'Qwen/Qwen3.8-27B-FP8': '017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def time_cuda(fn, torch, warmup, repeats, tokens):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        wall, event = [], []
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            before = time.perf_counter()
            start.record()
            output = fn()
            end.record()
            torch.cuda.synchronize()
            wall.append((time.perf_counter() - before) * 1000)
            event.append(start.elapsed_time(end))
            del output
    median = statistics.median(wall)
    return {'wall_median_ms': median, 'wall_samples_ms': wall, 'cuda_median_ms': statistics.median(event),
            'tokens_per_second': tokens / (median / 1000),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def comparison(a, b, torch):
    require(a.shape == b.shape, 'Output shape mismatch')
    require(torch.isfinite(a).all().item() and torch.isfinite(b).all().item(), 'Non-finite output')
    difference = (a.float() - b.float()).abs()
    absolute = difference.max().item()
    peak = absolute / max(b.float().abs().max().item(), 1e-12)
    require(torch.allclose(a.float(), b.float(), atol=2e-3, rtol=2e-3),
            'Real-weight logits mismatch: abs=%.3e, relative_peak=%.3e' % (absolute, peak))
    return {'max_absolute': absolute, 'relative_to_peak': peak,
            'argmax_agreement': (a.argmax(-1) == b.argmax(-1)).float().mean().item()}


def quantization_contract(repo, family, explicit_path=None):
    """Read the checkpoint's explicit quant contract from a .by1 description."""
    if explicit_path:
        filename = os.path.abspath(explicit_path)
    elif repo == 'openai/gpt-oss-20b':
        filename = os.path.join(by1paths.MODELS, 'gpt-oss-20b.by1')
    else:
        return None
    report, info = by1check.check(filename)
    require(not report.count(by1check.E), '.by1 quant contract failed validation')
    require(info.get('arch') == family,
            '.by1 quant contract family %s does not match checkpoint family %s'
            % (info.get('arch'), family))
    contract = by1full.compile_quantization(info)
    require(contract, '.by1 file has no validated quantization contract')
    contract['source_sha256'] = hashlib.sha256(pathlib.Path(filename).read_bytes()).hexdigest()
    return contract


def inputs(checkpoint, family, lengths, torch, T, with_image=True):
    cfg = T.AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    text_cfg = cfg.text_config if family == 'qwen3_5' else cfg
    generator = torch.Generator().manual_seed(20261013)
    batches = {}
    for length in lengths:
        ids = torch.randint(0, min(text_cfg.vocab_size, 10000), (1, length), generator=generator)
        batches['text-' + str(length)] = {'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'use_cache': False}
    if family == 'qwen3_5' and with_image:
        from PIL import Image
        processor = T.AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
        image = Image.new('RGB', (128, 128), color=(32, 120, 220))
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                    {'type': 'text', 'text': 'Describe this image briefly.'}]}]
        batch = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                              return_tensors='pt', add_generation_prompt=True)
        batches['image'] = dict(batch, use_cache=False)
    return batches


def shadow_config_text(repo, args, hub):
    """影子模式的 config.json 原文。**必须是真品**，见 by1shadow 模块头。

    查找顺序：--shadow-config > refs/*.config.json > 只拉 config.json 一个
    文件（几 KB，再慢的网也下得动）。三处都没有就跳过，不硬编。
    """
    if args.shadow_config:
        return pathlib.Path(args.shadow_config).read_text(encoding='utf-8')
    for name in (repo.replace('/', '__') + '.config.json',
                 repo.replace('/', '_') + '.config.json'):
        candidate = pathlib.Path(by1paths.REFS, name)
        if candidate.exists():
            return candidate.read_text(encoding='utf-8')
    try:
        path = hub.hf_hub_download(repo, 'config.json',
                                   revision=args.revision or PINNED_REVISIONS.get(repo))
    except Exception as exc:
        raise by1shadow.ShadowSkip('拿不到真品 config.json（--shadow-config / refs/ / 网络三处都没有）：%s' % exc)
    return pathlib.Path(path).read_text(encoding='utf-8')


def shadow_checkpoint(repo, family, cfg, mtp, args, hub, work):
    """造（或复用）影子 checkpoint，返回 (目录, manifest)。0 权重下载。"""
    config_text = shadow_config_text(repo, args, hub)
    out = work / ('shadow-' + family)
    if by1shadow.reusable(out, family, mtp, args.shadow_seed, args.shadow_dtype, config_text):
        manifest = json.loads((out / by1shadow.MANIFEST).read_text(encoding='utf-8'))
    else:
        manifest = by1shadow.build(config_text, family, mtp, out,
                                   seed=args.shadow_seed, dtype=args.shadow_dtype)
    return str(out), manifest


def benchmark(filename, repo, args, torch, T, hub, work, save, row):
    report, info = by1check.check(filename)
    require(not report.count(by1check.E), str(report.items))
    family, cfg, mtp = by1full.compile_config(info)
    quantization = quantization_contract(repo, family, args.quant_contract)
    if args.shadow:
        # 量化存储没法"随机"——随机字节不是合法量化块。显式跳过，
        # 不是失败也**不是验过**；真权重对拍仍然覆盖它。
        if quantization:
            raise by1shadow.ShadowSkip('量化契约（%s）的存储格式无法随机合成，影子模式跳过；真权重对拍覆盖它'
                                       % quantization['format'])
        if family == 'gpt_oss':
            raise by1shadow.ShadowSkip('gpt-oss 的 MXFP4 发布存储无法随机合成，影子模式跳过；真权重对拍覆盖它')
        checkpoint, manifest = shadow_checkpoint(repo, family, cfg, mtp, args, hub, work)
        row.update(repo=repo, revision='synthetic', download_endpoint='none',
                   checkpoint_bytes=manifest['bytes'],
                   checkpoint_files=manifest['shards'],
                   repository_safetensors_bytes=manifest['bytes'],
                   weights='synthetic', shadow=manifest)
    else:
        remote = hub.HfApi().model_info(repo,
            revision=args.revision or PINNED_REVISIONS.get(repo), files_metadata=True, timeout=30)
        repository_weights = [s for s in remote.siblings if s.rfilename.endswith('.safetensors')]
        checkpoint_weights = [s for s in repository_weights if '/' not in s.rfilename]
        row.update(repo=repo, revision=remote.sha, download_endpoint=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'),
                   checkpoint_bytes=sum(s.size or 0 for s in checkpoint_weights),
                   checkpoint_files=[s.rfilename for s in checkpoint_weights],
                   repository_safetensors_bytes=sum(s.size or 0 for s in repository_weights))
        ignore_patterns = ['tokenizer.json'] if family == 'gpt_oss' else None
        checkpoint = hub.snapshot_download(repo, revision=remote.sha,
            allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', '*.tiktoken'],
            ignore_patterns=ignore_patterns, max_workers=4)
    if quantization:
        row['quantization_contract'] = quantization
    save()
    row['checkpoint_config_sha256'] = hashlib.sha256(pathlib.Path(checkpoint, 'config.json').read_bytes()).hexdigest()
    if quantization and quantization['format'] == 'fp8_e4m3_block':
        import by1quant
        expected_count = (407 if repo == 'Qwen/Qwen3.8-27B-FP8' else None)
        row['checkpoint_quantized_storage'] = by1quant.audit_checkpoint_storage(
            checkpoint, quantization, expected_weight_count=expected_count)
        save()
    save()
    batches = inputs(checkpoint, family, args.lengths, torch, T, with_image=not args.shadow)
    kwargs = dict(dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='eager', local_files_only=True)
    native_quantization = bool(quantization) and (
        args.quant_contract is not None or family != 'gpt_oss'
        or args.gpt_oss_quant_mode == 'native')
    if family == 'gpt_oss' and not native_quantization:
        kwargs['quantization_config'] = T.Mxfp4Config(dequantize=True)
        row['quantization'] = 'published MXFP4 checkpoint dequantized to BF16 for both loaders'
    elif native_quantization:
        checkpoint_cfg = T.AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        import by1quant
        kwargs['quantization_config'] = by1quant.make_transformers_config(
            quantization, getattr(checkpoint_cfg, 'quantization_config', None), T)
        row['quantization'] = 'native %s storage requested for both loaders' % quantization['format']
    reference_class = T.AutoModelForImageTextToText if family == 'qwen3_5' else T.AutoModelForCausalLM
    row['runs'] = {}
    row['phase'] = 'original-load'
    save()
    original, loading = reference_class.from_pretrained(checkpoint, output_loading_info=True, **kwargs)
    extra = [k for k in loading.get('unexpected_keys', []) if not (family == 'qwen3_5' and k.startswith('mtp.'))]
    require(not extra and not loading.get('missing_keys') and not loading.get('mismatched_keys'),
            'Original loading is incomplete: ' + str(loading))
    row['original_ignored_mtp_keys'] = len([k for k in loading.get('unexpected_keys', []) if k.startswith('mtp.')])
    if native_quantization:
        import by1quant
        row['original_quantized_storage'] = by1quant.audit_runtime_storage(original, quantization, torch)
    original.eval()
    for name, batch in batches.items():
        moved = {k: v.to(device='cuda', dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                 if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        row['phase'] = 'original-' + name
        save()
        with torch.inference_mode():
            output = original(**moved)
            require(output.logits.is_cuda and torch.isfinite(output.logits).all().item(), 'Original not executing on GPU')
            torch.save(output.logits.detach().cpu(), work / (family + '-' + name + '-original.pt'))
            del output
        n_tokens = int(moved['input_ids'].numel())
        row['runs'][name] = {'input_shape': list(moved['input_ids'].shape),
                             'original': time_cuda(lambda: original(**moved), torch, args.warmup, args.repeats, n_tokens)}
        save()
    generation = dict(batches['text-' + str(args.lengths[0])])
    generation.pop('use_cache')
    generation = {k: v.cuda() for k, v in generation.items()}
    with torch.inference_mode():
        generated = original.generate(**generation, do_sample=False, min_new_tokens=args.new_tokens,
                                      max_new_tokens=args.new_tokens, use_cache=True)
        torch.save(generated.cpu(), work / (family + '-generation-original.pt'))
    row['generation'] = {'original': time_cuda(lambda: original.generate(**generation, do_sample=False,
        min_new_tokens=args.new_tokens, max_new_tokens=args.new_tokens, use_cache=True),
        torch, args.warmup, args.repeats, args.new_tokens)}
    del original, generated, moved
    gc.collect()
    torch.cuda.empty_cache()
    row['phase'] = 'by1-load'
    save()
    model = by1full.load_model(family, cfg, mtp, checkpoint,
                               quantization=(quantization if native_quantization else None),
                               **kwargs).eval()
    require(all(p.is_cuda for p in model.parameters()), 'by1 loader has CPU parameters')
    if native_quantization:
        row['by1_quantized_storage'] = model._by1_quantization_audit
    row['by1_parameters'] = sum(p.numel() for p in model.parameters())
    for name, batch in batches.items():
        moved = {k: v.to(device='cuda', dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                 if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        row['phase'] = 'by1-' + name
        save()
        with torch.inference_mode():
            output = model(**moved)
            require(output.logits.is_cuda, 'by1 output is not CUDA')
            expected = torch.load(work / (family + '-' + name + '-original.pt'), weights_only=True)
            row['runs'][name]['numerics'] = comparison(output.logits.cpu(), expected, torch)
            del output, expected
        row['runs'][name]['by1'] = time_cuda(lambda: model(**moved), torch, args.warmup, args.repeats,
                                            int(moved['input_ids'].numel()))
        row['runs'][name]['latency_ratio_by1_over_original'] = (row['runs'][name]['by1']['wall_median_ms'] /
                                                             row['runs'][name]['original']['wall_median_ms'])
        save()
    with torch.inference_mode():
        generated = model.generate(**generation, do_sample=False, min_new_tokens=args.new_tokens,
                                   max_new_tokens=args.new_tokens, use_cache=True)
        expected = torch.load(work / (family + '-generation-original.pt'), weights_only=True)
        require(torch.equal(generated.cpu(), expected), 'Greedy generation tokens differ from original')
    row['generation']['identical_tokens'] = True
    row['generation']['by1'] = time_cuda(lambda: model.generate(**generation, do_sample=False,
        min_new_tokens=args.new_tokens, max_new_tokens=args.new_tokens, use_cache=True),
        torch, args.warmup, args.repeats, args.new_tokens)
    if mtp:
        from checks import by1complete
        # 影子模式没有 processor 文件（tokenzier 之类要下载），图像批次换文本
        # 批次 —— 两个 loader 同输入，MTP 接线照样对得上；图像塔不在影子范围。
        mtp_batch = 'image' if not args.shadow else 'text-' + str(args.lengths[0])
        moved = {k: v.to(device='cuda', dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                 if isinstance(v, torch.Tensor) else v for k, v in batches[mtp_batch].items()}
        with torch.inference_mode():
            base = model(**moved, output_hidden_states=True)
            ids = moved['input_ids'].roll(-1, 1)
            positions = torch.arange(ids.shape[1], device='cuda')[None] + 1
            hidden = base.hidden_states[-1]
            draft = model.forward_mtp(ids, hidden, positions)['logits']
            formula = by1complete.mtp_formula(model, ids, hidden, positions, torch)
            row['mtp'] = {'numerics_vs_connection_formula': comparison(draft.cpu(), formula.cpu(), torch),
                          'reference_scope': 'independent connection formula using checkpoint MTP weights; not a vLLM benchmark',
                          'by1': time_cuda(lambda: model.forward_mtp(ids, hidden, positions), torch,
                                           args.warmup, args.repeats, ids.numel())}
            require(model.mtp.embed_tokens is model.get_input_embeddings() and model.mtp.lm_head is model.get_output_embeddings(),
                    'Real checkpoint lost MTP parameter sharing')
    row['phase'] = 'complete'
    save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', choices=[r for _, r in MODELS])
    parser.add_argument('--checkpoint',
                        help='Optional Hugging Face checkpoint repo for the selected .by1 model / '
                             '可选 Hugging Face checkpoint 仓库，用于所选 .by1 架构')
    parser.add_argument('--revision',
                        help='Optional immutable Hugging Face revision / '
                             '可选 Hugging Face 固定 revision（建议 commit SHA）')
    parser.add_argument('--lengths', type=int, nargs='+', default=[32, 128])
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--new-tokens', type=int, default=16)
    parser.add_argument('--gpt-oss-quant-mode', choices=['dequantized', 'native'],
                        default='dequantized',
                        help='GPT-OSS quantized checkpoint loading mode / '
                             'GPT-OSS 量化 checkpoint 装载方式')
    parser.add_argument('--quant-contract',
                        help='Optional .by1 file declaring a native quantized checkpoint; '
                             'also enables quantized image/MTP benchmark paths when present / '
                             '可选 .by1 量化契约；有图像塔/MTP 时也会进入对应对拍')
    parser.add_argument('--shadow', action='store_true',
                        help='Synthetic shadow checkpoint: real tensor names/shapes/dtypes and real '
                             'config.json, seeded random values; no weight download / '
                             '影子模式：真格式随机值 checkpoint，0 权重下载（验接线，不验真值）')
    parser.add_argument('--shadow-config',
                        help='Real config.json for --shadow (default: refs/ then config-only fetch) / '
                             '影子模式的真品 config.json（默认 refs/，再不行只拉 config 一个文件）')
    parser.add_argument('--shadow-seed', type=int, default=by1shadow.SEED)
    parser.add_argument('--shadow-dtype', default='bfloat16',
                        choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--report', default='cgen/gpu-benchmark.json')
    args = parser.parse_args(argv)
    if min(args.lengths) < 2 or args.warmup < 0 or args.repeats < 1 or args.new_tokens < 1:
        return by1skip.caller('Lengths/repeats/new-tokens must be positive; warmup must be nonnegative')
    if not args.shadow and (args.shadow_config or args.shadow_seed != by1shadow.SEED
                            or args.shadow_dtype != 'bfloat16'):
        return by1skip.caller('--shadow-config/--shadow-seed/--shadow-dtype 只在 --shadow 下有意义')
    report = {'schema': 1, 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'dtype': 'bfloat16', 'batch_size': 1, 'attention': 'eager', 'warmup': args.warmup,
              'repeats': args.repeats, 'lengths': args.lengths, 'new_tokens': args.new_tokens,
              'forward_atol': 2e-3, 'forward_rtol': 2e-3,
              'scope': 'Synthetic shadow checkpoint (real names/shapes/dtypes, real config.json, seeded '
                       'random values); official config vs by1-compiled config, same Transformers '
                       'operators. Verifies loading, wiring, tying, greedy-token agreement. NOT real '
                       'weight values, NOT quantized storage.' if args.shadow else
                       'Published checkpoint; official config vs by1-compiled config, same Transformers operators. Not an independent kernel comparison.',
              'models': []}
    target = pathlib.Path(args.report)
    work = target.parent / (target.stem + '-outputs')
    work.mkdir(parents=True, exist_ok=True)
    def save():
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    try:
        import torch
        import transformers as T
        import huggingface_hub as hub
    except ImportError as exc:
        report.update(status='skipped', reason=str(exc))
        save()
        return by1skip.skip(str(exc))
    if not torch.cuda.is_available():
        report.update(status='skipped', reason='CUDA unavailable')
        save()
        return by1skip.skip('No CUDA: benchmark not executed')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report['environment'] = dict(torch=torch.__version__, transformers=T.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    report['source_sha256'] = {p: hashlib.sha256(pathlib.Path(by1paths.ROOT, p).read_bytes()).hexdigest()
                              for p in ['src/checks/by1gpubench.py', 'src/by1full.py',
                                        'src/modelcheck/by1quant.py', 'src/modelcheck/by1shadow.py']}
    selected = [(f, r) for f, r in MODELS if args.model is None or r in args.model]
    if args.checkpoint and len(selected) != 1:
        return by1skip.caller('--checkpoint requires selecting exactly one --model / '
                              '--checkpoint 必须同时且只指定一个 --model')
    if args.quant_contract and len(selected) != 1:
        return by1skip.caller('--quant-contract requires selecting exactly one --model / '
                              '--quant-contract 必须同时且只指定一个 --model')
    if args.revision and len(selected) != 1:
        return by1skip.caller('--revision requires selecting exactly one --model / '
                              '--revision 必须同时且只指定一个 --model')
    for filename, default_repo in selected:
        repo = args.checkpoint or default_repo
        started = time.monotonic()
        row = {'model': filename, 'status': 'running', 'phase': 'download'}
        report['models'].append(row)
        save()
        print('START ' + repo, flush=True)
        try:
            benchmark(filename, repo, args, torch, T, hub, work, save, row)
            row['status'] = 'passed'
            print('[PASS] ' + repo, flush=True)
        except by1shadow.ShadowSkip as exc:
            # 影子模式做不了 ≠ 失败也 ≠ 验过。单列一行，报告里单列一格。
            row.update(status='skipped', reason=str(exc))
            print('%s %s: %s' % (by1skip.MARK, repo, exc), flush=True)
        except Exception as exc:
            row.update(status='failed', error=str(exc))
            print('[FAIL] %s: %s' % (repo, exc), flush=True)
        row['seconds'] = time.monotonic() - started
        save()
        gc.collect()
        torch.cuda.empty_cache()
    failed = [r['model'] for r in report['models'] if r['status'] == 'failed']
    skipped = [r['model'] for r in report['models'] if r['status'] == 'skipped']
    passed = [r['model'] for r in report['models'] if r['status'] == 'passed']
    report['status'] = 'failed' if failed else ('skipped' if not passed else 'passed')
    if skipped:
        report['skipped'] = skipped
    save()
    print('[%s] Real checkpoint benchmark: %d models, %d failures, %d skipped' %
          ('FAIL' if failed else 'PASS', len(selected), len(failed), len(skipped)), flush=True)
    if failed:
        return 1
    if skipped and not passed:
        return by1skip.skip('全部模型在影子模式下跳过（%s）；没有验过任何一项' % ', '.join(skipped))
    return 0


if __name__ == '__main__':
    _sys.exit(main())
