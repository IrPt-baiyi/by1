#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1gputrain -- 用真实 Qwen3-0.6B 权重对比 CUDA 反向传播和一次更新。

`--shadow`：影子权重（真格式随机值）—— 梯度与 AdamW 一步的对比是接线级的，
不需要真值，0 权重下载；报告标 `weights=synthetic`，**不替代真权重结论**。
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401

import datetime
import gc
import json
import math
import pathlib

import by1check
import by1full
import by1skip


MODEL_FILE = 'models/complete-Qwen3-0.6B.by1'
REPO = 'Qwen/Qwen3-0.6B'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def resolve_shadow_config(args):
    """影子 config **只认真品**：--shadow-config > refs/ > 只拉 config
    一个文件（几 KB）。声明合成的 config 会让声明对拍空转（by1shadow 头注）。"""
    if args.shadow_config:
        return pathlib.Path(args.shadow_config).read_text(encoding='utf-8')
    for name in (REPO.replace('/', '__') + '.config.json',
                 REPO.replace('/', '_') + '.config.json'):
        candidate = pathlib.Path(by1paths.REFS, name)
        if candidate.exists():
            return candidate.read_text(encoding='utf-8')
    import by1shadow
    return pathlib.Path(by1shadow.fetch_config(REPO)).read_text(encoding='utf-8')


def snapshot_step(model, torch, ids, labels, lr):
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)
    output = model(input_ids=ids, labels=labels, use_cache=False)
    require(torch.isfinite(output.loss).item(), 'Training loss is not finite')
    loss = float(output.loss.detach())
    logits = output.logits.detach().cpu().clone()
    output.loss.backward()

    gradients = {}
    grad_square = 0.0
    grad_count = 0
    watched_name, watched_before = None, None
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        require(grad is not None, 'Missing gradient for ' + name)
        require(torch.isfinite(grad).all().item(), 'Non-finite gradient for ' + name)
        cpu_grad = grad.detach().cpu().clone()
        gradients[name] = cpu_grad
        grad_square += float(cpu_grad.float().square().sum())
        grad_count += cpu_grad.numel()
        if name.endswith('layers.0.self_attn.q_proj.weight'):
            watched_name = name
            watched_before = parameter.detach().cpu().clone()
    require(watched_name is not None, 'Could not find the first Qwen3 query projection')
    require(grad_square > 0, 'All model gradients are zero')

    optimizer.step()
    watched_after = dict(model.named_parameters())[watched_name].detach().cpu()
    watch_delta = (watched_after.float() - watched_before.float()).abs().max().item()
    require(watch_delta > 0, 'AdamW did not update the watched query projection')

    updated = {name: parameter.detach().cpu().clone()
               for name, parameter in model.named_parameters()}
    peak_allocated = (torch.cuda.max_memory_allocated()
                      if torch.cuda.is_available() else None)
    del output, optimizer
    return {'loss': loss, 'logits': logits, 'gradients': gradients, 'updated': updated,
            'gradient_l2': math.sqrt(grad_square), 'gradient_values': grad_count,
            'watched_parameter': watched_name, 'watched_parameter_max_update': watch_delta,
            'peak_allocated_bytes': peak_allocated}


def compare_maps(reference, candidate, label, relative_l2_tolerance=2e-3,
                 relative_peak_tolerance=2e-2):
    require(reference.keys() == candidate.keys(), label + ' parameter names differ')
    error_sq, reference_sq, error_peak, reference_peak = 0.0, 0.0, 0.0, 0.0
    compared = 0
    for name in reference:
        ref, got = reference[name].reshape(-1), candidate[name].reshape(-1)
        require(ref.shape == got.shape, label + ' shape differs for ' + name)
        for start in range(0, ref.numel(), 1_000_000):
            r = ref[start:start + 1_000_000].float()
            g = got[start:start + 1_000_000].float()
            diff = (g - r).abs()
            error_sq += float(diff.double().square().sum())
            reference_sq += float(r.double().square().sum())
            error_peak = max(error_peak, float(diff.max()))
            reference_peak = max(reference_peak, float(r.abs().max()))
            compared += r.numel()
    relative_l2 = math.sqrt(error_sq) / max(math.sqrt(reference_sq), 1e-12)
    relative_peak = error_peak / max(reference_peak, 1e-12)
    require(relative_l2 <= relative_l2_tolerance and relative_peak <= relative_peak_tolerance,
            '%s differs: relative_l2=%.3e, relative_peak=%.3e' %
            (label, relative_l2, relative_peak))
    return {'values': compared, 'max_absolute': error_peak,
            'relative_l2': relative_l2, 'relative_to_peak': relative_peak}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='cgen/gpu-real-traincheck.json')
    parser.add_argument('--length', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--shadow', action='store_true',
                        help='影子权重（真格式随机值）：训练步对拍不需要真值，'
                             '0 权重下载；报告标 weights=synthetic')
    parser.add_argument('--shadow-config',
                        help='影子用的真品 config.json（默认 refs/，再不行只拉 config）')
    parser.add_argument('--shadow-seed', type=int, default=20261013)
    # **CPU 腿（2026-10-16）**：原来这里第一条就是 `if not cuda: skip`，
    # 于是"真权重训练步"这条判据在本机（无显卡）永远只显示"跳过"。
    # 内存账：0.6B fp32 全参数一步 = params 2.4 + grads 2.4 + AdamW 两态 4.8
    # ≈ 9.6 GB（本机可用 10.3 GB，太紧）；**bf16 减半 ≈ 4.8 GB**，放得下。
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='auto = 有卡用卡、没卡用 CPU')
    parser.add_argument('--dtype', default=None, choices=['bfloat16', 'float32'],
                        help='不给：cuda -> bfloat16；cpu -> float32（CPU 想省内存给 bfloat16）')
    parser.add_argument('--checkpoint',
                        help='本地 HF checkpoint 目录。不给：按 cgen/gpu-benchmark.json 的钉住 revision 现下')
    args = parser.parse_args(argv)
    target = pathlib.Path(args.report)
    report = {'schema': 1, 'status': 'running', 'started_at':
              datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'repo': REPO, 'model_file': MODEL_FILE, 'dtype': 'bfloat16',
              'batch_size': 1, 'sequence_length': args.length, 'optimizer': 'AdamW',
              'learning_rate': args.lr,
              'weights': 'synthetic' if args.shadow else 'published',
              'gradient_relative_l2_tolerance': 2e-3,
              'gradient_relative_peak_tolerance': 2e-2,
              'parameter_update_relative_l2_tolerance': 2e-3,
              'parameter_update_relative_peak_tolerance': 2e-2}

    def save():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    try:
        import torch
        import transformers as T
        import huggingface_hub as hub
        if args.device == 'cuda' and not torch.cuda.is_available():
            report.update(status='skipped', reason='CUDA requested but unavailable')
            save()
            return by1skip.skip('CUDA requested but unavailable; no real-weight training step was run')
        device = 'cuda' if (args.device == 'cuda'
                            or (args.device == 'auto' and torch.cuda.is_available())) else 'cpu'
        dtype = getattr(torch, args.dtype or ('bfloat16' if device == 'cuda' else 'float32'))
        require(args.length >= 2 and args.lr > 0, 'Length and learning rate must be positive')
        torch.set_num_threads(4)
        if device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats()
        report['dtype'] = str(dtype).replace('torch.', '')
        report['device'] = device
        report['environment'] = {'torch': torch.__version__, 'transformers': T.__version__,
                                 'device': device,
                                 'cuda': torch.version.cuda if device == 'cuda' else None,
                                 'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
                                 'gpu_memory_bytes': (torch.cuda.get_device_properties(0).total_memory
                                                      if device == 'cuda' else None)}

        check, info = by1check.check(MODEL_FILE)
        require(not check.count(by1check.E), str(check.items))
        family, config, mtp = by1full.compile_config(info)
        require(family == 'qwen3' and not mtp, 'This real-weight training check expects plain Qwen3')
        if args.shadow:
            # **训练步对拍不需要真值** —— 梯度和 AdamW 一步的对比是接线级的，
            # 随机值权重一样抓。0 权重下载；量化存储不适用（本模型无）。
            import by1shadow
            config_text = resolve_shadow_config(args)
            shadow_dir = target.parent / 'shadow-train-qwen3-0.6B'
            if by1shadow.reusable(shadow_dir, family, mtp, args.shadow_seed,
                                  'bfloat16', config_text):
                manifest = json.loads(
                    (shadow_dir / by1shadow.MANIFEST).read_text(encoding='utf-8'))
            else:
                manifest = by1shadow.build(config_text, family, mtp, shadow_dir,
                                           seed=args.shadow_seed, dtype='bfloat16')
            checkpoint = str(shadow_dir)
            report.update(revision='synthetic', checkpoint_bytes=manifest['bytes'],
                          download_endpoint='none', shadow=manifest)
        else:
            if args.checkpoint:
                checkpoint = args.checkpoint
                report.update(revision='local:' + pathlib.Path(checkpoint).name,
                              download_endpoint='none')
            else:
                benchmark_report = json.loads(
                    pathlib.Path('cgen/gpu-benchmark.json').read_text(encoding='utf-8'))
                row = next(item for item in benchmark_report['models']
                           if item.get('repo') == REPO)
                revision = row['revision']
                checkpoint = hub.snapshot_download(REPO, revision=revision,
                    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model',
                                    '*.jinja', '*.tiktoken'],
                    max_workers=4)
                report.update(revision=revision, checkpoint_bytes=row['checkpoint_bytes'],
                              download_endpoint=row['download_endpoint'])
        cfg = T.AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        generator = torch.Generator().manual_seed(20261013)
        ids = torch.randint(0, min(cfg.vocab_size, 10000), (1, args.length), generator=generator,
                            dtype=torch.long).to(device)
        labels = ids.clone()
        kwargs = {'dtype': dtype, 'device_map': {'': device},
                  'attn_implementation': 'eager', 'local_files_only': True}

        report['phase'] = 'original-step'
        save()
        original, loading = T.AutoModelForCausalLM.from_pretrained(
            checkpoint, output_loading_info=True, **kwargs)
        problems = {key: loading.get(key) for key in
                    ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')
                    if loading.get(key)}
        require(not problems, 'Official model load was incomplete: ' + str(problems))
        reference = snapshot_step(original, torch, ids, labels, args.lr)
        report['official'] = {key: reference[key] for key in
                              ('loss', 'gradient_l2', 'gradient_values',
                               'watched_parameter', 'watched_parameter_max_update',
                               'peak_allocated_bytes')}
        del original
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

        report['phase'] = 'by1-step'
        save()
        candidate = by1full.load_model(family, config, mtp, checkpoint, **kwargs).eval()
        actual = snapshot_step(candidate, torch, ids, labels, args.lr)
        report['by1'] = {key: actual[key] for key in
                         ('loss', 'gradient_l2', 'gradient_values',
                          'watched_parameter', 'watched_parameter_max_update',
                          'peak_allocated_bytes')}
        logits_diff = (actual['logits'].float() - reference['logits'].float()).abs()
        report['forward'] = {'max_absolute': logits_diff.max().item(),
                             'allclose': torch.allclose(actual['logits'].float(), reference['logits'].float(),
                                                        atol=2e-3, rtol=2e-3)}
        require(report['forward']['allclose'], 'Training logits differ from official model')
        require(abs(actual['loss'] - reference['loss']) <= 2e-3,
                'Training loss differs from official model')
        report['gradients'] = compare_maps(reference['gradients'], actual['gradients'],
            'Full-model gradients', report['gradient_relative_l2_tolerance'],
            report['gradient_relative_peak_tolerance'])
        report['updated_parameters'] = compare_maps(reference['updated'], actual['updated'],
            'AdamW-updated parameters', report['parameter_update_relative_l2_tolerance'],
            report['parameter_update_relative_peak_tolerance'])
        report['status'] = 'passed'
        report['phase'] = 'complete'
    except ImportError as exc:
        report.update(status='skipped', reason=str(exc))
        save()
        return by1skip.skip('GPU training dependencies unavailable: ' + str(exc))
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        save()
        print('[FAIL] Real-weight GPU train step: ' + str(exc), flush=True)
        return 1
    save()
    if report.get('weights') == 'synthetic':
        print('[PASS] Qwen3-0.6B **影子权重** BF16 backward + AdamW step; '
              'full gradients and updated parameters match the official model'
              '（接线对拍，非真权重结论）', flush=True)
    else:
        print('[PASS] Qwen3-0.6B real-weight BF16 backward + AdamW step; '
              'full gradients and updated parameters match the official model', flush=True)
    return 0


if __name__ == '__main__':
    _sys.exit(main())
