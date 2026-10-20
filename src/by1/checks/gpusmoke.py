#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1gpusmoke -- CUDA 前向、梯度、参数更新与完整视觉/MTP 分支的可复现验收。"""

import os as _os
import sys as _sys
# 引导（生成的，别手改）：往上找到有 by1paths.py 的那一层 —— 迁移后
# 模块在包内深处（src/by1/checks/…）—— 找**包所在的那一层**（src/）。
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p and not _os.path.isdir(_os.path.join(_p, 'by1')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import by1  # noqa: E402,F401
import by1paths  # noqa: E402,F401
import os as _os
import sys as _sys
# 引导（生成的，别手改）：往上找到有 by1paths.py 的那一层 —— 迁移后
# 模块在包内深处（src/by1/checks/…）—— 找**包所在的那一层**（src/）。
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p and not _os.path.isdir(_os.path.join(_p, 'by1')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import by1paths  # noqa: E402,F401

import argparse
import datetime
import hashlib
import json
import pathlib
import tempfile
import time

import by1codegen
import by1full
import by1skip
from by1.checks import complete as fixtures


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def compare(actual, expected, torch, label, atol, rtol):
    a = actual.detach().float().cpu()
    b = expected.detach().float().cpu()
    require(a.shape == b.shape, label + ': shape mismatch')
    require(torch.isfinite(a).all().item() and torch.isfinite(b).all().item(), label + ': NaN/Inf')
    absolute = (a - b).abs().max().item()
    relative = absolute / max(b.abs().max().item(), 1e-12)
    require(torch.allclose(a, b, atol=atol, rtol=rtol),
            '%s: abs=%.3e, relative_peak=%.3e' % (label, absolute, relative))
    return {'max_absolute': absolute, 'relative_to_peak': relative}


def gradients(model, reference, torch, atol, rtol, exclude_mtp=False, reduced_precision=False):
    other = dict(reference.named_parameters())
    count, maximum = 0, 0.0
    for name, param in model.named_parameters():
        if exclude_mtp and name.startswith('mtp.'):
            continue
        require(name in other, 'Reference missing parameter: ' + name)
        require(param.grad is not None and other[name].grad is not None, 'Missing gradient: ' + name)
        if reduced_precision:
            a, b = param.grad.detach().float().cpu(), other[name].grad.detach().float().cpu()
            require(torch.isfinite(a).all().item() and torch.isfinite(b).all().item(), 'Non-finite gradient: ' + name)
            delta = a - b
            absolute = delta.abs().max().item()
            peak = absolute / max(b.abs().max().item(), 1e-12)
            relative_l2 = delta.norm().item() / max(b.norm().item(), 1e-12)
            # BF16 rounding near zero makes elementwise relative error misleading.
            # Keep both the vector norm and peak error bounded instead.
            require(absolute <= atol or (peak <= rtol and relative_l2 <= rtol),
                    'gradient %s: abs=%.3e peak=%.3e l2=%.3e' % (name, absolute, peak, relative_l2))
            error = {'max_absolute': absolute}
        else:
            error = compare(param.grad, other[name].grad, torch, 'gradient ' + name, atol, rtol)
        maximum = max(maximum, error['max_absolute'])
        count += 1
    require(count > 0, 'No gradients compared')
    return {'compared_tensors': count, 'max_absolute': maximum}


def device_check(model, tensors, device):
    require(all(p.device == device for p in model.parameters()), 'Parameter did not move to requested device')
    require(all(b.device == device for b in model.buffers()), 'Buffer did not move to requested device')
    require(all(v.device == device for v in tensors.values() if hasattr(v, 'device')), 'Input on wrong device')


def update(model, torch, watched):
    params = dict(model.named_parameters())
    for name in watched:
        require(name in params, 'Update probe missing: ' + name)
        g = params[name].grad
        require(g is not None and torch.isfinite(g).all().item() and g.abs().max().item() > 0,
                'No finite nonzero gradient: ' + name)
    before = {name: params[name].detach().clone() for name in watched}
    for name, param in params.items():
        require(param.grad is not None and torch.isfinite(param.grad).all().item(), 'Invalid gradient: ' + name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    for name in watched:
        require(not torch.equal(before[name], params[name]), 'Optimizer did not update: ' + name)
        require(torch.isfinite(params[name]).all().item(), 'Optimizer produced NaN/Inf: ' + name)
    return watched


def native(torch, device, dtype, tolerances):
    import by1exec
    info = fixtures.description('hello.by1')
    ir = by1codegen.compile_ir(info)
    ns = {}
    exec(compile(by1codegen.render_ir(ir, 'hello.by1'), '<gpu-hello>', 'exec'), ns)
    torch.manual_seed(17)
    cpu = ns['build']().eval()
    gpu = ns['build']().to(device=device, dtype=dtype).eval()
    gpu.load_state_dict(cpu.state_dict(), strict=True)
    ids = torch.arange(8)[None]
    gpu_ids = ids.to(device)
    device_check(gpu, {'input_ids': gpu_ids}, device)
    expected = cpu(ids)
    actual = gpu(gpu_ids)
    require(actual.device == device, 'Output on wrong device')
    result = {'cpu_forward': compare(actual, expected, torch, 'native CPU', *tolerances[:2])}
    # The NumPy reference does not share the generated PyTorch operator code.
    shapes = by1exec.shapes_of(ir)
    state = cpu.state_dict()
    params = {}
    for key, tensor in state.items():
        candidate = key if key in shapes else key.removesuffix('.weight')
        require(candidate in shapes and tuple(tensor.shape) == tuple(shapes[candidate]), 'Unmapped tensor: ' + key)
        params[candidate] = tensor.detach().numpy().copy()
    require(set(params) == set(shapes), 'NumPy reference has missing weights')
    executor = by1exec.Exec(ir, params)
    executor.P['_g'] = {k: params[k] for k in ('embed.weight', 'head.weight', 'final_norm.w')}
    numpy_output = torch.from_numpy(executor(ids.numpy()))
    result['numpy_forward'] = compare(actual, numpy_output, torch, 'native NumPy', *tolerances[:2])
    loss = torch.nn.functional.cross_entropy(actual[:, :-1].float().flatten(0, 1), gpu_ids[:, 1:].flatten())
    reference_loss = torch.nn.functional.cross_entropy(expected[:, :-1].flatten(0, 1), ids[:, 1:].flatten())
    result['loss'] = compare(loss, reference_loss, torch, 'native loss', *tolerances[:2])
    loss.backward()
    reference_loss.backward()
    result['gradients'] = gradients(gpu, cpu, torch, *tolerances[2:], reduced_precision=dtype == torch.bfloat16)
    result['updated'] = update(gpu, torch, ['embed.weight', 'layers.0.op1.wq.weight', 'head.weight'])
    result['parameters'] = sum(p.numel() for p in gpu.parameters())
    result['output_shape'] = list(actual.shape)
    return result


def full(filename, torch, T, device, dtype, tolerances):
    family, cfg, mtp = fixtures.tiny(filename)
    torch.manual_seed(23)
    cpu = by1full.build_model(family, cfg, mtp).eval()
    gpu = by1full.build_model(family, cfg, mtp).to(device=device, dtype=dtype).eval()
    gpu.load_state_dict(cpu.state_dict(), strict=True)
    tied = cfg.get('text_config', cfg)['tie_word_embeddings']
    require((gpu.get_input_embeddings().weight is gpu.get_output_embeddings().weight) == tied, 'Incorrect weight tying')
    # Independent config construction, using the same HF operator implementation.
    ref = fixtures.reference(family, tied, torch, T).to(device=device, dtype=dtype).eval()
    ref.config._attn_implementation = 'eager'
    if family == 'qwen3_5':
        ref.config.text_config._attn_implementation = 'eager'
        ref.config.vision_config._attn_implementation = 'eager'
    ref.load_state_dict({k: v for k, v in gpu.state_dict().items() if not k.startswith('mtp.')}, strict=True)
    batch = fixtures.inputs(family, torch)
    batch['labels'] = batch['input_ids']
    moved = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
             if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    device_check(gpu, moved, device)
    expected = cpu(**batch)
    actual = gpu(**moved)
    independent = ref(**moved)
    require(actual.logits.device == device and actual.loss.device == device, 'Output/loss on wrong device')
    result = {'cpu_forward': compare(actual.logits, expected.logits, torch, 'CPU logits', *tolerances[:2]),
              'reference_forward': compare(actual.logits, independent.logits, torch, 'HF config logits', *tolerances[:2]),
              'loss': compare(actual.loss, expected.loss, torch, 'CPU loss', *tolerances[:2])}
    actual.loss.backward(retain_graph=bool(mtp))
    expected.loss.backward(retain_graph=bool(mtp))
    independent.loss.backward()
    result['cpu_gradients'] = gradients(gpu, cpu, torch, *tolerances[2:], exclude_mtp=True,
                                       reduced_precision=dtype == torch.bfloat16)
    result['reference_gradients'] = gradients(gpu, ref, torch, 3e-5, 2e-3, exclude_mtp=True)
    watched = [next(iter(dict(gpu.named_parameters())))]
    if mtp:
        require(gpu.mtp.embed_tokens is gpu.get_input_embeddings() and gpu.mtp.lm_head is gpu.get_output_embeddings(),
                'MTP sharing lost after device transfer')
        with torch.no_grad():
            video = fixtures.inputs(family, torch, video=True)
            video = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                     if isinstance(v, torch.Tensor) else v for k, v in video.items()}
            result['video_forward'] = compare(gpu(**video).logits, ref(**video).logits, torch, 'video logits', *tolerances[:2])
            changed = dict(moved)
            changed['pixel_values'] = moved['pixel_values'] * -10
            require(not torch.allclose(gpu(**changed).logits, actual.logits), 'Image has no effect on text logits')
        gpu.zero_grad(set_to_none=True)
        cpu.zero_grad(set_to_none=True)
        ids = moved['input_ids'].roll(-1, 1)
        positions = torch.arange(ids.shape[1], device=device)[None] + 1
        embeds = actual.hidden_states[0].roll(-1, 1)
        draft = gpu.forward_mtp(ids, actual.hidden_states[-1], positions, inputs_embeds=embeds, labels=ids)
        formula = fixtures.mtp_formula(gpu, ids, actual.hidden_states[-1], positions, torch, embeds)
        result['mtp_formula'] = compare(draft['logits'], formula, torch, 'MTP formula', *tolerances[:2])
        cpu_ids = batch['input_ids'].roll(-1, 1)
        cpu_draft = cpu.forward_mtp(cpu_ids, expected.hidden_states[-1], positions.cpu(),
                                   inputs_embeds=expected.hidden_states[0].roll(-1, 1), labels=cpu_ids)
        result['mtp_cpu_forward'] = compare(draft['logits'], cpu_draft['logits'], torch, 'MTP CPU', *tolerances[:2])
        draft['loss'].backward()
        cpu_draft['loss'].backward()
        result['mtp_cpu_gradients'] = gradients(gpu, cpu, torch, *tolerances[2:], reduced_precision=dtype == torch.bfloat16)
        watched = ['model.visual.patch_embed.proj.weight', 'mtp.fc.weight', 'lm_head.weight']
    result['updated'] = update(gpu, torch, watched)
    result['parameters'] = sum(p.numel() for p in gpu.parameters())
    result['output_shape'] = list(actual.logits.shape)
    if mtp:
        work = pathlib.Path(by1paths.ROOT, 'cgen')
        work.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='gpu-roundtrip-', dir=work) as tmp:
            gpu.save_pretrained(tmp)
            loaded = by1full.load_model(family, cfg, mtp, tmp, local_files_only=True).to(device=device, dtype=dtype).eval()
            require(loaded.mtp.embed_tokens is loaded.get_input_embeddings(), 'Reload lost MTP sharing')
            with torch.no_grad():
                base = gpu(**moved)
                again = loaded(**moved)
                result['checkpoint_forward'] = compare(again.logits, base.logits, torch, 'checkpoint text', *tolerances[:2])
                a = gpu.forward_mtp(ids, base.hidden_states[-1], positions)['logits']
                b = loaded.forward_mtp(ids, again.hidden_states[-1], positions)['logits']
                result['checkpoint_mtp'] = compare(b, a, torch, 'checkpoint MTP', *tolerances[:2])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='cpu 只检查验收器，不算 GPU 通过')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--report', default='cgen/gpu-smoke.json')
    args = parser.parse_args(argv)
    report = {'schema': 1, 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'requested_device': args.device, 'dtype': args.dtype, 'gpu_verified': False,
              'scope': 'random weights; native hello plus six scaled complete architectures; eager attention',
              'cases': [], 'source_sha256': {}}
    sources = ['src/checks/by1gpusmoke.py', 'src/checks/by1complete.py', 'src/by1full.py', 'src/by1codegen.py',
               'models/hello.by1'] + ['models/' + n for n in fixtures.GALLERY]
    for source in sources:
        report['source_sha256'][source] = hashlib.sha256(pathlib.Path(by1paths.ROOT, source).read_bytes()).hexdigest()
    def save():
        target = pathlib.Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    try:
        import torch
        import transformers as T
        import safetensors
    except ImportError as exc:
        report.update(status='skipped', reason=str(exc))
        save()
        return by1skip.skip('GPU 验收依赖缺失：' + str(exc))
    report['environment'] = {'torch': torch.__version__, 'transformers': T.__version__,
                             'safetensors': safetensors.__version__, 'cuda_runtime': torch.version.cuda}
    if args.device == 'cuda' and not torch.cuda.is_available():
        report.update(status='skipped', reason='CUDA unavailable')
        save()
        return by1skip.skip('CUDA 不可用；没有执行 GPU 验证')
    torch.set_num_threads(1)
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
    dtype = getattr(torch, args.dtype)
    if device.type == 'cuda':
        prop = torch.cuda.get_device_properties(device)
        report['environment'].update(gpu=prop.name, memory_bytes=prop.total_memory,
                                     compute_capability=list(torch.cuda.get_device_capability(device)))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            report.update(status='skipped', reason='BF16 unsupported')
            save()
            return by1skip.skip('设备不支持 BF16')
    tolerances = (3e-5, 5e-4, 3e-5, 2e-3) if dtype == torch.float32 else (2e-2, 8e-2, 2e-3, 3e-2)
    report['tolerances'] = dict(zip(['forward_atol', 'forward_rtol', 'gradient_atol', 'gradient_rtol'], tolerances))
    report['tolerances']['bf16_cpu_gradient_metric'] = 'both normalized L2 and relative peak; or max absolute <= gradient_atol'
    names = ['native-hello'] + fixtures.GALLERY
    for name in names:
        started = time.monotonic()
        row = {'name': name, 'status': 'running'}
        report['cases'].append(row)
        save()
        try:
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            row['metrics'] = native(torch, device, dtype, tolerances) if name == 'native-hello' else full(name, torch, T, device, dtype, tolerances)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                row['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            row['status'] = 'passed'
            print('  ok %s (%s, %s)' % (name, device, args.dtype), flush=True)
        except Exception as exc:
            row.update(status='failed', error=str(exc))
            print('  [FAIL] %s: %s' % (name, exc), flush=True)
        row['seconds'] = time.monotonic() - started
        save()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    failures = [r['name'] for r in report['cases'] if r['status'] != 'passed']
    report['status'] = 'failed' if failures else 'passed_gpu' if device.type == 'cuda' else 'passed_cpu_harness'
    report['gpu_verified'] = not failures and device.type == 'cuda'
    save()
    print('\n  [%s] %s: %d cases, %d failures; report %s' %
          ('FAIL' if failures else 'PASS', report['status'], len(names), len(failures), args.report), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    _sys.exit(main())
