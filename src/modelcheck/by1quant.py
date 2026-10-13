#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quantized checkpoint storage contracts used by emit and manifest checks.

These describe checkpoint storage, not model architecture or a runnable kernel.
"""

import re
import math
import pathlib


FORMATS = {
    "mxfp4": {
        "layout": "packed_blocks",
        "weight_dtype": "U8",
        "scale_dtype": "U8",
        "bias_dtype": "BF16",
        "scale_field": None,
        "fixed_block": 32,
        "fuse": True,
        "weight_shape": "[..., out, in/32, 16]",
        "scale_shape": "[..., out, in/32]",
        "scale_meaning": "packed E8M0 block scales",
    },
    "fp8_e4m3_block": {
        "layout": "weight_and_scale",
        "weight_dtype": "F8_E4M3",
        "scale_dtype": "F32",
        "scale_dtypes": ("F32", "BF16"),
        "bias_dtype": None,
        "scale_field": "required",
        "fixed_block": None,
        "fuse": False,
        "weight_shape": "unchanged",
        "scale_shape": "ceil(each weight dimension / block)",
        "scale_meaning": "inverse scale per block",
    },
}

_IDENTIFIER = re.compile(r"[A-Za-z_][\w.]*\Z")
_TENSOR_LIST = re.compile(
    r"\[\s*[A-Za-z_][\w.]*(?:\s*,\s*[A-Za-z_][\w.]*)*\s*\]\Z")
_MXFP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def decode_mxfp4_element(code):
    """Decode one OCP E2M1 element code (the surrounding byte packing is separate)."""
    if not isinstance(code, int) or not 0 <= code < 16:
        raise ValueError("MXFP4 element code must be an integer from 0 to 15")
    return _MXFP4_VALUES[code]


def decode_e8m0_scale(code):
    """Decode an OCP E8M0 byte to its power-of-two scale; 255 is the NaN code."""
    if not isinstance(code, int) or not 0 <= code <= 255:
        raise ValueError("E8M0 scale code must be an integer from 0 to 255")
    return math.nan if code == 255 else math.ldexp(1.0, code - 127)


def mxfp4_block_dot(left, right, left_scale, right_scale):
    """OCP MX block dot-product oracle for equal-length E2M1 code sequences."""
    if len(left) != 32 or len(right) != 32:
        raise ValueError("MXFP4 dot-product blocks must each contain 32 elements")
    return (left_scale * right_scale
            * sum(decode_mxfp4_element(a) * decode_mxfp4_element(b)
                  for a, b in zip(left, right)))


def decode_fp8_e4m3(code):
    """Decode one OCP FP8 E4M3 byte, including subnormals and its NaN code."""
    if not isinstance(code, int) or not 0 <= code <= 255:
        raise ValueError("FP8 E4M3 code must be an integer from 0 to 255")
    sign = -1.0 if code & 0x80 else 1.0
    exponent, mantissa = (code >> 3) & 0x0f, code & 0x07
    if exponent == 0x0f and mantissa == 0x07:
        return math.nan
    if exponent == 0:
        value = math.ldexp(mantissa / 8.0, 1 - 7)
    else:
        value = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return math.copysign(value, sign)


def dequant_fp8_e4m3(code, scale_inv):
    """Apply the checkpoint's ``weight_scale_inv`` convention to one E4M3 byte."""
    return decode_fp8_e4m3(code) * float(scale_inv)


def _list_names(value):
    """Return names from a validated list expression, else an empty list."""
    if not _TENSOR_LIST.fullmatch(str(value).strip()):
        return []
    return re.findall(r"[A-Za-z_][\w.]*", str(value).strip()[1:-1])


def compile_emit_quant(quant):
    """Normalize the lowered emit declaration into a serializable runtime spec."""
    errors = validate_emit_quant(quant)
    if errors:
        raise ValueError("; ".join(errors))
    result = {
        "format": str(quant["format"]).strip().strip('"'),
        "block": int(str(quant["block"]).strip().strip('"')),
        "tensor": _list_names(quant["tensor"]),
    }
    scale = str(quant.get("scale", "")).strip().strip('"')
    if scale:
        result["scale"] = scale
    scale_dtype = str(quant.get("scale_dtype", "")).strip().strip('"').upper()
    if scale_dtype:
        result["scale_dtype"] = scale_dtype
    fuses = quant.get("fuse") or {}
    if fuses:
        result["fuse"] = {name: _list_names(names) for name, names in fuses.items()}
    return result


def _config_mapping(config):
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "to_dict"):
        result = config.to_dict()
        return dict(result) if isinstance(result, dict) else {}
    return {}


def _enum_value(value):
    return getattr(value, "value", value)


def _scope_counts(names):
    counts = {"text_or_other": 0, "visual": 0, "mtp": 0}
    for name in names:
        lower = name.lower()
        if lower == "visual" or ".visual." in lower or lower.startswith("visual."):
            counts["visual"] += 1
        elif lower == "mtp" or ".mtp." in lower or lower.startswith("mtp."):
            counts["mtp"] += 1
        else:
            counts["text_or_other"] += 1
    return counts


def make_transformers_config(quant, checkpoint_quantization_config, transformers):
    """Build a strict native Transformers quantizer config for a declared format.

    Native config metadata such as ``modules_to_not_convert`` comes from the
    checkpoint. By1's declaration must agree on the format and block size;
    otherwise refusing the load is safer than silently changing precision.
    """
    metadata = _config_mapping(checkpoint_quantization_config)
    quant_method = str(_enum_value(metadata.get("quant_method", ""))).lower()
    modules = metadata.get("modules_to_not_convert") or None

    if quant["format"] == "mxfp4":
        if quant_method != "mxfp4":
            raise ValueError(".by1 声明 mxfp4，但 checkpoint quantization_config 不是 mxfp4")
        config_class = getattr(transformers, "Mxfp4Config", None)
        if config_class is None:
            raise ValueError("当前 Transformers 没有 Mxfp4Config")
        return config_class(dequantize=False, modules_to_not_convert=modules)

    if quant["format"] == "fp8_e4m3_block":
        if quant_method not in ("fp8", "finegrained_fp8"):
            raise ValueError(".by1 声明块 FP8，但 checkpoint quantization_config 不是 FP8")
        checkpoint_block = metadata.get("weight_block_size")
        expected_block = [quant["block"], quant["block"]]
        if checkpoint_block is not None and list(checkpoint_block) != expected_block:
            raise ValueError("FP8 block 与 checkpoint 不一致：%s vs %s"
                             % (checkpoint_block, expected_block))
        fmt = str(_enum_value(metadata.get("fmt", "e4m3"))).lower()
        if fmt not in ("e4m3", "e4m3fn"):
            raise ValueError("当前声明只支持 E4M3，checkpoint fmt=%s" % fmt)
        config_class = getattr(transformers, "FineGrainedFP8Config", None)
        if config_class is None:
            raise ValueError("当前 Transformers 没有 FineGrainedFP8Config")
        return config_class(
            activation_scheme=metadata.get("activation_scheme", "dynamic"),
            weight_block_size=tuple(expected_block), dequantize=False,
            modules_to_not_convert=modules,
            scale_fmt=metadata.get("scale_fmt", "float"),
        )

    raise ValueError("没有可运行的 Transformers 量化适配器：%s" % quant["format"])


def audit_runtime_storage(model, quant, torch):
    """Fail if native loading discarded the declared quantized representation."""
    state = model.state_dict()
    if quant["format"] == "mxfp4":
        packed = {name[:-7]: value for name, value in state.items()
                  if name.endswith("_blocks")}
        scales = {name[:-7]: value for name, value in state.items()
                  if name.endswith("_scales")}
        if not packed or packed.keys() != scales.keys():
            raise ValueError("MXFP4 加载后缺少成对的 _blocks/_scales")
        wrong = [(name, str(value.dtype)) for name, value in
                 list(packed.items()) + list(scales.items())
                 if value.dtype != torch.uint8]
        if wrong:
            raise ValueError("MXFP4 打包权重/scale 必须保留为 U8：%s" % wrong[:4])
        return {"format": "mxfp4", "packed_tensor_count": len(packed),
                "dtype": "uint8", "scope_tensor_counts": _scope_counts(packed)}

    if quant["format"] == "fp8_e4m3_block":
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            raise ValueError("当前 PyTorch 没有 float8_e4m3fn dtype")
        weights = {name[:-7]: value for name, value in state.items()
                   if name.endswith(".weight") and value.dtype == fp8_dtype}
        scales = {name[:-len("weight_scale_inv")].rstrip("."): value
                  for name, value in state.items()
                  if name.endswith("weight_scale_inv")}
        if not weights or weights.keys() != scales.keys():
            raise ValueError("FP8 加载后必须保留成对的 E4M3 权重和 weight_scale_inv")
        expected_scale_dtype = quant.get("scale_dtype", "F32").upper()
        torch_scale_dtype = {"F32": torch.float32, "BF16": torch.bfloat16}.get(
            expected_scale_dtype)
        if torch_scale_dtype is None:
            raise ValueError("不支持的 FP8 scale dtype: %s" % expected_scale_dtype)
        allowed_runtime_scale_dtypes = {torch.float32, torch.bfloat16}
        wrong = [(name, str(value.dtype)) for name, value in scales.items()
                 if value.dtype not in allowed_runtime_scale_dtypes]
        if wrong:
            raise ValueError("FP8 inverse scales 运行时只接受 F32/BF16：%s" % wrong[:4])
        scale_dtype_names = {
            torch.float32: "F32",
            torch.bfloat16: "BF16",
        }
        return {"format": "fp8_e4m3_block", "weight_tensor_count": len(weights),
                "scale_tensor_count": len(scales), "weight_dtype": "float8_e4m3fn",
                "contract_scale_dtype": expected_scale_dtype,
                "scale_dtypes": sorted({scale_dtype_names.get(value.dtype, str(value.dtype))
                                         for value in scales.values()}),
                "scope_tensor_counts": _scope_counts(weights)}

    raise ValueError("没有运行时存储审计器：%s" % quant["format"])


def audit_checkpoint_storage(checkpoint_dir, quant, expected_weight_count=None):
    """Inspect safetensors headers and verify raw quantized names, dtypes, and blocks."""
    if quant.get("format") != "fp8_e4m3_block":
        raise ValueError("checkpoint 头审计当前只支持 fp8_e4m3_block")
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ValueError("读取 checkpoint safetensors 头需要 safetensors") from exc

    files = sorted(pathlib.Path(checkpoint_dir).glob("*.safetensors"))
    if not files:
        raise ValueError("checkpoint 目录里没有 safetensors 分片")
    logical_names = tuple(str(name).strip() for name in quant.get("tensor", ()))
    scale_suffix = str(quant.get("scale", "")).strip()
    block = int(quant.get("block", 0))
    expected_scale_dtype = str(quant.get("scale_dtype", "F32")).upper()
    if not logical_names or not scale_suffix or block <= 0:
        raise ValueError("FP8 checkpoint 头审计缺少 tensor/scale/block 合同")
    expected_scale_safe_dtype = {"F32": "F32", "BF16": "BF16"}.get(
        expected_scale_dtype)
    if expected_scale_safe_dtype is None:
        raise ValueError("不支持的 FP8 checkpoint scale dtype: %s" % expected_scale_dtype)

    metadata = {}
    weight_dtypes, scale_dtypes = set(), set()
    shard_count = 0
    for filename in files:
        with safe_open(str(filename), framework="pt", device="cpu") as shard:
            shard_count += 1
            for name in shard.keys():
                is_weight = (name.endswith(".weight") and any(
                    name.endswith("." + logical) for logical in logical_names))
                paired_weight = name[:-len(scale_suffix)] + "weight" \
                    if name.endswith("." + scale_suffix) else ""
                is_scale = bool(paired_weight) and any(
                    paired_weight.endswith("." + logical) for logical in logical_names)
                if not (is_weight or is_scale):
                    continue
                if name in metadata:
                    raise ValueError("checkpoint 分片重复包含张量: %s" % name)
                tensor = shard.get_slice(name)
                metadata[name] = (str(tensor.get_dtype()).upper(),
                                  list(tensor.get_shape()))

    weights = scales = 0
    for name, (weight_dtype, weight_shape) in metadata.items():
        if not name.endswith(".weight"):
            continue
        scale_name = name[:-len("weight")] + scale_suffix
        if scale_name not in metadata:
            raise ValueError("FP8 权重缺少配对 scale: %s" % name)
        scale_dtype, scale_shape = metadata[scale_name]
        if weight_dtype not in ("F8_E4M3", "F8_E4M3FN"):
            raise ValueError("量化权重必须是 E4M3，%s 实际为 %s"
                             % (name, weight_dtype))
        if scale_dtype != expected_scale_safe_dtype:
            raise ValueError("%s scale dtype 契约为 %s，实际为 %s"
                             % (scale_name, expected_scale_safe_dtype, scale_dtype))
        if len(weight_shape) != 2:
            raise ValueError("FP8 block 权重必须为二维: %s %s"
                             % (name, weight_shape))
        expected_shape = [(dim + block - 1) // block for dim in weight_shape]
        if scale_shape != expected_shape:
            raise ValueError("%s scale shape 应为 %s，实际为 %s"
                             % (scale_name, expected_shape, scale_shape))
        weights += 1
        scales += 1
        weight_dtypes.add(weight_dtype)
        scale_dtypes.add(scale_dtype)

    raw_scale_count = sum(1 for name in metadata if name.endswith("." + scale_suffix))
    if not weights or weights != scales or scales != raw_scale_count:
        raise ValueError("FP8 checkpoint 权重/scale 数量不一致: %d/%d"
                         % (weights, raw_scale_count))
    if expected_weight_count is not None and weights != expected_weight_count:
        raise ValueError("FP8 checkpoint 应有 %d 对量化张量，实际 %d"
                         % (expected_weight_count, weights))
    return {"format": "fp8_e4m3_block", "shard_count": shard_count,
            "weight_tensor_count": weights, "scale_tensor_count": scales,
            "weight_dtypes": sorted(weight_dtypes),
            "scale_dtypes": sorted(scale_dtypes), "block": block}


def audit_shared_weights(model, tie_word_embeddings=False, has_mtp=False):
    """Verify storage aliases required by the loaded model declaration."""
    if tie_word_embeddings:
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        shared = input_weight is output_weight
        if (not shared and getattr(input_weight, "device", None) is not None
                and input_weight.device == getattr(output_weight, "device", None)):
            try:
                shared = (input_weight.untyped_storage().data_ptr() ==
                          output_weight.untyped_storage().data_ptr())
            except (AttributeError, RuntimeError):
                shared = False
        if not shared:
            raise ValueError("量化加载拆开了 input embedding/lm_head 的共享权重")
    if has_mtp:
        if model.mtp.embed_tokens is not model.get_input_embeddings():
            raise ValueError("量化加载拆开了 MTP/input embedding 的共享关系")
        if model.mtp.lm_head is not model.get_output_embeddings():
            raise ValueError("量化加载拆开了 MTP/lm_head 的共享关系")
    return {"tied_word_embeddings": bool(tie_word_embeddings), "mtp_aliases": bool(has_mtp)}


def validate_emit_quant(quant):
    """Return diagnostics for a lowered ``emit ... quant {}`` rule."""
    errors = []
    if not isinstance(quant, dict):
        return ["quant 必须是字段映射"]

    allowed = {"format", "block", "tensor", "scale", "scale_dtype", "fuse"}
    extra = sorted(set(quant) - allowed)
    if extra:
        errors.append("quant 含未支持字段：" + ", ".join(extra))

    fmt = str(quant.get("format", "")).strip().strip('"')
    spec = FORMATS.get(fmt)
    if spec is None:
        errors.append("quant.format 必须是 " + " 或 ".join(sorted(FORMATS)))

    raw_block = str(quant.get("block", "")).strip().strip('"')
    block = int(raw_block) if raw_block.isdigit() else 0
    if block <= 0:
        errors.append("quant.block 必须是正整数")
    elif spec and spec["fixed_block"] and block != spec["fixed_block"]:
        errors.append("mxfp4 的 block 固定为 32")

    tensor_names = _list_names(quant.get("tensor", ""))
    if not tensor_names:
        errors.append("quant.tensor 必须是非空张量名列表，例如 [gate_proj.weight]")
    elif len(tensor_names) != len(set(tensor_names)):
        errors.append("quant.tensor 不能重复声明同一个张量")

    scale = str(quant.get("scale", "")).strip().strip('"')
    scale_dtype = str(quant.get("scale_dtype", "")).strip().strip('"').upper()
    if scale and not _IDENTIFIER.fullmatch(scale):
        errors.append("quant.scale 必须是张量名")
    if spec:
        if spec["scale_field"] is None and scale:
            errors.append("mxfp4 不声明 quant.scale；它使用打包的 _blocks/_scales")
        elif spec["scale_field"] == "required" and not scale:
            errors.append(fmt + " 必须声明 quant.scale")
        if scale_dtype and scale_dtype not in spec.get("scale_dtypes", (spec["scale_dtype"],)):
            errors.append(fmt + " 的 scale_dtype 必须是 "
                          + " 或 ".join(spec.get("scale_dtypes", (spec["scale_dtype"],))))
        if fmt == "mxfp4" and scale_dtype:
            errors.append("mxfp4 的 scale_dtype 随打包格式固定，不能单独声明")

    fuses = quant.get("fuse") or {}
    if not isinstance(fuses, dict):
        errors.append("quant.fuse 必须是 fuse 名到张量列表的映射")
    else:
        if fuses and spec and not spec["fuse"]:
            errors.append(fmt + " 暂不支持 quant.fuse")
        for fused_name, value in fuses.items():
            if not _IDENTIFIER.fullmatch(str(fused_name)):
                errors.append("quant.fuse 的融合名必须是标识符")
                continue
            names = _list_names(value)
            if len(names) < 2:
                errors.append("quant.fuse 的值必须列出至少两个张量")
            elif len(names) != len(set(names)):
                errors.append("quant.fuse 不能重复列出同一个张量")
            elif tensor_names and any(name not in tensor_names for name in names):
                errors.append("quant.fuse 只能融合 quant.tensor 中已声明的张量")
    return errors


def selftest():
    """Deterministic numeric-oracle and accepted/rejected-format checks."""
    valid = (
        {"format": "mxfp4", "block": "32", "tensor": "[gate_exps, up_exps]",
         "fuse": {"gate_up_proj": "[gate_exps, up_exps]"}},
        {"format": "fp8_e4m3_block", "block": "128",
         "tensor": "[q_proj.weight]", "scale": "weight_scale_inv"},
        {"format": "fp8_e4m3_block", "block": "128",
         "tensor": "[q_proj.weight]", "scale": "weight_scale_inv", "scale_dtype": "BF16"},
    )
    invalid = (
        {"block": "32", "tensor": "[w]"},
        {"format": "mxfp4", "block": "16", "tensor": "[w]"},
        {"format": "fp8_e4m3_block", "block": "128", "tensor": "[w]"},
        {"format": "mxfp4", "block": "32", "tensor": "[w, w]"},
        {"format": "fp8_e4m3_block", "block": "128", "tensor": "[w]",
         "scale": "scales", "fuse": {"joined": "[w, other]"}},
        {"format": "mxfp4", "block": "32", "tensor": "[w]", "scale_dtype": "BF16"},
        {"format": "fp8_e4m3_block", "block": "128", "tensor": "[w]",
         "scale": "weight_scale_inv", "scale_dtype": "FP16"},
    )
    if any(validate_emit_quant(rule) for rule in valid):
        raise RuntimeError("valid quant format declaration was rejected")
    if any(not validate_emit_quant(rule) for rule in invalid):
        raise RuntimeError("invalid quant format declaration was accepted")
    if tuple(decode_mxfp4_element(code) for code in range(16)) != _MXFP4_VALUES:
        raise RuntimeError("MXFP4 E2M1 decode vector mismatch")
    if not (decode_e8m0_scale(126) == 0.5
            and decode_e8m0_scale(127) == 1.0
            and decode_e8m0_scale(128) == 2.0
            and math.isnan(decode_e8m0_scale(255))):
        raise RuntimeError("MXFP4 E8M0 scale vector mismatch")
    if mxfp4_block_dot([2, 4] + [0] * 30, [1, 3] + [0] * 30,
                       2.0, 0.5) != 3.5:
        raise RuntimeError("MXFP4 scaled dot-product vector mismatch")
    if not (decode_fp8_e4m3(0x01) == 2 ** -9
            and decode_fp8_e4m3(0x08) == 2 ** -6
            and decode_fp8_e4m3(0x38) == 1.0
            and decode_fp8_e4m3(0x7e) == 448.0
            and math.isnan(decode_fp8_e4m3(0x7f))
            and dequant_fp8_e4m3(0x38, 0.5) == 0.5):
        raise RuntimeError("FP8 E4M3 decode/scale vector mismatch")

    class _QuantConfig:
        def __init__(self, **kwargs):
            self.options = kwargs

        def to_dict(self):
            return dict(self.options)

    class _Transformers:
        Mxfp4Config = _QuantConfig
        FineGrainedFP8Config = _QuantConfig

    fp8_config = make_transformers_config(
        {"format": "fp8_e4m3_block", "block": 128},
        {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128],
         "modules_to_not_convert": ["visual"]}, _Transformers)
    if (fp8_config.options.get("dequantize") is not False
            or fp8_config.options.get("weight_block_size") != (128, 128)
            or fp8_config.options.get("modules_to_not_convert") != ["visual"]):
        raise RuntimeError("FP8 native Transformers config mismatch")
    if compile_emit_quant(valid[2]).get("scale_dtype") != "BF16":
        raise RuntimeError("FP8 scale dtype was not preserved in compiled contract")
    mxfp4_config = make_transformers_config(
        {"format": "mxfp4", "block": 32},
        {"quant_method": "mxfp4"}, _Transformers)
    if mxfp4_config.options.get("dequantize") is not False:
        raise RuntimeError("MXFP4 must request native packed storage")
    block_mismatch_rejected = False
    try:
        make_transformers_config(
            {"format": "fp8_e4m3_block", "block": 64},
            {"quant_method": "fp8", "weight_block_size": [128, 128]}, _Transformers)
    except ValueError:
        block_mismatch_rejected = True
    if not block_mismatch_rejected:
        raise RuntimeError("mismatched checkpoint FP8 block was accepted")

    class _Tensor:
        def __init__(self, dtype):
            self.dtype = dtype

    class _Model:
        def __init__(self, state):
            self._state = state

        def state_dict(self):
            return self._state

    class _Torch:
        uint8 = "u8"
        float32 = "f32"
        bfloat16 = "bf16"
        float8_e4m3fn = "f8e4m3"

    mxfp4_storage = audit_runtime_storage(_Model({
        "expert.weight_blocks": _Tensor("u8"),
        "expert.weight_scales": _Tensor("u8"),
    }), {"format": "mxfp4"}, _Torch)
    fp8_storage = audit_runtime_storage(_Model({
        "layer.weight": _Tensor("f8e4m3"),
        "layer.weight_scale_inv": _Tensor("f32"),
        "visual.proj.weight": _Tensor("f8e4m3"),
        "visual.proj.weight_scale_inv": _Tensor("f32"),
        "mtp.layers.0.proj.weight": _Tensor("f8e4m3"),
        "mtp.layers.0.proj.weight_scale_inv": _Tensor("f32"),
    }), {"format": "fp8_e4m3_block"}, _Torch)
    if (mxfp4_storage["packed_tensor_count"] != 1
            or fp8_storage["weight_tensor_count"] != 3
            or fp8_storage["scope_tensor_counts"] !=
            {"text_or_other": 1, "visual": 1, "mtp": 1}):
        raise RuntimeError("native storage audits missed quantized tensor pairs")
    bf16_fp8_storage = audit_runtime_storage(_Model({
        "layer.weight": _Tensor("f8e4m3"),
        "layer.weight_scale_inv": _Tensor("bf16"),
    }), {"format": "fp8_e4m3_block", "scale_dtype": "BF16"}, _Torch)
    if bf16_fp8_storage["scale_dtypes"] != ["BF16"]:
        raise RuntimeError("BF16 block scales were not preserved in storage audit")
    promoted_fp8_storage = audit_runtime_storage(_Model({
        "layer.weight": _Tensor("f8e4m3"),
        "layer.weight_scale_inv": _Tensor("f32"),
    }), {"format": "fp8_e4m3_block", "scale_dtype": "BF16"}, _Torch)
    if (promoted_fp8_storage["contract_scale_dtype"] != "BF16"
            or promoted_fp8_storage["scale_dtypes"] != ["F32"]):
        raise RuntimeError("exact BF16-to-F32 scale promotion was not reported")
    fallback_rejected = False
    try:
        audit_runtime_storage(_Model({"layer.weight": _Tensor("bf16")}),
                              {"format": "fp8_e4m3_block"}, _Torch)
    except ValueError:
        fallback_rejected = True
    if not fallback_rejected:
        raise RuntimeError("dequantized FP8 fallback passed the storage audit")

    class _Module:
        def __init__(self, weight=None):
            self.weight = weight if weight is not None else object()

    class _SharedModel:
        def __init__(self, output):
            self.input = _Module(shared_weight)
            self.output = output
            self.mtp = type("MTP", (), {"embed_tokens": self.input,
                                         "lm_head": self.output})()

        def get_input_embeddings(self):
            return self.input

        def get_output_embeddings(self):
            return self.output

    shared_weight = object()
    aliases = _SharedModel(_Module(shared_weight))
    sharing = audit_shared_weights(aliases, tie_word_embeddings=True, has_mtp=True)
    if not sharing["tied_word_embeddings"] or not sharing["mtp_aliases"]:
        raise RuntimeError("shared embedding/head/MTP identity audit mismatch")
    broken_sharing_rejected = False
    try:
        audit_shared_weights(_SharedModel(_Module()), tie_word_embeddings=True, has_mtp=True)
    except ValueError:
        broken_sharing_rejected = True
    if not broken_sharing_rejected:
        raise RuntimeError("broken weight sharing passed the audit")
