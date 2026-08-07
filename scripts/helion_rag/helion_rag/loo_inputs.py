"""Reconstruct runnable inputs for corpus workloads used by LOO evaluation."""

from __future__ import annotations

import ast
import importlib

_DTYPES = {
    "torch.float16": "float16",
    "torch.float32": "float32",
    "torch.bfloat16": "bfloat16",
    "torch.int64": "int64",
    "torch.int32": "int32",
    "torch.int8": "int8",
}

KERNEL_ENTRY = {
    "attention": ("examples.attention", "attention"),
    "helion_gdn_fwd_h": ("examples.gdn_fwd_h", "helion_gdn_fwd_h"),
    "helion_mamba2_chunk_scan_kernel": (
        "examples.mamba2_chunk_scan",
        "helion_mamba2_chunk_scan_kernel",
    ),
    "helion_mamba2_chunk_state_kernel": (
        "examples.mamba2_chunk_state",
        "helion_mamba2_chunk_state_kernel",
    ),
    "softmax_two_pass": ("examples.softmax", "softmax_two_pass"),
    "cross_entropy": ("examples.cross_entropy", "cross_entropy"),
    "jsd_forward": ("examples.jsd", "jsd_forward"),
    "kl_div_forward": ("examples.kl_div", "kl_div_forward"),
    "layer_norm_fwd": ("examples.layer_norm", "layer_norm_fwd"),
    "layer_norm_bwd": ("examples.layer_norm", "layer_norm_bwd"),
    "matmul": ("examples.matmul", "matmul"),
    "matmul_bf16_int4": ("examples.int4_gemm", "matmul_bf16_int4"),
    "rms_norm_fwd": ("examples.rms_norm", "rms_norm_fwd"),
    "rms_norm_bwd": ("examples.rms_norm", "rms_norm_bwd"),
    "rope_fwd": ("examples.rope", "rope_fwd"),
    "welford": ("examples.welford", "welford"),
}
SUPPORTED_KERNELS = tuple(KERNEL_ENTRY)


def _tensor(shape, dtype: str, *, int4: bool = False):
    import torch

    torch_dtype = getattr(torch, _DTYPES[dtype])
    shape = tuple(shape)
    if torch_dtype.is_floating_point:
        return torch.randn(shape, dtype=torch_dtype, device="cuda")
    low, high = (-8, 8) if int4 else (0, 2)
    return torch.randint(low, high, shape, dtype=torch_dtype, device="cuda")


def _generic(shapes, dtypes):
    return tuple(_tensor(shape, dtype) for shape, dtype in zip(shapes, dtypes))


def _cross_entropy(shapes, dtypes):
    import torch

    logits = _tensor(shapes[0], dtypes[0])
    labels = torch.randint(
        0,
        shapes[0][-1],
        tuple(shapes[1]),
        dtype=torch.int64,
        device="cuda",
    )
    return logits, labels


def _layer_norm_fwd(shapes, dtypes):
    x, weight, bias = _generic(shapes, dtypes)
    return x, [x.shape[-1]], weight, bias


def _gdn(shapes, dtypes):
    return (*_generic(shapes, dtypes), 64)


def _int4(shapes, dtypes):
    return _tensor(shapes[0], dtypes[0]), _tensor(shapes[1], dtypes[1], int4=True)


def _kl_div(shapes, dtypes):
    import torch

    y_pred = _tensor(shapes[0], dtypes[0])
    y_true = torch.rand(
        tuple(shapes[1]),
        dtype=getattr(torch, _DTYPES[dtypes[1]]),
        device="cuda",
    )
    return y_pred, y_true


_BUILDERS = {
    "cross_entropy": _cross_entropy,
    "layer_norm_fwd": _layer_norm_fwd,
    "helion_gdn_fwd_h": _gdn,
    "matmul_bf16_int4": _int4,
    "kl_div_forward": _kl_div,
}


def build_inputs(kernel_name: str, shapes, dtypes) -> tuple[object, ...]:
    """Build CUDA inputs from canonical corpus shape and dtype strings."""
    if isinstance(shapes, str):
        shapes = ast.literal_eval(shapes)
    if isinstance(dtypes, str):
        dtypes = ast.literal_eval(dtypes)
    builder = _BUILDERS.get(kernel_name, _generic)
    return tuple(builder(list(shapes), list(dtypes)))


def load_kernel(kernel_name: str):
    """Load the example kernel corresponding to a corpus kernel name."""
    module, attribute = KERNEL_ENTRY[kernel_name]
    return getattr(importlib.import_module(module), attribute)
