from __future__ import annotations

import functools
import importlib

import torch
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.utils import triton_type
from triton.backends.compiler import GPUTarget
import triton.language as tl


def supports_tensor_descriptor() -> bool:
    # call private func we can patch in testing
    return _supports_tensor_descriptor()


@functools.cache
def _supports_tensor_descriptor() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major < 9:
        return False
    try:
        return get_triton_tensor_descriptor_class() is not None
    except ImportError:
        return False


@functools.cache
def get_triton_tensor_descriptor_class_import_path() -> str:
    cls = get_triton_tensor_descriptor_class()
    return f"from {cls.__module__} import {cls.__qualname__}"


@functools.cache
def get_triton_tensor_descriptor_class() -> type[object]:
    """Attempt to import TensorDescriptor class from known Triton modules."""
    possible_modules = [
        "triton.tools.tensor_descriptor",
        "triton.tools.experimental_descriptor",
    ]
    for module_name in possible_modules:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "TensorDescriptor"):
                return module.TensorDescriptor
        except ImportError:
            continue
    raise ImportError(
        "TensorDescriptor class not found in any of the known Triton modules."
    )


@functools.cache
def torch_dtype_to_tl(torch_dtype: torch.dtype) -> object:
    """Return the `triton.language` dtype that matches a `torch.dtype`."""
    name_str = triton_type(torch_dtype).replace("tl.", "")
    return getattr(tl, name_str)


@functools.cache
def min_dot_size(
    device: torch.device, lhs: torch.dtype, rhs: torch.torch.dtype
) -> tuple[int, int, int]:
    if device.type != "cuda":
        # TODO(jansel): support non-cuda properly
        return (16, 16, 16)

    from triton.backends.nvidia.compiler import min_dot_size as min_dot_size_cuda

    props = DeviceProperties.create(device)
    return min_dot_size_cuda(
        GPUTarget(
            backend=props.type,
            arch=props.cc,
            warp_size=props.warp_size or 32,
        )
    )(torch_dtype_to_tl(lhs), torch_dtype_to_tl(rhs))


def warps_to_threads(num_warps: int) -> int:
    if torch.cuda.is_available():
        props = DeviceProperties.create(
            torch.device("cuda", torch.cuda.current_device())
        )
        return num_warps * (props.warp_size or 32)
    return num_warps * 32
