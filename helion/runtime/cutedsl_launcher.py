from __future__ import annotations

import threading
from typing import Any
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

# Cache compiled kernels: keyed on (kernel_fn_id, tensor_meta, scalar_values)
_compile_cache: dict[tuple[Any, ...], Any] = {}
_compile_lock = threading.Lock()


def _tensor_meta(t: torch.Tensor) -> tuple[Any, ...]:
    """Extract shape/stride/dtype metadata from a tensor for cache keying."""
    return (tuple(t.shape), tuple(t.stride()), t.dtype)


def default_cutedsl_launcher(
    kernel_fn: object,
    grid: tuple[int, ...],
    *args: object,
    num_warps: int = 4,
    num_stages: int = 2,
    shared_mem_bytes: int = 0,
    **kwargs: object,
) -> None:
    """Default launcher function that compiles and executes a CuteDSL kernel.

    Uses cute.compile() for compilation and from_dlpack() for tensor conversion.
    Compiled kernels are cached based on kernel identity and argument metadata.

    Args:
        kernel_fn: The CuteDSL kernel function (decorated with @cute.kernel).
        grid: Grid dimensions for the kernel launch.
        *args: Kernel arguments (tensors, scalars, etc.).
        num_warps: Number of warps per block.
        num_stages: Number of pipeline stages.
        shared_mem_bytes: Shared memory allocation in bytes.
        **kwargs: Additional keyword arguments (constexpr values like _BLOCK_SIZE_0).
    """
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    # Get current CUDA stream
    device = None
    for a in args:
        if isinstance(a, torch.Tensor):
            device = a.device
            break
    assert device is not None, "CuteDSL launcher requires at least one tensor argument"
    current_stream = torch.cuda.current_stream(device).cuda_stream

    # Build compile args: convert tensors to cute tensors, keep scalars as-is
    compile_args: list[object] = []
    cache_key_parts: list[Any] = [id(kernel_fn)]

    for a in args:
        if isinstance(a, torch.Tensor):
            cute_tensor = from_dlpack(a.detach(), assumed_align=16)
            if a.ndim > 0:
                cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=a.ndim - 1)
            compile_args.append(cute_tensor)
            cache_key_parts.append(_tensor_meta(a))
        else:
            compile_args.append(a)
            cache_key_parts.append(a)

    # Add constexpr kwargs (like _BLOCK_SIZE_0)
    for k, v in sorted(kwargs.items()):
        compile_args.append(v)
        cache_key_parts.append((k, v))

    # Add stream
    compile_args.append(current_stream)
    cache_key_parts.append("stream")

    cache_key = tuple(cache_key_parts)

    # Compile (with caching)
    with _compile_lock:
        compiled_fn = _compile_cache.get(cache_key)
        if compiled_fn is None:
            compiled_fn = cute.compile(
                kernel_fn,
                *compile_args,
            )
            _compile_cache[cache_key] = compiled_fn

    # Launch: pass raw torch tensors (detached) and scalar args
    launch_args: list[object] = []
    for a in args:
        if isinstance(a, torch.Tensor):
            launch_args.append(a.detach())
        else:
            launch_args.append(a)

    # Add constexpr kwargs as positional args (same order as compile)
    for _k, v in sorted(kwargs.items()):
        launch_args.append(v)

    # Add stream
    launch_args.append(current_stream)

    compiled_fn(*launch_args)
