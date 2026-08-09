"""Helion-dependency-free runtime launch helpers for the Triton backend.

This module holds the small set of runtime symbols that Helion's *generated*
Triton code depends on at execution time:

* :func:`default_launcher` -- invokes a compiled ``triton.jit`` kernel.
* :func:`get_num_sm` -- persistent-kernel grid size (host statement).
* :func:`set_triton_allocator` -- installs the scratch allocator used by TMA /
  tensor-descriptor kernels (device-function prefix statement).

It depends only on ``torch`` and ``triton`` -- no other ``helion`` module -- so
the ahead-of-time precompiler can bulk-export this file verbatim into a
standalone kernel with zero Helion runtime dependency.

Helion-specific behavior that is only meaningful in-process (translating
Triton's opaque shape errors into :class:`helion.exc.ShapeMismatch`, and the
CPU/TPU cases of :func:`get_num_sm`) lives in thin wrappers in
:mod:`helion.runtime`, not here.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import triton
except ImportError:
    triton = None  # type: ignore[assignment]

try:
    from triton._C.libtriton import (
        native_specialize_impl,  # pyrefly: ignore [missing-module-attribute]
    )
except ImportError:
    native_specialize_impl = None  # type: ignore[assignment]


if triton is not None:

    def _alloc_fn(size: int, alignment: int, stream: int | None) -> torch.Tensor:
        # Dynamically get device from Triton backend
        current_target = triton.runtime.driver.active.get_current_target()
        if current_target is None:
            raise RuntimeError("No active Triton target available")
        backend = current_target.backend
        return torch.empty(size, device=backend, dtype=torch.int8)

    def set_triton_allocator() -> None:
        try:
            from triton import set_allocator
            from triton.runtime._allocation import NullAllocator
            from triton.runtime._allocation import _allocator
        except ImportError:
            return
        if isinstance(_allocator, contextvars.ContextVar):
            existing = _allocator.get()
        else:  # older versions of Triton
            existing = _allocator
        # if allocator isn't NullAllocator, we assume it is set by the user
        if isinstance(existing, NullAllocator):
            set_allocator(_alloc_fn)

else:

    def set_triton_allocator() -> None:  # type: ignore[misc]
        pass


def get_num_sm(device: torch.device, *, reserved_sms: int = 0) -> int:
    """
    Get the number of streaming multiprocessors (SMs) for the specified GPU.

    Args:
        device: Device to query. Must be a GPU device (``cuda``/``xpu``/``mps``/
            ``mtia``); CPU/TPU handling lives in :func:`helion.runtime.get_num_sm`.
        reserved_sms: Number of SMs to keep free for other work (e.g., communication
            kernels). Defaults to 0 meaning all device SMs are available to Helion.

    Returns:
        Grid size to use for a persistent kernel on the device after accounting
        for any reserved SMs. Always at least 1.
    """
    available_sms: int
    assert device.type in [
        "cuda",
        "xpu",
        "mtia",
        "mps",
    ], "TODO: implement for other devices"
    if device.type == "cuda":
        available_sms = torch.cuda.get_device_properties(
            device.index
        ).multi_processor_count
    # TODO(EikanWang): gpu_subslice_count is an out-of-date term. we change update it to XeCore number.
    elif device.type == "xpu":
        available_sms = torch.xpu.get_device_properties(device.index).gpu_subslice_count
    elif device.type == "mps":
        available_sms = torch.backends.mps.get_core_count()
    elif device.type == "mtia":
        device_props = torch.mtia.get_device_properties(device.index)
        if "max_grid_height" in device_props and "max_grid_width" in device_props:
            available_sms = (
                device_props["max_grid_height"] * device_props["max_grid_width"]
            )
        else:
            raise RuntimeError(
                f"Unable to determine SM count for MTIA device. "
                f"Available properties: {list(device_props.keys())}"
            )
    else:
        raise NotImplementedError(
            f"get_num_sm not implemented for device type: {device.type}"
        )

    if reserved_sms <= 0:
        return available_sms
    return max(available_sms - reserved_sms, 1)


# CUs per XCD by base CDNA architecture.  Used to derive the live,
# partition-visible XCD count from the observed CU count (see get_num_xcd).
_CUS_PER_XCD: dict[str, int] = {
    "gfx942": 38,  # CDNA3 (MI300)
    "gfx950": 32,  # CDNA4 (MI350)
    "gfx951": 32,  # CDNA4 (MI355)
}

_MISSING_GLOBAL = object()
_PREPARED_LAUNCH_CACHE_LIMIT = 8
_RUNTIME_ONLY_OPTIONS = frozenset({"device", "device_type", "stream", "warmup"})


def _prepared_key_value(value: object) -> object:
    if type(value) is float:
        if value != value:
            raise TypeError("NaN cannot be a prepared-launch key")
        return value.hex()
    return value


def _prepared_options_key(
    ptx_options: str | None, kwargs: dict[str, object]
) -> tuple[object, ...] | None:
    """Return a conservative exact key for Triton compilation options."""
    if ptx_options is None and not kwargs:
        return ()
    if ptx_options is not None and type(ptx_options) is not str:
        return None
    result: list[object] = [("ptx_options", type(ptx_options), ptx_options)]
    supported_types = (
        bool,
        int,
        float,
        str,
        type(None),
        torch.dtype,
        torch.device,
    )
    for name, value in sorted(kwargs.items()):
        if type(value) not in supported_types:
            return None
        try:
            keyed_value = _prepared_key_value(value)
        except TypeError:
            return None
        result.append((name, type(value), keyed_value))
    return tuple(result)


def _prepared_launch_context(
    triton_kernel: object, grid: tuple[int, ...]
) -> tuple[object, object] | None:
    """Return live launch state when bypassing ``JITFunction.run`` is safe."""
    if triton is None or native_specialize_impl is None:
        return None
    runtime = triton.knobs.runtime
    compilation = triton.knobs.compilation
    pre_run_hooks = triton_kernel.pre_run_hooks  # type: ignore[union-attr]
    effective_debug = triton_kernel.debug or runtime.debug  # type: ignore[union-attr]
    if (
        type(pre_run_hooks) is not list
        or pre_run_hooks
        or type(effective_debug) is not bool
        or effective_debug
        or runtime.add_stages_inspection_hook is not None
        or type(compilation.instrumentation_mode) is not str
        or compilation.instrumentation_mode != ""
        # CompiledKernel.__getitem__ closes over a padded 3D grid. A custom
        # metadata callback must continue to observe Triton's original grid.
        or (len(grid) < 3 and triton_kernel.launch_metadata is not None)  # type: ignore[union-attr]
    ):
        return None
    active = triton.runtime.driver.active
    # pyrefly: ignore [missing-attribute]
    return active, active.get_current_device()


def _make_prepared_launch_guard(
    triton_kernel: object,
    active: object,
    device: object,
    device_cache: object,
    backend: object,
    kernel_cache: object,
    kernel_key: object,
    compiled: object,
    grid: tuple[int, ...],
    args: tuple[object, ...],
    num_warps: int,
    num_stages: int,
    launch_cooperative_grid: bool,
    option_key: tuple[object, ...],
) -> Callable[..., bool]:
    """Build an unrolled guard finer than Triton's specialization key."""
    if len(args) != len(triton_kernel.params):  # type: ignore[union-attr]
        raise TypeError("prepared launch requires every JIT parameter")
    specialize_impl = native_specialize_impl
    if specialize_impl is None:
        raise TypeError("Triton specialization is unavailable")
    namespace: dict[str, object] = {
        "active": active,
        "device": device,
        "device_cache": device_cache,
        "backend": backend,
        "kernel_cache": kernel_cache,
        "kernel_key": kernel_key,
        "compiled": compiled,
        "grid": grid,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "launch_cooperative_grid": launch_cooperative_grid,
        "option_key": option_key,
        "num_warps_type": type(num_warps),
        "num_stages_type": type(num_stages),
        "launch_cooperative_grid_type": type(launch_cooperative_grid),
        "missing": _MISSING_GLOBAL,
        "triton_kernel": triton_kernel,
        "specialize_impl": specialize_impl,
    }
    checks = [
        "current_active is active",
        "current_device == device",
        "current_device_cache is device_cache",
        "kernel_cache.get(kernel_key, missing) is compiled",
        "current_grid == grid",
        "type(current_num_warps) is num_warps_type and current_num_warps == num_warps",
        "type(current_num_stages) is num_stages_type and current_num_stages == num_stages",
        (
            "type(current_launch_cooperative_grid) is launch_cooperative_grid_type "
            "and current_launch_cooperative_grid == launch_cooperative_grid"
        ),
        "current_option_key == option_key",
        f"len(args) == {len(args)}",
    ]
    for index, arg in enumerate(args):
        prefix = f"args[{index}]"
        namespace[f"type_{index}"] = type(arg)
        checks.append(f"type({prefix}) is type_{index}")
        parameter = triton_kernel.params[index]  # type: ignore[union-attr]
        namespace[f"is_const_{index}"] = parameter.is_const
        namespace[f"specialize_{index}"] = not parameter.do_not_specialize
        namespace[f"align_{index}"] = not parameter.do_not_specialize_on_alignment
        if parameter.is_constexpr:
            namespace[f"value_{index}"] = _prepared_key_value(arg)
            checks.append(
                f"{prefix}.hex() == value_{index}"
                if type(arg) is float
                else f"{prefix} == value_{index}"
            )
        else:
            native_specialization = specialize_impl(
                backend,
                arg,
                parameter.is_const,
                not parameter.do_not_specialize,
                not parameter.do_not_specialize_on_alignment,
            )
            annotation_type = parameter.annotation_type
            if annotation_type:
                specializes_annotation = not (
                    parameter.do_not_specialize
                    or annotation_type == "u1"
                    or annotation_type.startswith(("fp", "bf"))
                )
                if specializes_annotation:
                    namespace[f"specialization_{index}"] = native_specialization[1:]
                    checks.append(
                        f"specialize_impl(backend, {prefix}, is_const_{index}, "
                        f"specialize_{index}, align_{index})[1:] "
                        f"== specialization_{index}"
                    )
            else:
                namespace[f"specialization_{index}"] = native_specialization
                checks.append(
                    f"specialize_impl(backend, {prefix}, is_const_{index}, "
                    f"specialize_{index}, align_{index}) == specialization_{index}"
                )

        if type(arg) in (torch.Tensor, torch.nn.Parameter):
            assert isinstance(arg, torch.Tensor)
            namespace[f"device_{index}"] = arg.device
            namespace[f"dtype_{index}"] = arg.dtype
            checks.extend(
                (
                    f"{prefix}.device == device_{index}",
                    f"{prefix}.dtype is dtype_{index}",
                )
            )
        elif type(arg) in (
            bool,
            int,
            float,
            str,
            type(None),
            torch.dtype,
            torch.device,
        ):
            pass
        else:
            raise TypeError(f"unsupported prepared-launch argument: {type(arg)!r}")
    used_globals = triton_kernel.used_global_vals  # type: ignore[union-attr]
    namespace["used_globals"] = used_globals
    checks.extend(
        (
            "triton_kernel.used_global_vals is used_globals",
            f"len(used_globals) == {len(used_globals)}",
        )
    )
    for index, ((name, global_id), entry) in enumerate(used_globals.items()):
        value, global_dict = entry
        namespace[f"used_global_key_{index}"] = (name, global_id)
        namespace[f"used_global_entry_{index}"] = entry
        namespace[f"global_name_{index}"] = name
        namespace[f"global_dict_{index}"] = global_dict
        namespace[f"global_value_{index}"] = value
        checks.extend(
            (
                (
                    f"used_globals.get(used_global_key_{index}, missing) "
                    f"is used_global_entry_{index}"
                ),
                (
                    f"not (global_dict_{index}.get(global_name_{index}, missing) "
                    f"!= global_value_{index})"
                ),
            )
        )
    parameters = (
        "current_active, current_device, current_device_cache, current_grid, args, "
        "current_num_warps, current_num_stages, current_launch_cooperative_grid, "
        "current_option_key"
    )
    return eval(f"lambda {parameters}: {' and '.join(checks)}", namespace)


def _prepared_launch_cache(
    triton_kernel: object,
) -> list[tuple[Callable[..., bool], tuple[object, ...]]]:
    cache = triton_kernel.__dict__.get("_helion_prepared_launches")  # type: ignore[union-attr]
    if cache is None:
        cache = []
        triton_kernel._helion_prepared_launches = cache  # type: ignore[union-attr]
    return cache


def _is_future_kernel(compiled: object) -> bool:
    compiled_type = type(compiled)
    return (
        compiled_type.__name__ == "FutureKernel"
        and compiled_type.__module__ == "triton.runtime._async_compile"
    )


def get_num_xcd(device: torch.device | int | None = None) -> int:
    """Number of XCDs visible for ``device`` on AMD CDNA, else ``1``.

    Derived from the live, partition-visible compute-unit count rather than the
    architecture name, so MI300A (6 XCDs) and compute-partition modes such as CPX
    (which expose a single XCD) are handled correctly.  Returns ``1`` -- which
    disables xcd_remap -- for unknown architectures or a CU count that does not
    look like an integer number of XCDs.
    """
    if not torch.cuda.is_available():
        return 1
    try:
        props = torch.cuda.get_device_properties(
            device if device is not None else torch.cuda.current_device()
        )
    except Exception:
        return 1
    arch = getattr(props, "gcnArchName", None)
    if not arch:
        return 1
    cus_per_xcd = _CUS_PER_XCD.get(arch.split(":")[0])
    if cus_per_xcd is None:
        return 1
    cu_count = props.multi_processor_count
    num_xcd = round(cu_count / cus_per_xcd)
    # Tolerate harvested parts, but bail out (return 1) if the live CU count does
    # not look like an integer number of XCDs.
    if num_xcd < 1 or abs(num_xcd * cus_per_xcd - cu_count) > cus_per_xcd // 4:
        return 1
    return num_xcd


def default_launcher(
    triton_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    num_warps: int,
    num_stages: int,
    ptx_options: str | None = None,
    launch_cooperative_grid: bool = False,
    **kwargs: object,
) -> object:
    """Launch through Triton, caching resolved binaries for repeat calls."""
    option_key = _prepared_options_key(ptx_options, kwargs)
    can_prepare = (
        option_key is not None
        and not _RUNTIME_ONLY_OPTIONS.intersection(kwargs)
        and type(grid) is tuple
        and 0 < len(grid) <= 3
        and all(type(value) is int for value in grid)
        and not torch.compiler.is_compiling()
    )
    prepared_context = None
    if can_prepare:
        prepared = None
        try:
            prepared_context = _prepared_launch_context(triton_kernel, grid)
            cache = triton_kernel.__dict__.get(  # type: ignore[union-attr]
                "_helion_prepared_launches"
            )
            if prepared_context is not None and cache:
                active, device = prepared_context
                device_cache = triton_kernel.device_caches.get(device)  # type: ignore[union-attr]
                for index, (guard, cached) in enumerate(cache):
                    if guard(
                        active,
                        device,
                        device_cache,
                        grid,
                        args,
                        num_warps,
                        num_stages,
                        launch_cooperative_grid,
                        option_key,
                    ):
                        if index:
                            cache.insert(0, cache.pop(index))
                        prepared = cached
                        break
        except Exception:
            pass
        if prepared is not None:
            runner, compiled = prepared
            runner(*args)  # pyrefly: ignore [not-callable]
            return compiled

    run_kwargs: dict = {
        "grid": grid,
        "warmup": False,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "launch_cooperative_grid": launch_cooperative_grid,
        **kwargs,
    }
    if ptx_options is not None:
        run_kwargs["ptx_options"] = ptx_options
    compiled = triton_kernel.run(*args, **run_kwargs)  # type: ignore[union-attr]
    if (
        can_prepare
        and prepared_context is not None
        and compiled is not None
        and not _is_future_kernel(compiled)
    ):
        try:
            assert option_key is not None
            active, device = prepared_context
            current_context = _prepared_launch_context(triton_kernel, grid)
            if not (
                current_context is not None
                and current_context[0] is active
                and current_context[1] == device
            ):
                return compiled
            device_cache = triton_kernel.device_caches.get(device)  # type: ignore[union-attr]
            if device_cache is None:
                return compiled
            kernel_cache = device_cache[0]  # pyrefly: ignore [unsupported-operation]
            backend = device_cache[3]  # pyrefly: ignore [unsupported-operation]
            kernel_key = next(
                (
                    key
                    for key, value in kernel_cache.items()  # pyrefly: ignore [missing-attribute]
                    if value is compiled
                ),
                _MISSING_GLOBAL,
            )
            if kernel_key is _MISSING_GLOBAL:
                return compiled
            guard = _make_prepared_launch_guard(
                triton_kernel,
                active,
                device,
                device_cache,
                backend,
                kernel_cache,
                kernel_key,
                compiled,
                grid,
                args,
                num_warps,
                num_stages,
                launch_cooperative_grid,
                option_key,
            )
            cache = _prepared_launch_cache(triton_kernel)
            normalized_grid = (
                grid[0],
                grid[1] if len(grid) > 1 else 1,
                grid[2] if len(grid) > 2 else 1,
            )
            cache.insert(
                0,
                (
                    guard,
                    (
                        compiled[normalized_grid],  # pyrefly: ignore [unsupported-operation]
                        compiled,
                    ),
                ),
            )
            if len(cache) > _PREPARED_LAUNCH_CACHE_LIMIT:
                cache.pop()
        except Exception:
            pass
    return compiled
