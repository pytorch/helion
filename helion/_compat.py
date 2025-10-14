from __future__ import annotations

import contextlib
import functools
from typing import Any
from typing import Callable
from typing import cast

import torch
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.utils import triton_type
import triton
from triton.backends.compiler import BaseBackend
from triton.backends.compiler import GPUTarget
import triton.language as tl
import triton.runtime.jit as triton_jit

NativeSpecializeImpl = Callable[
    [type[BaseBackend], object, bool, bool, bool], tuple[object, ...]
]
CreateSpecializeImpl = Callable[
    [Callable[..., object]], Callable[..., tuple[object, ...]]
]


def _make_specialize_impl_wrapper(
    *,
    native_impl: NativeSpecializeImpl | None = None,
    create_factory: CreateSpecializeImpl | None = None,
) -> Callable[..., object]:
    if native_impl is None:
        native_impl = cast(
            "NativeSpecializeImpl | None",
            getattr(triton_jit, "native_specialize_impl", None),
        )
    if native_impl is None and create_factory is None:
        raise AttributeError("native_specialize_impl unavailable")

    def specialize_impl_wrapper(
        *args: object,
        **kwargs: object,
    ) -> object:
        specialize_extra = cast(
            "Callable[..., object] | None",
            kwargs.pop("specialize_extra", None),
        )
        kwargs.pop("specialize_zero_one", None)
        backend_param = kwargs.pop("backend", None)
        args_list: list[object] = list(args)
        backend_type: type[BaseBackend]
        if backend_param is None and args_list:
            first = args_list[0]
            if isinstance(first, type) and issubclass(first, BaseBackend):
                backend_type = first
                args_list.pop(0)
            elif isinstance(first, BaseBackend):
                backend_type = type(first)
                args_list.pop(0)
            else:
                backend_type = BaseBackend
        elif isinstance(backend_param, type) and issubclass(backend_param, BaseBackend):
            backend_type = backend_param
        elif isinstance(backend_param, BaseBackend):
            backend_type = type(backend_param)
        else:
            backend_type = BaseBackend

        arg = kwargs.pop("arg", None)
        if arg is None:
            if args_list:
                arg = args_list.pop(0)
            else:
                raise TypeError("specialize_impl() missing positional argument 'arg'")

        def _pop_flag(
            key: str,
            *,
            alt_keys: tuple[str, ...] = (),
            default: bool | None = None,
        ) -> bool:
            value = kwargs.pop(key, None)
            if value is None:
                for alt in alt_keys:
                    value = kwargs.pop(alt, None)
                    if value is not None:
                        break
            if value is None:
                if args_list:
                    value = args_list.pop(0)
                elif default is not None:
                    value = default
                else:
                    raise TypeError(f"specialize_impl() missing argument '{key}'")
            return bool(value)

        is_const = _pop_flag("is_const")
        specialize_value = _pop_flag(
            "specialize_value",
            alt_keys=("specialize",),
            default=True,
        )
        align = _pop_flag("align", default=True)

        if native_impl is not None:
            result = native_impl(
                backend_type,
                arg,
                is_const,
                specialize_value,
                align,
            )
            if specialize_extra is not None:
                with contextlib.suppress(Exception):
                    specialize_extra(arg)
        else:
            assert create_factory is not None

            def _call_specialize_extra(
                extra_arg: object,
                kind: object,
                *,
                align: bool = True,
            ) -> object:
                if specialize_extra is None:
                    return None
                try:
                    return specialize_extra(extra_arg)
                except TypeError:
                    try:
                        return specialize_extra(extra_arg, kind, align=align)
                    except Exception:
                        return None
                except Exception:
                    return None

            impl = create_factory(_call_specialize_extra)
            result = impl(
                arg,
                is_const=is_const,
                specialize_value=specialize_value,
                align=align,
            )
        return result

    return specialize_impl_wrapper


def _ensure_triton_specialize_impl_alias() -> None:
    if hasattr(triton_jit, "specialize_impl"):
        return
    if hasattr(triton_jit, "native_specialize_impl"):
        module: Any = triton_jit
        module.specialize_impl = _make_specialize_impl_wrapper()  # type: ignore[assignment]
        return
    if hasattr(triton_jit, "create_specialize_impl"):
        module: Any = triton_jit
        module.specialize_impl = _make_specialize_impl_wrapper(
            create_factory=triton_jit.create_specialize_impl,
        )  # type: ignore[assignment]


_ensure_triton_specialize_impl_alias()


def _ensure_backend_specialization_alias() -> None:
    if hasattr(BaseBackend, "get_arg_specialization"):
        return
    if hasattr(BaseBackend, "get_tensor_specialization"):
        BaseBackend.get_arg_specialization = BaseBackend.get_tensor_specialization  # type: ignore[attr-defined]


_ensure_backend_specialization_alias()


@functools.cache
def get_triton_find_paths_if() -> Callable[..., object]:
    if hasattr(triton_jit, "find_paths_if"):
        return triton_jit.find_paths_if
    if hasattr(triton_jit, "_find_paths_if"):
        return triton_jit._find_paths_if  # type: ignore[attr-defined]
    raise AttributeError("Unable to locate Triton find_paths_if helper")


@functools.cache
def get_triton_iterable_path() -> Callable[..., object]:
    if hasattr(triton_jit, "get_iterable_path"):
        return triton_jit.get_iterable_path
    if hasattr(triton_jit, "_get_iterable_path"):
        return triton_jit._get_iterable_path  # type: ignore[attr-defined]
    raise AttributeError("Unable to locate Triton get_iterable_path helper")


def supports_tensor_descriptor() -> bool:
    # call private func we can patch in testing
    return _supports_tensor_descriptor()


@functools.cache
def _supports_tensor_descriptor() -> bool:
    def _cuda_tensor_desc_available() -> bool:
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
        return major >= 9

    def _xpu_tensor_desc_available() -> bool:
        if not torch.xpu.is_available():
            return False

        from packaging import version

        return version.parse(triton.__version__) >= version.parse("3.5")

    if not (_cuda_tensor_desc_available() or _xpu_tensor_desc_available()):
        return False

    return hasattr(triton.language, "make_tensor_descriptor") or hasattr(
        triton.language, "_experimental_make_tensor_descriptor"
    )


@functools.cache
def get_tensor_descriptor_fn_name() -> str:
    if hasattr(triton.language, "make_tensor_descriptor"):
        return "tl.make_tensor_descriptor"
    assert hasattr(triton.language, "_experimental_make_tensor_descriptor")
    return "tl._experimental_make_tensor_descriptor"


@functools.cache
def torch_dtype_to_tl(torch_dtype: torch.dtype) -> object:
    """Return the `triton.language` dtype that matches a `torch.dtype`."""
    name_str = triton_type(torch_dtype).replace("tl.", "")
    return getattr(tl, name_str)


def min_dot_size(
    device: torch.device, lhs: torch.dtype, rhs: torch.dtype
) -> tuple[int, int, int]:
    # call private func we can patch in testing
    return _min_dot_size(device, lhs, rhs)


@functools.cache
def _min_dot_size(
    device: torch.device, lhs: torch.dtype, rhs: torch.dtype
) -> tuple[int, int, int]:
    if device.type not in ["cuda", "xpu"]:
        # TODO(jansel): support other hardware backends properly besides CUDA and XPU
        return (16, 16, 16)

    if torch.xpu.is_available():
        from triton.backends.intel.compiler import (  # pyright: ignore[reportMissingImports]
            min_dot_size as min_dot_size_xpu,
        )

        device_properties = torch.xpu.get_device_properties()
        gpu_target_info = {
            k: getattr(device_properties, k)
            for k in device_properties.__dir__()
            if not k.startswith("_")
        }

        dot_size_val = min_dot_size_xpu(gpu_target_info)(
            torch_dtype_to_tl(lhs), torch_dtype_to_tl(rhs)
        )
        return tuple(int(v) for v in dot_size_val)  # pyright: ignore[reportReturnType]

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
