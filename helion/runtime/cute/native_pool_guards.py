"""Compile-only workers never create device contexts or invoke compiled code."""

from __future__ import annotations

from contextlib import ExitStack
from contextlib import contextmanager
import inspect
import os
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator


def reject(calls: list[str], label: str) -> Callable[..., None]:

    def fail(*args: object, **kwargs: object) -> None:
        calls.append(label)
        raise RuntimeError("Forbidden native-cache worker operation: " + label)

    return fail


@contextmanager
def worker_guards(calls: list[str]) -> Iterator[int]:
    from cuda.bindings import driver
    from cuda.bindings import runtime
    from cutlass.base_dsl import jit_executor
    from cutlass.cutlass_dsl import cuda_jit_executor
    from cutlass.cutlass_dsl import tvm_ffi_provider
    import torch
    from torch._dynamo.device_interface import CudaInterface
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    from helion.runtime.cute import launcher
    from helion.runtime.kernel import BoundKernel

    targets = []
    for module, prefix in ((driver, "cu"), (runtime, "cuda")):
        targets.extend(
            (module, name)
            for name in dir(module)
            if name.startswith(prefix)
            and name[len(prefix) : len(prefix) + 1].isupper()
            and callable(getattr(module, name))
            and (not isinstance(getattr(module, name), type))
        )
    targets.extend(
        (torch._C, name)
        for name in dir(torch._C)
        if name.startswith(("_cuda_", "_accelerator_"))
        and callable(getattr(torch._C, name))
    )
    targets.extend(
        (torch.cuda, name)
        for name in (
            "_lazy_init",
            "current_device",
            "get_device_properties",
            "get_device_capability",
            "get_device_name",
            "device_count",
            "current_stream",
            "synchronize",
            "set_stream",
            "_exchange_device",
            "_maybe_exchange_device",
            "cudart",
            "Stream",
            "ExternalStream",
            "Event",
            "CUDAGraph",
            "graph",
            "stream",
            "device",
            "set_device",
        )
    )
    targets.extend(
        (CudaInterface, name)
        for name in (
            "device",
            "Event",
            "Stream",
            "set_device",
            "stream",
            "current_stream",
            "set_stream",
            "_set_stream_by_id",
            "synchronize",
            "get_raw_stream",
            "exchange_device",
            "maybe_exchange_device",
            "memory_allocated",
            "is_bf16_supported",
            "current_device",
            "device_count",
            "get_device_properties",
        )
    )
    targets.extend(
        (CudaInterface.Worker, name)
        for name in ("set_device", "current_device", "get_device_properties")
    )
    targets.extend(
        [
            (jit_executor, "load_kernels_from_ir_module"),
            (jit_executor.JitCompiledFunction, "to"),
            (jit_executor.JitCompiledFunction, "get_aux_func"),
            (jit_executor.JitCompiledFunction, "run_compiled_program"),
            (jit_executor.JitCompiledFunction, "__call__"),
            (jit_executor.JitExecutor, "run_compiled_program"),
            (jit_executor.JitExecutor, "__call__"),
            (cuda_jit_executor.CudaDialectJitCompiledFunction, "to"),
            (cuda_jit_executor.CudaDialectJitCompiledFunction, "_deserializer"),
            (tvm_ffi_provider.TVMFFIJitCompiledFunctionBase, "to"),
            (tvm_ffi_provider.TVMFFIJitCompiledFunctionBase, "run_compiled_program"),
            (tvm_ffi_provider.TVMFFIJitCompiledFunction, "__call__"),
            (tvm_ffi_provider.TVMFFIJitCompiledFunctionWithKwargs, "__call__"),
            (launcher._CompiledCuteLauncher, "__call__"),
            (launcher, "_launch_cute_entry"),
            (BoundKernel, "to_code"),
            (BoundKernel, "compile_config"),
            (BoundKernel, "autotune"),
        ]
    )

    class NoDeviceTensor(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func: Callable[..., Any],
            types: object,
            args: tuple[object, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> object:
            kwargs = kwargs or {}
            leaves, _ = tree_flatten((args, kwargs))
            if any(
                isinstance(x, torch.Tensor) and x.device.type != "cpu" for x in leaves
            ):
                reject(calls, "non-CPU tensor dispatch")()
            device = kwargs.get("device")
            if device is not None and torch.device(device).type != "cpu":
                reject(calls, "non-CPU tensor allocation")()
            return func(*args, **kwargs)

    with ExitStack() as stack:
        for owner, name in targets:
            assert callable(getattr(owner, name)), (owner, name)
            inspect.getattr_static(owner, name)
            stack.enter_context(patch.object(owner, name, reject(calls, name)))
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        stack.enter_context(
            patch.object(torch.cuda, "is_available", return_value=False)
        )
        stack.enter_context(NoDeviceTensor())
        yield len(targets)
