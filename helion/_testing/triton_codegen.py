"""CPU-only Triton code generation helpers shared by the test suite."""

from __future__ import annotations

import ast
from contextlib import ExitStack
from contextlib import contextmanager
import importlib.util
import sys
from types import MappingProxyType
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from .cute_codegen import cuda_metadata_trace

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Mapping
    from pathlib import Path

    from ..runtime.kernel import BoundKernel


@pytest.fixture
def cpu_codegen() -> Generator[None, None, None]:
    """Generate Triton source in a CPU-only process for an sm90 target.

    The runtime sees 132 SMs and the config spec 128 compute units. The
    device probes report no CUDA, maxnreg, tensor descriptors, HIP or TileIR
    tunables, and initializing CUDA fails the test.
    """
    with ExitStack() as stack:
        for name, value in (
            ("torch.cuda.is_available", False),
            ("helion._compat._supports_maxnreg", False),
            ("helion._compat._supports_tensor_descriptor", False),
            ("helion._compat._is_hip", False),
            ("helion.language.loops.use_tileir_tunables", False),
            ("helion.autotuner.config_spec.num_compute_units", 128),
            ("helion.runtime.kernel.target_device_capability", (9, 0)),
            ("helion._compiler.compile_environment.target_device_capability", (9, 0)),
            ("helion.runtime.get_num_sm", 132),
        ):
            stack.enter_context(patch(name, return_value=value))
        stack.enter_context(
            patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden"))
        )
        yield


# The config every Gluon ring test starts from: tensor-descriptor operands, one
# persistent worker a SM, a two-stage ring (read-only: tests spread it).
GLUON_RING_CONFIG: Mapping[str, object] = MappingProxyType(
    {
        "indexing": "tensor_descriptor",
        "pid_type": "persistent_blocked",
        "num_sm_multiplier": 1,
        "num_stages": 2,
    }
)


def gluon_tail_ring_config(
    bound: BoundKernel, *, task_loop: int = -1, **extra: object
) -> dict[str, object]:
    """The ring config of the ``cross_loop_tmem_prefetch`` tests (their
    ``scale_then_*`` kernels): the ``task_loop``-th
    warp-specializable loop warp-specialized, 128-row weight tiles over K steps
    of 64, a four-stage ring; ``extra`` overrides."""
    slots: list[bool | None] = [None] * len(bound.config_spec.range_warp_specialize)
    slots[task_loop] = True
    block_sizes = [32, 64, 128, 32, 64]
    block_sizes += [64] * (len(bound.config_spec.block_sizes) - len(block_sizes))
    return {
        **GLUON_RING_CONFIG,
        "block_sizes": block_sizes,
        "range_warp_specializes": slots,
        "cross_loop_pipeline": "static",
        "num_warps": 4,
        "num_stages": 4,
        **extra,
    }


@contextmanager
def gluon_cpu_codegen() -> Generator[None, None, None]:
    """Generate Gluon (``backend="gluon"``) source in a CPU-only process for a
    148-SM sm100 target: kernels trace on ``cuda:0`` as on the GPU
    (``cute_codegen.cuda_metadata_trace``, so kernels with cross-loop
    dependencies bind), the Gluon, warp-specialization and tensor-descriptor
    probes report support, the ring lowering sees 2048 thread slots a SM, the
    dsmem transports' residency probe sees every SM in clusters of any size
    (``_cluster_capacity``), and initializing CUDA fails the test."""
    device = SimpleNamespace(max_threads_per_multi_processor=2048)
    with ExitStack() as stack:
        stack.enter_context(cuda_metadata_trace())
        for name, value in (
            ("torch.cuda.get_device_properties", device),
            ("helion._compat.supports_gluon", True),
            ("helion._compat._supports_tensor_descriptor", True),
            ("helion.language.loops._supports_warp_specialize", True),
            ("helion.autotuner.config_spec.num_compute_units", 148),
        ):
            stack.enter_context(patch(name, return_value=value))
        stack.enter_context(
            patch(
                "helion._compiler.cross_loop_dsmem.get_cluster_capacity",
                side_effect=_cluster_capacity,
            )
        )
        yield


def _cluster_capacity(device: object, cluster_size: int) -> int:
    """A model of ``launcher.get_cluster_capacity`` on a 148-SM GPU whose GPCs
    all divide into clusters of any size (the driver's answer is smaller on
    real parts: a GPC's leftover SMs stay idle)."""
    return 148 // cluster_size * cluster_size


def interpret_triton_selection(
    source: str,
    inputs: tuple[object, ...],
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_ops: set[str] | None = None,
) -> object:
    """Execute the exact generated body through Triton's CPU interpreter.

    ``source`` is written to ``path`` and imported; its last function, the
    host wrapper, runs on ``inputs``. The names of the ``sort`` and ``topk``
    helpers the kernel calls are added to ``selected_ops``.
    """
    from triton.runtime.interpreter import InterpretedFunction
    from triton.runtime.interpreter import _patch_lang
    from triton.runtime.jit import JITFunction

    path.write_text(source)
    spec = importlib.util.spec_from_file_location("selection_interpreter_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    def launch(
        kernel: JITFunction, grid: tuple[int, ...], *args: object, **kwargs: object
    ) -> object:
        # The standard library was imported before interpreter mode. Dispatch
        # its existing JIT helpers through the same interpreter, without
        # replacing top-k/sort arithmetic with a reference implementation.
        def device_call(
            function: JITFunction, *values: object, **options: object
        ) -> object:
            if selected_ops is not None and function.fn.__name__ in ("sort", "topk"):
                selected_ops.add(function.fn.__name__)
            interpreted = InterpretedFunction(function.fn)
            # InterpretedFunction.__call__ patches each helper's language
            # namespace without restoring it. Helpers may import tl.core,
            # which the outer grid's tl patch does not cover.
            scope = _patch_lang(function.fn)
            try:
                return interpreted.rewrite()(*values, **options)
            finally:
                scope.restore()

        with patch.object(JITFunction, "__call__", device_call):
            return InterpretedFunction(kernel.fn)[grid](*args, **kwargs)

    function = next(
        node.name
        for node in reversed(ast.parse(source).body)
        if isinstance(node, ast.FunctionDef)
    )
    return vars(module)[function](*inputs, _launcher=launch)
