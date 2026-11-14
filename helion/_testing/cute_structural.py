"""Shared CPU setup, source loading and cases for structural AOT tests."""

from __future__ import annotations

import ast
import importlib.util
import inspect
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._hardware import HardwareInfo
from helion._testing import cpu_render_target
from helion._testing import import_example
from helion._testing.cute_codegen import tcgen05_mma_support
from helion._testing.cute_structural_kernels import _joint_sum
from helion._testing.cute_structural_kernels import _matmul
from helion._testing.cute_structural_kernels import _offset_mm
from helion._testing.cute_structural_kernels import _pointwise
from helion._testing.cute_structural_kernels import _se_args
from helion.autotuner.aot_cache import AOTAutotuneCache
from helion.autotuner.base_search import BaseSearch
from helion.runtime.cute.structural_policy import CUTE_STRUCTURAL_SETTINGS
from helion.runtime.cute.structural_policy import CuteStructuralPolicy

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from pathlib import Path
    from types import ModuleType
    from typing import Any

    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

matmul_bf16_int4 = import_example("int4_gemm").matmul_bf16_int4
squeeze_and_excitation_net_fwd = import_example(
    "squeeze_and_excitation_net"
).squeeze_and_excitation_net_fwd


def _pointwise_bound(policy: CuteStructuralPolicy | None) -> BoundKernel[Any]:
    return helion.kernel(
        _pointwise,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_structural_policy=policy,
    )._bind_isolated((torch.empty(37),))


def _case(
    setting: str,
) -> tuple[Callable[..., Any], tuple[object, ...], str]:
    if setting == "cute_region_fission":
        return squeeze_and_excitation_net_fwd.fn, _se_args(), "fission"
    if setting == "cute_full_slice_matmul_tiling":
        return (
            _matmul,
            (
                torch.empty((32, 16), dtype=torch.float16),
                torch.empty((16, 64), dtype=torch.float16),
            ),
            "_helion_fullslice_k",
        )
    if setting == "cute_segmented_matmul_tiling":
        return (
            _offset_mm,
            (
                torch.empty((37, 16), dtype=torch.bfloat16),
                torch.empty((16, 24), dtype=torch.bfloat16),
                torch.tensor([0, 0, 1, 19, 37], dtype=torch.int64),
            ),
            "_helion_segment_group",
        )
    if setting == "cute_flatten_nested_reductions":
        return (
            _joint_sum,
            (torch.empty((37, 69)), torch.tensor([0, 0, 1, 19, 37], dtype=torch.int64)),
            "_helion_flat_reduction",
        )
    assert setting == "cute_materialize_transformed_operands"
    return (
        matmul_bf16_int4.fn,
        (
            torch.empty((16, 40), dtype=torch.bfloat16),
            torch.empty((20, 24), dtype=torch.int8),
        ),
        "_helion_materialized_operand",
    )


def _host(bound: BoundKernel[Any]) -> str:
    assert bound.host_function is not None
    return ast.dump(ast.Module(body=bound.host_function.body, type_ignores=[]))


def _replay(decorator: str, function: Callable[..., Any]) -> Kernel[Any]:
    # Evaluate the real public decorator: config construction and Settings
    # precedence must work without injecting missing replay-only kwargs.
    return eval(decorator.removeprefix("@"), {"helion": helion, "torch": torch})(
        function
    )


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for name in CUTE_STRUCTURAL_SETTINGS:
        monkeypatch.delenv(f"HELION_{name.upper()}", raising=False)
    monkeypatch.delenv("HELION_BACKEND", raising=False)
    monkeypatch.setenv("HELION_AUTOTUNE_RANDOM_SEED", "0")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    # Share one undo stack with test bodies so their changes cannot restore
    # fixture-local AOT settings after this fixture has finished.
    monkeypatch.setenv("HELION_CACHE_DIR", str(tmp_path / "local"))
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(tmp_path / "aot"))
    monkeypatch.setenv("HELION_AOT_MODE", "collect")
    monkeypatch.delenv("HELION_SKIP_CACHE", raising=False)
    AOTAutotuneCache.clear_caches()
    try:
        with (
            cpu_render_target(mma_support=tcgen05_mma_support()),
            patch(
                "helion.autotuner.aot_cache.get_hardware_info",
                return_value=HardwareInfo(
                    device_kind="cuda",
                    hardware_name="CPU SM100 proof",
                    runtime_version="13.0",
                    compute_capability="sm100",
                ),
            ),
        ):
            yield
    finally:
        AOTAutotuneCache.clear_caches()
        torch.set_num_threads(previous_threads)


def _search(policy: CuteStructuralPolicy | None) -> BaseSearch:
    bound = _pointwise_bound(policy)
    return BaseSearch(bound, (torch.empty(37),))


def _module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _function(tmp_path: Path) -> Callable[..., Any]:
    source = tmp_path / "original.py"
    source.write_text(
        "import torch\nimport helion.language as hl\n" + inspect.getsource(_pointwise)
    )
    return _module(source)._pointwise
