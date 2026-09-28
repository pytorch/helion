from __future__ import annotations

import ast
from dataclasses import dataclass
import inspect
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from helion._testing import skipUnlessBackends
from helion.runtime.cute import launcher

pytestmark = skipUnlessBackends(["cute"])


@pytest.fixture(autouse=True)
def no_cuda_initialization():
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        yield


def _offset_view(dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(17, dtype=dtype)[1:]


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32, torch.int64, torch.int8, torch.bool]
)
def test_element_alignment_metadata(dtype: torch.dtype) -> None:
    tensor = _offset_view(dtype)
    assert launcher._cute_tensor_pointer_alignment(SimpleNamespace(), tensor) == 16
    kernel = SimpleNamespace(_helion_cute_pointer_alignment=1)
    assert (
        launcher._cute_tensor_pointer_alignment(kernel, tensor) == tensor.element_size()
    )


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32, torch.int64, torch.int8, torch.bool]
)
def test_real_pointer_marshalling_accepts_element_aligned_views(
    dtype: torch.dtype,
) -> None:
    tensor = _offset_view(dtype)
    kernel = SimpleNamespace(_helion_cute_pointer_alignment=1)
    with (
        patch.object(launcher, "_validate_cute_launcher_tensor"),
        patch.object(
            launcher, "_cute_kernel_param_is_constexpr", return_value=(False,)
        ),
    ):
        entry = launcher._build_cute_schema_and_args(kernel, (tensor,), (1, 1, 1))
        assert len(entry.launch_args) == 4
        with pytest.raises(AssertionError, match="align"):
            launcher._build_cute_schema_and_args(
                SimpleNamespace(), (tensor,), (1, 1, 1)
            )


def test_alignment_participates_in_every_cache_key() -> None:
    kernel = SimpleNamespace(_helion_cute_source_hash="alignment-test")
    tensor = torch.empty(16, dtype=torch.bfloat16)
    grid, block = (1, 1, 1), (128, 1, 1)
    schema = (("tensor", "torch.bfloat16", 1),)
    with (
        patch.object(launcher, "_validate_cute_launcher_tensor"),
        patch.object(
            launcher, "_cute_kernel_param_is_constexpr", return_value=(False,)
        ),
    ):
        original = launcher._cute_launch_arg_cache_key(kernel, (tensor,), grid)
        guard = launcher._cute_last_launch_arg_guard(kernel, (tensor,), grid)
        disk = launcher._cute_disk_cache_key(kernel, schema, block, (), None, None)
        assert guard.matches(kernel, (tensor,), grid)
        kernel._helion_cute_pointer_alignment = 1
        assert launcher._cute_launch_arg_cache_key(kernel, (tensor,), grid) != original
        assert not guard.matches(kernel, (tensor,), grid)
        assert (
            launcher._cute_disk_cache_key(kernel, schema, block, (), None, None) != disk
        )
    first = launcher._cute_compiled_launcher_discriminator(schema, block, None, None)
    second = launcher._cute_compiled_launcher_discriminator(
        schema, block, None, None, pointer_alignment=1
    )
    assert first[0] != second[0]


def _fastpath(tensor: torch.Tensor, alignment: int) -> launcher._CuteFastRelaunch:
    return launcher._CuteFastRelaunch(
        executor=SimpleNamespace(run_compiled_program=lambda args: tuple(args)),
        exe_args=[tensor.data_ptr()],
        tensor_guards=(
            (
                0,
                tensor.device.type,
                tensor.device.index,
                tensor.dtype,
                tuple(tensor.shape),
                tensor.stride(),
            ),
        ),
        scalar_guards=(),
        constexpr_flags=(False,),
        tensor_slots=((0, 0, None),),
        by_ref_writers=[],
        by_val_slots=[],
        arg_count=1,
        grid=(1, 1, 1),
        block=(128, 1, 1),
        compile_options=None,
        device_index=0,
        last_raw=7,
        keepalive=(),
        pointer_alignment=alignment,
    )


def test_fast_relaunch_checks_alignment_and_patches_fresh_view_pointer() -> None:
    aligned = torch.empty(16, dtype=torch.bfloat16)
    offset = _offset_view(torch.bfloat16)
    grid, block = (1, 1, 1), (128, 1, 1)
    ordinary = _fastpath(aligned, 16)
    guarded = _fastpath(aligned, 1)
    with patch.object(torch._C, "_cuda_getCurrentRawStream", return_value=7):
        assert ordinary.try_launch((offset,), grid, block, None) == (False, None)
        assert guarded.try_launch((offset,), grid, block, None, 1) == (
            True,
            (offset.data_ptr(),),
        )
        assert guarded.try_launch((offset,), grid, block, None, 16) == (False, None)


class _PretendCudaTensor(torch.Tensor):
    """CPU storage with CUDA metadata, used only for pointer-marshalling tests."""

    @property
    def device(self) -> torch.device:  # pyrefly: ignore[bad-override]
        return torch.device("cuda:0")


def test_fast_relaunch_probe_clones_use_kernel_alignment() -> None:
    tensor = _offset_view(torch.bfloat16).as_subclass(_PretendCudaTensor)
    records: list[int] = []

    @dataclass
    class Pointer:
        value: int

    def make_ptr(
        _dtype: object, address: int, _space: object, *, assumed_align: int
    ) -> Pointer:
        assert address % assumed_align == 0
        records.append(assumed_align)
        return Pointer(address)

    def marshal(args: tuple[object, ...], kwargs: dict[str, object]):
        values = [
            arg.value if isinstance(arg, Pointer) else int(cast("Any", arg))
            for arg in args
        ]
        return values, ()

    kernel = SimpleNamespace(_helion_cute_pointer_alignment=1)
    launch = launcher._CuteLaunchArgCacheEntry(
        schema=(),
        launch_args=(Pointer(tensor.data_ptr()),),
        grouped_static_metadata=(),
        owned_tensors=(),
    )
    compiled = launcher._CompiledCuteLauncher(None, None)
    compiled._compiled = SimpleNamespace(
        execution_args=SimpleNamespace(generate_execution_args=marshal),
        _default_executor=SimpleNamespace(
            run_compiled_program=lambda args: tuple(args)
        ),
    )
    with (
        patch.object(
            launcher,
            "_get_cute_launcher_imports",
            return_value=("gmem", make_ptr, None),
        ),
        patch.object(
            launcher, "_cute_kernel_param_is_constexpr", return_value=(False,)
        ),
        patch.object(torch._C, "_cuda_getCurrentRawStream", return_value=7),
    ):
        state = launcher._cute_build_fast_relaunch(
            kernel, (tensor,), (1, 1, 1), (128, 1, 1), None, launch, compiled
        )
    assert state is not None
    assert state.pointer_alignment == 1
    assert records == [2, 2]


def test_every_pointer_constructor_uses_metadata() -> None:
    tree = ast.parse(inspect.getsource(launcher))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "make_ptr"
    ]
    assert len(calls) == 4
    for call in calls:
        argument = next(
            keyword.value for keyword in call.keywords if keyword.arg == "assumed_align"
        )
        assert isinstance(argument, ast.Call)
        assert isinstance(argument.func, ast.Name)
        assert argument.func.id == "_cute_tensor_pointer_alignment"
