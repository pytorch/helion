"""Exercise the real bound host and production launcher, stopping before CUDA."""

from __future__ import annotations

import ast
import inspect
import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode

pytest.importorskip("cuda.bindings.driver")

from cuda.bindings.driver import CUstream

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_host_paired_sum_seeds import _original

import helion
from helion._compiler.cute.paired_sum_runtime import paired_sum_cast_kernel
from helion.runtime.cute import launcher
from helion.runtime.cute import paired_sum

CUDA_0_DEVICE = "cuda:0"

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


_requires_pinned_aten_build = pytest.mark.skipif(
    torch.version.git_version != paired_sum._ATEN_SUM_GIT,
    reason="host sum ATen reuse requires the pinned PyTorch build",
)


@pytest.fixture(scope="module", autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch(
            "torch.cuda._exchange_device", side_effect=AssertionError("GPU forbidden")
        ),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def original_config(layout: str) -> helion.Config:
    return helion.Config(
        block_sizes=[4, 1],
        num_threads=[1, 1, 128],
        load_eviction_policies=[""] * 5,
        cute_vector_widths=[4, 1, 8],
        cute_lane_layouts=["blocked", "strided", "blocked"],
        cute_reduction_reloads=["register"],
        cute_reduction_schedule="pipelined",
        cute_reduction_group_rows=4,
        cute_cluster_n=1,
        cute_min_blocks_per_mp=0,
        cute_host_paired_sum=layout,
    )


@_requires_pinned_aten_build
@pytest.mark.parametrize("layout", ("off", "mapped", "narrow"))
def test_actual_bound_host_production_factory_and_pointer_stream_reuse(
    layout: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cuda_initialized_at_entry = torch.cuda.is_initialized()
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    bound, originals = _original(True)
    bound.set_config(original_config(layout))
    host = bound._run
    assert host is not None and bound._config is not None
    code = bound.to_code(original_config(layout))
    producer_name = next(
        node.name
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.decorator_list
    )
    producer = host.__globals__[producer_name]
    for kernel in (producer, paired_sum_cast_kernel):
        for key, value in (
            ("_helion_cute_compiled_launchers", {}),
            ("_helion_cute_launch_arg_cache", {}),
            ("_helion_cute_last_launch_cache", None),
            ("_helion_cute_fastpath", None),
        ):
            monkeypatch.setattr(kernel, key, value, raising=False)
    addresses = {}
    real_pointer = torch.Tensor.data_ptr

    def pointer(tensor: torch.Tensor) -> int:
        if not isinstance(tensor, FakeTensor):
            return real_pointer(tensor)
        storage = tensor.untyped_storage()._cdata
        addresses.setdefault(storage, (len(addresses) + 1) * 0x1000000)
        return addresses[storage] + tensor.storage_offset() * tensor.element_size()

    factories, launches, host_calls = [], [], []
    create = launcher._create_cute_wrapper

    def factory(*args: object, **kwargs: object) -> object:
        assert kwargs == {"num_sm": 148}
        wrapper = create(*args, **kwargs)
        factories.append(
            {
                "kernel": args[0].__name__,
                "schema": repr(args[1]),
                "block": args[2],
                "kwargs": kwargs,
                "wrapper": inspect.getsource(wrapper),
            }
        )
        return wrapper

    def compiled_call(compiled: object, *args: object) -> None:
        assert compiled._compile_options == "--enable-tvm-ffi"
        assert compiled._cache_key is not None
        launches.append({"object": id(compiled), "stream": str(args[-1])})

    stream = [CUstream(11)]

    def observe(kernel: object, grid: tuple, *values: object, **kwargs: object) -> None:
        host_calls.append((kernel, grid, values, kwargs))
        launcher.default_cute_launcher(kernel, grid, *values, **kwargs)

    # Fake tensors are admitted only by this explicit ABI-test override. The
    # independent runtime tests check the real guard declines FakeTensor/modes.
    with (
        FakeTensorMode(),
        patch.object(torch.Tensor, "data_ptr", pointer),
        patch.object(
            paired_sum, "_ordinary_tensor", lambda x: isinstance(x, FakeTensor)
        ),
        patch.object(paired_sum, "_get_current_dispatch_mode_stack", return_value=[]),
        patch.object(launcher, "get_num_sm", return_value=148),
        patch.object(launcher, "_ensure_cute_dsl_arch_env", return_value=None),
        patch.object(launcher, "_create_cute_wrapper", factory),
        patch.object(launcher._CompiledCuteLauncher, "__call__", compiled_call),
        patch.object(launcher, "_cute_current_stream", side_effect=lambda: stream[0]),
        patch.dict(host.__kwdefaults__, {"_launcher": observe}),
    ):
        inputs = []
        outputs = []
        for index in range(3):
            args = tuple(
                torch.empty_strided(
                    tensor.shape,
                    tensor.stride(),
                    dtype=tensor.dtype,
                    device=CUDA_0_DEVICE,
                    requires_grad=tensor.requires_grad,
                )
                if isinstance(tensor, torch.Tensor)
                else tensor
                for tensor in originals
            )
            inputs.append(args)
            stream[0] = CUstream(11 + index)
            outputs.append(bound(*args))
            assert bound._run is host
        expected = 1 if layout == "off" else 2
        assert len(factories) == expected
        assert len(launches) == 3 * expected
        assert len({entry["object"] for entry in launches}) == expected
        for index in range(3):
            assert all(
                entry["stream"] == str(CUstream(11 + index))
                for entry in launches[index * expected : (index + 1) * expected]
            )
        tensors = [tensor for values in outputs for tensor in values]
        assert len({tensor.untyped_storage()._cdata for tensor in tensors}) == 9
        assert all(not tensor.requires_grad for tensor in tensors)
        assert all(
            tuple(value.shape) == expected_shape
            for values in outputs
            for value, expected_shape in zip(
                values, ((256, 1024), (1024,), (1024,)), strict=True
            )
        )
        producer_calls = [call for call in host_calls if call[0] is producer]
        assert len(producer_calls) == 3
        partials = [
            value
            for call in producer_calls
            for value in call[2]
            if isinstance(value, torch.Tensor) and value.shape == (64, 1024)
        ]
        assert (
            len(partials)
            == len({tensor.untyped_storage()._cdata for tensor in partials})
            == 6
        )
        assert all(
            value.dtype == torch.float32 and not value.requires_grad
            for value in partials
        )
        if layout != "off":
            tail_calls = [
                call for call in host_calls if call[0] is paired_sum_cast_kernel
            ]
            assert len(tail_calls) == 3
            for index, call in enumerate(tail_calls):
                assert call[2][0] is partials[2 * index]
                assert call[2][1] is partials[2 * index + 1]
                assert call[2][2] is outputs[index][1]
                assert call[2][3] is outputs[index][2]
                assert call[3]["block"] == (
                    (32, 4, 1) if layout == "mapped" else (8, 4, 1)
                )
    assert torch.cuda.is_initialized() == cuda_initialized_at_entry
    (tmp_path / "actual-bound-abi.json").write_text(
        json.dumps(
            {
                "layout": layout,
                "factories": factories,
                "launches": launches,
                "fresh_pointer_sets": 3,
                "fresh_partial_buffers": len(partials),
                "fresh_outputs": len(tensors),
                "actual_bound_and_production_factory": True,
                "fake_tensor_admission_override_for_abi_only": True,
                "cuda_initialized_at_entry": cuda_initialized_at_entry,
                "cuda_initialized": torch.cuda.is_initialized(),
            },
            indent=2,
        )
        + "\n"
    )
