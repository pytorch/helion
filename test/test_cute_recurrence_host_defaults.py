"""Offline recurrence hosts leave shared-memory preferences to the SDK."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import pytest
import torch

from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_recurrence_host import chunk_runtime_plan
from helion._testing.cute_recurrence_host import chunk_runtime_view_args
from helion._testing.cute_recurrence_host import compile_host_offline
from helion._testing.cute_recurrence_host import gdn_test_wrapper
from helion.runtime.cute.launcher import _create_cute_wrapper
from helion.runtime.cute.sass_check import _cubin_blobs

pytestmark = pytest.mark.cpu_only


def test_chunk_recurrence_dv2_host_uses_sdk_carveout_default() -> None:
    def dummy_kernel(*_args: object) -> None:
        pass

    plan = chunk_runtime_plan()
    plan.update(
        {
            "kind": "chunk_recurrence_sm100",
            "threads": 512,
            "device_abi": 2,
            "input_stages": 8,
            "tma_stages": 6,
            "factor_tma_value_splits": 2,
            "output_acc_stages": 2,
            "output_smem_stages": 7,
            "output_store_wait_groups": 6,
            "tmem_cols": 512,
            "dv_partitions": 2,
            "smem_bytes": 138240,
            "scale_idx": 10,
        }
    )
    dummy_kernel._helion_cute_wrapper_plans = [plan]  # type: ignore[attr-defined]
    tensors = chunk_runtime_view_args()
    schema = (
        *(
            ("tensor", str(value.dtype), value.ndim, tuple(value.shape), value.stride())
            for value in tensors
        ),
        ("scalar", "float"),
    )
    wrapper = _create_cute_wrapper(dummy_kernel, schema, (512, 1, 1))
    dtypes = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
        torch.int32: cutlass.Int32,
    }
    pointers = [
        make_ptr(dtypes[value.dtype], 0, cute.AddressSpace.gmem, assumed_align=16)
        for value in tensors
    ]
    with cpu_target(forbid_native=False):
        compiled = compile_host_offline(
            wrapper,
            *pointers,
            1.0,
            1,
            12,
            2,
            cuda.CUstream(0),
        )
    assert "preferred_shared_memory_carveout" not in str(compiled.ir_module)
    assert len(_cubin_blobs(compiled.ir_module)) == 1


def test_gdn_recurrence_host_uses_sdk_carveout_default() -> None:
    pointers = [
        make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
        for dtype in (
            cutlass.BFloat16,
            cutlass.BFloat16,
            cutlass.BFloat16,
            cutlass.Float32,
            cutlass.BFloat16,
        )
    ]
    with cpu_target(forbid_native=False):
        compiled = compile_host_offline(
            gdn_test_wrapper(),
            *pointers,
            1,
            8,
            2,
            cuda.CUstream(0),
        )
    assert "preferred_shared_memory_carveout" not in str(compiled.ir_module)
    assert len(_cubin_blobs(compiled.ir_module)) == 1
