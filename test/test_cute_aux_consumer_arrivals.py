from __future__ import annotations

import ast

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from test.test_cute_aux_read_fence import _assert_aux_read_fence_order
from test.test_cute_aux_read_fence import _rank_two_inputs
from test.test_cute_aux_read_fence import _rank_two_multi_aux
from test.test_cute_batched_aux_tma import _config
from test.test_cute_batched_aux_tma import _cpu_codegen

from helion._compiler.cute.cute_mma import _emit_tcgen05_aux_pipeline_setup
from helion._compiler.cute.cute_mma import _Tcgen05AuxPerDescriptorRingNames
from helion._compiler.cute.cute_mma import _Tcgen05AuxPipelinePlan
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


def _group_size(code: str, name: str) -> int:
    for node in ast.walk(ast.parse(code)):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            assert isinstance(node.value, ast.Call)
            count = node.value.args[1]
            assert isinstance(count, ast.Call)
            return ast.literal_eval(count.args[0])
    raise AssertionError(f"Missing cooperative group {name}")


@pytest.mark.parametrize("epi_warps", [1, 2, 4, 8])
@pytest.mark.parametrize("tma", [False, True])
def test_aux_consumer_count_derived_from_epilogue_warps(
    epi_warps: int, tma: bool
) -> None:
    ring = _Tcgen05AuxPerDescriptorRingNames(
        smem_layout="ring_layout",
        smem_ptr="ring_ptr",
        smem="ring",
        tma_atom="tma_atom" if tma else None,
        tma_tensor="tma_tensor" if tma else None,
    )
    plan = _Tcgen05AuxPipelinePlan(
        barriers="barriers",
        producer_group="producers",
        consumer_group="consumers",
        pipeline="pipeline",
        producer_state="producer_state",
        consumer_state="consumer_state",
        rings=(ring,),
        use_tma_load=tma,
        stage_count=2,
        epi_tile_var="epi_tile",
    )
    code = "\n".join(
        ast.unparse(node)
        for node in _emit_tcgen05_aux_pipeline_setup(
            plan,
            descriptor_dtype_strs=("cutlass.BFloat16",),
            tile_shape_expr="epi_tile",
            c_input_warp_thread_count=32,
            epi_warp_count=epi_warps,
            defer_sync=False,
        )
    )
    assert _group_size(code, "consumers") == epi_warps * (1 if tma else 32)
    if not tma:
        assert _group_size(code, "producers") == 32
        assert "PipelineAsync.create" in code
    else:
        assert "PipelineTmaAsync.create" in code


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("fringe", [False, True])
@pytest.mark.parametrize("bn", [128, 256])
def test_simt_aux_arrival_codegen(
    dtype: torch.dtype, mixed: bool, fringe: bool, bn: int
) -> None:
    args = list(_rank_two_inputs(2, m=513 if fringe else 512, n=8192, dtype=dtype))
    if mixed:
        args[-1] = torch.empty_like(args[-1], dtype=torch.float32)
    with _cpu_codegen():
        code = _rank_two_multi_aux._bind_isolated(tuple(args)).to_code(
            _config(block_sizes=[128, bn, 64], tcgen05_aux_load_mode="simt")
        )
    assert _group_size(code, "tcgen05_aux_pipeline_consumer_group") == 4 * 32
    assert _group_size(code, "tcgen05_aux_pipeline_producer_group") == 32
    assert _assert_aux_read_fence_order(code) > 0
    # The existing per-reader fence is immediately followed by the direct
    # release: neither election nor an extra warp sync may intervene.
    assert "PipelineAsync.create" in code
    assert "PipelineTmaAsync.create" not in code


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_tma_aux_keeps_elected_release(dtype: torch.dtype) -> None:
    with _cpu_codegen():
        code = _rank_two_multi_aux._bind_isolated(
            _rank_two_inputs(2, dtype=dtype)
        ).to_code(_config(block_sizes=[128, 128, 64], tcgen05_aux_load_mode="tma"))
    assert _group_size(code, "tcgen05_aux_pipeline_consumer_group") == 4
    assert _assert_aux_read_fence_order(code, use_tma_load=True) > 0
    assert "PipelineTmaAsync.create" in code


@pytest.mark.parametrize("fringe", [False, True])
def test_direct_aux_has_no_empty_barrier_arrival(fringe: bool) -> None:
    with _cpu_codegen():
        code = _rank_two_multi_aux._bind_isolated(
            _rank_two_inputs(2, m=513 if fringe else 512, n=8192)
        ).to_code(
            _config(
                block_sizes=[128, 128, 64],
                tcgen05_aux_load_mode="simt",
                tcgen05_warp_spec_c_input_warps=0,
            )
        )
    assert "tcgen05_aux_pipeline" not in code
    assert _assert_aux_read_fence_order(code) == 0
