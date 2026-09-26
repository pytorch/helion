from __future__ import annotations

import ast

from examples.moe_matmul_ogs import moe_matmul_ogs
import pytest
import torch

from test._cute_binding import _cpu_bind

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._testing import skipUnlessBackends
from helion.runtime.cute.launcher import _append_cute_wrapper_plan
from helion.runtime.cute.launcher import _cute_wrapper_plan_bakes_tensor_shapes


def _bound(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    static: bool,
    *,
    lhs_offset: int = 0,
):
    rows, k, n, groups = shape
    lhs_storage = torch.empty(rows * k + lhs_offset, dtype=dtype)
    inputs = (
        lhs_storage[lhs_offset:].view(rows, k),
        torch.empty((groups, k, n), dtype=dtype),
        torch.empty(groups, dtype=torch.int32),
        torch.empty(groups + 1, dtype=torch.int32),
        torch.empty(rows, dtype=torch.int32),
        rows // groups + 32,
    )
    kernel = helion.kernel(
        moe_matmul_ogs.fn,
        backend="cute",
        static_shapes=static,
        autotune_effort="none",
    )
    return _cpu_bind(kernel, inputs)


@pytest.mark.parametrize(
    "shape", [(256, 128, 128, 4), (1024, 512, 512, 8), (4096, 1024, 1024, 16)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("static", [False, True])
@skipUnlessBackends(["cute"])
def test_original_gathered_contraction_emits_pipeline(shape, dtype, static) -> None:
    bound = _bound(shape, dtype, static)
    source = bound.to_code(
        helion.Config(
            block_sizes=[128, 32, 64],
            num_threads=[4, 32, 1],
            cute_vector_widths=[1, 1, 1, 1],
            cute_collective_mma=True,
            cute_collective_compute="tma_gather",
            cute_gathered_mma_n=512,
            cute_gathered_mma_stages=2,
        )
    )
    assert "gathered_mma_tma" in source
    assert "gather_four_rows" in source
    assert "num_threads=288" in source
    assert "_helion_pending_collective_mma" not in source
    assert "torch.zeros(" in source
    # The final store retains its row predicate and complete/tail vector check.
    assert "gathered_logical_row" in source
    assert "% 8 == 0" in source
    ast.parse(source)


@skipUnlessBackends(["cute"])
def test_gathered_seeds_keep_wide_columns_and_pipeline_depth() -> None:
    bound = _bound((1024, 512, 512, 8), torch.float16, False)
    assert bound.host_function is not None
    with bound.env:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    gathered = [
        seed
        for seed in seeds
        if seed.config.get("cute_collective_compute") == "tma_gather"
    ]
    assert {
        (seed["cute_gathered_mma_n"], seed["cute_gathered_mma_stages"])
        for seed in gathered
    } == {(128, 2), (128, 3), (128, 4), (256, 2), (256, 3), (256, 4), (512, 2)}
    for seed in gathered:
        normalized = helion.Config.from_dict(dict(seed.config))
        bound.config_spec.normalize(normalized)
        assert normalized["cute_collective_compute"] == "tma_gather"
        assert normalized["cute_gathered_mma_n"] == seed["cute_gathered_mma_n"]
        assert (
            normalized["cute_gathered_mma_stages"] == seed["cute_gathered_mma_stages"]
        )


def test_gathered_wrapper_uses_runtime_m_extent_and_exact_tensor_shapes() -> None:
    plan: dict[str, object] = {
        "kind": "gathered_mma_tma",
        "lhs_idx": 0,
        "rhs_idx": 1,
        "out_idx": 2,
        "m_extent_idx": 3,
        "groups": 4,
        "n_size": 128,
        "bn": 512,
        "stages": 2,
        "grid_cap": 65535,
        "kernel_args": ["aa", "ba", "bt", "mma", "al", "bl"],
    }
    statements = []
    arguments = []
    _append_cute_wrapper_plan(statements, arguments, plan)
    text = "\n".join(statements)
    assert "cutlass.Int32(arg3)" in text
    assert "_helion_make_gathered_tma(arg0, arg1, 512, 2)" in text
    assert arguments == plan["kernel_args"]
    assert _cute_wrapper_plan_bakes_tensor_shapes(plan)


def test_oversized_gathered_pipeline_is_rejected_by_wrapper() -> None:
    plan: dict[str, object] = {
        "kind": "gathered_mma_tma",
        "bn": 512,
        "stages": 3,
        "kernel_args": ["aa", "ba", "bt", "mma", "al", "bl"],
    }
    with pytest.raises(helion.exc.BackendUnsupported, match="wrapper geometry"):
        _append_cute_wrapper_plan([], [], plan)


@skipUnlessBackends(["cute"])
def test_gathered_mode_requires_collective_admission() -> None:
    bound = _bound((256, 128, 128, 4), torch.float16, False)
    with pytest.raises(helion.exc.BackendUnsupported, match="admitted collective"):
        bound.to_code(
            helion.Config(
                block_sizes=[128, 32, 64],
                num_threads=[4, 32, 1],
                cute_vector_widths=[1, 1, 1, 1],
                cute_collective_mma=False,
                cute_collective_compute="tma_gather",
            )
        )


@skipUnlessBackends(["cute"])
def test_gathered_mode_rejects_unaligned_input_after_binding() -> None:
    bound = _bound((256, 128, 128, 4), torch.float16, False, lhs_offset=1)
    with pytest.raises(helion.exc.BackendUnsupported, match="unproved gathered"):
        bound.to_code(
            helion.Config(
                block_sizes=[128, 32, 64],
                num_threads=[4, 32, 1],
                cute_vector_widths=[1, 1, 1, 1],
                cute_collective_mma=True,
                cute_collective_compute="tma_gather",
            )
        )
