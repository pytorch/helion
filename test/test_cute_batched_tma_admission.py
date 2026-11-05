"""CPU admission and codegen facts for batched (leading-axis) tcgen05 operands.

Two general changes are covered:

* the exact half -> FP32 -> same-half operand round trip, already proved for
  rank-2 operands, is also recognized on rank-3 (batched) operands by the
  operand analysis that plans leading-axis matmuls;
* a batched B whose per-batch matrix is K-contiguous (column-major) is staged
  through the same permuted rank-3 TensorMap as a row-major batched B, as a
  K-major operand; a batched A must still be K-major.

Unsupported casts, layouts and output maps keep the native path refused. No
CUDA work and no native compilation happen here.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import cute_mma
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion._testing.cute_dense_tcgen05 import wrapper_plans
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl
from helion.runtime.cute import launcher

# Byte-exact native module Inductor generated for a batched matmul whose output
# is stored through a head-split permutation. It is a portable fixture, not a
# changed frontend or benchmark.
INDUCTOR_PERMUTED_STORE_BMM = """import torch
import helion
import helion.language as hl

from torch._inductor.runtime.helion import HelionKernelWrapper
@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full", ref_mode=helion.RefMode.OFF)
def helion_op0_op1(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([512, 512, 64]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(512):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        tmp4 = hl.full([], 0.3535533905932738, dtype=torch.float32)
        tmp5 = acc * tmp4
        out0[tile_z.index.reshape([tile_z, 1, 1]) // 64, ((tile_y.index.reshape([1, tile_y, 1]) + 512*tile_z.index.reshape([tile_z, 1, 1])) % 512), (tile_z.index.reshape([tile_z, 1, 1]) % 64), ((tile_x.index.reshape([1, 1, tile_x]) + 64*tile_y.index.reshape([1, tile_y, 1]) + 32768*tile_z.index.reshape([tile_z, 1, 1])) % 64)] = tmp5.to(torch.bfloat16)
helion_op0_op1 = HelionKernelWrapper(helion_op0_op1)

def call_helion_op0_op1(arg0_1: torch.Tensor, arg1_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op0_op1(arg0_1, arg1_1, buf1)
"""


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def bmm_round_trip(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    z, m, k = A.size()
    n = B.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            lhs = A[tz, ty, tr].to(torch.float32).to(torch.bfloat16)
            rhs = B[tz, tr, tx].to(torch.float32).to(torch.bfloat16)
            acc = torch.baddbmm(acc, lhs, rhs)
        out[tz, ty, tx] = (acc * 0.5).to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def bmm_direct(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    z, m, k = A.size()
    n = B.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        out[tz, ty, tx] = (acc * 0.5).to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def bmm_cast_with_arithmetic(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor
) -> None:
    z, m, k = A.size()
    n = B.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            lhs = (A[tz, ty, tr].to(torch.float32) * 1.0).to(torch.bfloat16)
            rhs = B[tz, tr, tx].to(torch.float32).to(torch.bfloat16)
            acc = torch.baddbmm(acc, lhs, rhs)
        out[tz, ty, tx] = (acc * 0.5).to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def bmm_lossy_cast(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    z, m, k = A.size()
    n = B.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            lhs = A[tz, ty, tr].to(torch.float32).to(torch.float16)
            rhs = B[tz, tr, tx].to(torch.float32).to(torch.float16)
            acc = torch.baddbmm(acc, lhs, rhs)
        out[tz, ty, tx] = (acc * 0.5).to(torch.bfloat16)


Z, M, N, K = 4, 256, 128, 256


def _a_row() -> torch.Tensor:
    return torch.empty((Z, M, K), dtype=torch.bfloat16)


def _a_col() -> torch.Tensor:
    return torch.empty_strided((Z, M, K), (M * K, 1, M), dtype=torch.bfloat16)


def _b_k_major() -> torch.Tensor:
    return torch.empty_strided((Z, K, N), (K * N, 1, K), dtype=torch.bfloat16)


def _b_row() -> torch.Tensor:
    return torch.empty((Z, K, N), dtype=torch.bfloat16)


def _b_k_major_misaligned() -> torch.Tensor:
    # Each column is K + 1 elements apart: 514 bytes, not a 16-byte multiple.
    storage = torch.empty(Z * (K + 1) * N + 8, dtype=torch.bfloat16)
    return torch.as_strided(storage, (Z, K, N), ((K + 1) * N, 1, K + 1))


def _b_row_misaligned() -> torch.Tensor:
    storage = torch.empty(Z * K * (N + 1) + 8, dtype=torch.bfloat16)
    return torch.as_strided(storage, (Z, K, N), (K * (N + 1), N + 1, 1))


def _out() -> torch.Tensor:
    return torch.empty((Z, M, N), dtype=torch.bfloat16)


def _bind(kernel: helion.Kernel, args: tuple[torch.Tensor, ...]):
    fresh = helion.kernel(
        kernel.fn,
        settings=kernel.settings,
        cute_structural_policy=kernel.cute_structural_policy,
    )
    with patch.object(fresh, "_bind_isolated", side_effect=fresh.bind):
        return cpu_bind(fresh, args)


def _tcgen05_seeds(bound) -> list[helion.Config]:
    return [
        seed
        for seed in bound.config_spec.compiler_seed_configs
        if any(key.startswith("tcgen05_") for key in seed.config)
    ]


def _mma_nodes(bound):
    device_ir = bound.host_function.device_ir
    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            target = cute_mma._decode_cute_mma_target(node, device_ir=device_ir)
            if target is not None:
                yield node, target


def _ab_plan(code: str) -> dict[str, object] | None:
    try:
        plans = wrapper_plans(code)
    except StopIteration:
        return None
    ab = [plan for plan in plans if plan.get("kind") == "tcgen05_ab_tma"]
    assert len(ab) <= 1
    return ab[0] if ab else None


def _rendered_ab_wrapper(plan: dict[str, object]) -> str:
    body: list[str] = []
    launcher._append_cute_wrapper_plan(body, [], plan, num_sm=148)
    return "\n".join(body)


@pytest.mark.parametrize(
    ("kernel", "b_factory", "b_k_major"),
    [
        pytest.param(bmm_round_trip, _b_k_major, True, id="round_trip_k_major_b"),
        pytest.param(bmm_round_trip, _b_row, False, id="round_trip_row_major_b"),
        pytest.param(bmm_direct, _b_k_major, True, id="direct_k_major_b"),
        pytest.param(bmm_direct, _b_row, False, id="direct_row_major_b"),
    ],
)
def test_batched_operands_admit_native_search_and_rank3_tensormaps(
    kernel: helion.Kernel, b_factory, b_k_major: bool
) -> None:
    with cpu_codegen_target(), forbid_cuda_work():
        bound = _bind(kernel, (_a_row(), b_factory(), _out()))
        assert bound.config_spec.cute_tcgen05_search_enabled
        with bound.env, bound.host_function:
            ConfigGeneration(bound.config_spec).initial_population_view()
            ((node, target),) = list(_mma_nodes(bound))
            env = CompileEnvironment.current()
            analysis = cute_mma._analyze_mma_operands(
                target.lhs, target.rhs, env, allow_integer_casts=True
            )
            assert analysis is not None
            assert analysis.has_leading_passthrough
            assert analysis.lhs.matrix_major == "row"
            assert analysis.rhs.matrix_major == ("col" if b_k_major else "row")
            expected_casts = 2 if kernel is bmm_round_trip else 0
            assert len(analysis.lhs.exact_round_trip_casts) == expected_casts
            assert len(analysis.rhs.exact_round_trip_casts) == expected_casts
            assert (
                cute_mma.analyze_cute_mma_node(
                    node, device_ir=bound.host_function.device_ir
                )
                is not None
            )
        seeds = _tcgen05_seeds(bound)
        assert seeds
        code = bound.to_code(seeds[0])
    plan = _ab_plan(code)
    assert plan is not None
    assert plan["lhs_tma_order"] == (1, 2, 0)
    assert plan["rhs_tma_order"] == (2, 1, 0)
    assert "a_k_major" not in plan
    assert plan.get("b_k_major") is (True if b_k_major else None)
    assert code.count("cute.gemm(") == 1
    assert "'d_leading_passthrough': True" in code
    wrapper = _rendered_ab_wrapper(plan)
    # The B TensorMap is the same permuted (N, K, L) view of the original
    # argument for both majors; only the MMA/SMEM major follows its strides.
    assert (
        "cute.make_layout((arg1_shape2, arg1_shape1, arg1_shape0), "
        "stride=(arg1_stride2, arg1_stride1, arg1_stride0))" in wrapper
    )
    assert (
        "cute.make_layout((arg0_shape1, arg0_shape2, arg0_shape0), "
        "stride=(arg0_stride1, arg0_stride2, arg0_stride0))" in wrapper
    )
    b_major = "K" if b_k_major else "MN"
    assert (
        f"cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.{b_major}, "
        in wrapper
    )
    assert ("is_k_major=True" in wrapper) is b_k_major
    assert "make_tiled_tma_atom_B(" in wrapper


def test_round_trip_proof_is_rank3_only_when_leading_axis_is_requested() -> None:
    with cpu_codegen_target(), forbid_cuda_work():
        bound = _bind(bmm_round_trip, (_a_row(), _b_k_major(), _out()))
        with bound.env, bound.host_function:
            ((_node, target),) = list(_mma_nodes(bound))
            for operand in (target.lhs, target.rhs):
                assert cute_mma._exact_half_mma_operand_round_trip(operand) is None
                proved = cute_mma._exact_half_mma_operand_round_trip(
                    operand, allow_leading_axis=True
                )
                assert proved is not None
                load, casts = proved
                assert load.target is not None
                assert [cast.args[1] for cast in casts] == [
                    torch.float32,
                    torch.bfloat16,
                ]
                # The lossless-upcast unwrap used by fallback paths stays rank-2.
                assert cute_mma._unwrap_lossless_mma_upcast(casts[0]) is casts[0]


@pytest.mark.parametrize(
    ("kernel", "a_factory", "b_factory"),
    [
        pytest.param(bmm_round_trip, _a_col, _b_k_major, id="m_major_batched_a"),
        pytest.param(
            bmm_cast_with_arithmetic, _a_row, _b_k_major, id="cast_chain_arithmetic"
        ),
        pytest.param(bmm_lossy_cast, _a_row, _b_k_major, id="unequal_cast_dtype"),
    ],
)
def test_unsupported_batched_operands_keep_native_search_off(
    kernel: helion.Kernel, a_factory, b_factory
) -> None:
    with cpu_codegen_target(), forbid_cuda_work():
        bound = _bind(kernel, (a_factory(), b_factory(), _out()))
        assert not bound.config_spec.cute_tcgen05_search_enabled
        assert not _tcgen05_seeds(bound)
        with bound.env, bound.host_function:
            ((node, target),) = list(_mma_nodes(bound))
            env = CompileEnvironment.current()
            assert (
                cute_mma._analyze_mma_operands(
                    target.lhs, target.rhs, env, allow_integer_casts=True
                )
                is None
            )
            assert (
                cute_mma.analyze_cute_mma_node(
                    node, device_ir=bound.host_function.device_ir
                )
                is None
            )


@pytest.mark.parametrize(
    "b_factory",
    [
        pytest.param(_b_k_major_misaligned, id="k_major_b"),
        pytest.param(_b_row_misaligned, id="row_major_b"),
    ],
)
def test_misaligned_batched_b_builds_no_tensormaps(b_factory) -> None:
    # The existing alignment proof keeps both A/B TensorMaps off for an
    # unaligned outer stride; a K-major B follows the same rule as a row-major B.
    with cpu_codegen_target(), forbid_cuda_work():
        bound = _bind(bmm_round_trip, (_a_row(), b_factory(), _out()))
        seeds = _tcgen05_seeds(bound)
        assert seeds
        code = bound.to_code(seeds[0])
    assert _ab_plan(code) is None
    assert "tma_atom_b" not in code


def test_k_major_batched_b_partial_tiles_stay_off_the_ragged_path() -> None:
    config = helion.Config(pid_type="flat", block_sizes=[1, 128, 96, 64])
    with cpu_codegen_target(), forbid_cuda_work():
        results = {}
        for label, b_factory in (("k_major", _b_k_major), ("row", _b_row)):
            bound = _bind(bmm_round_trip, (_a_row(), b_factory(), _out()))
            with bound.env, bound.host_function:
                ((node, _target),) = list(_mma_nodes(bound))
                candidate = cute_mma.analyze_cute_mma_node(
                    node, device_ir=bound.host_function.device_ir
                )
                assert candidate is not None
                assert cute_mma._ragged_batched_tcgen05_config(config, 96)
                operands = candidate.operands
                assert not cute_mma._mma_tiles_are_static_full(
                    operands, bm=128, bn=96, bk=64
                )
                results[label] = (
                    cute_mma._batched_tcgen05_tiles_admitted(
                        operands, config, bm=128, bn=96, bk=64
                    ),
                    cute_mma._batched_tcgen05_tiles_admitted(
                        operands, config, bm=128, bn=64, bk=64
                    ),
                )
    # Whole tiles admit both majors; partial tiles keep the validated
    # MN-major-only ragged rule.
    assert results == {"k_major": (False, True), "row": (True, True)}


def test_inductor_permuted_output_store_is_proved_by_the_batch_digit_analysis(
    tmp_path,
) -> None:
    # Before the batch-digit output proof this store was refused; it is now
    # admitted through ``mapped_output.py`` (see
    # ``test_cute_batch_digit_output_store.py`` for the lowering facts).
    from torch._inductor.codecache import PyCodeCache

    module = PyCodeCache.load(INDUCTOR_PERMUTED_STORE_BMM)
    original = module.helion_op0_op1.kernel
    args = (
        torch.empty((512, 512, 512), dtype=torch.bfloat16),
        torch.empty_strided((512, 512, 64), (32768, 1, 512), dtype=torch.bfloat16),
        torch.empty((8, 512, 64, 64), dtype=torch.bfloat16),
    )
    with cpu_codegen_target(), forbid_cuda_work():
        bound = _bind(original, args)
        with bound.env, bound.host_function:
            ConfigGeneration(bound.config_spec).initial_population_view()
            ((node, target),) = list(_mma_nodes(bound))
            env = CompileEnvironment.current()
            analysis = cute_mma._analyze_mma_operands(
                target.lhs, target.rhs, env, allow_integer_casts=True
            )
            assert analysis is not None
            assert len(analysis.lhs.exact_round_trip_casts) == 2
            assert len(analysis.rhs.exact_round_trip_casts) == 2
            assert (analysis.lhs.matrix_major, analysis.rhs.matrix_major) == (
                "row",
                "col",
            )
            graphs = list(bound.host_function.device_ir.graphs)
            stores = cute_mma._analyze_mma_output_stores(node, analysis, graphs=graphs)
            assert stores is not None and stores.batch_digit_output is not None
            assert (
                cute_mma.analyze_cute_mma_node(
                    node, device_ir=bound.host_function.device_ir
                )
                is not None
            )
        assert bound.config_spec.cute_tcgen05_search_enabled
