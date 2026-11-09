"""Codegen and search-surface pins for the ``warp_mma`` matmul family.

CPU-only (the fake CUDA target of the module's ``cpu_render`` marker), with
the binders of ``test_cute_tcgen05_fixed_cost.py``: every test binds a kernel,
pins a config and inspects the generated CuTe source or the config surface.
The runtime numerics (sm_100 GPU) are at the end of the module.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from test.test_cute_tcgen05_fixed_chain import CPU_DEVICE
from test.test_cute_tcgen05_fixed_chain import _batched_gemm
from test.test_cute_tcgen05_fixed_chain import _bind_bmm
from test.test_cute_tcgen05_fixed_chain import _bind_gemm
from test.test_cute_tcgen05_fixed_chain import _bind_two_output_gemm
from test.test_cute_tcgen05_fixed_chain import _kernel_body
from test.test_cute_tcgen05_fixed_chain import _ordinary_gemm
from test.test_cute_tcgen05_fixed_chain import _source
from test.test_cute_tcgen05_fixed_chain import _two_output_gemm
from test.test_cute_tcgen05_fixed_cost import _bind_row_bias_gemm
from test.test_cute_tcgen05_fixed_cost import _closure_bias_epilogue
from test.test_cute_tcgen05_fixed_cost import _gemm_with_epilogue
from test.test_cute_tcgen05_fixed_cost import _row_bias_gemm

import helion
from helion._compiler.cute import cute_warp_mma_gemm as family
from helion._compiler.cute.state import cute_env_state
from helion._compiler.cute.tcgen05_config import seeds
from helion._compiler.cute.tcgen05_config import surface
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA

if TYPE_CHECKING:
    from collections.abc import Callable
from helion._testing import import_example
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl

# The B200 model with CUDA hidden, the inputs traced as ``cuda:0`` tensors
# of a 148-SM device, the reduction SMEM budget and an MMA support record
# carrying the e4m3 flag the fp8 binders consult (the family's own
# admission does not need it).
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.cpu_render(
        mma_support=SimpleNamespace(
            universal=True,
            warp_f16bf16=True,
            tcgen05_f16bf16=True,
            tcgen05_f8=True,
        ),
        forbid_native=False,
        fake_device_sm_count=148,
        reduction_smem_budget_bytes=232448,
    ),
]

FAMILY = family.WARP_MMA_FAMILY_KEY
WARPS = family.WARP_MMA_WARPS_KEY
RT = "_helion_warp_mma"


def _bind_gemm_k_major_b(
    shape: tuple[int, int, int], dtype: torch.dtype
) -> helion.runtime.kernel.BoundKernel:
    """A GEMM whose B is K-contiguous (the fp8 example's column-major y)."""
    m, n, k = shape
    args = (
        torch.empty((m, k), dtype=dtype, device=CPU_DEVICE),
        torch.empty((n, k), dtype=dtype, device=CPU_DEVICE).T,
    )
    return helion.kernel(_ordinary_gemm, backend="cute", static_shapes=True).bind(args)


def _bind_gemm_m_major_a(
    shape: tuple[int, int, int], dtype: torch.dtype, *, k_major_b: bool = False
) -> helion.runtime.kernel.BoundKernel:
    """A GEMM whose A is M-contiguous (``mat1.T @ grad_out`` of the matmul
    example's backward: the transposed view of a row-major matrix)."""
    m, n, k = shape
    a = torch.empty((k, m), dtype=dtype, device=CPU_DEVICE).T
    if k_major_b:
        b = torch.empty((n, k), dtype=dtype, device=CPU_DEVICE).T
    else:
        b = torch.empty((k, n), dtype=dtype, device=CPU_DEVICE)
    return helion.kernel(_ordinary_gemm, backend="cute", static_shapes=True).bind(
        (a, b)
    )


def _fp8_gemm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(torch.float16)
    return out


def _bind_fp8_gemm(
    shape: tuple[int, int, int], *, k_major_b: bool = True
) -> helion.runtime.kernel.BoundKernel:
    m, n, k = shape
    x = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=CPU_DEVICE)
    if k_major_b:
        y = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=CPU_DEVICE).T
    else:
        y = torch.empty((k, n), dtype=torch.float8_e4m3fn, device=CPU_DEVICE)
    return helion.kernel(_fp8_gemm, backend="cute", static_shapes=True).bind((x, y))


def _f32_out_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


def _relu_residual_gemm(
    a: torch.Tensor, b: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = torch.relu(acc + residual[tile_m, tile_n]).to(out.dtype)
    return out


def _rank0_scaled_gemm(
    a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = (acc * scale_a[()] * scale_b[()]).to(out.dtype)
    return out


def _extra_root_store_after(
    a: torch.Tensor, b: torch.Tensor, other: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    flag = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
        flag[tile_m, tile_n] = other[tile_m, tile_n] * 2
    return out, flag


def _extra_root_store_before(
    a: torch.Tensor, b: torch.Tensor, other: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    flag = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile((m, n)):
        flag[tile_m, tile_n] = other[tile_m, tile_n] * 2
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out, flag


def _inplace_residual(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = (acc + out[tile_m, tile_n]).to(out.dtype)
    return out


def _sliced_output(
    a: torch.Tensor, b: torch.Tensor, full: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = full[:, :n]
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return full


def _permuted_lhs_gemm(at: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    k, m = at.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=at.dtype, device=at.device)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, at[tile_k, tile_m].T, b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _bind(
    fn: object, *args: torch.Tensor, **settings: object
) -> helion.runtime.kernel.BoundKernel:
    return helion.kernel(fn, backend="cute", static_shapes=True, **settings).bind(args)  # pyrefly: ignore[bad-argument-type]


def _warp_config(
    block_sizes: list[int], warps: int, **overrides: object
) -> helion.Config:
    values: dict[str, object] = {
        "block_sizes": block_sizes,
        FAMILY: family.MATMUL_FAMILY_WARP_MMA,
        WARPS: warps,
    }
    values.update(overrides)
    return helion.Config(**values)


def _normalize(
    bound: helion.runtime.kernel.BoundKernel, config: helion.Config, *, fix: bool
) -> dict[str, object]:
    copied = helion.Config(
        **{k: list(v) if isinstance(v, list) else v for k, v in config.config.items()}
    )
    with bound.env:
        bound.config_spec.normalize(copied, _fix_invalid=fix)
    return copied.config


def _strict(
    bound: helion.runtime.kernel.BoundKernel, config: helion.Config
) -> dict[str, object]:
    return _normalize(bound, config, fix=False)


def _repaired(
    bound: helion.runtime.kernel.BoundKernel, config: helion.Config
) -> dict[str, object]:
    return _normalize(bound, config, fix=True)


def _plan(source: str) -> dict[str, object]:
    match = re.search(r"_helion_cute_wrapper_plans = (\[.*\])", source)
    assert match, source
    plans = eval(match.group(1))
    assert len(plans) == 1, plans
    return plans[0]


# --------------------------------------------------------------------------
# Renders
# --------------------------------------------------------------------------


def test_fp8_gemm_renders_one_warp_register_mma_tiles() -> None:
    # The Helion-Triton fp8 256^3 winner's structure: 512 one-warp 16x8 tiles,
    # K staged as two 128-wide cp.async groups, mma.sync.m16n8k32 e4m3.
    bound = _bind_fp8_gemm((256, 256, 256))
    source = _source(bound, _warp_config([16, 8, 128], 1))
    kernel = _kernel_body(source)
    assert f"{RT}.mma_e4m3(" in kernel
    assert kernel.count(f"{RT}.mma_e4m3(") == 8  # 2 chunks x 4 k32 steps x 1 atom
    assert kernel.count(f"{RT}.ldmatrix_x4(") == 8  # A fragments
    assert kernel.count(f"{RT}.ldmatrix_x2(") == 8  # K-major B, one n-tile
    assert "ldmatrix_x4_trans" not in kernel
    # A: 16 rows x 128 B per chunk = 128 packets over 32 threads; B: 8 x 128 B.
    assert kernel.count(f"{RT}.cp_async_16(") == 2 * (4 + 2)
    assert kernel.count("cute.arch.cp_async_commit_group()") == 2
    assert "cute.arch.cp_async_wait_group(1)" in kernel
    assert "cute.arch.cp_async_wait_group(0)" in kernel
    assert "cute.arch.sync_warp()" in kernel and "cute.arch.barrier()" not in kernel
    assert kernel.count(f"{RT}.pack_f16x2(") == 2
    assert kernel.count(f"{RT}.stg_b32(") == 2
    assert "tcgen05" not in kernel
    assert "block=(32, 1, 1)" in source
    plan = _plan(source)
    assert plan["kind"] == family.WARP_MMA_PLAN_KIND
    assert plan["total_tiles"] == 512 and plan["threads"] == 32
    assert "lhs_idx" in plan and "rhs_idx" in plan and "out_idx" in plan


def test_fp16_bias_gemm_renders_four_warp_tiles_with_the_bias_row() -> None:
    # fp16 256^3 + bias with the measured 64x8 four-warp tile: an N-major B
    # moves through ldmatrix.trans, the bias row is loaded before the first
    # cp.async wait and added on the fp32 fragments before the fp16 pack.
    bound = _bind_row_bias_gemm((256, 256, 256), torch.float16)
    source = _source(bound, _warp_config([64, 8, 128], 4))
    kernel = _kernel_body(source)
    assert kernel.count(f"{RT}.mma_f16(") == 2 * 8  # 2 chunks x 8 k16 steps
    assert kernel.count(f"{RT}.ldmatrix_x2_trans(") == 16
    assert "cute.arch.barrier()" in kernel and "sync_warp" not in kernel
    assert "block=(128, 1, 1)" in source
    bias_load = kernel.index(f"{RT}.unpack_f16x2({RT}.ldg_b32(")
    assert bias_load < kernel.index("cute.arch.cp_async_wait_group(")
    assert re.search(r"wm_acc_0_0_0 \+ wm_v0_0_0_0", kernel), kernel
    assert kernel.count(f"{RT}.pack_f16x2(") == 2
    plan = _plan(source)
    assert plan["total_tiles"] == 128 and plan["epi_aux_count"] == 1
    assert "epi_aux0_idx" in plan


def test_batched_gemm_tiles_every_batch_separately() -> None:
    bound = _bind_bmm((4, 64, 128, 128), torch.float16)
    source = _source(bound, _warp_config([1, 64, 8, 128], 4))
    kernel = _kernel_body(source)
    assert kernel.count(f"{RT}.mma_f16(") == 8
    assert "wm_pb = wm_pid // 16" in kernel
    assert _plan(source)["total_tiles"] == 64
    # The batch stride walks the operands and the output.
    assert "cutlass.Int64(wm_pb) * 16384" in kernel  # 64 x 128 fp16 A
    assert "cutlass.Int64(wm_pb) * 32768" in kernel  # 128 x 128 fp16 B


def test_two_warp_tiles_split_rows_first_and_share_b() -> None:
    bound = _bind_gemm((256, 256, 256), torch.bfloat16)
    source = _source(bound, _warp_config([32, 16, 256], 2))
    kernel = _kernel_body(source)
    assert "wm_wmi = wm_warp // 1" in kernel and "wm_wni = wm_warp % 1" in kernel
    assert kernel.count(f"{RT}.mma_bf16(") == 16 * 2  # 16 k16 steps x 2 n atoms
    assert kernel.count(f"{RT}.ldmatrix_x4_trans(") == 16  # both n-tiles per x4
    assert kernel.count("cute.arch.cp_async_commit_group()") == 1
    assert kernel.count(f"{RT}.pack_bf16x2(") == 4


def test_fp32_output_stores_pairs_without_a_pack() -> None:
    bound = _bind(
        _f32_out_gemm,
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
    )
    kernel = _kernel_body(_source(bound, _warp_config([16, 8, 128], 1)))
    assert kernel.count(f"{RT}.stg_v2_f32(") == 2
    assert "pack_" not in kernel


def test_residual_relu_epilogue_runs_on_the_fragments() -> None:
    bound = _bind(
        _relu_residual_gemm,
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
    )
    kernel = _kernel_body(_source(bound, _warp_config([32, 16, 128], 2)))
    # Exact-shape residual: one 4-byte pair per fragment row pair (each warp
    # owns one 16-row atom and two 8-column atoms), loaded at entry; relu
    # renders as the scalar conditional with NaN propagation.
    assert kernel.count(f"{RT}.unpack_f16x2({RT}.ldg_b32(") == 1 * 2 * 2
    assert kernel.count("if") >= 8 and "!=" in kernel
    assert "cute.where" not in kernel


def test_closure_bias_row_loads_element_by_element() -> None:
    # The matmul example's ``acc + bias[tile[1]]`` closure: the captured bias
    # row is not a bound input, so its base alignment is unprovable and the
    # row moves as single fp16 elements; the fresh output still stores pairs.
    bias = torch.empty((256,), dtype=torch.float16, device=CPU_DEVICE)
    bound = _bind(
        _gemm_with_epilogue,
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
        _closure_bias_epilogue(bias),  # pyrefly: ignore[bad-argument-type]
    )
    kernel = _kernel_body(_source(bound, _warp_config([64, 8, 128], 4)))
    assert kernel.count(f"{RT}.ldg_f16_as_f32(") == 2
    assert "ldg_b32" not in kernel
    assert kernel.count(f"{RT}.stg_b32(") == 2 and "stg_f16" not in kernel


def test_two_outputs_each_get_their_chain() -> None:
    bound = _bind_two_output_gemm((256, 256, 256), torch.float16)
    source = _source(bound, _warp_config([16, 8, 128], 1))
    kernel = _kernel_body(source)
    assert "wm_o0 = " in kernel and "wm_o1 = " in kernel
    assert kernel.count(f"{RT}.stg_b32(") == 4
    assert _plan(source)["out_idx"] is not None


def _bias_gelu_aux_gemm(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    aux = torch.empty_like(out)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        pre = acc * 1.25 + bias[tile_n]
        aux[tile_m, tile_n] = pre.to(a.dtype)
        out[tile_m, tile_n] = torch.nn.functional.gelu(pre).to(a.dtype)
    return out, aux


def test_gelu_aux_epilogue_is_part_of_the_gemm() -> None:
    # The chain's Python-level elementwise helpers (``_gelu_erf``) are pure:
    # the root-loop proof admits them like the aten pointwise ops, so the
    # cuBLASLt GELU_AUX pattern (bias prefix, two outputs) takes the
    # register-MMA family at small shapes.
    bound = _bind(
        _bias_gelu_aux_gemm,
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
        torch.empty((256,), dtype=torch.float16, device=CPU_DEVICE),
    )
    source = _source(bound, _warp_config([64, 16, 64], 1))
    kernel = _kernel_body(source)
    assert "wm_o0 = " in kernel and "wm_o1 = " in kernel
    assert "gelu_erf" in source
    assert _plan(source)["out_idx"] is not None


# --------------------------------------------------------------------------
# Admission, normalization and seeds
# --------------------------------------------------------------------------


def _fragment_low(bound: helion.runtime.kernel.BoundKernel, index: int) -> int:
    with bound.env:
        spec = bound.config_spec
        return spec.block_sizes[index]._fragment(spec).low


def test_batched_m_major_a_is_outside_the_search() -> None:
    # ``bmm(a.transpose(1, 2), b)``: the tcgen05 search records no static
    # extents for a batched M-contiguous A, so neither family is offered (the
    # admission's rank-3 M-major stride case is not reachable from a kernel);
    # the same bmm with a K-contiguous A admits the family.
    from test.test_cute_tcgen05_fixed_chain import _batched_gemm

    b = torch.empty((4, 128, 128), dtype=torch.float16, device=CPU_DEVICE)
    m_major = _bind(
        _batched_gemm,
        torch.empty((4, 128, 64), dtype=torch.float16, device=CPU_DEVICE).transpose(
            1, 2
        ),
        b,
    )
    with m_major.env:
        state = m_major.config_spec._cute_tcgen05_config
        assert not state.search_enabled and not state.warp_mma_admitted
        assert state.matmul_compile_time_static_extents is None
        with pytest.raises(InvalidConfig, match="tcgen05-enabled"):
            m_major.config_spec.normalized_config(_warp_config([1, 16, 8, 128], 1))
    k_major = _bind(
        _batched_gemm,
        torch.empty((4, 64, 128), dtype=torch.float16, device=CPU_DEVICE),
        b,
    )
    with k_major.env:
        state = k_major.config_spec._cute_tcgen05_config
        assert state.search_enabled and state.warp_mma_admitted


def test_admitted_problems_widen_the_block_floors_to_the_mma_atom() -> None:
    assert _fragment_low(_bind_gemm((256, 256, 256), torch.float16), 0) == 16
    assert _fragment_low(_bind_gemm((256, 256, 256), torch.float16), 1) == 8
    assert _fragment_low(_bind_fp8_gemm((256, 256, 256)), 1) == 8
    assert _fragment_low(_bind_bmm((4, 64, 128, 128), torch.float16), 1) == 16
    # Not admitted: a machine-filling grid, the 268M-MAC bmm, fp8 with an
    # N-major B and a dynamic-shape kernel keep the tcgen05 floors.
    assert _fragment_low(_bind_gemm((2048, 2048, 256), torch.float16), 0) == 128
    assert _fragment_low(_bind_bmm((8, 256, 512, 256), torch.float16), 1) == 64
    assert _fragment_low(_bind_fp8_gemm((256, 256, 256), k_major_b=False), 1) == 16


def test_campaign_1024_class_shards_are_not_admitted() -> None:
    # The three 1024-class campaign shards have small grids (64 / 32 tiles)
    # but 268M-1.07G multiply-adds, far beyond the latency-bound regime the
    # register-MMA tiles win (measured 3-5x slower than tcgen05 there): the
    # family is not admitted, its keys never enter their search surface and
    # their tcgen05 floors are unchanged.
    for bound in (
        _bind_fp8_gemm((1024, 1024, 1024)),
        _bind_row_bias_gemm((1024, 1024, 1024), torch.float16),
        _bind_bmm((8, 256, 512, 256), torch.float16),
    ):
        with bound.env:
            spec = bound.config_spec
            assert not spec._cute_tcgen05_config.warp_mma_admitted
            fields = spec.flat_config(lambda x: x.default()).config
            assert FAMILY not in fields and WARPS not in fields
            assert seeds._warp_mma_seed_configs(spec._cute_tcgen05_config) == []
    assert (
        _fragment_low(_bind_row_bias_gemm((1024, 1024, 1024), torch.float16), 0) == 64
    )
    # The three shape-0 shards are admitted.
    for bound in (
        _bind_fp8_gemm((256, 256, 256)),
        _bind_row_bias_gemm((256, 256, 256), torch.float16),
        _bind_bmm((4, 64, 128, 128), torch.float16),
    ):
        with bound.env:
            assert bound.config_spec._cute_tcgen05_config.warp_mma_admitted


def test_family_keys_appear_only_where_admitted() -> None:
    admitted = _bind_gemm((256, 256, 256), torch.float16)
    with admitted.env:
        fields = admitted.config_spec.flat_config(lambda x: x.default()).config
    assert fields[FAMILY] == family.MATMUL_FAMILY_TCGEN05
    assert fields[WARPS] == family.WARP_MMA_DEFAULT_WARPS
    full = _bind_gemm((2048, 2048, 256), torch.float16)
    with full.env:
        fields = full.config_spec.flat_config(lambda x: x.default()).config
    assert FAMILY not in fields and WARPS not in fields


def test_default_config_keeps_the_tcgen05_tile() -> None:
    # The widened floors do not move the default: the tcgen05 family clamps
    # the 16-row default back to its own floor.
    bound = _bind_gemm((256, 256, 256), torch.float16)
    with bound.env:
        config = bound.config_spec.default_config().config
    assert config[FAMILY] == family.MATMUL_FAMILY_TCGEN05
    assert config["block_sizes"][0] == 64 and config["block_sizes"][1] >= 8


def test_tcgen05_family_clamps_rows_below_its_floor() -> None:
    # Before the family, the block floors clamped an under-sized request
    # silently (a pinned ``block_sizes=[16, 16, 32]`` ran the 64-row tile);
    # the widened search floor keeps that contract for tcgen05 configs in
    # both normalization modes.
    bound = _bind_gemm((256, 256, 256), torch.float16)
    repaired = _repaired(bound, helion.Config(block_sizes=[16, 8, 128]))
    assert repaired["block_sizes"][:2] == [64, 8]
    strict = _strict(
        bound, helion.Config(block_sizes=[16, 8, 128], **{FAMILY: "tcgen05"})
    )
    assert strict["block_sizes"][:2] == [64, 8]
    fp8 = _bind_fp8_gemm((256, 256, 256))
    strict = _strict(fp8, helion.Config(block_sizes=[16, 8, 32]))
    assert strict["block_sizes"][:2] == [64, 16]


def test_warp_mma_requests_outside_the_admitted_region_fail_closed() -> None:
    full = _bind_gemm((2048, 2048, 256), torch.float16)
    with pytest.raises(InvalidConfig, match="not admitted"):
        _strict(full, _warp_config([64, 64, 128], 4))
    repaired = _repaired(full, _warp_config([64, 64, 128], 4))
    assert (
        repaired[FAMILY] == family.MATMUL_FAMILY_TCGEN05 if FAMILY in repaired else True
    )
    # Repair drops the inert keys from a non-admitted kernel's identity.
    assert repaired.get(FAMILY, "tcgen05") == "tcgen05"


def test_warp_mma_tiles_are_repaired_or_rejected() -> None:
    bound = _bind_gemm((256, 256, 256), torch.float16)
    # 128-row tiles exceed the family's 64-row tile; the repair halves them.
    repaired = _repaired(bound, _warp_config([128, 128, 64], 4))
    assert repaired["block_sizes"] == [64, 64, 64]
    assert repaired[WARPS] == 4
    with pytest.raises(InvalidConfig, match="block_m 128"):
        _strict(bound, _warp_config([128, 128, 64], 4))
    # Eight warps cannot tile 16x8: the repair takes the largest legal count.
    repaired = _repaired(bound, _warp_config([16, 8, 128], 8))
    assert repaired[WARPS] == 1
    with pytest.raises(InvalidConfig, match="warps cannot tile"):
        _strict(bound, _warp_config([16, 8, 128], 8))
    # The K chunk must divide K in whole mma steps.
    with pytest.raises(InvalidConfig, match="block_k"):
        _strict(bound, _warp_config([16, 8, 8], 1))


def test_warp_mma_pins_the_inert_tcgen05_knobs() -> None:
    bound = _bind_gemm((256, 256, 256), torch.float16)
    a = _repaired(bound, _warp_config([32, 16, 128], 2, tcgen05_ab_stages=5))
    b = _repaired(bound, _warp_config([32, 16, 128], 2, tcgen05_ab_stages=2))
    assert a["tcgen05_ab_stages"] == b["tcgen05_ab_stages"]
    assert a["tcgen05_c_store_mode"] == b["tcgen05_c_store_mode"]


def test_warp_mma_runs_one_cta_per_tile_on_the_flat_grid() -> None:
    # The persistent program-id strategies wrap the device body in a
    # ``virtual_pid`` loop (the first cold autotune lost 28 of 816 configs to
    # it: shared memory allocated inside the loop fails to compile); the
    # family pins the flat grid and every knob of the replaced Helion grid, so
    # equal kernels share one identity.
    bound = _bind_gemm((256, 256, 256), torch.float16)
    persistent = _warp_config(
        [16, 8, 128],
        1,
        pid_type="persistent_interleaved",
        l2_groupings=[8],
        loop_orders=[[1, 0]],
        indexing=["tensor_descriptor", "pointer", "tensor_descriptor"],
        cute_min_blocks_per_mp=1,
    )
    for config in (_strict(bound, persistent), _repaired(bound, persistent)):
        assert config["pid_type"] == "flat"
        assert config["l2_groupings"] == [1]
        assert config["loop_orders"] == [[0, 1]]
        assert config["indexing"] == ["pointer"] * 3
        assert config["cute_min_blocks_per_mp"] == 0
    source = _source(bound, persistent)
    assert "virtual_pid" not in source and "_NUM_SM" not in source
    assert _plan(source)["total_tiles"] == 512
    assert source == _source(bound, _warp_config([16, 8, 128], 1))


def test_batched_family_configs_pin_the_batch_block() -> None:
    bound = _bind_bmm((4, 64, 128, 128), torch.float16)
    repaired = _repaired(bound, _warp_config([2, 64, 8, 128], 4))
    assert repaired["block_sizes"][0] == 1
    with pytest.raises(InvalidConfig, match="batch block of 1"):
        _strict(bound, _warp_config([2, 64, 8, 128], 4))


def test_seeds_cover_the_measured_tiles() -> None:
    bound = _bind_row_bias_gemm((256, 256, 256), torch.float16)
    with bound.env:
        seed_configs = seeds._warp_mma_seed_configs(
            bound.config_spec._cute_tcgen05_config
        )
    tiles = sorted(
        (tuple(seed.config["block_sizes"]), seed.config[WARPS]) for seed in seed_configs
    )
    assert tiles == [
        ((16, 8, 128), 1),
        ((16, 16, 128), 1),
        ((16, 32, 128), 2),
        ((32, 16, 128), 2),
        ((32, 32, 128), 4),
        ((64, 8, 128), 4),
        ((64, 16, 128), 4),
    ], tiles
    assert all(seed.config[FAMILY] == "warp_mma" for seed in seed_configs)
    bmm = _bind_bmm((4, 64, 128, 128), torch.float16)
    with bmm.env:
        seed_configs = seeds._warp_mma_seed_configs(
            bmm.config_spec._cute_tcgen05_config
        )
    tiles = sorted(tuple(seed.config["block_sizes"]) for seed in seed_configs)
    assert tiles == [
        (1, 16, 8, 128),
        (1, 16, 16, 128),
        (1, 16, 32, 128),
        (1, 32, 16, 128),
        (1, 32, 32, 128),
        (1, 64, 8, 128),
        (1, 64, 16, 128),
    ], tiles
    full = _bind_gemm((2048, 2048, 256), torch.float16)
    with full.env:
        assert seeds._warp_mma_seed_configs(full.config_spec._cute_tcgen05_config) == []
    with bound.env:
        all_seeds = seeds.autotune_seed_configs(bound.config_spec._cute_tcgen05_config)
    assert sum(seed.config.get(FAMILY) == "warp_mma" for seed in all_seeds) == 7


def test_machine_filling_renders_carry_no_family_key() -> None:
    bound = _bind_gemm((2048, 2048, 256), torch.float16)
    with bound.env:
        config = bound.config_spec.default_config()
    assert FAMILY not in config.config
    assert "warp_mma" not in bound.to_code(config)


@pytest.mark.parametrize(
    "kernel_fn", [_extra_root_store_after, _extra_root_store_before]
)
def test_root_statements_outside_the_gemm_fail_closed(kernel_fn: object) -> None:
    # The family replaces the whole device body.  A second store in the root
    # tile loop (before or after the K loop) is not part of the matched GEMM:
    # it must refuse the family instead of silently dropping that store.  The
    # default (tcgen05) config of the same kernel runs the rest of the tile
    # body once per thread beside its matmul, where the per-element store has
    # no lane coordinates either: it refuses too rather than emitting a kernel
    # that reads undefined lane indices.
    bound = _bind(
        kernel_fn,
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
    )
    with pytest.raises(
        BackendUnsupported, match="root tile loop may only hold the GEMM"
    ):
        _source(bound, _warp_config([16, 8, 128], 1))
    with bound.env:
        default = bound.config_spec.default_config()
    with pytest.raises(BackendUnsupported, match="beside a collective matmul"):
        bound.to_code(default)


def test_admission_mirrors_the_detector_gates() -> None:
    # Kernels the detector always declines never receive the family keys or
    # its seeds (each would be a dead compile in a cold autotune).
    permuted = _bind(
        _permuted_lhs_gemm,
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
    )
    # 1032-byte rows on an operand above the staging size floor: the staging
    # pass relays it through an aligned copy, so the family is admitted; with
    # the pass disabled the detector's alignment gate declines as before.
    misaligned_args = (
        torch.empty((512, 516), dtype=torch.float16, device=CPU_DEVICE)[:, :512],
        torch.empty((512, 64), dtype=torch.float16, device=CPU_DEVICE),
    )
    staged_rows = _bind(_ordinary_gemm, *misaligned_args)
    with staged_rows.env:
        assert staged_rows.config_spec._cute_tcgen05_config.warp_mma_admitted
    misaligned_rows = _bind(
        _ordinary_gemm, *misaligned_args, cute_stage_unaligned_operands=False
    )
    for bound in (permuted, misaligned_rows):
        with bound.env:
            spec = bound.config_spec
            assert not spec._cute_tcgen05_config.warp_mma_admitted
            assert FAMILY not in spec.flat_config(lambda x: x.default()).config
            assert seeds._warp_mma_seed_configs(spec._cute_tcgen05_config) == []
        with pytest.raises(InvalidConfig, match="not admitted"):
            _strict(bound, _warp_config([16, 8, 128], 1))
    assert (
        family.warp_mma_admission_reason(
            static_m=256,
            static_n=256,
            static_k=256,
            leading=1,
            dtype=torch.float16,
            lhs_major="row",
            rhs_major="row",
            num_sms=148,
            lhs_strides=(260, 1),
            rhs_strides=(256, 1),
        )
        == "A rows must be K-contiguous and 16-byte aligned"
    )
    assert (
        family.warp_mma_admission_reason(
            static_m=256,
            static_n=256,
            static_k=256,
            leading=1,
            dtype=torch.float16,
            lhs_major="row",
            rhs_major="row",
            num_sms=148,
            lhs_strides=(256, 1),
            rhs_strides=(256, 1),
            operands_permuted=True,
        )
        == "permuted operands are not supported"
    )


def test_m_major_a_is_staged_transposed_and_read_with_ldmatrix_trans() -> None:
    # The ``mat1.T @ grad_out`` GEMM of the matmul backward: A is M-contiguous.
    # Its tile is staged as bk rows of bm elements (packets coalesced along
    # M, one k column per packet row) and the fragments come from
    # ``ldmatrix.trans`` like those of an N-contiguous B; the family is
    # admitted and seeded for it.
    bound = _bind_gemm_m_major_a((256, 256, 256), torch.float16)
    with bound.env:
        spec = bound.config_spec
        assert spec._cute_tcgen05_config.warp_mma_admitted
        assert seeds._warp_mma_seed_configs(spec._cute_tcgen05_config)
    body = _kernel_body(_source(bound, _warp_config([16, 32, 64], 2)))
    # The tile base steps along M by one element; packet rows step along K
    # by the column stride (256 elements = 512 B).
    assert "cutlass.Int64(wm_m0) * 2" in body
    assert (
        "wm_a_g + cutlass.Int64(0 + wm_prow) * 512 + cutlass.Int64(wm_pch * 16)" in body
    )
    assert (
        "wm_a_g + cutlass.Int64(64 + wm_prow) * 512 + cutlass.Int64(wm_pch * 16)"
        in body
    )
    # Both operands are N-/M-contiguous: every fragment comes through .trans.
    assert f"{RT}.ldmatrix_x4(" not in body
    # 4 chunks x 4 steps x (1 A atom + 1 B pair) per warp.
    assert body.count(f"{RT}.ldmatrix_x4_trans(") == 32
    assert "wm_arow = wm_l8 + 8 * (wm_lq // 2)" in body
    assert "wm_achunk = wm_wmi * 2 + wm_lq % 2" in body
    # Same shared footprint as the K-contiguous staging of the same tile.
    row_major = _kernel_body(
        _source(
            _bind_gemm((256, 256, 256), torch.float16), _warp_config([16, 32, 64], 2)
        )
    )
    smem = re.compile(r"alloc_smem\(cutlass.Uint8, (\d+)")
    assert smem.search(body).group(1) == smem.search(row_major).group(1) == "24576"
    # With a K-contiguous B only A goes through .trans.
    both = _kernel_body(
        _source(
            _bind_gemm_m_major_a((256, 256, 256), torch.bfloat16, k_major_b=True),
            _warp_config([32, 32, 64], 4),
        )
    )
    assert both.count(f"{RT}.ldmatrix_x4_trans(") == 16  # A: 4 chunks x 4 steps
    assert both.count(f"{RT}.ldmatrix_x4(") == 16  # B pairs
    assert f"{RT}.mma_bf16(" in both


def test_m_major_a_admission_keeps_the_e4m3_and_alignment_gates() -> None:
    def reason(**overrides: object) -> str | None:
        values: dict[str, object] = {
            "static_m": 256,
            "static_n": 256,
            "static_k": 256,
            "leading": 1,
            "dtype": torch.float16,
            "lhs_major": "col",
            "rhs_major": "row",
            "num_sms": 148,
            "lhs_strides": (1, 256),
            "rhs_strides": (256, 1),
        }
        values.update(overrides)
        return family.warp_mma_admission_reason(**values)  # pyrefly: ignore[bad-argument-type]

    assert reason() is None
    assert reason(rhs_major="col", rhs_strides=(1, 256)) is None
    assert (
        reason(dtype=torch.float8_e4m3fn, rhs_major="col", rhs_strides=(1, 256))
        == "an e4m3 A must be K-contiguous (ldmatrix.trans moves 16-bit elements)"
    )
    assert (
        reason(lhs_strides=(1, 260))
        == "an M-contiguous A needs 16-byte aligned columns"
    )
    assert reason(lhs_major=None) == "A must be K- or M-contiguous"
    # The strided-view binding of a column-major A with a misaligned column
    # stride is not admitted (no family keys, no seeds) once the staging
    # pass that would relay it through an aligned copy is disabled.
    misaligned = _bind(
        _ordinary_gemm,
        torch.empty((256, 260), dtype=torch.float16, device=CPU_DEVICE)[:, :256].T,
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
        cute_stage_unaligned_operands=False,
    )
    with misaligned.env:
        spec = misaligned.config_spec
        assert not spec._cute_tcgen05_config.warp_mma_admitted
        assert seeds._warp_mma_seed_configs(spec._cute_tcgen05_config) == []


def test_cast_operand_kernel_stays_fused_and_converts_in_registers() -> None:
    # bf16 x int16 at (128, 256, 128): the matmul reads ``w[k, n].to(bf16)``
    # of an int16 tile.  TMA packets cannot carry that cast to the tensor
    # core, so the tcgen05 family is withheld and the materialization pass
    # keeps the single launch: the register-MMA family is admitted, offered
    # alone, and converts the int16 packets in registers on the way to
    # shared memory (eight words per 16-byte packet, every packet of a
    # thread in flight before the first convert).
    example = import_example("bf16xint16_gemm")

    bound = _bind(
        example._bf16xint16_gemm.fn,
        torch.empty((128, 256), dtype=torch.bfloat16, device=CPU_DEVICE),
        torch.empty((256, 128), dtype=torch.int16, device=CPU_DEVICE),
    )
    with bound.env:
        spec = bound.config_spec
        state = spec._cute_tcgen05_config
        assert cute_env_state(bound.env).fission_plan is None
        assert len(bound.host_function.device_ir.graphs) == 2
        assert state.warp_mma_admitted and not state.tcgen05_family_admitted
        search = surface.optional_fragments(state, for_search=True)[FAMILY].choices
        assert tuple(search) == (family.MATMUL_FAMILY_WARP_MMA,)
        assert spec.default_config().config[FAMILY] == family.MATMUL_FAMILY_WARP_MMA
        seed_configs = seeds._warp_mma_seed_configs(state)
        assert seed_configs
    assert all(
        seed.config[FAMILY] == family.MATMUL_FAMILY_WARP_MMA for seed in seed_configs
    )
    source = _source(bound, seed_configs[0])
    body = _kernel_body(source)
    assert source.count("@cute.kernel") == 1
    assert f"{RT}.mma_bf16(" in body
    # A (bf16) still moves by cp.async; B (int16) is loaded as words and
    # converted into its swizzled slot after every packet is issued.
    assert f"{RT}.cp_async_16(wm_s_a" in body
    assert f"{RT}.cp_async_16(wm_s_b" not in body
    assert f"{RT}.ldg_v4_b32(wm_b_g" in body
    assert f"{RT}.cvt_s16x8_bf16_sts(wm_b" in body
    assert body.index(f"{RT}.ldg_v4_b32(") < body.index(f"{RT}.cvt_s16x8_bf16_sts(")
    assert body.rindex(f"{RT}.cvt_s16x8_bf16_sts(") < body.index("cp_async_wait_group")
    # An explicit tcgen05-family request fails closed with the reason.
    with bound.env, pytest.raises(InvalidConfig, match="cast matmul operand"):
        bound.config_spec.normalized_config(
            helion.Config(block_sizes=[64, 32, 128], **{FAMILY: "tcgen05"})
        )


def test_cast_operand_on_the_lhs_converts_a_packets() -> None:
    example = import_example("bf16xint16_gemm")

    bound = _bind(
        example._int16xbf16_gemm.fn,
        torch.empty((128, 256), dtype=torch.int16, device=CPU_DEVICE),
        torch.empty((256, 128), dtype=torch.bfloat16, device=CPU_DEVICE),
    )
    with bound.env:
        state = bound.config_spec._cute_tcgen05_config
        assert state.warp_mma_admitted and not state.tcgen05_family_admitted
        seed_configs = seeds._warp_mma_seed_configs(state)
    body = _kernel_body(_source(bound, seed_configs[0]))
    assert f"{RT}.ldg_v4_b32(wm_a_g" in body
    assert f"{RT}.cvt_s16x8_bf16_sts(wm_a" in body
    assert f"{RT}.cp_async_16(wm_s_a" not in body
    assert f"{RT}.cp_async_16(wm_s_b" in body


def test_cute_pdl_waits_before_the_packets_and_releases_before_the_stores() -> None:
    bound = _bind_row_bias_gemm((256, 256, 256), torch.float16)
    # (``cute_pdl`` defaults to True on the sm_90+ tcgen05 surface, which
    # the family shares; the knob-off render pins it off)
    plain = _source(bound, _warp_config([64, 8, 256], 4, cute_pdl=False))
    assert "griddepcontrol" not in plain and "'use_pdl': True" not in plain
    source = _source(bound, _warp_config([64, 8, 256], 4, cute_pdl=True))
    body = _kernel_body(source)
    assert "'use_pdl': True" in source
    assert body.count("cute.arch.griddepcontrol_wait()") == 1
    assert body.count("cute.arch.griddepcontrol_launch_dependents()") == 1
    wait = body.index("cute.arch.griddepcontrol_wait()")
    release = body.index("cute.arch.griddepcontrol_launch_dependents()")
    # The wait precedes every global access (the bias row, the operand
    # packets); the release follows the last MMA and precedes the stores.
    assert body.index("alloc_smem") < wait < body.index(f"{RT}.ldg_b32(")
    assert wait < body.index(f"{RT}.cp_async_16(")
    assert body.rindex(f"{RT}.mma_f16(") < release < body.index(f"{RT}.stg_b32(")
    # The knob survives the family's inert-knob pin.
    with bound.env:
        normalized = bound.config_spec.normalized_config(
            _warp_config([64, 8, 256], 4, cute_pdl=True)
        )
    assert normalized.config["cute_pdl"] is True
    stripped = [line for line in body.splitlines() if "griddepcontrol" not in line]
    assert "\n".join(stripped) == _kernel_body(plain)


def test_fissioned_kernel_region_renders_the_family() -> None:
    # A dynamic-shape bf16 x int16 kernel keeps the materialization fission
    # (the proof specializes the shapes the fused program would lack): an
    # operand-materialization region and a plain bf16 GEMM region (three
    # graphs).  The complete-root rule inspects the root that calls the K
    # loop; the other region keeps its own lowering, so the family is
    # admitted, seeded and renders the GEMM region.
    example = import_example("bf16xint16_gemm")

    bound = helion.kernel(
        example._bf16xint16_gemm.fn,  # pyrefly: ignore[bad-argument-type]
        backend="cute",
        static_shapes=False,
    ).bind(
        (
            torch.empty((128, 256), dtype=torch.bfloat16, device=CPU_DEVICE),
            torch.empty((256, 128), dtype=torch.int16, device=CPU_DEVICE),
        )
    )
    with bound.env:
        spec = bound.config_spec
        assert cute_env_state(bound.env).fission_plan is not None
        assert len(bound.host_function.device_ir.graphs) == 3
        assert spec._cute_tcgen05_config.warp_mma_admitted
        assert spec._cute_tcgen05_config.tcgen05_family_admitted
        seed_configs = seeds._warp_mma_seed_configs(spec._cute_tcgen05_config)
    assert seed_configs
    source = _source(bound, seed_configs[0])
    assert f"{RT}.mma_bf16(" in source
    assert f"{RT}.cvt_s16x8_bf16_sts(" not in source
    assert source.count("@cute.kernel") == 2  # the materialization region stays


def test_tile_reason_requires_powers_of_two() -> None:
    common = {"m": 256, "n": 256, "k": 256, "warps": 1, "dtype": torch.float16}
    assert family.warp_mma_tile_reason(bm=48, bn=8, bk=128, **common) == (
        "block_m 48 must be a power of two"
    )
    assert family.warp_mma_tile_reason(bm=16, bn=24, bk=128, **common) == (
        "block_n 24 must be a power of two"
    )
    assert family.warp_mma_tile_reason(bm=16, bn=8, bk=144, **common) == (
        "block_k 144 must be a power of two"
    )
    assert family.warp_mma_tile_reason(bm=16, bn=8, bk=128, **common) is None


def test_64_row_persistent_request_renders_the_flat_grid() -> None:
    # A 64-row tile is tcgen05-legal, so a persistent request used to
    # instantiate the tcgen05 persistent scheduler without a tcgen05 plan.
    bound = _bind_gemm((256, 256, 256), torch.float16)
    for pid_type in ("persistent_blocked", "persistent_interleaved", "xyz"):
        source = _source(bound, _warp_config([64, 8, 128], 4, pid_type=pid_type))
        assert "PersistentTileScheduler" not in source
        assert "virtual_pid" not in source and "_NUM_SM" not in source
        assert source == _source(bound, _warp_config([64, 8, 128], 4))


def test_launcher_never_stages_the_family_operands() -> None:
    # The body bakes every stride: a staged (contiguous) copy of a strided
    # output view would be addressed with the view's strides.  Under-aligned
    # A / B bases are refused at compile time; outputs and aux rows fall back
    # to element accesses instead.
    from helion.runtime.cute.launch_descriptor import STAGED_PLAN_OPERANDS

    assert family.WARP_MMA_PLAN_KIND not in STAGED_PLAN_OPERANDS
    bound = _bind(
        _inplace_residual,
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
        torch.empty((256, 258), dtype=torch.float16, device=CPU_DEVICE)[:, 1:257],
    )
    kernel = _kernel_body(_source(bound, _warp_config([16, 8, 128], 1)))
    # Row stride 516 B is baked; the 2-byte-offset view moves element-wise.
    assert "* 516" in kernel
    assert kernel.count(f"{RT}.stg_f16(") == 4 and "stg_b32" not in kernel
    assert kernel.count(f"{RT}.ldg_f16_as_f32(") == 4 and "ldg_b32" not in kernel


def test_unsupported_epilogue_fails_closed_at_codegen() -> None:
    # The register-MMA body replaces the whole kernel: an epilogue the chain
    # analysis does not admit (a reduction over the tile) is refused with
    # the reason instead of running another lowering.
    def rowsum_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m, k = a.shape
        _, n = b.shape
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
        for tile_m, tile_n in hl.tile((m, n)):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
            out[tile_m, tile_n] = (acc - acc.amax(dim=1, keepdim=True)).to(out.dtype)
        return out

    bound = _bind(
        rowsum_gemm,
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
    )
    with pytest.raises(BackendUnsupported, match="warp_mma GEMM family"):
        _source(bound, _warp_config([16, 8, 128], 1))


# Runtime numerics for the ``warp_mma`` matmul family (sm_100 GPU).
#
# Every admitted operand dtype (fp16, bf16, e4m3), both B layouts (N- and
# K-contiguous), batched problems, bias rows, residual tiles, fp32 outputs and
# every seeded tile / warp count are checked against fp32 torch matmuls of the
# upcast operands; repeated launches are bitwise identical (fixed K order).


def _sm100_gpu(test: Callable[..., object]) -> Callable[..., object]:
    """Skip ``test`` without a CUDA GPU of capability 10.0 or newer."""
    return skipIfNotCUDA()(
        skipIfCudaCapabilityLessThan(
            (10, 0),
            reason="the register-MMA family targets the sm_100 GPUs of the tcgen05 search",
        )(test)
    )


LAUNCHES = 8


def _config(block_sizes: list[int], warps: int) -> helion.Config:
    return helion.Config(
        block_sizes=block_sizes,
        **{FAMILY: family.MATMUL_FAMILY_WARP_MMA, WARPS: warps},
    )


def _run(kernel_fn, args: tuple[torch.Tensor, ...], config: helion.Config):
    kernel = helion.kernel(kernel_fn, backend="cute", static_shapes=True)
    bound = kernel.bind(args)
    bound.set_config(config)
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    source = bound.to_code(normalized)
    assert "_helion_warp_mma.mma_" in source, source
    assert "tcgen05_tmem_allocator" not in source
    outputs = [bound(*args) for _ in range(LAUNCHES)]
    torch.cuda.synchronize()
    first = outputs[0]
    for other in outputs[1:]:
        if isinstance(first, tuple):
            for lhs, rhs in zip(first, other, strict=True):
                assert torch.equal(lhs, rhs)
        else:
            assert torch.equal(first, other)
    return first, source


def _randn(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(shape, device=DEVICE, dtype=torch.float32).to(dtype)


def _k_major(b: torch.Tensor) -> torch.Tensor:
    return b.transpose(-1, -2).contiguous().transpose(-1, -2)


def _check(out: torch.Tensor, ref: torch.Tensor) -> None:
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


TILES = [
    ([16, 8, 128], 1),
    ([16, 16, 128], 1),
    ([32, 16, 128], 2),
    ([32, 32, 128], 4),
    ([64, 8, 128], 4),
    ([64, 32, 256], 4),
    ([32, 8, 32], 1),
    ([64, 64, 64], 8),
]


@_sm100_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k_major", [False, True])
@pytest.mark.parametrize(("block_sizes", "warps"), TILES)
def test_dense_gemm_matches_torch(
    dtype: torch.dtype, k_major: bool, block_sizes: list[int], warps: int
) -> None:
    torch.manual_seed(0)
    a = _randn(256, 256, dtype=dtype)
    b = _randn(256, 256, dtype=dtype)
    if k_major:
        b = _k_major(b)
    out, _ = _run(_ordinary_gemm, (a, b), _config(block_sizes, warps))
    _check(out, torch.matmul(a.float(), b.float()))


@_sm100_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k_major", [False, True])
@pytest.mark.parametrize(("block_sizes", "warps"), TILES)
def test_m_major_a_gemm_matches_torch(
    dtype: torch.dtype, k_major: bool, block_sizes: list[int], warps: int
) -> None:
    # ``mat1.T @ grad_out``: A is the transposed view of a row-major matrix.
    torch.manual_seed(0)
    a = _randn(256, 256, dtype=dtype).T
    b = _randn(256, 256, dtype=dtype)
    if k_major:
        b = _k_major(b)
    out, source = _run(_ordinary_gemm, (a, b), _config(block_sizes, warps))
    assert "_helion_warp_mma.ldmatrix_x4_trans(" in source
    _check(out, torch.matmul(a.float(), b.float()))


@_sm100_gpu
@pytest.mark.cuda
def test_matmul_backward_pair_runs_on_the_family() -> None:
    """Both GEMMs of the matmul example's backward (``grad_out @ mat2.T`` and
    ``mat1.T @ grad_out``) run on the family's measured 16x32 two-warp tile."""
    matmul_mod = import_example("matmul")

    torch.manual_seed(0)
    mat1 = _randn(256, 256, dtype=torch.float16)
    mat2 = _randn(256, 256, dtype=torch.float16)
    grad_out = _randn(256, 256, dtype=torch.float16)
    matmul = helion.kernel(matmul_mod.matmul.fn, backend="cute", static_shapes=True)
    for lhs, rhs in ((grad_out, mat2.T), (mat1.T, grad_out)):
        bound = matmul.bind((lhs, rhs))
        bound.set_config(_config([16, 32, 64], 2))
        with bound.env:
            normalized = bound.config_spec.normalized_config(_config([16, 32, 64], 2))
        assert "_helion_warp_mma.mma_f16(" in bound.to_code(normalized)
        _check(bound(lhs, rhs), torch.matmul(lhs.float(), rhs.float()))


@_sm100_gpu
@pytest.mark.cuda
@pytest.mark.parametrize(("block_sizes", "warps"), TILES)
def test_fp8_gemm_matches_scaled_mm(block_sizes: list[int], warps: int) -> None:
    torch.manual_seed(0)
    x = _randn(256, 256, dtype=torch.float8_e4m3fn)
    y = _k_major(_randn(256, 256, dtype=torch.float8_e4m3fn))
    out, source = _run(_fp8_gemm, (x, y), _config(block_sizes, warps))
    assert "_helion_warp_mma.mma_e4m3(" in source
    scale = torch.tensor(1.0, device=DEVICE)
    ref = torch._scaled_mm(
        x, y, scale, scale, use_fast_accum=False, out_dtype=torch.float16
    )
    _check(out, ref.float())
    _check(out, torch.matmul(x.float(), y.float()))


@_sm100_gpu
@pytest.mark.cuda
def test_fp8_gemm_rejects_an_n_major_b() -> None:
    x = _randn(256, 256, dtype=torch.float8_e4m3fn)
    y = _randn(256, 256, dtype=torch.float8_e4m3fn)
    kernel = helion.kernel(_fp8_gemm, backend="cute", static_shapes=True)
    bound = kernel.bind((x, y))
    with pytest.raises(
        helion.exc.InvalidConfig, match="not admitted|cute_matmul_family"
    ):
        bound.set_config(_config([16, 8, 128], 1))
        with bound.env:
            bound.config_spec.normalized_config(_config([16, 8, 128], 1))


@_sm100_gpu
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("block_sizes", "warps"), [([64, 8, 128], 4), ([16, 8, 128], 1), ([32, 32, 128], 4)]
)
def test_bias_row_epilogue(block_sizes: list[int], warps: int) -> None:
    torch.manual_seed(0)
    a = _randn(256, 256, dtype=torch.float16)
    b = _randn(256, 256, dtype=torch.float16)
    bias = _randn(256, dtype=torch.float16)
    out, source = _run(_row_bias_gemm, (a, b, bias), _config(block_sizes, warps))
    assert "_helion_warp_mma.unpack_f16x2(_helion_warp_mma.ldg_b32(" in source
    _check(out, torch.matmul(a.float(), b.float()) + bias.float())


@_sm100_gpu
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("block_sizes", "warps"),
    [
        ([1, 64, 8, 128], 4),
        ([1, 16, 8, 128], 1),
        ([1, 32, 32, 128], 4),
        ([1, 32, 16, 64], 2),
    ],
)
def test_batched_gemm(block_sizes: list[int], warps: int) -> None:
    torch.manual_seed(0)
    a = _randn(4, 64, 128, dtype=torch.float16)
    b = _randn(4, 128, 128, dtype=torch.float16)
    out, _ = _run(_batched_gemm, (a, b), _config(block_sizes, warps))
    _check(out, torch.bmm(a.float(), b.float()))


@_sm100_gpu
@pytest.mark.cuda
def test_fp32_output() -> None:
    torch.manual_seed(0)
    a = _randn(128, 64, dtype=torch.bfloat16)
    b = _randn(64, 192, dtype=torch.bfloat16)
    out, source = _run(_f32_out_gemm, (a, b), _config([32, 8, 64], 2))
    assert "_helion_warp_mma.stg_v2_f32(" in source
    assert out.dtype == torch.float32
    _check(out, torch.matmul(a.float(), b.float()))


@_sm100_gpu
@pytest.mark.cuda
def test_relu_residual_epilogue() -> None:
    torch.manual_seed(0)
    a = _randn(192, 256, dtype=torch.float16)
    b = _randn(256, 128, dtype=torch.float16)
    residual = _randn(192, 128, dtype=torch.float16)
    out, _ = _run(_relu_residual_gemm, (a, b, residual), _config([16, 16, 128], 1))
    ref = torch.relu(torch.matmul(a.float(), b.float()) + residual.float())
    _check(out, ref)


@_sm100_gpu
@pytest.mark.cuda
def test_rank0_scale_epilogue() -> None:
    # The fp8_gemm dequantization scales: rank-0 loads are tile-uniform
    # scalar leaves rendered inline on the fragments, and the launcher
    # marshals the 0-d tensors as one-element views.
    torch.manual_seed(0)
    a = _randn(128, 64, dtype=torch.bfloat16)
    b = _randn(64, 128, dtype=torch.bfloat16)
    scale_a = torch.tensor(0.5, device=DEVICE)
    scale_b = torch.tensor(3.0, device=DEVICE)
    out, source = _run(
        _rank0_scaled_gemm, (a, b, scale_a, scale_b), _config([32, 32, 64], 4)
    )
    assert "* cutlass.Float32(scale_a.iterator.load())" in source
    assert "* cutlass.Float32(scale_b.iterator.load())" in source
    _check(out, torch.matmul(a.float(), b.float()) * 1.5)


@_sm100_gpu
@pytest.mark.cuda
def test_two_outputs() -> None:
    torch.manual_seed(0)
    a = _randn(256, 256, dtype=torch.float16)
    b = _randn(256, 256, dtype=torch.float16)
    (out, out2), _ = _run(_two_output_gemm, (a, b), _config([32, 16, 128], 2))
    ref = torch.matmul(a.float(), b.float())
    _check(out, ref)
    _check(out2, torch.relu(ref))


@_sm100_gpu
@pytest.mark.cuda
def test_under_aligned_operand_fails_closed() -> None:
    # The cp.async packets need 16-byte-aligned operand bases; a view eight
    # bytes into a buffer is refused with the reason rather than lowered.
    torch.manual_seed(0)
    full = _randn(256, 264, dtype=torch.float16)
    a = full[:, 4:260]
    b = _randn(256, 256, dtype=torch.float16)
    kernel = helion.kernel(_ordinary_gemm, backend="cute", static_shapes=True)
    bound = kernel.bind((a, b))
    with pytest.raises(BackendUnsupported, match="16-byte aligned"):
        bound.set_config(_config([16, 8, 128], 1))
        bound(a, b)


@_sm100_gpu
@pytest.mark.cuda
def test_campaign_examples_run_on_the_family() -> None:
    """The three campaign variants on their measured tiles."""
    bmm_mod = import_example("bmm")
    fp8_mod = import_example("fp8_gemm")
    matmul_mod = import_example("matmul")

    torch.manual_seed(0)
    x = _randn(256, 256, dtype=torch.float8_e4m3fn)
    y = _k_major(_randn(256, 256, dtype=torch.float8_e4m3fn))
    fp8 = helion.kernel(fp8_mod.fp8_gemm.fn, backend="cute", static_shapes=True)
    bound = fp8.bind((x, y))
    bound.set_config(_config([16, 8, 128], 1))
    out = bound(x, y)
    _check(out, torch.matmul(x.float(), y.float()))

    a = _randn(256, 256, dtype=torch.float16)
    b = _randn(256, 256, dtype=torch.float16)
    bias = _randn(256, dtype=torch.float16)
    epilogue = lambda acc, tile: acc + bias[tile[1]]  # noqa: E731
    matmul = helion.kernel(matmul_mod.matmul.fn, backend="cute", static_shapes=True)
    bound = matmul.bind((a, b, epilogue))
    bound.set_config(_config([64, 8, 128], 4))
    out = bound(a, b, epilogue)
    _check(out, torch.addmm(bias.float(), a.float(), b.float()))

    a3 = _randn(4, 64, 128, dtype=torch.float16)
    b3 = _randn(4, 128, 128, dtype=torch.float16)
    bmm = helion.kernel(bmm_mod.bmm.fn, backend="cute", static_shapes=True)
    bound = bmm.bind((a3, b3))
    bound.set_config(_config([1, 64, 8, 128], 4))
    out = bound(a3, b3)
    _check(out, torch.bmm(a3.float(), b3.float()))


def _grouped_reference(
    a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """The jagged kernel's contract: rows of a group hold ``A[rows] @ B``;
    rows outside every group are left as allocated (NaN here)."""
    out = torch.full(
        (a.shape[0], b.shape[1]), float("nan"), dtype=a.dtype, device=a.device
    )
    bounds = offsets.tolist()
    for g in range(len(bounds) - 1):
        start, end = bounds[g], bounds[g + 1]
        if end - start > 0:
            out[start:end] = (a[start:end].float() @ b.float()).to(a.dtype)
    return out


@_sm100_gpu
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("block_sizes", "warps"),
    [([32, 32, 64], 4), ([64, 16, 64], 4), ([16, 32, 64], 2), ([16, 8, 128], 1)],
)
def test_grouped_row_union_matches_the_jagged_kernel(
    block_sizes: list[int], warps: int
) -> None:
    """``examples/grouped_gemm.py``'s jagged kernel on the family: the
    catalogue's shape-0 layout, then (same compiled kernel, ``static_shapes=
    False``) a layout with an empty group and one that leaves the tail rows
    outside every group."""
    grouped_mod = import_example("grouped_gemm")

    torch.manual_seed(0)
    a = _randn(640, 128, dtype=torch.bfloat16)
    b = _randn(128, 128, dtype=torch.bfloat16)
    offsets = torch.tensor([0, 64, 192, 384, 640], device=DEVICE, dtype=torch.int32)
    kernel = helion.kernel(
        grouped_mod.grouped_gemm_jagged.fn, backend="cute", static_shapes=False
    )
    bound = kernel.bind((a, b, offsets))
    config = _config(block_sizes, warps)
    bound.set_config(config)
    with bound.env:
        assert bound.config_spec._cute_tcgen05_config.warp_mma_admitted
        normalized = bound.config_spec.normalized_config(config)
    source = bound.to_code(normalized)
    assert "_helion_warp_mma.mma_bf16(" in source
    assert "group_offsets.iterator + cutlass.Int64(" in source
    assert "tcgen05_tmem_allocator" not in source
    for layout in (
        [0, 64, 192, 384, 640],
        [0, 100, 100, 300, 640],
        [0, 64, 64, 200, 500],
    ):
        offs = torch.tensor(layout, device=DEVICE, dtype=torch.int32)
        ref = _grouped_reference(a, b, offs)
        covered = ~torch.isnan(ref)
        outs = [bound(a, b, offs) for _ in range(3)]
        torch.cuda.synchronize()
        for other in outs[1:]:
            assert torch.equal(outs[0][covered], other[covered])
        _check(outs[0][covered], ref[covered].float())


@_sm100_gpu
@pytest.mark.cuda
def test_grouped_row_union_rebinds_per_shape_and_fails_closed_off_the_family() -> None:
    """``static_shapes=False``: the family is admitted from each binding's
    hints and codegen specializes ``(total_M, G)`` under shape guards, so a
    call with another packed size or group count rebinds (fresh heuristic,
    fresh body) instead of reusing the first binding's clamps.  A shape the
    family does not run (1000 rows) rejects the pinned family strictly and
    repairs to the tcgen05 lowering, which stays correct."""
    grouped_mod = import_example("grouped_gemm")

    torch.manual_seed(0)
    b = _randn(128, 128, dtype=torch.bfloat16)
    kernel = helion.kernel(
        grouped_mod.grouped_gemm_jagged.fn, backend="cute", static_shapes=False
    )
    config = _config([32, 32, 64], 4)
    seen: dict[int, int] = {}
    for total_m, layout in (
        (640, [0, 64, 192, 384, 640]),
        (1024, [0, 100, 100, 300, 1024]),
        (768, [0, 64, 64, 200, 500, 700, 768]),
        (640, [0, 10, 300, 300, 640]),
    ):
        a = _randn(total_m, 128, dtype=torch.bfloat16)
        offs = torch.tensor(layout, device=DEVICE, dtype=torch.int32)
        bound = kernel.bind((a, b, offs))
        # One bound kernel per (total_M, G); the same shape reuses it.
        assert seen.setdefault(id(bound), total_m) == total_m
        with bound.env:
            state = bound.config_spec._cute_tcgen05_config
            assert state.warp_mma_admitted
            assert state.warp_mma_grouped_extents == (total_m, 128, 128)
            normalized = bound.config_spec.normalized_config(config)
        source = bound.to_code(normalized)
        assert "_helion_warp_mma.mma_bf16(" in source
        assert f"cutlass.Int32({total_m})" in source
        assert source.count("group_offsets.iterator + cutlass.Int64(") == len(layout)
        bound.set_config(normalized)
        out = bound(a, b, offs)
        torch.cuda.synchronize()
        ref = _grouped_reference(a, b, offs)
        covered = ~torch.isnan(ref)
        _check(out[covered], ref[covered].float())
    assert len(seen) == 3

    a = _randn(1000, 128, dtype=torch.bfloat16)
    offs = torch.tensor([0, 100, 100, 300, 1000], device=DEVICE, dtype=torch.int32)
    bound = kernel.bind((a, b, offs))
    assert id(bound) not in seen
    with bound.env:
        assert not bound.config_spec._cute_tcgen05_config.warp_mma_admitted
        with pytest.raises(helion.exc.InvalidConfig, match="not admitted"):
            bound.config_spec.normalized_config(config)
        repaired = dict(config.config)
        bound.config_spec.normalize(repaired, _fix_invalid=True)
    assert repaired[FAMILY] == family.MATMUL_FAMILY_TCGEN05
    bound.set_config(helion.Config(**repaired))
    assert "_helion_warp_mma" not in bound.to_code(bound._config)
    out = bound(a, b, offs)
    torch.cuda.synchronize()
    ref = _grouped_reference(a, b, offs)
    covered = ~torch.isnan(ref)
    _check(out[covered], ref[covered].float())


@_sm100_gpu
@pytest.mark.cuda
def test_gelu_aux_epilogue_matches_torch() -> None:
    # The cuBLASLt GELU_AUX pattern (bias prefix, two outputs, the chain's
    # ``_gelu_erf`` helper) on the family: the exact erf on per-fragment
    # scalars against torch's gelu of the fp32 pre-activation.
    torch.manual_seed(0)
    a = _randn(256, 256, dtype=torch.float16)
    b = _randn(256, 256, dtype=torch.float16)
    bias = _randn(256, dtype=torch.float16)
    pre = torch.matmul(a.float(), b.float()) * 1.25 + bias.float()
    for block_sizes, warps in (([64, 16, 64], 1), ([32, 32, 128], 4)):
        (out, aux), source = _run(
            _bias_gelu_aux_gemm, (a, b, bias), _config(block_sizes, warps)
        )
        assert "gelu_erf" in source
        _check(aux, pre)
        _check(out, torch.nn.functional.gelu(pre))


@_sm100_gpu
@pytest.mark.cuda
def test_m_major_int16_cast_operand_matches_torch() -> None:
    # ``x[tile_m, tile_k].to(bf16)`` of an M-contiguous int16 A: the packets
    # are loaded as words, converted in registers into the M-major staging
    # and read by ``ldmatrix.trans``.
    example = import_example("bf16xint16_gemm")

    torch.manual_seed(0)
    x = torch.randint(-8, 9, (256, 128), device=DEVICE, dtype=torch.int16).T
    w = _randn(256, 128, dtype=torch.bfloat16)
    assert x.shape == (128, 256) and x.stride() == (1, 128)
    kernel = helion.kernel(example._int16xbf16_gemm.fn, backend="cute")
    bound = kernel.bind((x, w))
    with bound.env:
        state = bound.config_spec._cute_tcgen05_config
        assert state.warp_mma_admitted and not state.tcgen05_family_admitted
    for block_sizes, warps in (([16, 8, 128], 1), ([32, 16, 64], 2)):
        config = _config(block_sizes, warps)
        bound.set_config(config)
        with bound.env:
            normalized = bound.config_spec.normalized_config(config)
        source = bound.to_code(normalized)
        assert "_helion_warp_mma.cvt_s16x8_bf16_sts(wm_a" in source
        assert "_helion_warp_mma.ldmatrix_x4_trans(" in source
        out = bound(x, w)
        torch.cuda.synchronize()
        _check(out, torch.matmul(x.float(), w.float()))


@_sm100_gpu
@pytest.mark.cuda
@pytest.mark.parametrize(
    "kernel_fn", [_extra_root_store_after, _extra_root_store_before]
)
def test_extra_root_store_fails_closed_on_the_family(kernel_fn) -> None:
    # The family refuses a kernel whose root tile loop holds a store it did
    # not match instead of dropping it.  (The default tcgen05 config of this
    # kernel fails to compile at the base commit as well -- ``indices_0`` is
    # undefined for the SIMT store next to the role body -- a pre-existing
    # defect outside this family, recorded in the lane report.)
    torch.manual_seed(0)
    a = _randn(256, 256, dtype=torch.float16)
    b = _randn(256, 256, dtype=torch.float16)
    other = _randn(256, 256, dtype=torch.float16)
    kernel = helion.kernel(kernel_fn, backend="cute", static_shapes=True)
    bound = kernel.bind((a, b, other))
    with pytest.raises(
        BackendUnsupported, match="root tile loop may only hold the GEMM"
    ):
        bound.set_config(_config([16, 8, 128], 1))
        bound(a, b, other)


@_sm100_gpu
@pytest.mark.cuda
def test_in_place_residual_on_strided_output_views() -> None:
    # The output is also the residual aux: a 2-byte-offset view (element
    # loads / stores, row stride 516 B) and an 8-byte-offset view (pairs)
    # must both be read before they are written, in place, no staging copy.
    torch.manual_seed(0)
    a = _randn(256, 256, dtype=torch.float16)
    b = _randn(256, 256, dtype=torch.float16)
    ref = torch.matmul(a.float(), b.float())
    kernel = helion.kernel(_inplace_residual, backend="cute", static_shapes=True)
    for width, offset, block_sizes, warps in (
        (258, 1, [16, 8, 128], 1),
        (264, 4, [32, 16, 128], 2),
        (256, 0, [64, 8, 128], 4),
    ):
        full = _randn(256, width, dtype=torch.float16)
        view = full[:, offset : offset + 256]
        before = full.clone()
        bound = kernel.bind((a, b, view))
        bound.set_config(_config(block_sizes, warps))
        result = bound(a, b, view)
        torch.cuda.synchronize()
        _check(result, ref + before[:, offset : offset + 256].float())
        # Columns outside the view are untouched.
        assert torch.equal(full[:, :offset], before[:, :offset])
        assert torch.equal(full[:, offset + 256 :], before[:, offset + 256 :])


@_sm100_gpu
@pytest.mark.cuda
def test_sliced_output_leaves_the_rest() -> None:
    torch.manual_seed(0)
    a = _randn(256, 256, dtype=torch.float16)
    b = _randn(256, 256, dtype=torch.float16)
    full = _randn(256, 512, dtype=torch.float16)
    keep = full[:, 256:].clone()
    kernel = helion.kernel(_sliced_output, backend="cute", static_shapes=True)
    bound = kernel.bind((a, b, full))
    bound.set_config(_config([16, 8, 128], 1))
    result = bound(a, b, full)
    torch.cuda.synchronize()
    _check(result[:, :256], torch.matmul(a.float(), b.float()))
    assert torch.equal(result[:, 256:], keep)


@_sm100_gpu
@pytest.mark.cuda
def test_bf16xint16_materialized_operand_on_the_family() -> None:
    # The fissioned bf16 x int16 example: region 1 is a plain bf16 GEMM over
    # the materialized operand, admitted (and seeded) at (128, 256, 128).
    example = import_example("bf16xint16_gemm")

    torch.manual_seed(0)
    x = _randn(128, 256, dtype=torch.bfloat16)
    w = torch.randint(-(2**15), 2**15 - 1, (256, 128), device=DEVICE, dtype=torch.int16)
    kernel = helion.kernel(
        example._bf16xint16_gemm.fn, backend="cute", static_shapes=True
    )
    bound = kernel.bind((x, w))
    with bound.env:
        seed_configs = [
            seed
            for seed in seeds.autotune_seed_configs(
                bound.config_spec._cute_tcgen05_config
            )
            if seed.config.get(FAMILY) == family.MATMUL_FAMILY_WARP_MMA
        ]
    assert seed_configs
    ref = torch.matmul(x.float(), w.to(torch.bfloat16).float())
    for seed in seed_configs[:3]:
        bound = kernel.bind((x, w))
        bound.set_config(seed)
        with bound.env:
            normalized = bound.config_spec.normalized_config(seed)
        source = bound.to_code(normalized)
        assert "_helion_warp_mma.mma_bf16(" in source
        out = bound(x, w)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
