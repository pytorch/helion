"""Codegen and search-surface pins for the ``warp_mma`` matmul family.

CPU-only (fake CUDA target), like ``test_cute_tcgen05_fixed_cost.py`` whose
binders and fixture this file reuses: every test binds a kernel, pins a config
and inspects the generated CuTe source or the config surface.  Runtime
coverage lives in ``test_cute_warp_mma_gemm_gpu.py``.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from test.test_cute_tcgen05_fixed_chain import CPU_DEVICE
from test.test_cute_tcgen05_fixed_chain import _bind_bmm
from test.test_cute_tcgen05_fixed_chain import _bind_gemm
from test.test_cute_tcgen05_fixed_chain import _bind_two_output_gemm
from test.test_cute_tcgen05_fixed_chain import _cpu_only  # noqa: F401
from test.test_cute_tcgen05_fixed_chain import _kernel_body
from test.test_cute_tcgen05_fixed_chain import _ordinary_gemm
from test.test_cute_tcgen05_fixed_chain import _source
from test.test_cute_tcgen05_fixed_cost import _bind_row_bias_gemm
from test.test_cute_tcgen05_fixed_cost import _closure_bias_epilogue
from test.test_cute_tcgen05_fixed_cost import _gemm_with_epilogue

import helion
from helion._compiler.cute import cute_warp_mma_gemm as family
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl

if TYPE_CHECKING:
    import contextlib

pytestmark = skipUnlessBackends(["cute"])

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


@pytest.fixture(autouse=True)
def _fp8_capable() -> contextlib.AbstractContextManager[object]:
    # The shared CPU target stubs the MMA support without the e4m3 flag the
    # fp8 binders consult; the family's own admission does not need it.
    with patch_cute_mma_support(
        SimpleNamespace(
            universal=True,
            warp_f16bf16=True,
            warpgroup_f16bf16=True,
            tcgen05_f16bf16=True,
            tcgen05_f8=True,
        )
    ) as support:
        yield support


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


def _bind(fn: object, *args: torch.Tensor) -> helion.runtime.kernel.BoundKernel:
    return helion.kernel(fn, backend="cute", static_shapes=True).bind(args)  # pyrefly: ignore[bad-argument-type]


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


# --------------------------------------------------------------------------
# Admission, normalization and seeds
# --------------------------------------------------------------------------


def _fragment_low(bound: helion.runtime.kernel.BoundKernel, index: int) -> int:
    with bound.env:
        spec = bound.config_spec
        return spec.block_sizes[index]._fragment(spec).low


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
            assert spec._cute_tcgen05_config._warp_mma_seed_configs() == []
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
        seeds = bound.config_spec._cute_tcgen05_config._warp_mma_seed_configs()
    tiles = sorted(
        (tuple(seed.config["block_sizes"]), seed.config[WARPS]) for seed in seeds
    )
    assert tiles == [
        ((16, 8, 128), 1),
        ((16, 16, 128), 1),
        ((32, 16, 128), 2),
        ((32, 32, 128), 4),
        ((64, 8, 128), 4),
    ], tiles
    assert all(seed.config[FAMILY] == "warp_mma" for seed in seeds)
    bmm = _bind_bmm((4, 64, 128, 128), torch.float16)
    with bmm.env:
        seeds = bmm.config_spec._cute_tcgen05_config._warp_mma_seed_configs()
    tiles = sorted(tuple(seed.config["block_sizes"]) for seed in seeds)
    assert tiles == [
        (1, 16, 8, 128),
        (1, 16, 16, 128),
        (1, 32, 16, 128),
        (1, 32, 32, 128),
        (1, 64, 8, 128),
    ], tiles
    full = _bind_gemm((2048, 2048, 256), torch.float16)
    with full.env:
        assert full.config_spec._cute_tcgen05_config._warp_mma_seed_configs() == []
    with bound.env:
        all_seeds = bound.config_spec._cute_tcgen05_config.autotune_seed_configs()
    assert sum(seed.config.get(FAMILY) == "warp_mma" for seed in all_seeds) == 5


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
    # it must refuse the family instead of silently dropping that store; the
    # default (tcgen05) config of the same kernel keeps both stores.
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
    kernel = _kernel_body(bound.to_code(default))
    assert "def _helion_" in kernel and "other" in kernel and "flag" in kernel
    assert "_helion_warp_mma" not in kernel


def test_admission_mirrors_the_detector_gates() -> None:
    # Kernels the detector always declines never receive the family keys or
    # its seeds (each would be a dead compile in a cold autotune).
    permuted = _bind(
        _permuted_lhs_gemm,
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
    )
    misaligned_rows = _bind(
        _ordinary_gemm,
        torch.empty((256, 260), dtype=torch.float16, device=CPU_DEVICE)[:, :256],
        torch.empty((256, 256), dtype=torch.float16, device=CPU_DEVICE),
    )
    for bound in (permuted, misaligned_rows):
        with bound.env:
            spec = bound.config_spec
            assert not spec._cute_tcgen05_config.warp_mma_admitted
            assert FAMILY not in spec.flat_config(lambda x: x.default()).config
            assert spec._cute_tcgen05_config._warp_mma_seed_configs() == []
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


def test_fissioned_kernel_region_renders_the_family() -> None:
    # bf16 x int16 at (128, 256, 128): ``cute_materialize_transformed_operands``
    # fissions the kernel into an operand-materialization region and a plain
    # bf16 GEMM region (three graphs).  The complete-root rule inspects the
    # root that calls the K loop; the other region keeps its own lowering,
    # so the family is admitted, seeded and renders the GEMM region.
    import examples.bf16xint16_gemm as example

    bound = _bind(
        example._bf16xint16_gemm.fn,
        torch.empty((128, 256), dtype=torch.bfloat16, device=CPU_DEVICE),
        torch.empty((256, 128), dtype=torch.int16, device=CPU_DEVICE),
    )
    with bound.env:
        spec = bound.config_spec
        assert len(bound.host_function.device_ir.graphs) == 3
        assert spec._cute_tcgen05_config.warp_mma_admitted
        seeds = spec._cute_tcgen05_config._warp_mma_seed_configs()
    assert seeds
    source = _source(bound, seeds[0])
    assert f"{RT}.mma_bf16(" in source
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
    from helion.runtime.cute.launcher import _CUTE_STAGED_PLAN_OPERANDS

    assert family.WARP_MMA_PLAN_KIND not in _CUTE_STAGED_PLAN_OPERANDS
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
