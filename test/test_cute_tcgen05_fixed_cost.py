"""Codegen pins for the fixed-cost trims of the plain tcgen05 GEMM launch.

CPU-only (the fake CUDA target of the module's ``cpu_render`` marker), with
the binders of ``test_cute_tcgen05_fixed_chain.py``: every test binds a
kernel, pins a config and inspects the generated CuTe source.  The runtime
numerics (sm_100 GPU) are at the end of the module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import pytest
import torch

from test.cute_population_contracts import tcgen05_mma_support
from test.test_cute_tcgen05_fixed_chain import CPU_DEVICE
from test.test_cute_tcgen05_fixed_chain import PUBLICATION_HOISTED
from test.test_cute_tcgen05_fixed_chain import _addmm_gemm
from test.test_cute_tcgen05_fixed_chain import _batched_gemm
from test.test_cute_tcgen05_fixed_chain import _bias_gemm
from test.test_cute_tcgen05_fixed_chain import _bind_bias_gemm
from test.test_cute_tcgen05_fixed_chain import _bind_bmm
from test.test_cute_tcgen05_fixed_chain import _bind_gemm
from test.test_cute_tcgen05_fixed_chain import _config
from test.test_cute_tcgen05_fixed_chain import _gemm
from test.test_cute_tcgen05_fixed_chain import _kernel_body
from test.test_cute_tcgen05_fixed_chain import _plain_config
from test.test_cute_tcgen05_fixed_chain import _source

import helion
from helion._compiler.cute.tcgen05_config import facts
from helion._compiler.cute.tcgen05_config import seeds
from helion._compiler.cute.tcgen05_config import surface
from helion._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_DIRECT
from helion._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_NORMAL
from helion._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODES
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_NORMAL
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUTS
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing.cute_dense_tcgen05 import wrapper_plans
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl

# The B200 model with tcgen05 MMA support, CUDA hidden, the inputs traced
# as ``cuda:0`` tensors of a 148-SM device, and the reduction SMEM budget.
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.cpu_render(
        mma_support=tcgen05_mma_support(),
        forbid_native=False,
        fake_device_sm_count=148,
        reduction_smem_budget_bytes=232448,
    ),
]

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import BoundKernel


def _row_bias_gemm(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = (acc + bias[tile_n]).to(out.dtype)
    return out


def _bind_row_bias_gemm(shape: tuple[int, int, int], dtype: torch.dtype) -> BoundKernel:
    m, n, k = shape
    args = (
        torch.empty((m, k), dtype=dtype, device=CPU_DEVICE),
        torch.empty((k, n), dtype=dtype, device=CPU_DEVICE),
        torch.empty((n,), dtype=dtype, device=CPU_DEVICE),
    )
    return helion.kernel(_row_bias_gemm, backend="cute", static_shapes=True).bind(args)


def _sched_params(kernel: str) -> list[str]:
    return re.findall(r"PersistentTileSchedulerParams\(.*", kernel)


# --------------------------------------------------------------------------
# L1: the one-shot role scheduler no longer depends on the L2 swizzle knob
# --------------------------------------------------------------------------


@pytest.mark.parametrize("swizzle", [2, 4])
def test_one_tile_per_cta_takes_the_one_shot_path_for_any_l2_swizzle(
    swizzle: int,
) -> None:
    # 512x512 with 128x64 tiles: 32 tiles <= 148 SMs.  The campaign's tiny
    # winners carried swizzle 2 / 4 and used to fall back to the persistent
    # ``while`` form; with one tile per CTA the swizzle is a no-op, so the plan
    # drops it and the render equals the swizzle-1 render byte for byte.
    bound = _bind_gemm((512, 512, 256), torch.float16)
    swizzled = _source(
        bound,
        _plain_config(
            pid_type="persistent_interleaved", tcgen05_l2_swizzle_size=swizzle
        ),
    )
    plain = _source(
        bound,
        _plain_config(pid_type="persistent_interleaved", tcgen05_l2_swizzle_size=1),
    )
    assert swizzled == plain
    kernel = _kernel_body(swizzled)
    assert "while tcgen05_role_local" not in kernel
    assert "advance_to_next_work" not in kernel
    assert all("swizzle_size" not in params for params in _sched_params(kernel))
    assert PUBLICATION_HOISTED in kernel
    assert kernel.count("tcgen05_tmem_allocator.free(") == 1


def test_more_tiles_than_sms_keeps_the_swizzled_persistent_loop() -> None:
    # 2048x4096 with 128x64 tiles: 1024 tiles > 148 SMs -> the persistent form
    # keeps the configured swizzle (and its padding guard).
    kernel = _kernel_body(
        _source(
            _bind_gemm((2048, 4096, 256), torch.float16),
            _plain_config(pid_type="persistent_interleaved", tcgen05_l2_swizzle_size=2),
        )
    )
    assert "while tcgen05_role_local" in kernel
    params = _sched_params(kernel)
    assert params and all("swizzle_size=2" in p for p in params), params


# --------------------------------------------------------------------------
# L3: the L2-grouping decode folds its runtime divisor when the M tile count
# is static
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "grouping", "expected"),
    [
        # 4 M tiles, group of 32: a single group holds every M tile.
        ((512, 512, 256), 32, "group_size_m = num_pid_m"),
        # 4 M tiles, group of 2: every group is full.
        ((512, 512, 256), 2, "group_size_m = 2"),
        # 6 M tiles, group of 4: the last group is partial -> runtime min.
        ((768, 512, 256), 4, "group_size_m = min(num_pid_m - first_pid_m, 4)"),
    ],
    ids=["single_group", "full_groups", "partial_group"],
)
def test_static_l2_group_size_folds_to_a_constant(
    shape: tuple[int, int, int], grouping: int, expected: str
) -> None:
    source = _source(
        _bind_gemm(shape, torch.float16),
        _plain_config(l2_groupings=[grouping], pid_type="persistent_blocked"),
    )
    assert expected in source, source
    # The decode itself is unchanged: the fold only replaces the divisor.
    assert "pid_1 = inner_2d_pid % num_pid_in_group // group_size_m" in source


# --------------------------------------------------------------------------
# L2: ``tcgen05_c_store_mode="direct"`` -- register-direct vector stores
# --------------------------------------------------------------------------


def _assert_direct_store(kernel: str, output: str, dtype: str) -> None:
    assert "tcgen05_tma_store_atom" not in kernel
    assert "PipelineTmaStore" not in kernel
    assert "tcgen05_sD_ptr" not in kernel
    assert "tcgen05_epilog_sync_barrier.arrive_and_wait()" not in kernel
    assert "fence_view_async_shared" not in kernel
    assert "cute.nvgpu.CopyR2GOp()" in kernel
    # Statically full tiles: one unconditional vector copy per subtile, no
    # runtime full-tile predicate and no scalar edge loop.
    assert "tcgen05_full_tile" not in kernel
    assert "_edge_i" not in kernel
    assert (
        kernel.count(
            "cute.copy(tcgen05_simt_atom, tcgen05_tTR_rD, tcgen05_tTR_gC_subtile)"
        )
        == 1
    )
    # The provable 16-byte alignment is re-asserted on the per-subtile pointer
    # (same address, no rounding) and the copy atom widens to it.
    assert (
        f"tcgen05_num_bits = min(min({output}.iterator.alignment, 16) * 8, "
        f"cute.size(tcgen05_mcld) * {dtype}.width)"
    ) in kernel
    assert (
        f"cute.make_ptr({dtype}, tcgen05_tTR_gC_subtile.iterator.toint(), "
        f"cute.AddressSpace.gmem, assumed_align=min({output}.iterator.alignment, 16))"
    ) in kernel
    # Without a store drain the teardown frees TMEM (no in-epilogue free).
    assert kernel.count("tcgen05_tmem_allocator.free(") == 1
    assert "tcgen05_c_pipeline" not in kernel


def test_direct_store_mode_renders_vector_register_stores() -> None:
    bound = _bind_gemm((512, 512, 256), torch.float16)
    requested = _plain_config(
        pid_type="persistent_interleaved", tcgen05_c_store_mode="direct"
    )
    with bound.env:
        config = bound.config_spec.normalized_config(requested)
    # An exact alternative body: no diagnostic opt-in needed, and the
    # normalizer keeps the request.
    assert config[TCGEN05_C_STORE_MODE_CONFIG_KEY] == TCGEN05_C_STORE_MODE_DIRECT
    assert "tcgen05_diagnostic_invalid_output" not in config.config
    source = bound.to_code(config)
    kernel = _kernel_body(source)
    _assert_direct_store(kernel, "out", "cutlass.Float16")
    # The host side builds no D tensormap.
    assert "'kind': 'tcgen05_d_tma'" not in source
    # Still the one-shot chain: hoisted TMA role, merged init.
    assert PUBLICATION_HOISTED in kernel


def test_direct_store_mode_on_a_batched_output() -> None:
    kernel = _kernel_body(
        _source(
            _bind_bmm((8, 256, 256, 512), torch.float16),
            _plain_config(
                block_sizes=[1, 128, 64, 64],
                loop_orders=[[0, 1, 2]],
                tcgen05_c_store_mode="direct",
            ),
        )
    )
    _assert_direct_store(kernel, "out", "cutlass.Float16")
    assert "tcgen05_gmem2d" in kernel


def test_direct_store_mode_on_a_64_row_tile() -> None:
    # bm=64 switches the TMEM load atom; the alignment proof is per chunk and
    # does not depend on the atom.
    kernel = _kernel_body(
        _source(
            _bind_gemm((256, 256, 256), torch.float16),
            _plain_config(
                block_sizes=[64, 16, 128],
                pid_type="persistent_interleaved",
                tcgen05_c_store_mode="direct",
            ),
        )
    )
    _assert_direct_store(kernel, "out", "cutlass.Float16")


def test_direct_store_mode_with_a_bias_row_loads_the_row_before_the_wait() -> None:
    # The SIMT body's pre-wait aux placement: the bias fragment is loaded
    # ahead of the accumulator wait so its latency overlaps the MMA tail.
    kernel = _kernel_body(
        _source(
            _bind_bias_gemm((512, 512, 256), torch.float16),
            _plain_config(
                pid_type="persistent_interleaved",
                tcgen05_c_store_mode="direct",
                tcgen05_aux_load_placement="pre_acc_wait",
            ),
        )
    )
    _assert_direct_store(kernel, "out", "cutlass.Float16")
    aux_load = kernel.index("tcgen05_aux_loaded_0 = ")
    acc_wait = kernel.index("tcgen05_acc_pipeline.consumer_wait(")
    assert aux_load < acc_wait


def test_direct_store_mode_is_exact_and_searchable() -> None:
    bound = _bind_gemm((512, 512, 256), torch.float16)
    tcgen05 = bound.config_spec._cute_tcgen05_config
    with bound.env:
        search = surface.optional_fragments(tcgen05, for_search=True)
        validation = surface.optional_fragments(tcgen05, for_search=False)
    assert search[TCGEN05_C_STORE_MODE_CONFIG_KEY].choices == (
        TCGEN05_C_STORE_MODE_NORMAL,
        TCGEN05_C_STORE_MODE_DIRECT,
    )
    assert validation[TCGEN05_C_STORE_MODE_CONFIG_KEY].choices == TCGEN05_C_STORE_MODES
    # The invalid-output diagnostic still needs its opt-in.
    with bound.env, pytest.raises(InvalidConfig, match="changes output correctness"):
        bound.config_spec.normalized_config(
            _plain_config(tcgen05_c_store_mode="skip_epilogue_store")
        )


def test_normal_store_mode_render_is_unchanged_by_the_knob() -> None:
    bound = _bind_gemm((512, 512, 256), torch.float16)
    implicit = _source(bound, _plain_config(pid_type="persistent_interleaved"))
    explicit = _source(
        bound,
        _plain_config(pid_type="persistent_interleaved", tcgen05_c_store_mode="normal"),
    )
    assert implicit == explicit
    assert "tcgen05_tma_store_atom" in _kernel_body(implicit)


# --------------------------------------------------------------------------
# Seeds: the small-grid family covers batched GEMMs
# --------------------------------------------------------------------------


def test_small_grid_seeds_cover_batched_gemms() -> None:
    # 4 x 64x128x128: four standard tiles on 148 SMs.  The batch axis is one
    # of the seed's block sizes (pinned to 1) and multiplies the tile count.
    bound = _bind_bmm((4, 64, 128, 128), torch.float16)
    with bound.env:
        seed_configs = seeds._small_grid_seed_configs(
            bound.config_spec._cute_tcgen05_config
        )
    tiles = sorted({tuple(seed.config["block_sizes"]) for seed in seed_configs})
    assert tiles == [(1, 64, 16, 128), (1, 64, 32, 128), (1, 64, 64, 128)], tiles
    for seed in seed_configs:
        assert seed.config["pid_type"] == "persistent_interleaved"
        # K = 128 in one 128-wide step: the ring is never deeper than the loop.
        assert seed.config["tcgen05_ab_stages"] == 1


def test_small_grid_seeds_stop_once_the_batched_grid_fills_the_machine() -> None:
    # 64 x 128x128x256: 64 standard tiles would qualify unbatched, but the
    # batch axis fills the machine 64-fold.
    bound = _bind_bmm((64, 256, 256, 256), torch.float16)
    with bound.env:
        seed_configs = seeds._small_grid_seed_configs(
            bound.config_spec._cute_tcgen05_config
        )
    assert seed_configs == []


def test_small_grid_seeds_prefetch_the_bias_row() -> None:
    # 256^3 + bias: every small-grid seed loads its bias row ahead of the
    # accumulator wait (the promoted row stage), like the one-wave seeds.
    bound = _bind_row_bias_gemm((256, 256, 256), torch.float16)
    with bound.env:
        seed_configs = seeds._small_grid_seed_configs(
            bound.config_spec._cute_tcgen05_config
        )
    assert seed_configs
    for seed in seed_configs:
        assert seed.config["tcgen05_aux_load_placement"] == "pre_acc_wait", seed


def test_small_grid_seeds_twin_narrow_tiles_with_the_direct_store() -> None:
    bound = _bind_gemm((256, 256, 256), torch.float16)
    with bound.env:
        seed_configs = seeds._small_grid_seed_configs(
            bound.config_spec._cute_tcgen05_config
        )
    by_mode: dict[str, list[tuple[int, int]]] = {}
    for seed in seed_configs:
        mode = seed.config.get(
            TCGEN05_C_STORE_MODE_CONFIG_KEY, TCGEN05_C_STORE_MODE_NORMAL
        )
        by_mode.setdefault(mode, []).append(tuple(seed.config["block_sizes"][:2]))
    # Every tile keeps its TMA-store seed; only the narrow ones (<= 32 columns)
    # get the register-direct twin.
    assert sorted(by_mode[TCGEN05_C_STORE_MODE_NORMAL]) == [
        (64, 16),
        (64, 32),
        (64, 64),
        (128, 16),
        (128, 32),
        (128, 64),
    ]
    assert sorted(by_mode[TCGEN05_C_STORE_MODE_DIRECT]) == [
        (64, 16),
        (64, 32),
        (128, 16),
        (128, 32),
    ]


# --------------------------------------------------------------------------
# Search floor: grids smaller than the machine admit 64-row tiles
# --------------------------------------------------------------------------


def _m_block_fragment_low(bound: BoundKernel) -> int:
    with bound.env:
        fragments = facts._matmul_block_fragments(
            bound.config_spec._cute_tcgen05_config
        )
    assert fragments is not None
    return fragments[0].low


def test_small_grids_admit_64_row_tiles_in_search() -> None:
    # 256^3: four 128x128 tiles on 148 SMs.  The operands admit 256-row tiles,
    # which used to pin the search floor at 128 rows; the 64-row one-CTA tile
    # (cuBLAS runs this shape as 128 CTAs of 64x8) is searchable, and the
    # register-MMA family admitted on this latency-bound problem widens the
    # floor to its 16-row atom (tcgen05-family configs clamp back to 64, see
    # test_cute_warp_mma_gemm.py).
    assert _m_block_fragment_low(_bind_gemm((256, 256, 256), torch.float16)) == 16
    # A grid that fills the machine keeps the 128-row floor.
    assert _m_block_fragment_low(_bind_gemm((2048, 2048, 256), torch.float16)) == 128
    # The batch axis counts toward the grid: 64 x 256x256 fills it.
    assert _m_block_fragment_low(_bind_bmm((64, 256, 256, 256), torch.float16)) == 128
    # 8 x 256x512x256 is small-grid but beyond the register-MMA family's
    # latency-bound regime: the 64-row tcgen05 floor stays.
    assert _m_block_fragment_low(_bind_bmm((8, 256, 512, 256), torch.float16)) == 64


def test_small_grid_seeds_keep_64_row_tiles_within_one_wave() -> None:
    # 512x1024x256: 32 standard tiles, so both row widths are searchable.  The
    # 64-row seeds appear only while their grid fits the 148 SMs (64x64 -> 128
    # CTAs; 64x32 and 64x16 would need a second wave), next to every 128-row
    # tile.
    bound = _bind_gemm((512, 1024, 256), torch.float16)
    with bound.env:
        seed_configs = seeds._small_grid_seed_configs(
            bound.config_spec._cute_tcgen05_config
        )
    tiles = sorted({tuple(seed.config["block_sizes"]) for seed in seed_configs})
    assert tiles == [
        (64, 64, 128),
        (128, 16, 128),
        (128, 32, 128),
        (128, 64, 128),
    ], tiles


# --------------------------------------------------------------------------
# N=8 tiles launch one physical warp per role row, like every other tile
# --------------------------------------------------------------------------


def _inline_n8_config() -> helion.Config:
    """The campaign's bmm s0 winner: 64x8 tiles on the inline ``xyz`` path."""
    return helion.Config(
        block_sizes=[1, 64, 8, 64],
        loop_orders=[[0, 1, 2]],
        l2_groupings=[16],
        indexing=["pointer", "tensor_descriptor", "tensor_descriptor"],
        pid_type="xyz",
        tcgen05_persistence_model="non_persistent",
        tcgen05_cluster_m=1,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=4,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=4,
        tcgen05_tvm_ffi_launch=False,
    )


def _tma_prologue(kernel: str) -> str:
    """The TMA-load prologue: from the first full-tile gate to the K loop."""
    start = kernel.index("tcgen05_tma_initial_full_tile =")
    return kernel[start : kernel.index("for tile_offset_3 in range(", start)]


def test_narrow_n8_tile_launches_exactly_the_role_warps() -> None:
    # The N=8 tile used to keep a tile-wide 64-lane SIMT M axis, and the role
    # launch multiplies that width by the warp count: block (64, 6, 1), twelve
    # warps for six roles.  The six spare warps were "not epilogue" warps and
    # arrived on the 96-thread pipeline-init barrier, so the TMA warp could
    # pass it before warp 0 had initialized the mbarriers and fault on its
    # first ``try_wait`` (the flaky ``unspecified launch failure`` of this
    # config).  One warp per role row, like every other tile.
    source = _source(_bind_bmm((4, 64, 128, 128), torch.float16), _inline_n8_config())
    kernel = _kernel_body(source)
    assert "block=(32, 6, 1)" in source
    assert "barrier_id=3, num_threads=96)" in kernel
    assert "tcgen05_exec_active = tcgen05_warp_idx == cutlass.Int32(4)" in kernel
    assert "tcgen05_tma_warp = tcgen05_warp_idx == cutlass.Int32(5)" in kernel
    assert "cute.copy(tcgen05_tma_store_atom" in kernel
    # The 32-lane row leaves root lane loops over M, which the MMA lowering
    # suppresses, so the thread-barrier pass no longer runs over a lane-free
    # root body and no longer separates the TMA prologue stages (opaque
    # copies on one SMEM tensor) with CTA-wide barriers.
    assert "cute.arch.sync_threads()" not in _tma_prologue(kernel)
    # The inline path's own CTA-wide syncs (acc handoff, epilogue, drain) stay.
    assert kernel.count("cute.arch.sync_threads()") == 3


@pytest.mark.parametrize("ab_stages", [1, 2], ids=["ab1", "ab2"])
def test_narrow_n8_one_shot_chain_has_no_cta_wide_barrier(ab_stages: int) -> None:
    # The same tile on the one-shot role-local path, with 2 K tiles.  The
    # hoisted TMA-load role issues the whole K loop ahead of the MMA role, so a
    # CTA-wide barrier between the two deadlocks as soon as the K extent
    # exceeds the AB ring: the TMA warp cannot reach the barrier before the
    # MMA warp releases a stage, and the MMA warp waits at the barrier.  The
    # thread-barrier pass put one there for the N=8 family (the 30 s
    # autotuner timeouts of every ``tcgen05_ab_stages=1`` sample).
    source = _source(
        _bind_bmm((4, 64, 128, 128), torch.float16),
        _plain_config(
            block_sizes=[1, 64, 8, 64],
            loop_orders=[[0, 1, 2]],
            pid_type="persistent_interleaved",
            tcgen05_ab_stages=ab_stages,
            tcgen05_c_stages=4,
        ),
    )
    kernel = _kernel_body(source)
    assert "block=(32, 6, 1)" in source
    assert PUBLICATION_HOISTED in kernel
    assert "while tcgen05_role_local" not in kernel
    assert "cute.arch.sync_threads()" not in kernel
    # Both K tiles are issued by the hoisted role: the prologue's fresh-ring
    # acquire and the K loop's waiting acquire (the ring is reused).
    assert (
        "tcgen05_ab_pipeline.producer_acquire(tcgen05_ab_producer_state, True)"
        in kernel
    )
    assert (
        "tcgen05_ab_pipeline.producer_acquire(tcgen05_ab_producer_state, "
        "tcgen05_ab_producer_try_token)"
    ) in kernel


def test_narrow_n8_persistent_chain_has_no_cta_wide_barrier() -> None:
    # 8 x 2 x 32 = 512 tiles of 128x8 over 148 CTAs: the persistent while,
    # several tiles per CTA, four K tiles over a two-stage ring.
    source = _source(
        _bind_bmm((8, 256, 256, 256), torch.float16),
        _plain_config(
            block_sizes=[1, 128, 8, 64],
            loop_orders=[[0, 1, 2]],
            tcgen05_c_stages=4,
        ),
    )
    kernel = _kernel_body(source)
    assert "block=(32, 6, 1)" in source
    assert "while tcgen05_role_local" in kernel
    assert "cute.arch.sync_threads()" not in kernel


def test_wider_tiles_render_unchanged_by_the_n8_rule() -> None:
    # A 64x16 tile already mapped a 32-lane M axis; its render is the same
    # six-warp launch with the inline path's three syncs.
    config = _inline_n8_config()
    source = _source(
        _bind_bmm((4, 64, 128, 128), torch.float16),
        helion.Config(**{**config.config, "block_sizes": [1, 64, 16, 64]}),
    )
    kernel = _kernel_body(source)
    assert "block=(32, 6, 1)" in source
    assert kernel.count("cute.arch.sync_threads()") == 3


# --------------------------------------------------------------------------
# The TMA store needs a provably TensorMap-legal output (review F2)
# --------------------------------------------------------------------------


def _gemm_into(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _bind_gemm_into(out: torch.Tensor, *, k: int = 256) -> BoundKernel:
    m, n = out.shape
    args = (
        torch.empty((m, k), dtype=out.dtype, device=out.device),
        torch.empty((k, n), dtype=out.dtype, device=out.device),
        out,
    )
    return helion.kernel(_gemm_into, backend="cute", static_shapes=True).bind(args)


# Outputs the TMA store may address: a TensorMap needs a 16-byte base and
# 16-byte outer strides, with either matrix axis contiguous.
TMA_STORE_LEGAL_OUTPUTS = (
    "contiguous",
    "row_stride_16B",
    "column_major",
    "column_major_stride_16B",
)
# ... and views that violate exactly one of those constraints.
TMA_STORE_ILLEGAL_OUTPUTS = (
    "row_stride_8B",
    "column_major_stride_8B",
    "base_8B",
)


def _output_view(
    case: str, dtype: torch.dtype, device: str | torch.device = CPU_DEVICE
) -> torch.Tensor:
    """A 512x512 output of ``dtype`` whose layout matches ``case``."""
    rows, cols = 512, 512
    # Eight bytes per row (or ahead of the base): never a 16-byte multiple.
    pad = 8 // dtype.itemsize
    if case == "contiguous":
        return torch.zeros(rows, cols, dtype=dtype, device=device)
    if case == "row_stride_16B":
        return torch.zeros(rows, cols + 2 * pad, dtype=dtype, device=device)[:, :cols]
    if case == "row_stride_8B":
        return torch.zeros(rows, cols + pad, dtype=dtype, device=device)[:, :cols]
    if case == "column_major":
        return torch.zeros(cols, rows, dtype=dtype, device=device).T
    if case == "column_major_stride_16B":
        return torch.zeros(cols, rows + 2 * pad, dtype=dtype, device=device)[:, :rows].T
    if case == "column_major_stride_8B":
        return torch.zeros(cols, rows + pad, dtype=dtype, device=device)[:, :rows].T
    assert case == "base_8B"
    flat = torch.zeros(rows * cols + 2 * pad, dtype=dtype, device=device)
    return flat[pad : pad + rows * cols].view(rows, cols)


@pytest.mark.parametrize("case", TMA_STORE_LEGAL_OUTPUTS)
def test_tma_store_keeps_proven_output_arguments(case: str) -> None:
    # The output is a kernel argument: its base and stride residues come
    # from the bound kernel's pointer-alignment specialization.
    source = _source(
        _bind_gemm_into(_output_view(case, torch.float16)),
        _plain_config(pid_type="persistent_interleaved"),
    )
    kernel = _kernel_body(source)
    assert "cute.copy(tcgen05_tma_store_atom" in kernel
    assert "cute.nvgpu.CopyR2GOp()" not in kernel
    (output_plan,) = [
        plan for plan in wrapper_plans(source) if plan["kind"] == "tcgen05_d_tma"
    ]
    column_major = case.startswith("column_major")
    assert output_plan.get("d_column_major", False) is column_major
    assert (output_plan["bm"], output_plan["bn"]) == (128, 64)
    layout = "COL_MAJOR" if column_major else "ROW_MAJOR"
    assert f"tcgen05_c_layout = cutlass.utils.layout.LayoutEnum.{layout}" in kernel


@pytest.mark.parametrize("case", TMA_STORE_ILLEGAL_OUTPUTS)
def test_unproven_outputs_take_the_simt_store(case: str) -> None:
    source = _source(
        _bind_gemm_into(_output_view(case, torch.float16)),
        _plain_config(pid_type="persistent_interleaved"),
    )
    kernel = _kernel_body(source)
    # No D tensormap anywhere: not built by the host, not stored through.
    assert "tcgen05_tma_store_atom" not in kernel
    assert "PipelineTmaStore" not in kernel
    assert "'kind': 'tcgen05_d_tma'" not in source
    # The SIMT body addresses every element itself ...
    assert "cute.nvgpu.CopyR2GOp()" in kernel
    if case == "base_8B":
        # ... claiming no more than the pointer's runtime alignment on an
        # output whose strides prove the chunk layout (the launcher passes
        # the 8-byte base as ``assumed_align=8``: 64-bit stores) ...
        assert "assumed_align=min(out.iterator.alignment, 16)" in kernel
        assert "gmem, assumed_align=16)" not in kernel
    else:
        # ... and with no alignment claim at all on an unproven stride.
        assert "assumed_align=min(" not in kernel
    # ... while A and B keep their proven TMA loads.
    assert "cute.nvgpu.cpasync.prefetch_descriptor(" in kernel


def test_output_proof_is_independent_of_the_store_knob() -> None:
    # The register-direct store never used the D tensormap; an unproven
    # output renders the same body with and without the knob.
    out = _output_view("row_stride_8B", torch.float16)
    bound = _bind_gemm_into(out)
    implicit = _source(bound, _plain_config(pid_type="persistent_interleaved"))
    direct = _source(
        bound,
        _plain_config(pid_type="persistent_interleaved", tcgen05_c_store_mode="direct"),
    )
    assert "tcgen05_tma_store_atom" not in _kernel_body(implicit)
    assert "cute.nvgpu.CopyR2GOp()" in _kernel_body(direct)


# --------------------------------------------------------------------------
# Fresh outputs from the tensor-method factories and ``torch.empty_strided``
# are proven like ``torch.empty`` (review F1 of the post-reboot review)
# --------------------------------------------------------------------------


def _gemm_new_empty(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = a.new_empty([m, n])
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _gemm_new_zeros(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = a.new_zeros((m, n))
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _gemm_new_ones(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = a.new_ones([m, n], dtype=a.dtype)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _gemm_new_full(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = a.new_full([m, n], 0.0)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _gemm_empty_strided(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty_strided((m, n), (n, 1), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _gemm_empty_strided_column_major(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty_strided((m, n), (1, m), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


# Factories whose fresh, row-major result the TMA store may address ...
FRESH_FACTORY_GEMMS = {
    "new_empty": _gemm_new_empty,
    "new_zeros": _gemm_new_zeros,
    "new_ones": _gemm_new_ones,
    "new_full": _gemm_new_full,
    "empty_strided": _gemm_empty_strided,
}


def _bind_fresh_factory_gemm(
    kernel_fn: Callable[..., torch.Tensor],
    shape: tuple[int, int, int],
    dtype: torch.dtype,
) -> BoundKernel:
    m, n, k = shape
    args = (
        torch.empty((m, k), dtype=dtype, device=CPU_DEVICE),
        torch.empty((k, n), dtype=dtype, device=CPU_DEVICE),
    )
    return helion.kernel(kernel_fn, backend="cute", static_shapes=True).bind(args)


@pytest.mark.parametrize("factory", sorted(FRESH_FACTORY_GEMMS))
def test_tma_store_keeps_tensor_method_factory_outputs(factory: str) -> None:
    source = _source(
        _bind_fresh_factory_gemm(
            FRESH_FACTORY_GEMMS[factory], (512, 512, 256), torch.float16
        ),
        _plain_config(pid_type="persistent_interleaved"),
    )
    kernel = _kernel_body(source)
    assert "cute.copy(tcgen05_tma_store_atom" in kernel
    assert "cute.nvgpu.CopyR2GOp()" not in kernel
    assert "'kind': 'tcgen05_d_tma'" in source


def test_fresh_column_major_strided_output_keeps_the_tma_store() -> None:
    # The fresh allocation proves the base alignment and its M-contiguous
    # stride selects a column-major D tile and descriptor.
    source = _source(
        _bind_fresh_factory_gemm(
            _gemm_empty_strided_column_major, (512, 512, 256), torch.float16
        ),
        _plain_config(pid_type="persistent_interleaved"),
    )
    kernel = _kernel_body(source)
    assert "cute.copy(tcgen05_tma_store_atom" in kernel
    assert "PipelineTmaStore.create" in kernel
    assert "cute.nvgpu.CopyR2GOp()" not in kernel
    assert "tcgen05_c_layout = cutlass.utils.layout.LayoutEnum.COL_MAJOR" in kernel
    (output_plan,) = [
        plan for plan in wrapper_plans(source) if plan["kind"] == "tcgen05_d_tma"
    ]
    assert output_plan["d_column_major"] is True
    assert (output_plan["bm"], output_plan["bn"]) == (128, 64)


# --------------------------------------------------------------------------
# An explicit M-axis thread count other than one warp never reaches the
# tcgen05 role launch (review F3 of the post-reboot review)
# --------------------------------------------------------------------------


def test_explicit_m_axis_other_than_one_warp_takes_the_generic_simt_path() -> None:
    # The tcgen05 role launch is one physical warp per role row.  An explicit
    # ``num_threads`` M count of exactly 32 keeps the role launch; any other
    # explicit width is declined by the MMA detection before planning (and
    # by the pipeline itself) and renders the generic SIMT kernel instead of
    # launching warps the roles and the init barrier do not count.
    bound = _bind_gemm((512, 512, 256), torch.float16)
    for pid_type in ("flat", "persistent_blocked"):
        one_warp = _source(
            bound, _plain_config(num_threads=[32, 8, 0], pid_type=pid_type)
        )
        assert "block=(32, 6, 1)" in one_warp
        assert "cute.copy(tcgen05_tma_store_atom" in _kernel_body(one_warp)
    for num_threads in ([16, 8, 0], [64, 8, 0], [128, 8, 0]):
        generic = _source(
            bound, _plain_config(num_threads=num_threads, pid_type="flat")
        )
        assert f"block=({num_threads[0]}, 8, 1)" in generic, generic
        assert "tcgen05_tmem_alloc_barrier" not in generic
        assert "tcgen05_tma_store_atom" not in generic
        assert "mma_active =" not in generic
        # The persistent tcgen05 schedulers address role warps the generic
        # kernel does not launch: an unsupported pairing, said so up front
        # (it used to trip an internal assertion for the 16-thread request).
        with pytest.raises(BackendUnsupported, match="persistent tcgen05 pid_type"):
            _source(bound, _plain_config(num_threads=num_threads))


def test_explicit_n_axis_other_than_the_resolved_width_takes_the_generic_simt_path() -> (
    None
):
    # The role launch's y extent is the plan's warp count clamped by the SIMT
    # N axis, which the planning resolves to min(bn, 8).  An explicit N count
    # of exactly that width keeps the role launch; four used to launch a
    # four-warp CTA whose roles still addressed warps 4 and 5 (the MMA and
    # TMA warps never started: a deadlock), so any other explicit width is
    # declined like the M axis.
    bound = _bind_gemm((512, 512, 256), torch.float16)
    for pid_type in ("flat", "persistent_blocked"):
        resolved = _source(
            bound, _plain_config(num_threads=[0, 8, 0], pid_type=pid_type)
        )
        assert "block=(32, 6, 1)" in resolved
        assert "cute.copy(tcgen05_tma_store_atom" in _kernel_body(resolved)
    for n_threads in (2, 4, 16):
        generic = _source(
            bound, _plain_config(num_threads=[0, n_threads, 0], pid_type="flat")
        )
        assert f"block=(32, {n_threads}, 1)" in generic, generic
        assert "tcgen05_tmem_alloc_barrier" not in generic
        assert "tcgen05_warp_idx == cutlass.Int32(4)" not in generic
        with pytest.raises(BackendUnsupported, match="persistent tcgen05 pid_type"):
            _source(bound, _plain_config(num_threads=[0, n_threads, 0]))
    # The N=8 tile resolves to eight as well; four is declined there too.
    narrow = _bind_gemm((512, 8, 256), torch.float16)
    kept = _source(
        narrow, _plain_config(block_sizes=[128, 8, 64], num_threads=[0, 8, 0])
    )
    assert "block=(32, 6, 1)" in kept
    declined = _source(
        narrow,
        _plain_config(block_sizes=[128, 8, 64], num_threads=[0, 4, 0], pid_type="flat"),
    )
    assert "block=(32, 4, 1)" in declined
    assert "tcgen05_tmem_alloc_barrier" not in declined


# --------------------------------------------------------------------------
# An epilogue callable may read a module-level global (lifted globals are
# faked once per host function, like closure cells)
# --------------------------------------------------------------------------

# A CPU tensor at import time: the pins below only render.
GLOBAL_BIAS = torch.empty((512,), dtype=torch.float16)


def _gemm_with_epilogue(
    a: torch.Tensor, b: torch.Tensor, epilogue: Callable[..., torch.Tensor]
) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = epilogue(acc, (tile_m, tile_n)).to(out.dtype)
    return out


def _closure_bias_epilogue(bias: torch.Tensor) -> Callable[..., torch.Tensor]:
    return lambda acc, tile: acc + bias[tile[1]]


def _bind_epilogue_gemm(epilogue: Callable[..., torch.Tensor]) -> BoundKernel:
    args = (
        torch.empty((512, 256), dtype=torch.float16, device=CPU_DEVICE),
        torch.empty((256, 512), dtype=torch.float16, device=CPU_DEVICE),
        epilogue,
    )
    return helion.kernel(_gemm_with_epilogue, backend="cute", static_shapes=True).bind(
        args
    )


def test_epilogue_lambda_reading_a_module_global_renders_like_a_closure() -> None:
    # A static-shapes kernel fakes a global tensor through
    # ``torch.empty_strided``; read again while the device body was traced,
    # that allocation was recorded as a device ``empty_strided`` node with no
    # host origin and codegen failed with a KeyError in the aux-store splice.
    # The global is now faked once per host function and arrives as a
    # ``_global_source`` argument, like a closure cell.
    config = _plain_config(pid_type="persistent_interleaved")
    source = _source(
        _bind_epilogue_gemm(lambda acc, tile: acc + GLOBAL_BIAS[tile[1]]), config
    )
    kernel = _kernel_body(source)
    # The kernel's own module is imported as ``_source_module``.
    assert "_source_module.GLOBAL_BIAS" in source
    assert "empty_strided" not in kernel
    assert "cute.copy(tcgen05_tma_store_atom" in kernel
    assert "cute.nvgpu.CopyR2GOp()" not in kernel
    closure = _kernel_body(
        _source(_bind_epilogue_gemm(_closure_bias_epilogue(GLOBAL_BIAS)), config)
    )
    # Same kernel up to the bias argument's name.
    assert kernel.count("\n") == closure.count("\n")
    assert "cute.copy(tcgen05_tma_store_atom" in closure


# --------------------------------------------------------------------------
# Search hygiene: direct store x split epilogue layout (review F4)
# --------------------------------------------------------------------------

_TWO_CTA_BN256 = {
    "block_sizes": [256, 256, 64],
    "tcgen05_cta_group": "two",
    "tcgen05_cluster_m": 2,
    "pid_type": "persistent_blocked",
}


def test_direct_store_demotes_a_split_epilogue_layout() -> None:
    bound = _bind_gemm((2048, 2048, 256), torch.float16)
    for layout in TCGEN05_EPILOGUE_LAYOUTS:
        if layout == TCGEN05_EPILOGUE_LAYOUT_NORMAL:
            continue
        requested = _plain_config(
            **_TWO_CTA_BN256,
            tcgen05_c_store_mode="direct",
            tcgen05_epilogue_layout=layout,
        )
        # Repaired (search) configs fall back to the normal layout and keep
        # the store mode ...
        repaired = dict(requested.config)
        with bound.env:
            bound.config_spec.normalize(repaired, _fix_invalid=True)
        assert repaired[TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY] == (
            TCGEN05_EPILOGUE_LAYOUT_NORMAL
        )
        assert repaired[TCGEN05_C_STORE_MODE_CONFIG_KEY] == TCGEN05_C_STORE_MODE_DIRECT
        # ... explicit ones are rejected up front, not at codegen.
        with bound.env, pytest.raises(InvalidConfig, match="register-direct"):
            bound.config_spec.normalized_config(requested)
    # The TMA store keeps its layouts.
    with bound.env:
        kept = bound.config_spec.normalized_config(
            _plain_config(**_TWO_CTA_BN256, tcgen05_epilogue_layout="split_first_t2r")
        )
    assert kept[TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY] == "split_first_t2r"


# Runtime numerics for the tcgen05 fixed-cost trims (sm_100 GPU).
#
# The register-direct output store must be bitwise identical to the
# SMEM-staged TMA store of the same config (same conversion, same values), the
# one-shot path reached from a swizzled config must match the swizzle-1 kernel,
# and the folded L2-grouping decode must visit every tile exactly once.  Every
# case launches many times: a TMEM leak, a barrier-count mismatch or a bad
# alignment claim shows up as a hang, a fault or a wrong result within a few
# launches per SM.


def _sm100_gpu(test: Callable[..., object]) -> Callable[..., object]:
    """Skip ``test`` without a CUDA GPU of capability 10.0 or newer."""
    return skipIfNotCUDA()(
        skipIfCudaCapabilityLessThan((10, 0), reason="tcgen05 requires an sm_100 GPU")(
            test
        )
    )


LAUNCHES = 40


def _outputs(
    kernel_fn, args: tuple[torch.Tensor, ...], config: helion.Config
) -> tuple[list[torch.Tensor], str]:
    kernel = helion.kernel(kernel_fn, backend="cute", static_shapes=True)
    bound = kernel.bind(args)
    bound.set_config(config)
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    source = bound.to_code(normalized)
    outputs = [bound(*args) for _ in range(LAUNCHES)]
    torch.cuda.synchronize()
    return outputs, source


def _assert_direct_matches_tma(
    kernel_fn,
    args: tuple[torch.Tensor, ...],
    config: helion.Config,
    expected: torch.Tensor,
    **tol: float,
) -> None:
    normal, normal_source = _outputs(kernel_fn, args, config)
    direct, direct_source = _outputs(
        kernel_fn,
        args,
        helion.Config(**{**config.config, "tcgen05_c_store_mode": "direct"}),
    )
    assert "cute.copy(tcgen05_tma_store_atom" in normal_source
    assert "tcgen05_tma_store_atom" not in direct_source
    assert "cute.nvgpu.CopyR2GOp()" in direct_source
    # The 16-byte alignment proof held: vector stores, no per-element loop.
    assert "assumed_align=min(" in direct_source, direct_source
    assert "_edge_i" not in direct_source
    for out in normal:
        torch.testing.assert_close(out, expected.to(out.dtype), **tol)
    for out in direct:
        assert torch.equal(out, normal[0])


@_sm100_gpu
@pytest.mark.cuda
def test_direct_store_plain_fp16_one_tile_per_cta() -> None:
    torch.manual_seed(0)
    a = torch.randn(512, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 512, device=DEVICE, dtype=torch.float16)
    _assert_direct_matches_tma(
        _gemm,
        (a, b),
        _config(pid_type="persistent_interleaved"),
        a @ b,
        rtol=1e-2,
        atol=1e-2,
    )


@_sm100_gpu
@pytest.mark.cuda
def test_direct_store_plain_bf16_128_wide_tile() -> None:
    torch.manual_seed(0)
    a = torch.randn(1024, 512, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(512, 1024, device=DEVICE, dtype=torch.bfloat16)
    # 8 x 8 = 64 tiles of 128x128: four 32-column subtiles per tile.
    _assert_direct_matches_tma(
        _addmm_gemm,
        (a, b),
        _config(block_sizes=[128, 128, 64], pid_type="persistent_interleaved"),
        a @ b,
        rtol=2e-2,
        atol=2e-2,
    )


@_sm100_gpu
@pytest.mark.cuda
def test_direct_store_persistent_tiles() -> None:
    torch.manual_seed(0)
    a = torch.randn(2048, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 2048, device=DEVICE, dtype=torch.float16)
    # 16 x 32 = 512 tiles > 148 SMs: the persistent while, several tiles per
    # CTA, each stored straight from registers.
    _assert_direct_matches_tma(
        _gemm,
        (a, b),
        _config(pid_type="persistent_blocked", l2_groupings=[8]),
        a @ b,
        rtol=1e-2,
        atol=1e-2,
    )


@_sm100_gpu
@pytest.mark.cuda
def test_direct_store_batched_output() -> None:
    torch.manual_seed(0)
    a = torch.randn(8, 256, 512, device=DEVICE, dtype=torch.float16)
    b = torch.randn(8, 512, 256, device=DEVICE, dtype=torch.float16)
    _assert_direct_matches_tma(
        _batched_gemm,
        (a, b),
        _config(block_sizes=[1, 128, 64, 64], loop_orders=[[0, 1, 2]]),
        torch.bmm(a, b),
        rtol=1e-2,
        atol=1e-2,
    )


@_sm100_gpu
@pytest.mark.cuda
def test_direct_store_64_row_tile() -> None:
    # bm=64 uses a different TMEM load atom (and so a different per-thread
    # chunk layout); the alignment claim must hold for it too.
    torch.manual_seed(0)
    a = torch.randn(256, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 256, device=DEVICE, dtype=torch.float16)
    _assert_direct_matches_tma(
        _gemm,
        (a, b),
        _config(block_sizes=[64, 16, 128], pid_type="persistent_interleaved"),
        a @ b,
        rtol=1e-2,
        atol=1e-2,
    )


@_sm100_gpu
@pytest.mark.cuda
def test_direct_store_fp8_narrow_tile() -> None:
    torch.manual_seed(0)
    a = torch.randn(256, 256, device=DEVICE, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    b = (
        torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
        .to(torch.float8_e4m3fn)
        .T.contiguous()
        .T
    )
    expected = a.float() @ b.float()
    _assert_direct_matches_tma(
        _gemm,
        (a, b),
        _config(
            block_sizes=[128, 16, 128],
            indexing=["pointer", "pointer", "pointer"],
            pid_type="persistent_interleaved",
        ),
        expected,
        rtol=5e-2,
        atol=5e-2,
    )


@_sm100_gpu
@pytest.mark.cuda
def test_direct_store_bias_row_pre_wait() -> None:
    torch.manual_seed(0)
    a = torch.randn(512, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 512, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(512, 512, device=DEVICE, dtype=torch.float16)
    _assert_direct_matches_tma(
        _bias_gemm,
        (a, b, bias),
        _config(
            pid_type="persistent_interleaved",
            tcgen05_aux_load_placement="pre_acc_wait",
        ),
        (a @ b).float() + bias.float(),
        rtol=1e-2,
        atol=1e-2,
    )


@_sm100_gpu
@pytest.mark.cuda
def test_one_shot_path_from_a_swizzled_config_matches_swizzle_one() -> None:
    torch.manual_seed(0)
    a = torch.randn(512, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 512, device=DEVICE, dtype=torch.float16)
    plain, plain_source = _outputs(
        _gemm,
        (a, b),
        _config(pid_type="persistent_interleaved", tcgen05_l2_swizzle_size=1),
    )
    swizzled, swizzled_source = _outputs(
        _gemm,
        (a, b),
        _config(pid_type="persistent_interleaved", tcgen05_l2_swizzle_size=4),
    )
    assert "while tcgen05_role_local" not in swizzled_source
    assert "swizzle_size" not in swizzled_source
    for out in plain:
        torch.testing.assert_close(out, (a @ b).to(out.dtype), rtol=1e-2, atol=1e-2)
    for out in swizzled:
        assert torch.equal(out, plain[0])


@_sm100_gpu
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("m", "grouping", "folded"),
    [
        (2048, 32, "group_size_m = num_pid_m"),
        (2048, 2, "group_size_m = 2"),
        (768, 4, "group_size_m = min(num_pid_m - first_pid_m, 4)"),
    ],
    ids=["single_group", "full_groups", "partial_group"],
)
def test_folded_l2_grouping_visits_every_tile(
    m: int, grouping: int, folded: str
) -> None:
    torch.manual_seed(0)
    a = torch.randn(m, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 2048, device=DEVICE, dtype=torch.float16)
    grouped, source = _outputs(
        _gemm, (a, b), _config(pid_type="persistent_blocked", l2_groupings=[grouping])
    )
    assert folded in source
    ungrouped, _ = _outputs(
        _gemm, (a, b), _config(pid_type="persistent_blocked", l2_groupings=[1])
    )
    for out in ungrouped:
        torch.testing.assert_close(out, (a @ b).to(out.dtype), rtol=1e-2, atol=1e-2)
    # Same tiles, same per-tile math: the raster order cannot change a value.
    for out in grouped:
        assert torch.equal(out, ungrouped[0])


@_sm100_gpu
@pytest.mark.cuda
def test_direct_store_row_vector_bias_pre_wait() -> None:
    # The matmul s0 winner's shape: a trailing-axis bias row on a 128x16 tile
    # loaded ahead of the accumulator wait (the promoted FP32 row stage) and
    # stored straight from registers.
    torch.manual_seed(0)
    a = torch.randn(256, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 256, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(256, device=DEVICE, dtype=torch.float16)
    _assert_direct_matches_tma(
        _row_bias_gemm,
        (a, b, bias),
        _config(
            block_sizes=[128, 16, 64],
            indexing=["tensor_descriptor"] * 4,
            pid_type="persistent_interleaved",
            tcgen05_aux_load_placement="pre_acc_wait",
        ),
        (a.float() @ b.float() + bias.float()).to(torch.float16),
        rtol=2e-2,
        atol=2e-2,
    )


# --------------------------------------------------------------------------
# N=8 tiles: six role warps, no CTA-wide barrier on the role-local chain
# --------------------------------------------------------------------------

N8_INLINE_CONFIG: dict[str, object] = {
    # The campaign's bmm s0 winner: 64x8 tiles, inline ``xyz`` path, SMEM-staged
    # TMA store.  Its N=8 tile used to launch 64-lane rows (twelve warps for six
    # roles); the spare warps raced the TMA warp through the pipeline-init
    # barrier ahead of warp 0's mbarrier init, a flaky illegal-instruction
    # fault on the second launch of a process.
    "block_sizes": [1, 64, 8, 64],
    "loop_orders": [[0, 1, 2]],
    "l2_groupings": [16],
    "indexing": ["pointer", "tensor_descriptor", "tensor_descriptor"],
    "pid_type": "xyz",
    "tcgen05_persistence_model": "non_persistent",
    "tcgen05_cluster_m": 1,
    "tcgen05_cluster_n": 1,
    "tcgen05_ab_stages": 4,
    "tcgen05_acc_stages": 2,
    "tcgen05_c_stages": 4,
    "tcgen05_tvm_ffi_launch": False,
}


@_sm100_gpu
@pytest.mark.cuda
def test_narrow_n8_inline_store_is_stable_across_streams_and_graph_replay() -> None:
    from helion.autotuner.benchmarking import _make_cudagraph_replay

    torch.manual_seed(0)
    a = torch.randn(4, 64, 128, device=DEVICE, dtype=torch.float16)
    b = torch.randn(4, 128, 128, device=DEVICE, dtype=torch.float16)
    config = helion.Config(**N8_INLINE_CONFIG)
    bound = helion.kernel(_batched_gemm, backend="cute", static_shapes=True).bind(
        (a, b)
    )
    bound.set_config(config)
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    source = bound.to_code(normalized)
    assert "block=(32, 6, 1)" in source
    assert "cute.copy(tcgen05_tma_store_atom" in source
    # The fault needed a launch after the first (the launcher's fast relaunch)
    # on another stream; alternate streams and allocate a fresh output each
    # time, as the autotuner's capture helper does.
    side = torch.cuda.Stream()
    outputs: list[torch.Tensor] = []
    for launch in range(LAUNCHES):
        stream = side if launch % 2 else torch.cuda.default_stream()
        with torch.cuda.stream(stream):
            outputs.append(bound(a, b))
    torch.cuda.synchronize()
    replay = _make_cudagraph_replay(lambda: bound(a, b))
    torch.cuda.synchronize()
    replays: list[torch.Tensor] = []
    for _ in range(5):
        replays.append(replay().clone())
        torch.cuda.synchronize()
    torch.testing.assert_close(outputs[0], torch.bmm(a, b), rtol=1e-2, atol=1e-2)
    for out in [*outputs[1:], *replays]:
        assert torch.equal(out, outputs[0])


def _launch_batched_gemm_in_subprocess(
    tmp_path: Path,
    shape: tuple[int, int, int, int],
    config: dict[str, object],
    *,
    timeout: float,
) -> None:
    """Compile, launch and check a bmm config in a child process under a timeout.

    A deadlocked kernel cannot be interrupted in-process (``synchronize``
    blocks); the child is killed by ``subprocess.run`` instead and the test
    fails with the timeout.  The probe is a file: Helion reads the kernel's
    source through ``inspect``.
    """
    probe = textwrap.dedent(
        f"""
        import json
        import torch
        import helion
        import helion.language as hl

        def bmm(a, b):
            batch, m, k = a.shape
            _, _, n = b.shape
            out = torch.empty((batch, m, n), dtype=a.dtype, device=a.device)
            for tile_b, tile_m, tile_n in hl.tile((batch, m, n)):
                acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.baddbmm(
                        acc, a[tile_b, tile_m, tile_k], b[tile_b, tile_k, tile_n]
                    )
                out[tile_b, tile_m, tile_n] = acc.to(out.dtype)
            return out

        torch.manual_seed(0)
        batch, m, n, k = {shape!r}
        a = torch.randn(batch, m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(batch, k, n, device="cuda", dtype=torch.float16)
        bound = helion.kernel(bmm, backend="cute", static_shapes=True).bind((a, b))
        bound.set_config(helion.Config(**json.loads({json.dumps(config)!r})))
        first = bound(a, b)
        torch.cuda.synchronize()
        torch.testing.assert_close(first, torch.bmm(a, b), rtol=1e-2, atol=1e-2)
        for _ in range(8):
            assert torch.equal(bound(a, b), first)
        torch.cuda.synchronize()
        print("PROBE_OK", helion.__file__)
        """
    )
    probe_path = tmp_path / "n8_probe.py"
    probe_path.write_text(probe)
    # A script's own directory heads ``sys.path``, not the cwd: point the child
    # at the Helion this process imported (the checkout under test), not at an
    # installed one.
    helion_root = str(Path(helion.__file__).resolve().parents[1])
    env = {**os.environ, "HELION_BACKEND": "cute"}
    env["PYTHONPATH"] = os.pathsep.join(
        [helion_root, *filter(None, [env.get("PYTHONPATH")])]
    )
    try:
        result = subprocess.run(
            [sys.executable, str(probe_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as timed_out:
        raise AssertionError(
            f"the launch did not complete within {timeout:.0f}s "
            f"(stdout={timed_out.stdout!r}, stderr={str(timed_out.stderr)[-2000:]!r})"
        ) from None
    assert result.returncode == 0 and "PROBE_OK" in result.stdout, result.stderr[-4000:]


@_sm100_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("ab_stages", [1, 2], ids=["ab1", "ab2"])
def test_narrow_n8_one_shot_chain_completes_with_more_k_tiles_than_stages(
    tmp_path: Path, ab_stages: int
) -> None:
    # 4 x 1 x 16 = 64 tiles of 64x8 <= 148 SMs: the one-shot role-local chain
    # with 2 K tiles over a 1- or 2-stage ring.  A CTA-wide barrier between
    # the hoisted TMA-load role and the MMA role deadlocked the single-stage
    # ring (every ``tcgen05_ab_stages=1`` autotune sample timed out).
    _launch_batched_gemm_in_subprocess(
        tmp_path,
        (4, 64, 128, 128),
        {
            "block_sizes": [1, 64, 8, 64],
            "loop_orders": [[0, 1, 2]],
            "indexing": ["tensor_descriptor"] * 3,
            "pid_type": "persistent_interleaved",
            "tcgen05_persistence_model": "static_persistent",
            "tcgen05_cluster_m": 1,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": ab_stages,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 4,
            "tcgen05_tvm_ffi_launch": False,
        },
        timeout=300.0,
    )


@_sm100_gpu
@pytest.mark.cuda
def test_narrow_n8_one_shot_chain_completes_with_four_k_tiles_over_two_stages(
    tmp_path: Path,
) -> None:
    # 8 x 2 x 16 = 256 tiles would not be one-shot; 2 x 1 x 32 = 64 tiles of
    # 128x8 are, with 4 K tiles over a 2-stage ring.
    _launch_batched_gemm_in_subprocess(
        tmp_path,
        (2, 128, 256, 256),
        {
            "block_sizes": [1, 128, 8, 64],
            "loop_orders": [[0, 1, 2]],
            "indexing": ["tensor_descriptor"] * 3,
            "pid_type": "persistent_interleaved",
            "tcgen05_persistence_model": "static_persistent",
            "tcgen05_cluster_m": 1,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": 2,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 4,
            "tcgen05_tvm_ffi_launch": False,
        },
        timeout=300.0,
    )


_OUTPUT_VIEW_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float8_e4m3fn,
)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8)


def _assert_column_major_tma_store(source: str) -> None:
    assert "cute.copy(tcgen05_tma_store_atom" in source
    assert "cute.nvgpu.CopyR2GOp()" not in source
    assert "'kind': 'tcgen05_d_tma'" in source
    assert "'d_column_major': True" in source
    assert "'rank3_mnl_tensor': True" in source
    assert "tcgen05_c_layout = cutlass.utils.layout.LayoutEnum.COL_MAJOR" in source


def _assert_output_view_stored_exactly(
    case: str, dtype: torch.dtype, *, column_major_tma: bool
) -> None:
    torch.manual_seed(0)
    in_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16
    a = torch.randn(512, 256, device=DEVICE, dtype=in_dtype)
    b = torch.randn(256, 512, device=DEVICE, dtype=in_dtype)
    config = _config(pid_type="persistent_interleaved")
    contiguous = _output_view("contiguous", dtype, device=DEVICE)
    _, tma_source = _outputs(_gemm_into, (a, b, contiguous), config)
    assert "cute.copy(tcgen05_tma_store_atom" in tma_source
    view = _output_view(case, dtype, device=DEVICE)
    outputs, source = _outputs(_gemm_into, (a, b, view), config)
    if column_major_tma:
        _assert_column_major_tma_store(source)
        assert view.stride(0) == 1
        assert view.stride(1) * dtype.itemsize % 16 == 0
    else:
        assert "tcgen05_tma_store_atom" not in source
        assert "'kind': 'tcgen05_d_tma'" not in source
        assert "cute.nvgpu.CopyR2GOp()" in source
    assert all(out is view for out in outputs)
    assert torch.equal(_bits(view), _bits(contiguous))
    # Stores into a padded view must leave the surrounding allocation intact.
    if case in ("row_stride_8B", "column_major_stride_16B", "column_major_stride_8B"):
        assert view._base is not None
        assert torch.count_nonzero(_bits(view._base[:, 512:])).item() == 0
    elif case == "base_8B":
        assert view._base is not None
        pad = 8 // dtype.itemsize
        assert torch.count_nonzero(_bits(view._base[:pad])).item() == 0
        assert torch.count_nonzero(_bits(view._base[-pad:])).item() == 0
    expected = a.float() @ b.float()
    if dtype == torch.float8_e4m3fn:
        tolerance = {"rtol": 1e-1, "atol": 5e-1}
    else:
        tolerance = {"rtol": 2e-2, "atol": 2e-2}
    torch.testing.assert_close(view.float(), expected, **tolerance)


@_sm100_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", _OUTPUT_VIEW_DTYPES, ids=str)
@pytest.mark.parametrize("case", TMA_STORE_ILLEGAL_OUTPUTS)
def test_unproven_output_views_are_stored_exactly(
    case: str, dtype: torch.dtype
) -> None:
    # An 8-byte outer stride or base alignment violates a TensorMap constraint.
    # The SIMT fallback must preserve the exact result and surrounding storage.
    _assert_output_view_stored_exactly(case, dtype, column_major_tma=False)


@_sm100_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", _OUTPUT_VIEW_DTYPES, ids=str)
@pytest.mark.parametrize("case", ["column_major", "column_major_stride_16B"])
def test_column_major_output_views_keep_the_tma_store(
    case: str, dtype: torch.dtype
) -> None:
    # Proven M-contiguous outputs use a column-major D descriptor and SMEM tile.
    # The store must remain bitwise equal to the row-major TMA result.
    _assert_output_view_stored_exactly(case, dtype, column_major_tma=True)


@_sm100_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("factory", sorted(FRESH_FACTORY_GEMMS))
def test_tensor_method_factory_outputs_keep_the_tma_store(factory: str) -> None:
    # An output allocated through ``a.new_*`` or ``torch.empty_strided`` is
    # proven fresh like ``torch.empty``: the TMA store is kept and every
    # launch is bitwise the ``torch.empty`` kernel's result.
    torch.manual_seed(0)
    a = torch.randn(512, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 512, device=DEVICE, dtype=torch.float16)
    config = _config(pid_type="persistent_interleaved")
    reference, reference_source = _outputs(_gemm, (a, b), config)
    assert "cute.copy(tcgen05_tma_store_atom" in reference_source
    outputs, source = _outputs(FRESH_FACTORY_GEMMS[factory], (a, b), config)
    assert "cute.copy(tcgen05_tma_store_atom" in source
    assert "cute.nvgpu.CopyR2GOp()" not in source
    for out in outputs:
        assert torch.equal(out, reference[0])
    torch.testing.assert_close(
        outputs[0].float(), a.float() @ b.float(), rtol=2e-2, atol=2e-2
    )


@_sm100_gpu
@pytest.mark.cuda
def test_fresh_column_major_strided_output_keeps_the_tma_store_on_device() -> None:
    # A fresh M-contiguous output uses column-major TMA without changing the
    # logical result or its strides.
    torch.manual_seed(0)
    a = torch.randn(512, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 512, device=DEVICE, dtype=torch.float16)
    config = _config(pid_type="persistent_interleaved")
    reference, _ = _outputs(_gemm, (a, b), config)
    outputs, source = _outputs(_gemm_empty_strided_column_major, (a, b), config)
    _assert_column_major_tma_store(source)
    for out in outputs:
        assert out.stride() == (1, 512)
        assert torch.equal(out.contiguous(), reference[0])


# Assigned by the test (a CUDA tensor cannot be created at import time); the
# epilogue lambda below reads it as a module-level global.
GPU_GLOBAL_BIAS: torch.Tensor | None = None


@_sm100_gpu
@pytest.mark.cuda
def test_epilogue_lambda_reading_a_module_global_matches_the_closure_form() -> None:
    global GPU_GLOBAL_BIAS
    torch.manual_seed(0)
    a = torch.randn(512, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 512, device=DEVICE, dtype=torch.float16)
    GPU_GLOBAL_BIAS = torch.randn(512, device=DEVICE, dtype=torch.float16)
    config = _config(pid_type="persistent_interleaved")
    closure, closure_source = _outputs(
        _gemm_with_epilogue, (a, b, _closure_bias_epilogue(GPU_GLOBAL_BIAS)), config
    )
    outputs, source = _outputs(
        _gemm_with_epilogue,
        (a, b, lambda acc, tile: acc + GPU_GLOBAL_BIAS[tile[1]]),
        config,
    )
    assert "cute.copy(tcgen05_tma_store_atom" in source
    assert "cute.copy(tcgen05_tma_store_atom" in closure_source
    # The global lives in the kernel's own module.
    assert "_source_module.GPU_GLOBAL_BIAS" in source
    for out in outputs:
        assert torch.equal(out, closure[0])
    expected = a.float() @ b.float() + GPU_GLOBAL_BIAS.float()
    torch.testing.assert_close(outputs[0].float(), expected, rtol=2e-2, atol=2e-2)
