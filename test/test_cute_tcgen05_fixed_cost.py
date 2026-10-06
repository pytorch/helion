"""Codegen pins for the fixed-cost trims of the plain tcgen05 GEMM launch.

CPU-only (fake CUDA target), like ``test_cute_tcgen05_fixed_chain.py`` whose
binders and fixture this file reuses: every test binds a kernel, pins a
config and inspects the generated CuTe source.  Runtime coverage lives in
``test_cute_tcgen05_fixed_cost_gpu.py``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
import torch

from test.test_cute_tcgen05_fixed_chain import CPU_DEVICE
from test.test_cute_tcgen05_fixed_chain import PUBLICATION_HOISTED
from test.test_cute_tcgen05_fixed_chain import _bind_bias_gemm
from test.test_cute_tcgen05_fixed_chain import _bind_bmm
from test.test_cute_tcgen05_fixed_chain import _bind_gemm
from test.test_cute_tcgen05_fixed_chain import _cpu_only  # noqa: F401
from test.test_cute_tcgen05_fixed_chain import _kernel_body
from test.test_cute_tcgen05_fixed_chain import _plain_config
from test.test_cute_tcgen05_fixed_chain import _source

import helion
from helion._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_DIRECT
from helion._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_NORMAL
from helion._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODES
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_NORMAL
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUTS
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

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
        search = tcgen05.optional_fragments(for_search=True)
        validation = tcgen05.optional_fragments(for_search=False)
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
        seeds = bound.config_spec._cute_tcgen05_config._small_grid_seed_configs()
    tiles = sorted({tuple(seed.config["block_sizes"]) for seed in seeds})
    assert tiles == [(1, 64, 16, 128), (1, 64, 32, 128), (1, 64, 64, 128)], tiles
    for seed in seeds:
        assert seed.config["pid_type"] == "persistent_interleaved"
        # K = 128 in one 128-wide step: the ring is never deeper than the loop.
        assert seed.config["tcgen05_ab_stages"] == 1


def test_small_grid_seeds_stop_once_the_batched_grid_fills_the_machine() -> None:
    # 64 x 128x128x256: 64 standard tiles would qualify unbatched, but the
    # batch axis fills the machine 64-fold.
    bound = _bind_bmm((64, 256, 256, 256), torch.float16)
    with bound.env:
        seeds = bound.config_spec._cute_tcgen05_config._small_grid_seed_configs()
    assert seeds == []


def test_small_grid_seeds_prefetch_the_bias_row() -> None:
    # 256^3 + bias: every small-grid seed loads its bias row ahead of the
    # accumulator wait (the promoted row stage), like the one-wave seeds.
    bound = _bind_row_bias_gemm((256, 256, 256), torch.float16)
    with bound.env:
        seeds = bound.config_spec._cute_tcgen05_config._small_grid_seed_configs()
    assert seeds
    for seed in seeds:
        assert seed.config["tcgen05_aux_load_placement"] == "pre_acc_wait", seed


def test_small_grid_seeds_twin_narrow_tiles_with_the_direct_store() -> None:
    bound = _bind_gemm((256, 256, 256), torch.float16)
    with bound.env:
        seeds = bound.config_spec._cute_tcgen05_config._small_grid_seed_configs()
    by_mode: dict[str, list[tuple[int, int]]] = {}
    for seed in seeds:
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
        fragments = bound.config_spec._cute_tcgen05_config._matmul_block_fragments()
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
        seeds = bound.config_spec._cute_tcgen05_config._small_grid_seed_configs()
    tiles = sorted({tuple(seed.config["block_sizes"]) for seed in seeds})
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


# Outputs the TMA store may address (a TensorMap needs a 16-byte base and
# 16-byte outer strides over an N-contiguous matrix) ...
TMA_STORE_LEGAL_OUTPUTS = ("contiguous", "row_stride_16B")
# ... and views that violate exactly one of those constraints.
TMA_STORE_ILLEGAL_OUTPUTS = ("row_stride_8B", "column_major", "base_8B")


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
    assert "'kind': 'tcgen05_d_tma'" in source


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
    # The SIMT body addresses every element itself (no alignment claim on
    # an unproven output) ...
    assert "cute.nvgpu.CopyR2GOp()" in kernel
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


def test_fresh_column_major_strided_output_takes_the_simt_store() -> None:
    # ... freshness says nothing about the layout: an N-major
    # ``torch.empty_strided`` result is not what the staged row-major D tile
    # addresses, so it keeps the exact SIMT store.
    source = _source(
        _bind_fresh_factory_gemm(
            _gemm_empty_strided_column_major, (512, 512, 256), torch.float16
        ),
        _plain_config(pid_type="persistent_interleaved"),
    )
    kernel = _kernel_body(source)
    assert "tcgen05_tma_store_atom" not in kernel
    assert "'kind': 'tcgen05_d_tma'" not in source
    assert "cute.nvgpu.CopyR2GOp()" in kernel


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
