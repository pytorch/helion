"""cp.async producers for a tcgen05 A operand no TensorMap can address.

A TensorMap needs a 16-byte base and 16-byte outer strides. A bf16 ``x.T``
whose source rows are 30522 wide has an M-contiguous A with a K stride of
61044 bytes, so the TMA pipeline cannot load it, and the host pass
``stage_aligned_operands`` re-lays it out first: a transposing padded copy of
the whole operand ahead of the GEMM.

This module plans the epilogue warps of the flat one-CTA kernel -- idle until
the accumulator is ready -- as the A producer instead: each k-block's A tile
lands in the MMA's shared-memory stage through ``cp.async`` copies straight
from the operand, while the TMA warp keeps loading B. Rows ``P`` apart (the
period of the stride's byte residue mod 16) start at the same residue, so the
rows split into classes copied at the widest legal width: with a 61044-byte
stride rows ``k % 4 == 0`` start 16-byte aligned (``cp.async.cg``, 16 bytes),
rows ``k % 4 == 2`` 8-byte aligned and the odd rows 4-byte aligned
(``cp.async.ca``). Generated code, per k-block::

    tcgen05_ab_pipeline.sync_object_empty.wait(state.index, state.phase)
    cute.copy(
        tcgen05_cpa_0_0_tc,
        tcgen05_cpa_0_0_g[None, None, None, k],
        tcgen05_cpa_0_0_s[None, None, None, state.index],
        pred=tcgen05_cpa_0_0_p,
    )
    ...
    cute.arch.cp_async_mbarrier_arrive_noinc(tcgen05_cpa_full + state.index)

and the MMA warp waits ``tcgen05_cpa_full`` beside the TMA full barrier. A
vector is copied when its first row is inside M (``Tcgen05CpAsyncAPlan``);
a vector that starts past M is predicated off, and whatever its stage rows
hold only feeds output rows the epilogue never stores.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment

# The producer threads: the four epilogue warps.
TCGEN05_CP_ASYNC_A_THREADS = 128
# One stage of the MN-major 16-bit A tile per 128-row subtile: (128 m, 64 k).
TCGEN05_CP_ASYNC_A_TILE_M = 128
TCGEN05_CP_ASYNC_A_BLOCK_K = 64


@dataclasses.dataclass(frozen=True)
class Tcgen05CpAsyncALayout:
    """The static layout of an M-contiguous 16-bit A operand.

    ``m_extent`` and ``k_extent`` are the logical sizes and ``k_stride`` the
    element stride between K rows; row ``k`` starts ``2 * k * k_stride``
    bytes past a 16-byte aligned base.
    """

    m_extent: int
    k_extent: int
    k_stride: int

    def row_copy_bytes(self, row: int) -> int:
        """Widest ``cp.async`` size (4, 8 or 16 bytes) row ``row`` admits."""
        residue = (2 * row * self.k_stride) % 16
        if residue == 0:
            return 16
        if residue == 8:
            return 8
        return 4

    @property
    def period(self) -> int:
        """Rows after which the row-start residue repeats: 1, 2 or 4 for the
        even K strides the producer takes."""
        for period in (1, 2, 4):
            if (2 * period * self.k_stride) % 16 == 0:
                return period
        raise AssertionError("an even stride repeats within 4 rows")


def tcgen05_cp_async_a_operand_layout(
    env: CompileEnvironment, tensor: torch.Tensor
) -> Tcgen05CpAsyncALayout | None:
    """The layout when ``tensor`` is an A operand the producer serves.

    A rank-2 16-bit (M, K) tensor with static sizes and strides, M contiguous
    and an even K stride (every row 4-byte aligned), K a whole number of
    64-wide k-blocks and M even (element pairs end at the extent), on a base
    ``cute_tensor_base_is_aligned`` proves 16-byte aligned.
    """
    from .memory_ops import cute_tensor_base_is_aligned

    layout = tcgen05_cp_async_a_strided_layout(tensor)
    if layout is None or not cute_tensor_base_is_aligned(env, tensor, 16):
        return None
    return layout


def tcgen05_cp_async_a_strided_layout(
    tensor: torch.Tensor,
) -> Tcgen05CpAsyncALayout | None:
    """``tcgen05_cp_async_a_operand_layout``'s size and stride conditions."""
    if tensor.ndim != 2 or tensor.dtype not in (torch.float16, torch.bfloat16):
        return None
    values = (*tensor.size(), *tensor.stride())
    if not all(isinstance(value, int) for value in values):
        return None
    m_extent, k_extent, m_stride, k_stride = (int(value) for value in values)
    if (
        m_stride != 1
        or k_stride < m_extent
        or k_stride % 2
        or m_extent % 2
        or k_extent % TCGEN05_CP_ASYNC_A_BLOCK_K
    ):
        return None
    return Tcgen05CpAsyncALayout(
        m_extent=m_extent, k_extent=k_extent, k_stride=k_stride
    )


@dataclasses.dataclass(frozen=True)
class Tcgen05CpAsyncAPlan:
    """Generated-code names and layout of the cp.async A producer.

    ``a_tensor`` is the kernel's A argument; ``smem_as`` the MMA's A stage
    tensors (one per 128-row subtile); ``m_offset`` the tile's first M row;
    ``pipeline`` the AB ``PipelineTmaUmma`` whose empty barriers release the
    stages; ``prefix`` names this plan's locals.
    """

    layout: Tcgen05CpAsyncALayout
    a_tensor: str
    smem_as: tuple[str, ...]
    m_offset: str
    pipeline: str
    epi_tidx: str
    warp_idx: str
    lane_idx: str
    ab_stages: int
    prefix: str
    dtype_str: str

    @property
    def full_barriers(self) -> str:
        return f"{self.prefix}_full"

    @property
    def producer_state(self) -> str:
        return f"{self.prefix}_state"

    def _classes(self) -> list[tuple[int, int]]:
        """``(first row, copy bytes)`` of each row class within a k-block.

        A copy of a vector whose first row is inside M runs whole: the rows
        past M land in the stage rows of output rows the epilogue never
        stores, read from the next K row of the operand. Only the operand's
        last K row has no next row, so its class copies element pairs when M
        would cut its vectors (M is even).
        """
        layout = self.layout
        last_row_class = (layout.k_extent - 1) % layout.period
        classes = []
        for row in range(layout.period):
            nbytes = layout.row_copy_bytes(row)
            if row == last_row_class and layout.m_extent % (nbytes // 2):
                nbytes = 4
            classes.append((row, nbytes))
        return classes

    def _copy_name(self, sub: int, cls: int) -> str:
        return f"{self.prefix}_{sub}_{cls}"

    def setup_lines(self) -> list[str]:
        """Barriers, tiled copies, partitions and M predicates, ahead of the
        barrier fence.

        Class ``c`` views rows ``c, c + P, ...`` of every k-block (``P`` the
        period) of the operand and of the stage: the 16-bit MN-major SW128
        stage of ``make_smem_layout_a`` puts element ``(m, k)`` at
        ``m % 64 + 64 * bk * (m // 64) + 64 * k`` before the swizzle, linear
        in ``k``. One predicated copy per class serves every tile: a second,
        element-pair copy for the edge tile kept its partitions live through
        the K loop and spilled the M-paired kernel's 255 registers (4.6x
        slower). The predicate repeats along K (stride 0).
        """
        layout = self.layout
        period = layout.period
        bk = TCGEN05_CP_ASYNC_A_BLOCK_K
        tile_m = TCGEN05_CP_ASYNC_A_TILE_M
        rows = bk // period
        p = self.prefix
        lines = [
            (
                f"{self.full_barriers} = cute.arch.alloc_smem(cutlass.Int64, "
                f"cutlass.Int32({self.ab_stages}))"
            ),
            (
                f"if {self.warp_idx} == cutlass.Int32(0) and "
                f"{self.lane_idx} == cutlass.Int32(0):"
            ),
            f"    for _{p}_s in cutlass.range_constexpr({self.ab_stages}):",
            (
                f"        cute.arch.mbarrier_init({self.full_barriers} + _{p}_s, "
                f"cutlass.Int32({TCGEN05_CP_ASYNC_A_THREADS}))"
            ),
            (
                f"{p}_a = cute.make_tensor(cute.make_ptr({self.dtype_str}, "
                f"{self.a_tensor}.iterator.toint(), cute.AddressSpace.gmem, "
                "assumed_align=16), cute.make_layout(("
                f"{layout.m_extent}, {layout.k_extent}), stride=(1, "
                f"{layout.k_stride})))"
            ),
            (
                f"{self.producer_state} = cutlass.pipeline.make_pipeline_state("
                f"cutlass.pipeline.PipelineUserType.Producer, {self.ab_stages})"
            ),
        ]
        for sub, smem in enumerate(self.smem_as):
            tile_index = (
                f"{self.m_offset} // cutlass.Int32({tile_m}) + cutlass.Int32({sub})"
            )
            for cls, (row, nbytes) in enumerate(self._classes()):
                elems = nbytes // 2
                threads_m = tile_m // elems
                threads_k = TCGEN05_CP_ASYNC_A_THREADS // threads_m
                assert rows % threads_k == 0, (rows, threads_k)
                name = self._copy_name(sub, cls)
                cache = (
                    "cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL"
                    if nbytes == 16
                    else ""
                )
                lines += [
                    (
                        f"{name}_tc = cute.make_tiled_copy_tv(cute.make_copy_atom("
                        f"cute.nvgpu.cpasync.CopyG2SOp({cache}), {self.dtype_str}, "
                        f"num_bits_per_copy={nbytes * 8}), cute.make_layout(("
                        f"{threads_m}, {threads_k}), stride=(1, {threads_m})), "
                        f"cute.make_layout(({elems}, 1)))"
                    ),
                    f"{name}_th = {name}_tc.get_slice({self.epi_tidx})",
                    (
                        f"{name}_g = {name}_th.partition_S(cute.local_tile("
                        f"cute.make_tensor({p}_a.iterator + {row * layout.k_stride}, "
                        f"cute.make_layout(({layout.m_extent}, "
                        f"{layout.k_extent // period}), stride=(1, "
                        f"{period * layout.k_stride}))), ({tile_m}, {rows}), "
                        f"({tile_index}, None)))"
                    ),
                    (
                        f"{name}_s = {name}_th.partition_D(cute.make_tensor("
                        f"{smem}.iterator + {64 * row}, cute.make_layout(((64, 2), "
                        f"{rows}, {self.ab_stages}), stride=((1, {64 * bk}), "
                        f"{64 * period}, {tile_m * bk}))))"
                    ),
                    (
                        f"{name}_c = {name}_th.partition_S("
                        f"cute.make_identity_tensor(({tile_m}, {rows})))"
                    ),
                    (
                        f"{name}_p = cute.make_rmem_tensor(cute.make_layout(("
                        f"1, cute.size({name}_c, mode=[1]), "
                        f"cute.size({name}_c, mode=[2])), stride=(0, 1, 0)), "
                        "cutlass.Boolean)"
                    ),
                    (
                        f"for _{p}_m in cutlass.range_constexpr("
                        f"cute.size({name}_c, mode=[1])):"
                    ),
                    (
                        f"    {name}_p[0, _{p}_m, 0] = {self.m_offset} + "
                        f"cutlass.Int32({sub * tile_m}) + "
                        f"{name}_c[0, _{p}_m, 0][0] < "
                        f"cutlass.Int32({layout.m_extent})"
                    ),
                ]
        return lines

    def issue_lines(self, k_block: str) -> list[str]:
        """One k-block's stage: wait for it to empty, copy, arrive on fill."""
        state = self.producer_state
        return [
            f"{self.pipeline}.sync_object_empty.wait({state}.index, {state}.phase)",
            *(
                (
                    f"cute.copy({name}_tc, {name}_g[None, None, None, {k_block}], "
                    f"{name}_s[None, None, None, {state}.index], pred={name}_p)"
                )
                for sub in range(len(self.smem_as))
                for cls in range(len(self._classes()))
                for name in (self._copy_name(sub, cls),)
            ),
            (
                "cute.arch.cp_async_mbarrier_arrive_noinc("
                f"{self.full_barriers} + {state}.index)"
            ),
            f"{state}.advance()",
        ]

    def consumer_wait_line(self, consumer_state: str) -> str:
        """The MMA warp's wait for a stage's copies, then the proxy fence:
        ``cp.async`` writes the stage through the generic proxy and the UMMA
        reads it through the async proxy, and the arrive fires when the copies
        land, so only the waiting warp can order the two::

            cute.arch.mbarrier_wait(tcgen05_cpa_full + state.index, state.phase)
            cute.arch.fence_view_async_shared()
        """
        return (
            f"cute.arch.mbarrier_wait({self.full_barriers} + "
            f"{consumer_state}.index, {consumer_state}.phase)\n"
            "cute.arch.fence_view_async_shared()"
        )
