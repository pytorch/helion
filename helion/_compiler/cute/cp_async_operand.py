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
a vector that starts past M is predicated off. The DSL lowers a predicated
copy to ``cp.async``'s source-size operand, so an off vector reads nothing
and zero-fills its stage bytes::

    selp.b32 %r9, 0, 8, %p1;
    cp.async.ca.shared.global [%r8], [%rd3], 8, %r9;

A batched A (``bmm(x.transpose(1, 2), y)``) is copied from the CTA's own
matrix: the base moves by the batch index times the batch stride. Over a K
tail (K not a whole number of k-blocks) the last k-block's rows past K are
predicated off too, so they are zeros the MMA multiplies with the zeros B's
TMA box fills past K; garbage there (another tile's rows, or whatever the
shared memory held before the kernel) could be a NaN, which 0 does not
cancel.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment

# The producer threads: the four epilogue warps.
TCGEN05_CP_ASYNC_A_THREADS = 128
# One stage of the MN-major 16-bit A tile per 128-row subtile: (128 m, bk k).
TCGEN05_CP_ASYNC_A_TILE_M = 128
# The default k-block depth, and the depths the producer copies (a layout's
# shallowest: ``Tcgen05CpAsyncALayout.min_block_k``). A short K loop runs best
# on shallow k-blocks (``_cp_async_a_default``): bf16 768 x 196x64x196 on
# 256-row tiles, GB300, of stock Inductor (Triton, 24.1 us): bk 64 ab 4 0.82,
# bk 32 ab 3 1.036, bk 16 ab 4 1.096.
TCGEN05_CP_ASYNC_A_BLOCK_K = 64
TCGEN05_CP_ASYNC_A_BLOCK_KS = (16, 32, 64)


@dataclasses.dataclass(frozen=True)
class Tcgen05CpAsyncALayout:
    """The static layout of an M-contiguous 16-bit A operand.

    ``m_extent`` and ``k_extent`` are the logical sizes and ``k_stride`` the
    element stride between K rows; row ``k`` starts ``2 * k * k_stride``
    bytes past a 16-byte aligned base. ``batch_stride`` is the element stride
    between the matrices of a batched (rank-3) A, whose every matrix starts
    16-byte aligned too; 0 for a single matrix.
    """

    m_extent: int
    k_extent: int
    k_stride: int
    batch_stride: int = 0

    def class_rows(self, row: int) -> int:
        """K rows of the class starting at row ``row`` (``row < period``):
        rows ``row, row + period, ...`` below ``k_extent``."""
        return -(-(self.k_extent - row) // self.period)

    @property
    def min_block_k(self) -> int:
        """The shallowest k-block (of ``TCGEN05_CP_ASYNC_A_BLOCK_KS``) whose
        class rows fill whole thread rows of every class's copy: a copy of
        ``nbytes``-byte vectors lays its 128 threads out ``(128 // e, e)``
        over (m, k) (``e = nbytes // 2``), so the 16-byte class (row 0,
        always) needs a multiple of 8 rows a k-block, ``8 * period`` K
        rows: 16 at period 2, 32 at period 4 (16 there gives the class 4
        rows for its 8 thread rows)."""
        return next(
            depth
            for depth in TCGEN05_CP_ASYNC_A_BLOCK_KS
            if (depth // self.period) % 8 == 0
        )

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

    A 16-bit (M, K) tensor, or a batch of them (L, M, K), with static sizes
    and strides, M contiguous and an even K stride (every row 4-byte
    aligned), M even (element pairs end at the extent) and K at least the
    shallowest k-block its row classes admit (``Tcgen05CpAsyncALayout.
    min_block_k``: 16 deep, 32 where K rows sit 4 bytes off 16;
    ``cp_async_a_config_ok`` judges a config's depth against K: whole
    k-blocks, or a K tail past the first), on a base
    ``cute_tensor_base_is_aligned`` proves 16-byte aligned; a batch's
    matrices start a 16-byte multiple apart.
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
    if tensor.ndim not in (2, 3) or tensor.dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        return None
    values = (*tensor.size(), *tensor.stride())
    if not all(isinstance(value, int) for value in values):
        return None
    batch_stride = 0
    if tensor.ndim == 3:
        batch_extent, m_extent, k_extent = (int(value) for value in tensor.size())
        batch_stride, m_stride, k_stride = (int(value) for value in tensor.stride())
        if batch_extent == 1:
            batch_stride = 0
        elif batch_stride <= 0 or (2 * batch_stride) % 16:
            # every matrix's rows sit at the residues of the first one's
            return None
    else:
        m_extent, k_extent, m_stride, k_stride = (int(value) for value in values)
    if m_stride != 1 or k_stride < m_extent or k_stride % 2 or m_extent % 2:
        return None
    layout = Tcgen05CpAsyncALayout(
        m_extent=m_extent,
        k_extent=k_extent,
        k_stride=k_stride,
        batch_stride=batch_stride,
    )
    # At least the shallowest k-block the row classes admit, which
    # ``cp_async_a_config_ok`` admits for any K from it on (whole, or a tail
    # past the first): K = 24 over 4084-byte rows (4 mod 16, 32-deep k-blocks
    # at least) has no depth, and is staged.
    if k_extent < layout.min_block_k:
        return None
    return layout


@dataclasses.dataclass(frozen=True)
class Tcgen05CpAsyncAPlan:
    """Generated-code names and layout of the cp.async A producer.

    ``a_tensor`` is the kernel's A argument; ``smem_as`` the MMA's A stage
    tensors (one per 128-row subtile); ``m_offset`` the tile's first M row;
    ``batch_index`` the tile's matrix of a batched A (``None`` for a single
    matrix); ``pipeline`` the AB ``PipelineTmaUmma`` whose empty barriers
    release the stages; ``prefix`` names this plan's locals.
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
    batch_index: str | None = None
    block_k: int = TCGEN05_CP_ASYNC_A_BLOCK_K

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
        barrier fence (``bk`` below is the plan's ``block_k``).

        Class ``c`` views rows ``c, c + P, ...`` of every k-block (``P`` the
        period) of the operand and of the stage: the 16-bit MN-major SW128
        stage of ``make_smem_layout_a`` puts element ``(m, k)`` at
        ``m % 64 + 64 * bk * (m // 64) + 64 * k`` before the swizzle, linear
        in ``k``. One predicated copy per class serves every tile: a second,
        element-pair copy for the edge tile kept its partitions live through
        the K loop and spilled the M-paired kernel's 255 registers (4.6x
        slower). The predicate repeats along K (stride 0); over a K tail each
        issue widens it with the k-block's rows (``issue_lines``).

        A batched A's view starts at the tile's own matrix (bf16
        [768, 196, 196] with strides (38464, 1, 196))::

            tcgen05_cpa_a = cute.make_tensor(
                cute.make_ptr(
                    cutlass.BFloat16,
                    (
                        in0.iterator
                        + cutlass.Int64(cutlass.Int32(indices_0)) * cutlass.Int64(38464)
                    ).toint(),
                    ...,
                ),
                cute.make_layout((196, 196), stride=(1, 196)),
            )
        """
        layout = self.layout
        period = layout.period
        bk = self.block_k
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
                f"{self._matrix_base()}.toint(), cute.AddressSpace.gmem, "
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
                        f"{layout.class_rows(row)}), stride=(1, "
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
                if layout.k_extent % bk:
                    lines.append(
                        f"{name}_pk = cute.make_rmem_tensor(cute.make_layout(("
                        f"1, cute.size({name}_c, mode=[1]), "
                        f"cute.size({name}_c, mode=[2]))), cutlass.Boolean)"
                    )
        return lines

    def _matrix_base(self) -> str:
        """The A tensor's iterator at the tile's matrix: the operand's own for
        a single matrix, moved by the batch index times the batch stride for
        a batched A (in 64 bits: the offset of the last matrix may pass
        2**31 elements)."""
        if self.batch_index is None or not self.layout.batch_stride:
            return f"{self.a_tensor}.iterator"
        return (
            f"({self.a_tensor}.iterator + cutlass.Int64({self.batch_index}) * "
            f"cutlass.Int64({self.layout.batch_stride}))"
        )

    def issue_lines(self, k_block: str) -> list[str]:
        """One k-block's stage: wait for it to empty, copy, arrive on fill.

        Over a K tail every copy runs under the k-block's own predicate: the
        M predicate and, per vector, its class row below the class's K rows,
        so the last k-block zero-fills its rows past K (K = 196 in 64-deep
        k-blocks, k-block 3, odd rows: ``3 * 32 + row < 98``)::

            for _tcgen05_cpa_m in cutlass.range_constexpr(...):
                for _tcgen05_cpa_k in cutlass.range_constexpr(...):
                    tcgen05_cpa_0_1_pk[0, _tcgen05_cpa_m, _tcgen05_cpa_k] = (
                        tcgen05_cpa_0_1_p[0, _tcgen05_cpa_m, 0]
                        and k * 32 + tcgen05_cpa_0_1_c[0, ...][1] < 98
                    )
            cute.copy(
                tcgen05_cpa_0_1_tc,
                tcgen05_cpa_0_1_g[None, None, None, k],
                tcgen05_cpa_0_1_s[...],
                pred=tcgen05_cpa_0_1_pk,
            )

        A whole number of k-blocks keeps the M predicate alone.
        """
        state = self.producer_state
        layout = self.layout
        k_tail = layout.k_extent % self.block_k != 0
        rows = self.block_k // layout.period
        p = self.prefix
        lines = [
            f"{self.pipeline}.sync_object_empty.wait({state}.index, {state}.phase)"
        ]
        for sub in range(len(self.smem_as)):
            for cls, (row, _nbytes) in enumerate(self._classes()):
                name = self._copy_name(sub, cls)
                pred = f"{name}_p"
                if k_tail:
                    pred = f"{name}_pk"
                    lines += [
                        (
                            f"for _{p}_m in cutlass.range_constexpr("
                            f"cute.size({name}_c, mode=[1])):"
                        ),
                        (
                            f"    for _{p}_k in cutlass.range_constexpr("
                            f"cute.size({name}_c, mode=[2])):"
                        ),
                        (
                            f"        {pred}[0, _{p}_m, _{p}_k] = "
                            f"{name}_p[0, _{p}_m, 0] and ({k_block}) * "
                            f"cutlass.Int32({rows}) + "
                            f"{name}_c[0, _{p}_m, _{p}_k][1] < "
                            f"cutlass.Int32({layout.class_rows(row)})"
                        ),
                    ]
                lines.append(
                    f"cute.copy({name}_tc, {name}_g[None, None, None, {k_block}], "
                    f"{name}_s[None, None, None, {state}.index], pred={pred})"
                )
        lines += [
            (
                "cute.arch.cp_async_mbarrier_arrive_noinc("
                f"{self.full_barriers} + {state}.index)"
            ),
            f"{state}.advance()",
        ]
        return lines

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
