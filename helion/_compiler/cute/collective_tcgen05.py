"""Native SM100 compute for independently proven collective scalar recipes.

The caller retains operand generation, masks, row traversal, and the scalar
epilogue. This module owns only shared operand layouts and a synchronous
single-CTA TCGEN05/TMEM lifetime. It does not require a unique scatter index.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from .collective_tmem_operand import CollectiveTmemOperand


def _statements(source: str) -> list[ast.stmt]:
    return ast.parse(source).body


@dataclass
class CollectiveOperandSmem:
    """Reuse dead A/B storage across synchronous collective contractions.

    Each collective finishes all asynchronous copies and MMA reads, then
    reaches a full-CTA barrier before returning to its surrounding code.
    Operand views are compiler-owned and cannot escape their collective.
    Accumulator storage has separate lifetimes and is never pooled here.
    """

    prefix: str
    a_bytes: int = 0
    b_bytes: int = 0

    def pointers(
        self, prefix: str, dtype: str, a_elements: int, b_elements: int
    ) -> str:
        element_bytes = 4 if dtype == "cutlass.TFloat32" else 2
        self.a_bytes = max(self.a_bytes, a_elements * element_bytes)
        self.b_bytes = max(self.b_bytes, b_elements * element_bytes)
        a_pointer = (
            f"{prefix}_ap = cute.recast_ptr({self.prefix}_ap, dtype={dtype})\n"
            if a_elements
            else ""
        )
        return (
            a_pointer
            + f"{prefix}_bp = cute.recast_ptr({self.prefix}_bp, dtype={dtype})\n"
        )

    def prologue(self) -> list[ast.stmt]:
        return _statements(
            (
                f"{self.prefix}_ap = cute.arch.alloc_smem(cutlass.Uint8, {self.a_bytes}, alignment=1024)\n"
                if self.a_bytes
                else ""
            )
            + f"{self.prefix}_bp = cute.arch.alloc_smem(cutlass.Uint8, {self.b_bytes}, alignment=1024)\n"
        )


@dataclass
class CollectiveTmemResource:
    """One allocation shared by sequential collective sites in one CTA.

    Allocation and deallocation enclose the whole device body, including any
    serial row loops. Release the allocation permit immediately after the single
    allocation so other CTAs can allocate while this CTA computes. Each tile
    finishes its TMEM reads and reaches a full-CTA barrier before another tile
    can reuse this allocation; no later tile performs another allocation.
    """

    prefix: str
    columns: int
    operand_columns: int = 0

    def reserve_operand(self, columns: int) -> None:
        # Accumulator columns are fixed from every native site before lowering.
        # Late recipe admission may reserve an additional, disjoint A range.
        # All sites finish before reusing that range, so only its maximum is
        # needed. The allocation prologue is emitted after every site is lowered.
        self.operand_columns = max(self.operand_columns, columns)

    @property
    def allocation_columns(self) -> int:
        return 1 << (self.columns + self.operand_columns - 1).bit_length()

    @property
    def pointer(self) -> str:
        return f"{self.prefix}_ptr"

    def prologue(self) -> list[ast.stmt]:
        p = self.prefix
        return _statements(f"""
{p}_holding = cute.arch.alloc_smem(cutlass.Int32, 1)
{p}_barrier = cutlass.pipeline.NamedBarrier(barrier_id=1, num_threads=128)
{p}_allocator = cutlass.utils.TmemAllocator({p}_holding, barrier_for_retrieve={p}_barrier, allocator_warp_id=0)
{p}_allocator.allocate({self.allocation_columns})
{p}_allocator.relinquish_alloc_permit()
{p}_allocator.wait_for_alloc()
{self.pointer} = {p}_allocator.retrieve_ptr(cutlass.Float32)
{p}_done = cute.arch.alloc_smem(cutlass.Int64, 1)
if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == 0:
    with cute.arch.elect_one():
        cute.arch.mbarrier_init({p}_done, 1)
cute.arch.mbarrier_init_fence()
cute.arch.sync_threads()
{p}_phase = cutlass.Int32(0)
""")

    def epilogue(self) -> list[ast.stmt]:
        p = self.prefix
        return _statements(f"""
cute.arch.fence_view_async_tmem_load()
cute.arch.sync_threads()
{p}_allocator.free({self.pointer})
""")


@dataclass(frozen=True)
class CollectiveTcgen05Plan:
    prefix: str
    tid: str
    bm: int
    bn: int
    bk: int
    dtype: str
    resource: CollectiveTmemResource
    b_k_major: bool = False
    zero_seed: bool = True
    operands: CollectiveOperandSmem | None = None
    a_in_tmem: bool = False

    @property
    def tmem_a(self) -> CollectiveTmemOperand | None:
        return (
            CollectiveTmemOperand(
                self.prefix,
                self.tid,
                self.bm,
                self.bk,
                self.dtype,
                self.resource.columns,
            )
            if self.a_in_tmem
            else None
        )

    @property
    def c_swizzle(self) -> tuple[int, int, int]:
        # Native TMEM copies assign consecutive rows to lanes. A 16-byte
        # shared store from each of eight such lanes should use eight disjoint
        # bank groups. XOR the row's low three bits into the 4-float column
        # group, keeping both vector contiguity and scalar N-lane reads intact.
        return (3, 2, self.bn.bit_length() - 3)

    def __post_init__(self) -> None:
        # These are the surrounding collective's existing tile bounds, narrowed
        # to the native instruction's M choices. N remains fully threaded.
        if self.bm not in (64, 128) or self.bn not in (32, 64):
            raise ValueError("native collective requires M=64/128 and N=32/64")
        if self.bk not in (16, 32, 64, 128):
            raise ValueError("native collective requires a supported k16 tile")
        if self.dtype not in (
            "cutlass.Float16",
            "cutlass.BFloat16",
            "cutlass.TFloat32",
        ):
            raise ValueError("native collective requires FP16, BF16 or TF32 operands")
        if self.resource.columns < self.bn:
            raise ValueError("TMEM allocation is smaller than the accumulator")
        if (operand := self.tmem_a) is not None:
            self.resource.reserve_operand(operand.columns)

    def setup(self, thread_index: str) -> list[ast.stmt]:
        p, bm, bn, bk, dtype = (
            self.prefix,
            self.bm,
            self.bn,
            self.bk,
            self.dtype,
        )
        element_bytes = 4 if dtype == "cutlass.TFloat32" else 2
        a_swizzle = min(128, element_bytes * bk)
        b_swizzle = min(128, element_bytes * (bk if self.b_k_major else bn))
        b_major = "K" if self.b_k_major else "MN"
        b_swizzle_suffix = (
            "_32B" if dtype == "cutlass.TFloat32" and not self.b_k_major else ""
        )
        b_order = (0, 1) if self.b_k_major else (1, 0)
        c_bits, c_base, c_shift = self.c_swizzle
        tmem_a = self.tmem_a
        a_elements = 0 if tmem_a is not None else bm * bk
        operand_pointers = (
            self.operands.pointers(p, dtype, a_elements, bn * bk)
            if self.operands is not None
            else (
                f"{p}_ap = cute.arch.alloc_smem({dtype}, {a_elements}, alignment=1024)\n"
                if a_elements
                else ""
            )
            + f"{p}_bp = cute.arch.alloc_smem({dtype}, {bn * bk}, alignment=1024)\n"
        )
        a_setup = (
            f"""
{p}_al = cute.tile_to_shape(cute.nvgpu.tcgen05.make_smem_layout_atom(cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW{a_swizzle}, {dtype}), ({bm}, {bk}), order=(0, 1))
{p}_a = cute.make_tensor(cute.recast_ptr({p}_ap, {p}_al.inner, dtype={dtype}), {p}_al.outer)
"""
            if tmem_a is None
            else ""
        )
        setup = _statements(f"""
{self.tid} = {thread_index}
{operand_pointers}
{p}_cp = cute.arch.alloc_smem(cutlass.Float32, {bm * bn}, alignment=128)
{a_setup}
{p}_bl = cute.tile_to_shape(cute.nvgpu.tcgen05.make_smem_layout_atom(cute.nvgpu.tcgen05.SmemLayoutAtomKind.{b_major}_SW{b_swizzle}{b_swizzle_suffix}, {dtype}), ({bn}, {bk}), order={b_order})
{p}_b = cute.make_tensor(cute.recast_ptr({p}_bp, {p}_bl.inner, dtype={dtype}), {p}_bl.outer)
{p}_c = cute.make_tensor({p}_cp, cute.make_composed_layout(cute.make_swizzle({c_bits}, {c_base}, {c_shift}), 0, cute.make_layout(({bm}, {bn}), stride=({bn}, 1))))
{p}_mma = cutlass.utils.blackwell_helpers.make_trivial_tiled_mma({dtype}, {dtype}, cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.{b_major}, cutlass.Float32, cute.nvgpu.tcgen05.CtaGroup.ONE, ({bm}, {bn}), cute.nvgpu.tcgen05.OperandSource.{"TMEM" if tmem_a is not None else "SMEM"})
{p}_thr = {p}_mma.get_slice(0)
""")
        setup.extend(
            tmem_a.setup(self.resource.pointer)
            if tmem_a is not None
            else _statements(
                f"{p}_ra = {p}_thr.make_fragment_A({p}_thr.partition_A({p}_a))"
            )
        )
        setup.extend(
            _statements(f"""
{p}_rb = {p}_thr.make_fragment_B({p}_thr.partition_B({p}_b))
{p}_acc_base = {p}_mma.make_fragment_C({p}_mma.partition_shape_C(({bm}, {bn})))
{p}_acc = cute.make_tensor({self.resource.pointer}, {p}_acc_base.layout)
""")
        )
        return setup

    def seed_from_shared(self) -> list[ast.stmt]:
        """Upload the exact FP32 seed before the first native accumulation.

        The store operations are the inverses of the FP32 row-major drain's
        16x128b (M64) and 32x32b (M128) operations. Partitioning the same logical
        C tensor keeps the upload independent of physical TMEM row strides.
        The caller synchronizes the shared seed stores before this upload.
        """
        p = self.prefix
        operation, repetitions = (
            ("St16x128bOp", self.bn // 4) if self.bm == 64 else ("St32x32bOp", self.bn)
        )
        return _statements(f"""
{p}_seed_mn = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout({p}_acc)
{p}_seed_atom = cute.make_copy_atom(cute.nvgpu.tcgen05.{operation}(cute.nvgpu.tcgen05.Repetition({repetitions})), cutlass.Float32)
{p}_seed_copy = cute.nvgpu.tcgen05.make_tmem_copy({p}_seed_atom, {p}_seed_mn)
{p}_seed_thread = {p}_seed_copy.get_slice({self.tid})
{p}_seed_source = {p}_seed_thread.partition_S({p}_c)
{p}_seed_destination = {p}_seed_thread.partition_D({p}_seed_mn)
{p}_seed_registers = cute.make_rmem_tensor({p}_seed_source.shape, cutlass.Float32)
cute.autovec_copy({p}_seed_source, {p}_seed_registers)
cute.copy({p}_seed_copy, {p}_seed_registers, {p}_seed_destination)
cute.arch.fence_view_async_tmem_store()
cute.arch.sync_threads()
""")

    def compute(self, k_offset: ast.expr, k_start: ast.expr) -> list[ast.stmt]:
        p, r = self.prefix, self.resource.prefix
        # All ordinary shared stores happen-before the async proxy fence and
        # CTA barrier. The commit covers every MMA read of this A/B stage;
        # producers cannot reuse it until the parity wait completes.
        accumulate = (
            "True"
            if not self.zero_seed
            else f"({ast.unparse(k_offset)}) != ({ast.unparse(k_start)}) or {p}_ki != 0"
        )
        return _statements(f"""
cute.arch.fence_proxy(kind="async.shared", space="cta")
cute.arch.sync_threads()
if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == 0:
    for {p}_ki in cutlass.range({self.bk // (8 if self.dtype == "cutlass.TFloat32" else 16)}, unroll=1):
        {p}_mma.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, {accumulate})
        cute.gemm({p}_mma, {p}_acc, {p}_ra[None, None, {p}_ki], {p}_rb[None, None, {p}_ki], {p}_acc)
    with cute.arch.elect_one():
        cute.nvgpu.tcgen05.commit({r}_done)
cute.arch.mbarrier_wait({r}_done, {r}_phase)
{r}_phase = {r}_phase ^ 1
cute.arch.sync_threads()
""")

    def finish(self, k_start: ast.expr, k_stop: ast.expr) -> list[ast.stmt]:
        p = self.prefix
        # Transform the native fragment's ((M,N),1,1) layout with CUTLASS's
        # public helper; do not assume a physical TMEM row or column stride.
        # An empty reduction reads TMEM only when the seed was uploaded.
        initialized = (
            "True"
            if not self.zero_seed
            else f"({ast.unparse(k_start)}) < ({ast.unparse(k_stop)})"
        )
        return _statements(f"""
{p}_acc_mn = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout({p}_acc)
{p}_tmem_copy_atom = cutlass.utils.blackwell_helpers.get_tmem_load_op(({self.bm}, {self.bn}, {self.bk}), cutlass.utils.layout.LayoutEnum.ROW_MAJOR, cutlass.Float32, cutlass.Float32, ({self.bm}, {self.bn}), False)
{p}_tmem_copy = cute.nvgpu.tcgen05.make_tmem_copy({p}_tmem_copy_atom, {p}_acc_mn)
{p}_tmem_thread = {p}_tmem_copy.get_slice({self.tid})
{p}_tmem_source = {p}_tmem_thread.partition_S({p}_acc_mn)
{p}_tmem_destination = {p}_tmem_thread.partition_D({p}_c)
{p}_tmem_registers = cute.make_rmem_tensor({p}_tmem_destination.shape, cutlass.Float32)
{p}_tmem_registers.fill(0.0)
if {initialized}:
    cute.copy({p}_tmem_copy, {p}_tmem_source, {p}_tmem_registers)
cute.autovec_copy({p}_tmem_registers, {p}_tmem_destination)
cute.arch.fence_view_async_tmem_load()
cute.arch.sync_threads()
""")
