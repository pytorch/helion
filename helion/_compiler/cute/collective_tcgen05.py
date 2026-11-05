"""Native SM100 compute for independently proven collective scalar recipes.

The caller retains operand generation, masks, row traversal, and the scalar
epilogue. This module owns shared operand layouts and a single-CTA
TCGEN05/TMEM lifetime. A guarded two-stage path overlaps operand production,
but every dependent MMA and TMEM publication still waits for completion.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ..ast_read_writes import ast_rename
from .collective_tmem_operand import CollectiveTmemOperand
from .scalar_recipe import _clone


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
    operand_stages: int = 1

    def b_stage_stride_aligned(self) -> bool:
        # K_SW/MN_SW atoms span eight swizzle-width byte vectors. The base
        # is 1024-byte aligned; each slot must keep those same swizzle bits.
        element_bytes = 4 if self.dtype == "cutlass.TFloat32" else 2
        swizzle_bytes = min(
            128, element_bytes * (self.bk if self.b_k_major else self.bn)
        )
        atom_bytes = 8 * swizzle_bytes
        slot_bytes = self.bn * self.bk * element_bytes
        return 1024 % atom_bytes == 0 and slot_bytes % atom_bytes == 0

    def pipeline_stages(
        self,
        requested: int,
        start: ast.expr,
        stop: ast.expr,
        *,
        static_shapes: bool,
        unmasked_recipes: bool,
        independent_recipes: bool,
        single_site: bool,
        no_other_smem: bool,
        capacity_bytes: int,
    ) -> int:
        """Admit one register-produced A ahead of the live TMEM A.

        B has two disjoint shared slots. The existing A register fragment is
        the producer slot; the current MMA reads only the published TMEM slot.
        Wider rings, masked ranges and unknown arenas retain stage one.
        """

        def literal(value: ast.expr) -> int | None:
            if (
                isinstance(value, ast.Call)
                and ast.unparse(value.func) == "cutlass.Int32"
                and len(value.args) == 1
                and not value.keywords
            ):
                value = value.args[0]
            if isinstance(value, ast.Constant) and type(value.value) is int:
                return value.value
            return None

        first, last = literal(start), literal(stop)
        if not (
            type(requested) is int
            and requested == 2
            and self.a_in_tmem
            and self.dtype in ("cutlass.Float16", "cutlass.BFloat16")
            and self.b_stage_stride_aligned()
            and static_shapes
            and unmasked_recipes
            and independent_recipes
            and single_site
            and no_other_smem
            and first is not None
            and last is not None
            and 0 <= first < last < (1 << 31)
            and last - first >= 2 * self.bk
            and (last - first) % self.bk == 0
        ):
            return 1
        # Three allocator-alignment gaps plus the TMEM/barrier runtime arena.
        required = 2 * self.bn * self.bk * 2 + self.bm * self.bn * 4 + 4096
        return 2 if required <= capacity_bytes else 1

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
        if (
            type(self.operand_stages) is not int
            or self.operand_stages not in (1, 2)
            or (
                self.operand_stages == 2
                and (not self.a_in_tmem or self.dtype == "cutlass.TFloat32")
            )
        ):
            raise ValueError("two-stage native operands require FP16/BF16 TMEM A")
        if self.operand_stages == 2 and not self.b_stage_stride_aligned():
            raise ValueError("native B stage stride must preserve the swizzle atom")
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
        b_elements = bn * bk * self.operand_stages
        operand_pointers = (
            self.operands.pointers(p, dtype, a_elements, b_elements)
            if self.operands is not None
            else (
                f"{p}_ap = cute.arch.alloc_smem({dtype}, {a_elements}, alignment=1024)\n"
                if a_elements
                else ""
            )
            + f"{p}_bp = cute.arch.alloc_smem({dtype}, {b_elements}, alignment=1024)\n"
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
        return [*self.issue_mma(k_offset, k_start), *self.wait_mma()]

    def issue_mma(self, k_offset: ast.expr, k_start: ast.expr) -> list[ast.stmt]:
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
""")

    def wait_mma(self) -> list[ast.stmt]:
        r = self.resource.prefix
        return _statements(f"""
cute.arch.mbarrier_wait({r}_done, {r}_phase)
{r}_phase = {r}_phase ^ 1
cute.arch.sync_threads()
""")

    def bind_b_stage(
        self, stage: str, view: str, *, fragment: str | None = None
    ) -> list[ast.stmt]:
        p = self.prefix
        result = _statements(f"""
{view} = cute.make_tensor(cute.recast_ptr({p}_bp + ({stage}) * {self.bn * self.bk}, {p}_bl.inner, dtype={self.dtype}), {p}_bl.outer)
""")
        if fragment is not None:
            result.extend(
                _statements(
                    f"{fragment} = {p}_thr.make_fragment_B({p}_thr.partition_B({view}))"
                )
            )
        return result

    def pipeline_loop(
        self,
        loop: ast.For,
        a_copy: ast.stmt,
        a_publish: list[ast.stmt],
        b_copy: ast.stmt,
        start: ast.expr,
        stop: ast.expr,
        *,
        async_copies: bool,
    ) -> list[ast.stmt]:
        """Keep current A in TMEM while producing next A in registers.

        Every MMA completes before the next TMEM publication or accumulator
        update. The alternate B slot is never the slot an outstanding MMA
        reads. The original recipes and typed index arithmetic are unchanged.
        """
        if self.operand_stages != 2 or not self.a_in_tmem:
            raise ValueError("native operand overlap requires a two-stage TMEM-A plan")
        if not isinstance(loop.target, ast.Name):
            raise ValueError("native operand overlap requires one induction variable")
        p = self.prefix
        offset = loop.target.id
        recipe_k, next_k, stage = f"{p}_recipe_k", f"{p}_next_k", f"{p}_b_stage"
        prefill_b, current_b, prefetch_b = (
            f"{p}_b_prefill",
            f"{p}_b_current",
            f"{p}_b_prefetch",
        )
        current_rb = f"{p}_rb_current"

        def recipe(statement: ast.stmt, b_view: str | None = None) -> ast.stmt:
            renames = {offset: recipe_k}
            if b_view is not None:
                renames[f"{p}_b"] = b_view
            return ast_rename(_clone(statement), renames)

        commit = (
            _statements("cute.arch.cp_async_commit_group()") if async_copies else []
        )
        wait = (
            _statements("cute.arch.cp_async_wait_group(0)\ncute.arch.sync_threads()")
            if async_copies
            else []
        )
        prefill = [
            *_statements(f"{recipe_k} = cutlass.Int32({ast.unparse(start)})"),
            *self.bind_b_stage("0", prefill_b),
            recipe(a_copy),
            *(_clone(statement) for statement in a_publish),
            recipe(b_copy, prefill_b),
            *(_clone(statement) for statement in commit),
            *(_clone(statement) for statement in wait),
        ]
        prefetch = ast.If(
            test=ast.parse(
                f"{next_k} < cutlass.Int64({ast.unparse(stop)})", mode="eval"
            ).body,
            body=[
                *_statements(f"{recipe_k} = cutlass.Int32({next_k})"),
                recipe(a_copy),
                *self.bind_b_stage(f"{stage} ^ 1", prefetch_b),
                recipe(b_copy, prefetch_b),
                *(_clone(statement) for statement in commit),
            ],
            orelse=[],
        )
        publish = ast.If(
            test=_clone(prefetch.test),
            body=[_clone(statement) for statement in a_publish],
            orelse=[],
        )
        reduction = _clone(loop)
        reduction.body = [
            *_statements(
                f"{stage} = cutlass.Int32(((cutlass.Int64({offset}) - "
                f"cutlass.Int64({ast.unparse(start)})) // {self.bk}) % 2)"
            ),
            *self.bind_b_stage(stage, current_b, fragment=current_rb),
            *ast_rename(
                ast.Module(body=self.issue_mma(loop.target, start), type_ignores=[]),
                {f"{p}_rb": current_rb},
            ).body,
            *_statements(f"{next_k} = cutlass.Int64({offset}) + {self.bk}"),
            prefetch,
            *self.wait_mma(),
            *wait,
            publish,
        ]
        return ast.fix_missing_locations(
            ast.Module(body=[*prefill, reduction], type_ignores=[])
        ).body

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
