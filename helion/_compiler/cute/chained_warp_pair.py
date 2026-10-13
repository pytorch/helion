"""Adjacent same-lane native C to A transport with explicit team ownership.

Planning grants a layout, never readiness. Original emission retains both C
publications and joins; completed pair receipts can later authorize a grouped
schedule and a separate full-body read cut for the first shared result.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from functools import lru_cache
import importlib
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import cast

import torch

from .chained_execution import ChainedExecution
from .chained_fragment_epilogue import FragmentExpression
from .chained_fragment_epilogue import bind_fragment_expression
from .chained_warp_stage import _execution
from .chained_warp_stage import _snapshot
from .chained_warp_stage import warp_revision
from .physical_use_frontier import PhysicalUseFrontier

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_matmul import _ScanInput
    from .chained_root_warp_stage import WarpOperandLayout


def _reject(message: str) -> NoReturn:
    from .chained_matmul import _UnsupportedChain

    raise _UnsupportedChain(message)


@dataclass(frozen=True)
class WarpPairLayout:
    producer: tuple[int, int, int]
    consumer: tuple[int, int, int]
    source_dtype: torch.dtype
    dtype: torch.dtype
    threads: int
    slots: int
    panels: int

    @property
    def topology(self) -> tuple[int, int, int]:
        return (self.threads // 32, 1, 1)


@dataclass(frozen=True)
class WarpPairTeam:
    """One contiguous physical owner for both members of a proved pair.

    The scheduler owns the named barrier and establishes the local indices
    inside ``predicate``. A CTA barrier is never legal inside this scope.
    """

    layout: WarpPairLayout
    execution: ChainedExecution
    first_thread: int
    cta_threads: int
    barrier_id: int

    def __post_init__(self) -> None:
        if (
            self.execution.threads != self.layout.threads
            or type(self.first_thread) is not int
            or self.first_thread < 0
            or self.first_thread % 32
            or type(self.cta_threads) is not int
            or self.cta_threads % 32
            or self.first_thread + self.layout.threads > self.cta_threads
            or type(self.barrier_id) is not int
            or not 1 <= self.barrier_id <= 15
            or not self.execution.thread.isidentifier()
            or not self.execution.warp.isidentifier()
            or self.execution.thread == "chain_thread"
            or self.execution.warp == "chain_warp"
            or not self.execution.barriers.isidentifier()
            or len(
                {self.execution.thread, self.execution.warp, self.execution.barriers}
            )
            != 3
            or self.execution.sync != f"{self.execution.barriers}.arrive_and_wait()"
        ):
            _reject("warp pair lacks a contiguous local team and named join")

    @property
    def predicate(self) -> str:
        return (
            f"{self.first_thread} <= chain_thread < "
            f"{self.first_thread + self.layout.threads}"
        )

    @property
    def bindings(self) -> tuple[str, str]:
        return (
            f"{self.execution.thread} = chain_thread - {self.first_thread}",
            f"{self.execution.warp} = {self.execution.thread} // 32",
        )

    @property
    def declaration(self) -> str:
        return (
            f"{self.execution.barriers} = chain_pair_pipeline.NamedBarrier("
            f"barrier_id={self.barrier_id}, num_threads={self.layout.threads})"
        )


@lru_cache(maxsize=128)
def native_pair_layout(
    producer: tuple[int, int, int],
    consumer: tuple[int, int, int],
    source_dtype: torch.dtype,
    dtype: torch.dtype,
) -> WarpPairLayout | None:
    """Prove the actual C rmem, whole-A storage and every increasing K slice."""
    if (
        source_dtype not in (torch.float16, torch.bfloat16)
        or dtype not in (torch.float16, torch.bfloat16)
        or any(type(n) is not int or not 0 < n <= 32 for n in (*producer, *consumer))
        or producer[0] not in (16, 32)
        or consumer[0] != producer[0]
        or producer[1] != consumer[2]
        or any(n % 8 for n in (*producer[1:], *consumer[1:]))
    ):
        return None
    import cutlass
    import cutlass.cute as cute

    ir = importlib.import_module("cutlass._mlir.ir")
    threads = 2 * producer[0]
    topology = (threads // 32, 1, 1)
    kinds = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mmas = tuple(
                cute.make_tiled_mma(
                    cute.make_mma_atom(
                        cute.nvgpu.warp.MmaF16BF16Op(
                            kinds[kind],
                            cutlass.Float32,
                            (16, 8, 16 if shape[2] % 16 == 0 else 8),
                        )
                    ),
                    atom_layout_mnk=topology,
                )
                for shape, kind in ((producer, source_dtype), (consumer, dtype))
            )
            all_c: list[list[tuple[int, int]]] = [[], []]
            slots = panels = 0
            for lane in range(threads):
                source = (
                    mmas[0]
                    .get_slice(lane)
                    .partition_C(cute.make_identity_tensor(producer[:2]))
                )
                target = (
                    mmas[1]
                    .get_slice(lane)
                    .partition_A(cute.make_identity_tensor((consumer[0], consumer[2])))
                )
                source_coords = tuple(
                    (int(source[i][0]), int(source[i][1]))
                    for i in range(cute.size(source))
                )
                target_coords = tuple(
                    (int(target[i][0]), int(target[i][1]))
                    for i in range(cute.size(target))
                )
                if source_coords != target_coords:
                    return None
                all_c[0].extend(source_coords)
                destination = (
                    mmas[1]
                    .get_slice(lane)
                    .partition_C(cute.make_identity_tensor(consumer[:2]))
                )
                all_c[1].extend(
                    (int(destination[i][0]), int(destination[i][1]))
                    for i in range(cute.size(destination))
                )
                if lane == 0:
                    c = cute.make_rmem_tensor(
                        mmas[0].partition_shape_C(producer[:2]), cutlass.Float32
                    )
                    a = mmas[1].make_fragment_A(target.shape)
                    slots = int(cute.size(a))
                    panels = int(cute.size(a, mode=[2]))
                    if int(cute.size(c)) != slots:
                        return None
                    for tensor in (c, a):
                        if sorted(int(tensor.layout(i)) for i in range(slots)) != list(
                            range(slots)
                        ):
                            return None
                    offsets: list[int] = []
                    logical = [int(a.layout(i)) for i in range(slots)]
                    for kk in range(panels):
                        panel = a[None, None, kk]
                        coordinates = target[None, None, kk]
                        indices = [
                            int(a.layout((i, j, kk)))
                            for j in range(cute.size(panel, mode=[1]))
                            for i in range(cute.size(panel, mode=[0]))
                        ]
                        if [target_coords[logical.index(i)] for i in indices] != [
                            (int(coordinates[i][0]), int(coordinates[i][1]))
                            for i in range(cute.size(coordinates))
                        ]:
                            return None
                        offsets.extend(indices)
                    if sorted(offsets) != list(range(slots)):
                        return None
            for shape, coordinates in zip((producer, consumer), all_c, strict=True):
                expected = {(r, c) for r in range(shape[0]) for c in range(shape[1])}
                if len(coordinates) != len(expected) or set(coordinates) != expected:
                    return None
        if not module.operation.verify():
            return None
    return WarpPairLayout(
        producer, consumer, source_dtype, dtype, threads, slots, panels
    )


def _storage(plan: ChainedMatmulPlan) -> object:
    return _snapshot(
        (
            plan.preparation_pipeline,
            plan.contraction_groups,
            plan.root_typed_operands,
            plan.initialized_accumulator,
            plan.late_rhs_reuse,
            plan.k_schedule,
            plan.warp_mma_stages,
            plan.warp_workspace,
            plan.warp_result_transport,
            plan.root_exports,
            plan.scan_exports,
        )
    )


@lru_cache(maxsize=128)
def native_pair_copies(
    shape: tuple[int, int, int], dtype: torch.dtype, axes: tuple[int, int]
) -> bool:
    """Independent PTX m8n8 packet delivery for the original padded A/B views."""
    if (
        dtype not in (torch.float16, torch.bfloat16)
        or shape[0] not in (16, 32)
        or any(type(n) is not int or not 0 < n <= 32 or n % 8 for n in shape)
        or axes not in ((0, 0), (0, 1), (1, 0), (1, 1))
    ):
        return False
    import cutlass
    import cutlass.cute as cute

    kind = cutlass.Float16 if dtype is torch.float16 else cutlass.BFloat16
    ir = importlib.import_module("cutlass._mlir.ir")
    atom_k = 16 if shape[2] % 16 == 0 else 8
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = cute.make_tiled_mma(
                cute.make_mma_atom(
                    cute.nvgpu.warp.MmaF16BF16Op(kind, cutlass.Float32, (16, 8, atom_k))
                ),
                atom_layout_mnk=(shape[0] // 16, 1, 1),
            )
            for role, extent, inner, matrices in (
                ("A", shape[0], axes[0], atom_k // 4),
                ("B", shape[1], axes[1], atom_k // 8),
            ):
                identity = cute.make_identity_tensor((extent, shape[2]))
                layout = cute.make_layout(
                    (extent, shape[2]),
                    stride=(shape[2] + 8, 1) if inner else (1, extent + 8),
                )
                capacity = extent * shape[2] + 8 * max(extent, shape[2])
                copy = getattr(cute, f"make_tiled_copy_{role}")(
                    cute.make_copy_atom(
                        cute.nvgpu.warp.LdMatrix8x8x16bOp(
                            transpose=inner == 0, num_matrices=matrices
                        ),
                        kind,
                    ),
                    mma,
                )
                for warp in range(shape[0] // 16):
                    sources = [
                        copy.get_slice(warp * 32 + lane).partition_S(identity)
                        for lane in range(32)
                    ]
                    fragments = [
                        getattr(mma.get_slice(warp * 32 + lane), f"partition_{role}")(
                            identity
                        )
                        for lane in range(32)
                    ]
                    destinations = [
                        copy.get_slice(warp * 32 + lane).retile(fragments[lane])
                        for lane in range(32)
                    ]
                    for tile in range(cute.size(sources[0], mode=[1])):
                        for kk in range(shape[2] // atom_k):
                            packets = []
                            for lane in range(8 * matrices):
                                panel = sources[lane][None, tile, kk]
                                coords = [
                                    tuple(map(int, panel[i]))
                                    for i in range(cute.size(panel))
                                ]
                                offsets = [int(layout(coord)) for coord in coords]
                                if (
                                    len(offsets) != 8
                                    or offsets
                                    != list(range(offsets[0], offsets[0] + 8))
                                    or offsets[0] % 8
                                    or offsets[0] < 0
                                    or offsets[-1] >= capacity
                                    or any(
                                        not (
                                            0 <= r < extent
                                            and kk * atom_k <= c < (kk + 1) * atom_k
                                        )
                                        for r, c in coords
                                    )
                                ):
                                    return False
                                packets.append(coords)
                            for lane in range(32):
                                delivered = []
                                for matrix in range(matrices):
                                    for half in range(2):
                                        r, c = lane // 4, 2 * (lane % 4) + half
                                        if inner == 0:
                                            r, c = c, r
                                        delivered.append(packets[matrix * 8 + r][c])
                                panel = destinations[lane][None, tile, kk]
                                if delivered != [
                                    tuple(map(int, panel[i]))
                                    for i in range(cute.size(panel))
                                ]:
                                    return False
        return bool(module.operation.verify())


@dataclass(frozen=True)
class WarpPairStage:
    """One actual original-stage issue, tied to a single transport attempt."""

    owner: WarpPairAttempt
    stage: int
    producer: bool
    _issue: tuple[str, ...] | None = field(default=None, init=False, repr=False)
    _prefix: tuple[str, ...] | None = field(default=None, init=False, repr=False)
    _boundaries: tuple[tuple[Node, str], ...] | None = field(
        default=None, init=False, repr=False
    )
    _aliases: tuple[tuple[str, str], ...] = field(default=(), init=False, repr=False)
    _inner_axes: tuple[tuple[str, int], ...] = field(default=(), init=False, repr=False)
    _publication: tuple[str, ...] = field(default=(), init=False, repr=False)

    @property
    def topology(self) -> tuple[int, int, int]:
        return self.owner.layout.topology

    @property
    def register_a(self) -> str | None:
        return None if self.producer else f"{self.owner.prefix}_a"

    def check_issue(
        self,
        prefix: str,
        dtype: str,
        shape: tuple[int, int, int],
        threads: int,
        seed: Sequence[str],
        inner_axes: dict[str, int],
        execution: ChainedExecution | None = None,
    ) -> None:
        self.owner.check()
        execution = execution or self.owner.execution
        expected = self.owner.source if self.producer else self.owner.source + 1
        if (
            self is not (self.owner.producer if self.producer else self.owner.consumer)
            or self.stage != expected
            or self._issue is not None
            or self._prefix is None
            or prefix != f"chain_{expected}"
            or shape != self.owner.plan.shapes[expected]
            or threads != self.owner.layout.threads
            or dtype != self.owner.dtype_name(expected)
            or list(seed) != [f"{prefix}_acc.fill(0.0)"]
            or self.owner.phase != ("declared" if self.producer else "bound")
            or set(inner_axes) != {"a", "b"}
            or execution != self.owner.execution
            or self.owner.team is not None
            and execution is not self.owner.team.execution
            or not native_pair_copies(
                shape,
                self.owner.plan.operand_dtype(expected),
                (inner_axes["a"], inner_axes["b"]),
            )
        ):
            _reject("warp pair issue owner, geometry, seed or readiness changed")
        object.__setattr__(self, "_inner_axes", tuple(inner_axes.items()))

    def before_issue(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        prefix: Sequence[str],
    ) -> None:
        self.owner.check()
        required = (
            self.owner._prefix + self.owner._declaration
            if self.producer
            else self.owner._joined
        )
        if (
            cg is not self.owner.codegen
            or plan is not self.owner.plan
            or self._prefix is not None
            or self.owner.phase != ("declared" if self.producer else "bound")
            or not required
            or tuple(prefix[: len(required)]) != required
            or not self.producer
            and tuple(boundaries.items()) != self.owner._boundaries
        ):
            _reject(
                "warp pair before-issue declaration, prefix or original boundaries changed"
            )
        object.__setattr__(self, "_prefix", tuple(prefix))
        object.__setattr__(self, "_boundaries", tuple(boundaries.items()))
        object.__setattr__(self, "_aliases", tuple(plan.tensor_aliases.items()))

    def record_issue(self, lines: Sequence[str]) -> None:
        if self._issue is not None or not lines:
            _reject("warp pair issue emitted twice")
        object.__setattr__(self, "_issue", tuple(lines))


@dataclass
class WarpPairAttempt:
    codegen: GenerateAST
    plan: ChainedMatmulPlan
    source: int
    layout: WarpPairLayout
    expression: FragmentExpression
    revision: object
    config: object
    storage: object
    phase: str = field(default="planned", init=False)
    producer: WarpPairStage = field(init=False)
    consumer: WarpPairStage = field(init=False)
    _fill: tuple[str, ...] = field(default=(), init=False, repr=False)
    _prefix: tuple[str, ...] = field(default=(), init=False, repr=False)
    _operand: WarpOperandLayout | None = field(default=None, init=False, repr=False)
    _operand_facts: object = field(default=None, init=False, repr=False)
    _selection: object = field(init=False, repr=False)
    _declaration: tuple[str, ...] = field(default=(), init=False, repr=False)
    _joined: tuple[str, ...] = field(default=(), init=False, repr=False)
    _boundaries: tuple[tuple[Node, str], ...] = field(
        default=(), init=False, repr=False
    )
    _aliases: tuple[tuple[str, str], ...] = field(default=(), init=False, repr=False)
    team: WarpPairTeam | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.producer = WarpPairStage(self, self.source, True)
        self.consumer = WarpPairStage(self, self.source + 1, False)
        self._selection = self.selection()

    def selection(self) -> object:
        return (
            id(self.codegen),
            id(self.plan),
            self.source,
            self.layout,
            id(self.expression),
            self.revision,
            self.config,
            self.storage,
            id(self.producer),
            id(self.consumer),
            None
            if self.team is None
            else (
                id(self.team),
                self.team.layout,
                _execution(self.team.execution),
                self.team.first_thread,
                self.team.cta_threads,
                self.team.barrier_id,
            ),
        )

    @property
    def execution(self) -> ChainedExecution:
        return (
            ChainedExecution(self.layout.threads)
            if self.team is None
            else self.team.execution
        )

    def bind_team(self, team: WarpPairTeam) -> None:
        """Bind before emission; a consumed attempt cannot acquire a new owner."""
        self.check()
        if (
            self.phase != "planned"
            or self.team is not None
            or team.layout != self.layout
            or team.cta_threads != self.plan.threads
        ):
            _reject("warp pair team changed after planning")
        self.team = team
        self._selection = self.selection()

    @property
    def prefix(self) -> str:
        return f"chain_pair_{self.source}"

    def dtype_name(self, stage: int) -> str:
        return (
            "cutlass.Float16"
            if self.plan.operand_dtype(stage) is torch.float16
            else "cutlass.BFloat16"
        )

    def check(self) -> None:
        if (
            self.selection() != self._selection
            or self.producer.owner is not self
            or self.consumer.owner is not self
            or not self.producer.producer
            or self.consumer.producer
            or self.producer.stage != self.source
            or self.consumer.stage != self.source + 1
            or not 0 <= self.source < len(self.plan.dots) - 1
            or self.expression.source is not self.plan.dots[self.source]
            or self.expression.target is not self.plan.dots[self.source + 1].args[0]
            or self.expression.fragment
            != f"chain_{self.source}_acc[{self.prefix}_index]"
            or self.expression.coordinates
            != (f"{self.prefix}_row", f"{self.prefix}_col")
            or self.expression.target_coordinates != self.expression.coordinates
            or self.expression.dtype != self.dtype_name(self.source + 1)
            or warp_revision(self.plan, aliases=False) != self.revision
            or _snapshot(self.codegen.device_function.config.config) != self.config
            or _storage(self.plan) != self.storage
            or native_pair_layout(
                self.plan.shapes[self.source],
                self.plan.shapes[self.source + 1],
                self.plan.operand_dtype(self.source),
                self.plan.operand_dtype(self.source + 1),
            )
            != self.layout
            or self._operand is not None
            and (id(self._operand), _snapshot(self._operand.__dict__))
            != self._operand_facts
            or tuple(self.plan.tensor_aliases.items())[: len(self._aliases)]
            != self._aliases
            or any(
                tuple(self.plan.tensor_aliases.items())[: len(token._aliases)]
                != token._aliases
                for token in (self.producer, self.consumer)
            )
        ):
            _reject("warp pair graph, config, storage or native layout changed")
        self.expression.check()

    def declare(self, prefix: Sequence[str]) -> list[str]:
        self.check()
        if self.phase != "planned":
            _reject("warp pair private A declared twice")
        self.phase = "declared"
        self._prefix = tuple(prefix)
        name, shape = self.prefix, self.layout.consumer
        atom_k = 16 if shape[2] % 16 == 0 else 8
        self._declaration = (
            f"{name}_mma = cute.make_tiled_mma(cute.make_mma_atom(cute.nvgpu.warp.MmaF16BF16Op({self.dtype_name(self.source + 1)}, cutlass.Float32, (16, 8, {atom_k}))), atom_layout_mnk={self.layout.topology!r})",
            f"{name}_thr = {name}_mma.get_slice({self.execution.thread if self.team is not None else f'chain_thread % {self.layout.threads}'})",
            f"{name}_coords = {name}_thr.partition_A(cute.make_identity_tensor(({shape[0]}, {shape[2]})))",
            f"{name}_a = {name}_mma.make_fragment_A({name}_coords.shape)",
        )
        return list(self._declaration)

    def fill(self) -> list[str]:
        from .chained_matmul import _indent

        self.check()
        if self.phase != "declared" or self.producer._issue is None:
            _reject("warp pair A fill precedes its original producer issue")
        value, name = self.expression, self.prefix
        self._fill = (
            f"for {name}_index in cutlass.range_constexpr(cute.size({name}_a)):",
            f"    {value.coordinates[0]}, {value.coordinates[1]} = {name}_coords[{name}_index]",
            _indent(value.expression.lines),
            f"    {name}_a[{name}_index] = {value.masked_value()}",
        )
        self.phase = "filled"
        return list(self._fill)

    def bind_operand(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        stage: int,
        layout: WarpOperandLayout,
        prefix: Sequence[str],
    ) -> None:
        self.check()
        expected = self.layout.consumer
        if (
            cg is not self.codegen
            or plan is not self.plan
            or self.phase != "joined"
            or stage != self.source + 1
            or layout.role != "a"
            or layout.node is not self.expression.target
            or layout.shape != (expected[0], expected[2])
            or layout.dtype != self.layout.dtype
            or layout.inner not in (0, 1)
            or layout.stride
            != ((expected[2] + 8, 1) if layout.inner else (1, expected[0] + 8))
            or boundaries.get(plan.dots[self.source]) != f"chain_{self.source}_c"
            or not self._joined
            or tuple(prefix[: len(self._joined)]) != self._joined
        ):
            _reject("warp pair original A layout or producer publication changed")
        self._operand = layout
        self._operand_facts = id(layout), _snapshot(layout.__dict__)
        self._boundaries = tuple(boundaries.items())
        self._aliases = tuple(plan.tensor_aliases.items())
        self.phase = "bound"

    def complete(
        self,
        stage: int,
        prefix: Sequence[str],
        publication: Sequence[str],
        boundaries: dict[Node, str],
    ) -> None:
        """Called only after the original active compute, C store and CTA join."""
        from .chained_matmul import _indent

        self.check()
        token = self.producer if stage == self.source else self.consumer
        source = token is self.producer
        if (
            stage != token.stage
            or self.phase != ("filled" if source else "bound")
            or token._issue is None
            or token._prefix is None
            or not publication
            or not prefix
            or prefix[-1] != self.execution.sync
            or tuple(prefix[: len(token._prefix)]) != token._prefix
            or tuple(boundaries.items()) != token._boundaries
        ):
            _reject("warp pair original stage completion changed")
        block = [*token._issue, *(self._fill if source else ()), *publication]
        actual = list(prefix)
        expected = (
            [f"if chain_thread < {self.layout.threads}:", _indent(block)]
            if self.team is None and self.layout.threads < self.plan.threads
            else block
        )
        if actual[-len(expected) - 1 : -1] != expected:
            _reject("warp pair issue, fill, shared publication or join was reordered")
        object.__setattr__(token, "_publication", tuple(publication))
        self.phase = "joined" if source else "consumed"
        if source:
            self._joined = tuple(prefix)


def plan_warp_pairs(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scans: list[_ScanInput],
    schedule: str,
) -> tuple[WarpPairAttempt, ...]:
    if cg.device_function.config.config.get("cute_chained_warp_register_a") is not True:
        return ()
    if (
        plan.strategy != "warp"
        or plan.loop is not None
        or schedule not in ("coalesced", "cp_async")
        or plan.preparation_pipeline is not None
        or plan.contraction_groups is not None
        or plan.root_typed_operands is not None
        or plan.initialized_accumulator is not None
        or plan.late_rhs_reuse is not None
        or plan.k_schedule is not None
        or plan.warp_mma_stages
        or plan.warp_workspace is not None
        or plan.warp_result_transport is not None
        and plan.warp_result_transport.mode != "direct"
        or plan.pointwise_cache is not None
    ):
        _reject("warp register A requires original unaliased straight-line warp stages")
    return _collect_warp_pairs(cg, plan, boundaries, scans)


def _collect_warp_pairs(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scans: list[_ScanInput],
) -> tuple[WarpPairAttempt, ...]:
    """Shared graph, expression and physical-owner admission for adjacent pairs."""
    from . import chained_matmul as chain

    frontier = PhysicalUseFrontier(tuple(plan.dots[0].graph.nodes))
    all_boundaries = {
        **boundaries,
        **{node: f"chain_{i}_c" for i, node in enumerate(plan.dots)},
    }
    pairs: list[WarpPairAttempt] = []
    occupied: set[int] = set()
    for stage in range(1, len(plan.dots)):
        producer, consumer = plan.dots[stage - 1 : stage + 1]
        if (
            plan.warp_result_transport is not None
            and plan.warp_result_transport.terminal is not None
            and consumer is plan.warp_result_transport.terminal.source
        ):
            # Pair completion still owns its original shared C publication.
            continue
        operand = cast("Node", consumer.args[0])
        if (
            stage - 1 in occupied
            or producer.args[2] is not None
            or consumer.args[2] is not None
            or chain._shape(operand) != plan.shapes[stage - 1][:2]
            or not frontier.exclusive_use(producer, operand, consumer)
            or producer in chain._ancestors(cast("Node", consumer.args[1]))
            or plan.region is not None
            and any(
                producer in chain._ancestors(node)
                for node in (*plan.region.scans, *plan.region.reductions)
            )
        ):
            continue
        layout = native_pair_layout(
            plan.shapes[stage - 1],
            plan.shapes[stage],
            plan.operand_dtype(stage - 1),
            plan.operand_dtype(stage),
        )
        if layout is None or layout.threads > plan.threads:
            continue
        # A single-source unary/view region can be evaluated at the producer
        # without moving any other tensor read across its original boundary.
        pending, seen = [operand], set()
        valid = True
        while pending:
            node = pending.pop()
            if node is producer or node in seen:
                continue
            seen.add(node)
            if (
                len(seen) > 32
                or node.op != "call_function"
                or node.is_impure()
                or len(node.all_input_nodes) != 1
                or not isinstance(node.meta.get("val"), torch.Tensor)
            ):
                valid = False
                break
            pending.extend(node.all_input_nodes)
        if not valid:
            continue
        name = f"chain_pair_{stage - 1}"
        coords = (f"{name}_row", f"{name}_col")
        try:
            expression = bind_fragment_expression(
                cg,
                plan,
                all_boundaries,
                scans,
                producer,
                operand,
                coords,
                coords,
                f"chain_{stage - 1}_acc[{name}_index]",
                "cutlass.Float16"
                if layout.dtype is torch.float16
                else "cutlass.BFloat16",
            )
        except chain._UnsupportedChain:
            continue
        if expression.expression.loaded_inputs or expression.expression.accesses:
            continue
        pairs.append(
            WarpPairAttempt(
                cg,
                plan,
                stage - 1,
                layout,
                expression,
                warp_revision(plan, aliases=False),
                _snapshot(cg.device_function.config.config),
                _storage(plan),
            )
        )
        occupied.update((stage - 1, stage))
    frontier.check()
    if not pairs:
        _reject("warp register A lacks an exclusive adjacent native pair")
    return tuple(pairs)


def has_warp_pair_candidate(
    graphs: Sequence[GraphInfo], env: CompileEnvironment
) -> bool:
    """Find a static small unary C-to-A path; codegen still proves ownership."""
    from ..compile_environment import FixedBlockSizeSource
    from ..compile_environment import ReductionLoopBlockSizeSource
    from .chained_matmul import _classify_chained_graph

    graph = _classify_chained_graph(graphs)
    if graph is None or graph.loop is not None:
        return False

    def extent(value: object) -> int | None:
        if type(value) is int:
            return value
        if not isinstance(value, torch.SymInt):
            return None
        block_id = env.get_block_id(value)
        if block_id is None:
            return None
        block = env.block_sizes[env.canonical_block_id(block_id)]
        source = block.block_size_source
        if isinstance(source, FixedBlockSizeSource):
            return source.value if type(source.value) is int else None
        if isinstance(source, ReductionLoopBlockSizeSource):
            return env._search_invariant_extent_for_numel_constraint(block)
        return None

    frontier = PhysicalUseFrontier(tuple(graph.dots[0].graph.nodes))
    for producer, consumer in zip(graph.dots[:-1], graph.dots[1:], strict=True):
        if producer.args[2] is not None or consumer.args[2] is not None:
            continue
        operands = (
            producer.args[0],
            producer.args[1],
            consumer.args[0],
            consumer.args[1],
        )
        tensors = [cast("Node", node).meta.get("val") for node in operands]
        if any(
            not isinstance(value, torch.Tensor)
            or value.dtype not in (torch.float16, torch.bfloat16)
            or value.ndim != 2
            for value in tensors
        ):
            continue
        shapes = [
            tuple(extent(size) for size in cast("torch.Tensor", value).shape)
            for value in tensors
        ]
        if any(
            type(size) is not int or not 0 < size <= 32
            for shape in shapes
            for size in shape
        ):
            continue
        left, right, second_left, second_right = (
            cast("tuple[int, int]", shape) for shape in shapes
        )
        if (
            left[0] not in (16, 32)
            or second_left[0] != left[0]
            or left[1] != right[0]
            or right[1] != second_left[1]
            or second_left[1] != second_right[0]
            or any(
                size % 8
                for shape in (left, right, second_left, second_right)
                for size in shape
            )
        ):
            continue
        operand = cast("Node", consumer.args[0])
        if not frontier.exclusive_use(producer, operand, consumer):
            continue
        pending, seen = [operand], set()
        while pending:
            node = pending.pop()
            if node is producer or node in seen:
                continue
            seen.add(node)
            if (
                len(seen) > 32
                or node.op != "call_function"
                or node.is_impure()
                or len(node.all_input_nodes) != 1
                or not isinstance(node.meta.get("val"), torch.Tensor)
            ):
                break
            pending.extend(node.all_input_nodes)
        else:
            frontier.check()
            return True
    frontier.check()
    return False
