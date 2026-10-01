"""Private binding of the existing loop transport to a shared native edge.

This does not discover a transport or grant an owner lifetime. The original
LoopTmemTransport and enclosing RecurrenceBody retain those obligations. One
binding follows their selected source stage through issue, read, unchanged
point lowering, publication and the original final join. Intermediate returns
are not completion receipts.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
import inspect
from itertools import starmap
from textwrap import dedent
from types import CodeType
from types import FunctionType
from typing import TYPE_CHECKING

import torch

from . import chained_matmul as chain
from .chained_completed_store import _records
from .chained_execution import ChainedExecution
from .chained_loop_tmem_slots import _valid_candidate
from .chained_pipeline_storage import _facts
from .chained_root_stage import _execution_fields

if TYPE_CHECKING:
    from torch.fx import Node

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from .chained_loop_tmem_transport import LoopTmemTransport
    from .chained_matmul import ChainedMatmulPlan
    from .chained_tcgen_stage import StageGeometry
    from .chunk_recurrence import CuteChunkRecurrencePlan


def _lines(lines: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    # Permit only uniform enclosing-role indentation, not a changed scope.
    return tuple(dedent("\n".join(lines)).splitlines())


@dataclass(frozen=True)
class MatchedPreparedProjection:
    """Original nodes retained only after the existing complete semantic match.

    The private explicit-host fixture uses this capture, not a reconstructed
    graph or a general-stage plan. This is source selection authority only;
    neither it nor returning the host keyword certifies GPU completion.
    """

    root: GraphInfo
    loop: GraphInfo
    source: Node
    operand: Node
    state: Node
    revision: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from ..backend_registry import repair_backend_codegen

        # Complete first-use handler registration before freezing its identity.
        # Later changes still invalidate the full original graph witness.
        repair_backend_codegen("cute")
        object.__setattr__(self, "revision", self._current())

    def _current(self) -> tuple[object, ...]:
        return (
            tuple(
                id(value)
                for value in (
                    self.root,
                    self.loop,
                    self.source,
                    self.operand,
                    self.state,
                )
            ),
            tuple(
                (
                    node,
                    node.op,
                    node.target,
                    _records(node.args),
                    _records(node.kwargs),
                    _records(node.meta),
                    tuple(node.users),
                )
                for graph in (self.root.graph, self.loop.graph)
                for node in graph.nodes
            ),
        )

    def check(self) -> None:
        if self._current() != self.revision:
            raise chain._UnsupportedChain("matched projection graph changed")


def _native_source(function: object, executor: str) -> tuple[object, ...]:
    """Original CuTe source dependency, not transformed-bytecode attestation.

    CuTe keeps the original inner code while replacing its active code during
    preprocessing. Inspect that retained code so decorators remain included;
    never normalize source or adopt the post-compilation context.
    """
    if (
        not isinstance(function, FunctionType)
        or function.__code__.co_freevars != ("executor_name", "func")
        or function.__closure__ is None
    ):
        raise chain._UnsupportedChain("unknown original CuTe wrapper")
    kind, inner = (cell.cell_contents for cell in function.__closure__)
    if (
        kind != executor
        or not isinstance(inner, FunctionType)
        or vars(function).get("__wrapped__") is not inner
        or vars(inner).get("_preprocess_enabled") is not True
    ):
        raise chain._UnsupportedChain("changed original CuTe wrapper")
    code = vars(inner).get("_original_code", inner.__code__)
    if not isinstance(code, CodeType):
        raise chain._UnsupportedChain("invalid original CuTe code")
    # Keep strong references as well as identities: CodeType equality is
    # structural, and retaining only ids would permit object-id recycling.
    return (
        function,
        function.__code__,
        id(function.__code__),
        inner,
        code,
        id(code),
        kind,
        _records(vars(inner).get("_decorator_location")),
        inspect.getsource(code),
    )


@dataclass(eq=False)
class PreparedProjectionHost:
    """Private original-host selection, bound to complete selected native code.

    The host/kernel are the original entries; no callback or replacement root
    is installed. Publication is still performed by the shared executor in
    the original reader role. The caller must check this binding before and
    after the complete original-host compilation, never after just issue/read.
    """

    plan: CuteChunkRecurrencePlan
    _context: object = field(init=False, repr=False)
    _bindings: tuple[object, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            self.plan.prepared_projection is None
            or self.plan.schedule != "sm100_tmem"
            or self.plan.pipeline != "wide"
            or self.plan.dv_partitions != 2
            or self.plan.state.fake.dtype is not torch.float32
        ):
            raise chain._UnsupportedChain("unselected original projection host")
        self._bindings = self._binding_identities()
        self._check_bindings()
        self._context = self._current()

    def _binding_identities(self) -> tuple[object, ...]:
        projection = self.plan.prepared_projection
        continuation = self.plan.prepared_continuation
        epoch = self.plan.prepared_epoch
        # Hold the original objects strongly, including their immutable captured
        # revisions. A replacement proof cannot reset the host's authority.
        return (
            self.plan,
            projection,
            None if projection is None else projection.revision,
            continuation,
            None if continuation is None else continuation._revision,
            None if continuation is None else continuation.graph,
            None if continuation is None else continuation.graph._revision,
            epoch,
            None if epoch is None else epoch._identity,
            None if epoch is None else epoch.projected,
            None if epoch is None else epoch.projected._revision,
        )

    def _check_bindings(self) -> None:
        if any(
            current is not original
            for current, original in zip(
                self._binding_identities(), self._bindings, strict=True
            )
        ):
            raise chain._UnsupportedChain("original prepared host binding changed")
        projection = self.plan.prepared_projection
        continuation = self.plan.prepared_continuation
        epoch = self.plan.prepared_epoch
        if projection is None:
            raise chain._UnsupportedChain("original projection witness lost")
        if epoch is not None:
            if (
                epoch.projection is not projection
                or epoch.continuation is not continuation
            ):
                raise chain._UnsupportedChain("foreign prepared host epoch binding")
            # The epoch checks its projection, continuation and K segments.
            epoch.check()
        else:
            projection.check()
            if continuation is not None:
                continuation.check()

    def _current(self) -> tuple[object, ...]:
        from . import chunk_recurrence_sm100 as original
        from . import prepared_tcgen_edge as shared

        # Exact complete native spans include the original point body, role
        # branches, both half waits, and all arrival contributions. This is a
        # dependency pin, not source-spelling based eligibility.
        functions = (
            (original.host_chain_dv2, "_func"),
            (original.kernel_chain_dv2, "_kernel_helper"),
            (original._issue_prepared_state_k, "_func"),
            (original.tcgen05_chain_stage_vmx_input_tmem, "_func"),
            (original.prepared_residual_point, "_func"),
            (original.prepared_output_point, "_func"),
            (original.tcgen05_rhs_token_pair_from_16x256b_fragment, "_func"),
            (original.tcgen05_chain_store_output_smem, "_func"),
            (shared._prepare_descriptor_issue, "_func"),
            (shared._issue_atom, "_func"),
            (shared._read_companion, "_func"),
            (shared.execute_prepared_wait, "_func"),
            (shared.execute_prepared_issue, "_func"),
            (shared.execute_prepared_read, "_func"),
            (shared.pack_layout_f_state, "_func"),
            (shared.scale_layout_f_state, "_func"),
            (shared.execute_prepared_store, "_func"),
            (shared.execute_prepared_store_completion, "_func"),
            (shared.execute_prepared_publication, "_func"),
            (shared._continuation_control, "_func"),
            (shared.execute_prepared_continuation, "_func"),
        )
        return (
            id(self.plan),
            _records(
                tuple(
                    (item.name, getattr(self.plan, item.name))
                    for item in fields(self.plan)
                    if item.name
                    not in (
                        "prepared_projection",
                        "prepared_continuation",
                        "prepared_epoch",
                    )
                )
            ),
            tuple(starmap(_native_source, functions)),
            _records((original.ROLES, original._TMEM_LAYOUT)),
            tuple(
                (name, value)
                for name, value in vars(original).items()
                if type(value) in (int, str, bool, float)
            ),
        )

    def check(self) -> None:
        self._check_bindings()
        if self._current() != self._context:
            raise chain._UnsupportedChain("original prepared host context changed")

    def host_keywords(self) -> dict[str, bool]:
        self.check()
        return {"PREPARED_EDGE": True}


@dataclass(eq=False)
class PreparedTmemEdge:
    codegen: GenerateAST
    plan: ChainedMatmulPlan
    boundaries: dict[Node, str]
    stage: int
    geometry: StageGeometry
    phase: str
    execution: ChainedExecution
    transport: LoopTmemTransport
    _context: object = field(default=None, init=False, repr=False)
    _maps: tuple[tuple[dict[str, str], tuple[tuple[str, str], ...]], ...] = field(
        default=(), init=False, repr=False
    )
    _boundary_owner: object = field(default=None, init=False, repr=False)
    _boundary_values: tuple[tuple[Node, str], ...] = field(
        default=(), init=False, repr=False
    )
    _steps: tuple[tuple[str, ...], ...] = field(default=(), init=False, repr=False)
    _complete: tuple[str, ...] | None = field(default=None, init=False, repr=False)
    _publication: tuple[str, ...] = field(default=(), init=False, repr=False)
    _stage_setup: tuple[str, ...] | None = field(default=None, init=False, repr=False)
    _load_setup: tuple[str, ...] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        candidate = self.transport.slot.candidate
        if (
            self.plan.loop is None
            or self.plan.region is not candidate.region
            or self.stage != candidate.source_stage
            or self.plan.dots[self.stage] is not candidate.source
            or self.geometry != candidate.source_geometry
            or self.execution.threads < 128
            or not _valid_candidate(candidate, self.transport.slot.column_offset)
            or self.transport.slot.columns != candidate.physical_shape[1] // 2
        ):
            raise chain._UnsupportedChain("invalid prepared TCgen edge selection")
        transport_execution = ChainedExecution(
            128,
            thread=self.execution.thread,
            warp=self.execution.warp,
            sync="chain_tmem_barrier.arrive_and_wait()",
            tmem=self.execution.tmem,
        )
        # The already-lowered _Expression is rendered once, with its original
        # full-domain typed publication. Only the native store tail delegates.
        self._publication = tuple(self.transport.emit(transport_execution))
        prefix = self.transport.prefix
        if self._publication[-4:] != (
            transport_execution.sync,
            f"cute.copy({prefix}_copy, {prefix}_packed, {prefix}_destination)",
            "cute.arch.fence_view_async_tmem_store()",
            transport_execution.sync,
        ):
            raise chain._UnsupportedChain("unknown original TMEM publication")
        self._context, self._maps = self._current()
        self._boundary_owner = self.boundaries
        self._boundary_values = tuple(self.boundaries.items())

    def _current(
        self,
    ) -> tuple[object, tuple[tuple[dict[str, str], tuple[tuple[str, str], ...]], ...]]:
        aliases: dict[int, tuple[dict[str, str], tuple[tuple[str, str], ...]]] = {}
        facts = _records((self.plan, self.transport, self.geometry), aliases=aliases)
        return (
            (
                tuple(
                    id(value)
                    for value in (
                        self.codegen,
                        self.plan,
                        self.transport,
                        self.geometry,
                        self.execution,
                    )
                ),
                facts,
                _facts(self.plan),
                _records(self.codegen.device_function.config.config),
                _execution_fields(self.execution),
                self.stage,
                self.phase,
                self._publication,
            ),
            tuple(aliases.values()),
        )

    def check(self) -> None:
        context, maps = self._current()
        if (
            context != self._context
            or self.boundaries is not self._boundary_owner
            or tuple(self.boundaries.items()) != self._boundary_values
            or len(maps) != len(self._maps)
            or any(
                current is not old
                or (
                    tuple(current.items())
                    if self._steps
                    else tuple(current.items())[: len(values)]
                )
                != values
                for (old, values), (current, _) in zip(self._maps, maps, strict=True)
            )
        ):
            raise chain._UnsupportedChain("prepared TCgen edge context changed")

    def validate(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        stage: int,
        geometry: StageGeometry,
        phase: str,
        execution: ChainedExecution,
        transport: LoopTmemTransport,
    ) -> None:
        self.check()
        if (
            cg is not self.codegen
            or plan is not self.plan
            or boundaries is not self.boundaries
            or stage != self.stage
            or geometry is not self.geometry
            or phase != self.phase
            or execution is not self.execution
            or transport is not self.transport
            or self._steps
            or self._complete is not None
        ):
            raise chain._UnsupportedChain("foreign or already used prepared edge")

    def _advance(self, expected: int, lines: list[str]) -> list[str]:
        self.check()
        if len(self._steps) != expected or self._complete is not None:
            raise chain._UnsupportedChain("prepared edge emission order changed")
        if expected == 0:
            # Original fill/seed lowering can register aliases before issue.
            # After issue all original point expressions are already lowered.
            self._maps = self._current()[1]
        self._steps = (*self._steps, _lines(lines))
        return lines

    def capture_stage_setup(self, lines: list[str]) -> None:
        """Original emitter prefix, captured before any edge callback returns."""
        self.check()
        if self._stage_setup is not None or self._steps or self._complete is not None:
            raise chain._UnsupportedChain("prepared edge stage setup repeated")
        self._stage_setup = tuple(lines)

    def capture_load_setup(self, prefix: str, lines: list[str]) -> None:
        """Original load factory setup, captured before its read callback."""
        self.check()
        if (
            prefix != f"chain_{self.stage}"
            or self._stage_setup is None
            or self._load_setup is not None
            or len(self._steps) != 1
            or self._complete is not None
        ):
            raise chain._UnsupportedChain("prepared edge load setup changed")
        self._load_setup = tuple(lines)

    def issue(self, prefix: str, initialized: bool, tmem_a: bool) -> list[str]:
        execution = self.execution
        if (
            prefix != f"chain_{self.stage}"
            or tmem_a
            or initialized != (self.plan.dots[self.stage].args[2] is not None)
            or self._stage_setup is None
        ):
            raise chain._UnsupportedChain("prepared edge seed policy changed")
        return self._advance(
            0,
            [
                "from helion._compiler.cute import prepared_tcgen_edge",
                f"{prefix}_rb = {prefix}_mma.make_fragment_B({prefix}_slice.partition_B({prefix}_b))",
                f"prepared_tcgen_edge.execute_prepared_issue({prefix}_ra, {prefix}_rb, {prefix}_acc, {prefix}_mma, {execution.warp} == 0, {execution.barriers} + {self.stage}, {self.phase}, None, None, False, {tmem_a}, 0, cute.size({prefix}_ra, mode=[2]), 16, None, {initialized}, True, True)",
            ],
        )

    def read(self, prefix: str) -> list[str]:
        if prefix != f"chain_{self.stage}" or self._load_setup is None:
            raise chain._UnsupportedChain("prepared edge read owner changed")
        return self._advance(
            1,
            [
                f"{prefix}_values, {prefix}_unused_companion = prepared_tcgen_edge.execute_prepared_read({prefix}_source, {prefix}_values, {prefix}_copy, None, None, None, False, None, 0)"
            ],
        )

    def publish(
        self,
        prefix: str,
        transport: LoopTmemTransport,
        execution: ChainedExecution,
    ) -> list[str]:
        if (
            transport is not self.transport
            or prefix != transport.prefix
            or execution.threads != 128
            or execution.thread != self.execution.thread
            or execution.warp != self.execution.warp
            or execution.tmem != self.execution.tmem
            or execution.sync != "chain_tmem_barrier.arrive_and_wait()"
        ):
            raise chain._UnsupportedChain("prepared edge publication owner changed")
        # The original emitter built this point program once from the captured
        # transport, including its exact casts, bounds and typed zero.
        return self._advance(
            2,
            [
                *self._publication[:-4],
                f"prepared_tcgen_edge.execute_prepared_publication({prefix}_packed, {prefix}_destination, {prefix}_copy, chain_tmem_barrier, None, False, None)",
            ],
        )

    def finish(self, lines: list[str]) -> None:
        self.check()
        if (
            len(self._steps) != 3
            or self._complete is not None
            or self._stage_setup is None
            or self._load_setup is None
        ):
            raise chain._UnsupportedChain("prepared edge publication incomplete")
        # The expected whole stage is assembled from the original setup
        # captures and independently owned operation tuples, never from the
        # observed aggregate or callback return lists. This admits no extra
        # effect, control flow, changed setup, or early exit between spans.
        expected = [*self._stage_setup, *self._steps[0]]
        result = [*self._load_setup, *self._steps[1], *self._steps[2]]
        if self.execution.threads > 128:
            result = [f"if {self.execution.thread} < 128:", chain._indent(result)]
        expected.extend((*result, self.execution.sync))
        if ast.dump(ast.parse("\n".join(lines))) != ast.dump(
            ast.parse("\n".join(expected))
        ):
            raise chain._UnsupportedChain("prepared edge complete stage changed")
        self._complete = tuple(lines)

    def completed(self, lines: list[str]) -> bool:
        self.check()
        return self._complete is not None and self._complete == tuple(lines)
