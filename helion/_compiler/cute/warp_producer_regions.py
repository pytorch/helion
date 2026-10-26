"""One-row producer prefixes with scalar publications and an unchanged CTA suffix."""

from __future__ import annotations

import ast
from copy import copy
from dataclasses import dataclass
import math
import operator
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

import torch

from ... import exc
from ...language import _tracing_ops
from ...language import memory_ops
from ...language import scan_ops
from ...language import tile_ops
from ..ast_extension import ExtendedAST
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..inductor_lowering import ReductionLowering
from .completed_scan import pure
from .pure_producer_regions import opaque_producer
from .register_loads import host_load_is_readonly

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from torch._inductor.ir import Reduction
    from torch.fx import Graph
    from torch.fx import Node

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo
    from .computed_fragment import Fragment
    from .computed_fragment import FragmentCompiler

KEY = "cute_fragment_warp_producer_regions"
DTYPES = (torch.float32, torch.float64, torch.int32, torch.int64)


@dataclass(frozen=True)
class WarpProducerRegion:
    nodes: tuple[Node, ...]
    capacity: int


def producer_prefix(
    graph: Graph,
    env: CompileEnvironment,
    graphs: Sequence[GraphInfo],
    shape: Callable[[object], tuple[int, ...]],
) -> WarpProducerRegion | None:
    """A current straight-line prefix; all data edges at its cut are scalar.

    Emission additionally validates every physical read and exact selected
    materialization. Lazy scalar expressions keep their ordinary suffix reads.
    """
    nodes = tuple(graph.nodes)
    if any(
        _tracing_ops.is_for_loop_target(n.target)
        or n.target in (_tracing_ops._if, _tracing_ops._while_loop)
        or n.op == "placeholder"
        for n in nodes
    ):
        return None
    selected: list[Node] = []
    capacity = 0
    scans = 0
    for node in nodes:
        value = node.meta.get("val")
        target = node.target
        if target in (_tracing_ops._host_tensor, _tracing_ops._get_symnode):
            selected.append(node)
            continue
        if node.op == "output":
            break
        if not isinstance(value, torch.Tensor):
            if target in (
                torch.ops.aten.sym_size.int,
                operator.add,
                operator.sub,
                operator.mul,
                operator.floordiv,
                tile_ops.tile_begin,
                tile_ops.tile_end,
                tile_ops.tile_id,
            ):
                selected.append(node)
                continue
            break
        dims = shape(value.shape)
        size = math.prod(dims)
        if not (size == 1 or (dims and math.prod(dims[:-1]) == 1)):
            break
        if size != 1:
            if not 0 < size <= 1024 or capacity not in (0, size):
                break
            capacity = size
        if target is memory_ops.load:
            if not host_load_is_readonly(node, env, graphs):
                break
        elif target is scan_ops._associative_scan:
            if not (
                value.dtype in DTYPES
                and len(node.args) == 5
                and not node.kwargs
                and node.args[2] in (-1, len(dims) - 1)
                and node.args[3] is False
                and node.args[4] is False
                and size > 1
            ):
                break
            scans += 1
        elif isinstance(node.meta.get("lowering"), ReductionLowering):
            lowering = cast("ReductionLowering", node.meta["lowering"])
            if not (
                size == 1
                and value.dtype in DTYPES
                and lowering.reduction_type in ("sum", "min", "max")
                and shape(lowering.buffer.data.ranges) in ((), (1,))
                and shape(cast("Reduction", lowering.buffer.data).reduction_ranges)
                == (capacity,)
            ):
                break
        elif opaque_producer(node):
            if size != 1 or value.dtype not in DTYPES:
                break
        elif target not in (_tracing_ops._new_var, tile_ops.tile_index) and not pure(
            node
        ):
            break
        selected.append(node)
    if not scans:
        return None
    selected_set = set(selected)
    for node in selected:
        if any(user not in selected_set for user in node.users):
            value = node.meta.get("val")
            if (
                isinstance(value, torch.Tensor)
                and node.target is not _tracing_ops._host_tensor
                and math.prod(shape(value.shape)) != 1
            ):
                return None
    return WarpProducerRegion(tuple(selected), capacity)


@dataclass(frozen=True)
class CopyStep:
    statements: tuple[ast.AST, ...]
    source: Fragment
    target: Fragment
    workers: int


@dataclass(frozen=True)
class ScanStep:
    statements: tuple[ast.AST, ...]
    source: Fragment
    target: Fragment
    totals: Fragment


@dataclass(frozen=True)
class PublicationStep:
    statements: tuple[ast.AST, ...]
    target: Fragment


@dataclass(frozen=True)
class Register:
    name: str
    fragment: Fragment
    dtype: str
    replicated: bool = False


class _Decline(Exception):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise _Decline


def _name(node: ast.AST) -> str:
    if not isinstance(node, ast.Name):
        raise _Decline
    return node.id


def _text(node: ast.AST) -> str:
    return ast.unparse(node)


def _has_opaque(statements: Sequence[ast.AST]) -> bool:
    return any(
        isinstance(node, ast.Call)
        and _text(node.func) == "_cute_inline_asm_elementwise"
        for statement in statements
        for node in ast.walk(statement)
    )


_CloneT = TypeVar("_CloneT")


def _clone(value: _CloneT) -> _CloneT:
    if isinstance(value, list):
        return cast("_CloneT", [_clone(item) for item in value])
    if isinstance(value, ast.AST):
        fields = {name: _clone(item) for name, item in ast.iter_fields(value)}
        if isinstance(value, ExtendedAST):
            return cast("_CloneT", value.new(fields))
        return cast("_CloneT", ast.copy_location(type(value)(**fields), value))
    return value


def _statements(text: str) -> list[ast.stmt]:
    return ast.parse(text).body


class _Reads(ast.NodeTransformer):
    """Replace only exact current storage generations at their proved owner.

    The bounded integer interpreter proves each emitted flattened address for
    every physical coordinate. It does not evaluate tensor data or host loads.
    Unknown coordinates decline; masked accesses retain their original guards.
    """

    def __init__(
        self,
        registers: dict[str, Register],
        thread: str,
        lane: str,
        slot: str,
        capacity: int,
        variables: dict[str, Callable[[int], int]],
        shared: frozenset[str],
    ) -> None:
        self.registers = registers
        self.shared = shared
        self.thread = thread
        self.lane = lane
        self.slot = slot
        self.capacity = capacity
        self.variables = variables
        self.definitions: dict[str, ast.expr] = {}

    def integer(
        self, expr: ast.expr, rank: int, seen: frozenset[str] = frozenset()
    ) -> int:
        if isinstance(expr, ast.Constant) and type(expr.value) is int:
            return expr.value
        if isinstance(expr, ast.Name):
            if expr.id in self.variables:
                return self.variables[expr.id](rank)
            _require(expr.id in self.definitions and expr.id not in seen)
            return self.integer(self.definitions[expr.id], rank, seen | {expr.id})
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
            return -self.integer(expr.operand, rank, seen)
        if isinstance(expr, ast.BinOp):
            operations = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.FloorDiv: operator.floordiv,
                ast.Mod: operator.mod,
            }
            operation = next(
                (fn for kind, fn in operations.items() if isinstance(expr.op, kind)),
                None,
            )
            if operation is None:
                raise _Decline
            return operation(
                self.integer(expr.left, rank, seen),
                self.integer(expr.right, rank, seen),
            )
        raise _Decline

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.definitions[node.targets[0].id] = _clone(node.value)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        # A physical storage alias/iterator cannot bypass generation checking.
        _require(node.id not in self.shared)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if not isinstance(node.value, ast.Name) or node.value.id not in self.registers:
            return self.generic_visit(node)
        register = self.registers[node.value.id]
        count = math.prod(register.fragment.shape)
        if register.replicated:
            _require(isinstance(node.ctx, ast.Load))
            _require(
                all(
                    self.integer(node.slice, rank) == rank % 32
                    for rank in range(self.capacity)
                )
            )
            expr = f"{register.dtype}(0)"
            for index in reversed(range(count)):
                expr = (
                    f"{register.name}[{index}] if {self.lane} == {index} else ({expr})"
                )
            return expr_from_string(expr)
        if count == 1:
            _require(
                all(
                    self.integer(node.slice, rank) == 0 for rank in range(self.capacity)
                )
            )
            return ast.Name(id=register.name, ctx=node.ctx)
        _require(
            all(self.integer(node.slice, rank) == rank for rank in range(self.capacity))
        )
        return ast.Subscript(
            value=ast.Name(id=register.name, ctx=ast.Load()),
            slice=ast.Name(id=self.slot, ctx=ast.Load()),
            ctx=node.ctx,
        )


class WarpProducerRecorder:
    """Current emitter provenance; retained references do not alter held/scopes."""

    def __init__(self, compiler: FragmentCompiler, plan: WarpProducerRegion) -> None:
        self.compiler = compiler
        self.plan = plan
        self.nodes = frozenset(plan.nodes)
        self.active = False
        self.steps: list[CopyStep | ScanStep | PublicationStep] = []
        self.suffix_steps: list[CopyStep | ScanStep | PublicationStep] = []
        self.frontier: set[str] | None = None

    def enter(self, node: Node) -> None:
        self.active = node in self.nodes
        if not self.active and self.frontier is None:
            self.frontier = self.compiler.live_buffers()

    def copy(
        self,
        statements: Sequence[ast.AST],
        source: Fragment,
        target: Fragment,
        workers: int,
    ) -> None:
        destination = self.steps if self.active else self.suffix_steps
        destination.append(CopyStep(tuple(statements), source, target, workers))

    def scan(
        self,
        statements: Sequence[ast.AST],
        source: Fragment,
        target: Fragment,
        totals: Fragment,
    ) -> None:
        destination = self.steps if self.active else self.suffix_steps
        destination.append(ScanStep(tuple(statements), source, target, totals))

    def publication(self, statements: Sequence[ast.AST], target: Fragment) -> None:
        destination = self.steps if self.active else self.suffix_steps
        destination.append(PublicationStep(tuple(statements), target))

    def lower(self, body: list[ast.AST]) -> None:
        """Build/validate privately, then replace the complete prefix once."""
        namespace = self.compiler.df.namespace
        prospective = copy(namespace)
        prospective.__dict__ = {
            key: copy(value) for key, value in vars(namespace).items()
        }
        self.compiler.df.namespace = prospective
        try:
            try:
                self._lower(body)
            except (_Decline, ZeroDivisionError):
                raise exc.InvalidConfig(
                    "warp producer prefix has unsupported current ownership or storage reads"
                ) from None
        except BaseException:
            self.compiler.df.namespace = namespace
            raise

    def _lower(self, body: list[ast.AST]) -> None:
        compiler = self.compiler
        _require(bool(self.steps))
        frontier = self.frontier
        if frontier is None:
            raise _Decline
        original = [statement for step in self.steps for statement in step.statements]
        start = next((i for i, stmt in enumerate(body) if stmt is original[0]), -1)
        _require(start >= 0 and body[start : start + len(original)] == original)
        for step in self.steps:
            _require(_text(step.statements[-1]) == "cute.arch.sync_threads()")
        capacity = self.plan.capacity
        slots = (capacity + 31) // 32
        lane = compiler.df.new_var("fragment_producer_lane")
        slot = compiler.df.new_var("fragment_producer_slot")
        registers: dict[str, Register] = {}
        emitted: list[ast.stmt] = []

        def register(fragment: Fragment, *, replicated: bool = False) -> Register:
            _require(fragment.storage is not None and fragment.dtype in DTYPES)
            _require(
                any(
                    name == fragment.storage
                    and dtype == fragment.dtype
                    and size >= math.prod(fragment.shape)
                    for name, dtype, size in compiler.buffers
                )
            )
            count = math.prod(fragment.shape)
            _require(count in (1, capacity) or replicated and count == slots)
            name = compiler.df.new_var("fragment_producer_value")
            result = Register(
                name, fragment, compiler.dtype(fragment.dtype), replicated
            )
            dtype = compiler.dtype(fragment.dtype)
            if count == 1 and not replicated:
                emitted.extend(_statements(f"{name} = {dtype}(0)"))
            else:
                emitted.extend(
                    _statements(
                        f"{name} = cute.make_rmem_tensor(({slots},), {dtype})\n{name}.fill({dtype}(0))"
                    )
                )
            return result

        def reads(
            nodes: Sequence[ast.AST],
            variables: dict[str, Callable[[int], int]],
            mapping: dict[str, Register] | None = None,
        ) -> list[ast.stmt]:
            rewrite = _Reads(
                registers if mapping is None else mapping,
                compiler.thread,
                lane,
                slot,
                capacity,
                {compiler.thread: lambda rank: rank % 32, **variables},
                frozenset(name for name, _dtype, _size in compiler.buffers),
            )
            return [cast("ast.stmt", rewrite.visit(_clone(stmt))) for stmt in nodes]

        def broadcast(value: Register) -> None:
            dtype = compiler.dtype(value.fragment.dtype)
            emitted.extend(
                _statements(
                    f"{value.name} = {dtype}(cute.arch.shuffle_sync({value.name}, 0, mask=0xffffffff, mask_and_clamp=31))"
                )
            )

        for step in self.steps:
            if isinstance(step, ScanStep):
                self._scan(step, emitted, registers, register, reads, lane, slot)
                continue
            if not isinstance(step, CopyStep):
                raise _Decline
            _require(
                len(step.statements) == 2 and isinstance(step.statements[0], ast.For)
            )
            outer = cast("ast.For", step.statements[0])
            _require(isinstance(outer.target, ast.Name) and not outer.orelse)
            size = math.prod(step.target.shape)
            _require(
                step.source.shape == step.target.shape
                and step.source.dtype == step.target.dtype
            )
            # Opaque scalar programs keep the exact original physical thread0.
            # A scalar FX value may otherwise be inlined into a vector recipe.
            if size != 1 or step.workers != 1:
                _require(not _has_opaque(step.statements))
            expected_start = (
                compiler.thread if step.workers == 1 else f"{compiler.thread} // 32"
            )
            _require(
                _text(outer.iter)
                == f"range({expected_start}, {size}, {compiler.threads // step.workers})"
            )
            target = register(step.target)
            mapping = registers | {cast("str", step.target.storage): target}
            if size == 1:
                variables: dict[str, Callable[[int], int]] = {
                    _name(outer.target): lambda _rank: 0
                }
                inside = list(outer.body)
                if step.workers == 32:
                    loops = [
                        node
                        for node in ast.walk(outer)
                        if isinstance(node, ast.For) and node is not outer
                    ]
                    _require(len(loops) == 1)
                    reduction = loops[0]
                    _require(isinstance(reduction.target, ast.Name))
                    _require(
                        _text(reduction.iter)
                        == f"range({compiler.thread} % 32, {capacity}, 32)"
                    )
                    variables[_name(reduction.target)] = lambda rank: rank
                    replacement = _statements(
                        f"for {slot} in cutlass.range_constexpr({slots}):\n    {_name(reduction.target)} = {lane} + 32 * {slot}\n    if {_name(reduction.target)} < {capacity}:\n        pass"
                    )[0]
                    cast("ast.If", cast("ast.For", replacement).body[-1]).body = cast(
                        "list[ast.stmt]", _clone(reduction.body)
                    )
                    _require(reduction in inside)
                    inside[inside.index(reduction)] = replacement
                    emitted.extend(reads(inside, variables, mapping))
                else:
                    _require(step.workers == 1)
                    branch = cast(
                        "ast.If", _statements(f"if {lane} == 0:\n    pass")[0]
                    )
                    branch.body = reads(inside, variables, mapping)
                    emitted.append(branch)
                broadcast(target)
            else:
                _require(size == capacity and step.workers == 1)
                replacement = cast(
                    "ast.For",
                    _statements(
                        f"for {slot} in cutlass.range_constexpr({slots}):\n    {_name(outer.target)} = {lane} + 32 * {slot}\n    if {_name(outer.target)} < {capacity}:\n        pass"
                    )[0],
                )
                cast("ast.If", replacement.body[-1]).body = reads(
                    outer.body, {_name(outer.target): lambda rank: rank}, mapping
                )
                emitted.append(replacement)
            registers[cast("str", step.target.storage)] = target
        _require(frontier <= registers.keys())
        publications: list[ast.stmt] = []
        for name in sorted(frontier):
            value = registers[name]
            _require(math.prod(value.fragment.shape) == 1)
            publications.extend(
                _statements(
                    f"{value.fragment.read(tuple('0' for _ in value.fragment.shape))} = {compiler.dtype(value.fragment.dtype)}({value.name})"
                )
            )
        publication_index = compiler.df.new_var("fragment_producer_publication")
        publication = cast(
            "ast.For",
            _statements(
                f"for {publication_index} in range({compiler.thread}, 1, {compiler.threads}):\n    pass"
            )[0],
        )
        publication.body = publications
        # Define only the scalar frontier outside the warp branch. This keeps
        # the existing elected-owner publication shape, so the established
        # published-scalar pass sees the same suffix epoch without new rules.
        initial = [
            statement_from_string(
                f"{registers[name].name} = {compiler.dtype(registers[name].fragment.dtype)}(0)"
            )
            for name in sorted(frontier)
        ]
        # The existing allocator has already proved subsequent physical reuse.
        # Conservatively check every surviving prefix-generation read until its
        # first suffix publication; unknown shared references cannot disappear.
        pending = set(registers) - frontier
        suffix = body[start + len(original) :]
        for statement in suffix:
            store_bases = {
                id(node.value)
                for node in ast.walk(statement)
                if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
            }
            loads = {
                node.id
                for node in ast.walk(statement)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and id(node) not in store_bases
            }
            _require(not pending.intersection(loads))
            for step in self.suffix_steps:
                if (
                    isinstance(step, (CopyStep, PublicationStep))
                    and statement is step.statements[-1]
                ):
                    pending.discard(step.target.storage)
                elif isinstance(step, ScanStep) and statement is step.statements[1]:
                    # The canonical partial phase initializes the complete
                    # physical result and each logical chunk total.
                    pending.discard(step.target.storage)
                    pending.discard(step.totals.storage)
        outer = cast("ast.If", _statements(f"if {compiler.thread} < 32:\n    pass")[0])
        outer.body = emitted
        replacement = [
            statement_from_string(f"{lane} = {compiler.thread} % 32"),
            *initial,
            outer,
            publication,
            statement_from_string("cute.arch.sync_threads()"),
        ]
        body[start : start + len(original)] = [
            ast.fix_missing_locations(statement) for statement in replacement
        ]

    def _scan(
        self,
        step: ScanStep,
        emitted: list[ast.stmt],
        registers: dict[str, Register],
        register: Callable,
        reads: Callable,
        lane: str,
        slot: str,
    ) -> None:
        compiler = self.compiler
        capacity = self.plan.capacity
        slots = (capacity + 31) // 32
        _require(not _has_opaque(step.statements))
        _require(
            step.source.shape == step.target.shape
            and step.source.dtype == step.target.dtype == step.totals.dtype
        )
        _require(
            len(step.statements) == 4
            and all(isinstance(step.statements[i], ast.For) for i in (0, 2))
        )
        partial, carry = (cast("ast.For", step.statements[i]) for i in (0, 2))
        for loop in (partial, carry):
            _require(
                isinstance(loop.target, ast.Name)
                and _text(loop.iter)
                == f"range({compiler.thread} // 32, {slots}, {compiler.threads // 32})"
            )
        result = register(step.target)
        totals = register(step.totals, replicated=True)
        mapping = registers | {cast("str", step.target.storage): result}
        # This exact emitter's final statement publishes the shuffled total in
        # lane 0. That shuffle value is uniform in every lane of this chunk.
        tail = partial.body[-1]
        _require(
            isinstance(tail, ast.If)
            and len(tail.body) == 1
            and isinstance(tail.body[0], ast.Assign)
        )
        store = cast("ast.Assign", cast("ast.If", tail).body[0])
        _require(
            isinstance(store.targets[0], ast.Subscript)
            and isinstance(store.targets[0].value, ast.Name)
            and store.targets[0].value.id == step.totals.storage
        )
        total_store = ast.Assign(
            targets=[cast("ast.expr", expr_from_string(f"{totals.name}[{slot}]"))],
            value=_clone(store.value),
        )
        cast("ast.Subscript", total_store.targets[0]).ctx = ast.Store()
        inside = reads(
            [*partial.body[:-1], total_store],
            {_name(partial.target): lambda rank: rank // 32},
            mapping,
        )
        loop = cast(
            "ast.For",
            _statements(
                f"for {slot} in cutlass.range_constexpr({slots}):\n    {_name(partial.target)} = {slot}"
            )[0],
        )
        loop.body.extend(inside)
        emitted.append(loop)
        registers[cast("str", step.target.storage)] = result
        registers[cast("str", step.totals.storage)] = totals
        # The canonical carry begins with lane/rank, then the identical padded
        # summary tree. Keep rank and chunk-dependent carry in the slot loop.
        _require(
            len(carry.body) >= 4
            and all(isinstance(n, ast.Assign) for n in carry.body[:3])
        )
        split = next(
            (
                i
                for i, n in enumerate(carry.body)
                if i > 2
                and isinstance(n, ast.Assign)
                and isinstance(n.value, ast.Call)
                and _text(n.value.func) == "cute.arch.shuffle_sync"
            ),
            -1,
        )
        _require(split > 2)
        summary = [carry.body[0], *carry.body[2:split]]
        _require(
            not any(
                isinstance(n, ast.Name) and n.id == _name(carry.target)
                for s in summary
                for n in ast.walk(s)
            )
        )
        emitted.extend(reads(summary, {}))
        tail = [carry.body[1], *carry.body[split:]]
        inside = reads(
            tail,
            {
                _name(carry.target): lambda rank: rank // 32,
                _name(cast("ast.Assign", carry.body[0]).targets[0]): lambda rank: (
                    rank % 32
                ),
            },
        )
        loop = cast(
            "ast.For",
            _statements(
                f"for {slot} in cutlass.range_constexpr({slots}):\n    {_name(carry.target)} = {slot}"
            )[0],
        )
        loop.body.extend(inside)
        emitted.append(loop)
