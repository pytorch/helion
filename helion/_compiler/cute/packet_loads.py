"""Packetize readonly host loads while retaining dense fragment publication."""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..ast_extension import statement_from_string
from .memory_ops import tensor_has_specialized_base_alignment
from .published_scalars import _integer_slot
from .published_scalars import immutable_publications

if TYPE_CHECKING:
    from collections.abc import Callable

    from .computed_fragment import Fragment
    from .computed_fragment import FragmentCompiler
    from .computed_fragment import HostTensor

KEY = "cute_fragment_packet_loads"
_PURE = frozenset(
    [
        f"cutlass.{name}"
        for name in ("Int32", "Int64", "Uint32", "Uint64", "Float32", "Boolean")
    ]
    + [
        f"operator.{name}"
        for name in (
            "add",
            "sub",
            "mul",
            "floordiv",
            "mod",
            "lt",
            "le",
            "gt",
            "ge",
            "eq",
            "ne",
            "and_",
            "or_",
            "xor",
            "lshift",
            "rshift",
        )
    ]
    + ["_cute_python_mod", "min", "max"]
)


def _load_assignment(
    body: list[ast.AST], tensor_name: str, shared_slots: frozenset[str] = frozenset()
) -> tuple[ast.Assign, ast.expr] | None:
    """Prove that address preparation contains only pure scalar operations.

    Only explicitly proved published shared slots may augment the coordinate
    recipe. Gathers, atomics and unknown calls decline. No expression is moved
    out of its original conditional scope.
    """
    loads = []
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.stmt) and not isinstance(
                node, (ast.Assign, ast.If, ast.Pass)
            ):
                return None
            if isinstance(node, ast.Subscript):
                if not (
                    isinstance(node.ctx, ast.Load)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in shared_slots
                    and _integer_slot(node.slice) == 0
                ):
                    return None
            if isinstance(node, ast.Assign) and (
                len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name)
            ):
                return None
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load"
                    and not node.args
                    and not node.keywords
                ):
                    loads.append(node)
                elif ast.unparse(node.func) not in _PURE:
                    return None
    if len(loads) != 1:
        return None
    load = loads[0]
    assert isinstance(load.func, ast.Attribute)
    pointer = load.func.value
    if not (
        isinstance(pointer, ast.BinOp)
        and isinstance(pointer.op, ast.Add)
        and ast.unparse(pointer.left) == f"{tensor_name}.iterator"
    ):
        return None
    parents = {
        child: parent
        for statement in body
        for parent in ast.walk(statement)
        for child in ast.iter_child_nodes(parent)
    }
    node: ast.AST = load
    while node in parents and isinstance(parents[node], ast.Call):
        node = parents[node]
    parent = parents.get(node)
    if not isinstance(parent, ast.Assign) or parent.value is not node:
        return None
    return parent, pointer.right


def _published_dependencies(
    compiler: FragmentCompiler, dependencies: tuple[Fragment, ...]
) -> frozenset[str]:
    """Known shared scalar epochs; every transitive allocation stays held.

    Register-owned snapshots cannot be read by the packet's different physical
    owner. Unknown storage and publications outside this lexical body decline.
    This proof does not move a read or require a constant effective address.
    """
    seen: set[int] = set()
    pending = list(dependencies)
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if value.resident and value.storage is None and not value.dependencies:
            return frozenset()
        pending.extend(value.dependencies)
    buffers = compiler.referenced_buffers(dependencies)
    allocated = {name for name, _dtype, _capacity in compiler.buffers}
    if not buffers <= allocated & compiler.live_buffers():
        return frozenset()
    if buffers & compiler.pending_local_atomics:
        return frozenset()
    published = immutable_publications(
        compiler.cg.statements_stack[-1], buffers, compiler.thread, compiler.threads
    )
    return frozenset(published)


class _CaptureAddress(ast.NodeTransformer):
    def __init__(
        self, assignment: ast.Assign, address: str, predicate: str, offset: ast.expr
    ) -> None:
        self.assignment = assignment
        self.address = address
        self.predicate = predicate
        self.offset = offset

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        if node is self.assignment:
            return ast.parse(
                f"{self.address} = cutlass.Int64({ast.unparse(self.offset)})\n"
                f"{self.predicate} = True"
            ).body
        return node


def materialize_packet_load(
    compiler: FragmentCompiler,
    tensor: HostTensor,
    value: Fragment,
    load: Callable[[tuple[str, ...]], str],
    dependencies: tuple[Fragment, ...] = (),
) -> Fragment | None:
    """Publish the same scalar values at the same dense shared coordinates.

    The caller proves the host allocation is readonly for the complete root.
    Each packet owns four consecutive *logical* slots, independent of pointer
    strides. Original address/mask code runs first; only four valid consecutive
    aligned effective addresses select LDG.128. Every other packet scalarizes.
    The final existing CTA barrier protects all later cross-thread consumers.
    """
    if (
        not value.shape
        or value.dtype != torch.float32
        or not tensor_has_specialized_base_alignment(compiler.env, tensor.value, 16)
    ):
        return None
    count = math.prod(value.shape)
    if count < 4:
        return None
    packet_index = compiler.df.new_var("fragment_packet_index")
    loop = cast(
        "ast.For",
        statement_from_string(
            f"for {packet_index} in range({compiler.thread}, {(count + 3) // 4}, {compiler.threads}):\n    pass"
        ),
    )
    loop.body.clear()
    captures = []
    shared_slots = _published_dependencies(compiler, dependencies)
    shared_reads: set[str] = set()
    for lane in range(4):
        index = f"({packet_index} * 4 + {lane})"
        coords = compiler.coordinates(index, value.shape)
        statements: list[ast.AST] = []
        with compiler.cg.set_statements(statements):
            name = load(coords)
        assignment = _load_assignment(statements, tensor.name, shared_slots)
        if assignment is None:
            return None
        shared_reads.update(
            node.value.id
            for statement in statements
            for node in ast.walk(statement)
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
        )
        captures.append((index, coords, name, statements, *assignment))
    compiler.held.append((value, *dependencies) if shared_reads else value)
    result = compiler.allocate(value)
    assert result.storage not in shared_reads
    addresses = []
    predicates = []
    names = []
    with compiler.cg.set_statements(cast("list[ast.AST]", loop.body)):
        for index, _coords, name, statements, assignment, offset in captures:
            address = compiler.df.new_var("fragment_packet_address")
            predicate = compiler.df.new_var("fragment_packet_valid")
            addresses.append(address)
            predicates.append(predicate)
            names.append(name)
            compiler.emit(f"{address} = cutlass.Int64(0)")
            compiler.emit(f"{predicate} = False")
            compiler.emit(f"{name} = cutlass.Float32(0)")

            captured = []
            transform = _CaptureAddress(assignment, address, predicate, offset)
            for statement in statements:
                rewritten = transform.visit(statement)
                captured.extend(
                    rewritten if isinstance(rewritten, list) else [rewritten]
                )
            if count % 4:
                captured = [
                    ast.If(
                        test=ast.parse(f"{index} < {count}", mode="eval").body,
                        body=cast("list[ast.stmt]", captured),
                        orelse=[],
                    )
                ]
            for statement in captured:
                compiler.cg.add_statement(statement)
        guard = " and ".join(
            [
                *predicates,
                f"{addresses[0]} % 4 == 0",
                *(
                    f"{address} == {addresses[0]} + {lane}"
                    for lane, address in enumerate(addresses[1:], 1)
                ),
            ]
        )
        packet = compiler.df.new_var("fragment_packet_values")
        source = compiler.df.new_var("fragment_packet_source")
        fast = f"{packet} = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Float32)\n{source} = cute.make_tensor(({tensor.name}.iterator + {addresses[0]}).align(16), cute.make_layout((4,)))\ncute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128), {source}, {packet})\n"
        fast += "\n".join(
            f"{name} = cutlass.Float32({packet}[{lane}])"
            for lane, name in enumerate(names)
        )
        slow = "\n".join(
            f"if {valid}:\n    {name} = cutlass.Float32(({tensor.name}.iterator + {address}).load())"
            for valid, name, address in zip(predicates, names, addresses, strict=True)
        )
        branch = cast("ast.If", statement_from_string(f"if {guard}:\n    pass"))
        branch.body = ast.parse(fast).body
        branch.orelse = ast.parse(slow).body
        compiler.cg.add_statement(branch)
        for index, coords, name, *_rest in captures:
            store = f"{result.read(coords)} = {name}"
            compiler.emit(f"if {index} < {count}:\n    {store}" if count % 4 else store)
    compiler.cg.add_statement(loop)
    compiler.synchronize()
    compiler.held.pop()
    return result
