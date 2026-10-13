"""Pack already independent row programs without changing their scalar recipes.

The caller must supply an existing RowParallelPlan batch after its original
input-completion checks. This helper grants no movement or alias permission.
Only the common register-only suffix is shared. Initializers and publications
remain per-group, and every participating physical warp is complete.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from dataclasses import fields
from typing import TYPE_CHECKING
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from .chained_execution import ChainedExecution
    from .chained_row_registers import RowRegisterIsland


@dataclass(frozen=True)
class RowProgram:
    rows: int
    columns: int
    slots: int
    row: str
    column: str
    registers: str
    output: str
    warp: int
    source: str


@dataclass(frozen=True)
class RowPackingProof:
    """Bind the original row programs to one exact subgroup transformation.

    This grants neither earlier readiness nor a different collective scope.
    The enclosing original RowBatchCompletion still owns the CTA publication.
    """

    islands: tuple[RowRegisterIsland, ...]
    programs: tuple[RowProgram, ...]
    execution: ChainedExecution
    names: tuple[tuple[str, str], ...]
    source: str
    selection: tuple[object, ...]

    def facts(self) -> tuple[object, ...]:
        from .chained_row_registers import _record

        return (
            tuple(_record(island) for island in self.islands),
            tuple(tuple(getattr(p, f.name) for f in fields(p)) for p in self.programs),
            self.execution,
            self.names,
            self.source,
        )

    def check(
        self,
        islands: tuple[RowRegisterIsland, ...],
        execution: ChainedExecution,
        phase: tuple[str, ...],
    ) -> None:
        from .chained_matmul import _UnsupportedChain

        if (
            len(islands) != len(self.islands)
            or any(a is not b for a, b in zip(islands, self.islands, strict=True))
            or execution != self.execution
            or self.facts() != self.selection
            or phase != (*self.source.splitlines(), execution.sync)
        ):
            raise _UnsupportedChain("packed row ownership or emitted phase changed")
        names = iter(self.names)

        def replay(preferred: str) -> str:
            pair = next(names, None)
            if pair is None or pair[0] != preferred:
                raise _UnsupportedChain("packed row local name binding changed")
            return pair[1]

        result = pack(
            list(self.programs), execution.thread, execution.threads, new_var=replay
        )
        if result != self.source or next(names, None) is not None:
            raise _UnsupportedChain("packed row scalar program changed")

    def publication_names(
        self,
        island: RowRegisterIsland,
        execution: ChainedExecution,
        phase: tuple[str, ...],
    ) -> tuple[str, str, str]:
        """Return only the scalar renaming replayed by this exact packed phase.

        Output ownership stays with the selected original island. Packing only
        substitutes the first program's row, column and register names inside
        each original group's publication; no address or cast is inferred here.
        """
        from .chained_matmul import _UnsupportedChain

        self.check(self.islands, execution, phase)
        matches = [i for i, item in enumerate(self.islands) if item is island]
        if len(matches) != 1 or len(self.programs) != len(self.islands):
            raise _UnsupportedChain("packed publication island identity changed")
        program = self.programs[matches[0]]
        if (
            program.output != island.name
            or (program.rows, program.columns) != island.shape
            or program.slots != island.register_slots
        ):
            raise _UnsupportedChain("packed publication original owner changed")
        first = self.programs[0]
        return first.row, first.column, first.registers


def names(tree: ast.AST, context: type[ast.expr_context]) -> set[str]:
    return {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, context)
    }


class Rename(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(
            ast.Name(id=self.mapping.get(node.id, node.id), ctx=node.ctx), node
        )


def expression(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


class FreshNames:
    def __init__(
        self, reserved: set[str], new_var: Callable[[str], str] | None = None
    ) -> None:
        self.reserved = reserved
        self.allocated: dict[str, str] = {}
        self.new_var = new_var

    def get(self, preferred: str) -> str:
        if preferred not in self.allocated:
            if self.new_var is None:
                name = preferred
                suffix = 0
                while name in self.reserved:
                    suffix += 1
                    name = f"{preferred}_{suffix}"
            else:
                name = self.new_var(preferred)
                while name in self.reserved:
                    name = self.new_var(preferred)
            self.reserved.add(name)
            self.allocated[preferred] = name
        return self.allocated[preferred]


def canonical(
    statement: ast.stmt,
    program: RowProgram,
    first: RowProgram,
    index: int,
    fresh: FreshNames,
) -> ast.stmt:
    mapping = {
        program.row: first.row,
        program.column: first.column,
        program.registers: first.registers,
    }
    for node in ast.walk(statement):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id not in mapping
        ):
            mapping[node.id] = fresh.get(
                f"{first.registers}_event_{index}_local_{len(mapping)}"
            )
    return cast("ast.stmt", Rename(mapping).visit(copy.deepcopy(statement)))


def suffix_is_register_only(
    statements: list[ast.stmt], program: RowProgram, thread: str
) -> bool:
    module = ast.Module(body=statements, type_ignores=[])
    if thread in names(module, ast.Load):
        return False
    allowed = {
        "cutlass.Float32",
        "cutlass.BFloat16",
        "cutlass.Int32",
        "cutlass.range_constexpr",
        "cute.arch.shuffle_sync",
        "operator.eq",
        "operator.gt",
        "operator.lt",
    }
    shuffles = 0
    for node in ast.walk(module):
        if isinstance(node, ast.If) and any(
            isinstance(child, ast.Call)
            and ast.unparse(child.func) == "cute.arch.shuffle_sync"
            for child in ast.walk(node)
        ):
            return False
        if isinstance(node, ast.Subscript):
            if (
                not isinstance(node.value, ast.Name)
                or node.value.id != program.registers
            ):
                return False
        if isinstance(node, ast.Call):
            function = ast.unparse(node.func)
            if function not in allowed:
                return False
            if function == "cute.arch.shuffle_sync":
                if (
                    len(node.args) != 2
                    or not isinstance(node.args[1], ast.Constant)
                    or type(node.args[1].value) is not int
                    or not 0 <= node.args[1].value < program.rows
                ):
                    return False
                if {k.arg: ast.unparse(k.value) for k in node.keywords} != {
                    "mask": "-1",
                    "mask_and_clamp": "31",
                }:
                    return False
                shuffles += 1
    return shuffles > 0


def pack(
    programs: list[RowProgram],
    thread: str,
    threads: int,
    *,
    new_var: Callable[[str], str] | None = None,
) -> str | None:
    if len(programs) < 2:
        return None
    first = programs[0]
    width = first.rows
    active = len(programs) * width
    if width not in (2, 4, 8, 16) or active % 32 or active > threads:
        return None
    if any(
        (p.rows, p.columns, p.slots) != (first.rows, first.columns, first.slots)
        for p in programs
    ):
        return None
    bodies: list[list[ast.stmt]] = []
    for program in programs:
        parsed = ast.parse(program.source).body
        if len(parsed) != 1 or not isinstance(parsed[0], ast.If) or parsed[0].orelse:
            return None
        guard = parsed[0]
        if ast.dump(guard.test) != ast.dump(
            expression(f"{thread} // 32 == {program.warp}")
        ):
            return None
        body = guard.body
        if (
            len(body) < 5
            or not isinstance(body[0], ast.Assign)
            or not isinstance(body[1], ast.Assign)
        ):
            return None
        if ast.unparse(body[0]) != f"{program.row} = {thread} - {32 * program.warp}":
            return None
        if (
            ast.unparse(body[1])
            != f"{program.registers} = cute.make_rmem_tensor(({program.slots},), cutlass.Float32)"
        ):
            return None
        # Every event-local temporary must die within its original statement.
        # Row, column and the explicitly allocated payload are the only carried
        # local state; external sources remain unchanged in each initializer.
        for index, statement in enumerate(body[2:]):
            local = names(statement, ast.Store) - {program.column}
            other = ast.Module(
                body=[s for i, s in enumerate(body[2:]) if i != index], type_ignores=[]
            )
            if local & names(other, ast.Load):
                return None
        bodies.append(body)
    if len({len(body) for body in bodies}) != 1:
        return None
    fresh = FreshNames(
        {thread}
        | {
            node.id
            for body in bodies
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name)
        },
        new_var,
    )
    normalized = [
        [canonical(s, p, first, i, fresh) for i, s in enumerate(body)]
        for p, body in zip(programs, bodies, strict=True)
    ]
    split = len(bodies[0]) - 1
    while split > 2 and all(
        ast.dump(body[split - 1]) == ast.dump(normalized[0][split - 1])
        for body in normalized[1:]
    ):
        split -= 1
    suffix = normalized[0][split:-1]
    if split < 3 or not suffix_is_register_only(suffix, first, thread):
        return None
    for program, body in zip(programs, bodies, strict=True):
        prefix = ast.Module(body=body[2:split], type_ignores=[])
        if thread in names(prefix, ast.Load):
            return None
        for node in ast.walk(prefix):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("shuffle_sync", "sync_threads", "store")
            ):
                return None
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Store)
                and (
                    not isinstance(node.value, ast.Name)
                    or node.value.id != program.registers
                )
            ):
                return None
        # The original emitter publishes exactly its final register image.
        publication = body[-1]
        expected = ast.parse(
            f"if {program.row} < {width}:\n    for {program.column} in cutlass.range_constexpr({program.columns}):\n        {program.output}[{program.row}, {program.column}] = {program.registers}[0 + {program.column}]"
        ).body[0]
        expected = cast("ast.If", expected)
        expected_loop = cast("ast.For", expected.body[0])
        expected_store = cast("ast.Assign", expected_loop.body[0])
        if (
            not isinstance(publication, ast.If)
            or ast.dump(publication.test) != ast.dump(expected.test)
            or publication.orelse
            or len(publication.body) != 1
            or not isinstance(publication.body[0], ast.For)
        ):
            return None
        loop = publication.body[0]
        if (
            ast.dump(loop.target) != ast.dump(expected_loop.target)
            or ast.dump(loop.iter) != ast.dump(expected_loop.iter)
            or loop.orelse
            or len(loop.body) != 1
            or not isinstance(loop.body[0], ast.Assign)
        ):
            return None
        assignment = loop.body[0]
        if (
            len(assignment.targets) != 1
            or ast.dump(assignment.targets[0]) != ast.dump(expected_store.targets[0])
            or not isinstance(assignment.value, ast.Subscript)
            or not isinstance(assignment.value.value, ast.Name)
            or assignment.value.value.id != program.registers
        ):
            return None
    segment = ((32 - width) << 8) | 31
    for node in ast.walk(ast.Module(body=suffix, type_ignores=[])):
        if (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "cute.arch.shuffle_sync"
        ):
            next(
                k for k in node.keywords if k.arg == "mask_and_clamp"
            ).value = ast.Constant(value=segment)
    group = fresh.get(f"{first.registers}_group")
    merged = ast.parse(
        f"{first.row} = {thread} % {width}\n{group} = {thread} // {width}"
    ).body
    merged.append(copy.deepcopy(bodies[0][1]))
    for index, (program, body) in enumerate(zip(programs, bodies, strict=True)):
        mapping = {
            program.row: first.row,
            program.column: first.column,
            program.registers: first.registers,
        }
        prefix = [Rename(mapping).visit(copy.deepcopy(s)) for s in body[2:split]]
        merged.append(
            ast.If(test=expression(f"{group} == {index}"), body=prefix, orelse=[])
        )
    merged.extend(suffix)
    for index, (program, body) in enumerate(zip(programs, bodies, strict=True)):
        mapping = {
            program.row: first.row,
            program.column: first.column,
            program.registers: first.registers,
        }
        publication = Rename(mapping).visit(copy.deepcopy(body[-1]))
        merged.append(
            ast.If(
                test=expression(f"{group} == {index}"), body=[publication], orelse=[]
            )
        )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.If(test=expression(f"{thread} < {active}"), body=merged, orelse=[])
            ],
            type_ignores=[],
        )
    )
    return ast.unparse(module) + "\n"


def pack_emitted_batch(
    islands: list[RowRegisterIsland],
    programs: list[list[str]],
    execution: ChainedExecution,
    new_var: Callable[[str], str],
    *,
    receipts: list[RowPackingProof] | None = None,
    original_warps: tuple[int, ...] | None = None,
) -> str | None:
    """Called only after the existing row batch's joined input checks.

    The original execution proof supplies a contiguous 1D thread range.
    The caller verifies the actual ordinary-root launch before this adapter.
    A temporary thread binding lets the existing integer-loop bounds proof
    normalize equivalent masks before comparing typed scalar statements.
    The temporary binding is never emitted and grants no grid or memory fact.
    """
    from .proven_loop_bounds import simplify_proven_loop_bounds

    owners = tuple(range(len(islands))) if original_warps is None else original_warps
    if len(owners) != len(islands) or any(
        type(warp) is not int or not 0 <= warp < execution.threads // 32
        for warp in owners
    ):
        return None
    rows = []
    for warp, island, lines in zip(owners, islands, programs, strict=True):
        tree = ast.parse("\n".join(lines))
        prefix = ast.parse(f"{execution.thread} = cute.arch.thread_idx()[0]").body
        simplified = simplify_proven_loop_bounds(
            prefix + tree.body,
            enabled=True,
            thread_block_dims=(execution.threads, 1, 1),
            constexpr_values={},
        )
        guards = [node for node in simplified if isinstance(node, ast.If)]
        if len(guards) != 1:
            return None
        rows.append(
            RowProgram(
                island.shape[0],
                island.shape[1],
                island.register_slots,
                f"{island.name}_row",
                f"{island.name}_column",
                f"{island.name}_values",
                island.name,
                warp,
                ast.unparse(guards[0]),
            )
        )
    allocated: list[tuple[str, str]] = []

    def allocate(preferred: str) -> str:
        name = new_var(preferred)
        allocated.append((preferred, name))
        return name

    result = pack(rows, execution.thread, execution.threads, new_var=allocate)
    if result is not None and receipts is not None:
        proof = RowPackingProof(
            tuple(islands), tuple(rows), execution, tuple(allocated), result, ()
        )
        proof = RowPackingProof(
            proof.islands,
            proof.programs,
            proof.execution,
            proof.names,
            proof.source,
            proof.facts(),
        )
        proof.check(tuple(islands), execution, (*result.splitlines(), execution.sync))
        receipts.append(proof)
    return result
