"""Independent FP32 sums across a cached rolled vector traversal."""

from __future__ import annotations

import ast

from ..ast_read_writes import ReadWrites
from .vector_reduction_packets import _clone
from .vector_reduction_packets import _float_zero
from .vector_reduction_packets import _Substitute


def split_rolled_vector_sums(body: list[ast.stmt]) -> list[ast.stmt]:
    from ..tile_strategy import _is_proven_relocatable_assignment

    used = {
        node.id
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    reads = ReadWrites.from_ast(ast.Module(body=body, type_ignores=[])).reads
    counter = 0

    def fresh(kind: str) -> str:
        nonlocal counter
        while (name := f"_cute_rolled_{kind}_{counter}") in used:
            counter += 1
        counter += 1
        used.add(name)
        return name

    def split(init: ast.stmt, outer: ast.For) -> list[ast.stmt] | None:
        if not (
            isinstance(init, ast.Assign)
            and len(init.targets) == 1
            and isinstance(init.targets[0], ast.Name)
            and _float_zero(init.value)
            and not outer.orelse
        ):
            return None
        accumulator = init.targets[0].id
        updates: list[ast.Assign] = []
        leaves: list[ast.For] = []

        def visit(statements: list[ast.stmt], owner: ast.For) -> bool:
            for statement in statements:
                if isinstance(statement, ast.For):
                    if statement.orelse or not visit(statement.body, statement):
                        return False
                elif (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and _is_proven_relocatable_assignment(statement, allow_load=True)
                ):
                    if statement.targets[0].id == accumulator:
                        value = statement.value
                        if not (
                            isinstance(value, ast.BinOp)
                            and isinstance(value.op, ast.Add)
                            and isinstance(value.left, ast.Name)
                            and value.left.id == accumulator
                            and accumulator
                            not in ReadWrites.from_ast(value.right).reads
                            and statement is owner.body[-1]
                        ):
                            return False
                        updates.append(statement)
                        leaves.append(owner)
                    elif (
                        statement.targets[0].id
                        in ReadWrites.from_ast(statement.value).reads
                    ):
                        return False
                elif not isinstance(statement, ast.Pass):
                    return False
            return True

        if not visit(outer.body, outer) or len(updates) != 1:
            return None
        inner = leaves[0]
        iterator = inner.iter
        if not (
            inner is not outer
            and isinstance(inner.target, ast.Name)
            and isinstance(iterator, ast.Call)
            and ast.unparse(iterator.func) == "cutlass.range_constexpr"
            and len(iterator.args) == 1
            and isinstance(iterator.args[0], ast.Constant)
            and iterator.args[0].value in (2, 4, 8)
        ):
            return None
        width = iterator.args[0].value
        inner_reads = ReadWrites.from_ast(inner).reads[inner.target.id]
        if reads[inner.target.id] != inner_reads:
            return None
        if ReadWrites.from_ast(outer).reads[accumulator] != 1:
            return None
        locals_ = {
            statement.targets[0].id
            for statement in inner.body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id != accumulator
        }
        assigned: set[str] = set()
        for statement in inner.body:
            if not isinstance(statement, ast.Assign):
                return None
            if (set(ReadWrites.from_ast(statement.value).reads) & locals_) - assigned:
                return None
            assigned.update(
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            )
        outside = [
            node
            for node in ast.walk(outer)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in locals_
        ]
        internal = {
            id(node)
            for node in ast.walk(inner)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        if any(id(node) not in internal for node in outside):
            return None
        partials = [fresh("sum") for _ in range(width)]
        replacement: list[ast.stmt] = []
        for lane, partial in enumerate(partials):
            substitute = _Substitute(
                {
                    inner.target.id: ast.Constant(lane),
                    accumulator: ast.Name(id=partial, ctx=ast.Load()),
                }
            )
            replacement.extend(
                substitute.visit(_clone(statement)) for statement in inner.body
            )

        class Replace(ast.NodeTransformer):
            def visit_For(self, node: ast.For) -> ast.AST | list[ast.stmt]:
                if node is inner:
                    return replacement
                return self.generic_visit(node)

        # Replacements are built from clones; the source loop owns the same
        # nodes until this successful proof replaces the innermost traversal.
        changed = Replace().visit(outer)
        assert isinstance(changed, ast.For)
        result = [
            ast.parse(f"{partial} = cutlass.Float32(0)").body[0] for partial in partials
        ]
        result.append(changed)
        while len(partials) > 1:
            half = len(partials) // 2
            joined = []
            for index in range(half):
                name = fresh("join")
                result.append(
                    ast.parse(
                        f"{name} = {partials[index]} + {partials[index + half]}"
                    ).body[0]
                )
                joined.append(name)
            partials = joined
        result.append(ast.parse(f"{accumulator} = {partials[0]}").body[0])
        return result

    result: list[ast.stmt] = []
    for statement in body:
        if isinstance(statement, ast.For) and result:
            replacement = split(result[-1], statement)
            if replacement is not None:
                result.pop()
                result.extend(replacement)
                continue
        result.append(statement)
    return result
