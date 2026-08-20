"""Fuse triples of identical CuTe shared-memory sum reductions."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import os

from ..ast_extension import create


@dataclass
class _ReductionGroup:
    lane_stmts: list[ast.Assign]
    result_name: str
    input_value: ast.expr
    identity: ast.expr
    keywords: list[ast.keyword]
    cast_stmt: ast.Assign


def _simple_assign(stmt: ast.stmt) -> tuple[str, ast.expr] | None:
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    target = stmt.targets[0]
    if not isinstance(target, ast.Name):
        return None
    return target.id, stmt.value


def _match_group(body: list[ast.stmt], start: int) -> _ReductionGroup | None:
    if start + 5 > len(body):
        return None
    assigns = [_simple_assign(stmt) for stmt in body[start : start + 5]]
    if any(item is None for item in assigns):
        return None
    lane, lane_in_group, lane_mod, result, cast_result = assigns
    assert lane is not None
    assert lane_in_group is not None
    assert lane_mod is not None
    assert result is not None
    assert cast_result is not None
    result_name, call = result
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_cute_grouped_reduce_shared_two_stage"
        and len(call.args) == 6
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == "sum"
    ):
        return None
    expected_names = [lane[0], lane_in_group[0], lane_mod[0]]
    if not all(
        isinstance(arg, ast.Name) and arg.id == expected
        for arg, expected in zip(call.args[3:], expected_names, strict=True)
    ):
        return None
    cast_call = cast_result[1]
    if not (
        isinstance(cast_call, ast.Call)
        and len(cast_call.args) == 1
        and isinstance(cast_call.args[0], ast.Name)
        and cast_call.args[0].id == result_name
    ):
        return None
    return _ReductionGroup(
        lane_stmts=[body[start], body[start + 1], body[start + 2]],  # type: ignore[list-item]
        result_name=result_name,
        input_value=call.args[0],
        identity=call.args[2],
        keywords=call.keywords,
        cast_stmt=body[start + 4],  # type: ignore[arg-type]
    )


def _normalized_lane_setup(group: _ReductionGroup) -> tuple[str, str, str]:
    mapping = {
        group.lane_stmts[0].targets[0].id: "lane",  # type: ignore[union-attr]
        group.lane_stmts[1].targets[0].id: "lane_in_group",  # type: ignore[union-attr]
        group.lane_stmts[2].targets[0].id: "lane_mod",  # type: ignore[union-attr]
    }

    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in mapping:
                return ast.copy_location(
                    ast.Name(id=mapping[node.id], ctx=node.ctx), node
                )
            return node

    def normalize(stmt: ast.Assign) -> str:
        return ast.dump(
            Normalize().visit(
                ast.fix_missing_locations(
                    ast.parse(ast.unparse(stmt.value), mode="eval").body
                )
            )
        )

    lane, lane_in_group, lane_mod = group.lane_stmts
    return normalize(lane), normalize(lane_in_group), normalize(lane_mod)


def _compatible(groups: list[_ReductionGroup]) -> bool:
    first = groups[0]
    setup = _normalized_lane_setup(first)
    identity = ast.dump(first.identity)
    keywords = [(kw.arg, ast.dump(kw.value)) for kw in first.keywords]
    return all(
        _normalized_lane_setup(group) == setup
        and ast.dump(group.identity) == identity
        and [(kw.arg, ast.dump(kw.value)) for kw in group.keywords] == keywords
        for group in groups[1:]
    )


def _fuse_body(body: list[ast.stmt], *, warp0_epilogue: bool = False) -> list[ast.stmt]:
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.For, ast.If, ast.With)):
            stmt.body = _fuse_body(stmt.body, warp0_epilogue=warp0_epilogue)
            if isinstance(stmt, (ast.For, ast.If)):
                stmt.orelse = _fuse_body(stmt.orelse, warp0_epilogue=warp0_epilogue)

    result: list[ast.stmt] = []
    index = 0
    while index < len(body):
        groups = [_match_group(body, index + 5 * offset) for offset in range(3)]
        if all(group is not None for group in groups):
            matched = [group for group in groups if group is not None]
            if _compatible(matched):
                first = matched[0]
                lane_names = [
                    stmt.targets[0].id  # type: ignore[union-attr]
                    for stmt in first.lane_stmts
                ]
                call = create(
                    ast.Call,
                    func=create(
                        ast.Name,
                        id="_cute_grouped_reduce_shared_two_stage_sum3",
                        ctx=ast.Load(),
                    ),
                    args=[
                        *(group.input_value for group in matched),
                        first.identity,
                        *(
                            create(ast.Name, id=name, ctx=ast.Load())
                            for name in lane_names
                        ),
                    ],
                    keywords=first.keywords,
                )
                tuple_assign = create(
                    ast.Assign,
                    targets=[
                        create(
                            ast.Tuple,
                            elts=[
                                create(ast.Name, id=group.result_name, ctx=ast.Store())
                                for group in matched
                            ],
                            ctx=ast.Store(),
                        )
                    ],
                    value=call,
                )
                result.extend(first.lane_stmts)
                result.append(tuple_assign)
                casts = [group.cast_stmt for group in matched]
                if warp0_epilogue or os.environ.get(
                    "HELION_CUTE_PACKED_BF16X2_WARP0_EPILOGUE"
                ):
                    result.append(
                        create(
                            ast.If,
                            test=create(
                                ast.Compare,
                                left=create(
                                    ast.Name,
                                    id=lane_names[1],
                                    ctx=ast.Load(),
                                ),
                                ops=[create(ast.Lt)],
                                comparators=[create(ast.Constant, value=32)],
                            ),
                            body=[*casts, *body[index + 15 :]],
                            orelse=[],
                        )
                    )
                    return result
                result.extend(casts)
                index += 15
                continue
        result.append(body[index])
        index += 1
    return result


def fuse_shared_sum_reductions(
    body: list[ast.stmt], *, warp0_epilogue: bool = False
) -> list[ast.stmt]:
    """Fuse three consecutive compatible two-stage sum reductions."""
    return _fuse_body(body, warp0_epilogue=warp0_epilogue)
