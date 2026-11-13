from __future__ import annotations

import ast
from collections import Counter
from types import SimpleNamespace


def generated_output_address_coverage(
    source: str, output_name: str, blocks: int, threads: int
) -> dict[int, int]:
    """Count every emitted output store for the selected launch geometry."""
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(
            ast.Module(
                body=[n for n in tree.body if isinstance(n, ast.Assign)]
                + function.body,
                type_ignores=[],
            )
        )
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    stores = []

    def visit(
        node: ast.AST,
        loops: tuple[ast.For, ...] = (),
        guards: tuple[ast.expr, ...] = (),
    ) -> None:
        if isinstance(node, ast.For):
            for child in node.body:
                visit(child, (*loops, node), guards)
            return
        if isinstance(node, ast.If):
            for child in node.body:
                visit(child, loops, (*guards, node.test))
            for child in node.orelse:
                visit(
                    child,
                    loops,
                    (*guards, ast.UnaryOp(op=ast.Not(), operand=node.test)),
                )
            return
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "store"
            and any(
                isinstance(n, ast.Name) and n.id == output_name
                for n in ast.walk(node.func.value)
            )
        ):
            stores.append((node.func.value, loops, guards))
        for child in ast.iter_child_nodes(node):
            visit(child, loops, guards)

    visit(function)
    assert len(stores) == 1
    pointer, loops, guards = stores[0]
    assert len(loops) == 1
    loop = loops[0]
    assert isinstance(loop.target, ast.Name)
    result = Counter()
    for block in range(blocks):
        for thread in range(threads):
            env: dict[str, object] = {
                "cutlass": SimpleNamespace(
                    Int32=int, Int64=int, Uint32=int, Uint64=int
                ),
                "cute": SimpleNamespace(
                    arch=SimpleNamespace(
                        thread_idx=lambda thread=thread: (thread, 0, 0),
                        block_idx=lambda block=block: (block, 0, 0),
                    )
                ),
                "range": range,
                output_name: SimpleNamespace(
                    iterator=0, layout=SimpleNamespace(stride=(1,))
                ),
            }

            def value(
                expression: ast.expr, env: dict[str, object] = env
            ) -> int | range:
                for node in ast.walk(expression):
                    if (
                        isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load)
                        and node.id not in env
                    ):
                        assert node.id in assignments, node.id
                        env[node.id] = value(assignments[node.id])
                return eval(
                    compile(
                        ast.fix_missing_locations(ast.Expression(expression)),
                        "<generated-address>",
                        "eval",
                    ),
                    {"__builtins__": {}},
                    env,
                )

            iterations = value(loop.iter)
            assert isinstance(iterations, range)
            for position in iterations:
                # Local scalar assignments must be re-evaluated at every loop
                # iteration; retain only the external primitive environment.
                for name in assignments:
                    env.pop(name, None)
                env[loop.target.id] = position
                if all(value(guard) for guard in guards):
                    address = value(pointer)
                    assert isinstance(address, int)
                    result[address] += 1
    return dict(sorted(result.items()))
