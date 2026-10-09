"""Owner-private scheduling for proved read-only scalar finalizer loops."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import cast

from ...exc import InvalidConfig
from ..ast_extension import ExtendedAST
from ..ast_extension import statement_from_string
from ..device_ir import ForLoopGraphInfo
from ..device_ir import IfGraphInfo
from ..device_ir import control_flow_parent_entries
from .local_atomic import local_atomic_allocations
from .local_atomic import terminal_loop_symbols

if TYPE_CHECKING:
    from torch.fx import Node

    from ..device_ir import GraphInfo
    from .computed_fragment import FragmentCompiler


def _copy_ast(node: ast.AST) -> ast.AST:
    fields = {
        name: _copy_ast(value)
        if isinstance(value, ast.AST)
        else [_copy_ast(item) if isinstance(item, ast.AST) else item for item in value]
        if isinstance(value, list)
        else value
        for name, value in ast.iter_fields(node)
    }
    if isinstance(node, ExtendedAST):
        return cast("ast.AST", node.copy(**fields))
    return ast.copy_location(type(node)(**fields), node)


def private_scalar_loop_nodes(graphs: list[GraphInfo]) -> frozenset[Node]:
    """Use the existing local-finalizer proof; never claim unrelated loops."""
    if not local_atomic_allocations(graphs):
        return frozenset()
    parents = control_flow_parent_entries(graphs)
    by_graph = {info.graph: info for info in graphs}
    result = set()
    for graph_id, (node, _slot) in parents.items():
        # A branch's direct scalar loops have a finalizer as their parent.
        if not isinstance(by_graph[node.graph], IfGraphInfo):
            continue
        info = next(info for info in graphs if info.graph_id == graph_id)
        if type(info) is not ForLoopGraphInfo:
            continue
        try:
            terminal_loop_symbols(node, graphs)
        except InvalidConfig:
            continue
        result.add(node)
    return frozenset(result)


def privatize_scalar_loop(
    compiler: FragmentCompiler, loop: ast.For, outgoing: set[str]
) -> list[ast.stmt]:
    """Verify the emitted scalar ownership, then retain only written slots.

    The caller has proved a uniform, side-effect-free scalar loop. This second
    proof checks its actual physical schedule. Every operation already belongs
    to thread zero. Read-only shared/global loads stay at their original place;
    only carry/snapshot slots become registers. Publish live carries, including
    their incoming values on zero-trip loops, before the uniform exit barrier.
    """

    def reject() -> NoReturn:
        raise InvalidConfig(
            "private scalar loops require a proved thread-zero scalar schedule"
        )

    buffers = {name: compiler.dtype(dtype) for name, dtype, _count in compiler.buffers}
    flat: list[ast.stmt] = []
    for statement in loop.body:
        if (
            isinstance(statement, ast.Expr)
            and ast.unparse(statement.value) == "cute.arch.sync_threads()"
        ):
            continue
        if not isinstance(statement, ast.For) or statement.orelse:
            reject()
        assert isinstance(statement, ast.For)
        it = statement.iter
        if (
            not isinstance(it, ast.Call)
            or not isinstance(it.func, ast.Name)
            or it.func.id != "range"
            or len(it.args) != 3
            or it.keywords
            or ast.unparse(it.args[0]) != compiler.thread
            or ast.unparse(it.args[1]) != "1"
            or ast.unparse(it.args[2]) != str(compiler.threads)
            or not isinstance(statement.target, ast.Name)
        ):
            reject()
        flat.append(
            cast(
                "ast.stmt",
                statement_from_string(f"{ast.unparse(statement.target)} = 0"),
            )
        )
        flat.extend(cast("ast.stmt", _copy_ast(item)) for item in statement.body)
    written: set[str] = set()
    first_reads: set[str] = set()
    casts = set(buffers.values()) | {
        "cutlass.Int32",
        "cutlass.Int64",
        "cutlass.Boolean",
    }
    for statement in flat:
        reads: set[str] = set()
        stores: set[str] = set()
        for parent in ast.walk(statement):
            for child in ast.iter_child_nodes(parent):
                if isinstance(child, ast.Name) and child.id in buffers:
                    if (
                        not isinstance(parent, ast.Subscript)
                        or parent.value is not child
                    ):
                        reject()  # No escaped shared pointer or hidden storage alias.
        for node in ast.walk(statement):
            if isinstance(
                node,
                (
                    ast.For,
                    ast.While,
                    ast.AugAssign,
                    ast.Break,
                    ast.Continue,
                    ast.Return,
                ),
            ):
                reject()
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if not (
                    name in casts | {"_cute_python_mod", "min", "max"}
                    or name
                    in {
                        "operator.add",
                        "operator.sub",
                        "operator.mul",
                        "operator.truediv",
                        "operator.floordiv",
                        "operator.mod",
                        "operator.neg",
                        "operator.pos",
                        "operator.eq",
                        "operator.ne",
                        "operator.gt",
                        "operator.ge",
                        "operator.lt",
                        "operator.le",
                        "operator.and_",
                        "operator.or_",
                        "operator.xor",
                        "operator.not_",
                        "operator.invert",
                        "operator.lshift",
                        "operator.rshift",
                    }
                    or isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load"
                ):
                    reject()
            if isinstance(node, ast.Subscript):
                if (
                    not isinstance(node.value, ast.Name)
                    or node.value.id not in buffers
                    or ast.unparse(node.slice) != "0"
                ):
                    reject()
                (stores if isinstance(node.ctx, ast.Store) else reads).add(
                    node.value.id
                )
        if stores and not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Subscript)
            and len(stores) == 1
        ):
            reject()
        first_reads.update(reads - written)
        written.update(stores)
    if not outgoing or not outgoing <= written or (first_reads & written) - outgoing:
        reject()
    names = {
        name: compiler.df.new_var("fragment_private_scalar") for name in sorted(written)
    }

    class Private(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            if isinstance(node.value, ast.Name) and node.value.id in names:
                return ast.copy_location(
                    ast.Name(
                        names[node.value.id],
                        ast.Store() if isinstance(node.ctx, ast.Store) else ast.Load(),
                    ),
                    node,
                )
            return self.generic_visit(node)

    body: list[ast.stmt] = []
    for name, private in names.items():
        initial = f"{name}[0]" if name in outgoing else f"{buffers[name]}(0)"
        body.append(cast("ast.stmt", statement_from_string(f"{private} = {initial}")))
    changed = _copy_ast(loop)
    assert isinstance(changed, ast.For)
    changed.body = [cast("ast.stmt", Private().visit(stmt)) for stmt in flat]
    body.append(changed)
    for name in sorted(outgoing):
        body.append(
            cast("ast.stmt", statement_from_string(f"{name}[0] = {names[name]}"))
        )
    owner = statement_from_string(f"if {compiler.thread} == 0:\n    pass")
    assert isinstance(owner, ast.If)
    owner.body = body
    return [
        owner,
        cast("ast.stmt", statement_from_string("cute.arch.sync_threads()")),
    ]
