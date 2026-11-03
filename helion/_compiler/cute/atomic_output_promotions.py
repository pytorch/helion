"""Prove private host storage for atomic outputs.

Role: analysis and device-IR pass.  ``fresh_half_atomic_outputs`` promotes half atomic
outputs to FP32; ``private_fresh_host_bindings`` is the same proof for the
exact zero-add elision of ``cute/atomic_ops.py``.
``collect_cute_half_atomic_output_promotions`` finds the promotable outputs
of a traced kernel and ``rewrite_cute_half_atomic_output_allocations`` /
``promote_cute_root_graph_host_tensors`` apply the promotion to the host AST
and the device graphs (``ir_passes.canonicalize_traced_graphs_pass``).
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from ..host_function import HostFunction


def fresh_half_atomic_outputs(
    body: Sequence[ast.stmt], promotions: Mapping[str, torch.dtype]
) -> dict[str, torch.dtype]:
    """Require a fresh allocation whose host references cannot escape as aliases."""
    return {
        name: promotions[name] for name in private_fresh_host_bindings(body, promotions)
    }


def private_fresh_host_bindings(
    body: Sequence[ast.stmt], names: Collection[str]
) -> set[str]:
    """The ``names`` bound exactly once, to a fresh factory allocation, whose
    host references never escape: every later mention is metadata, the
    destination of ``hl.atomic_add``, or the return value.

    A factory is a ``torch.*`` allocation call or a ``Tensor.new_*`` method,
    which allocates afresh whatever its receiver (the receiver only supplies
    the default dtype and device)."""
    factories = {
        f"torch.{name}"
        for name in (
            "empty",
            "empty_like",
            "zeros",
            "zeros_like",
            "ones",
            "ones_like",
            "full",
            "full_like",
        )
    }
    factory_methods = {"new_empty", "new_zeros", "new_ones", "new_full"}
    module = ast.Module(body=list(body), type_ignores=[])
    parents = {
        id(child): parent
        for parent in ast.walk(module)
        for child in ast.iter_child_nodes(parent)
    }
    bindings: dict[str, list[ast.Name]] = {name: [] for name in names}
    escaped: set[str] = set()
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in names
        ):
            parent = parents[id(node)]
            ancestor = parent
            captured = False
            while ancestor is not module:
                if isinstance(
                    ancestor,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
                ):
                    captured = True
                    break
                ancestor = parents[id(ancestor)]
            if captured:
                escaped.add(node.id)
                continue
            if isinstance(parent, ast.Attribute) and parent.attr in {
                "dtype",
                "device",
                "shape",
                "ndim",
            }:
                continue
            if isinstance(parent, ast.Attribute) and parent.attr in {
                "size",
                "stride",
                "numel",
                "element_size",
            }:
                call = parents[id(parent)]
                # A bound method retains its Tensor receiver in __self__.
                # Only an immediate metadata call prevents that receiver from
                # escaping under another name.
                if isinstance(call, ast.Call) and call.func is parent:
                    continue
            if (
                isinstance(parent, ast.Call)
                and ast.unparse(parent.func)
                in {"hl.atomic_add", "helion.language.atomic_add"}
                and parent.args
                and parent.args[0] is node
            ):
                continue
            # Returning the destination after execution does not make its
            # contents available to the device under another host binding.
            while isinstance(parent, (ast.Tuple, ast.List)):
                parent = parents[id(parent)]
            if not isinstance(parent, ast.Return):
                escaped.add(node.id)
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id in bindings
        ):
            bindings[node.id].append(node)
    result: set[str] = set()
    for assignment in ast.walk(module):
        if (
            not isinstance(assignment, ast.Assign)
            or len(assignment.targets) != 1
            or not isinstance(assignment.targets[0], ast.Name)
        ):
            continue
        target = assignment.targets[0]
        value = assignment.value
        if (
            bindings.get(target.id) == [target]
            and target.id not in escaped
            and isinstance(value, ast.Call)
            and (
                ast.unparse(value.func) in factories
                or (
                    isinstance(value.func, ast.Attribute)
                    and value.func.attr in factory_methods
                )
            )
            and all(keyword.arg not in (None, "out") for keyword in value.keywords)
        ):
            result.add(target.id)
    return result


def collect_cute_half_atomic_output_promotions(
    graph_infos: list[GraphInfo],
) -> dict[str, torch.dtype]:
    from ...language import atomic_add
    from ...language._tracing_ops import _host_tensor
    from ..ast_extension import ExtendedAST
    from ..host_function import HostFunction
    from ..type_info import CallableType
    from ..variable_origin import ArgumentOrigin

    promotions: dict[str, torch.dtype] = {}
    host_fn = HostFunction.current()
    host_tensor_nodes: dict[str, list[torch.fx.Node]] = {}
    storage_names: dict[int, set[str]] = {}

    for graph_info in graph_infos:
        for node in graph_info.graph.nodes:
            if node.op == "call_function" and node.target is _host_tensor:
                target_name = node.args[0]
                if isinstance(target_name, str):
                    host_tensor_nodes.setdefault(target_name, []).append(node)
                    value = node.meta.get("val")
                    if isinstance(value, torch.Tensor):
                        storage_names.setdefault(
                            id(value.untyped_storage()), set()
                        ).add(target_name)

    def is_promotable_target(node: torch.fx.Node) -> bool:
        target_val = node.meta.get("val")
        if (
            not isinstance(target_val, torch.Tensor)
            or target_val.dtype != torch.float16
        ):
            return False
        origin = host_fn.tensor_to_origin.get(target_val)
        if origin is None or isinstance(origin, ArgumentOrigin):
            return False
        if not node.users:
            return False
        for user in node.users:
            if user.op != "call_function" or user.target is not atomic_add:
                return False
            if len(user.args) < 3 or user.args[0] is not node or len(user.users) != 0:
                return False
            value_node = user.args[2]
            if not isinstance(value_node, torch.fx.Node):
                return False
            value_val = value_node.meta.get("val")
            if not isinstance(value_val, torch.Tensor) or value_val.dtype not in (
                torch.float16,
                torch.float32,
            ):
                return False
        return True

    for target_name, nodes in host_tensor_nodes.items():
        # A second host name can read a view of this allocation without being a
        # user of the atomic destination node. Check the original storage here:
        # promoting FakeTensors below creates distinct float32 storage and can
        # no longer establish this alias relationship.
        if all(
            is_promotable_target(node)
            and storage_names[id(node.meta["val"].untyped_storage())] == {target_name}
            for node in nodes
        ):
            promotions[target_name] = torch.float16

    # Host indexing and control-flow type joins can construct fresh FakeTensor
    # metadata even for runtime views. Storage identity therefore supplements
    # the host reference proof; it cannot replace that proof.
    factory_functions = (
        torch.empty,
        torch.empty_like,
        torch.full,
        torch.full_like,
        torch.ones,
        torch.ones_like,
        torch.zeros,
        torch.zeros_like,
    )
    proven_factories = set()
    for assignment in ast.walk(ast.Module(body=host_fn.body, type_ignores=[])):
        if (
            isinstance(assignment, ast.Assign)
            and len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Name)
            and assignment.targets[0].id in promotions
            and isinstance(assignment.value, ast.Call)
            and isinstance(assignment.value.func, ExtendedAST)
            and isinstance(assignment.value.func._type_info, CallableType)
            and any(
                assignment.value.func._type_info.value is factory
                for factory in factory_functions
            )
        ):
            # The spelling torch.zeros alone does not prove allocation if a
            # local binding or closure shadows the imported module.
            proven_factories.add(assignment.targets[0].id)
    return fresh_half_atomic_outputs(
        host_fn.body,
        {name: dtype for name, dtype in promotions.items() if name in proven_factories},
    )


def rewrite_cute_half_atomic_output_allocations(
    host_fn: HostFunction,
    promotions: dict[str, torch.dtype],
) -> None:
    from ..ast_extension import create
    from ..ast_extension import expr_from_string

    torch_factory_names = {
        "empty",
        "empty_like",
        "full",
        "full_like",
        "ones",
        "ones_like",
        "zeros",
        "zeros_like",
    }

    def dtype_expr(dtype: torch.dtype) -> ast.expr:
        expr = expr_from_string(f"torch.{str(dtype).split('.', 1)[1]}")
        assert isinstance(expr, ast.expr)
        return expr

    def is_torch_factory_call(call: ast.Call) -> bool:
        return (
            isinstance(call.func, ast.Attribute)
            and call.func.attr in torch_factory_names
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "torch"
        )

    def rewrite_allocation_dtype(call: ast.Call) -> None:
        dtype = dtype_expr(torch.float32)
        for kwarg in call.keywords:
            if kwarg.arg == "dtype":
                kwarg.value = dtype
                return
        if is_torch_factory_call(call):
            call.keywords.append(create(ast.keyword, arg="dtype", value=dtype))

    def rewrite_return_expr(expr: ast.expr) -> ast.expr:
        if isinstance(expr, ast.Name) and expr.id in promotions:
            cast_expr = expr_from_string(
                "{value}.to({dtype})",
                value=expr,
                dtype=dtype_expr(promotions[expr.id]),
            )
            assert isinstance(cast_expr, ast.expr)
            return cast_expr
        if isinstance(expr, ast.Tuple):
            return create(
                ast.Tuple,
                elts=[rewrite_return_expr(elt) for elt in expr.elts],
                ctx=expr.ctx,
            )
        if isinstance(expr, ast.List):
            return create(
                ast.List,
                elts=[rewrite_return_expr(elt) for elt in expr.elts],
                ctx=expr.ctx,
            )
        return expr

    for stmt in ast.walk(ast.Module(body=host_fn.body, type_ignores=[])):
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id in promotions
            and isinstance(stmt.value, ast.Call)
        ):
            rewrite_allocation_dtype(stmt.value)
        elif isinstance(stmt, ast.Return) and stmt.value is not None:
            stmt.value = rewrite_return_expr(stmt.value)


def promote_cute_root_graph_host_tensors(
    graph_infos: list[GraphInfo],
    promotions: dict[str, torch.dtype],
) -> None:
    from ...language._tracing_ops import _host_tensor
    from ..host_function import HostFunction

    host_fn = HostFunction.current()
    for graph_info in graph_infos:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target is not _host_tensor:
                continue
            target_name = node.args[0]
            if not isinstance(target_name, str) or target_name not in promotions:
                continue
            value = node.meta.get("val")
            if isinstance(value, torch.Tensor):
                promoted_value = value.to(dtype=torch.float32)
                if origin := host_fn.tensor_to_origin.get(value):
                    host_fn.tensor_to_origin[promoted_value] = origin
                node.meta["val"] = promoted_value
