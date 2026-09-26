"""Give direct full-slice CuTe matmuls an ordinary tunable contraction loop.

The first envelope accepts two-dimensional half/BF16 input tensors under a
two-axis output tile. A private FP32 accumulator spans the entire contraction;
the original result dtype is restored once, before any following epilogue.
No memory operation outside the replaced expression moves across the loop.
The contraction extents must already be provably equal; sample values cannot
establish equality for distinct dynamic shape symbols.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import language as hl
from ..ast_extension import ExtendedAST
from ..ast_extension import create
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from ..host_function import HostFunction


_FACTORIES = (
    torch.empty,
    torch.empty_like,
    torch.zeros,
    torch.zeros_like,
    torch.ones,
    torch.ones_like,
    torch.full,
    torch.full_like,
)
_MATMULS = (torch.mm, torch.matmul)
_METADATA_ATTRIBUTES = frozenset({"shape", "dtype", "device", "ndim"})
_METADATA_METHODS = frozenset({"size", "stride", "dim", "ndimension"})


def _bound_names(host: HostFunction) -> set[str]:
    return {
        *host.params.arguments,
        *(
            node.id
            for statement in host.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
        ),
    }


def _global_value(node: ast.AST, host: HostFunction, local_names: set[str]) -> object:
    if isinstance(node, ast.Name) and node.id not in local_names:
        return host.fn.__globals__.get(node.id)
    if isinstance(node, ast.Attribute):
        parent = _global_value(node.value, host, local_names)
        if parent is torch or parent is hl:
            return vars(parent).get(node.attr)
    return None


def _global_reference(
    value: object, host: HostFunction, local_names: set[str]
) -> ast.expr | None:
    """Resolve real bindings without changing the user's globals or aliases."""
    for name, candidate in host.fn.__globals__.items():
        if name in local_names or not name.isidentifier():
            continue
        if candidate is value:
            return create(ast.Name, id=name, ctx=ast.Load())
        if candidate is torch or candidate is hl:
            for member, item in vars(candidate).items():
                if item is value and member.isidentifier():
                    return create(
                        ast.Attribute,
                        value=create(ast.Name, id=name, ctx=ast.Load()),
                        attr=member,
                        ctx=ast.Load(),
                    )
    return None


def _metadata_expression(
    node: ast.AST,
    tensors: set[str],
    metadata: set[str],
    host: HostFunction,
    local_names: set[str],
) -> bool:
    def recurse(value: ast.AST) -> bool:
        return _metadata_expression(value, tensors, metadata, host, local_names)

    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name) and node.id in metadata:
        return True
    if isinstance(node, (ast.Name, ast.Attribute)):
        value = _global_value(node, host, local_names)
        if type(value) in (
            int,
            float,
            bool,
            str,
            torch.dtype,
            torch.device,
            torch.memory_format,
        ):
            return True
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id in tensors
            and node.attr in _METADATA_ATTRIBUTES
        )
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(recurse(item) for item in node.elts)
    if isinstance(node, ast.Subscript):
        return recurse(node.value) and recurse(node.slice)
    if isinstance(node, ast.Call):
        if _global_value(node.func, host, local_names) is torch.promote_types:
            return not node.keywords and all(recurse(arg) for arg in node.args)
        return (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in tensors
            and node.func.attr in _METADATA_METHODS
            and not node.keywords
            and all(recurse(arg) for arg in node.args)
        )
    if isinstance(node, ast.BinOp):
        return (
            isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod))
            and recurse(node.left)
            and recurse(node.right)
        )
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)) and recurse(
            node.operand
        )
    if isinstance(node, ast.Compare):
        return recurse(node.left) and all(recurse(item) for item in node.comparators)
    if isinstance(node, ast.BoolOp):
        return all(recurse(item) for item in node.values)
    return False


def _pure_host_prelude(
    host: HostFunction,
    local_names: set[str],
    *,
    allow_block_size_registration: bool = False,
) -> bool:
    """Do not infer input metadata across arbitrary host code or aliases."""
    tensors = {
        name
        for name, value in host.params.arguments.items()
        if isinstance(value, torch.Tensor)
    }
    metadata = set(host.params.arguments) - tensors
    for statement in host.body:
        if isinstance(statement, ast.For):
            return True
        if isinstance(statement, ast.Expr) and isinstance(
            statement.value, ast.Constant
        ):
            continue
        if isinstance(statement, ast.Assert):
            if not _metadata_expression(
                statement.test, tensors, metadata, host, local_names
            ):
                return False
            continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return False
        target = statement.targets[0]
        names = (
            [target.id]
            if isinstance(target, ast.Name)
            else [item.id for item in target.elts if isinstance(item, ast.Name)]
            if isinstance(target, (ast.Tuple, ast.List))
            else []
        )
        if (
            not names
            or isinstance(target, (ast.Tuple, ast.List))
            and len(names) != len(target.elts)
        ):
            return False
        if set(names) & (set(host.params.arguments) | tensors):
            return False
        value = statement.value
        if isinstance(value, ast.Call) and any(
            _global_value(value.func, host, local_names) is factory
            for factory in _FACTORIES
        ):
            if len(names) != 1 or any(
                keyword.arg in (None, "out") for keyword in value.keywords
            ):
                return False
            if not all(
                isinstance(arg, ast.Name)
                and arg.id in tensors
                or _metadata_expression(arg, tensors, metadata, host, local_names)
                for arg in (*value.args, *(keyword.value for keyword in value.keywords))
            ):
                return False
            tensors.add(names[0])
        elif (
            allow_block_size_registration
            and isinstance(value, ast.Call)
            and _global_value(value.func, host, local_names) is hl.register_block_size
            and not value.keywords
            and all(
                _metadata_expression(arg, tensors, metadata, host, local_names)
                for arg in value.args
            )
        ) or _metadata_expression(value, tensors, metadata, host, local_names):
            metadata.update(names)
        else:
            return False
    return False


def _input_binding_is_stable(
    name: str, host: HostFunction, local_names: set[str]
) -> bool:
    """Reject rebindings and escapes that could mutate input shape or dtype."""
    for statement in host.body:
        parents = {
            child: parent
            for parent in ast.walk(statement)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(statement):
            if not isinstance(node, ast.Name) or node.id != name:
                continue
            if not isinstance(node.ctx, ast.Load):
                return False
            parent = parents.get(node)
            if isinstance(parent, ast.Subscript) and parent.value is node:
                continue
            if isinstance(parent, ast.Attribute) and parent.value is node:
                if (
                    isinstance(parent.ctx, ast.Load)
                    and parent.attr in _METADATA_ATTRIBUTES | _METADATA_METHODS
                ):
                    continue
                return False
            if isinstance(parent, ast.Call) and any(
                _global_value(parent.func, host, local_names) is function
                for function in (*_FACTORIES, *_MATMULS)
            ):
                continue
            return False
    return True


def _full_slice(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Slice)
        and node.lower is None
        and node.upper is None
        and node.step is None
    )


def _operand(node: ast.AST, free_name: str, contraction_dimension: int) -> str | None:
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and isinstance(node.slice, ast.Tuple)
        and len(node.slice.elts) == 2
    ):
        return None
    contraction = node.slice.elts[contraction_dimension]
    free = node.slice.elts[1 - contraction_dimension]
    return (
        node.value.id
        if _full_slice(contraction)
        and isinstance(free, ast.Name)
        and free.id == free_name
        else None
    )


def stripmine_full_slice_matmuls(host: HostFunction) -> int:
    """Replace eligible statements, leaving rejected source completely unchanged."""
    env = CompileEnvironment.current()
    if env.backend_name != "cute" or not env.settings.cute_full_slice_matmul_tiling:
        return 0
    local_names = _bound_names(host)
    if not _pure_host_prelude(host, local_names):
        return 0
    input_tensors = {
        name: value
        for name, value in host.params.arguments.items()
        if isinstance(value, torch.Tensor)
        and value.ndim == 2
        and value.dtype in (torch.float16, torch.bfloat16)
        and _input_binding_is_stable(name, host, local_names)
    }
    if not input_tensors:
        return 0
    changed = 0
    used_names = local_names | set(host.fn.__globals__)

    def fresh(prefix: str) -> str:
        while prefix in used_names:
            prefix += "_"
        used_names.add(prefix)
        return prefix

    for root in host.body:
        if not (
            isinstance(root, ast.For)
            and isinstance(root.iter, ast.Call)
            and _global_value(root.iter.func, host, local_names) is hl.tile
            and len(root.iter.args) == 1
            and isinstance(root.iter.args[0], (ast.Tuple, ast.List))
            and len(root.iter.args[0].elts) == 2
            and isinstance(root.target, (ast.Tuple, ast.List))
            and len(root.target.elts) == 2
            and all(isinstance(item, ast.Name) for item in root.target.elts)
            and not root.orelse
        ):
            continue
        m_name, n_name = (cast("ast.Name", item).id for item in root.target.elts)
        body: list[ast.stmt] = []
        for statement in root.body:
            body.append(statement)
            if (
                not isinstance(statement, ast.Assign)
                or len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
            ):
                continue
            expression = statement.value
            if isinstance(expression, ast.BinOp) and isinstance(
                expression.op, ast.MatMult
            ):
                lhs, rhs = expression.left, expression.right
            elif (
                isinstance(expression, ast.Call)
                and len(expression.args) == 2
                and not expression.keywords
                and any(
                    _global_value(expression.func, host, local_names) is function
                    for function in _MATMULS
                )
            ):
                lhs, rhs = expression.args
            else:
                continue
            lhs_name = _operand(lhs, m_name, 1)
            rhs_name = _operand(rhs, n_name, 0)
            if lhs_name not in input_tensors or rhs_name not in input_tensors:
                continue
            lhs_value, rhs_value = input_tensors[lhs_name], input_tensors[rhs_name]
            if lhs_value.dtype != rhs_value.dtype or not env.known_equal(
                lhs_value.size(1), rhs_value.size(0)
            ):
                continue
            assert isinstance(statement, ExtendedAST)
            with statement:
                zeros = _global_reference(hl.zeros, host, local_names)
                addmm = _global_reference(torch.addmm, host, local_names)
                float32 = _global_reference(torch.float32, host, local_names)
                if zeros is None or addmm is None or float32 is None:
                    continue
                accumulator = fresh("_helion_fullslice_acc")
                contraction = fresh("_helion_fullslice_k")
                initial = statement_from_string(
                    f"{accumulator} = {{zeros}}([{m_name}, {n_name}], dtype={{dtype}})",
                    zeros=zeros,
                    dtype=float32,
                )
                loop = statement_from_string(
                    f"for {contraction} in {{tile}}({lhs_name}.size(1)):\n"
                    f"    {accumulator} = {{addmm}}({accumulator}, "
                    f"{lhs_name}[{m_name}, {contraction}], {rhs_name}[{contraction}, {n_name}])",
                    tile=root.iter.func,
                    addmm=addmm,
                )
                finish = statement_from_string(
                    f"{{target}} = {accumulator}.to({lhs_name}.dtype)",
                    target=statement.targets[0],
                )
                body[-1:] = [initial, loop, finish]
                changed += 1
        root.body = body
    return changed
