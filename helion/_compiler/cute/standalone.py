"""Export concrete CuTe launch wrappers without the Helion runtime."""

from __future__ import annotations

import ast
import copy
from graphlib import CycleError
from graphlib import TopologicalSorter
import importlib
import importlib.util
import inspect
import math
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch
from torch._inductor.codecache import PyCodeCache
from torch.utils._pytree import tree_leaves

from ...runtime.cute.launcher import _create_cute_wrapper
from ...runtime.cute.launcher import _cute_kernel_param_is_constexpr
from ...runtime.cute.launcher import _normalize_cute_scalar
from ..compile_environment import _replay_tensor_input_source
from ..output_code_utils import _check_kernel_name_not_shadowed
from ..output_code_utils import _GlobalToNonlocal
from .memory_ops import _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
from .memory_ops import _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY

if TYPE_CHECKING:
    from ...runtime.kernel import BoundKernel


def _bound_names(statement: ast.stmt) -> set[str]:
    if isinstance(statement, (ast.FunctionDef, ast.ClassDef)):
        return {statement.name}
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return {alias.asname or alias.name.split(".")[0] for alias in statement.names}
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        return {
            node.id
            for target in targets
            for node in ast.walk(target)
            if isinstance(node, ast.Name)
        }
    return set()


def _helper_namespace(module_name: str) -> str:
    return "_standalone_" + module_name.rsplit(".", 1)[-1].lstrip("_")


class _StructAnnotations(ast.NodeTransformer):
    """Keep CuTe struct field types concrete under generated future annotations."""

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        if not any(
            ast.unparse(decorator) == "cute.struct" for decorator in node.decorator_list
        ):
            return node
        fields = [
            (statement.target.id, statement.annotation)
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        ]
        if fields:
            node.body = [
                statement
                for statement in node.body
                if not isinstance(statement, ast.AnnAssign)
            ]
            node.body.insert(
                0,
                ast.Assign(
                    targets=[ast.Name(id="__annotations__", ctx=ast.Store())],
                    value=ast.Dict(
                        keys=[ast.Constant(value=name) for name, _ in fields],
                        values=[annotation for _, annotation in fields],
                    ),
                ),
            )
        return node


class _Helpers:
    """Copy only the native helper definitions referenced by generated code."""

    def __init__(self) -> None:
        self.needed: dict[str, set[str]] = {}
        self.packages: dict[str, str | None] = {}
        self.bindings: dict[str, dict[str, ast.stmt]] = {}

    def require(self, module_name: str, names: set[str]) -> str:
        if module_name in (
            "helion.runtime.cute.paired_sum",
            "helion.runtime.cute.single_sum",
        ):
            raise NotImplementedError(
                f"Standalone CuTe export cannot capture the runtime-dependent launch plan in {module_name}"
            )
        if module_name not in self.bindings:
            module = importlib.import_module(module_name)
            self.packages[module_name] = module.__package__
            source = inspect.getsource(module)
            statements = ast.parse(source).body
            self.bindings[module_name] = {
                name: statement
                for statement in statements
                for name in _bound_names(statement)
            }
            self.needed[module_name] = set()
        bindings = self.bindings[module_name]
        pending = list(names)
        while pending:
            name = pending.pop()
            if name in self.needed[module_name]:
                continue
            if name not in bindings:
                raise NotImplementedError(
                    f"Cannot export CuTe helper {module_name}.{name}"
                )
            self.needed[module_name].add(name)
            statement = bindings[name]
            pending.extend(
                node.id
                for node in ast.walk(statement)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in bindings
            )
        return _helper_namespace(module_name)

    def rewrite_import(
        self,
        node: ast.Import | ast.ImportFrom,
        body: ast.AST,
        package: str | None = None,
    ) -> list[ast.stmt]:
        output: list[ast.stmt] = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith("helion."):
                    output.append(ast.Import(names=[alias]))
                    continue
                if alias.asname is None:
                    raise NotImplementedError(
                        "Standalone CuTe helper imports require an alias"
                    )
                local = alias.asname
                names = {
                    child.attr
                    for child in ast.walk(body)
                    if isinstance(child, ast.Attribute)
                    and isinstance(child.value, ast.Name)
                    and child.value.id == local
                }
                namespace = self.require(alias.name, names)
                output.extend(ast.parse(f"{local} = {namespace}").body)
            return output
        module_name = node.module or ""
        if node.level:
            assert package is not None
            module_name = importlib.util.resolve_name(
                "." * node.level + module_name, package
            )
        if not module_name.startswith("helion."):
            return [node]
        namespace = self.require(module_name, {alias.name for alias in node.names})
        return [
            ast.Assign(
                targets=[ast.Name(id=alias.asname or alias.name, ctx=ast.Store())],
                value=ast.Attribute(
                    value=ast.Name(id=namespace, ctx=ast.Load()),
                    attr=alias.name,
                    ctx=ast.Load(),
                ),
            )
            for alias in node.names
        ]

    def emit(self) -> list[ast.stmt]:
        emitted: dict[str, list[ast.stmt]] = {}
        resolved: dict[str, frozenset[str]] = {}
        dependencies: dict[str, set[str]] = {}
        # Imported helpers can add symbols to an already visited module.
        while resolved != {
            name: frozenset(symbols) for name, symbols in self.needed.items()
        }:
            for name in list(self.needed):
                symbols = frozenset(self.needed[name])
                if resolved.get(name) == symbols:
                    continue
                selected = {self.bindings[name][symbol] for symbol in self.needed[name]}
                whole = ast.Module(
                    body=copy.deepcopy(
                        sorted(
                            selected, key=lambda node: (node.lineno, node.col_offset)
                        )
                    ),
                    type_ignores=[],
                )
                rewritten = self.rewrite(whole, self.packages[name])
                emitted[name] = rewritten.body
                referenced_names = {
                    node.id
                    for node in ast.walk(rewritten)
                    if isinstance(node, ast.Name)
                }
                dependencies[name] = {
                    dependency
                    for dependency in self.needed
                    if dependency != name
                    and _helper_namespace(dependency) in referenced_names
                }
                resolved[name] = symbols
        result: list[ast.stmt] = []
        try:
            ordered = tuple(
                TopologicalSorter(
                    {name: sorted(deps) for name, deps in dependencies.items()}
                ).static_order()
            )
        except CycleError as error:
            raise NotImplementedError(
                "Cannot export cyclic CuTe helper imports"
            ) from error
        for name in ordered:
            namespace = _helper_namespace(name)
            transformer = _GlobalToNonlocal(set(self.needed[name]))
            body = [transformer.visit(statement) for statement in emitted[name]]
            exported = sorted(self.needed[name])
            body.extend(
                ast.parse(
                    "return types.SimpleNamespace("
                    + ", ".join(f"{key}={key}" for key in exported)
                    + ")"
                ).body
            )
            function = ast.FunctionDef(
                name=f"_make{namespace}",
                args=ast.arguments(
                    posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
                ),
                body=body,
                decorator_list=[],
            )
            result.append(_StructAnnotations().visit(function))
            result.extend(ast.parse(f"{namespace} = _make{namespace}()").body)
        return result

    def rewrite(self, body: ast.Module, package: str | None = None) -> ast.Module:
        helpers = self

        class Imports(ast.NodeTransformer):
            def visit_Import(self, node: ast.Import) -> list[ast.stmt]:
                return helpers.rewrite_import(node, body, package)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> list[ast.stmt]:
                return helpers.rewrite_import(node, body, package)

        return Imports().visit(body)


def _argument_schema(
    arg: object,
    pointer_residues: dict[torch.UntypedStorage, int],
    *,
    specialized_vars: set[Any] | None = None,
) -> tuple[Any, ...]:
    if isinstance(arg, torch.Tensor):
        alignment = math.gcd(
            16,
            pointer_residues.get(arg.untyped_storage(), 0)
            + int(arg.storage_offset()) * arg.element_size(),
        )
        return (
            "tensor",
            str(arg.dtype),
            arg.ndim,
            tuple(map(int, arg.shape)),
            tuple(map(int, arg.stride())),
            alignment,
        )
    if isinstance(arg, (tuple, list)):
        return (
            type(arg).__name__,
            tuple(
                _argument_schema(
                    item, pointer_residues, specialized_vars=specialized_vars
                )
                for item in arg
            ),
        )
    if isinstance(arg, dict):
        return (
            "dict",
            tuple(
                (
                    key,
                    _argument_schema(
                        value, pointer_residues, specialized_vars=specialized_vars
                    ),
                )
                for key, value in arg.items()
            ),
        )
    constant = specialized_vars is not None
    if constant and isinstance(arg, (torch.SymBool, torch.SymInt, torch.SymFloat)):
        constant = bool(arg._sympy_().free_symbols & specialized_vars)
    if constant:
        if isinstance(arg, torch.device):
            return ("device", str(arg))
        if isinstance(arg, (bool, int, float, str, type(None), torch.dtype)):
            return ("constant", arg)
        if isinstance(arg, (torch.SymBool, torch.SymInt, torch.SymFloat)):
            return ("constant", _normalize_cute_scalar(arg)[1])
    for kind, classes in (
        ("bool", (bool, torch.SymBool)),
        ("int", (int, torch.SymInt)),
        ("float", (float, torch.SymFloat)),
    ):
        if isinstance(arg, classes):
            return ("scalar", kind)
    raise NotImplementedError(f"Unsupported standalone CuTe argument type: {type(arg)}")


def _input_specializations(
    bound: BoundKernel[Any],
) -> tuple[dict[torch.UntypedStorage, int], list[ast.stmt]]:
    """Preserve the compiler's input alignment and aliasing assumptions."""
    if bound.env.runtime_input_specializations.keys() - {
        _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY,
        _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY,
    }:
        raise NotImplementedError(
            "Standalone CuTe export cannot preserve runtime input specializations"
        )
    pointer_residues: dict[torch.UntypedStorage, int] = {}
    fake_by_name = dict(
        zip(bound.kernel.signature.parameters, bound.fake_args, strict=True)
    )
    guards: list[ast.stmt] = []

    class SourceExpression(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "L"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                return ast.Name(id=node.slice.value, ctx=ast.Load())
            self.generic_visit(node)
            return node

    for key, specialization in bound.env.runtime_input_specializations.items():
        facts = bound.env.bound_runtime_input_specialization_results[key]
        source_args = ", ".join(
            ast.unparse(SourceExpression().visit(ast.parse(item.name, mode="eval")))
            for item in specialization.sources
        )
        if key == _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY:
            residues = []
            for item, signature in zip(
                specialization.sources, cast("tuple[Any, ...]", facts), strict=True
            ):
                tensor = _replay_tensor_input_source(item, fake_by_name)
                if not isinstance(tensor, torch.Tensor) or signature is None:
                    raise NotImplementedError(
                        "Standalone CuTe export needs captured input alignment facts"
                    )
                residue = signature[0]
                residues.append(residue)
                pointer_residues[tensor.untyped_storage()] = (
                    residue - int(tensor.storage_offset()) * tensor.element_size()
                ) % 16
            guards.extend(
                ast.parse(
                    f"_standalone_validate_alignment(({source_args},), {tuple(residues)!r})"
                ).body
            )
        else:
            guards.extend(
                ast.parse(
                    f"_standalone_validate_aliases(({source_args},), {facts!r})"
                ).body
            )
    # Dtypes outside the compiler's vectorization classifier still need native
    # pointer alignment. Use the original tensors while their weak refs are live.
    with bound._runtime_arg_values_for_codegen():
        for tensor in tree_leaves(bound.fake_args):
            if (
                isinstance(tensor, torch.Tensor)
                and tensor.untyped_storage() not in pointer_residues
            ):
                actual = bound.env.runtime_value_for_tensor(tensor)
                if isinstance(actual, torch.Tensor):
                    pointer_residues[tensor.untyped_storage()] = (
                        actual.data_ptr()
                        - int(tensor.storage_offset()) * tensor.element_size()
                    ) % 16
    return pointer_residues, guards


def _capture_launches(
    bound: BoundKernel[Any],
    source: str,
    pointer_residues: dict[torch.UntypedStorage, int],
) -> tuple[list[ast.stmt], dict[str, tuple[Any, ...]]]:
    """Run the host on FakeTensors to build native wrappers for its launches."""
    module = PyCodeCache.load(source)
    entrypoint = getattr(module, bound.kernel.name)
    wrappers: list[ast.stmt] = []
    launch_specs: dict[str, tuple[Any, ...]] = {}
    num_sm = torch.cuda.get_device_properties(bound.env.device).multi_processor_count

    def capture(
        kernel: object, grid: tuple[int, ...], *args: object, **kwargs: object
    ) -> None:
        if getattr(kernel, "_helion_cute_disable_bake_tensor_shapes", False):
            raise NotImplementedError(
                "Standalone CuTe export cannot bake this kernel's runtime tensor layouts"
            )
        plans = getattr(kernel, "_helion_cute_wrapper_plans", ())
        if any(
            plan.get("kind") not in ("helion_flash", "helion_small_biased_attention")
            for plan in plans
        ):
            raise NotImplementedError(
                "Standalone CuTe export does not yet support this kernel's descriptor-preparation plan"
            )
        block = tuple(cast("tuple[int, ...]", kwargs.get("block", (256, 1, 1))))
        block = (*block, *(1 for _ in range(3 - len(block))))
        flags = _cute_kernel_param_is_constexpr(kernel)
        schema: list[tuple[Any, ...]] = []
        for index, arg in enumerate(args):
            if index < len(flags) and flags[index]:
                kind, value = _normalize_cute_scalar(arg)
                schema.append(("scalar_constexpr", kind, value, value))
            else:
                schema.append(_argument_schema(arg, pointer_residues))
        kernel_name = cast("Any", kernel).__name__
        options = kwargs.get("cute_compile_options")
        if kernel_name in launch_specs:
            previous_schema, _, previous_block, previous_options = launch_specs[
                kernel_name
            ]
            if (tuple(schema), block, options) != (
                previous_schema,
                previous_block,
                previous_options,
            ):
                raise NotImplementedError(
                    "Standalone CuTe export requires one launch schema per device function"
                )
            return
        native = _create_cute_wrapper(
            kernel, tuple(schema), cast("tuple[int, int, int]", block), num_sm=num_sm
        )
        wrapper = ast.parse(inspect.getsource(cast("Any", native))).body[0]
        assert isinstance(wrapper, ast.FunctionDef)
        wrapper.name = f"_standalone_launch_{len(launch_specs)}"
        for node in ast.walk(wrapper):
            if isinstance(node, ast.Name) and node.id == "_kernel":
                node.id = kernel_name
        wrappers.append(wrapper)
        launch_specs[kernel_name] = (tuple(schema), wrapper.name, block, options)

    with bound.env:
        entrypoint(*bound.fake_args, _launcher=capture)
    if not launch_specs:
        raise NotImplementedError(
            "No CuTe launches were captured for standalone export"
        )
    return wrappers, launch_specs


def build_standalone_code(
    bound: BoundKernel[Any], import_lines: list[str], body_root: ast.Module
) -> ast.Module:
    """Capture static launch schemas and embed readable native CuTe wrappers."""
    if not bound.settings.static_shapes:
        raise NotImplementedError("Standalone CuTe export requires static_shapes=True")
    pointer_residues, guards = _input_specializations(bound)
    source = (
        "from __future__ import annotations\n"
        + "\n".join(import_lines)
        + "\n"
        + ast.unparse(body_root)
    )
    wrappers, launch_specs = _capture_launches(bound, source, pointer_residues)
    helpers = _Helpers()
    tree = ast.parse(source)
    statements = []
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom) and statement.module in (
            "__future__",
            "helion.runtime",
        ):
            continue
        if isinstance(statement, ast.Import) and any(
            alias.name == "helion" for alias in statement.names
        ):
            continue
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            if any(
                alias.asname == "_source_module"
                or (alias.asname or "").startswith("_global_source")
                for alias in statement.names
            ):
                raise NotImplementedError(
                    "Standalone CuTe export cannot import values from the original kernel module"
                )
        statements.append(statement)
    tree.body = statements
    tree = helpers.rewrite(tree)
    imports: list[ast.stmt] = []
    body: list[ast.stmt] = []
    for statement in tree.body:
        (
            imports if isinstance(statement, (ast.Import, ast.ImportFrom)) else body
        ).append(statement)
    host = next(
        statement
        for statement in body
        if isinstance(statement, ast.FunctionDef)
        and statement.name == bound.kernel.name
    )
    argument_specs = tuple(
        _argument_schema(
            arg, pointer_residues, specialized_vars=bound.env.specialized_vars
        )
        for arg in bound.fake_args
    )
    names = list(bound.kernel.signature.parameters)
    host.body[:0] = [
        *ast.parse(
            f"_standalone_validate_args(({', '.join(names)},), {argument_specs!r})"
        ).body,
        *guards,
    ]
    runtime_path = (
        Path(__file__).parents[2] / "runtime" / "cute" / "standalone_launcher.py"
    )
    runtime = ast.parse(runtime_path.read_text()).body
    runtime = [
        statement
        for statement in runtime
        if not (
            isinstance(statement, ast.ImportFrom) and statement.module == "__future__"
        )
    ]
    table = (
        "_STANDALONE_LAUNCHES = {\n"
        + "\n".join(
            f"    {name!r}: ({schema!r}, {wrapper}, {block!r}, {options!r}),"
            for name, (schema, wrapper, block, options) in launch_specs.items()
        )
        + "\n}"
    )
    import_lines[:] = [ast.unparse(statement) for statement in imports]
    body_root = ast.Module(
        body=[*runtime, *helpers.emit(), *body, *wrappers, *ast.parse(table).body],
        type_ignores=[],
    )
    ast.fix_missing_locations(body_root)
    _check_kernel_name_not_shadowed(body_root, import_lines, bound.kernel.name)
    return body_root
