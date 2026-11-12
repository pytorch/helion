"""Export concrete CuTe launch wrappers without the Helion runtime."""

from __future__ import annotations

import ast
import copy
from graphlib import CycleError
from graphlib import TopologicalSorter
import importlib
import importlib.util
import inspect
from itertools import starmap
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch
from torch._dynamo.source import GetItemSource
from torch._dynamo.source import LocalSource
from torch._inductor.codecache import PyCodeCache
from torch._inductor.codegen.simd import constant_repr
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
from torch.utils._pytree import tree_leaves

from ...runtime.cute.arg_signature import normalize_scalar
from ...runtime.cute.launch_descriptor import STAGED_OPERAND_ALIGNMENT
from ...runtime.cute.launcher import _create_cute_wrapper
from ...runtime.cute.launcher import _cute_kernel_param_is_constexpr
from ...runtime.cute.launcher import _cute_staged_plan_operands
from ..compile_environment import _replay_tensor_input_source
from ..output_code_utils import _check_kernel_name_not_shadowed
from ..output_code_utils import _GlobalToNonlocal
from ..runtime_tensor_guards import _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
from .memory_ops import _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from torch._guards import Source

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


def _decorator_name(decorator: ast.expr) -> str:
    """``cute.jit`` for both ``@cute.jit`` and ``@cute.jit(...)``."""
    return ast.unparse(decorator.func if isinstance(decorator, ast.Call) else decorator)


class _StructAnnotations(ast.NodeTransformer):
    """Keep CuTe struct and dataclass field types concrete under generated future
    annotations.

    CuTe DSL reads field types as objects: a dataclass field annotated
    ``cutlass.Constexpr[float]`` is a compile-time constant only if its type is
    not the string the future import makes of it.
    """

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        if not any(
            _decorator_name(decorator)
            in ("cute.struct", "dataclass", "dataclasses.dataclass")
            for decorator in node.decorator_list
        ):
            return node
        fields = [
            (statement.target.id, statement.annotation)
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        ]
        if fields:
            body: list[ast.stmt] = []
            for statement in node.body:
                if not isinstance(statement, ast.AnnAssign):
                    body.append(statement)
                elif statement.value is not None:
                    # A field default stays a class attribute, where dataclasses
                    # find it.
                    body.append(
                        ast.Assign(targets=[statement.target], value=statement.value)
                    )
            node.body = body
            # After the docstring, which stays the class's ``__doc__``.
            first = body[0].value if body and isinstance(body[0], ast.Expr) else None
            docstring = isinstance(first, ast.Constant) and isinstance(first.value, str)
            node.body.insert(
                int(docstring),
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
        if module_name != "helion" and not module_name.startswith("helion."):
            return [node]
        module = __import__(module_name, fromlist=[alias.name for alias in node.names])
        for alias in node.names:
            value = getattr(module, alias.name)
            if inspect.ismodule(value):
                # A from-import can bind a submodule rather than a definition.
                output.extend(
                    self.rewrite_import(
                        ast.Import(
                            names=[
                                ast.alias(
                                    name=value.__name__,
                                    asname=alias.asname or alias.name,
                                )
                            ]
                        ),
                        body,
                    )
                )
                continue
            namespace = self.require(module_name, {alias.name})
            output.append(
                ast.Assign(
                    targets=[ast.Name(id=alias.asname or alias.name, ctx=ast.Store())],
                    value=ast.Attribute(
                        value=ast.Name(id=namespace, ctx=ast.Load()),
                        attr=alias.name,
                        ctx=ast.Load(),
                    ),
                )
            )
        return output

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
            "tuple" if isinstance(arg, tuple) else "list",
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
            return ("constant", normalize_scalar(arg)[1])
    for kind, classes in (
        ("bool", (bool, torch.SymBool)),
        ("int", (int, torch.SymInt)),
        ("float", (float, torch.SymFloat)),
    ):
        if isinstance(arg, classes):
            return ("scalar", kind)
    raise NotImplementedError(f"Unsupported standalone CuTe argument type: {type(arg)}")


def _schema_repr(value: object) -> str:
    """Render nested schemas as Python expressions, including non-finite floats."""
    if isinstance(value, tuple):
        items = ", ".join(_schema_repr(item) for item in value)
        return f"({items}{',' if len(value) == 1 else ''})"
    if isinstance(value, float):
        return constant_repr(value)
    return repr(value)


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

    def source_expression(source: Source) -> str:
        if isinstance(source, LocalSource):
            return source.local_name
        assert isinstance(source, GetItemSource)
        base = source_expression(source.base)
        value = _replay_tensor_input_source(source.base, fake_by_name)
        if isinstance(source.index, str) and not isinstance(value, dict):
            # Namedtuple fields are recorded as string-indexed sources.
            return f"{base}.{source.index}"
        return f"{base}[{source.index!r}]"

    for key, specialization in bound.env.runtime_input_specializations.items():
        facts = bound.env.bound_runtime_input_specialization_results[key]
        source_args = ", ".join(map(source_expression, specialization.sources))
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


def _mma_dag_alias_spec(
    plans: Sequence[dict[str, object]], schema: Sequence[tuple[Any, ...]]
) -> tuple[tuple[int, ...], tuple[bool, ...]] | None:
    """The launch positions and store flags of a cute_mma_dag alias plan
    (the launcher's ``MmaDagAliasGuard``) for the exported launcher's span
    check; None without the plan."""
    alias = [plan for plan in plans if plan.get("kind") == "helion_mma_dag_alias"]
    if not alias:
        return None
    if len(alias) > 1:
        raise NotImplementedError(
            "Standalone CuTe export expects one mma_dag alias plan per kernel"
        )
    indices = alias[0].get("tensor_idx")
    stored = alias[0].get("stored")
    if (
        not isinstance(indices, list)
        or not isinstance(stored, list)
        or len(indices) != len(stored)
        or any(
            type(index) is not int
            or not 0 <= index < len(schema)
            or schema[index][0] != "tensor"
            for index in indices
        )
        or any(type(flag) is not bool for flag in stored)
    ):
        raise NotImplementedError(
            "Standalone CuTe export needs a well-formed mma_dag alias plan"
        )
    return tuple(indices), tuple(stored)


# Opts standalone export into hard-coding the bool and float arguments whose
# values capture needs (see :func:`_concretize_needed_scalars`).
HARDCODE_SCALARS_ENV = "HELION_CUTE_EXPORT_HARDCODE_SCALARS"


def _capture_launches(
    bound: BoundKernel[Any],
    source: str,
    pointer_residues: dict[torch.UntypedStorage, int],
    scalars: dict[str, bool | float],
) -> tuple[list[ast.stmt], dict[str, tuple[Any, ...]], dict[str, bool | float]]:
    """Run the host on FakeTensors to build native wrappers for its launches.

    Also returns the ``scalars`` (from :func:`_scalar_values`) the host code
    needed concrete (:func:`_concretize_needed_scalars`), with
    ``HELION_CUTE_EXPORT_HARDCODE_SCALARS=1``; without it the host code runs
    with every scalar symbolic, and none is concrete."""
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
        if getattr(kernel, "_helion_cute_sass_checks", ()):
            # The exported launcher has no SASS verification step
            # (``helion.runtime.cute.sass_check``); a kernel whose
            # instruction forms need one (hinted mma_dag packets) is not
            # exported.  Pin the plain config (``load_eviction_policies``
            # all "") to export this kernel.
            raise NotImplementedError(
                "Standalone CuTe export does not support kernels whose SASS "
                "the launcher must verify (hinted cp.async packets)"
            )
        plans = getattr(kernel, "_helion_cute_wrapper_plans", ())
        if any(
            plan.get("kind")
            not in (
                "helion_flash",
                # The fused flash backward's plan only selects its launch
                # path; like the forward's, its one host preparation, the
                # staging of under-aligned operands, is refused below.
                "helion_flash_bwd",
                "helion_small_biased_attention",
                # Launch-time argument metadata: the exported launcher runs
                # the alias check from the launch table (no host preparation).
                "helion_mma_dag_alias",
            )
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
                kind, value = normalize_scalar(arg)
                schema.append(("scalar_constexpr", kind, value, value))
            else:
                schema.append(_argument_schema(arg, pointer_residues))
        _check_staged_operand_alignment(_cute_staged_plan_operands(kernel), schema)
        kernel_name = cast("Any", kernel).__name__
        options = kwargs.get("cute_compile_options")
        alias = _mma_dag_alias_spec(plans, schema)
        if kernel_name in launch_specs:
            previous_schema, _, previous_block, previous_options, previous_alias = (
                launch_specs[kernel_name]
            )
            if (tuple(schema), block, options, alias) != (
                previous_schema,
                previous_block,
                previous_options,
                previous_alias,
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
        launch_specs[kernel_name] = (
            tuple(schema),
            wrapper.name,
            block,
            options,
            alias,
        )

    fakes = dict(zip(bound.kernel.signature.parameters, bound.fake_args, strict=True))

    def run(values: dict[str, bool | float]) -> None:
        wrappers.clear()
        launch_specs.clear()
        with bound.env:
            entrypoint(*starmap(values.get, fakes.items()), _launcher=capture)

    if os.environ.get(HARDCODE_SCALARS_ENV, "") == "1":
        concrete = _concretize_needed_scalars(run, scalars, fakes)
    else:
        # Off by default: a host branch on a bool or float argument raises
        # GuardOnDataDependentSymNode, as export always did.
        run({})
        concrete = {}
    if not launch_specs:
        raise NotImplementedError(
            "No CuTe launches were captured for standalone export"
        )
    return wrappers, launch_specs, concrete


def _concretize_needed_scalars(
    run: Callable[[dict[str, bool | float]], None],
    scalars: dict[str, bool | float],
    fakes: dict[str, object],
) -> dict[str, bool | float]:
    """Run the host code (``run``, given the concrete values) with the bool
    and float arguments symbolic, making concrete only those whose values it
    needs, and return them.

    These arguments trace as unbacked symbols that carry no value, so a host
    branch on one, or a launch that needs it as a constexpr, raises
    GuardOnDataDependentSymNode; its condition names the symbols it read, and
    the run repeats with those arguments set to their ``scalars`` values. A
    scalar the host code only computes with or passes on stays symbolic, a
    runtime argument of the export."""
    names = {
        symbol: name
        for name, fake in fakes.items()
        if name in scalars and isinstance(fake, (torch.SymBool, torch.SymFloat))
        for symbol in fake._sympy_().free_symbols
    }
    concrete: dict[str, bool | float] = {}
    while True:
        try:
            run(concrete)
        except GuardOnDataDependentSymNode as error:
            needed = {
                names[symbol] for symbol in error.cond.free_symbols if symbol in names
            } - concrete.keys()
            if not needed:
                raise NotImplementedError(
                    "Standalone CuTe export cannot run host code that branches on "
                    "a value other than a top-level bool or float argument"
                ) from error
            concrete.update((name, scalars[name]) for name in needed)
        else:
            return concrete


def _check_staged_operand_alignment(
    staged: Sequence[tuple[int, bool]], schema: Sequence[tuple[Any, ...]]
) -> None:
    """Refuse a launch whose staged operands (``STAGED_PLAN_OPERANDS``) are
    under-aligned. ``default_cute_launcher`` copies such an operand to a
    16-byte-aligned buffer before the launch; the exported launcher has no
    such step, and a TMA descriptor over the under-aligned pointer would read
    and write the wrong bytes without faulting. An aligned export is safe:
    its tensor schema rejects under-aligned calls."""
    for index, _ in staged:
        entry = schema[index]
        if entry[0] == "tensor" and entry[-1] < STAGED_OPERAND_ALIGNMENT:
            raise NotImplementedError(
                "Standalone CuTe export cannot stage an under-aligned operand; "
                "export with 16-byte-aligned tensors"
            )


def _scalar_values(bound: BoundKernel[Any]) -> dict[str, bool | float]:
    """Values of the kernel's bool and float arguments: the call's being
    exported (the environment's runtime argument values), else the bound
    kernel's own. These key a bound kernel by type, so one serves every
    value."""
    runtime = bound.env.runtime_arg_values_by_name
    return {
        name: value
        if isinstance(value := runtime.get(name), (bool, float))
        and type(value) is type(default)
        else default
        for name, default in bound._runtime_scalar_args_by_name.items()
    }


def omitted_parameters(bound: BoundKernel[Any]) -> set[str]:
    """The parameters the standalone entry point omits: callable arguments
    (score/mask mods) only shape tracing, and their effect is compiled into
    the kernel. Decided on the fake arguments, for the export and for the
    inputs hand-off saves alike."""
    return {
        name
        for name, fake in zip(
            bound.kernel.signature.parameters, bound.fake_args, strict=True
        )
        if callable(fake) and not isinstance(fake, type)
    }


def _drop_parameters(function: ast.FunctionDef, names: set[str]) -> None:
    """Remove ``names`` from a function signature, keeping defaults aligned."""
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    defaults = [None] * (len(positional) - len(arguments.defaults)) + list(
        arguments.defaults
    )
    kept = [
        (arg, default)
        for arg, default in zip(positional, defaults, strict=True)
        if arg.arg not in names
    ]
    posonly = sum(arg.arg not in names for arg in arguments.posonlyargs)
    arguments.posonlyargs = [arg for arg, _ in kept[:posonly]]
    arguments.args = [arg for arg, _ in kept[posonly:]]
    # Defaulted parameters are a suffix, so the kept defaults still are.
    arguments.defaults = [default for _, default in kept if default is not None]
    kwonly = [
        (arg, default)
        for arg, default in zip(
            arguments.kwonlyargs, arguments.kw_defaults, strict=True
        )
        if arg.arg not in names
    ]
    arguments.kwonlyargs = [arg for arg, _ in kwonly]
    arguments.kw_defaults = [default for _, default in kwonly]


def _constant_expr(value: bool | float) -> ast.expr:
    """The expression of a pinned scalar's value (``float("inf")`` and the
    like for non-finite floats)."""
    return ast.parse(_schema_repr(value), mode="eval").body


def _pin_defaults(function: ast.FunctionDef, pinned: dict[str, bool | float]) -> None:
    """Make the default of each hard-coded parameter its pinned value: the
    source default could only fail the constant check. Kernel parameters are
    positional (``Kernel`` rejects keyword-only ones)."""
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    offset = len(positional) - len(arguments.defaults)
    for index in range(len(arguments.defaults)):
        if (name := positional[offset + index].arg) in pinned:
            arguments.defaults[index] = _constant_expr(pinned[name])


def _fit_repro_call(
    call: ast.FunctionDef,
    kernel_name: str,
    dropped: set[str],
    pinned: dict[str, bool | float],
) -> None:
    """Fit the repro caller (``emit_repro_caller``) to the entry point: drop
    the arguments it omits and pass the hard-coded scalars their values
    instead of placeholders."""
    body: list[ast.stmt] = []
    for statement in call.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(target := statement.targets[0], ast.Name)
        ):
            if target.id in dropped:
                continue
            if target.id in pinned:
                statement.value = _constant_expr(pinned[target.id])
        elif (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == kernel_name
        ):
            statement.value.args = [
                arg
                for arg in statement.value.args
                if not (isinstance(arg, ast.Name) and arg.id in dropped)
            ]
        body.append(statement)
    call.body = body


def build_standalone_code(
    bound: BoundKernel[Any], import_lines: list[str], body_root: ast.Module
) -> ast.Module:
    """Capture static launch schemas and embed readable native CuTe wrappers."""
    if not bound.settings.static_shapes:
        raise NotImplementedError("Standalone CuTe export requires static_shapes=True")
    # Callable arguments are compiled into the kernel and the entry point
    # omits them, so host code must not use one; refuse before capture runs
    # the host code with their fake stand-ins.
    dropped = omitted_parameters(bound)
    host_source = next(
        statement
        for statement in body_root.body
        if isinstance(statement, ast.FunctionDef)
        and statement.name == bound.kernel.name
    )
    if any(
        isinstance(node, ast.Name) and node.id in dropped
        for statement in host_source.body
        for node in ast.walk(statement)
    ):
        raise NotImplementedError(
            "Standalone CuTe export cannot omit a callable argument the host code "
            "uses (calls, or reads the closure or attributes of)"
        )
    pointer_residues, guards = _input_specializations(bound)
    source = (
        "from __future__ import annotations\n"
        + "\n".join(import_lines)
        + "\n"
        + ast.unparse(body_root)
    )
    wrappers, launch_specs, pinned = _capture_launches(
        bound, source, pointer_residues, _scalar_values(bound)
    )
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
    names = list(bound.kernel.signature.parameters)
    kept = [
        (name, arg)
        for name, arg in zip(names, bound.fake_args, strict=True)
        if name not in dropped
    ]
    _drop_parameters(host, dropped)
    _pin_defaults(host, pinned)
    # Capture made the ``pinned`` bool and float arguments concrete because
    # their values were needed (a host branch, a constexpr launch parameter),
    # so the export hard-codes them as constants every call must match. The
    # others stay runtime scalars.
    for statement in body:
        if (
            isinstance(statement, ast.FunctionDef)
            and statement.name == "call"
            and statement is not host
        ):
            _fit_repro_call(statement, bound.kernel.name, dropped, pinned)
    argument_specs = tuple(
        ("constant", pinned[name])
        if name in pinned
        else _argument_schema(
            arg, pointer_residues, specialized_vars=bound.env.specialized_vars
        )
        for name, arg in kept
    )
    host.body[:0] = [
        *ast.parse(
            f"_standalone_validate_args(({', '.join(name for name, _ in kept)},), "
            f"{_schema_repr(argument_specs)})"
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
            f"    {name!r}: ({_schema_repr(schema)}, {wrapper}, {block!r}, {options!r}, {alias!r}),"
            for name, (schema, wrapper, block, options, alias) in launch_specs.items()
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
