"""Dependency-free ("standalone") output code for :meth:`BoundKernel.to_code`.

By default ``to_code`` returns code that still imports ``helion`` at runtime (for
the launcher and a few ``helion.runtime.*`` helpers). Passing
:class:`~helion.runtime.kernel.OutputCodeOptions` with ``allow_helion_deps=False``
instead yields a self-contained module whose only dependencies are ``torch`` + the
backend DSL (e.g. ``triton``) -- so clients can check in a helion-generated kernel
without taking on helion as a dependency.

This module holds the backend-neutral machinery. :func:`build_dependency_free_code`
is an AST transform over ``to_code``'s ``body_root`` (run before it is unparsed): it
drops the helion imports (from ``import_lines``) and prepends -- as AST statements --
the dependency-free launcher inlined as a local ``helion.runtime`` shim (so the
body's verbatim ``helion.runtime.<fn>(...)`` calls resolve without the real package)
plus any in-kernel runtime helpers the body references (via
``Backend.embedded_helper_source``). The per-backend launcher info lives on the
backend classes (``Backend.dependency_free_launcher_info`` in
:mod:`helion._compiler.backend`). Backends that need workload-specific native
launch wrappers can instead implement ``Backend.build_standalone_code``.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import symtable
from typing import TYPE_CHECKING
from typing import Any

from .._utils import cdiv
from .._utils import next_power_of_2
from .backend import dedupe_preserve_order
from .backend import read_launcher_source

if TYPE_CHECKING:
    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from ..runtime.kernel import OutputCodeOptions
    from .backend import LauncherInfo


_SHIM_DOC = (
    "Inlined dependency-free Helion runtime shim -- this file does NOT need the real "
    "helion package. The generated kernel below is emitted verbatim, so it still "
    "calls the runtime helpers as ``helion.runtime.<fn>(...)``; those resolve against "
    "the ``helion`` namespace built from this function. The launcher is inlined in "
    "this private scope so its symbols can never collide with the kernel name."
)
_HOST_HELPERS = (cdiv, next_power_of_2)


def _artifact_global_reads(table: symtable.SymbolTable) -> set[str]:
    """Use Python's scope analysis, including nested functions and defaults."""
    names = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_global() and symbol.is_referenced()
    }
    for child in table.get_children():
        names.update(_artifact_global_reads(child))
    return names


def _artifact_annotation_reads(tree: ast.AST) -> set[str]:
    # Future/quoted annotations are absent from Python's runtime symbol table,
    # but may subsequently be evaluated by the backend or typing machinery.
    names: set[str] = set()
    for node in ast.walk(tree):
        annotation = None
        if isinstance(node, (ast.arg, ast.AnnAssign)):
            annotation = node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotation = node.returns
        if annotation is None:
            continue
        for value in ast.walk(annotation):
            if isinstance(value, ast.Name):
                names.add(value.id)
            elif isinstance(value, ast.Constant) and isinstance(value.value, str):
                try:
                    expression = ast.parse(value.value, mode="eval")
                except SyntaxError:
                    continue
                names.update(
                    item.id
                    for item in ast.walk(expression)
                    if isinstance(item, ast.Name)
                )
    return names


def _artifact_runtime_nodes(tree: ast.Module) -> set[ast.AST]:
    """Recognize the exact compiler-owned runtime, including optional CUDA APIs."""
    from .triton.backend import TritonBackend

    canonical = _runtime_shim_statements(TritonBackend().dependency_free_launcher_info)[
        0
    ]
    canonical_dump = ast.dump(canonical)
    return {
        child
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and ast.dump(node) == canonical_dump
        for child in ast.walk(node)
    }


def _check_artifact_helion_namespace(tree: ast.Module) -> None:
    """Reject references the inlined Helion namespace cannot satisfy."""
    from .triton.backend import TritonBackend

    info = TritonBackend().dependency_free_launcher_info
    runtime_names = {*info.runtime_helper_names, info.launcher_symbol}
    host_names = {helper.__name__ for helper in _HOST_HELPERS}
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Name)
            and node.id == "helion"
            and isinstance(node.ctx, ast.Load)
        ):
            continue
        attribute = parents.get(node)
        if isinstance(attribute, ast.Attribute) and attribute.value is node:
            if attribute.attr in host_names:
                continue
            if attribute.attr == "runtime":
                runtime_attribute = parents.get(attribute)
                if (
                    isinstance(runtime_attribute, ast.Attribute)
                    and runtime_attribute.value is attribute
                    and runtime_attribute.attr in runtime_names
                ):
                    continue
        raise ValueError(
            "artifact source uses an unsupported Helion namespace reference"
        )


def close_artifact_source(source: str) -> str:
    """Require emitted artifact code to depend only on its declared runtimes.

    This checks the generated program, not equivalence to future executions of
    the original Python kernel. Torch, Triton and the standard-library modules
    used by native code generation are trusted runtime dependencies, as is the
    exact native launcher (whose optional residency checks use CUDA bindings).
    This is not a sandbox for untrusted source or a monkeypatched installation.
    """
    tree = ast.parse(source)
    _check_artifact_helion_namespace(tree)
    runtime_nodes = _artifact_runtime_nodes(tree)
    source = ast.unparse(tree)
    table = symtable.symtable(source, "<helion-artifact>", "exec")
    referenced = _artifact_global_reads(table) | _artifact_annotation_reads(tree)
    # These are dependencies of the emitted launcher/dispatcher, not an
    # allowlist of operations in the source Helion kernel.
    runtime_modules = {
        "__future__",
        "collections",
        "contextvars",
        "functools",
        "inspect",
        "math",
        "operator",
        "torch",
        "triton",
        "types",
        "typing",
        "weakref",
    }
    dynamic_names = {
        "__import__",
        "__builtins__",
        "eval",
        "exec",
        "compile",
        "globals",
        "locals",
        "import_module",
        "load_module",
        "exec_module",
    }
    for node in ast.walk(tree):
        if node in runtime_nodes:
            continue
        if (
            (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in dynamic_names
            )
            or isinstance(node, ast.Attribute)
            and node.attr in dynamic_names
        ):
            raise ValueError("artifact source uses dynamic Python namespace access")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "vars"
            and not node.args
        ):
            raise ValueError("artifact source uses dynamic Python namespace access")
    for node in tuple(ast.walk(tree)):
        if node in runtime_nodes or not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        kept = []
        for alias in node.names:
            if alias.name in dynamic_names:
                raise ValueError("artifact source uses dynamic Python namespace access")
            module = alias.name if isinstance(node, ast.Import) else node.module or ""
            if alias.name == "*" or isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError(
                    "artifact source cannot use relative or wildcard imports"
                )
            if module.split(".")[0] in runtime_modules:
                kept.append(alias)
                continue
            name = alias.asname or (
                alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name
            )
            # Only discard compiler-emitted top-level imports. Nested imports
            # are rejected conservatively; they may have runtime side effects.
            if node not in tree.body or name in referenced:
                raise ValueError(
                    f"artifact source depends on external module {module!r}"
                )
        if kept:
            node.names = kept
        else:
            tree.body.remove(node)
    closed = ast.unparse(tree) + "\n"
    closed_table = symtable.symtable(closed, "<helion-artifact>", "exec")
    bound = {
        symbol.get_name()
        for symbol in closed_table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }
    implicit = {
        "__name__",
        "__doc__",
        "__package__",
        "__file__",
        "__spec__",
        "__loader__",
        "__cached__",
    }
    missing = (
        _artifact_global_reads(closed_table) - bound - set(vars(builtins)) - implicit
    )
    if missing:
        raise ValueError(
            f"artifact source has unbound global names: {', '.join(sorted(missing))}"
        )
    return closed


def build_dependency_free_code(
    bound: BoundKernel[Any],
    options: OutputCodeOptions,
    import_lines: list[str],
    body_root: ast.Module,
) -> ast.Module:
    """Rewrite ``to_code``'s split output into a self-contained (helion-free) module.

    An optional AST processing step over ``body_root``, run before it is unparsed: it
    mutates ``import_lines`` in place (drops the helion imports, adds ``types`` for
    the shim) and prepends -- as AST statements -- the inlined ``helion.runtime``
    shim and any in-kernel runtime helpers the body references (topk /
    compact-worklist). Returns ``body_root`` so :meth:`BoundKernel.to_code` unparses
    it in the shared path.

    Torch-tensor only; ``jax_fn`` is handled by :func:`build_jax_fn_module`.
    """
    backend = bound.env.backend
    # CuTe builds launch wrappers from bound inputs; other backends use the shim below.
    specialized = backend.build_standalone_code(bound, import_lines, body_root)
    if specialized is not None:
        return specialized
    info = backend.dependency_free_launcher_info
    kernel_name = bound.kernel.name
    _reject_body_helion_imports(body_root, kernel_name)
    embedded = backend.embedded_helper_source(ast.unparse(body_root))
    import_lines[:] = dedupe_preserve_order(
        [imp for imp in import_lines if "helion" not in imp] + ["import types"]
    )
    preamble: list[ast.stmt] = _runtime_shim_statements(info)
    if embedded:
        preamble += ast.parse(embedded).body
    body_root.body[:0] = preamble
    ast.fix_missing_locations(body_root)
    _check_kernel_name_not_shadowed(body_root, import_lines, kernel_name)
    return body_root


def capture_jax_launch_metadata(
    bound: BoundKernel[Any], config: Config | dict[str, object]
) -> object:
    """Capture ``to_code(jax_fn=True)`` launch metadata (Pallas only) -- the one
    non-AST step: the backend compiles the kernel and runs a capturing launch on real
    tensors, so this must be called *outside* the fake-tensor env. The result feeds
    :func:`build_jax_fn_module`."""
    return bound.env.backend.capture_jax_launch_metadata(bound, config)


def build_jax_fn_module(
    bound: BoundKernel[Any],
    options: OutputCodeOptions,
    import_lines: list[str],
    body_root: ast.Module,
    meta: object,
) -> ast.Module:
    """Rewrite ``body_root`` into the jax-native standalone module (AST in, AST out).

    An optional AST processing step for ``to_code(jax_fn=True)``: the emitted
    entrypoint operates on ``jax.Array`` inputs. Orthogonal to ``allow_helion_deps``:
    ``False`` inlines the launch core (``jax`` the only runtime dependency), ``True``
    imports it from helion (``jax`` + ``helion``). ``meta`` is the pre-captured value
    from :func:`capture_jax_launch_metadata`; ``import_lines`` is mutated in place to
    the jax import set. Pallas only.
    """
    return bound.env.backend.build_jax_fn_code(
        body_root, import_lines, meta, allow_helion_deps=options.allow_helion_deps
    )


def _reject_body_helion_imports(body_root: ast.Module, kernel_name: str) -> None:
    """Raise if the body AST imports helion anywhere (an in-kernel helper the
    standalone can't satisfy). Module-level helion imports are handled separately
    (dropped from ``import_lines``); this walks the body for a stray one."""
    for node in ast.walk(body_root):
        if isinstance(node, ast.Import) and any(
            "helion" in alias.name for alias in node.names
        ):
            _raise_needs_helion(kernel_name)
        if (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and "helion" in node.module
        ):
            _raise_needs_helion(kernel_name)


def _raise_needs_helion(kernel_name: str) -> None:
    raise NotImplementedError(
        f"cannot export {kernel_name!r} without helion deps: the generated kernel "
        "still imports helion in its body (an in-kernel runtime helper is not "
        "inlined yet)."
    )


def _runtime_shim_statements(info: LauncherInfo) -> list[ast.stmt]:
    """The dependency-free launcher inlined as a local ``helion.runtime`` shim,
    returned as AST statements to prepend to the module body.

    The launcher is inlined inside ``_make_helion_runtime`` -- a private function
    scope, so none of its symbols (``get_num_sm``, ``default_launcher``, ...) land at
    module level where they could clash with the generated kernel -- and re-exported
    on a ``types.SimpleNamespace`` as ``helion.runtime``. Pure host helpers are
    embedded from their canonical implementations on the outer ``helion``
    namespace, so generated host expressions need no Helion package either.

    Built by editing the AST: parse a small template, splice the launcher body into
    the function and the exports into the returned namespace -- no string surgery.
    """
    launcher_ast = ast.parse(read_launcher_source(info.launcher_module))
    # Everything but the module docstring and ``from __future__`` (illegal inside a
    # function) is inlined into ``_make_helion_runtime``; the launcher's own imports
    # come along as function-local imports.
    launcher_stmts = [
        stmt
        for stmt in launcher_ast.body
        if not _is_future_import(stmt) and not _is_docstring_expr(stmt)
    ]
    # The launcher's module-level state (e.g. a ``global``-mutated cache) becomes a
    # ``_make_helion_runtime`` local once inlined, so rewrite those ``global``
    # declarations to ``nonlocal`` -- otherwise a nested helper would look for a
    # module global that no longer exists.
    global_to_nonlocal = _GlobalToNonlocal(set(_module_level_names(launcher_ast)))
    launcher_stmts = [global_to_nonlocal.visit(stmt) for stmt in launcher_stmts]
    exports = dedupe_preserve_order([*info.runtime_helper_names, info.launcher_symbol])
    module = ast.parse(
        "def _make_helion_runtime():\n"
        "    return types.SimpleNamespace()\n"
        "helion = _make_helion_runtime()\n"
        f"{info.launcher_alias} = helion.runtime.{info.launcher_symbol}\n"
    )
    make_fn = module.body[0]
    assert isinstance(make_fn, ast.FunctionDef)
    return_stmt = make_fn.body[0]
    assert isinstance(return_stmt, ast.Return)
    namespace_call = return_stmt.value
    assert isinstance(namespace_call, ast.Call)
    # Docstring explaining the shim, then the inlined launcher, then the return.
    host_helpers = [
        ast.parse(inspect.getsource(helper)).body[0] for helper in _HOST_HELPERS
    ]
    make_fn.body = [
        ast.Expr(ast.Constant(_SHIM_DOC)),
        *launcher_stmts,
        *host_helpers,
        return_stmt,
    ]
    runtime_namespace = ast.Call(
        func=namespace_call.func,
        args=[],
        keywords=[
            ast.keyword(arg=name, value=ast.Name(id=name, ctx=ast.Load()))
            for name in exports
        ],
    )
    namespace_call.keywords = [
        ast.keyword(arg="runtime", value=runtime_namespace),
        *(
            ast.keyword(
                arg=helper.__name__, value=ast.Name(id=helper.__name__, ctx=ast.Load())
            )
            for helper in _HOST_HELPERS
        ),
    ]
    ast.fix_missing_locations(module)
    return module.body


def dependency_free_runtime_source(info: LauncherInfo) -> str:
    """Render the canonical dependency-free runtime shim as module source."""
    module = ast.Module(body=_runtime_shim_statements(info), type_ignores=[])
    ast.fix_missing_locations(module)
    return ast.unparse(module)


def _is_future_import(node: ast.stmt) -> bool:
    """True for a ``from __future__ import ...`` statement."""
    return isinstance(node, ast.ImportFrom) and node.module == "__future__"


def _is_docstring_expr(node: ast.stmt) -> bool:
    """True for a bare string-literal statement (a module/function docstring)."""
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _check_kernel_name_not_shadowed(
    body_root: ast.Module, import_lines: list[str], kernel_name: str
) -> None:
    """Ensure the kernel's host-wrapper is the only top-level binding of its name.

    Launcher helpers live inside ``_make_helion_runtime`` (a private scope), so a
    kernel named e.g. ``get_num_sm`` is fine. This only fires if the name still
    clashes with a genuine module-level symbol (an import, the shim, the device
    kernel) -- in which case we raise rather than emit a module whose later ``def``
    silently shadows the earlier binding.
    """
    import_names = (
        _module_level_names(ast.parse("\n".join(import_lines))) if import_lines else []
    )
    names = [*import_names, *_module_level_names(body_root)]
    if names.count(kernel_name) > 1:
        raise ValueError(
            f"cannot export kernel {kernel_name!r} without helion deps: its name "
            "collides with another top-level symbol in the generated module"
        )


def _module_level_names(tree: ast.Module) -> list[str]:
    """Top-level names a module binds (defs, classes, assignments, imports)."""
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names.extend(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.extend((a.asname or a.name).split(".")[0] for a in node.names)
    return names


class _GlobalToNonlocal(ast.NodeTransformer):
    """Rewrite ``global X`` -> ``nonlocal X`` for names the shim inlines as
    ``_make_helion_runtime`` locals (the launcher's module-level state), so a
    nested launcher helper still rebinds the shared value rather than looking for a
    module global that no longer exists. Names not inlined as locals are left as
    ``global`` (there are none in the current launchers, but this stays correct)."""

    def __init__(self, local_names: set[str]) -> None:
        self._local_names = local_names

    def visit_Global(self, node: ast.Global) -> ast.stmt:
        if node.names and all(name in self._local_names for name in node.names):
            return ast.copy_location(ast.Nonlocal(names=node.names), node)
        return node
