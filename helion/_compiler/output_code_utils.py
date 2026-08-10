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
:mod:`helion._compiler.backend`).
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Any

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
    on a ``types.SimpleNamespace`` as ``helion.runtime`` so the body's verbatim
    ``helion.runtime.<fn>(...)`` calls resolve against it.

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
        "helion = types.SimpleNamespace(runtime=_make_helion_runtime())\n"
        f"{info.launcher_alias} = helion.runtime.{info.launcher_symbol}\n"
    )
    make_fn = module.body[0]
    assert isinstance(make_fn, ast.FunctionDef)
    return_stmt = make_fn.body[0]
    assert isinstance(return_stmt, ast.Return)
    namespace_call = return_stmt.value
    assert isinstance(namespace_call, ast.Call)
    # Docstring explaining the shim, then the inlined launcher, then the return.
    make_fn.body = [ast.Expr(ast.Constant(_SHIM_DOC)), *launcher_stmts, return_stmt]
    namespace_call.keywords = [
        ast.keyword(arg=name, value=ast.Name(id=name, ctx=ast.Load()))
        for name in exports
    ]
    ast.fix_missing_locations(module)
    return module.body


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
