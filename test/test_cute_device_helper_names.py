"""Static invariants of the CuTe device-helper modules and the emitters naming them.

Generated kernels reach device helpers through a module alias the preamble
imports (``import helion._compiler.cute._warp_mma_runtime as _helion_warp_mma``)
and the emitters often compose the attribute name in an f-string
(``cvt_{src}x8_{dst}_sts``, ``sel_{op}_f32``, ``ldg_f16_as_f32{suffix}``,
``lane_group_{kind}``, ``{epi_stg_store_fn}_rows``).  A word-match dead-code
sweep cannot see those names, so this module checks, without a GPU:

* every ``<alias>.<name>`` an emitter can render -- each composed placeholder
  expanded over the values its selector table / assignments admit -- resolves
  to an attribute of the runtime module behind that alias;
* every ``FlashAttentionConfig`` field is read (``cfg.`` / ``config.`` /
  ``self.``) by some module;
* every ``device/ptx.py`` row, through every runtime module re-exporting it,
  traces to the inline-asm op of the hand-written wrapper it replaced.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
import itertools
from pathlib import Path
import re
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

pytest.importorskip("cutlass")

import cutlass
from cutlass._mlir import ir
from cutlass._mlir.dialects import func

import helion._compiler.cute as cute_package
from helion._compiler.cute.cute_flash import FlashAttentionConfig

pytestmark = pytest.mark.cpu_only

CUTE_ROOT = Path(cute_package.__file__).parent
CUTE_PACKAGE = cute_package.__name__

# Placeholders whose value set is not visible in the emitter source (plan
# fields, call-site data); keyed by (emitter module, placeholder source text).
PLACEHOLDER_DOMAINS: dict[tuple[str, str], tuple[str, ...]] = {
    # ``plan.load_hints`` values (``policy_var`` docstring).
    ("cute_mma_dag_codegen", "hint"): ("evict_last", "evict_first"),
}

# Composed names the sweep must keep seeing: each was once reported dead by a
# word-match audit.  (module basename, helper name)
COMPOSED_NAME_PINS: tuple[tuple[str, str], ...] = (
    ("_warp_mma_runtime", "sel_eq_f32"),
    ("_warp_mma_runtime", "sel_ne_f32"),
    ("_warp_mma_runtime", "sel_lt_f32"),
    ("_warp_mma_runtime", "sel_le_f32"),
    ("_warp_mma_runtime", "cvt_s16x8_f16_sts"),
    ("_warp_mma_runtime", "cvt_u16x8_bf16_sts"),
    ("_warp_mma_runtime", "cvt_u16x8_f16_sts"),
    ("_flash_row_mma_runtime", "lane_group_min"),
    ("_mma_dag_runtime", "ldg_f16_as_f32_pred"),
    ("_flash_runtime", "fa4_store_o_smem_to_gmem_whole_rows"),
)

_IMPORT_AS = re.compile(r"import (helion\.[\w.]+) as (\{?[A-Za-z_]\w*\}?)")
_LITERAL_ALIAS = re.compile(r"(_helion_[a-z0-9_]+)\.")
_IDENT = re.compile(r"[A-Za-z0-9_]*")

Token = tuple[str, object]  # ("text", str) | ("expr", ast.expr)


@dataclasses.dataclass(frozen=True)
class _Site:
    module: str  # dotted module name of the emitter
    lineno: int
    aliases: frozenset[str]  # alias strings this site renders through
    parts: tuple[Token, ...]  # the attribute pattern after the dot
    scopes: tuple[ast.AST, ...]  # innermost function first, module last

    def text(self) -> str:
        return "".join(
            part if kind == "text" else "{" + ast.unparse(part) + "}"  # type: ignore[arg-type]
            for kind, part in self.parts
        )


class _ModuleIndex:
    """One parsed emitter module: parent links, scopes and value assignments."""

    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path
        self.tree = ast.parse(path.read_text(), filename=str(path))
        self.parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                self.parents[child] = node
        self.docstrings = {
            id(body[0].value)
            for node in ast.walk(self.tree)
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            for body in (node.body,)
            if body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        }
        self.functions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions.setdefault(node.name, []).append(node)
        self._module = None

    @property
    def module(self) -> object:
        if self._module is None:
            self._module = importlib.import_module(self.name)
        return self._module

    def scopes(self, node: ast.AST) -> tuple[ast.AST, ...]:
        chain: list[ast.AST] = []
        current: ast.AST | None = node
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chain.append(current)
            current = self.parents.get(current)
        chain.append(self.tree)
        return tuple(chain)

    def strings(self) -> Iterator[ast.Constant | ast.JoinedStr]:
        """Every string literal and f-string that is not a docstring."""
        inside_joined: set[int] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.JoinedStr):
                inside_joined.update(
                    id(value)
                    for value in node.values
                    if isinstance(value, ast.Constant)
                )
        for node in ast.walk(self.tree):
            if isinstance(node, ast.JoinedStr) or (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in inside_joined
                and id(node) not in self.docstrings
            ):
                yield node


def _tokens(node: ast.Constant | ast.JoinedStr) -> list[Token]:
    if isinstance(node, ast.Constant):
        return [("text", node.value)]
    tokens: list[Token] = []
    for value in node.values:
        if isinstance(value, ast.Constant):
            tokens.append(("text", value.value))
        elif isinstance(value, ast.FormattedValue):
            tokens.append(("expr", value.value))
    return tokens


def _consume(tokens: Sequence[Token], index: int, position: int) -> tuple[Token, ...]:
    """The attribute pattern starting inside ``tokens[index]`` at ``position``:
    identifier text, then any placeholder that immediately continues it."""
    parts: list[Token] = []
    while True:
        kind, value = tokens[index]
        assert kind == "text" and isinstance(value, str)
        match = _IDENT.match(value, position)
        assert match is not None
        if match.group():
            parts.append(("text", match.group()))
        if match.end() != len(value) or index + 1 >= len(tokens):
            return tuple(parts)
        index += 1
        if tokens[index][0] != "expr":
            return tuple(parts)
        parts.append(tokens[index])
        index += 1
        position = 0
        if index >= len(tokens) or tokens[index][0] != "text":
            return tuple(parts)


def _collect_modules() -> dict[str, _ModuleIndex]:
    modules: dict[str, _ModuleIndex] = {}
    for path in sorted(CUTE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(CUTE_ROOT).with_suffix("")
        dotted = ".".join((CUTE_PACKAGE, *relative.parts))
        if relative.name == "__init__":
            dotted = ".".join((CUTE_PACKAGE, *relative.parts[:-1]))
        modules[dotted] = _ModuleIndex(dotted, path)
    return modules


def _alias_targets(modules: dict[str, _ModuleIndex]) -> dict[str, str]:
    """Alias string rendered into generated preambles -> runtime module name."""
    targets: dict[str, str] = {}
    for index in modules.values():
        for match in _IMPORT_AS.finditer(index.path.read_text()):
            target, alias = match.groups()
            if alias.startswith("{"):
                alias = getattr(index.module, alias[1:-1])
            assert isinstance(alias, str)
            previous = targets.setdefault(alias, target)
            assert previous == target, (alias, previous, target)
    return targets


def _alias_variables(
    index: _ModuleIndex, targets: dict[str, str]
) -> dict[str, set[str]]:
    """Names an emitter module binds to alias strings (``_RT = "_helion_rsb"``,
    ``rt = _RUNTIME``) -> the alias strings they can denote."""
    variables: dict[str, set[str]] = {}
    for node in ast.walk(index.tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if isinstance(node.value, ast.Constant) and node.value.value in targets:
            variables.setdefault(target.id, set()).add(node.value.value)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(index.tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Name)
                and node.value.id in variables
            ):
                before = len(variables.get(node.targets[0].id, ()))
                variables.setdefault(node.targets[0].id, set()).update(
                    variables[node.value.id]
                )
                changed |= len(variables[node.targets[0].id]) != before
    return variables


def _sites(
    index: _ModuleIndex, targets: dict[str, str], variables: dict[str, set[str]]
) -> list[_Site]:
    sites: list[_Site] = []
    for node in index.strings():
        tokens = _tokens(node)
        scopes = index.scopes(node)
        for position, (kind, value) in enumerate(tokens):
            if kind == "text":
                assert isinstance(value, str)
                for match in _LITERAL_ALIAS.finditer(value):
                    if match.group(1) not in targets:
                        continue
                    parts = _consume(tokens, position, match.end())
                    if parts:
                        sites.append(
                            _Site(
                                index.name,
                                node.lineno,
                                frozenset({match.group(1)}),
                                parts,
                                scopes,
                            )
                        )
            elif (
                isinstance(value, ast.Name)
                and value.id in variables
                and position + 1 < len(tokens)
                and tokens[position + 1][0] == "text"
                and str(tokens[position + 1][1]).startswith(".")
            ):
                parts = _consume(tokens, position + 1, 1)
                if parts:
                    sites.append(
                        _Site(
                            index.name,
                            node.lineno,
                            frozenset(variables[value.id]),
                            parts,
                            scopes,
                        )
                    )
    return sites


class _Undetermined(Exception):
    """A placeholder whose value set the emitter source does not show."""


class _Domains:
    """Value sets of f-string placeholders, read from the emitter's own
    assignments, selector dicts, conditional expressions, return statements
    and call sites."""

    def __init__(self, index: _ModuleIndex) -> None:
        self.index = index

    def of(self, expr: ast.expr, scopes: Sequence[ast.AST], depth: int = 0) -> set[str]:
        if depth > 6:
            raise _Undetermined(ast.unparse(expr))
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, str):
                return {expr.value}
            raise _Undetermined(ast.unparse(expr))
        if isinstance(expr, ast.JoinedStr):
            parts: list[set[str]] = []
            for value in expr.values:
                if isinstance(value, ast.Constant):
                    parts.append({str(value.value)})
                elif isinstance(value, ast.FormattedValue):
                    parts.append(self.of(value.value, scopes, depth + 1))
            return {"".join(piece) for piece in itertools.product(*parts)}
        if isinstance(expr, ast.IfExp):
            return self.of(expr.body, scopes, depth + 1) | self.of(
                expr.orelse, scopes, depth + 1
            )
        if isinstance(expr, ast.Name):
            return self.of_name(expr.id, scopes, depth)
        if isinstance(expr, ast.Attribute):
            return self.of_attribute(expr, depth)
        if isinstance(expr, ast.Subscript):
            return self.of_subscript(expr, scopes, depth)
        if isinstance(expr, ast.Call):
            return self.of_call(expr, scopes, depth)
        raise _Undetermined(ast.unparse(expr))

    def _assigned(
        self, name: str, scopes: Sequence[ast.AST], depth: int
    ) -> set[str] | None:
        for scope in scopes:
            values: set[str] = set()
            for node in ast.walk(scope):
                if (
                    isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == name
                        for target in node.targets
                    )
                ) or (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == name
                    and node.value is not None
                ):
                    try:
                        values |= self.of(node.value, scopes, depth + 1)
                    except _Undetermined:
                        continue
            if values:
                return values
        return None

    def of_name(self, name: str, scopes: Sequence[ast.AST], depth: int) -> set[str]:
        assigned = self._assigned(name, scopes, depth)
        if assigned:
            return assigned
        for scope in scopes:
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parameters = [arg.arg for arg in scope.args.args]
                if name in parameters:
                    return self.of_argument(scope, parameters.index(name), name, depth)
                if any(arg.arg == name for arg in scope.args.kwonlyargs):
                    # Keyword-only parameters (the hoisted fa4 role emitters
                    # take their captured strings this way) are matched at
                    # the call sites by keyword alone.
                    return self.of_argument(scope, -1, name, depth)
        value = getattr(self.index.module, name, None)
        if isinstance(value, str):
            return {value}
        raise _Undetermined(name)

    def of_attribute(self, expr: ast.Attribute, depth: int) -> set[str]:
        """The values a context field (``ctx.name``) is assigned anywhere in
        the module: the hoisted flash body emitters read the names their
        ``FlashBodyContext`` builder chose, so every ``<obj>.name = ...``
        of the module is a candidate."""
        values: set[str] = set()
        for node in ast.walk(self.index.tree):
            if isinstance(node, ast.Assign):
                if not any(
                    isinstance(target, ast.Attribute) and target.attr == expr.attr
                    for target in node.targets
                ):
                    continue
                value = node.value
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Attribute)
                and node.target.attr == expr.attr
                and node.value is not None
            ):
                value = node.value
            else:
                continue
            try:
                values |= self.of(value, self.index.scopes(value), depth + 1)
            except _Undetermined:
                continue
        if not values:
            raise _Undetermined(ast.unparse(expr))
        return values

    def of_argument(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        position: int,
        name: str,
        depth: int,
    ) -> set[str]:
        """The values every call site of ``function`` passes for parameter
        ``name`` (methods drop ``self``)."""
        values: set[str] = set()
        method = bool(function.args.args) and function.args.args[0].arg == "self"
        for node in ast.walk(self.index.tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if called != function.name:
                continue
            argument: ast.expr | None = None
            for keyword in node.keywords:
                if keyword.arg == name:
                    argument = keyword.value
            if argument is None:
                index = (
                    position - 1
                    if method and isinstance(func, ast.Attribute)
                    else position
                )
                if 0 <= index < len(node.args):
                    argument = node.args[index]
            if argument is None:
                continue
            values |= self.of(argument, self.index.scopes(node), depth + 1)
        if not values:
            raise _Undetermined(name)
        return values

    def imported(self, name: str) -> object:
        """The object a (possibly function-local) ``from ... import`` binds
        to ``name`` in this module, or None."""
        for node in ast.walk(self.index.tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            for alias in node.names:
                if (alias.asname or alias.name) != name:
                    continue
                module_name = importlib.util.resolve_name(
                    "." * node.level + (node.module or ""),
                    self.index.name.rsplit(".", 1)[0],
                )
                return getattr(importlib.import_module(module_name), alias.name)
        return None

    def of_subscript(
        self, expr: ast.Subscript, scopes: Sequence[ast.AST], depth: int
    ) -> set[str]:
        table = expr.value
        if isinstance(table, ast.Name):
            value = getattr(self.index.module, table.id, None)
            if value is None:
                value = self.imported(table.id)
            if isinstance(value, dict):
                return {item for item in value.values() if isinstance(item, str)}
            for scope in scopes:
                for node in ast.walk(scope):
                    if (
                        isinstance(node, ast.Assign)
                        and any(
                            isinstance(target, ast.Name) and target.id == table.id
                            for target in node.targets
                        )
                        and isinstance(node.value, ast.Dict)
                    ):
                        return set().union(
                            *(
                                self.of(item, scopes, depth + 1)
                                for item in node.value.values
                                if item is not None
                            )
                        )
        if isinstance(table, ast.Dict):
            return set().union(
                *(
                    self.of(item, scopes, depth + 1)
                    for item in table.values
                    if item is not None
                )
            )
        raise _Undetermined(ast.unparse(expr))

    def of_call(
        self, expr: ast.Call, scopes: Sequence[ast.AST], depth: int
    ) -> set[str]:
        func = expr.func
        called = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        values: set[str] = set()
        for function in self.index.functions.get(called or "", ()):
            for node in ast.walk(function):
                if isinstance(node, ast.Return) and node.value is not None:
                    values |= self.of(
                        node.value, self.index.scopes(node.value), depth + 1
                    )
        if not values:
            raise _Undetermined(ast.unparse(expr))
        return values


def _expand(site: _Site, domains: _Domains) -> set[str]:
    """Every attribute name the site can render; ``_Undetermined`` names the
    placeholder whose values are invisible (add it to PLACEHOLDER_DOMAINS)."""
    pieces: list[set[str]] = []
    for kind, part in site.parts:
        if kind == "text":
            assert isinstance(part, str)
            pieces.append({part})
            continue
        assert isinstance(part, ast.expr)
        key = (site.module.rsplit(".", 1)[-1], ast.unparse(part))
        if key in PLACEHOLDER_DOMAINS:
            pieces.append(set(PLACEHOLDER_DOMAINS[key]))
            continue
        try:
            pieces.append(domains.of(part, site.scopes))
        except _Undetermined as error:
            raise _Undetermined(
                f"{site.module}:{site.lineno}: cannot determine the values of "
                f"{{{ast.unparse(part)}}} in `<rt>.{site.text()}`: {error}; add the "
                f"value set to PLACEHOLDER_DOMAINS[{key!r}]"
            ) from None
    return {"".join(piece) for piece in itertools.product(*pieces)}


def _render_sites() -> tuple[dict[str, str], list[tuple[_Site, set[str]]]]:
    modules = _collect_modules()
    targets = _alias_targets(modules)
    rendered: list[tuple[_Site, set[str]]] = []
    undetermined: list[str] = []
    for index in modules.values():
        source = index.path.read_text()
        if "_helion_" not in source:
            continue
        variables = _alias_variables(index, targets)
        sites = _sites(index, targets, variables)
        if not sites:
            continue
        domains = _Domains(index)
        for site in sites:
            try:
                rendered.append((site, _expand(site, domains)))
            except _Undetermined as error:
                undetermined.append(str(error))
    assert not undetermined, "\n".join(undetermined)
    return targets, rendered


def test_every_rendered_helper_name_resolves() -> None:
    targets, rendered = _render_sites()
    assert rendered, "no `<alias>.<name>` sites found in the emitters"
    resolved: set[tuple[str, str]] = set()
    failures: list[str] = []
    for site, names in rendered:
        for alias in sorted(site.aliases):
            runtime = importlib.import_module(targets[alias])
            short = targets[alias].rsplit(".", 1)[-1]
            for name in sorted(names):
                if hasattr(runtime, name):
                    resolved.add((short, name))
                else:
                    failures.append(
                        f"{site.module}:{site.lineno}: {alias}.{name} "
                        f"(pattern `{site.text()}`) is not defined by {targets[alias]}"
                    )
    assert not failures, "\n".join(failures)
    missing = [pin for pin in COMPOSED_NAME_PINS if pin not in resolved]
    assert not missing, f"composed names no longer seen by the sweep: {missing}"


# The ``device/ptx.py`` rows each runtime module re-exports (every row is
# re-exported by at least one module; ``test_every_ptx_row_is_pinned`` keeps
# the two tables and the module in step).
PTX_ROW_EXPORTS: dict[str, tuple[str, ...]] = {
    "_paged_attention_runtime": ("add_bf16x2",),
    "_flash_row_mma_runtime": (
        "bulk_reduce_add_f32",
        "cp_async_16",
        "cp_async_16_zfill",
        "finite_or_zero",
        "ldmatrix_x2_trans",
        "ldmatrix_x4",
        "ldmatrix_x4_trans",
        "lds_f32",
        "lds_v2_f32",
        "lds_v4_f32",
        "mask_key_after",
        "movmatrix_trans",
        "stg_b32",
        "stg_f32",
        "stg_v4_b32",
        "sts_f32",
        "sts_v2_f32",
        "unpack_f16x2",
    ),
    "_warp_mma_runtime": (
        "cp_async_16",
        "cp_async_16_ca",
        "cp_async_4",
        "ldmatrix_x2",
        "ldmatrix_x2_trans",
        "ldmatrix_x4",
        "ldmatrix_x4_trans",
        "lds_bf16_as_f32",
        "lds_f16_as_f32",
        "lds_s32",
        "lds_v4_b32",
        "mma_e4m3",
        "mma_tf32",
        "round_bf16_f32",
        "sel_eq_f32",
        "sel_ge_f32",
        "sel_gt_f32",
        "sel_le_f32",
        "sel_lt_f32",
        "sel_ne_f32",
        "stg_b32",
        "stg_bf16",
        "stg_f16",
        "stg_f32",
        "stg_v2_b32",
        "stg_v2_f32",
        "stg_v4_b32",
        "stg_v4_f32",
        "sts_v4_b32",
        "unpack_f16x2",
    ),
    "_mma_dag_runtime": (
        "bits_f32",
        "cp_async_16",
        "cp_async_16_ca",
        "cp_async_16_l2",
        "cp_async_16_l2_zfill",
        "cp_async_16_zfill",
        "cp_async_4",
        "f32_bits",
        "l2_policy_evict_first",
        "l2_policy_evict_last",
        "ldg_b32_pred",
        "ldg_bf16_as_f32_pred",
        "ldg_f16_as_f32_pred",
        "ldg_f32_pred",
        "ldg_s32_pred",
        "ldg_s64",
        "ldg_s8_pred",
        "ldg_u8_pred",
        "ldg_v2_f32_pred",
        "ldmatrix_x2",
        "ldmatrix_x2_trans",
        "ldmatrix_x4",
        "ldmatrix_x4_trans",
        "lds_b32",
        "lds_bf16_as_f32",
        "lds_f16_as_f32",
        "lds_f32",
        "lds_s32",
        "lds_v2_f32",
        "lds_v4_b32",
        "lds_v4_f32",
        "mma_tf32",
        "movmatrix_trans",
        "round_bf16_f32",
        "shfl_down_add_f32",
        "shfl_up_add_f32",
        "stg_b32",
        "stg_bf16",
        "stg_f16",
        "stg_f32",
        "stg_v2_f32",
        "stg_v4_b32",
        "sts_b32",
        "sts_f32",
        "sts_v2_b32",
        "sts_v2_f32",
        "sts_v4_b32",
        "sts_v4_f32",
        "unpack_f16x2",
    ),
    "_mma_dag_tcgen05_runtime": (
        "atom_add_release_gpu_u32",
        "fence_acq_rel_gpu",
        "ld_acquire_gpu_u32",
        "ldcg_f32",
        "lds_v4_f32",
        "sts_v4_f32",
    ),
    "affine_recurrence_primitives": (
        "movmatrix_b16",
        "prefetch_global_l2",
        "store_u32x4_if_valid",
        "tma_load_3d",
        "tma_store_3d",
        "tma_store_commit_group",
    ),
    "chunk_prepare_split_alias_device": (
        "fence_tensormap_acquire",
        "movmatrix_b16",
        "mul_bf16x2",
        "tma_load_4d",
        "tma_store_3d",
        "tma_store_commit_group",
    ),
    "chunk_recurrence_dv4_sm100": (
        "fence_tensormap_acquire",
        "movmatrix_b16",
        "sub_bf16x2",
        "tma_load_3d",
        "tma_store_3d",
        "tma_store_commit_group",
    ),
    # Rows generated kernels import from ``device.ptx`` itself
    # (``CuteBackend.library_imports``; ``pair_lane_math`` renders them).
    "device.ptx": ("exact_div_fast_f32x2", "exp_f32x2_parts"),
    "device.half_extrema": (
        "max_nan_bf16x2",
        "max_nan_f16x2",
        "min_nan_bf16x2",
        "min_nan_f16x2",
        "unpack_f16x2",
    ),
}

# Traced ``llvm.inline_asm`` of each row: (asm text, constraint string, carries
# the side-effect flag, operand MLIR types, result MLIR type, DSL return kind),
# pinned from the hand-written wrappers the rows replaced (traced the same way
# in the tree before the table).  ``ptx_op`` composes the constraint string
# from the operand coercions, so the pins check the builder, not only the
# table text.  A ``smem_ptr`` / ``gmem_ptr`` operand is a ``cute.Pointer`` the
# row reads through ``toint`` (an i32 / i64 asm operand).  The return kind is
# the Python value the row hands the kernel -- ``None``, one DSL scalar type
# name or the tuple of them -- because an ``i32`` result is ``Int32`` or
# ``Uint32`` by the row's ``outs`` alone, and that signedness is part of the
# traced program downstream of the asm.
_ReturnKind = str | tuple[str, ...] | None
_Fingerprint = tuple[str, str, bool, tuple[str, ...], str | None, _ReturnKind]
_F32_PAIR = "!llvm.struct<(f32, f32)>"
_F32_QUAD = "!llvm.struct<(f32, f32, f32, f32)>"
_I32_PAIR = "!llvm.struct<(i32, i32)>"
_I32_QUAD = "!llvm.struct<(i32, i32, i32, i32)>"
PTX_ROW_FINGERPRINTS: dict[str, _Fingerprint] = {
    "add_bf16x2": (
        "add.rn.bf16x2 $0, $1, $2;",
        "=r,r,r",
        False,
        ("i32", "i32"),
        "i32",
        "Int32",
    ),
    "atom_add_release_gpu_u32": (
        "atom.add.release.gpu.global.u32 $0, [$1], $2;",
        "=r,l,r,~{memory}",
        True,
        ("i64", "i32"),
        "i32",
        "Uint32",
    ),
    "bits_f32": (
        "mov.b32 $0, $1;",
        "=f,r",
        False,
        ("i32",),
        "f32",
        "Float32",
    ),
    "bulk_reduce_add_f32": (
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,r,r,~{memory}",
        True,
        ("i64", "i32", "i32"),
        None,
        None,
    ),
    "cp_async_16": (
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l,~{memory}",
        True,
        ("i32", "i64"),
        None,
        None,
    ),
    "cp_async_16_ca": (
        "cp.async.ca.shared.global [$0], [$1], 16;",
        "r,l,~{memory}",
        True,
        ("i32", "i64"),
        None,
        None,
    ),
    "cp_async_16_l2": (
        "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, $2;",
        "r,l,l,~{memory}",
        True,
        ("i32", "i64", "i64"),
        None,
        None,
    ),
    "cp_async_16_l2_zfill": (
        "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, $2, $3;",
        "r,l,r,l,~{memory}",
        True,
        ("i32", "i64", "i32", "i64"),
        None,
        None,
    ),
    "cp_async_16_zfill": (
        "cp.async.cg.shared.global [$0], [$1], 16, $2;",
        "r,l,r,~{memory}",
        True,
        ("i32", "i64", "i32"),
        None,
        None,
    ),
    "cp_async_4": (
        "cp.async.ca.shared.global [$0], [$1], 4;",
        "r,l,~{memory}",
        True,
        ("i32", "i64"),
        None,
        None,
    ),
    "exact_div_fast_f32x2": (
        (
            "{.reg .f32 nb; .reg .b64 a, yy, nbb, q, e, r;"
            " neg.f32 nb, $4; mov.b64 a, {$2, $3};"
            " mov.b64 yy, {$5, $5}; mov.b64 nbb, {nb, nb};"
            " mul.rn.f32x2 q, a, yy; fma.rn.f32x2 e, q, nbb, a;"
            " fma.rn.f32x2 r, e, yy, q; mov.b64 {$0, $1}, r;}"
        ),
        "=f,=f,f,f,f,f",
        False,
        ("f32", "f32", "f32", "f32"),
        _F32_PAIR,
        ("Float32", "Float32"),
    ),
    "exp_f32x2_parts": (
        (
            "{.reg .f32 t0, t1, n0, n1, r0, r1, c1, half, k252, kmag, kpos, l2e, c2;"
            " .reg .b64 x, t, n, nt, r, kk252, kkmag, kkpos, ll2e, cc2;"
            " mov.b32 c1, 0f3BBB989D; mov.b32 half, 0f3F000000;"
            " mov.b32 k252, 0f437C0000; mov.b32 kmag, 0f4B400001;"
            " mov.b32 kpos, 0f4B40007F; mov.b32 l2e, 0f3FB8AA3B; mov.b32 c2, 0f32A57060;"
            " fma.rn.f32 t0, $4, c1, half; cvt.sat.f32.f32 t0, t0;"
            " fma.rn.f32 t1, $5, c1, half; cvt.sat.f32.f32 t1, t1;"
            " mov.b64 t, {t0, t1}; mov.b64 x, {$4, $5};"
            " mov.b64 kk252, {k252, k252}; mov.b64 kkmag, {kmag, kmag};"
            " mov.b64 kkpos, {kpos, kpos}; mov.b64 ll2e, {l2e, l2e}; mov.b64 cc2, {c2, c2};"
            " fma.rm.f32x2 n, t, kk252, kkmag; sub.rn.f32x2 nt, kkpos, n;"
            " fma.rn.f32x2 r, x, ll2e, nt; fma.rn.f32x2 r, x, cc2, r;"
            " mov.b64 {n0, n1}, n; shl.b32 $2, n0, 23; shl.b32 $3, n1, 23;"
            " mov.b64 {r0, r1}, r; ex2.approx.ftz.f32 $0, r0; ex2.approx.ftz.f32 $1, r1;}"
        ),
        "=f,=f,=f,=f,f,f",
        False,
        ("f32", "f32"),
        _F32_QUAD,
        ("Float32", "Float32", "Float32", "Float32"),
    ),
    "f32_bits": (
        "mov.b32 $0, $1;",
        "=r,f",
        False,
        ("f32",),
        "i32",
        "Uint32",
    ),
    "fence_acq_rel_gpu": (
        "fence.acq_rel.gpu;",
        "~{memory}",
        True,
        (),
        None,
        None,
    ),
    "fence_tensormap_acquire": (
        "fence.proxy.tensormap::generic.acquire.gpu [$0], 128;",
        "l",
        True,
        ("i64",),
        None,
        None,
    ),
    "finite_or_zero": (
        "{.reg .pred p; setp.eq.f32 p, $1, 0fFF800000; selp.f32 $0, 0f00000000, $1, p;}",
        "=f,f",
        False,
        ("f32",),
        "f32",
        "Float32",
    ),
    "l2_policy_evict_first": (
        "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
        "=l",
        True,
        (),
        "i64",
        "Int64",
    ),
    "l2_policy_evict_last": (
        "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;",
        "=l",
        True,
        (),
        "i64",
        "Int64",
    ),
    "ld_acquire_gpu_u32": (
        "ld.acquire.gpu.global.u32 $0, [$1];",
        "=r,l,~{memory}",
        True,
        ("i64",),
        "i32",
        "Uint32",
    ),
    "ldcg_f32": (
        "ld.global.cg.f32 $0, [$1];",
        "=f,l,~{memory}",
        True,
        ("i64",),
        "f32",
        "Float32",
    ),
    "ldg_b32_pred": (
        "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.b32 $0, 0; @p ld.global.b32 $0, [$1]; }",
        "=r,l,r,~{memory}",
        True,
        ("i64", "i32"),
        "i32",
        "Uint32",
    ),
    "ldg_bf16_as_f32_pred": (
        "{ .reg .pred p; .reg .b16 h; setp.ne.s32 p, $2, 0; mov.b16 h, 0; @p ld.global.b16 h, [$1]; cvt.f32.bf16 $0, h; }",
        "=f,l,r,~{memory}",
        True,
        ("i64", "i32"),
        "f32",
        "Float32",
    ),
    "ldg_f16_as_f32_pred": (
        "{ .reg .pred p; .reg .b16 h; setp.ne.s32 p, $2, 0; mov.b16 h, 0; @p ld.global.b16 h, [$1]; cvt.f32.f16 $0, h; }",
        "=f,l,r,~{memory}",
        True,
        ("i64", "i32"),
        "f32",
        "Float32",
    ),
    "ldg_f32_pred": (
        "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.f32 $0, 0f00000000; @p ld.global.f32 $0, [$1]; }",
        "=f,l,r,~{memory}",
        True,
        ("i64", "i32"),
        "f32",
        "Float32",
    ),
    "ldg_s32_pred": (
        "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.s32 $0, 0; @p ld.global.s32 $0, [$1]; }",
        "=r,l,r,~{memory}",
        True,
        ("i64", "i32"),
        "i32",
        "Int32",
    ),
    "ldg_s64": (
        "ld.global.s64 $0, [$1];",
        "=l,l,~{memory}",
        True,
        ("i64",),
        "i64",
        "Int64",
    ),
    "ldg_s8_pred": (
        "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.s32 $0, 0; @p ld.global.s8 $0, [$1]; }",
        "=r,l,r,~{memory}",
        True,
        ("i64", "i32"),
        "i32",
        "Int32",
    ),
    "ldg_u8_pred": (
        "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.u32 $0, 0; @p ld.global.u8 $0, [$1]; }",
        "=r,l,r,~{memory}",
        True,
        ("i64", "i32"),
        "i32",
        "Uint32",
    ),
    "ldg_v2_f32_pred": (
        "{ .reg .pred p; setp.ne.s32 p, $3, 0; mov.f32 $0, 0f00000000; mov.f32 $1, 0f00000000; @p ld.global.v2.f32 {$0, $1}, [$2]; }",
        "=f,=f,l,r,~{memory}",
        True,
        ("i64", "i32"),
        _F32_PAIR,
        ("Float32", "Float32"),
    ),
    "ldmatrix_x2": (
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [$2];",
        "=r,=r,r,~{memory}",
        True,
        ("i32",),
        _I32_PAIR,
        ("Uint32", "Uint32"),
    ),
    "ldmatrix_x2_trans": (
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0, $1}, [$2];",
        "=r,=r,r,~{memory}",
        True,
        ("i32",),
        _I32_PAIR,
        ("Uint32", "Uint32"),
    ),
    "ldmatrix_x4": (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}",
        True,
        ("i32",),
        _I32_QUAD,
        ("Uint32", "Uint32", "Uint32", "Uint32"),
    ),
    "ldmatrix_x4_trans": (
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}",
        True,
        ("i32",),
        _I32_QUAD,
        ("Uint32", "Uint32", "Uint32", "Uint32"),
    ),
    "lds_b32": (
        "ld.shared.b32 $0, [$1];",
        "=r,r,~{memory}",
        True,
        ("i32",),
        "i32",
        "Uint32",
    ),
    "lds_bf16_as_f32": (
        "{ .reg .b16 h; ld.shared.b16 h, [$1]; cvt.f32.bf16 $0, h; }",
        "=f,r,~{memory}",
        True,
        ("i32",),
        "f32",
        "Float32",
    ),
    "lds_f16_as_f32": (
        "{ .reg .b16 h; ld.shared.b16 h, [$1]; cvt.f32.f16 $0, h; }",
        "=f,r,~{memory}",
        True,
        ("i32",),
        "f32",
        "Float32",
    ),
    "lds_f32": (
        "ld.shared.f32 $0, [$1];",
        "=f,r,~{memory}",
        True,
        ("i32",),
        "f32",
        "Float32",
    ),
    "lds_s32": (
        "ld.shared.s32 $0, [$1];",
        "=r,r,~{memory}",
        True,
        ("i32",),
        "i32",
        "Int32",
    ),
    "lds_v2_f32": (
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r,~{memory}",
        True,
        ("i32",),
        _F32_PAIR,
        ("Float32", "Float32"),
    ),
    "lds_v4_b32": (
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}",
        True,
        ("i32",),
        _I32_QUAD,
        ("Uint32", "Uint32", "Uint32", "Uint32"),
    ),
    "lds_v4_f32": (
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r,~{memory}",
        True,
        ("i32",),
        _F32_QUAD,
        ("Float32", "Float32", "Float32", "Float32"),
    ),
    "mask_key_after": (
        "{.reg .pred p; setp.gt.s32 p, $2, $3; selp.f32 $0, 0fFF800000, $1, p;}",
        "=f,f,r,r",
        False,
        ("f32", "i32", "i32"),
        "f32",
        "Float32",
    ),
    "mma_e4m3": (
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        False,
        ("i32", "i32", "i32", "i32", "i32", "i32", "f32", "f32", "f32", "f32"),
        _F32_QUAD,
        ("Float32", "Float32", "Float32", "Float32"),
    ),
    "mma_tf32": (
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        False,
        ("i32", "i32", "i32", "i32", "i32", "i32", "f32", "f32", "f32", "f32"),
        _F32_QUAD,
        ("Float32", "Float32", "Float32", "Float32"),
    ),
    "max_nan_bf16x2": (
        "max.NaN.bf16x2 $0, $1, $2;",
        "=r,r,r",
        False,
        ("i32", "i32"),
        "i32",
        "Uint32",
    ),
    "max_nan_f16x2": (
        "max.NaN.f16x2 $0, $1, $2;",
        "=r,r,r",
        False,
        ("i32", "i32"),
        "i32",
        "Uint32",
    ),
    "min_nan_bf16x2": (
        "min.NaN.bf16x2 $0, $1, $2;",
        "=r,r,r",
        False,
        ("i32", "i32"),
        "i32",
        "Uint32",
    ),
    "min_nan_f16x2": (
        "min.NaN.f16x2 $0, $1, $2;",
        "=r,r,r",
        False,
        ("i32", "i32"),
        "i32",
        "Uint32",
    ),
    "movmatrix_b16": (
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
        "=r,r",
        False,
        ("i32",),
        "i32",
        "Int32",
    ),
    "movmatrix_trans": (
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
        "=r,r",
        False,
        ("i32",),
        "i32",
        "Uint32",
    ),
    "mul_bf16x2": (
        "mul.rn.bf16x2 $0, $1, $2;",
        "=r,r,r",
        False,
        ("i32", "i32"),
        "i32",
        "Int32",
    ),
    "prefetch_global_l2": (
        "prefetch.global.L2 [$0];",
        "l",
        True,
        ("gmem_ptr",),
        None,
        None,
    ),
    "round_bf16_f32": (
        "{ .reg .b32 w; cvt.rn.bf16x2.f32 w, $1, $1; and.b32 w, w, 0xffff0000; mov.b32 $0, w; }",
        "=f,f",
        False,
        ("f32",),
        "f32",
        "Float32",
    ),
    "sel_eq_f32": (
        "{ .reg .pred p; setp.eq.s32 p, $1, $2; selp.f32 $0, $3, $4, p; }",
        "=f,r,r,f,f",
        False,
        ("i32", "i32", "f32", "f32"),
        "f32",
        "Float32",
    ),
    "sel_ge_f32": (
        "{ .reg .pred p; setp.ge.s32 p, $1, $2; selp.f32 $0, $3, $4, p; }",
        "=f,r,r,f,f",
        False,
        ("i32", "i32", "f32", "f32"),
        "f32",
        "Float32",
    ),
    "sel_gt_f32": (
        "{ .reg .pred p; setp.gt.s32 p, $1, $2; selp.f32 $0, $3, $4, p; }",
        "=f,r,r,f,f",
        False,
        ("i32", "i32", "f32", "f32"),
        "f32",
        "Float32",
    ),
    "sel_le_f32": (
        "{ .reg .pred p; setp.le.s32 p, $1, $2; selp.f32 $0, $3, $4, p; }",
        "=f,r,r,f,f",
        False,
        ("i32", "i32", "f32", "f32"),
        "f32",
        "Float32",
    ),
    "sel_lt_f32": (
        "{ .reg .pred p; setp.lt.s32 p, $1, $2; selp.f32 $0, $3, $4, p; }",
        "=f,r,r,f,f",
        False,
        ("i32", "i32", "f32", "f32"),
        "f32",
        "Float32",
    ),
    "sel_ne_f32": (
        "{ .reg .pred p; setp.ne.s32 p, $1, $2; selp.f32 $0, $3, $4, p; }",
        "=f,r,r,f,f",
        False,
        ("i32", "i32", "f32", "f32"),
        "f32",
        "Float32",
    ),
    "shfl_down_add_f32": (
        "{.reg .f32 t; .reg .pred p; shfl.sync.down.b32 t|p, $1, $2, 0x1f, 0xffffffff; @p add.f32 $0, $1, t; @!p mov.f32 $0, $1;}",
        "=f,f,r",
        True,
        ("f32", "i32"),
        "f32",
        "Float32",
    ),
    "shfl_up_add_f32": (
        "{.reg .f32 t; .reg .pred p; shfl.sync.up.b32 t|p, $1, $2, 0x0, 0xffffffff; @p add.f32 $0, $1, t; @!p mov.f32 $0, $1;}",
        "=f,f,r",
        True,
        ("f32", "i32"),
        "f32",
        "Float32",
    ),
    "stg_b32": (
        "st.global.b32 [$0], $1;",
        "l,r,~{memory}",
        True,
        ("i64", "i32"),
        None,
        None,
    ),
    "stg_bf16": (
        "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; st.global.b16 [$0], h; }",
        "l,f,~{memory}",
        True,
        ("i64", "f32"),
        None,
        None,
    ),
    "stg_f16": (
        "{ .reg .b16 h; cvt.rn.f16.f32 h, $1; st.global.b16 [$0], h; }",
        "l,f,~{memory}",
        True,
        ("i64", "f32"),
        None,
        None,
    ),
    "stg_f32": (
        "st.global.f32 [$0], $1;",
        "l,f,~{memory}",
        True,
        ("i64", "f32"),
        None,
        None,
    ),
    "stg_v2_b32": (
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r,~{memory}",
        True,
        ("i64", "i32", "i32"),
        None,
        None,
    ),
    "stg_v2_f32": (
        "st.global.v2.f32 [$0], {$1, $2};",
        "l,f,f,~{memory}",
        True,
        ("i64", "f32", "f32"),
        None,
        None,
    ),
    "stg_v4_b32": (
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        True,
        ("i64", "i32", "i32", "i32", "i32"),
        None,
        None,
    ),
    "stg_v4_f32": (
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f,~{memory}",
        True,
        ("i64", "f32", "f32", "f32", "f32"),
        None,
        None,
    ),
    "store_u32x4_if_valid": (
        "{ .reg .pred valid; setp.lt.u64 valid, $5, $6; @valid st.global.L1::no_allocate.v4.u32 [$0], {$1, $2, $3, $4}; }",
        "l,r,r,r,r,l,l",
        True,
        ("gmem_ptr", "i32", "i32", "i32", "i32", "i64", "i64"),
        None,
        None,
    ),
    "sts_b32": (
        "st.shared.b32 [$0], $1;",
        "r,r,~{memory}",
        True,
        ("i32", "i32"),
        None,
        None,
    ),
    "sts_f32": (
        "st.shared.f32 [$0], $1;",
        "r,f,~{memory}",
        True,
        ("i32", "f32"),
        None,
        None,
    ),
    "sts_v2_b32": (
        "st.shared.v2.b32 [$0], {$1, $2};",
        "r,r,r,~{memory}",
        True,
        ("i32", "i32", "i32"),
        None,
        None,
    ),
    "sts_v2_f32": (
        "st.shared.v2.f32 [$0], {$1, $2};",
        "r,f,f,~{memory}",
        True,
        ("i32", "f32", "f32"),
        None,
        None,
    ),
    "sts_v4_b32": (
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r,~{memory}",
        True,
        ("i32", "i32", "i32", "i32", "i32"),
        None,
        None,
    ),
    "sts_v4_f32": (
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f,~{memory}",
        True,
        ("i32", "f32", "f32", "f32", "f32"),
        None,
        None,
    ),
    "sub_bf16x2": (
        "sub.rn.bf16x2 $0, $1, $2;",
        "=r,r,r",
        False,
        ("i32", "i32"),
        "i32",
        "Int32",
    ),
    "tma_load_3d": (
        "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$3, $4, $5}], [$2];",
        "r,l,r,r,r,r",
        True,
        ("smem_ptr", "i64", "smem_ptr", "i32", "i32", "i32"),
        None,
        None,
    ),
    "tma_load_4d": (
        "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$3, $4, $5, $6}], [$2];",
        "r,l,r,r,r,r,r",
        True,
        ("smem_ptr", "i64", "smem_ptr", "i32", "i32", "i32", "i32"),
        None,
        None,
    ),
    "tma_store_3d": (
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [$0, {$2, $3, $4}], [$1];",
        "l,r,r,r,r",
        True,
        ("i64", "smem_ptr", "i32", "i32", "i32"),
        None,
        None,
    ),
    "tma_store_commit_group": (
        "cp.async.bulk.commit_group;",
        "",
        True,
        (),
        None,
        None,
    ),
    "unpack_f16x2": (
        "{.reg .b16 lo, hi; mov.b32 {lo, hi}, $2; cvt.f32.f16 $0, lo; cvt.f32.f16 $1, hi;}",
        "=f,=f,r",
        False,
        ("i32",),
        _F32_PAIR,
        ("Float32", "Float32"),
    ),
}
# The bf16 unpack helpers stay one def per runtime (cvt vs shl/and lowering;
# ``_mma_dag_runtime`` re-exports the warp one), not table rows.
_BF16_UNPACK_ASM = {
    "_flash_row_mma_runtime": (
        "{.reg .b16 lo, hi; mov.b32 {lo, hi}, $2; "
        "cvt.f32.bf16 $0, lo; cvt.f32.bf16 $1, hi;}"
    ),
    "_warp_mma_runtime": (
        "{ .reg .b32 lo, hi; shl.b32 lo, $2, 16; and.b32 hi, $2, 0xffff0000; "
        "mov.b32 $0, lo; mov.b32 $1, hi; }"
    ),
}
_BF16_UNPACK_ASM["_mma_dag_runtime"] = _BF16_UNPACK_ASM["_warp_mma_runtime"]
INLINE_ASM_FINGERPRINTS: dict[tuple[str, str], _Fingerprint] = {
    **{
        (module_name, name): PTX_ROW_FINGERPRINTS[name]
        for module_name, names in PTX_ROW_EXPORTS.items()
        for name in names
    },
    **{
        (module_name, "unpack_bf16x2"): (
            asm,
            "=f,=f,r",
            False,
            ("i32",),
            _F32_PAIR,
            ("Float32", "Float32"),
        )
        for module_name, asm in _BF16_UNPACK_ASM.items()
    },
}
# The MLIR operand type behind each operand spec; the pointer specs stand for a
# ``cute.Pointer`` whose ``toint`` is that address.
_OPERAND_TYPES = {"smem_ptr": "i32", "gmem_ptr": "i64"}


class _Pointer:
    """A ``cute.Pointer`` stand-in: ``toint`` is all the rows read of one."""

    def __init__(self, address: object) -> None:
        self.address = address

    def toint(self, *, loc: object = None, ip: object = None) -> object:
        return self.address


def _return_kind(value: object) -> _ReturnKind:
    """The pin's shape of a helper's return value: the DSL type name of a
    scalar, the names of a tuple's members, ``None`` for no result."""
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(type(member).__name__ for member in value)
    return type(value).__name__


def _operand(kind: str) -> object:
    """A traceable operand per operand spec (the helpers coerce Python scalars)."""
    if kind == "smem_ptr":
        return _Pointer(cutlass.Int32(16))
    if kind == "gmem_ptr":
        return _Pointer(cutlass.Int64(16))
    return {"i32": 16, "i64": 16, "f32": 1.0}[kind]


def _ptx_rows() -> dict[str, object]:
    ptx = importlib.import_module(f"{CUTE_PACKAGE}.device.ptx")
    return {
        name: value
        for name, value in vars(ptx).items()
        if callable(value)
        and not isinstance(value, type)
        and not name.startswith("_")
        # the row builder and the result-struct unpackers the rows share
        and name not in {"ptx_op", "words", "floats"}
        and value.__module__ == ptx.__name__
    }


def test_ptx_rows_are_the_runtime_modules_helpers() -> None:
    """The table rows are the objects the runtime modules export, so the
    disk-cache fingerprint of every kernel naming them follows ptx.py."""
    ptx = importlib.import_module(f"{CUTE_PACKAGE}.device.ptx")
    for module_name, names in PTX_ROW_EXPORTS.items():
        module = importlib.import_module(f"{CUTE_PACKAGE}.{module_name}")
        for name in names:
            assert getattr(module, name) is getattr(ptx, name), (module_name, name)
            assert getattr(module, name).__module__ == ptx.__name__


def test_every_ptx_row_is_pinned() -> None:
    """Every row of the table has a fingerprint pin and a re-exporting
    runtime module; a row nothing renders is dead."""
    rows = set(_ptx_rows())
    assert rows == set(PTX_ROW_FINGERPRINTS)
    assert rows == {name for names in PTX_ROW_EXPORTS.values() for name in names}


@pytest.mark.parametrize(("module_name", "name"), sorted(INLINE_ASM_FINGERPRINTS))
def test_inline_asm_helpers_trace_to_their_pinned_ops(
    module_name: str, name: str
) -> None:
    """Trace the helper in a bare MLIR context (no GPU, no ``cute.jit``) and
    compare its ``llvm.inline_asm`` op with the pinned fingerprint."""
    helper = getattr(importlib.import_module(f"{CUTE_PACKAGE}.{module_name}"), name)
    asm, constraints, side_effects, operand_types, result_type, return_kind = (
        INLINE_ASM_FINGERPRINTS[module_name, name]
    )
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            block = func.FuncOp("probe", ([], [])).add_entry_block()
        with ir.InsertionPoint(block):
            returned = helper(*(_operand(kind) for kind in operand_types))
            func.ReturnOp([])
        asm_ops = [
            op for op in block.operations if op.operation.name == "llvm.inline_asm"
        ]
        assert len(asm_ops) == 1, [op.operation.name for op in block.operations]
        (op,) = asm_ops
        attributes = op.operation.attributes
        assert ir.StringAttr(attributes["asm_string"]).value == asm
        assert ir.StringAttr(attributes["constraints"]).value == constraints
        assert ("has_side_effects" in attributes) is side_effects
        assert tuple(str(value.type) for value in op.operands) == tuple(
            _OPERAND_TYPES.get(kind, kind) for kind in operand_types
        )
        assert [str(result.type) for result in op.results] == (
            [result_type] if result_type is not None else []
        )
        assert _return_kind(returned) == return_kind


def test_every_flash_config_field_is_read() -> None:
    """Every field is read through a config receiver (``cfg.``, ``config.``
    or ``self.`` -- the names the resolver, emitters and policy use), so a
    field sharing its name with some other attribute does not count."""
    roots = (
        CUTE_ROOT,
        CUTE_ROOT.parents[1] / "autotuner",
        CUTE_ROOT.parent / "autotuner_heuristics",
        CUTE_ROOT.parents[1] / "runtime",
    )
    source = "\n".join(
        path.read_text()
        for root in roots
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    )
    unread = [
        field.name
        for field in dataclasses.fields(FlashAttentionConfig)
        if re.search(rf"\b(?:cfg|config|self)\.{re.escape(field.name)}\b", source)
        is None
    ]
    assert not unread, f"FlashAttentionConfig fields nothing reads: {unread}"
