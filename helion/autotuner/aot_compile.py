"""
AOT Standalone Compilation
==========================

Generates a standalone ``.py`` file from Helion kernels. Triton output has no
Helion dependency; backends without a dependency-free launcher export retain
that launcher import. The output contains generated kernel code and a heuristic
dispatcher.

Usage::

    python -m helion.autotuner.aot_runner --standalone \\
        -- python examples/aot_compile_example.py

Writes ``<source>_<kernel>_standalone.py`` next to each kernel source file.
Static dispatch keys include tensor dtype, shape, stride, and any descriptor
safety predicates relevant to the generated file, but never tensor contents.
Any tensor-derived value that affects generated code must therefore also be
exposed as a scalar/container argument or checked by the emitted runtime wrapper.
"""

from __future__ import annotations

import ast
import inspect
import io
import logging
from pathlib import Path
import re
import symtable
import tempfile
import textwrap
import tokenize
from typing import TYPE_CHECKING

import torch

from .._compiler.compile_environment import (
    tensor_descriptor_runtime_alignment_signature,
)
from .._compiler.output_code_utils import _check_kernel_name_not_shadowed
from .._compiler.output_code_utils import dependency_free_runtime_names
from .._compiler.output_code_utils import dependency_free_runtime_source
from .._compiler.triton.backend import TritonBackend
from ..runtime.kernel import _dynamic_tensor_width_key
from ..runtime.kernel import _tensor_index_class

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..runtime.structural_policy import StructuralPolicy
    from .aot_structural_export import RuntimeGuard
    from .aot_structural_export import SpecializedInputFacts

log: logging.Logger = logging.getLogger(__name__)


def _standalone_value_key(
    value: object, *, tensor_descriptor_guards: bool = False
) -> tuple[object, ...]:
    """Normalize one static call value into a deterministic literal-safe key."""
    if isinstance(value, torch.Tensor):
        result = (
            "tensor",
            str(value.dtype),
            tuple(value.shape),
            tuple(value.stride()),
        )
        if tensor_descriptor_guards:
            return (*result, *tensor_descriptor_runtime_alignment_signature(value))
        return result
    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        # The binder compares the binary64 representation, including NaN payloads.
        from struct import pack

        return ("float", pack("!d", value))
    if type(value) is str:
        return ("str", value)
    if isinstance(value, torch.dtype):
        return ("dtype", str(value))
    if isinstance(value, torch.device):
        return ("device", str(value))
    if isinstance(value, tuple) and type(value) not in (tuple, torch.Size):
        # A caller can supply the actual wrapper while a standalone artifact
        # must remain importable without Helion. A wrapper already has its
        # defining module loaded; do not mistake a user tuple with a value field.
        from sys import modules

        constexpr = getattr(modules.get("helion.language.constexpr"), "ConstExpr", ())
        if isinstance(value, constexpr):
            return _standalone_value_key(
                value[0], tensor_descriptor_guards=tensor_descriptor_guards
            )
    if type(value) is tuple or type(value) is torch.Size:
        return (
            "tuple",
            tuple(
                _standalone_value_key(
                    item, tensor_descriptor_guards=tensor_descriptor_guards
                )
                for item in value
            ),
        )
    if type(value) is list:
        return (
            "list",
            tuple(
                _standalone_value_key(
                    item, tensor_descriptor_guards=tensor_descriptor_guards
                )
                for item in value
            ),
        )
    if type(value) is dict:
        items = [
            (
                _standalone_value_key(
                    key, tensor_descriptor_guards=tensor_descriptor_guards
                ),
                _standalone_value_key(
                    item, tensor_descriptor_guards=tensor_descriptor_guards
                ),
            )
            for key, item in value.items()
        ]
        return ("dict", tuple(sorted(items, key=repr)))
    raise TypeError(
        "standalone static dispatch does not support "
        f"{type(value).__qualname__} arguments"
    )


def _standalone_call_key(
    args: tuple[object, ...], *, tensor_descriptor_guards: bool = False
) -> tuple[object, ...]:
    """Return the static key for signature-normalized positional arguments."""
    return _standalone_value_key(
        args, tensor_descriptor_guards=tensor_descriptor_guards
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _split_imports_and_body(code: str) -> tuple[list[str], str]:
    """Split generated Triton code into import lines and everything after."""
    lines = code.split("\n")
    imports: list[str] = []
    body_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("from __future__", "import ", "from ")):
            imports.append(stripped)
            body_start = i + 1
        elif stripped == "" or stripped.startswith("#"):
            body_start = i + 1
        else:
            break
    return imports, "\n".join(lines[body_start:])


def _is_supported_helion_import(import_line: str) -> bool:
    """Whether an import matches the generated Triton wrapper contract."""
    return import_line in {
        "import helion",
        "from helion.runtime import default_launcher as _default_launcher",
        # The tensor maps of box-form prefetches, inlined as ``to_code``'s
        # Helion-free output does (``TritonBackend.embedded_helper_source``).
        TritonBackend().library_imports["helion_tensor_maps"],
        TritonBackend().library_imports["helion_triton_helpers"],
    }


def _runtime_references(body: str) -> set[str]:
    """Collect direct ``helion.runtime.<name>`` references from generated code."""
    result: set[str] = set()
    for node in ast.walk(ast.parse(body)):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "runtime"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "helion"
        ):
            result.add(node.attr)
    return result


def _rename_config_symbols(body: str, kernel_name: str, config_idx: int) -> str:
    """
    Rename module-level symbols so multiple configs coexist in one file.

    Appends ``_c<N>`` to: ``_helion_<kernel>``, the host wrapper
    ``def <kernel>(``, and every ``tl.constexpr`` constant.
    """
    suffix = f"_c{config_idx}"
    device_name = f"_helion_{kernel_name}"
    constexpr_names = set(
        re.findall(r"^(_[A-Z][A-Z0-9_]*)\s*=\s*tl\.constexpr\(", body, re.MULTILINE)
    )
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(body).readline):
        name = token.string
        if token.type == tokenize.NAME:
            if name == device_name or name.startswith(f"{device_name}_"):
                name = device_name + suffix + name[len(device_name) :]
            elif name in constexpr_names:
                name += suffix
            elif (
                name == kernel_name
                and token.start[1] == 4
                and token.line.startswith("def ")
            ):
                name = f"_{name}{suffix}"
        tokens.append(token._replace(string=name))
    return tokenize.untokenize(tokens)


def _referenced_globals(table: symtable.SymbolTable) -> set[str]:
    """Global reads, including those in nested functions and comprehensions."""
    names = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_global() and symbol.is_referenced()
    }
    for child in table.get_children():
        names.update(_referenced_globals(child))
    return names


def _imports_helion(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "helion" for alias in node.names):
                return True
        elif (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] == "helion"
        ):
            return True
    return False


def _key_requires_helion(source: str) -> bool:
    tree = ast.parse(source)
    if _imports_helion(tree):
        return True
    modules: set[str] = set()
    loaders = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                a.asname or a.name for a in node.names if a.name == "importlib"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            loaders.update(
                a.asname or a.name for a in node.names if a.name == "import_module"
            )
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        fn = call.func
        if (isinstance(fn, ast.Name) and fn.id in loaders) or (
            isinstance(fn, ast.Attribute)
            and fn.attr == "import_module"
            and isinstance(fn.value, ast.Name)
            and fn.value.id in modules
        ):
            name = (
                call.args[0]
                if call.args
                else next((kw.value for kw in call.keywords if kw.arg == "name"), None)
            )
            if not (
                isinstance(name, ast.Constant)
                and isinstance(name.value, str)
                and name.value.partition(".")[0] not in ("", "helion")
            ):
                return True
    return False


def _standalone_heuristic_source(source: str, kernel_name: str) -> str:
    """Keep model initialization and the selector portable without a tuner dependency."""
    if not _key_requires_helion(source):
        return source
    tree = ast.parse(source)
    namespace_reads = {
        "globals",
        "locals",
        "vars",
        "exec",
        "eval",
        "__builtins__",
        "__import__",
    }
    # Compile mode discovers configs through this entry point. The exported file
    # dispatches with key_<kernel> instead. Keep defaults and annotations eager;
    # decorators or a reference from the remaining code can make the body live.
    tuner_name = f"autotune_{kernel_name}"
    tuners = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == tuner_name
    ]
    if len(tuners) == 1 and not tuners[0].decorator_list:
        tuner = tuners[0]
        original_body = tuner.body
        tuner.body = [ast.Pass()]
        rest = _referenced_globals(
            symtable.symtable(ast.unparse(tree), "<standalone selector>", "exec")
        )
        if tuner_name in rest or namespace_reads & rest:
            tuner.body = original_body
    used = _referenced_globals(
        symtable.symtable(ast.unparse(tree), "<standalone selector>", "exec")
    )
    # An indirect namespace lookup can observe even an otherwise unused import.
    if used & namespace_reads:
        raise ValueError("Standalone heuristic selector requires a Helion import")
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep = []
            for alias in node.names:
                module = (
                    alias.name if isinstance(node, ast.Import) else node.module or ""
                )
                if module.split(".")[0] != "helion":
                    keep.append(alias)
                elif (
                    alias.name == "*"
                    or (alias.asname or alias.name.split(".")[0]) in used
                ):
                    raise ValueError(
                        "Standalone heuristic selector requires a Helion import"
                    )
            if not keep:
                continue
            node.names = keep
        body.append(node)
    tree.body = body
    portable_source = ast.unparse(tree)
    if _key_requires_helion(portable_source):
        raise ValueError("Standalone heuristic selector requires a Helion import")
    return portable_source


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_standalone_file(
    kernel_name: str,
    triton_codes: list[str],
    heuristic_code: str,
    output_dir: Path,
    kernel_source_file: str | None = None,
    dispatch_keys: list[tuple[object, ...]] | None = None,
    tensor_descriptor_guards: bool = False,
    *,
    structural_policy: StructuralPolicy | None = None,
    user_key: Callable[..., object] | None = None,
    _key_source: tuple[str, str] | None = None,
    dynamic_groups: list[tuple[tuple[object, ...], list[int]]] | None = None,
    index_limit: int | None = None,
    expected_width: int | None = None,
    specialized_input_facts: SpecializedInputFacts = (),
    runtime_guards: tuple[RuntimeGuard, ...] = (),
    runtime_guard_plans: tuple[tuple[RuntimeGuard, ...], ...] | None = None,
) -> Path:
    """
    Generate one standalone ``.py`` file containing every selected config.

    Backends with a dependency-free launcher produce Helion-free output. CuTe
    currently retains its Helion runtime launcher import.

    Each config's symbols get a ``_c<N>`` suffix to avoid collisions. Static
    shape variants can provide ``dispatch_keys`` so the public function rejects
    uncompiled call signatures instead of applying a shape-incompatible config.

    Args:
        kernel_name: Name of the kernel function.
        triton_codes: Triton code strings, one per selected config.
        heuristic_code: Generated heuristic Python source.
        output_dir: Fallback directory when *kernel_source_file* is ``None``.
        kernel_source_file: When set, writes next to the source file.
        dispatch_keys: Optional static call keys parallel to *triton_codes*.
        tensor_descriptor_guards: Include tensor-descriptor runtime predicates in
            static dispatch keys.
        expected_width: Bound index width class for a dynamic artifact with one
            compiled binding. Reject incompatible calls before invoking the child.
        specialized_input_facts: Binding-time literals and tensor metadata that an
            ordinary dynamic child substituted as constants.
        index_limit: Fixed index dtype maximum for dynamic exports;
            by default use automatic Int32/Int64 indexing.
        runtime_guards: Legacy single structural guard plan. Static keys are
            ``(call_key, guard_key)``; dynamic-group keys wrap that pair as
            ``("exact" | "dynamic", (call_key, guard_key))``.
        runtime_guard_plans: Structural plans carried by static keys as
            ``(plan, call_key, guard_key)`` and dynamic-group keys as
            ``("exact" | "dynamic", (plan, call_key, guard_key))``. Every key's
            plan must be listed. Do not combine explicit plans with nonempty
            ``runtime_guards``.

    Returns:
        Path to the generated file.
    """
    if (
        user_key is not None
        and dispatch_keys is None
        and (structural_policy is not None or len(triton_codes) > 1)
    ):
        from .aot_structural_export import _user_key_source

        _key_source = _user_key_source(user_key)
    if structural_policy is not None:
        if expected_width is not None:
            raise ValueError(
                "Structural standalone widths are selected by dynamic_groups"
            )
        from .aot_structural_export import generate_policy_standalone

        return generate_policy_standalone(
            kernel_name,
            triton_codes,
            heuristic_code,
            output_dir,
            kernel_source_file,
            dispatch_keys,
            structural_policy,
            _key_source,
            dynamic_groups,
            runtime_guards,
            runtime_guard_plans,
            index_limit,
        )
    if dispatch_keys is not None:
        if len(dispatch_keys) != len(triton_codes):
            raise ValueError("dispatch_keys must match triton_codes")
        if len(set(dispatch_keys)) != len(dispatch_keys):
            raise ValueError("dispatch_keys must be unique")
        ordered = sorted(
            zip(dispatch_keys, triton_codes, strict=True),
            key=lambda item: repr(item[0]),
        )
        dispatch_keys = [key for key, _code in ordered]
        triton_codes = [code for _key, code in ordered]
    n = len(triton_codes)

    # -- collect imports & bodies -------------------------------------------
    all_imports: set[str] = set()
    bodies: list[str] = []
    needs_runtime = False
    runtime_source: str | None = None
    runtime_references: set[str] = set()
    has_helion_deps = False

    for i, code in enumerate(triton_codes):
        imports, body = _split_imports_and_body(code)
        for imp in imports:
            if "helion" in imp:
                if imp == "import helion.language as hl":
                    # The generated module postpones annotations. Retain the
                    # usual refusal if hl is also read by executable code.
                    table = symtable.symtable(
                        "from __future__ import annotations\n" + body, "<child>", "exec"
                    )
                    if "hl" not in _referenced_globals(table):
                        continue
                if "default_cute_launcher" in imp:
                    # CuTe does not yet expose a dependency-free launcher. Keep
                    # its required runtime import instead of silently emitting
                    # a module whose wrapper references an undefined name.
                    all_imports.add(imp)
                    has_helion_deps = True
                    continue
                if not _is_supported_helion_import(imp):
                    raise ValueError(
                        f"unsupported Helion import in standalone AOT input: {imp!r}"
                    )
                if "default_launcher" in imp:
                    needs_runtime = True
                continue
            all_imports.add(imp)
        references = _runtime_references(body)
        runtime_references.update(references)
        needs_runtime = needs_runtime or bool(references)
        bodies.append(_rename_config_symbols(body, kernel_name, i))
    if dispatch_keys is not None:
        all_imports.add("import inspect")
        all_imports.add("import torch")

    if needs_runtime:
        launcher_info = TritonBackend().dependency_free_launcher_info
        if unknown := runtime_references.difference(
            dependency_free_runtime_names(launcher_info)
        ):
            raise ValueError(
                "unsupported Helion runtime helper in standalone AOT input: "
                f"{', '.join(sorted(unknown))}"
            )
        runtime_source = dependency_free_runtime_source(launcher_info)
        all_imports.add("import types")
    embedded = TritonBackend().embedded_helper_source("\n".join(bodies))
    if embedded:
        all_imports.add("import types")

    # -- assemble -----------------------------------------------------------
    parts: list[str] = [
        f"# Auto-generated standalone Triton kernel for '{kernel_name}'.",
        (
            "# Uses Helion's CuTe runtime launcher."
            if has_helion_deps
            else "# No Helion dependency required at runtime."
        ),
        "",
        "from __future__ import annotations\n",
    ]
    for imp in sorted(all_imports):
        if "from __future__" not in imp:
            parts.append(imp)
    parts.append("")

    if runtime_source is not None:
        parts.append(runtime_source)
    if embedded:
        parts.append(embedded)
    if dispatch_keys is not None:
        parts.extend(
            [
                textwrap.dedent(
                    inspect.getsource(tensor_descriptor_runtime_alignment_signature)
                ),
                textwrap.dedent(inspect.getsource(_standalone_value_key)),
                textwrap.dedent(inspect.getsource(_standalone_call_key)),
            ]
        )

    sep = "=" * 65
    for i, body in enumerate(bodies):
        parts.extend([f"\n# {sep}", f"# Config {i}", f"# {sep}\n", body])

    selector_name = "_helion_aot_select"
    while kernel_name.startswith(selector_name):
        selector_name += "_"
    if n > 1 and dispatch_keys is None:
        # The selector is a separate Python module. Its imports and builtins
        # must not resolve against the publicly named kernel or its children.
        model_source = _standalone_heuristic_source(heuristic_code, kernel_name)
        key_name = f"key_{kernel_name}"
        parts.extend(
            [
                f"\n# {sep}",
                "# Heuristic dispatch",
                f"# {sep}\n",
                f"def {selector_name}():",
                "    from builtins import BaseException, exec, globals",
                "    from hashlib import sha256",
                "    import os",
                "    from sys import modules",
                "    from types import ModuleType",
                f"    source = {model_source!r}",
                "    path = os.path.realpath(globals().get('__file__', __name__))",
                "    digest = sha256(os.fsencode(path) + b'\\x00' + source.encode()).hexdigest()",
                "    name = __name__ + '._helion_selector_' + digest",
                "    model = modules.get(name)",
                "    if model is None:",
                "        model = ModuleType(name)",
                "        model.__file__ = path",
                "        modules[name] = model",
                "        try:",
                "            exec(source, model.__dict__)",
                "        except BaseException:",
                "            del modules[name]",
                "            raise",
                f"    if {key_name!r} in model.__dict__:",
                f"        return model.__dict__[{key_name!r}]",
                "    return lambda *args: model._predict(model._extract_features(*args))",
                f"{selector_name} = {selector_name}()",
            ]
        )

    if dispatch_keys is not None:
        parts.extend([f"\n# {sep}", "# Static call dispatch", f"# {sep}\n"])
        parts.append("_STANDALONE_VARIANTS = {")
        for i, key in enumerate(dispatch_keys):
            parts.append(f"    {key!r}: _{kernel_name}_c{i},")
        parts.extend(
            [
                "}",
                f"_STANDALONE_SIGNATURE = inspect.signature(_{kernel_name}_c0)",
                "_STANDALONE_PARAMETER_NAMES = tuple(",
                "    name",
                "    for name, parameter in _STANDALONE_SIGNATURE.parameters.items()",
                "    if parameter.kind in (",
                "        inspect.Parameter.POSITIONAL_ONLY,",
                "        inspect.Parameter.POSITIONAL_OR_KEYWORD,",
                "    )",
                ")",
                f"\ndef {kernel_name}(*args, **kwargs):",
                "    bound = _STANDALONE_SIGNATURE.bind(*args, **kwargs)",
                "    bound.apply_defaults()",
                "    key = _standalone_call_key(",
                "        tuple(bound.arguments[name] for name in _STANDALONE_PARAMETER_NAMES),",
                f"        tensor_descriptor_guards={tensor_descriptor_guards!r},",
                "    )",
                "    try:",
                "        fn = _STANDALONE_VARIANTS[key]",
                "    except KeyError:",
                "        raise ValueError(",
                '            f"No standalone variant for call signature {key!r}"',
                "        ) from None",
                "    return fn(*args, **kwargs)",
                "",
            ]
        )
    else:
        select_expr = "0" if n == 1 else f"{selector_name}(*args)"
        guard_name = "_helion_aot_width_guard"
        while kernel_name.startswith(guard_name):
            guard_name += "_"
        if specialized_input_facts or expected_width is not None or n > 1:
            children = ", ".join(f"_{kernel_name}_c{i}" for i in range(n))
            parts.extend(
                [
                    f"def {guard_name}():",
                    "    from builtins import RuntimeError, ValueError, any, bool, int, isinstance, max, next, tuple",
                    "    import inspect",
                    "    import torch",
                    f"    signatures = tuple(inspect.signature(child) for child in ({children},))",
                    "    signature = next((sig for sig in signatures if not any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())), signatures[0])",
                    "    names = tuple(name for name, p in signature.parameters.items() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))",
                ]
            )
            if expected_width is not None:
                parts.extend(
                    [
                        textwrap.indent(inspect.getsource(_tensor_index_class), "    "),
                        textwrap.indent(
                            inspect.getsource(_dynamic_tensor_width_key), "    "
                        ),
                    ]
                )
            features = "normalized"
            if _key_source is not None:
                from .aot_kernel import _flatten_key_value

                key_source, key_name = _key_source
                if not has_helion_deps and _key_requires_helion(key_source):
                    from ..runtime.structural_policy import StructuralPolicyError

                    raise StructuralPolicyError(
                        "Dependency-free standalone user key requires a Helion import "
                        "or an unresolved dynamic import"
                    )
                features = "_flatten_key_value(user_key(*normalized))"
                # Keep captured globals in the key's own namespace. The public
                # kernel can itself be named like a builtin used by the flattener.
                parts.extend(
                    [
                        "    from builtins import exec, float, list, str",
                        "    key_globals = {}",
                        f"    exec({key_source!r}, key_globals)",
                        f"    user_key = key_globals[{key_name!r}]",
                        textwrap.indent(inspect.getsource(_flatten_key_value), "    "),
                    ]
                )
            if specialized_input_facts:
                from .aot_structural_export import _matches_specialized_input
                from .aot_structural_export import _project_input

                parts.extend(
                    [
                        "    from builtins import TypeError, dict, float, getattr, len, list, object, repr, sorted, str, type",
                        textwrap.indent(
                            inspect.getsource(_standalone_value_key), "    "
                        ),
                        textwrap.indent(inspect.getsource(_project_input), "    "),
                        textwrap.indent(
                            inspect.getsource(_matches_specialized_input), "    "
                        ),
                    ]
                )
            parts.extend(
                [
                    "    def check(args, kwargs):",
                    "        bound = signature.bind(*args, **kwargs)",
                    "        bound.apply_defaults()",
                ]
            )
            if expected_width is not None:
                parts.extend(
                    [
                        "        tensors = torch.utils._pytree.tree_leaves(tuple(bound.arguments.values()))",
                        f"        width = max((_dynamic_tensor_width_key(x, {index_limit!r}) for x in tensors if isinstance(x, torch.Tensor)), default=0)",
                        f"        if width > {expected_width}:",
                        '            raise ValueError("No standalone variant for the requested index width")',
                    ]
                )
            if specialized_input_facts:
                parts.extend(
                    [
                        "        values = tuple(bound.arguments[name] for name in names)",
                        f"        if any(not _matches_specialized_input(values, path, value) for path, value in {specialized_input_facts!r}):",
                        '            raise ValueError("No standalone variant for the specialized input value")',
                    ]
                )
            parts.extend(
                [
                    "        normalized = tuple(bound.arguments[name] for name in names) if names else args",
                    f"        return {features}",
                    "    return check",
                    f"{guard_name} = {guard_name}()",
                ]
            )
        parts.append(f"\ndef {kernel_name}(*args, **kwargs):")
        if specialized_input_facts or expected_width is not None or n > 1:
            parts.append(f"    normalized = {guard_name}(args, kwargs)")
            select_expr = select_expr.replace("*args", "*normalized")
        parts.append("    return [")
        for i in range(n):
            parts.append(f"        _{kernel_name}_c{i},")
        parts.extend([f"    ][{select_expr}](*args, **kwargs)", ""])

    content = "\n".join(parts)
    _check_kernel_name_not_shadowed(ast.parse(content), [], kernel_name)

    # -- write --------------------------------------------------------------
    if kernel_source_file is not None:
        source_path = Path(kernel_source_file)
        out_file = (
            source_path.parent / f"{source_path.stem}_{kernel_name}_standalone.py"
        )
    else:
        out_file = output_dir / f"{kernel_name}_standalone.py"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=out_file.parent,
            prefix=f".{out_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp:
            temp_path = Path(temp.name)
            temp.write(content)
        temp_path.replace(out_file)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    log.info("Standalone file: %s", out_file)
    return out_file
