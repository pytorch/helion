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

Writes ``<source>_<kernel>_standalone.py`` next to each canonical kernel source
file, or ``<kernel>_standalone.py`` in the requested output directory when the
source is unavailable or cannot be resolved safely.
Static dispatch keys include tensor dtype, shape, and stride, but never tensor
contents. Any tensor-derived value that affects generated code must therefore
also be exposed as a scalar/container argument or checked by the emitted runtime
wrapper.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import logging
import marshal
from pathlib import Path
import re
import tempfile
import textwrap
from typing import TYPE_CHECKING

import torch

from .._compiler.output_code_utils import _check_kernel_name_not_shadowed
from .._compiler.output_code_utils import dependency_free_runtime_source

if TYPE_CHECKING:
    from types import CodeType

log: logging.Logger = logging.getLogger(__name__)


def _standalone_value_key(value: object) -> tuple[object, ...]:
    """Normalize one static call value into a deterministic literal-safe key."""
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            str(value.dtype),
            tuple(value.shape),
            tuple(value.stride()),
        )
    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        return ("float", value.hex())
    if type(value) is str:
        return ("str", value)
    if isinstance(value, torch.dtype):
        return ("dtype", str(value))
    if isinstance(value, torch.device):
        return ("device", str(value))
    if type(value) is tuple:
        return ("tuple", tuple(_standalone_value_key(item) for item in value))
    if type(value) is list:
        return ("list", tuple(_standalone_value_key(item) for item in value))
    if type(value) is dict:
        items = [
            (_standalone_value_key(key), _standalone_value_key(item))
            for key, item in value.items()
        ]
        return ("dict", tuple(sorted(items, key=repr)))
    raise TypeError(
        "standalone static dispatch does not support "
        f"{type(value).__qualname__} arguments"
    )


def _standalone_call_key(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return the static key for signature-normalized positional arguments."""
    return _standalone_value_key(args)


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
    sfx = f"_c{config_idx}"

    body = body.replace(f"_helion_{kernel_name}", f"_helion_{kernel_name}{sfx}")

    body = re.sub(
        rf"^(def ){kernel_name}\(",
        rf"\g<1>_{kernel_name}{sfx}(",
        body,
        flags=re.MULTILINE,
    )

    # Word-boundary rename, longest first so _BLOCK_SIZE_0_1 is renamed
    # before _BLOCK_SIZE_0.
    constexpr_names = re.findall(
        r"^(_[A-Z][A-Z0-9_]*)\s*=\s*tl\.constexpr\(", body, re.MULTILINE
    )
    for name in sorted(constexpr_names, key=len, reverse=True):
        # pyrefly: ignore [bad-specialization]
        body = re.sub(rf"\b{re.escape(name)}\b", f"{name}{sfx}", body)

    return body


def _extract_heuristic_body(heuristic_code: str, kernel_name: str) -> str:
    """
    Extract the config-index selection logic from generated heuristic code.

    Keeps ``key_<kernel>`` (decision-tree backend) or ``_extract_features``
    / ``_predict`` (nearest-neighbor backend).  Strips everything else.
    """
    lines = heuristic_code.split("\n")
    out: list[str] = []

    in_docstring = False
    in_multiline_list = False
    skip_fn: str | None = None
    skip_header = True

    for line in lines:
        stripped = line.strip()

        # --- docstrings ---
        if stripped.startswith('"""'):
            if stripped.count('"""') >= 2 and len(stripped) > 3:
                continue
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue

        # --- leading imports / blanks / comments ---
        if skip_header and (
            stripped.startswith(("import ", "from ", "#")) or stripped == ""
        ):
            continue
        skip_header = False

        # --- module-level constants ---
        if stripped.startswith(("CONFIGS", "FEATURE_NAMES", "USED_FEATURES")):
            if "[" in stripped and "]" not in stripped:
                in_multiline_list = True
            continue
        if in_multiline_list:
            if "]" in stripped:
                in_multiline_list = False
            continue

        # --- functions: keep only those relevant to this kernel ---
        if stripped.startswith("def "):
            keep_prefixes = (
                f"def key_{kernel_name}(",
                "def _extract_features(",
                "def _predict(",
                "def _get_dtype_cat(",
            )
            if not any(stripped.startswith(p) for p in keep_prefixes):
                skip_fn = stripped
                continue
        if skip_fn is not None:
            if stripped and not line[0].isspace():
                skip_fn = None
            else:
                continue

        out.append(line)

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def canonical_kernel_source_path(
    kernel_source_file: str | Path | None,
) -> Path | None:
    """Return one durable file-backed source identity, or ``None`` for fallback."""
    if kernel_source_file is None:
        return None
    try:
        source_path = Path(kernel_source_file).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    return source_path if source_path.is_file() else None


def kernel_source_identity(
    kernel_source_file: str | Path | None,
    code_object: CodeType | None = None,
) -> tuple[Path | None, str]:
    """Return canonical source placement and one durable cache identity.

    Production callers provide the code object so distinct interactive kernels
    sharing labels such as ``<string>`` remain isolated. String-only callers
    retain a stable label hash when no code object is available.
    """
    source_path = canonical_kernel_source_path(kernel_source_file)
    if source_path is not None:
        return source_path, str(source_path)
    identity_bytes = (
        str(kernel_source_file).encode("utf-8", errors="surrogatepass")
        if code_object is None
        else marshal.dumps(code_object)
    )
    return None, f"<non-file-kernel:{hashlib.sha256(identity_bytes).hexdigest()}>"


def standalone_output_path(
    *,
    kernel_name: str,
    output_dir: Path,
    kernel_source_file: str | Path | None,
) -> Path:
    """Return the canonical standalone output path shared by writers and checks."""
    source_path = canonical_kernel_source_path(kernel_source_file)
    if source_path is not None:
        return source_path.parent / f"{source_path.stem}_{kernel_name}_standalone.py"
    return output_dir / f"{kernel_name}_standalone.py"


def generate_standalone_file(
    kernel_name: str,
    triton_codes: list[str],
    heuristic_code: str,
    output_dir: Path,
    kernel_source_file: str | Path | None = None,
    dispatch_keys: list[tuple[object, ...]] | None = None,
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
        output_dir: Fallback directory when the source is not a resolvable file.
        kernel_source_file: Optional source identity used for canonical placement.
        dispatch_keys: Optional static call keys parallel to *triton_codes*.

    Returns:
        Path to the generated file.
    """
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
        from .._compiler.triton.backend import TritonBackend

        launcher_info = TritonBackend().dependency_free_launcher_info
        supported_runtime_names = {
            launcher_info.launcher_symbol,
            *launcher_info.runtime_helper_names,
        }
        if unknown := runtime_references - supported_runtime_names:
            raise ValueError(
                "unsupported Helion runtime helper in standalone AOT input: "
                f"{', '.join(sorted(unknown))}"
            )
        runtime_source = dependency_free_runtime_source(launcher_info)
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
    if dispatch_keys is not None:
        parts.extend(
            [
                textwrap.dedent(inspect.getsource(_standalone_value_key)),
                textwrap.dedent(inspect.getsource(_standalone_call_key)),
            ]
        )

    sep = "=" * 65
    for i, body in enumerate(bodies):
        parts.extend([f"\n# {sep}", f"# Config {i}", f"# {sep}\n", body])

    has_key_dispatch = f"def key_{kernel_name}(" in heuristic_code
    if dispatch_keys is None and (n > 1 or has_key_dispatch):
        # Heuristic dispatch for multiple configs or an abstaining key.
        parts.extend([f"\n# {sep}", "# Heuristic dispatch", f"# {sep}\n"])
        parts.append(_extract_heuristic_body(heuristic_code, kernel_name))

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
                "        tuple(bound.arguments[name] for name in _STANDALONE_PARAMETER_NAMES)",
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
        if has_key_dispatch:
            select_expr = f"key_{kernel_name}(*args)"
        elif n == 1:
            select_expr = "0"
        else:
            select_expr = "_predict(_extract_features(*args))"
        parts.extend(
            [
                f"\ndef {kernel_name}(*args, **kwargs):",
                f"    selection = {select_expr}",
                "    if selection is None:",
                "        raise ValueError(",
                '            "No standalone variant: the AOT heuristic abstained"',
                "        )",
                "    return [",
            ]
        )
        for i in range(n):
            parts.append(f"        _{kernel_name}_c{i},")
        parts.extend(["    ][selection](*args, **kwargs)", ""])

    content = "\n".join(parts)
    _check_kernel_name_not_shadowed(ast.parse(content), [], kernel_name)

    # -- write --------------------------------------------------------------
    out_file = standalone_output_path(
        kernel_name=kernel_name,
        output_dir=output_dir,
        kernel_source_file=kernel_source_file,
    )

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
