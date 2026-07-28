"""Ahead-of-time precompilation of Helion kernels into standalone files.

:func:`precompile` compiles a Helion kernel for one concrete set of sample
inputs and writes a standalone Python file that runs the kernel with **no Helion
dependency** -- only ``torch`` plus the backend DSL (``triton``; ``jax`` for the
Pallas backend). This differs from :func:`helion.pretuned_kernel`, which
pre-*tunes* a kernel but still runs the full Helion compile stack at call time.
Here the generated file is self-contained: import it and call the entrypoint.

This module is backend-neutral: it defines the public API, the
:class:`BackendPrecompiler` interface, and the shared assembly helpers. Each
backend's precompiler lives next to its dependency-free launcher, in
``helion.runtime.<backend>.precompile`` (e.g.
:mod:`helion.runtime.triton.precompile`).

Example::

    @helion.kernel(config=helion.Config(block_sizes=[128]))
    def add(x, y): ...


    helion.precompile(
        helion.PrecompilationInput(
            kernel=add,
            sample_inputs=(x, y),
            output_path="add_precompiled.py",
        )
    )
"""

from __future__ import annotations

import abc
import dataclasses
import inspect
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

if TYPE_CHECKING:
    from .kernel import Kernel


@dataclasses.dataclass
class PrecompilationInput:
    """Inputs to :func:`precompile`.

    Attributes:
        kernel: The ``@helion.kernel`` to precompile. Its ``Settings`` and
            ``Config`` are honored exactly as in a normal call: an explicit
            ``config=``/``configs=`` is used as-is, otherwise the autotuner
            selects one for ``sample_inputs``.
        sample_inputs: Concrete arguments the kernel is compiled for. With
            ``static_shapes=True`` (the default) the generated kernel is
            specialized to these exact shapes/strides.
        output_path: Path of the standalone ``.py`` file to write.
        entrypoint_name: Name of the exported function. Defaults to the
            kernel's own name when ``None``.
        jax_fn: Pallas only -- export a ``jax.Array`` entrypoint (jax-only)
            instead of a torch-tensor one. Ignored by the Triton backend.
    """

    kernel: Kernel[Any]
    sample_inputs: Sequence[Any]
    output_path: str
    entrypoint_name: str | None = None
    jax_fn: bool = False


@dataclasses.dataclass
class PrecompilationResult:
    """Result of :func:`precompile`.

    Attributes:
        output_path: Path of the written standalone file.
        entrypoint_name: Name of the exported entrypoint function.
        source: The full generated source (also what was written to disk).
    """

    output_path: str
    entrypoint_name: str
    source: str


class BackendPrecompiler(abc.ABC):
    """Builds a standalone, helion-free file for one backend.

    The default :meth:`precompile` generates backend code and inlines the
    dependency-free launcher, parameterized by the class attributes below (set by
    each backend subclass). A backend may override :meth:`precompile` entirely for
    a different emission (e.g. a pure-JAX standalone).

    Subclasses live next to the launcher they inline, in
    ``helion.runtime.<backend>.precompile``.
    """

    launcher_module: str  # dotted path of the dep-free launcher, inlined verbatim
    launcher_symbol: str  # public launcher name that module defines
    launcher_alias: str  # underscore alias generated code binds it to
    deps: str  # runtime deps, for the header comment
    # ``helion.runtime.<fn>(`` calls the backend emits, rewritten to bare ``<fn>(``:
    helion_call_rewrites: tuple[str, ...] = ()

    def precompile(
        self,
        input: PrecompilationInput,  # noqa: A002
        args: tuple[object, ...],
    ) -> PrecompilationResult:
        """Generate backend code for ``args`` and inline the dep-free launcher."""
        if input.jax_fn:
            raise NotImplementedError(
                "jax_fn=True is not supported by this backend's precompiler"
            )
        bound = input.kernel.bind(args)
        # Resolve the config exactly as a normal call would: honor an explicit
        # config/configs, otherwise autotune for these inputs.
        bound.ensure_config_exists(args)
        config = bound._implicit_config()
        assert config is not None, "ensure_config_exists did not resolve a config"

        generated = bound.to_triton_code(config)
        entrypoint = input.entrypoint_name or input.kernel.name
        source = _build_standalone(generated, input.kernel.name, entrypoint, self)
        return _write_standalone(input, entrypoint, source)


def precompile(input: PrecompilationInput) -> PrecompilationResult:  # noqa: A002
    """Compile ``input.kernel`` for ``input.sample_inputs`` into a standalone file.

    See :class:`PrecompilationInput` / :class:`PrecompilationResult`. Raises
    ``NotImplementedError`` for backend / mode combinations not yet supported.
    """
    kernel = input.kernel
    args = tuple(input.sample_inputs)
    return _backend_precompiler(kernel.settings.backend).precompile(input, args)


def _backend_precompiler(backend: str) -> BackendPrecompiler:
    """The precompiler for ``backend``, imported lazily from its backend package.

    Backend-specific precompile logic lives beside the backend launcher, so this
    orchestrator stays free of Triton-/Pallas-specific knowledge.
    """
    if backend in ("triton", "tileir"):
        # tileir emits Triton-compatible code (same launcher / helion.runtime
        # calls), so it uses the Triton precompiler.
        from .triton.precompile import TritonPrecompiler

        return TritonPrecompiler()
    if backend == "pallas":
        from .pallas.precompile import PallasPrecompiler

        return PallasPrecompiler()
    raise NotImplementedError(
        f"helion.precompile does not yet support the {backend!r} backend"
    )


# ---------------------------------------------------------------------------
# Shared standalone-assembly helpers (backend-neutral)
# ---------------------------------------------------------------------------


def _build_standalone(
    generated: str,
    kernel_name: str,
    entrypoint_name: str,
    precompiler: BackendPrecompiler,
) -> str:
    """Turn generated backend code into a self-contained, Helion-free module by
    inlining the dependency-free launcher named by ``precompiler``."""
    gen_imports, gen_body = _split_module_source(generated)
    gen_imports = [imp for imp in gen_imports if "helion" not in imp]

    launcher_imports, launcher_body = _load_launcher_source(precompiler.launcher_module)

    imports = _dedupe_preserve_order([*gen_imports, *launcher_imports])

    body = _rewrite_helion_calls(gen_body, precompiler.helion_call_rewrites)
    if entrypoint_name != kernel_name:
        body = _rename_entrypoint(body, kernel_name, entrypoint_name)

    # Fail loudly rather than emit a standalone that secretly still needs helion.
    # In-function helion imports (e.g. the topk / compact-worklist in-kernel
    # helpers) survive `_split_module_source` (which only strips leading imports),
    # so a kernel using those features would otherwise produce a broken "helion-
    # free" file. The inlined launcher itself is helion-free by construction.
    if "import helion" in body or "from helion" in body:
        raise NotImplementedError(
            f"cannot precompile {kernel_name!r}: the generated kernel still "
            "references helion (an in-kernel runtime helper is not inlined yet). "
            "This kernel uses a feature (e.g. topk or compact-worklist) whose "
            "helper embedding is not implemented for precompile yet."
        )

    launcher_path = precompiler.launcher_module.replace(".", "/")
    parts = [
        f"# Auto-generated by helion.precompile for kernel '{kernel_name}'.",
        f"# Standalone: depends only on {precompiler.deps}, no helion at runtime.",
        "from __future__ import annotations",
        "",
        *imports,
        "",
        f"# --- inlined launcher ({launcher_path}.py) ---",
        launcher_body,
        f"{precompiler.launcher_alias} = {precompiler.launcher_symbol}",
        "",
        "# --- generated kernel ---",
        body,
        "",
    ]
    return "\n".join(parts)


def _write_standalone(
    input: PrecompilationInput,  # noqa: A002
    entrypoint: str,
    source: str,
) -> PrecompilationResult:
    """Write ``source`` to ``input.output_path`` and build the result.

    A do-not-edit banner naming the source kernel file (relative to the generated
    file) is prepended so the file always states that it is generated and where
    the real source lives.
    """
    out_path = Path(input.output_path)
    if out_path.parent != Path():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    source = _generated_header(input, out_path) + source
    out_path.write_text(source)
    return PrecompilationResult(
        output_path=str(out_path), entrypoint_name=entrypoint, source=source
    )


def _generated_header(input: PrecompilationInput, out_path: Path) -> str:  # noqa: A002
    """A strong 'do not edit' banner naming the Helion source kernel file, shown
    as a path relative to the generated file."""
    src_file = inspect.getsourcefile(input.kernel.fn)
    if src_file is not None:
        source_ref = os.path.relpath(
            os.path.abspath(src_file), os.path.abspath(out_path.parent)
        )
    else:
        source_ref = f"<{input.kernel.name}: source file unavailable>"
    bar = "# " + "=" * 76
    lines = [
        bar,
        "# THIS FILE IS AUTO-GENERATED BY helion.precompile -- DO NOT EDIT BY HAND.",
        "#",
        "# It is regenerated in full on every helion.precompile run, so any hand",
        "# edits here WILL BE SILENTLY OVERWRITTEN. Do not patch this file.",
        "#",
        "# To change what this kernel does, edit the Helion source kernel it was",
        "# generated from and re-run helion.precompile:",
        f"#     {source_ref}",
        "#",
        "# If a manual patch to this generated file is genuinely unavoidable, add a",
        "# comment at the very top of this file explaining exactly what changed and",
        "# why, so the patch is visible in review and a later regeneration does not",
        "# quietly discard it.",
        bar,
        "",
    ]
    return "\n".join(lines) + "\n"


def _load_launcher_source(module_name: str) -> tuple[list[str], str]:
    """Return ``(imports, body)`` of a dependency-free launcher module."""
    import importlib

    module = importlib.import_module(module_name)
    assert module.__file__ is not None
    source = Path(module.__file__).read_text()
    return _split_module_source(source)


def _rewrite_helion_calls(body: str, names: tuple[str, ...]) -> str:
    """Rewrite ``helion.runtime.<fn>(`` calls to the inlined bare ``<fn>(``."""
    for name in names:
        body = body.replace(f"helion.runtime.{name}(", f"{name}(")
    return body


def _rename_entrypoint(body: str, kernel_name: str, entrypoint_name: str) -> str:
    """Rename the public host-wrapper ``def <kernel_name>(`` to the entrypoint.

    Only the top-level host wrapper is renamed; the ``_helion_<kernel_name>``
    device kernel keeps its name.
    """
    return re.sub(
        rf"^def {re.escape(kernel_name)}\(",
        f"def {entrypoint_name}(",
        body,
        flags=re.MULTILINE,
    )


def _split_module_source(source: str) -> tuple[list[str], str]:
    """Split module source into its leading imports and the remaining body.

    Skips a leading module docstring and ``from __future__`` imports (the
    standalone supplies its own). Collects top-level ``import``/``from ... import``
    lines; everything from the first real statement onward is the body. This is
    line-based (not AST) so the generated code's ``# src[...]`` provenance
    comments and formatting are preserved verbatim.
    """
    lines = source.split("\n")
    i = 0
    n = len(lines)

    # Skip a leading module docstring, if any.
    while i < n and lines[i].strip() == "":
        i += 1
    if i < n and (stripped := lines[i].lstrip()).startswith(('"""', "'''")):
        quote = stripped[:3]
        # single-line docstring (opens and closes on the same line)
        if len(stripped) >= 6 and stripped.endswith(quote):
            i += 1
        else:
            i += 1
            while i < n and quote not in lines[i]:
                i += 1
            i += 1  # consume the closing-quote line

    imports: list[str] = []
    body_start = i
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith("from __future__"):
            body_start = i + 1
        elif stripped.startswith(("import ", "from ")):
            imports.append(stripped)
            body_start = i + 1
        elif stripped == "" or stripped.startswith("#"):
            body_start = i + 1
        else:
            break
        i += 1

    body = "\n".join(lines[body_start:]).strip("\n")
    return imports, body


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
